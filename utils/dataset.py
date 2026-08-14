"""
读取 PocketXMol 的 assembly 索引与分片 LMDB，并按任务—数据库权重持续产出单分子样本。

训练主入口是 :class:`ForeverTaskDataset`：它先从 assembly 找到 ``data_id``，再由
:class:`SingleDatabase` 按复合键合并分子、口袋与扭转字段，最后依次执行特征化、任务构造和加噪变换。
测试入口 :class:`TestTaskDataset` 为同一读取逻辑补充有限长度的 map-style 接口。
本模块不生成新的数据文件；返回的核心对象是含 LMDB 原始字段及 ``task``、``db``、``key`` 元数据的
单样本 ``PocketMolData``，其后才由 PyG ``DataLoader`` 拼成批次。

"""

from itertools import cycle
import os
from typing import Iterator
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
import lmdb
from torch.utils.data import get_worker_info
from torch.distributed import get_rank, get_world_size

from torch.utils.data import Dataset, Sampler, IterableDataset
from torch_geometric.data import Batch
try:
    from .train import shuffled_cyclic_iterator
except:
    import sys
    sys.path.append('.')
    from utils.train import shuffled_cyclic_iterator

LMDB_CONFIGS = {  # seems not used
    'geom': ['mols', 'torsion', 'decom'],
    'qm9': ['mols', 'torsion', 'decom'],
    'unmi': ['mols', 'torsion', 'decom'],
    'csd': ['pocmol10', 'torsion', 'decom'],
    'pbdock': ['pocmol10', 'torsion', 'decom'],
    'moad': ['pocmol10', 'torsion', 'decom'],
    'apep': ['pocmol10', 'torsion', 'decom'],

    'cremp': ['mols'],
    'pepbdb': ['pocmol10'],
}

    
def make_split(df_dict, provided_test=[], test_size_dict=None,
    ratio_val=0.1, ratio_test = 0.1,
    max_val=10000, max_test=10000):
    test_size_dict = test_size_dict if test_size_dict is not None else {}
    
    # find the task with the minimum number of data_id first
    df_num = pd.DataFrame(None, index=[task for task in df_dict.keys()],
                          columns=['train_size', 'train_ratio', 'val_size',
                                   'val_ratio', 'test_size', 'test_ratio'])
    df_num['total'] = [df.shape[0] for df in df_dict.values()]
    
    sorted_tasks = [key for key, value in sorted(
        df_dict.items(), key=lambda item: item[1]['data_id'].unique().shape[0])]
    # test set
    test_ids = set(provided_test)
    for task in sorted_tasks:
        df = df_dict[task]
        data_id_counts = df['data_id'].value_counts()
        if task in test_size_dict.keys():
            num_data_test = test_size_dict[task]
        else:
            num_data_test =  min(int(data_id_counts.shape[0] * ratio_test), max_test)
        if len(test_ids) < num_data_test: # no enough data for test
            # add more data to test
            data_id_counts_remain = data_id_counts[~data_id_counts.index.isin(test_ids)]
            num_data_test_reamin = int(num_data_test - len(test_ids))
            test_ids_this = np.random.choice(data_id_counts_remain.index.values,
                        p=data_id_counts_remain.values/np.sum(data_id_counts_remain.values),
                        size=num_data_test_reamin, replace=False)
            test_ids.update(test_ids_this)

    df_num['test_size'] = [df_dict[task]['data_id'].isin(test_ids).sum() for task in df_dict.keys()]
    df_num['test_ratio'] = df_num['test_size'] / df_num['total']

    # val set
    val_ids = set()
    for task in sorted_tasks:
        df = df_dict[task]
        data_id_counts = df['data_id'].value_counts()
        num_data_val = min(int(data_id_counts.shape[0] * ratio_val), max_val)
        data_id_counts = df[df['data_id'].isin(test_ids) == False]['data_id'].value_counts()
        if len(val_ids) < num_data_val: # no enough data for val
            # add more data to val
            data_id_counts_remain = data_id_counts[~data_id_counts.index.isin(val_ids)]
            num_data_val_remain = int(num_data_val - len(val_ids))
            val_ids_this = np.random.choice(data_id_counts_remain.index.values,
                        p=data_id_counts_remain.values/np.sum(data_id_counts_remain.values),
                        size=num_data_val_remain, replace=False)
            val_ids.update(val_ids_this)
    
    df_num['val_size'] = [df_dict[task]['data_id'].isin(val_ids).sum() for task in df_dict.keys()]
    df_num['val_ratio'] = df_num['val_size'] / df_num['total']
    
    # train set
    train_ids = set()
    for task in sorted_tasks:
        df = df_dict[task]
        train_ids_this = df[~df['data_id'].isin(test_ids | val_ids)]['data_id'].unique()
        train_ids.update(train_ids_this)
    
    df_num['train_size'] = [df_dict[task]['data_id'].isin(train_ids).sum() for task in df_dict.keys()]
    df_num['train_ratio'] = df_num['train_size'] / df_num['total']
    
    # save
    assert train_ids & val_ids & test_ids == set()
    all_ids = list(train_ids) + list(val_ids) + list(test_ids)
    labels = ['train'] * len(train_ids) + ['val'] * len(val_ids) + ['test'] * len(test_ids)
    df_split = pd.DataFrame({'data_id': all_ids, 'split': labels})

    return df_split, df_num
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # df_split.to_csv(save_path, index=False)
    # df_num.to_csv(save_path.replace('.csv', '_num.csv'))
    
    

class RegenDataset(Dataset):
    # make a dataset from previous generated path
    def __init__(self, gen_path, task, file2input, transforms=None):
        super().__init__()
        self.gen_path = gen_path
        self.task = task
        self.file2input = file2input
        self.transforms = transforms
        
        self.gen_name = os.path.basename(gen_path)
        self.df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
        self.sdf_dir = os.path.join(gen_path, f'{self.gen_name}_SDF')
        if not os.path.exists(self.sdf_dir):
            self.sdf_dir = os.path.join(gen_path, 'SDF')
        
        # load pocket_pdb
        pocket_path = os.path.join(self.sdf_dir, '0_inputs', 'pocket_block.pdb')
        with open(pocket_path, 'r') as f:
            self.pocket_pdb = f.read()
        
        # load sdf mol list
        df_succ = self.df_gen[self.df_gen['tag'].isna()]
        self.filename_list = df_succ['filename'].values

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):
        filename = self.filename_list[index]
        data = self.file2input(
            # mol=os.path.join(self.sdf_dir, filename),
            mol=os.path.join(self.sdf_dir, filename.replace('.pdb', '_mol.sdf')),  # pdb may be broken
            pdb=deepcopy(self.pocket_pdb),
            data_id=self.gen_name + '_' + filename,
            pdbid=''
        )
        data.update({'task': self.task,'db':self.gen_name, 'key':''})
        if self.transforms is not None:
            data = self.transforms(data)
        return data



class UseDataset(Dataset):
    """
    把一个已经构造好的分子—口袋样本复制为固定次数的推理输入。

    构造参数:
        - data: PocketMolData，``get_input_from_file`` 生成的口袋—配体模板；其原始叶见下方“单样本输出字段”。
        - n: int, 数据集长度；每个索引都从同一个 ``data`` 深拷贝，不共享张量容器状态。
        - task: str, 写入每个样本 ``task`` 字段的任务名，例如 ``conf`` 或 ``dock``。
        - transforms: callable|None, 依次生成模型字段的推理变换；``None`` 表示只添加元数据。

    单样本输出字段:
        - task: str, 当前推理任务名。
        - db: str, 固定为 ``use``，表示输入不是由训练数据库索引读取。
        - key: str, 固定为空字符串，表示没有 LMDB 复合键。
        - data_id: str，调用方给定的样本标识。
        - pdbid: str，受体结构标识。
        - smiles: str，固定二维配体图的规范 SMILES。
        - element: LongTensor，形状为 (N,)，配体原子序数。
        - pos_all_confs: FloatTensor，形状为 (C, N, 3)，输入 conformer 世界坐标，单位 Å。
        - i_conf_list: list[int]，长度为 C，合法 conformer 的输入编号。
        - num_confs: int 标量 C，合法 conformer 数。
        - bond_index: LongTensor，形状为 (2, 2M)，双向化学键端点。
        - bond_type: LongTensor，形状为 (2M,)，与 ``bond_index`` 列对齐的键类别。
        - num_atoms: int 标量 N，配体原子数。
        - num_bonds: int 标量 M，无向化学键数。
        - pocket_element: LongTensor，形状为 (P,)，口袋原子序数。
        - pocket_pos: FloatTensor，形状为 (P, 3)，变换前为世界坐标，变换后为口袋中心局部坐标，单位 Å。
        - pocket_is_backbone: BoolTensor，形状为 (P,)，真值表示口袋原子属于蛋白主链。
        - pocket_atom_name: list[str]，长度为 P，PDB 原子名。
        - pocket_atom_to_aa_type: LongTensor，形状为 (P,)，口袋原子所属氨基酸类别。
        - pocket_molecule_name: str|None，口袋 PDB ``HEADER`` 名称。
        - bond_rotatable: LongTensor，形状为 (2M,)，与 ``bond_index`` 对齐的可旋转键标记。
        - tor_twisted_pairs: dict[tuple[int, int], list[set[int], set[int]]]，可旋转键到断键两侧非轴原子的映射。
        - fixed_dist_torsion: Tensor，形状为 (N, N)，1 表示原子对距离不随内部扭转改变。
        - tor_bond_mat: Tensor，形状为 (N, N)，逐原子对可旋转键标记。
        - path_mat: Tensor，形状为 (N, N)，化学图最短路径长度，单位为键数。
        - nbh_dict: dict[int, list[int]]，每个原子编号到一跳邻居编号列表的映射。
        - matches_graph: LongTensor，形状为 (M_g, S_g)，忽略手性的压缩自同构映射。
        - matches_iso: LongTensor，形状为 (M_i, S_i)，考虑手性的压缩自同构映射。
        - brics.subgraphs: list[list[int]]，BRICS 片段的原子编号列表。
        - brics.anchors_list: list[set[int]]，每个 BRICS 片段的锚原子编号。
        - brics.nbh_subgraphs: list[list[int]]，每个 BRICS 片段的相邻片段编号。
        - brics.connections: dict[tuple[int, int], tuple[int, int]]，有向片段对到锚原子对的映射。
        - mmpa.subgraphs: list[list[int]]，MMPA 片段的原子编号列表。
        - mmpa.anchors_list: list[set[int]]，每个 MMPA 片段的锚原子编号。
        - mmpa.nbh_subgraphs: list[list[int]]，每个 MMPA 片段的相邻片段编号。
        - mmpa.connections: dict[tuple[int, int], tuple[int, int]]，有向片段对到锚原子对的映射。
        - pocket_atom_feature: FloatTensor，形状为 (P, 25)，口袋元素、氨基酸和主链标记特征。
        - pocket_knn_edge_index: LongTensor，形状为 (2, E_p)，口袋 kNN 有向边端点。
        - pocket_center: FloatTensor，形状为 (1, 3)，世界坐标到模型局部坐标的平移原点，单位 Å。
        - num_nodes: int 标量 N，PyG 配体节点数。
        - node_type: LongTensor，形状为 (N,)，模型原子类别编号。
        - node_pos: FloatTensor，形状为 (N, 3)，配体局部坐标，单位 Å。
        - i_conf: int 标量，本候选实际选择的输入 conformer 编号。
        - halfedge_index: LongTensor，形状为 (2, H)，完全图上三角半边端点，H=N(N-1)/2。
        - halfedge_type: LongTensor，形状为 (H,)，完全图半边类别，0 表示非键。
        - is_peptide: LongTensor，形状为 (N,)，小分子 use/docking 为全 0。
        - task_setting: str，当前 ``free``、``flexible``、``torsional`` 或 ``rigid`` 运动模式。
        - fixed_node: LongTensor，形状为 (N,)，1 表示原子类别为条件。
        - fixed_pos: LongTensor，形状为 (N,)，1 表示坐标为条件。
        - fixed_halfedge: LongTensor，形状为 (H,)，1 表示半边类别为条件。
        - fixed_halfdist: LongTensor，形状为 (H,)，1 表示半边端点距离必须保持。
        - n_domain: LongTensor 标量，刚体域数。
        - domain_node_index: LongTensor，形状为 (2, K)，第一行是刚体域编号，第二行是域内原子编号。
        - tor_bonds_anno: LongTensor，形状为 (T, 3)，每行是执行层级和两个旋转轴端点。
        - twisted_nodes_anno: LongTensor，形状为 (W, 2)，每行是扭转行号和随动原子编号。
        - dihedral_pairs_anno: LongTensor，形状为 (Q, 3)，每行是扭转行号和两个二面角外侧端点。
        - gt_node_type: LongTensor，形状为 (N,)，推理模式保留的原子类别真值副本。
        - gt_node_pos: FloatTensor，形状为 (N, 3)，推理模式保留的局部坐标真值副本，单位 Å。
        - gt_halfedge_type: LongTensor，形状为 (H,)，推理模式保留的半边类别真值副本。
        - fixed_halfdist_flex: LongTensor，形状为 (H,)，推理模式保留的原始柔性距离约束。
    """
    def __init__(self, data, n, task='', transforms=None):
        super().__init__()
        # ``self.data``：单样本容器模板；每次 __getitem__ 都 deepcopy，避免重复候选共享采样状态。
        self.data = data
        # ``self.n``：int，固定重复次数，也是数据集长度。
        self.n = n
        # ``self.task``：str，写入每个重复样本的 task 字段。
        self.task = task
        # ``self.transforms``：callable|None，单样本特征/任务变换链。
        self.transforms = transforms
        
    def __len__(self):
        return self.n

    def __getitem__(self, index):
        """深拷贝输入模板、写入 use 元数据，并执行单候选推理变换。

        输入参数:
            - index: int，候选编号，取值范围为 ``[0, n)``；当前实现只用于 Dataset 协议，不改变模板选择。

        返回字段:
            - data.task: str，当前推理任务名。
            - data.db: str，固定为 ``use``，表示样本不来自训练 LMDB 索引。
            - data.key: str，固定为空字符串，表示没有 LMDB 复合键。
            - data.data_id: str，调用方给定的样本标识。
            - data.pdbid: str，受体结构标识。
            - data.smiles: str，固定二维配体图的规范 SMILES。
            - data.element: LongTensor，形状为 (N,)，配体原子序数。
            - data.pos_all_confs: FloatTensor，形状为 (C, N, 3)，输入 conformer 世界坐标，单位 Å。
            - data.i_conf_list: list[int]，长度为 C，合法 conformer 的输入编号。
            - data.num_confs: int 标量 C，合法 conformer 数。
            - data.bond_index: LongTensor，形状为 (2, 2M)，双向化学键端点，索引配体原子维。
            - data.bond_type: LongTensor，形状为 (2M,)，与 ``bond_index`` 列对齐的键类别。
            - data.num_atoms: int 标量 N，配体原子数。
            - data.num_bonds: int 标量 M，无向化学键数。
            - data.pocket_element: LongTensor，形状为 (P,)，口袋原子序数。
            - data.pocket_pos: FloatTensor，形状为 (P, 3)，变换后与配体同原点的口袋局部坐标，单位 Å。
            - data.pocket_is_backbone: BoolTensor，形状为 (P,)，真值表示口袋原子属于蛋白主链。
            - data.pocket_atom_name: list[str]，长度为 P，PDB 原子名。
            - data.pocket_atom_to_aa_type: LongTensor，形状为 (P,)，口袋原子所属氨基酸类别。
            - data.pocket_molecule_name: str|None，口袋 PDB ``HEADER`` 名称。
            - data.bond_rotatable: LongTensor，形状为 (2M,)，与双向键对齐的可旋转标记。
            - data.tor_twisted_pairs: dict[tuple[int, int], list[set[int], set[int]]]，可旋转键两侧非轴原子集合。
            - data.fixed_dist_torsion: Tensor，形状为 (N, N)，1 表示原子对距离不随内部扭转改变。
            - data.tor_bond_mat: Tensor，形状为 (N, N)，逐原子对可旋转键标记。
            - data.path_mat: Tensor，形状为 (N, N)，化学图最短路径长度，单位为键数。
            - data.nbh_dict: dict[int, list[int]]，每个原子编号到一跳邻居编号列表的映射。
            - data.matches_graph: LongTensor，形状为 (M_g, S_g)，忽略手性的压缩自同构映射。
            - data.matches_iso: LongTensor，形状为 (M_i, S_i)，考虑手性的压缩自同构映射。
            - data.brics.subgraphs: list[list[int]]，BRICS 片段原子编号。
            - data.brics.anchors_list: list[set[int]]，BRICS 片段锚原子编号。
            - data.brics.nbh_subgraphs: list[list[int]]，BRICS 片段邻接表。
            - data.brics.connections: dict[tuple[int, int], tuple[int, int]]，BRICS 有向片段连接的锚原子对。
            - data.mmpa.subgraphs: list[list[int]]，MMPA 片段原子编号。
            - data.mmpa.anchors_list: list[set[int]]，MMPA 片段锚原子编号。
            - data.mmpa.nbh_subgraphs: list[list[int]]，MMPA 片段邻接表。
            - data.mmpa.connections: dict[tuple[int, int], tuple[int, int]]，MMPA 有向片段连接的锚原子对。
            - data.pocket_atom_feature: FloatTensor，形状为 (P, 25)，口袋元素、氨基酸和主链标记特征。
            - data.pocket_knn_edge_index: LongTensor，形状为 (2, E_p)，口袋 kNN 有向边端点。
            - data.pocket_center: FloatTensor，形状为 (1, 3)，世界坐标到模型局部坐标的平移原点，单位 Å。
            - data.num_nodes: int 标量 N，PyG 配体节点数。
            - data.node_type: LongTensor，形状为 (N,)，模型原子类别编号。
            - data.node_pos: FloatTensor，形状为 (N, 3)，配体局部坐标，单位 Å。
            - data.i_conf: int 标量，本候选实际选择的输入 conformer 编号。
            - data.halfedge_index: LongTensor，形状为 (2, H)，完全图上三角半边端点，索引配体原子维。
            - data.halfedge_type: LongTensor，形状为 (H,)，完全图半边类别，0 表示非键。
            - data.is_peptide: LongTensor，形状为 (N,)，小分子 use/docking 为全 0。
            - data.task_setting: str，当前 ``free``、``flexible``、``torsional`` 或 ``rigid`` 运动模式。
            - data.fixed_node: LongTensor，形状为 (N,)，0 表示原子类别待恢复，1 表示条件。
            - data.fixed_pos: LongTensor，形状为 (N,)，0 表示坐标待恢复，1 表示条件。
            - data.fixed_halfedge: LongTensor，形状为 (H,)，0 表示半边类别待恢复，1 表示条件。
            - data.fixed_halfdist: LongTensor，形状为 (H,)，0 表示端点距离可恢复，1 表示必须保持。
            - data.n_domain: LongTensor 标量，刚体域数。
            - data.domain_node_index: LongTensor，形状为 (2, K)，第一行是刚体域号，第二行是域内原子编号。
            - data.tor_bonds_anno: LongTensor，形状为 (T, 3)，每行是执行层级和两个扭转轴端点。
            - data.twisted_nodes_anno: LongTensor，形状为 (W, 2)，每行是扭转行号和随动原子编号。
            - data.dihedral_pairs_anno: LongTensor，形状为 (Q, 3)，每行是扭转行号和两个二面角外侧端点。
            - data.gt_node_type: LongTensor，形状为 (N,)，推理模式保留的原子类别真值副本。
            - data.gt_node_pos: FloatTensor，形状为 (N, 3)，推理模式保留的局部坐标真值副本，单位 Å。
            - data.gt_halfedge_type: LongTensor，形状为 (H,)，推理模式保留的半边类别真值副本。
            - data.fixed_halfdist_flex: LongTensor，形状为 (H,)，推理模式保留的原始柔性距离约束。

        注意:
            - ``deepcopy`` 保证不同候选的原地加噪、task prompt 与当前状态互不共享。
            - ``pocket_atom_feature`` 及其后的模型输入、prompt、运动注释和 ``gt_*`` 叶只在 ``transforms`` 非空且对应变换启用时存在。
        """
        # ``data``：深拷贝隔离不同重复样本；后续噪声器会原地覆盖 ``node_pos``、``pos_in`` 等字段。
        data = deepcopy(self.data)
        # ``data.task``：str，当前 use 推理任务名。
        # ``data.db``：str，固定为 ``use``，表示不从逻辑 LMDB 数据库读取。
        # ``data.key``：str，固定为空，表示没有 LMDB 复合键。
        data.update({
            # ``data.task``：str，当前 use 推理任务名。
            'task': self.task,
            # ``data.db``：str，固定为 ``use``。
            'db': 'use',
            # ``data.key``：str，固定为空字符串。
            'key': '',
        })
        if self.transforms is not None:
            # ``data``：单样本容器，执行变换后新增模型特征、fixed prompt 与几何注释。
            data = self.transforms(data)
        return data

class TestTaskDataset(Dataset):
    """
    把单一任务—数据库的 ``ForeverTaskDataset`` 包装成有限长度的测试数据集。

    构造参数:
        - ``*args``/``**kwargs``: 原样传给 ``ForeverTaskDataset``；正式测试配置要求 ``mode='test'`` 且 ``task_db_weights`` 含 ``name`` 与 ``db``。

    长度与索引:
        - ``size``: int, assembly 中该任务—数据库可读取的样本数；由 ``task_db_size(task_dbs[0])`` 解析回退规则后得到。
        - ``index``: int, 范围 ``[0, size - 1]``；直接交给内部数据集的 ``__getitem__``，不使用无限迭代器与 worker 分片。

    返回值:
        - 单个已变换样本；字段契约与 ``ForeverTaskDataset.__getitem__`` 相同。
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        # ``self.forever_dataset``：ForeverTaskDataset，测试模式下仅使用其按索引读取能力，不调用无限迭代器。
        self.forever_dataset = ForeverTaskDataset(*args, **kwargs)
        # ``self.size``：int，初始化时缓存当前任务—数据库组合的有限样本数。
        self.size = len(self)
        
    def __getitem__(self, index):
        if index >= self.size:
            raise IndexError(f'index {index} out of range {len(self)}')
        return self.forever_dataset[index]
    
    def __len__(self):
        return self.forever_dataset.task_db_size(self.forever_dataset.task_dbs[0])
        # return self.forever_dataset.total_size
    


# class MaxSizeDatasetWrapper(IterableDataset):
#     def __init__(self, dataset, max_size,
#                  follow_batch, exclude_keys):
#         super().__init__()
#         self.dataset = dataset
#         self.max_size = max_size
#         self.follow_batch = follow_batch
#         self.exclude_keys = exclude_keys

#     def __iter__(self) -> Iterator:
#         cum_size = 0
#         data_list = []
#         for data in self.dataset:
#             cum_size += data.num_atoms
#             if cum_size > self.max_size:
#                 yield Batch.from_data_list(data_list, follow_batch=self.follow_batch,
#                                            exclude_keys=self.exclude_keys)
#                 cum_size = data.num_atoms
#                 data_list = [data]
#             else:
#                 data_list.append(data)


class ForeverTaskDataset(IterableDataset):
    """
    按任务权重和任务内数据库权重持续采样 PocketXMol 训练/验证样本。

    构造参数:
        - dataset_cfg: 映射式配置；``root`` 为数据根目录，``assembly_path`` 为索引文件相对路径，``dbs`` 为数据库配置列表。
        - dataset_cfg.dbs[*].name: str, 逻辑数据库名；被 ``task_db_weights.*.db_ratio`` 和 assembly 键共同引用。
        - dataset_cfg.dbs[*].lmdb_root: str, 相对 ``root`` 的 LMDB 目录。
        - dataset_cfg.dbs[*].lmdb_path: dict[str, str], 子库名到 LMDB 文件名的映射，例如 ``mols``/``pocmol10``、``torsion`` 与 ``decom``。
        - task_db_weights: 训练/验证时为 ``dict[task, config]``；测试时为含 ``name``、``db`` 的单任务映射。
        - task_db_weights.<task>.weight: float, 任务采样概率；传给 ``np.random.choice``，同层权重必须合计为 1。
        - task_db_weights.<task>.db_ratio: dict[str, float], 当前任务可用数据库的相对权重；构造时归一化为概率。
        - mode: str, ``train`` 持续循环，``val`` 走完一次 worker 分片后停止，``test`` 只提供按索引读取。
        - split: str|None, assembly 文件后缀使用的划分名；``None`` 时取 ``mode``。
        - transforms: callable|None, 在单样本尚未进入 PyG ``DataLoader`` 前执行的变换链。
        - shuffle: bool, True 时在 worker 所属数据库区间内随机取索引；False 时顺序循环该区间。
        - global_rank: int，当前 DDP 进程编号。
        - world_size: int，DDP 进程总数。
        - num_workers: int，每个进程的 DataLoader worker 数。

    assembly 契约:
        - ``*_train.pkl``/``*_val.pkl``: dict，或相同键值契约的 ``*_train.lmdb``/``*_val.lmdb``。
        - all_dbs: list[str], 当前划分包含的数据库名；决定 worker 分片循环的数据库集合。
        - <db>: int, 该数据库当前划分的样本数。
        - <db>-<index>: str|None, 数据库局部索引到 ``data_id`` 的映射；缺失时依次尝试 ``<task>-<db>-<index>`` 和 ranker 单库键。
        - datatask_dbs: list[str], 测试/ranker assembly 中可用的任务—数据库组合；回退时使用第一个名称。
        - <task>-<db>: int|None, 特定任务—数据库的测试样本数；缺失时回退到 ``<db>`` 再回退到 ``datatask_dbs[0]``。

    单样本读取:
        - 输入索引: 迭代模式使用 ``(task, db_name, index)``，测试模式使用数据库局部整数 ``index``。
        - LMDB 复合键: 分号分隔多个 ``<subdb>/<data_id>`` 片段；``SingleDatabase`` 按顺序读取并用 ``update`` 合并字段。
        - task: str, 当前任务名；构象生成使用 ``conf``，小分子 docking 使用 ``dock``。
        - db: str, 当前逻辑数据库名。
        - key: str, 实际读取的完整 LMDB 复合键，保留各子库片段顺序。
        - element: LongTensor，形状为 (N,)，配体原子序数。
        - pos_all_confs: FloatTensor，形状为 (C, N, 3)，输入 conformer 坐标，单位 Å。
        - i_conf_list: list[int]，长度为 C，合法 conformer 的输入编号。
        - num_confs: int 标量 C，合法 conformer 数。
        - bond_index: LongTensor，形状为 (2, 2M)，双向化学键端点。
        - bond_type: LongTensor，形状为 (2M,)，与 ``bond_index`` 列对齐的键类别。
        - num_atoms: int 标量 N，配体原子数。
        - num_bonds: int 标量 M，无向化学键数。
        - pocket_element: LongTensor，形状为 (P,)，docking 口袋原子序数；构象数据可缺失。
        - pocket_pos: FloatTensor，形状为 (P, 3)，docking 口袋坐标；变换后与配体使用同一局部原点，单位 Å。
        - pocket_is_backbone: BoolTensor，形状为 (P,)，口袋原子主链标记。
        - pocket_atom_name: list[str]，长度为 P，口袋 PDB 原子名。
        - pocket_atom_to_aa_type: LongTensor，形状为 (P,)，口袋原子所属氨基酸类别。
        - pocket_molecule_name: str|None，口袋 PDB ``HEADER`` 名称。
        - pdbid: str，受体结构标识。
        - data_id: str，当前样本标识。
        - smiles: str，固定二维配体图的规范 SMILES。
        - bond_rotatable: LongTensor，形状为 (2M,)，与双向键对齐的可旋转标记。
        - tor_twisted_pairs: dict[tuple[int, int], list[set[int], set[int]]]，可旋转键两侧非轴原子集合。
        - fixed_dist_torsion: Tensor，形状为 (N, N)，扭转下保持距离的 0/1 矩阵。
        - tor_bond_mat: Tensor，形状为 (N, N)，逐原子对可旋转键标记。
        - path_mat: Tensor，形状为 (N, N)，化学图最短路径，单位为键数。
        - nbh_dict: dict[int, list[int]]，逐原子一跳邻居编号。
        - matches_graph: LongTensor，形状为 (M_g, S_g)，忽略手性的压缩自同构映射。
        - matches_iso: LongTensor，形状为 (M_i, S_i)，考虑手性的压缩自同构映射。
        - brics.subgraphs: list[list[int]]，BRICS 片段原子编号。
        - brics.anchors_list: list[set[int]]，BRICS 片段锚原子编号。
        - brics.nbh_subgraphs: list[list[int]]，BRICS 片段邻接表。
        - brics.connections: dict[tuple[int, int], tuple[int, int]]，BRICS 有向片段连接的锚原子对。
        - mmpa.subgraphs: list[list[int]]，MMPA 片段原子编号。
        - mmpa.anchors_list: list[set[int]]，MMPA 片段锚原子编号。
        - mmpa.nbh_subgraphs: list[list[int]]，MMPA 片段邻接表。
        - mmpa.connections: dict[tuple[int, int], tuple[int, int]]，MMPA 有向片段连接的锚原子对。
        - pocket_atom_feature: FloatTensor，形状为 (P, 25)，变换后的口袋离散特征。
        - pocket_knn_edge_index: LongTensor，形状为 (2, E_p)，变换后的口袋 kNN 有向边。
        - pocket_center: FloatTensor，形状为 (1, 3)，世界坐标到模型局部坐标的原点，单位 Å。
        - num_nodes: int 标量 N，PyG 配体节点数。
        - node_type: LongTensor，形状为 (N,)，模型原子类别编号。
        - node_pos: FloatTensor，形状为 (N, 3)，模型局部配体坐标，单位 Å。
        - i_conf: int 标量，本样本选择的 conformer 编号。
        - halfedge_index: LongTensor，形状为 (2, H)，完全图半边端点。
        - halfedge_type: LongTensor，形状为 (H,)，完全图半边类别。
        - is_peptide: LongTensor，形状为 (N,)，小分子任务为全 0。
        - task_setting: str，当前运动模式；训练模式会在 DataLoader 前排除该非张量字段。
        - fixed_node: LongTensor，形状为 (N,)，原子类别条件掩码。
        - fixed_pos: LongTensor，形状为 (N,)，坐标条件掩码。
        - fixed_halfedge: LongTensor，形状为 (H,)，半边类别条件掩码。
        - fixed_halfdist: LongTensor，形状为 (H,)，半边距离条件掩码。
        - n_domain: LongTensor 标量，刚体域数。
        - domain_node_index: LongTensor，形状为 (2, K)，刚体域—原子归属索引。
        - tor_bonds_anno: LongTensor，形状为 (T, 3)，扭转层级与轴端点。
        - twisted_nodes_anno: LongTensor，形状为 (W, 2)，扭转行号与随动原子索引。
        - dihedral_pairs_anno: LongTensor，形状为 (Q, 3)，扭转行号与二面角外侧端点。

    迭代边界:
        - ``sampler_index_list[g][db]`` 是第 g 个全局采样器在数据库 db 上的半开区间 ``(start, end)``。
        - 每轮每个全局采样器产生 ``total_size // total_samplers`` 个样本；余数不会在该轮额外产出。
        - 任务变换在 DataLoader 批处理前执行，因此 ``MixedTransform`` 和 ``MixedSampleNoiser`` 每次只接收一个已确定任务的样本。
    """
    def __init__(self, dataset_cfg, task_db_weights, mode,
                split=None, transforms=None, shuffle=False, **kwargs):
        super().__init__()
        
        # ``self.dataset_cfg``：EasyDict，保存数据根、assembly 相对路径与逻辑数据库列表。
        self.dataset_cfg = dataset_cfg
        # ``self.mode``：str，train/val/test；决定采样表与迭代终止行为。
        self.mode = mode
        assert mode in ['train', 'val', 'test'], 'Unknown mode: %s' % mode
        if split is None:
            # ``split``：str，缺省沿用 mode，决定 assembly 文件名后缀。
            split = mode
        # ``self.split``：str，实际读取的 assembly 划分名。
        self.split = split
        # ``self.transforms``：callable|None，LMDB 字段合并后、PyG 拼批前执行的单样本变换链。
        self.transforms = transforms
        # ``self.shuffle``：bool，True 在 worker 区间内随机取索引，False 为每库独立顺序循环。
        self.shuffle = shuffle
        
        # 训练/验证的两级采样表：先按 ``task_weights`` 选任务，再按归一化后的 ``db_weights[task]`` 选数据库。
        if mode in ['train', 'val']:
            # ``self.task_dbs``：list[tuple[str, str]]，列出配置中每个任务允许读取的全部任务—数据库组合。
            # ``task``：str，当前 ``task_db_weights`` 顶层任务名。
            # ``value``：Mapping，当前任务的 ``weight`` 与 ``db_ratio`` 配置叶。
            # ``db_name``：str，当前 ``value.db_ratio`` 中允许读取的逻辑数据库名。
            self.task_dbs = [(task, db_name) for task, value in task_db_weights.items() for db_name in value['db_ratio']]
            # ``self.task_weights.tasks``：list[str]，按配置迭代顺序保存可采样任务名。
            # ``self.task_weights.weights``：list[float]，与 ``tasks`` 逐项对齐，原样传给 NumPy 的概率参数 p。
            # ``self.task_weights``：dict，保存以上任务类别与第一级采样权重叶。
            self.task_weights = {
                # ``tasks``：list[str]，第 i 项是第一级采样的任务候选。
                'tasks': list(task_db_weights.keys()),
                # ``weights``：list[float]，第 i 项是 ``tasks[i]`` 的未归一化配置权重。
                # ``value``：Mapping，当前任务配置；此推导式只读取其 ``weight`` 叶。
                'weights': [value['weight'] for value in task_db_weights.values()]
            }
            # ``self.db_weights[task].dbs``：list[str]，当前任务允许读取的逻辑数据库名。
            # ``self.db_weights[task].weights``：ndarray，形状为 (D_t,)，与 ``dbs`` 逐项对齐且总和为 1。
            # ``self.db_weights``：dict[str, dict]，保存每个任务的以上第二级数据库采样叶。
            # ``task``：str，当前 ``task_db_weights`` 顶层任务名。
            # ``value``：Mapping，当前任务的 ``weight`` 与 ``db_ratio`` 配置叶。
            self.db_weights = {task: {
                    # ``dbs``：list[str]，当前 task 的第二级采样候选。
                    'dbs': list(value['db_ratio'].keys()),
                    # ``weights``：ndarray，形状为 (D_t,)，当前 task 的归一化数据库概率。
                    'weights': np.array(list(value['db_ratio'].values())) / sum(value['db_ratio'].values())
                } for task, value in task_db_weights.items()
            }

        else:
            # ``task``：str，测试配置指定的唯一任务名。
            task = task_db_weights['name']
            # ``db_name``：str，测试配置指定的唯一逻辑数据库名。
            db_name = task_db_weights['db']
            # ``self.task_dbs``：list[tuple[str,str]] 长度 1，统一测试模式与训练模式的任务—数据库表示。
            self.task_dbs = [(task, db_name)]
            

        # ``self.root``：``assembly_path`` 在 setup 中按 ``split`` 改写后缀；dbs_config 决定各逻辑数据库可合并的子 LMDB。
        self.root = dataset_cfg['root']
        # ``self.assembly_path``：path，data root 与 assembly 相对路径的拼接结果；setup 再按 split 改后缀。
        self.assembly_path = os.path.join(self.root, dataset_cfg['assembly_path'])
        # ``self.dbs_config``：list[config]，每项定义一个逻辑数据库的 lmdb_root 与 lmdb_path 叶子。
        self.dbs_config = dataset_cfg['dbs']
        # used dbs
        # self.used_dbs = [db.name for db in self.dbs_config]
        
        # ``self.catalog_dict``：dict|LMDBDatabase|None，setup(set_index=True) 后承载 assembly 键值索引。
        self.catalog_dict = None
        # ``self.db_dict``：dict[str,SingleDatabase]|None，setup(set_db=True) 后承载各逻辑数据库连接。
        self.db_dict = None
        # run only once
        self.setup(set_index=True, set_db=True)
        
        # get global id
        if mode != 'test':
            # ``self.gpu_rank``：int，当前 DDP 进程的全局 rank。
            self.gpu_rank = kwargs['global_rank']
            # ``self.world_size``：int，DDP 进程总数。
            self.world_size = kwargs['world_size']
            # ``self.num_workers``：int，每个进程的 DataLoader worker 数。
            self.num_workers = kwargs['num_workers']
            self.setup(set_iter=True)


    def setup(self, set_index=False, set_db=False, set_iter=False,
              ):
        """
        分别初始化 assembly 索引、子 LMDB 连接和全局 worker 索引区间。

        输入开关:
            - set_index: bool, 读取 assembly 并设置 ``catalog_dict``。
            - set_db: bool, 按 ``dataset_cfg.dbs`` 构造 ``db_dict: dict[str, SingleDatabase]``。
            - set_iter: bool, 按 ``num_workers * world_size`` 为每个数据库建立互不重叠的半开索引区间。

        ``catalog_dict`` 字段:
            - CSV 测试 assembly: ``datatask_dbs`` 保存唯一任务—数据库名，``<datatask-db>`` 保存行数，``<datatask-db>-<i>`` 保存第 i 行 ``data_id``。
            - pickle.all_dbs: list[str]，当前划分包含的逻辑数据库名。
            - pickle.<db>: int，数据库 ``db`` 在当前划分的样本数。
            - pickle.<db>-<index>: str|None，数据库局部索引到 ``data_id`` 的映射。
            - pickle.datatask_dbs: list[str]，测试或 ranker assembly 可用的 ``<task>-<db>`` 组合。
            - pickle.<task>-<db>: int|None，特定任务—数据库组合的样本数。
            - pickle 文件路径: 把 ``assembly_path`` 的 ``.pkl`` 替换为 ``_<split>.pkl``。
            - LMDB.all_dbs: list[str]，与 pickle 同义的数据库名列表叶。
            - LMDB.<db>: int，与 pickle 同义的数据库样本数叶。
            - LMDB.<db>-<index>: str|None，与 pickle 同义的索引—样本标识叶。
            - LMDB.datatask_dbs: list[str]，与 pickle 同义的任务—数据库组合叶。
            - LMDB.<task>-<db>: int|None，与 pickle 同义的组合样本数叶。
            - LMDB 文件路径: 把 ``assembly_path`` 的 ``.lmdb`` 替换为 ``_<split>.lmdb``。

        ``sampler_index_list`` 结构:
            - 外层: ``list[dict]``，长度为 ``total_samplers = num_workers * world_size``。
            - 每个元素: ``dict[db_name, tuple[int, int]]``，值是该全局采样器在数据库局部索引上的半开区间。
            - 分配规则: 每个区间先获得 ``floor(db_size / total_samplers)`` 个索引，余数按循环采样器编号各补 1 个。
        """
        if set_index:
            # setup index
            if self.assembly_path.endswith('.csv'):  # for test only. use the data_id columns
                # ``df_ass``：DataFrame, (S, C)，测试 CSV assembly；必须含 data_id 列。
                df_ass = pd.read_csv(self.assembly_path)
                # ``data_id_list``：object, (S,), 第 i 个值是测试 CSV 第 i 行的分子/复合物标识。
                data_id_list = df_ass['data_id'].values
                assert len(self.task_dbs) == 1, 'only test can use csv assembly'
                # ``datatask_db``：str ``<task>-<db>``，唯一测试任务—数据库组合的 assembly 前缀。
                datatask_db = '-'.join(self.task_dbs[0])
                # ``db_size``：int S，测试 CSV 行数。
                db_size = len(data_id_list)
                # ``catalog_dict``：dict，先写组合清单与总数，随后追加逐索引 data_id 键。
                catalog_dict = {
                    # ``catalog_dict.datatask_dbs``：list[str]，只含当前 CSV 对应的任务—数据库组合名。
                    'datatask_dbs': [datatask_db],
                    # ``catalog_dict[datatask_db]``：int，当前 CSV 的样本行数 S。
                    datatask_db: db_size,
                }
                # ``i``：int，当前 CSV 行号，取值范围为 ``[0, S)``，同时索引 ``data_id_list``。
                catalog_dict.update({
                    f'{datatask_db}-{i}':data_id_list[i] for i in range(len(data_id_list))
                })
            elif self.assembly_path.endswith('.pkl'):
                # raise NotImplementedError
                # ``assembly_path``：path，把模板 .pkl 后缀替换为 _<split>.pkl。
                assembly_path = self.assembly_path.replace('.pkl', f'_{self.split}.pkl')
                # ``f``：二进制文件句柄，指向当前划分的 pickle assembly。
                with open(assembly_path, 'rb') as f:
                    # ``catalog_dict``：dict，pickle 中保存的 assembly 键值映射。
                    catalog_dict = pickle.load(f)
                    # catalog_dict = {task_db:value for task_db, value in catalog_dict.items()}
            elif self.assembly_path.endswith('.lmdb'):
                # ``assembly_path``：path，把模板 .lmdb 后缀替换为 _<split>.lmdb。
                assembly_path = self.assembly_path.replace('.lmdb', f'_{self.split}.lmdb')
                # ``catalog_dict``：LMDBDatabase，只读 assembly 键值包装；允许最多 126 个读者。
                catalog_dict = LMDBDatabase(assembly_path, max_readers=126)
            else:
                raise ValueError('Unsupported appendix of assembly_path', self.assembly_path)
            # if self.shuffle:
            #     self.catalog_dict = {task_db: np.random.permutation(value) for task_db, value in catalog_dict.items()}
            # else:
            self.catalog_dict = catalog_dict

        if set_db:
            # load db
            # ``db_dict``：dict[str, SingleDatabase]，键为 dbs 配置中的逻辑数据库名。
            db_dict = {}
            # ``db_config.name``：str，当前逻辑数据库名。
            # ``db_config.lmdb_root``：str，当前逻辑数据库的子 LMDB 相对根目录。
            # ``db_config.lmdb_path``：dict[str, str]，当前逻辑数据库的子库名到 LMDB 文件名映射。
            # ``db_config``：配置映射，保存以上三个数据库布局叶。
            for db_config in self.dbs_config:
                # ``db_name``：str，逻辑数据库名；同时是 db_dict、assembly all_dbs 和采样权重的公共键。
                db_name = db_config.name
                # ``db_dict``：SingleDatabase，打开该逻辑库配置列出的所有只读子 LMDB。
                db_dict[db_name] = SingleDatabase(db_config, self.root)
                # db_dict[db_name] = DB_DICT[db_name](db_config, self.root)
            # ``self.db_dict``：dict[str, SingleDatabase]，完成全部逻辑数据库只读连接后的注册映射。
            self.db_dict = db_dict

        if set_iter:
            # global_id, total_samplers = self._get_global_id(self.gpu_rank, self.world_size)

            # int G，每进程 worker 数乘 DDP 进程数；也是 sampler_index_list 外层长度。
            self.total_samplers = total_samplers = self.num_workers * self.world_size
            # ``samplers_cycler``：无限整数迭代器；只用于轮转“哪个全局采样器获得当前数据库的一个余数索引”。
            samplers_cycler = shuffled_cyclic_iterator(total_samplers, shuffle=self.shuffle)
            # print(self.mode,  total_samplers, id(self.catalog_dict))
            
            # ``_``：int，当前初始化的全局采样器编号；值本身不写入空映射。
            # ``sampler_index_list``：list[dict]，长度 G，第 g 项将保存每个数据库分配给全局采样器 g 的半开区间。
            sampler_index_list = [{} for _ in range(total_samplers)]
            # ``db_name``：str，当前分配索引区间的逻辑数据库名，索引 ``catalog_dict`` 的数据库样本数叶。
            for db_name in self.catalog_dict['all_dbs']:
                
                # calc basic num per sampler
                db_size = self.catalog_dict[db_name]
                # ``size_per_sampler``：浮点整值；每个全局采样器在当前数据库中必得的基础样本数。
                size_per_sampler = np.floor(db_size / total_samplers)

                # who should have a bonus
                n_remaining = db_size % total_samplers
                # ``_``：int，当前余数奖励序号；只控制从 ``samplers_cycler`` 取值的次数。
                # ``index_sampler_bonus``：list[int]，长度为 n_remaining，记录本数据库各余数索引依次奖励给哪个全局采样器。
                index_sampler_bonus = [next(samplers_cycler) for _ in range(n_remaining)]
                # ``who_has_bonus``：bool, (total_samplers,), True 表示该全局采样器在当前数据库中多分到一个索引。
                who_has_bonus = np.zeros(total_samplers, dtype=bool)
                # ``who_has_bonus``：(G,) 掩码写入，把本数据库获得额外一个索引的采样器标为 True。
                who_has_bonus[index_sampler_bonus] = True
                
                # ``global_id``：int，当前全局采样器编号，索引 ``sampler_index_list`` 第一维与 ``who_has_bonus``。
                for global_id in range(total_samplers):
                    # calc index for this sampler
                    # ``index_start``：int，当前采样器在本数据库局部索引空间中的半开区间起点。
                    index_start = int(global_id * size_per_sampler + np.sum(who_has_bonus[:global_id]))
                    # ``index_end``：int，当前采样器在本数据库的局部半开区间终点；获得 bonus 时比基础长度多 1。
                    index_end = int(index_start + size_per_sampler + int(who_has_bonus[global_id]))
                    sampler_index_list[global_id].update({db_name: (index_start, index_end)})
                    
            # ``self.sampler_index_list``：list[dict] 长度 G，完成所有数据库分区后的只读索引范围表。
            self.sampler_index_list = sampler_index_list
            

    def _get_global_id(self):
        """
        把 DDP rank 与当前 DataLoader worker 编号映射为全局采样器编号。

        返回值:
            - global_id: int, ``gpu_rank * num_workers + worker_id``；索引 ``sampler_index_list`` 第一维，范围为 ``[0, total_samplers - 1]``。

        关键检查:
            - 运行时 ``get_worker_info().num_workers`` 必须与构造数据集时记录的 ``self.num_workers`` 一致。
        """
        # per gpu
        worker_info = get_worker_info()
        # ``worker_id``：int，当前进程内 DataLoader worker 编号；主进程直读时为 0。
        worker_id = worker_info.id if worker_info is not None else 0
        # ``num_workers``：int，运行时 DataLoader worker 总数；主进程直读时为 1。
        num_workers = worker_info.num_workers if worker_info is not None else 1
        assert num_workers == self.num_workers
        # ``global_id``：int，跨 DDP rank 唯一的采样器编号，索引 sampler_index_list 第一维。
        global_id = self.gpu_rank * self.num_workers + worker_id
        print(f'The processor {global_id}/{self.total_samplers} ({self.gpu_rank}-{worker_id}) is ready for {self.split}!')
        return global_id

    def __iter__(self):
        """
        生成当前 DDP rank/DataLoader worker 专属的任务—数据库样本流。

        每次产出:
            - 一个已经执行 ``transforms`` 的单样本 ``PocketMolData``；任务和数据库在读取该样本前已经确定。

        采样顺序:
            - 先用 ``task_weights`` 选择任务，再用 ``db_weights[task]`` 选择数据库。
            - ``shuffle=True`` 时从当前 worker 的 ``(start, end)`` 中均匀随机取数据库局部索引。
            - ``shuffle=False`` 时每个数据库维护独立循环迭代器，按局部索引顺序取值。
            - ``train`` 完成一轮后继续；``val`` 完成一轮后退出。
        """
        # ``global_id``：int，当前 DDP rank/worker 的全局采样器编号。
        global_id = self._get_global_id()
        # self.task_inf_iter = {(task, db): shuffled_cyclic_iterator((int(index_start), int(index_end)), shuffle=self.shuffle)
        #                       for task, db, index_start, index_end in self.sampler_index_list[global_id]}
        print('iter from global_id:', global_id)
        # ``sample_index``：dict[db_name, tuple[int, int]]，只覆盖当前全局采样器可访问的数据库局部索引区间。
        sample_index = self.sampler_index_list[global_id]
        if not self.shuffle:
            # ``db_name``：str，当前为顺序采样建立独立循环器的逻辑数据库名。
            # ``index_range``：tuple[int, int]，当前 worker 在该数据库局部索引上的半开区间 ``(start, end)``。
            # ``task_inf_iter``：dict[str, cycle]，每个数据库独立循环其分配半开区间，避免切换任务时共享游标。
            task_inf_iter = {db_name: cycle(range(*index_range)) 
                             for db_name, index_range in sample_index.items()}
        while True:
            # ``_``：int，当前轮内的产出计数；只控制每个全局采样器的固定产出次数，不参与样本定位。
            for _ in range(self.total_size//self.total_samplers):
                # ``task``：str，按顶层任务概率采到的当前单样本任务名。
                task = np.random.choice(self.task_weights['tasks'], p=self.task_weights['weights'])
                # ``task_db_weight.dbs``：list[str]，当前任务允许采样的逻辑数据库名。
                # ``task_db_weight.weights``：ndarray，形状为 (D_t,)，与 ``dbs`` 对齐且总和为 1 的数据库概率。
                # ``task_db_weight``：dict，当前任务的上述第二级数据库采样叶。
                task_db_weight = self.db_weights[task]
                # ``db``：str，在当前任务允许的数据库中按归一化概率采到的逻辑库名。
                db = np.random.choice(task_db_weight['dbs'], p=task_db_weight['weights'])
                # get an index
                if self.shuffle:
                    # ``index``：int，在当前 worker 对 db 的半开区间内均匀随机取得数据库局部索引。
                    index = np.random.randint(*sample_index[db])
                else:
                    # ``index``：int，从 db 独立循环迭代器取得下一个数据库局部索引。
                    index = next(task_inf_iter[db])
                yield self[(task, db, index)]
            if self.mode != 'train':
                break

            
    def get_data_key(self, task, db_name, data_id):
        """
        根据逻辑数据库布局构造 ``SingleDatabase`` 可解析的复合 LMDB 键。

        输入参数:
            - task: str, 当前任务名；``conf`` 与 ``dock`` 不追加任务专用子库。
            - db_name: str, ``dataset_cfg.dbs[*].name`` 中的逻辑数据库名。
            - data_id: str, assembly 为当前数据库局部索引解析出的分子或复合物标识。

        返回值:
            - key: str, 分号分隔的有序 ``<subdb>/<record-key>`` 列表；第一个子库提供基础 ``PocketMolData``，后续子库用 ``update`` 补充扭转或分解字段。

        构象/docking 相关分支:
            - geom: ``mols/<data_id>;torsion/<data_id>;decom/<data_id>``。
            - qm9: ``mols/<data_id>;torsion/<data_id>;decom/<data_id>``。
            - unmi: ``mols/<data_id>;torsion/<data_id>;decom/<data_id>``。
            - csd: ``pocmol10/<data_id>;torsion/<data_id>;decom/<data_id>``。
            - pbdock: ``pocmol10/<data_id>;torsion/<data_id>;decom/<data_id>``。
            - pbdockS: ``pocmol10/<data_id>;torsion/<data_id>;decom/<data_id>``。
            - pbdockL: ``pocmol10/<data_id>;torsion/<data_id>;decom/<data_id>``。
            - moad: ``pocmol10/<data_id>;torsion/<data_id>;decom/<data_id>``。
            - cremp: ``mols/<data_id>``，没有独立扭转与分解子库。
            - poseb: ``pocmol10/<data_id>``，用于 docking 测试集。
            - poseboff: ``pocmol10/<data_id>``，用于 docking 测试集。
            - 其他数据库: ``<db_name>/<data_id>``，供自定义测试/使用配置按同名子库读取。
        """
        
        if task == 'linking':
            data_id_split = data_id.split('/')
            if len(data_id_split) == 2:  # actually it is sep_id
                data_id, index_sep = data_id_split
            else:
                data_id = data_id_split[0]
                index_sep = '0'
        
        # the new process version has merged the torsion and decom
        if db_name in ['geom', 'qm9', 'unmi', ]:
            # ``key``：str，基础分子 -> 扭转 -> 分解的有序复合键；后者同名字段覆盖前者。
            key = f'mols/{data_id};torsion/{data_id};decom/{data_id}'
        elif db_name in ['csd', 'pbdock', 'pbdockS', 'pbdockL', 'moad']:
            # ``key``：str，基础口袋-配体复合物 -> 扭转 -> 分解的有序复合键。
            key = f'pocmol10/{data_id};torsion/{data_id};decom/{data_id}'
        elif db_name in ['cremp', 'protacdb']:
            # ``key``：str，仅含基础分子子库的复合键。
            key = f'mols/{data_id}'
        elif db_name in ['apep']:
            key = f'pocmol10/{data_id};torsion/{data_id};decom/{data_id};peptide/{data_id}'
        elif db_name in ['pepbdb', 'qbpep']:
            key = f'pocmol10/{data_id};peptide/{data_id}'
        elif db_name in ['poseb', 'poseboff']:
            # ``key``：str，docking 测试集只读取基础口袋-配体记录。
            key = f'pocmol10/{data_id}'
        else:  # new db. typically for test/use
            # raise ValueError(f'unknown db_name {db_name}')
            # ``key``：str，自定义数据库约定子库名与逻辑 db_name 相同。
            key = f'{db_name}/{data_id}'
        
        if task == 'linking':
            key += f';linking/{data_id}/{index_sep}'
        elif task == 'growing':
            key += f';growing/{data_id}'
        return key


    def __getitem__(self, index):
        """
        解析一个任务—数据库索引，合并 LMDB 字段并执行单样本变换。

        输入参数:
            - index: int|tuple, 测试模式使用数据库局部整数；迭代模式通常使用 ``(task, db_name, idx)``，四元组额外接受但不使用首项 ``datatask``。

        assembly 查找顺序:
            - 先读 ``<db_name>-<idx>``。
            - 值为 ``None`` 时再读 ``<task>-<db_name>-<idx>``。
            - 仍为 ``None`` 时取 ``datatask_dbs[0]``，再读 ``<datatask-db>-<idx>``。

        返回字段:
            - task: str, 当前任务名。
            - db: str, 当前逻辑数据库名。
            - key: str, ``get_data_key`` 生成并实际读取的复合键。
            - data_id: str，assembly 当前索引解析出的样本标识。
            - pdbid: str，受体结构标识。
            - smiles: str，固定二维配体图的规范 SMILES。
            - element: LongTensor，形状为 (N,)，配体原子序数。
            - pos_all_confs: FloatTensor，形状为 (C, N, 3)，输入 conformer 坐标，单位 Å。
            - i_conf_list: list[int]，长度为 C，合法 conformer 输入编号。
            - num_confs: int 标量 C，合法 conformer 数。
            - bond_index: LongTensor，形状为 (2, 2M)，双向化学键端点。
            - bond_type: LongTensor，形状为 (2M,)，逐双向键类别。
            - num_atoms: int 标量 N，配体原子数。
            - num_bonds: int 标量 M，无向化学键数。
            - pocket_element: LongTensor，形状为 (P,)，docking 口袋原子序数；构象数据可缺失。
            - pocket_pos: FloatTensor，形状为 (P, 3)，变换后与配体同原点的口袋局部坐标，单位 Å。
            - pocket_is_backbone: BoolTensor，形状为 (P,)，口袋原子主链标记。
            - pocket_atom_name: list[str]，长度为 P，PDB 原子名。
            - pocket_atom_to_aa_type: LongTensor，形状为 (P,)，口袋原子所属氨基酸类别。
            - pocket_molecule_name: str|None，口袋 PDB ``HEADER`` 名称。
            - bond_rotatable: LongTensor，形状为 (2M,)，可旋转键标记。
            - tor_twisted_pairs: dict[tuple[int, int], list[set[int], set[int]]]，可旋转键两侧非轴原子集合。
            - fixed_dist_torsion: Tensor，形状为 (N, N)，扭转下保持距离的 0/1 矩阵。
            - tor_bond_mat: Tensor，形状为 (N, N)，逐原子对可旋转键标记。
            - path_mat: Tensor，形状为 (N, N)，化学图最短路径，单位为键数。
            - nbh_dict: dict[int, list[int]]，逐原子一跳邻居编号。
            - matches_graph: LongTensor，形状为 (M_g, S_g)，忽略手性的压缩自同构映射。
            - matches_iso: LongTensor，形状为 (M_i, S_i)，考虑手性的压缩自同构映射。
            - brics.subgraphs: list[list[int]]，BRICS 片段原子编号。
            - brics.anchors_list: list[set[int]]，BRICS 片段锚原子编号。
            - brics.nbh_subgraphs: list[list[int]]，BRICS 片段邻接表。
            - brics.connections: dict[tuple[int, int], tuple[int, int]]，BRICS 片段连接锚原子对。
            - mmpa.subgraphs: list[list[int]]，MMPA 片段原子编号。
            - mmpa.anchors_list: list[set[int]]，MMPA 片段锚原子编号。
            - mmpa.nbh_subgraphs: list[list[int]]，MMPA 片段邻接表。
            - mmpa.connections: dict[tuple[int, int], tuple[int, int]]，MMPA 片段连接锚原子对。
            - pocket_atom_feature: FloatTensor，形状为 (P, 25)，口袋离散特征。
            - pocket_knn_edge_index: LongTensor，形状为 (2, E_p)，口袋 kNN 有向边。
            - pocket_center: FloatTensor，形状为 (1, 3)，模型局部坐标原点，单位 Å。
            - num_nodes: int 标量 N，PyG 配体节点数。
            - node_type: LongTensor，形状为 (N,)，模型原子类别编号。
            - node_pos: FloatTensor，形状为 (N, 3)，模型局部配体坐标，单位 Å。
            - i_conf: int 标量，当前选择的 conformer 编号。
            - halfedge_index: LongTensor，形状为 (2, H)，完全图半边端点。
            - halfedge_type: LongTensor，形状为 (H,)，完全图半边类别。
            - is_peptide: LongTensor，形状为 (N,)，小分子任务为全 0。
            - task_setting: str，当前运动模式；训练模式在批处理前排除。
            - fixed_node: LongTensor，形状为 (N,)，原子类别条件掩码。
            - fixed_pos: LongTensor，形状为 (N,)，坐标条件掩码。
            - fixed_halfedge: LongTensor，形状为 (H,)，半边类别条件掩码。
            - fixed_halfdist: LongTensor，形状为 (H,)，半边距离条件掩码。
            - n_domain: LongTensor 标量，刚体域数。
            - domain_node_index: LongTensor，形状为 (2, K)，刚体域—原子归属索引。
            - tor_bonds_anno: LongTensor，形状为 (T, 3)，扭转层级与轴端点。
            - twisted_nodes_anno: LongTensor，形状为 (W, 2)，扭转行号与随动原子编号。
            - dihedral_pairs_anno: LongTensor，形状为 (Q, 3)，扭转行号与二面角外侧端点。
        """
        # select task
        if isinstance(index, tuple):
            if len(index) == 4:
                # ``datatask``：str，四元索引中的数据任务前缀；当前函数后续不读取。
                # ``task``：str，当前图级任务名。
                # ``db_name``：str，当前逻辑数据库名。
                # ``idx``：int，当前数据库局部样本索引。
                datatask, task, db_name, idx = index
            else:
                # ``task``：str，训练或验证迭代器给出的图级任务名。
                # ``db_name``：str，训练或验证迭代器给出的逻辑数据库名。
                # ``idx``：int，训练或验证迭代器给出的数据库局部样本索引。
                task, db_name, idx = index
        else: # single task mode (usually for test)
            # ``task``：str，测试模式构造时指定的唯一图级任务名。
            # ``db_name``：str，测试模式构造时指定的唯一逻辑数据库名。
            task, db_name = self.task_dbs[0]
            # ``idx``：int，测试调用方给出的数据库局部索引。
            idx = index
        
        # prepare key
        if isinstance(self.catalog_dict, dict):
            # ``key``：str ``<db>-<idx>``，优先尝试的普通 dict assembly 键。
            key = '-'.join([db_name, str(idx)])
            # ``data_id``：str|None，命中时为 data_id；缺键显式设 None 以进入任务专用回退。
            data_id = self.catalog_dict[key] if key in self.catalog_dict else None
        else:
            # ``data_id``：str|None，LMDBDatabase assembly 对 ``<db>-<idx>`` 的读取结果。
            data_id = self.catalog_dict['-'.join([db_name, str(idx)])]
        if data_id is None:
            # ``data_id``：str|None，第二优先级 ``<task>-<db>-<idx>`` 的任务专用 data_id。
            data_id = self.catalog_dict['-'.join([task, db_name, str(idx)])]
            if data_id is None:  # use ranker for non-dock test sets
                # ``db``：str，ranker assembly 声明的第一个任务—数据库复合前缀。
                db = self.catalog_dict['datatask_dbs'][0]
                # ``data_id``：str|None，最终按 ``<datatask-db>-<idx>`` 读取 data_id。
                data_id = self.catalog_dict['-'.join([db, str(idx)])]

        # select db
        key = self.get_data_key(task, db_name, data_id)
        
        # select data
        # ``data`` 是第一个子 LMDB 保存的可更新容器；后续子库字段已由 ``SingleDatabase`` 原地合并。
        data = self.db_dict[db_name][key]
        # ``data.task``：str，当前采样任务名。
        # ``data.db``：str，当前逻辑数据库名。
        # ``data.key``：str，实际读取并合并的 LMDB 复合键。
        data.update({
            # ``data.task``：str，当前采样任务名。
            'task': task,
            # ``data.db``：str，当前逻辑数据库名。
            'db': db_name,
            # ``data.key``：str，实际读取的完整 LMDB 复合键。
            'key': key,
        })
        if self.transforms is not None:
            # ``data``：单样本容器，执行特征化、任务 prompt 与训练噪声后返回。
            data = self.transforms(data)
        #     data, level_dict = self.transforms(data)
        # if True:
        #     import time
        #     import torch
        #     new_data = deepcopy(data)
        #     new_data.update({f'level_{key}':value for key, value in level_dict.items()})
        #     torch.save(new_data, 'data/inputs/0115_pepbdb_1skg_B/{}.pt'.format(time.time()))
        return data
    
    @property
    def total_size(self):
        """返回 ``all_dbs`` 列出的数据库样本数之和；该值决定每个验证轮次的总产出上限。"""
        # if self.catalog_dict is None:
        #     self.setup(set_index=True)
        # ``db_name``：str，assembly ``all_dbs`` 中当前累加样本数的逻辑数据库名。
        return sum(self.catalog_dict[db_name] for db_name in self.catalog_dict['all_dbs'])

    # def datatask_db_size(self, datatask_db):
    #     # if self.catalog_dict is None:
    #     #     self.setup(set_index=True)
    #     # return len(self.catalog_dict[datatask_db])
    #     return self.catalog_dict[datatask_db]
    
    def task_db_size(self, task_db):
        """
        按 assembly 回退顺序取得一个任务—数据库组合的有限测试长度。

        输入参数:
            - task_db: tuple[str, str], ``(task, db)``。

        返回值:
            - size: int, 依次读取 ``<task>-<db>``、``<db>``、``<datatask_dbs[0]>`` 后得到的样本数。
        """
        # ``task``：str，待查询样本数的图级任务名。
        # ``db``：str，待查询样本数的逻辑数据库名。
        task, db = task_db
        # ``size``：int|None，最高优先级的任务—数据库专用样本数。
        size = self.catalog_dict[task+'-'+db]
        if size is None:
            # ``size``：int|None，回退到数据库全局样本数。
            size = self.catalog_dict[db]
            if size is None: # use ranker for non-dock test sets
                # ``db``：str，ranker assembly 的第一个复合数据库前缀；覆盖局部 db 变量。
                db = self.catalog_dict['datatask_dbs'][0]
                # ``size``：int|None，最终回退前缀对应的样本数。
                size = self.catalog_dict[db]
        return size
        # datatask = self.task_to_datatask[task]
        # return self.datatask_db_size(datatask+'-'+db)



class LMDBDatabase(Dataset):
    """
    对单个 LMDB 文件提供 pickle 键值的最小读写包装。

    构造参数:
        - db_path: str, LMDB 文件路径；当前项目以 ``subdir=False`` 把环境保存在单个文件中。
        - map_size: int, LMDB 地址空间上限，默认 ``10e11`` 字节；只读打开时不会据此扩写文件。
        - readonly: bool, True 时关闭锁、预读和内存初始化，只允许读取；False 时允许 ``add``/``add_one`` 写事务。
        - ``**kwargs``: 原样传给 ``lmdb.open``，例如 assembly 使用 ``max_readers=126``。

    键值契约:
        - 键: str，在 LMDB 中以默认编码保存为 bytes。
        - 值: 任意可 pickle Python 对象；``__getitem__`` 反序列化后返回，缺失键返回 ``None``。
        - assembly LMDB: 值通常是数据库名列表、样本数或 ``data_id`` 字符串。
        - 数据 LMDB: 第一个子库值通常是可更新的 ``PocketMolData``，后续子库值通常是字段 dict。

    副作用:
    """
    def __init__(self, db_path, map_size=int(10e11), readonly=True, **kwargs):
        super().__init__()
        # ``self.db_path``：path，单文件 LMDB 环境路径。
        self.db_path = db_path
        # ``self.map_size``：int，LMDB 虚拟地址空间上限字节数。
        self.map_size = map_size
        if readonly:
            # ``self.env``：lmdb.Environment，只读、无锁、关闭预读/预初始化的多 worker 读取连接。
            self.env = lmdb.open(
                self.db_path,
                map_size=self.map_size,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,  **kwargs
            )
        else:
            # ``self.env``：lmdb.Environment，可写连接；训练/采样主线不进入此分支。
            self.env = lmdb.open(
                self.db_path,
                map_size=self.map_size,
                subdir=False,
                readonly=False,  **kwargs
            )

    def add(self, data_dict):
        """把一个字符串键映射的全部叶值序列化后写入同一 LMDB 事务。

        输入参数:
            - data_dict: dict[str, pickle-able]，键按默认文本编码转为 bytes，值由 ``pickle.dumps`` 序列化。

        返回值:
            - None: 事务正常退出时全部键值一起提交；同名键按 LMDB ``put`` 语义覆盖。
        """
        # ``txn``：LMDB Transaction，覆盖 ``data_dict`` 全部键值的单个原子写事务。
        with self.env.begin(write=True) as txn:
            # ``key``：str，当前 LMDB 文本键。
            # ``value``：当前键对应的可 pickle Python 记录。
            for key, value in data_dict.items():
                txn.put(
                    key = key.encode(),
                    value = pickle.dumps(value)
                )

    def add_one(self, key, value):
        """把一个字符串键及其可 pickle 值写入独立 LMDB 事务。

        输入参数:
            - key: str，按默认文本编码写入 LMDB 的记录键。
            - value: pickle-able，序列化后写入的 Python 记录。

        返回值:
            - None: 事务正常退出时提交；同名键按 LMDB ``put`` 语义覆盖。
        """
        # ``txn``：LMDB Transaction，只覆盖当前一个键值的原子写事务。
        with self.env.begin(write=True) as txn:
            txn.put(
                key = key.encode(),
                value = pickle.dumps(value)
            )

    def close(self):
        self.env.close()

    def __getitem__(self, key):
        """
        读取并反序列化一个键。

        输入参数:
            - key: str|int, int 会先转成十进制字符串。

        返回值:
            - value: 反序列化后的 Python 对象；LMDB 中不存在该键时为 ``None``。
        """
        if isinstance(key, int):
            # ``key``：str，把十进制整数索引规范化为 LMDB 使用的文本键。
            key = str(key)
        # ``txn``：LMDB Transaction，当前键读取使用的只读事务。
        with self.env.begin() as txn:
            # ``value``：bytes|None，按默认编码后的键读取原始 pickle 字节；缺键为 None。
            value = txn.get(key.encode())
        if value is None:
            return None
        else:
            return pickle.loads(value)
    
    def __len__(self):
        """返回当前 LMDB 环境中已提交记录的键数量。"""
        # ``txn``：LMDB Transaction，用于读取环境统计信息而不修改记录。
        with self.env.begin() as txn:
            return txn.stat()['entries']
        
    def get_all_keys(self):
        """按 LMDB 游标顺序返回全部已解码字符串键。"""
        # ``txn``：LMDB Transaction，为只读游标提供一致的键快照。
        with self.env.begin() as txn:
            # ``k``：bytes，LMDB 游标当前记录的原始键；按默认文本编码解码为 str。
            return [k.decode() for k in txn.cursor().iternext(values=False)]


# @register_database('csd')  # pbdock use same logic as geom
# @register_database('pbdock')  # pbdock use same logic as geom
# @register_database('geom')
class SingleDatabase(Dataset): # modify from Drug3DDataset. directly the keys
    """
    管理一个逻辑数据库的多个子 LMDB，并按复合键把记录合并为单样本容器。

    构造参数:
        - config.name: str, 逻辑数据库名；由外层 ``ForeverTaskDataset.db_dict`` 用作键，本类不直接读取。
        - config.lmdb_root: str, 相对数据根目录的子 LMDB 目录。
        - config.lmdb_path: dict[str, str], 子库名到文件名的映射；子库名必须与复合键的首段一致。
        - root: str, 所有训练/测试数据库的共同根目录。

    内部字段:
        - lmdb_path: dict[str, str], 子库名到绝对/拼接后 LMDB 文件路径的映射。
        - db_dict: dict[str, LMDBDatabase], 已打开的只读子库连接。

    ``__getitem__`` 输入与输出:
        - key: str, 分号分隔的 ``<subdb>/<data_id>`` 或 ``<subdb>/<data_id>/<index_sep>`` 片段。
        - 第一个片段: 必须返回支持 ``update`` 的基础样本容器；其对象随后被原地补充字段。
        - 后续片段: 必须返回字段映射；同名字段按片段顺序覆盖先前值。
        - 返回值: 合并后的基础样本容器，保留第一个片段的具体类型。
    """
    def __init__(self, config, root):
        super().__init__()
        # ``self.config``：EasyDict，一个逻辑数据库的 name/lmdb_root/lmdb_path 配置。
        self.config = config
        # ``lmdb_root``：path，相对共同 root 的子 LMDB 目录。
        lmdb_root = config['lmdb_root']
        # ``key``：str，当前配置的子库名；必须匹配复合键片段首段。
        # ``value``：str，当前子库相对 ``lmdb_root`` 的 LMDB 文件名。
        # ``self.lmdb_path``：dict[str, str]，子库名到完整 LMDB 路径的映射。
        self.lmdb_path = {key: os.path.join(root, lmdb_root, value) for key, value in config['lmdb_path'].items()}
        
        # ``self.db_dict``：dict[str,LMDBDatabase]|None，连接前显式初始化为空状态。
        self.db_dict = None
        self._connect_db()
        
    def _connect_db(self):
        """
        为 ``lmdb_path`` 中的每个子库建立只读 ``LMDBDatabase`` 连接。

        副作用:
            - 重建 ``db_dict``；键为子库名，值为已打开环境。
        """
        # ``self.db_dict``：dict[str,LMDBDatabase]，重建连接映射；每个配置子库恰有一项。
        self.db_dict = {}
        # ``lmdb_name``：str，当前子库名。
        # ``lmdb_path``：str，当前子库对应的单文件 LMDB 路径。
        for lmdb_name, lmdb_path in self.lmdb_path.items():
            # ``self.db_dict``：LMDBDatabase，只读打开当前子库路径并按子库名登记。
            self.db_dict[lmdb_name] = LMDBDatabase(lmdb_path)

    def _close_db(self):
        # ``db``：tuple[str, LMDBDatabase]，``dict.items()`` 当前返回的键值对；既有代码随后调用 ``db.close()``，因此该关闭路径会触发 AttributeError，本学习分支只记录不修复。
        for db in self.db_dict.items():
            db.close()
        # ``self.db_dict``：None，标记连接映射已关闭；注意当前 _close_db 循环沿用原实现。
        self.db_dict = None

    def __len__(self):
        raise NotImplementedError('Please implement __len__ method')

    def __getitem__(self, key):
        """
        按复合键顺序读取并合并多个子 LMDB 记录。

        输入参数:
            - key: str, ``;`` 分隔片段；每段用 ``/`` 分成子库名、``data_id`` 和可选分隔编号。

        返回值:
            - data: 第一个片段返回的可更新样本容器；后续片段返回的映射已经按顺序执行 ``data.update(fetch)``。

        失败边界:
            - 子库名必须存在于 ``db_dict``，记录必须存在且满足首段可更新/后续段为映射的结构，否则由索引或 ``update`` 操作直接报错。
        """
        # if self.db_dict is None:
        #     self._connect_db()
        
        # ``key_split``：list[str]，顺序同时决定基础容器来源和字段覆盖优先级。
        key_split = key.split(';')
        # ``i``：int，当前复合键片段序号；0 表示用于初始化返回容器的基础记录。
        # ``key_this_db``：str，当前 ``<subdb>/<record>[/<index>]`` 片段。
        for i, key_this_db in enumerate(key_split):
            # ``key_list``：list[str]，当前片段按 ``/`` 拆成子库名、记录标识和可选分隔编号。
            key_list = key_this_db.split('/')
            if len(key_list) == 2:
                # ``lmdb_name``：str，普通复合键片段中的子库名。
                # ``data_id``：str，普通复合键片段中的子库记录标识。
                lmdb_name, data_id = key_list
                # ``fetch``：任意可 pickle 对象，当前子库 ``data_id`` 对应的反序列化记录。
                fetch = self.db_dict[lmdb_name][data_id]
            elif len(key_list) == 3:
                lmdb_name, data_id, index_sep = key_list
                fetch = self.db_dict[lmdb_name]['/'.join([data_id, index_sep])]
                # fetch = self.db_dict[lmdb_name][data_id][int(index_sep)]
            # initial fetch
            if i == 0:
                # assert lmdb_name == 'mols' or lmdb_name.startswith('pocmol'),\
                #     'You must fetch mols or pocmol lmdb as the first lmdb since it contains torch.Data'
                # ``data``：首个子库返回的可更新基础样本容器；保留其具体 ``PocketMolData`` 类型。
                data = fetch
            else:
                data.update(fetch)
        
        return data
