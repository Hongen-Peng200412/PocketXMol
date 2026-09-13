"""
把 LMDB 中的分子—口袋字段逐步变换为 PocketXMol 可批处理的图字段和任务 prompt。

构象生成与小分子 docking 的阅读主线依次是 :class:`FeaturizePocket`、:class:`FeaturizeMol`、
:class:`MixedTransform` 和 :class:`ConfTransform`；``dock`` 注册名复用 ``ConfTransform``，只在配置的
setting 分布和后续噪声先验上与 ``conf`` 区分。推理时 :class:`OverwriteStartPos` 可替换 flexible docking
初始构象，:class:`OverwritePos` 可把上一轮生成坐标重新装回同一输入契约。

本模块不落盘；返回的核心对象仍是原 ``PocketMolData``，但会新增原子/半边离散类别、中心化坐标、
口袋 kNN 图、fixed prompt、刚体域索引、可旋转键注释及可选真值副本。
"""

# Standard library imports
import sys
from itertools import product

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn.pool import knn_graph
from torch_geometric.transforms import Compose
from torch_geometric.utils import (
    bipartite_subgraph,
    sort_edge_index,
    subgraph,
    to_undirected,
)
from torch_scatter import scatter_min

# Local imports
sys.path.append('.')
from models.transition import *
from process.utils_process import process_raw
from utils.data import Mol3DData, PocketMolData
from utils.dataset import *
from utils.misc import *
from utils.train_noise import (
    combine_vectors_indexed,
    get_vector,
    get_vector_list,
)

# Configuration constants for different generation modes
CONF_SETTINGS = ['free', 'flexible', 'torsional', 'rigid']

MASKFILL_SETTINGS = {
    'decomposition': ['brics', 'mmpa', 'atom'],  # Fragment decomposition strategies
    'order': ['tree', 'inv_tree', 'random'],      # Generation order
    'part1_pert': ['fixed', 'free', 'small', 'rigid', 'flexible'],  # Reference fragment perturbation
    'known_anchor': ['all', 'partial', 'none']    # Anchor atom knowledge
}
PEPDESIGN_SETTINGS = {
    'mode': ['full', 'sc', 'packing']  # full=backbone+sidechain, sc=sidechain, packing=sc position only
}

# Transform registry
_TRAIN_DICT = {}


def register_transforms(name: str):
    """
    把变换类登记到模块级名称注册表。

    输入参数:
        - name: str, 配置 ``transforms.*.name`` 使用的查找键；``conf`` 与 ``dock`` 可以指向同一个类。

    返回值:
        - decorator: callable, 接收一个类并把 ``_TRAIN_DICT[name]`` 设为该类，随后原样返回该类。

    副作用:
        - 修改进程内全局 ``_TRAIN_DICT``；同名注册会覆盖先前值。
    """
    def decorator(cls):
        # ``_TRAIN_DICT``：type，变换类本身；注册阶段不实例化，get_transforms 再按配置构造对象。
        _TRAIN_DICT[name] = cls
        return cls
    return decorator

# XXX
def get_transforms(config, *args, **kwargs):
    """
    按 ``config.name`` 实例化一个已注册变换。

    输入参数:
        - config: 映射式配置，必须支持 ``config.name``；其余嵌套字段由选中的变换类读取。
        - ``*args``/``**kwargs``: 原样传给变换类构造器；训练主线会传 ``mode`` 和离散类别数。

    返回值:
        - transform: callable, ``_TRAIN_DICT[config.name]`` 的实例；未知名称直接触发字典 ``KeyError``。
    """
    # ``name``：str，_TRAIN_DICT 的注册键；构象/docking 分别为 conf/dock。
    name = config.name
    return _TRAIN_DICT[name](config, *args, **kwargs)

# XXX
class FeaturizePocket(object):
    """
    把原始口袋原子属性编码为节点特征、kNN 图和以口袋中心为原点的坐标。

    配置字段:
        - knn: int, 每个口袋原子连接的近邻数量；传给 ``torch_geometric.nn.knn_graph``。
        - center: 长度 3 的 float 列表|None, 预设世界坐标中心；None 时取全部口袋原子的算术平均。

    类别约定:
        - 元素通道 0..3: 依次为 C(6)、N(7)、O(8)、S(16)；出现其他原子序数会断言失败。
        - 氨基酸通道 4..23: ``pocket_atom_to_aa_type`` 的 0..19 one-hot。
        - 主链通道 24: ``pocket_is_backbone`` 的 0/1 值。

    输入样本字段:
        - pocket_element: int64, (P,), 每个口袋原子的原子序数。
        - pocket_atom_to_aa_type: int64, (P,), 每个口袋原子的氨基酸类别编号，范围 ``[0, 19]``。
        - pocket_is_backbone: bool 或 0/1, (P,), True/1 表示该原子属于蛋白主链。
        - pocket_pos: (P, 3), 口袋原子世界坐标；最后一维按 XYZ 排列，单位 Å。
        - pdbid: str|NaN, 受体结构标识；NaN 在输出中归一为空字符串。

    输出样本字段:
        - pocket_atom_feature: float32, (P, 25), ``[元素 one-hot(4), 氨基酸 one-hot(20), 主链标记(1)]``。
        - pocket_knn_edge_index: int64, (2, E_p), 口袋有向 kNN 边；第一行是消息聚合目标原子编号，第二行是近邻来源原子编号。
        - pocket_center: float32, (1, 3), 选定的世界坐标原点，最后一维按 XYZ 排列，单位 Å。
        - pocket_pos: float32, (P, 3), ``原世界坐标 - pocket_center``，单位 Å。
        - pdbid: str, 原标识或空字符串。

    空口袋分支:
        - 当字段缺失或 P=0 时返回 ``pocket_atom_feature(0,25)``、``pocket_knn_edge_index(2,0)``、``pocket_pos(0,3)`` 和 ``pocket_center(0,3)``。
        - 空口袋用于无受体构象生成；后续模型仍会接收这些空张量，不在本类伪造口袋节点。
    """

    def __init__(self, config):
        super().__init__()
        # ``config.knn``：int，每个口袋点的近邻数量上限。
        # ``config.center``：list[float]|None，长度为 3 的预设世界坐标中心，单位 Å。
        # ``self.config``：EasyDict，保留上述口袋 kNN 与中心叶。
        self.config = config
        # ``self.knn``：int，每个口袋原子最多连接的近邻数量；作为 ``knn_graph`` 的 ``k``。
        self.knn = config.knn
        # ``self.atomic_numbers``：LongTensor，形状为 (4,)；元素 one-hot 的列 0..3 依次对应 C、N、O、S。
        self.atomic_numbers = torch.LongTensor([
            6,   # C：元素通道 0。
            7,   # N：元素通道 1。
            8,   # O：元素通道 2。
            16,  # S：元素通道 3。
        ])
        # ``self.max_num_aa``：int，口袋氨基酸 one-hot 类别数；类别编号范围为 ``[0, 20)``。
        self.max_num_aa = 20
        
        # ``self.follow_batch``：list[str]；要求 PyG DataLoader 为 ``pocket_pos`` 生成 ``pocket_pos_batch``，形状为 (P,)。
        self.follow_batch = ['pocket_pos']
        # ``self.exclude_keys``：list[str]；以下原始逐原子叶在生成 ``pocket_atom_feature`` 后不由 PyG 默认拼接。
        self.exclude_keys = [
            'pocket_element',          # (P,)，原子序数；已编码进元素 one-hot 通道。
            'pocket_molecule_name',    # 长度为 P 的序列，逐原子所属受体链/分子名称。
            'pocket_is_backbone',      # (P,)，主链标记；已编码进最后一个特征通道。
            'pocket_atom_name',        # 长度为 P 的序列，逐原子 PDB 原子名。
            'pocket_atom_to_aa_type',  # (P,)，氨基酸类别；已编码进中间 20 个 one-hot 通道。
        ]

        # ``self.preset_pocket_center``：list[float]|None，长度为 3；非空时覆盖按口袋原子均值计算的世界坐标中心，单位 Å。
        self.preset_pocket_center = config.get('center', None)

    @property
    def feature_dim(self):
        """返回口袋节点特征宽度 25，即 4 个元素通道、20 个氨基酸通道和 1 个主链通道。"""
        return self.atomic_numbers.size(0) + self.max_num_aa + 1  # 1 for is_backbone

    def __call__(self, data:PocketMolData):
        """把原始口袋原子叶编码为局部坐标特征图。

        输入字段:
            - data.pocket_element: LongTensor，形状为 (P,)，口袋原子序数。
            - data.pocket_atom_to_aa_type: LongTensor，形状为 (P,)，氨基酸类别编号，范围为 ``[0, 20)``。
            - data.pocket_is_backbone: BoolTensor|LongTensor，形状为 (P,)，1 表示口袋原子属于蛋白主链。
            - data.pocket_pos: FloatTensor，形状为 (P, 3)，口袋世界坐标，单位 Å。
            - data.pdbid: str|NaN，受体结构标识。
            - data.data_id: str，空口袋告警使用的样本标识。

        输出字段:
            - data.pocket_atom_feature: FloatTensor，形状为 (P, 25)，元素、氨基酸和主链标记拼接特征。
            - data.pocket_knn_edge_index: LongTensor，形状为 (2, E_p)，口袋有向 kNN 边端点。
            - data.pocket_center: FloatTensor，形状为 (1, 3)，世界坐标到模型局部坐标的平移原点，单位 Å。
            - data.pocket_pos: FloatTensor，形状为 (P, 3)，减去 ``pocket_center`` 后的口袋局部坐标，单位 Å。
            - data.pdbid: str，原受体标识或由 NaN 归一化得到的空字符串。

        空口袋输出:
            - data.pocket_atom_feature: FloatTensor，形状为 (0, 25)。
            - data.pocket_knn_edge_index: LongTensor，形状为 (2, 0)。
            - data.pocket_center: FloatTensor，形状为 (0, 3)。
            - data.pocket_pos: FloatTensor，形状为 (0, 3)。
        """
        
        # no pocket data, make dummy
        # if ('pocket_pos' not in data.keys) or (len(data['pocket_pos']) == 0):
        #     if ('pocket_pos' in data.keys) and len(data['pocket_pos'] == 0):
        if ('pocket_pos' not in data) or (len(data['pocket_pos']) == 0):
            if ('pocket_pos' in data) and len(data['pocket_pos'] == 0):
                # ``pdbid``：str|NaN，空口袋仍保留输入受体标识；此条件沿用原实现的布尔长度判断。
                pdbid = data['pdbid']
                print(f'Warning: empty pocket: {data["data_id"]}')
            else:
                # ``pdbid``：str，无 ``pocket_pos`` 字段时无法关联受体，使用空标识。
                pdbid = ''
            data.update({
                # ``data.pocket_atom_feature``：FloatTensor，形状为 (0, 25)；空口袋没有节点特征行。
                'pocket_atom_feature': torch.empty([0, self.feature_dim], dtype=torch.float),
                # ``data.pocket_knn_edge_index``：LongTensor，形状为 (2, 0)；空口袋没有 kNN 边。
                'pocket_knn_edge_index': torch.empty([2, 0], dtype=torch.long),
                # ``data.pocket_pos``：FloatTensor，形状为 (0, 3)；空口袋没有局部坐标行，单位 Å。
                'pocket_pos': torch.empty([0, 3], dtype=torch.float),
                # ``data.pocket_center``：FloatTensor，形状为 (0, 3)；无口袋原子时不定义世界坐标中心，单位 Å。
                'pocket_center': torch.empty([0, 3], dtype=torch.float),
                # ``data.pdbid``：str|NaN，保留已有受体标识，或在字段缺失时写入空字符串。
                'pdbid': pdbid,
            })
            return data
        
        # if len(data['pocket_pos']) == 0:
            # raise ValueError(f"empty pocket: {data['data_id']}")
        # pocket atom features
        # ``e``：标量 Tensor，当前口袋原子的原子序数；逐项检查是否属于 C/N/O/S 词表。
        assert all([(e in self.atomic_numbers) for e in data.pocket_element]), 'unknown element in pocket'
        # ``element``：LongTensor，形状为 (P, 4)；每行恰有一个 1，列顺序为 C/N/O/S。
        element = (data.pocket_element.view(-1, 1) == self.atomic_numbers.view(1, -1)).long()   # [P] -> [P, 1]；[4] -> [1, 4]
        # ``amino_acid``：LongTensor，形状为 (P, 20)；每行按氨基酸类别编号置一个 1。
        amino_acid = F.one_hot(data.pocket_atom_to_aa_type, num_classes=self.max_num_aa)  # [P] -> [P, 20]
        # ``is_backbone``：LongTensor，形状为 (P, 1)；1 表示蛋白主链原子。
        is_backbone = data.pocket_is_backbone.view(-1, 1).long()  # [P] -> [P, 1]
        # ``x``：LongTensor，形状为 (P, 25)；按通道拼接元素、氨基酸和主链标记。
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)  # [P, 4] + [P, 20] + [P, 1] -> [P, 25]
        # ``data.pocket_atom_feature``：FloatTensor，形状为 (P, 25)；模型口袋节点编码器的输入特征。
        data['pocket_atom_feature'] = x.float()
        
        if 'is_atom_remain' in data:  # apply cut for pep, so need to update pocket
            raise ValueError('not supported anymore: is_atom_remain')
            is_atom_remain = data['is_atom_remain']
            pocket_pos = data['pocket_pos']
            node_pos = data['pos_all_confs'][0]
            dist = torch.norm(pocket_pos[:, None] - node_pos[None], dim=-1) # (n_pocket, n_node)
            threshold = dist.min(dim=1)[0].max()
            is_pocket_remain = dist[:, is_atom_remain].min(dim=1)[0] < threshold
            data['pocket_atom_feature'] = data['pocket_atom_feature'][is_pocket_remain]
            data['pocket_pos'] = data['pocket_pos'][is_pocket_remain]
        # pocket inner edge features
        # ``pocket_knn_edge_index``：LongTensor，形状为 (2, E_p)；每个口袋原子最多接收 ``knn`` 个近邻消息，端点编号索引 ``pocket_pos`` 第一维。
        pocket_knn_edge_index = knn_graph(
            data.pocket_pos, k=self.knn, flow='target_to_source')
        # ``data.pocket_knn_edge_index``：LongTensor，形状为 (2, E_p)；口袋图有向边，供模型 ``pocket_encoder`` 直接读取。
        data['pocket_knn_edge_index'] = pocket_knn_edge_index
        
        # pocket pos and center
        if self.preset_pocket_center is not None:
            # ``pocket_center``：FloatTensor，形状为 (1, 3)；配置给定的世界坐标中心，列顺序为 XYZ，单位 Å。
            pocket_center = torch.tensor(self.preset_pocket_center).reshape(1, 3).float()
        else:
            # ``pocket_center``：FloatTensor，形状为 (1, 3)；P 个口袋原子世界坐标的算术平均，列顺序为 XYZ，单位 Å。
            pocket_center = data.pocket_pos.mean(dim=0, keepdim=True)
        # ``data.pocket_center``：FloatTensor，形状为 (1, 3)；世界坐标到模型局部坐标的平移原点，单位 Å。
        data['pocket_center'] = pocket_center
        # ``data.pocket_pos``：[P, 3] - [1, 3] -> [P, 3]；广播减中心，得到口袋局部坐标，单位 Å。
        data['pocket_pos'] = data.pocket_pos - data.pocket_center
        # ``pdbid``：str|NaN，输入受体标识。
        pdbid = data['pdbid']
        # ``data.pdbid``：str，利用 NaN 不等于自身的性质把缺失标识归一为空字符串。
        data['pdbid'] = pdbid if (pdbid == pdbid) else ''
        return data

# XXX
class FeaturizeMol(object):
    """
    把原始配体元素、构象与双向化学键编码为离散节点、中心化坐标和完全图半边。

    配置字段:
        - chem.atomic_numbers: list[int], 节点类别编号到原子序数的有序映射；列表索引就是 ``node_type``。
        - chem.mol_bond_types: list[int], 化学键类别表；当前值 ``[1,2,3,4]`` 对应单/双/三/芳香键。
        - use_mask_node: bool, True 时在元素类别之后追加一个 node mask 类。
        - use_mask_edge: bool, True 时在非键类和化学键类之后追加一个 edge mask 类。
        - mol_as_pocket_center: bool, True 时已有口袋样本会改用配体质心作为共同局部原点，并同步平移口袋坐标与 ``pocket_center``。
        - is_peptide: bool, 写入每个配体原子的 ``is_peptide`` 0/1 特征；小分子构象/docking 为 False。

    输入样本字段:
        - element: int64, (N,), 每个配体原子的原子序数。
        - pos_all_confs: (C, N, 3), C 个候选构象的世界坐标，最后一维按 XYZ 排列，单位 Å。
        - i_conf_list: (C,), 每个候选构象的原数据编号；与 ``pos_all_confs`` 第一维对齐。
        - num_atoms: int 标量 N。
        - num_bonds: int 标量 M，只计无向化学键一次。
        - bond_index: int64, (2, 2M), 双向化学键端点；数值索引 ``element`` 第一维。
        - bond_type: int64, (2M,), 每个有向键的 1..4 类别；与 ``bond_index`` 第二维对齐。
        - pocket_center: (1, 3), 可选口袋世界坐标中心，单位 Å；存在且非空时用作配体局部原点。
        - pocket_pos: (P, 3), 可选已中心化口袋坐标；仅 ``mol_as_pocket_center=True`` 时再次平移。

    输出样本字段:
        - num_nodes: int 标量 N，PyG 节点数。
        - node_type: int64, (N,), 每个原子的元素类别编号，范围 ``[0, num_element-1]``。
        - node_pos: float32, (N, 3), 选中构象的局部坐标，最后一维按 XYZ 排列，单位 Å。
        - i_conf: 标量，选中构象在原数据中的编号 ``i_conf_list[idx]``。
        - halfedge_index: int64, (2, H), 完全图上三角端点，H=N(N-1)/2；每列满足 ``i<j``。
        - halfedge_type: int64, (H,), 0 表示非键，1..4 表示单/双/三/芳香键；与 ``halfedge_index`` 第二维对齐。
        - is_peptide: int64, (N,), 小分子构象与 docking 为全 0。

    符号 & 含义
        - C	当前分子通过筛选后保留的构象数
        - N	配体原子数
        - M	配体无向化学键数
        - P	当前裁剪口袋中的原子数
        - H	完全图无序原子对数，即 N(N-1)/2

    批处理声明:
        - follow_batch: ``['node_type', 'halfedge_type']``，使 PyG 生成逐原子与逐半边图编号。
        - exclude_keys: 原始多构象、双向键、扭转辅助映射等不适合默认拼接的字段；任务变换必须在批处理前消费它们。
    """
    def __init__(self, config):
        super().__init__()
        # ``atomic_numbers``：list[int]，节点类别索引到原子序数的有序词表；列表位置就是 ``node_type`` 类别编号。
        atomic_numbers = config.chem.atomic_numbers
        # ``mol_bond_types``：list[int]，真实化学键类别的有序词表；当前 1/2/3/4 对应单/双/三/芳香键。
        mol_bond_types = config.chem.mol_bond_types
        # ``use_mask_node``：bool，是否在元素词表末尾追加 node mask 类。
        use_mask_node = config.use_mask_node
        # ``use_mask_edge``：bool，是否在非键与真实键词表末尾追加 edge mask 类。
        use_mask_edge = config.use_mask_edge
        
        # ``self.atomic_numbers``：LongTensor，形状为 (K_elem,)；有序原子序数词表，行号等于节点类别编号。
        self.atomic_numbers = torch.LongTensor(atomic_numbers)
        # ``self.mol_bond_types``：LongTensor，形状为 (K_bond,)；有序真实化学键类别词表。
        self.mol_bond_types = torch.LongTensor(mol_bond_types)
        # ``self.num_element``：int，真实元素类别数 K_elem；不含可选 node mask 类。
        self.num_element = self.atomic_numbers.size(0)
        # ``self.num_bond_types``：int，真实化学键类别数 K_bond；不含非键和可选 edge mask 类。
        self.num_bond_types = self.mol_bond_types.size(0)

        # ``self.num_node_types``：int，K_elem 加可选 mask 类后的模型节点分类头输出宽度。
        self.num_node_types = self.num_element + int(use_mask_node)
        # ``self.num_edge_types``：int，非键 1 类、K_bond 个真实键类与可选 mask 类之和。
        self.num_edge_types = self.num_bond_types + 1 + int(use_mask_edge) # + 1 for the non-bonded edges
        # ``self.use_mask_node``：bool，保留 node mask 开关，供解码边界和其他变换读取。
        self.use_mask_node = use_mask_node
        # ``self.use_mask_edge``：bool，保留 edge mask 开关，供解码边界和其他变换读取。
        self.use_mask_edge = use_mask_edge
        
        # ``self.ele_to_nodetype``：dict[int, int]，原子序数到 0-based 节点类别编号的正向映射。
        # ``ele``：int，当前有序词表中的原子序数。
        # ``i``：int，当前原子序数对应的 0-based 节点类别编号。
        self.ele_to_nodetype = {ele: i for i, ele in enumerate(atomic_numbers)}
        # ``self.nodetype_to_ele``：dict[int, int]，0-based 节点类别编号到原子序数的逆向映射；只覆盖真实元素类。
        # ``i``：int，当前 0-based 节点类别编号。
        # ``ele``：int，当前节点类别对应的原子序数。
        self.nodetype_to_ele = {i: ele for i, ele in enumerate(atomic_numbers)}

        # ``self.follow_batch``：list[str]；要求 PyG 额外生成 ``node_type_batch(N,)`` 与 ``halfedge_type_batch(H,)``。
        self.follow_batch = [
            'node_type',      # (N,)，逐配体原子类别；生成逐原子图编号。
            'halfedge_type',  # (H,)，逐完全图半边类别；生成逐半边图编号。
        ]

        # ``self.exclude_keys``：list[str]；以下单样本原始叶或辅助叶不由 PyG 默认拼接。
        self.exclude_keys = [
            'orig_keys',                 # list[str]，原始样本键名快照。
            'pos_all_confs',             # (C, N, 3)，候选构象世界坐标；变换前已抽取一帧。
            'num_confs',                 # int，候选构象数 C。
            'i_conf_list',               # 长度为 C 的序列，候选构象原数据编号。
            'bond_index',                # (2, 2M)，原始双向化学键端点。
            'bond_type',                 # (2M,)，原始双向化学键类别。
            'num_bonds',                 # int，无向化学键数 M。
            'num_atoms',                 # int，配体原子数 N；已复制到 ``num_nodes``。
            'bond_rotatable',            # (2M,)，原始双向键可旋转标记。
            'tor_twisted_pairs',         # dict，规范化旋转轴到断键两侧原子集合的映射。
            'fixed_dist_torsion',        # (N, N)，扭转下保持不变的原子对距离标记。
            'path_mat',                  # (N, N)，化学图最短路径长度矩阵。
            'nbh_dict',                  # dict[int, list[int]]，逐原子一跳邻居索引。
            'tor_bond_mat',              # (N, N)，对称可旋转键邻接矩阵。
            'matches_graph',             # 图匹配辅助对象；构象采样前的单分子元数据。
            'matches_iso',               # 同构匹配辅助对象；构象采样前的单分子元数据。
            'mmpa',
            'brics',
            'peptide_pos',
            'peptide_atom_name',
            'peptide_is_backbone',
            'peptide_res_id',
            'peptide_atom_to_aa_type',
            'peptide_res_index',
            'peptide_seq',
            'peptide_pep_len',
            'peptide_pep_path',
        ]

        # ``self.mol_as_pocket_center``：bool；为 True 时把共同局部原点从口袋中心移到配体质心，并同步平移口袋。
        self.mol_as_pocket_center = config.get('mol_as_pocket_center', False)
        # ``self.is_peptide``：bool，为每个节点写入 1/0 的 ``is_peptide`` 特征；小分子任务为 False。
        self.is_peptide = config.get('is_peptide', False)
    
    def __call__(self, data: Mol3DData):
        """把原始配体记录编码为模型使用的局部完全图。

        输入字段:
            - data.element: LongTensor，形状为 (N,)，逐配体原子的原子序数。
            - data.pos_all_confs: FloatTensor，形状为 (C, N, 3)，候选 conformer 世界坐标，单位 Å。
            - data.i_conf_list: Sequence[int]，长度为 C，逐候选 conformer 的原数据编号。
                这里C是所有读取的合法构象。在分子对接任务中只读取一个沉积构象, 在构象生成任务中C有多个, 但 [FeaturizeMol (line 1048)](/C:/Users/15919/Desktop/PocketXMol/utils/transforms.py:1042) 每次只随机选择其中一个构象，得到 (N, 3) 的 node_pos。
            - data.num_atoms: int 标量 N，配体原子数。
            - data.num_bonds: int 标量 M，无向化学键数。
            - data.bond_index: LongTensor，形状为 (2, 2M)，双向化学键端点，数值索引 ``element`` 第一维。
            - data.bond_type: LongTensor，形状为 (2M,)，与 ``bond_index`` 列对齐的键类别。
            - data.pocket_center: FloatTensor|缺省，形状为 (1, 3)，docking 局部坐标原点的世界坐标，单位 Å, 最开始是口袋原子的中心; 但会在 ``mol_as_pocket_center=True`` 时变成配体的中心————————真正的docking任务用口袋中心 , 配体构象生成任务用配体中心 , 
            - data.pocket_pos: FloatTensor|缺省，形状为 (P, 3)，已中心化口袋坐标, 最开始以 data.pocket_center 为中心来中心化; 但会在 ``mol_as_pocket_center=True`` 时以配体原子中心为中心再次平移。P=当前裁剪口袋中的原子数

        输出字段:
            - data.num_nodes: int 标量 N，PyG 配体节点数。
            - data.node_type: LongTensor，形状为 (N,)，逐原子元素类别编号。
            - data.node_pos: FloatTensor，形状为 (N, 3)，选中 conformer 的模型局部坐标，单位 Å。
            - data.i_conf: int 标量，选中 conformer 在原数据中的编号。
            - data.halfedge_index: LongTensor，形状为 (2, H)，完全图上三角端点(无自环完全图中满足 i < j 的全部无序原子对端点索引, 如(0,1), (0,2), (0,3), (1,2), (1,3), (2,3))，H=N(N-1)/2。
            - data.halfedge_type: LongTensor，形状为 (H,)，0 为非键，1/2/3/4 为单/双/三/芳香键。
            - data.is_peptide: LongTensor，形状为 (N,)，小分子构象与 docking 路径为全 0。

        返回值:
            - data: Mol3DData，原地写入以上输出叶后的同一容器。

        符号 & 含义
            - C	当前分子通过筛选后保留的构象数
            - N	配体原子数
            - M	配体无向化学键数
            - P	当前裁剪口袋中的原子数
            - H	完全图无序原子对数，即 N(N-1)/2
        """
        # ``data.num_nodes``：int，配体原子数 N；把原始 ``num_atoms`` 显式登记为 PyG 节点数。
        data.num_nodes = data.num_atoms
        
        # node type
        # ``ele``：标量 Tensor，当前配体原子的原子序数；逐项检查是否属于配置元素词表。
        assert np.all([ele in self.atomic_numbers for ele in data.element]), 'unknown element'
        # ``ele``：标量 Tensor，当前配体原子的原子序数；用于查询 ``ele_to_nodetype``。
        # ``data.node_type``：LongTensor，形状为 (N,)；第 i 个值是 ``element[i]`` 在有序 ``atomic_numbers`` 中的索引(0-started)。
        data.node_type = torch.LongTensor([self.ele_to_nodetype[ele.item()] for ele in data.element])
        
        # atom pos: sample a conformer from data.pos_all_confs; then move to origin
        if data.get('task', '') == 'linking':
            idx = 0
        else:
            # ``idx``：int，从 ``[0, C)`` 均匀采样的候选构象行号。
            idx = np.random.randint(data.pos_all_confs.shape[0])
        # ``atom_pos``：FloatTensor，形状为 (N, 3)；选中单个构象的世界坐标，列顺序为 XYZ，单位 Å。
        atom_pos = data.pos_all_confs[idx].float()  # [C, N, 3] -> [N, 3]
        
        # move to center
        # if move:
        if len(getattr(data, 'pocket_center', [])) > 0:
            # ``atom_pos``：[N, 3] - [1, 3] -> [N, 3]；转为以口袋中心为原点的 docking 局部坐标，单位 Å。
            atom_pos = atom_pos - data.pocket_center
            if self.mol_as_pocket_center:  # change the global center as the mol center
                # ``mol_center``：FloatTensor，形状为 (1, 3)；已减口袋中心后的配体质心，成为新的共同局部原点，单位 Å。
                mol_center = atom_pos.mean(dim=0, keepdim=True)
                # ``atom_pos``：[N, 3] - [1, 3] -> [N, 3]；进一步把配体质心移到局部原点，单位 Å。
                atom_pos = atom_pos - mol_center
                # ``data.pocket_pos``：[P, 3] - [1, 3] -> [P, 3]；同步平移口袋，保持口袋—配体相对几何不变，单位 Å。
                data['pocket_pos'] = data['pocket_pos'] - mol_center
                # ``data.pocket_center``：[1, 3] + [1, 3] -> [1, 3]；世界坐标中心加上局部 ``mol_center``，保持坐标变换可逆，单位 Å。
                data['pocket_center'] = data['pocket_center'] + mol_center
        else:
            # ``atom_pos``：[N, 3] -> [N, 3]；无口袋构象任务减配体几何中心，转为质心局部坐标，单位 Å。
            atom_pos = atom_pos - atom_pos.mean(dim=0)

        # ``data.node_pos``：FloatTensor，形状为 (N, 3)；最终模型局部坐标，列顺序为 XYZ，单位 Å。
        data.node_pos = atom_pos
        # ``data.i_conf``：int 标量，所选构象在原数据中的编号；不是候选数组行号 ``idx``。
        data.i_conf = data.i_conf_list[idx]
        
        # build half edge (not full because perturb for edge_ij should be the same as edge_ji)
        # ``edge_type_mat``：LongTensor，形状为 (N, N)；稠密有向键类别矩阵，0 表示两个原子没有化学键。
        edge_type_mat = torch.zeros([data.num_nodes, data.num_nodes], dtype=torch.long)
        # ``i``：int，当前双向化学键列号，取值范围为 [0, 2M)。
        for i in range(data.num_bonds * 2):  # multiply by to is for symmtric of bond index
            # ``edge_type_mat[u, v]``：int64 标量；把第 i 条有向键类别写到源原子 u、目标原子 v 对应的矩阵单元。
            edge_type_mat[data.bond_index[0, i], data.bond_index[1, i]] = data.bond_type[i]
        # ``halfedge_index``：LongTensor，形状为 (2, H)；N 个原子的完全图上三角端点，H=N(N-1)/2，每列满足 i<j。
        halfedge_index = torch.triu_indices(data.num_nodes, data.num_nodes, offset=1)
        # ``halfedge_type``：LongTensor，形状为 (H,)；每条无向半边的类别，从对称有向键矩阵只取 i<j 一侧。
        halfedge_type = edge_type_mat[halfedge_index[0], halfedge_index[1]]  # [N, N] + [2, H] -> [H]
        assert len(halfedge_type) == len(halfedge_index[0])
        # max_bond = torch.norm(atom_pos[data.bond_index[0]] - atom_pos[data.bond_index[1]], dim=-1).max()
        # assert max_bond < 4, f'bond length too long: {max_bond.item()} in {data.data_id}'
        
        # ``data.halfedge_index``：LongTensor，形状为 (2, H)；写入完全图无向半边端点。
        data.halfedge_index = halfedge_index
        # ``data.halfedge_type``：LongTensor，形状为 (H,)；写入与半边列一一对齐的非键/化学键类别。
        data.halfedge_type = halfedge_type
        assert (data.halfedge_type > 0).sum() == data.num_bonds
        
        if 'is_atom_remain' in data:
            raise ValueError('not supported anymore: is_atom_remain')
        
        if self.is_peptide:
            data['is_peptide'] = torch.ones([data.num_nodes], dtype=torch.long)
        else:
            # ``data.is_peptide``：LongTensor，形状为 (N,)；小分子构象/docking 的附加节点特征全为 0。
            data['is_peptide'] = torch.zeros([data.num_nodes], dtype=torch.long)  # default is not peptide

        return data
    
    def decode_output(self, node, pos, halfedge, halfedge_index,
                      pocket_center=None):
        """
        把采样后的离散节点/半边类别和坐标解码为重建分子所需的 NumPy 字段。

        形状符号:
            - N: 采样图中的节点槽位数，包含可能仍为 mask/非法类别的槽位。
            - H: 采样图完全图的半边数。
            - N_keep: 节点类别落在真实元素表内的保留原子数。
            - M_keep: 类别落在真实化学键表内且两端原子均保留的无向键数。

        输入参数:
            - node: int, (N,), 已取 argmax/采样后的节点类别编号；负数、mask 类和越界值均被删除。
            - pos: float, (N, 3), 节点局部坐标，最后一维按 XYZ 排列，单位 Å。
            - halfedge: int, (H,), 已取 argmax/采样后的半边类别；0 为非键，1..num_bond_types 为真实键，其他值删除。
            - halfedge_index: int64, (2, H), 半边端点；数值索引 ``node``/``pos`` 第一维。
            - pocket_center: float, (1, 3)|None, 原始世界坐标中心，单位 Å；非 None 时广播加回所有节点坐标。

        返回值（存在键类型）:
            - element: int64, (N_keep,), 保留节点的原子序数；按 ``node`` 原顺序排列。
            - atom_pos: float, (N_keep, 3), 保留原子的世界坐标或原局部坐标，单位 Å。

        返回值（存在键类别时额外字段）:
            - bond_type: int, (2M_keep,), 每条保留无向键的类别复制两次，与双向 ``bond_index`` 对齐。
            - bond_index: int64, (2, 2M_keep), 压缩原子编号后的双向键端点；前 M_keep 列为 i<j，后 M_keep 列为反向。
            - atom_pos_masked: float, (N-N_keep, 3), 被节点类别过滤掉的槽位坐标，供诊断未解码节点。

        索引重映射:
            - 若删除节点，``edge_index_changer`` 把原 N 个槽位映射到 ``[0,N_keep-1]``；被删槽位映射为 -1。
            - 任一端点映射为 -1 的化学键同步删除，保证 ``bond_index`` 只引用 ``element`` 第一维。
        """
        # move back to pocekt center in pdb
        if pocket_center is not None:
            # ``pos``：[N, 3] + [1, 3] -> [N, 3]；把模型局部坐标平移回世界坐标，单位 Å。
            pos = pos + pocket_center
        # get atom and element
        # if is_prob:
        #     pred_atom = softmax(pred_node, axis=-1)
        #     atom_type = np.argmax(pred_atom, axis=-1)
        #     atom_prob = np.max(pred_atom, axis=-1)
        # else:
        # ``atom_type``：np.ndarray，形状为 (N,)；节点离散类别的别名，尚未过滤 mask/越界槽位。
        atom_type = node
        # atom_prob = np.ones(len(atom_type))
        # ``isnot_masked_atom``：np.ndarray[bool]，形状为 (N,)；True 表示类别可映射为真实元素，同时遮盖 ``node`` 与 ``pos`` 第一维。
        isnot_masked_atom = (atom_type < self.num_element) & (atom_type >= 0)
        # see me: 配体构象生成 conf、真正的 docking dock：最终结果中一般不会出现。
        if not isnot_masked_atom.all():
            # ``edge_index_changer``：np.ndarray[int64]，形状为 (N,)；原节点槽位到压缩后原子编号的映射，被删除槽位为 -1。
            edge_index_changer = - np.ones(len(isnot_masked_atom), dtype=np.int64)
            # ``edge_index_changer[isnot_masked_atom]``：[N_keep] -> [N_keep]；按原槽位顺序填入连续压缩编号 0..N_keep-1。
            edge_index_changer[isnot_masked_atom] = np.arange(isnot_masked_atom.sum())
        # ``atom_type``：[N] -> [N_keep]；删除 mask、负数和越界节点类别，保留原槽位相对顺序。
        atom_type = atom_type[isnot_masked_atom]
        # atom_prob = atom_prob[isnot_masked_atom]
        # ``i``：int，当前保留节点的类别编号；查询 ``nodetype_to_ele`` 得到原子序数。
        # ``element``：np.ndarray[int64]，形状为 (N_keep,)；节点类别按 ``nodetype_to_ele`` 反查得到的原子序数。
        element = np.array([self.nodetype_to_ele[i] for i in atom_type])
        
        # get pos
        # ``atom_pos``：np.ndarray，形状为 (N_keep, 3)；保留原子的世界坐标或原局部坐标，布尔掩码作用于 ``pos`` 第一维，单位 Å。
        atom_pos = pos[isnot_masked_atom]
        # ``atom_pos_masked``：np.ndarray，形状为 (N-N_keep, 3)；非法或 mask 节点槽位的坐标，仅作为诊断字段返回，单位 Å。
        atom_pos_masked = pos[~isnot_masked_atom]
        
        # get bond
        if self.num_edge_types == 1:
            return {
                # ``element``：np.ndarray[int64]，形状为 (N_keep,)；保留节点的原子序数。
                'element': element,
                # ``atom_pos``：np.ndarray，形状为 (N_keep, 3)；保留原子的世界坐标或原局部坐标，单位 Å。
                'atom_pos': atom_pos,
                # 'atom_prob': atom_prob,
            }
        # if is_prob:
        #     pred_halfedge = softmax(pred_halfedge, axis=-1)
        #     edge_type = np.argmax(pred_halfedge, axis=-1)  # omit half for simplicity
        #     edge_prob = np.max(pred_halfedge, axis=-1)
        # else:
        # ``edge_type``：np.ndarray，形状为 (H,)；无向半边离散类别的别名。
        edge_type = halfedge
        # edge_prob = np.ones(len(edge_type))
        
        # ``is_bond``：np.ndarray[bool]，形状为 (H,)；True 表示半边类别是 1..K_bond 的真实化学键，同时遮盖类别和端点列。
        is_bond = (edge_type > 0) & (edge_type <= self.num_bond_types)  # larger is mask type
        # ``bond_type``：np.ndarray，形状为 (M_raw,)；从 H 条半边筛出的真实无向化学键类别。
        bond_type = edge_type[is_bond]
        # bond_prob = edge_prob[is_bond]
        # ``bond_index``：np.ndarray[int64]，形状为 (2, M_raw)；真实键的原节点槽位端点，尚未删除指向 mask 节点的列。
        bond_index = halfedge_index[:, is_bond]
        if not isnot_masked_atom.all():
            # [2, M_raw] -> [2, M_raw]，把原槽位编号换成压缩原子编号；无对应原子的端点变为 -1。
            # ``bond_index``：[2, M_raw] -> [2, M_raw]；按槽位到保留原子的映射重编号，删除节点端点变为 -1。
            bond_index = edge_index_changer[bond_index]
            # ``bond_for_masked_atom``：np.ndarray[bool]，形状为 (M_raw,)；True 表示该预测键至少连接一个已删除节点。
            bond_for_masked_atom = (bond_index < 0).any(axis=0)
            # ``bond_index``：[2, M_raw] -> [2, M_keep]；删除任一端点没有保留原子的键列。
            bond_index = bond_index[:, ~bond_for_masked_atom]
            # ``bond_type``：[M_raw] -> [M_keep]；按相同键列掩码同步过滤键类别。
            bond_type = bond_type[~bond_for_masked_atom]
            # bond_prob = bond_prob[~bond_for_masked_atom]

        # [M_keep] -> [2M_keep]；类别顺序与正向、反向两组端点逐列对齐。
        # ``bond_type``：[M_keep] -> [2M_keep]；为正反两个方向各复制一次无向键类别。
        bond_type = np.concatenate([bond_type, bond_type])
        # bond_prob = np.concatenate([bond_prob, bond_prob])
        # [2, M_keep] -> [2, 2M_keep]；交换两行得到每条无向键的反向端点。
        # ``bond_index``：[2, M_keep] -> [2, 2M_keep]；在列轴拼接原端点与交换端点后的反向边。
        bond_index = np.concatenate([bond_index, bond_index[::-1]], axis=1)
        
        return {
            # ``element``：np.ndarray[int64]，形状为 (N_keep,)；保留节点的原子序数。
            'element': element,
            # ``atom_pos``：np.ndarray，形状为 (N_keep, 3)；保留原子的世界坐标或原局部坐标，单位 Å。
            'atom_pos': atom_pos,
            # ``bond_type``：np.ndarray，形状为 (2M_keep,)；与双向端点列对齐的化学键类别。
            'bond_type': bond_type,
            # ``bond_index``：np.ndarray[int64]，形状为 (2, 2M_keep)；压缩编号后的双向化学键端点。
            'bond_index': bond_index,
            
            # 'atom_prob': atom_prob,
            # 'bond_prob': bond_prob,
            
            # ``atom_pos_masked``：np.ndarray，形状为 (N-N_keep, 3)；未解码节点槽位的坐标，单位 Å。
            'atom_pos_masked': atom_pos_masked,
        }

# XXX
@register_transforms('mixed')
class MixedTransform:
    """
    在单样本层按 ``data['task']`` 分派到对应任务变换。

    配置字段:
        - individual: list[config], 每项必须含唯一 ``name``；构造时由 ``get_transforms`` 建成 ``transform_dict[name]``。

    输入与输出:
        - data.task: str, 当前样本任务名；必须是 ``transform_dict`` 的键。
        - 返回值: 选中任务变换的返回对象；构象 ``conf`` 与 docking ``dock`` 均由 ``ConfTransform`` 原地补充字段。

    批处理边界:
        - ``ForeverTaskDataset`` 在 PyG DataLoader 拼批前调用本类，因此这里的 task 是单个字符串，不是长度 B 的任务列表。
        - ``exclude_keys`` 是所有子变换排除字段的串联列表，供 DataLoader 跳过不能默认拼接的单样本辅助结构。
    """
    def __init__(self, config, **kwargs):
        # ``self.config``：EasyDict，包含 individual 子任务变换列表。
        self.config = config
        # ``self.transform_dict``：dict[str, callable]，任务名到已实例化单样本变换的映射。
        self.transform_dict = {}
        for task_cfg in config.individual:
            # ``self.transform_dict``：callable，以 task_cfg.name 为键；conf 和 dock 分别构造 ConfTransform 实例。
            self.transform_dict[task_cfg.name] = get_transforms(task_cfg, **kwargs)

        # ``self.exclude_keys``：list[str]，按 individual 顺序串联所有子变换的排除字段；不去重。
        self.exclude_keys = sum([getattr(t, 'exclude_keys', []) 
                                 for t in self.transform_dict.values()], [])

    def __call__(self, data: Mol3DData):
        # ``task``：str，当前单样本任务名；必须精确匹配 transform_dict 的一个键。
        task = data['task']
        return self.transform_dict[task](data)

# XXX
@register_transforms('conf')
@register_transforms('dock')
class ConfTransform:
    """
    为分子构象生成和小分子 docking 构造位置任务 prompt、刚体域与可旋转键注释.

    注册名称:
        - ``conf``: 分子构象生成; 训练 reduced 配置几乎总选 ``free``, 少量选择 ``torsional``.
        - ``dock``: 小分子 docking; 训练 reduced 配置几乎总选 ``free``, 少量选择 ``flexible``.

    配置字段:
        - settings: dict[str, float], setting 名到采样概率; 键只能来自 ``free``、``flexible``、``torsional``、``rigid``, 概率原样传给 ``np.random.choice``.
        - mode: 由构造 kwargs 提供; ``train`` 不复制真值且排除 ``task_setting``, 其他模式保留真值并补充 flexible 距离标记.
        - free_no_geometry: bool, 缺省 False 保持原路径; True 时当前 free 验证/采样也使用空刚体域和扭转注释, 不读取真实扭转资产或启用 flexible 修正.

    输入样本核心字段:
        - node_type: int64, (N,), 真值原子类别.
        - node_pos: (N, 3), 真值局部坐标, 最后一维按 XYZ 排列, 单位 Å.
        - halfedge_index: int64, (2, H), 完全图上三角半边端点.
        - halfedge_type: int64, (H,), 真值半边类别.

        - fixed_dist_torsion: 0/1, (N, N), 1 表示该原子对距离在绕任意可旋转键变化时保持不变, 0 表示可因扭转改变.
        - path_mat: float, (N, N), RDKit 化学图最短路径长度矩阵.
        - nbh_dict: dict[int, list[int]], 每个原子编号到一跳相邻原子编号列表的映射.
        - tor_bond_mat: 0/1, (N, N), 对称可旋转键邻接矩阵; 1 表示对应化学键可扭转.
        - tor_twisted_pairs: dict[tuple[int,int], list[set[int],set[int]]], 断开规范化可旋转键后两侧除轴端点外的原子集合.

    通用输出字段:
        - task_setting: str, 本样本采到的 ``free``/``flexible``/``torsional``/``rigid``.
        - fixed_node: 0/1, (N,), 全 1; 构象与 docking 均不改变元素类别.
        - fixed_pos: 0/1, (N,), 默认全 0; 坐标是两个目标任务的去噪变量.
        - fixed_halfedge: 0/1, (H,), 全 1; 构象与 docking 均不改变化学键类别.
        - fixed_halfdist: 0/1, (H,), ``fixed_distmat`` 在半边端点处的取值; 1 表示损失/修正器应保持该端点距离.

        - n_domain: int64 标量, 刚体域数量; 当前非空分支把全部 N 个原子放入同一个域.
        - domain_node_index: int64, (2, N_domain_nodes), 第一行是刚体域编号, 第二行是原子编号.
        - tor_bonds_anno: int64, (T, 3), 每行 ``[BFS 层级, 远离中心的轴端点, 靠近中心的轴端点]``.
        - twisted_nodes_anno: int64, (W, 2), 每行 ``[tor_bonds_anno 行号, 随该键旋转的原子编号]``.
        - dihedral_pairs_anno: int64, (Q, 3), 每行 ``[tor_bonds_anno 行号, 轴左端邻居, 轴右端邻居]``, 用于构造四原子二面角.

    推理额外字段:
        - gt_node_type: int64, (N,), 变换时真值原子类别副本.
        - gt_node_pos: (N, 3), 变换时真值局部坐标副本, 单位 Å.
        - gt_halfedge_type: int64, (H,), 变换时真值半边类别副本.
        - fixed_halfdist_flex: 0/1, (H,), 原路径从 fixed_dist_torsion 取半边值; 本项目 free_no_geometry=True 且 setting=free 时为兼容原接口的全0张量.

    setting 语义:
        - free: ``fixed_halfdist`` 全 0, 允许所有原子对距离变化; 训练时扭转/刚体注释为空.
        - flexible/torsional: ``fixed_halfdist`` 来自 ``fixed_dist_torsion``, 只允许跨可旋转键两侧的距离变化.
        - rigid: ``fixed_halfdist`` 全 1, 所有分子内距离固定; 扭转注释为空但刚体域覆盖全部原子.
    """
    def __init__(self, config, **kwargs):
        # ``config.settings.free``：float|缺省，逐原子自由坐标模式采样权重。
        # ``config.settings.flexible``：float|缺省，整体平移、旋转和内部扭转模式采样权重。
        # ``config.settings.torsional``：float|缺省，仅内部扭转模式采样权重。
        # ``config.settings.rigid``：float|缺省，仅整体平移与旋转模式采样权重。
        # ``self.config``：EasyDict，保留上述 setting 权重。
        self.config = config

        # ``self.mode``：str，``train`` 与非训练值决定是否排除 ``task_setting``、是否复制 ``gt_*`` 字段。
        # ``mode``：str，与 ``self.mode`` 同值；仅用于当前构造函数的训练分支判断。
        self.mode = mode = kwargs.get('mode', 'test')
        # ``self.exclude_keys``：list[str]；以下单样本扭转与图匹配辅助叶在任务变换后不进入 PyG Batch。
        self.exclude_keys = [
            'bond_rotatable',      # (2M,)，原始双向化学键的可旋转标记。
            'tor_twisted_pairs',   # dict，规范化旋转轴到断键两侧非轴原子集合的映射。
            'fixed_dist_torsion',  # (N, N)，扭转下保持不变的原子对距离标记。
            'path_mat',            # (N, N)，化学图最短路径长度矩阵。
            'nbh_dict',            # dict[int, list[int]]，逐原子一跳邻居索引。
            'tor_bond_mat',        # (N, N)，对称可旋转键邻接矩阵。
            'matches_graph',       # 图匹配辅助对象；当前任务变换不批处理。
            'matches_iso',         # 同构匹配辅助对象；当前任务变换不批处理。
        ]
        if mode == 'train':
            self.exclude_keys.extend([
                'task_setting',  # str，训练时的随机运动模式；模型前向不直接消费字符串。
            ])
        
    
        # ``self.settings``：tuple[list[str], list[float]]；索引 0 为运动模式名，索引 1 为逐项对齐的采样概率。
        self.settings = (list(config.settings.keys()), list(config.settings.values()))
        # ``s``：str，当前待校验的运动模式名；必须属于 ``CONF_SETTINGS``。
        assert np.all((s in CONF_SETTINGS) for s in self.settings[0]), f'unknown conf/dock setting {self.settings[0]}'

        self.fix_some = config.get('fix_some', None)
    
    def __call__(self, data: PocketMolData):
        """选择构象约束模式, 并写入 fixed prompt、刚体域、扭转和推理真值叶.

        输入字段:
            - data.node_type: LongTensor, 形状为 (N,), 干净原子类别.
            - data.node_pos: FloatTensor, 形状为 (N, 3), 干净配体局部坐标, 单位 Å.
            - data.halfedge_index: LongTensor, 形状为 (2, H), 完全图半边端点.
            - data.halfedge_type: LongTensor, 形状为 (H,), 干净半边类别.

            以下图几何字段只由真实扭转/刚体分支读取; free_no_geometry=True 的 free 路径不要求它们存在.
            - data.fixed_dist_torsion: Tensor, 形状为 (N, N), 扭转下保持距离的 0/1 矩阵.
            - data.path_mat: Tensor, 形状为 (N, N), 化学图最短路径(在分子的化学键连接图中, 两个原子之间最少需要经过多少条键).
            - data.nbh_dict: dict[int, list[int]], 逐原子一跳邻居编号.
            - data.tor_bond_mat: Tensor, 形状为 (N, N), 逐原子对可旋转键标记.
            - data.tor_twisted_pairs: dict[tuple[int, int], list[set[int], set[int]]], 表示 可旋转键两侧非轴原子集合. 单个key-value-pair是: (可旋转键端点原子A下标, 可旋转键端点原子B下表): [A这一侧的非轴原子集合, B这一侧的非轴原子集合].

        输出字段:
            - data.task_setting: str, 采到的 ``free``、``flexible``、``torsional`` 或 ``rigid``.
            - data.fixed_node: LongTensor, 形状为 (N,), 原子类别条件掩码.
            - data.fixed_pos: LongTensor, 形状为 (N,), 坐标条件掩码.
            - data.fixed_halfedge: LongTensor, 形状为 (H,), 半边类别条件掩码.
            - data.fixed_halfdist: LongTensor, 形状为 (H,), 半边距离条件掩码.

            本项目 free 路径的 n_domain 为标量0, 其余刚体/扭转索引为约定形状的空张量, 供原 loss、批处理和采样接口使用.
            - data.n_domain: LongTensor 标量, 刚体域数.在处理整体平移和旋转时, 被视为同一个整体的一组原子, 即, 对这个刚体域施加刚体变换等于对内部每个原子施加变换.
            - data.domain_node_index: LongTensor, (2, K), 第一行索引刚体域, 第二行索引配体原子; K 为登记的域—原子归属对数, 本项目 free 无几何分支 K=0, 真实整体刚体分支 K=N.
            - data.tor_bonds_anno: LongTensor, 形状为 (T, 3), 扭转执行层级、轴的内侧端点、轴的外侧端点, 一个原子可以出现多次.每条可旋转键只旋转远离分子图中心的那一侧, 靠近中心的一侧保持不动.
            - data.twisted_nodes_anno: LongTensor, 形状为 (W, 2), [tor_bonds_anno 的行号, 随该轴转动的某个原子编号], 一个原子可以出现多次.
            - data.dihedral_pairs_anno: LongTensor, 形状为 (Q, 3), [tor_bonds_anno 的行号, 第一个外侧原子, 第二个外侧原子].如果轴两侧分别有 P 和 R 个非轴邻居, 代码会构造笛卡尔积, 因此产生: PxR 个二面角实例.

        推理额外字段:
            - gt_node_type: int64, (N,), 变换时真值原子类别副本.
            - gt_node_pos: (N, 3), 变换时真值局部坐标副本, 单位 Å.
            - gt_halfedge_type: int64, (H,), 变换时真值半边类别副本.
            - fixed_halfdist_flex: 0/1, (H,), 原路径从 fixed_dist_torsion 按半边端点抽取; 本项目显式 free_no_geometry 的 free 路径为全0, 不供 flexible 修正读取.

        setting 语义:
            - free: ``fixed_halfdist`` 全 0, 允许所有原子对距离变化; 训练时扭转/刚体注释为空.
            - flexible/torsional: ``fixed_halfdist`` 来自 ``fixed_dist_torsion``, 只允许跨可旋转键两侧的距离变化.
            - rigid: ``fixed_halfdist`` 全 1, 所有分子内距离固定; 扭转注释为空但刚体域覆盖全部原子.
        """
        # sample setting
        # ``setting``: str, 本样本唯一的构象约束模式; 决定固定距离 prompt 和扭转注释分支.
        setting = self.sample_setting()
        if 'is_atom_remain'  in data:  # peptide is cut. only applicable for free setting
            raise ValueError('not supported anymore: is_atom_remain')
            setting = 'free'
        data.update({
            # ``data.task_setting``: str, 本样本采到的 ``free``、``flexible``、``torsional`` 或 ``rigid`` 运动模式.
            'task_setting': setting,
        })

        # set fixed
        # ``data``: PocketMolData, 新增 ``fixed_node``、``fixed_pos``、``fixed_halfedge`` 与 ``fixed_halfdist`` 叶.
        data = self.set_fixed(data, setting)
        
        # torsional 
        # ``data``: PocketMolData, 新增 ``n_domain``、``domain_node_index`` 与三张扭转注释表.
        data = self.set_torsional_feat(data, setting)
        
        # for the sample mode, prepare init data
        if self.mode != 'train':
            # ``data``: PocketMolData, 非训练模式新增 ``gt_node_type``、``gt_node_pos`` 与 ``gt_halfedge_type`` 副本.
            data = self.prepare_sample(data, setting)
            
        if self.mode != 'train':
            if setting == 'free' and self.config.get('free_no_geometry', False):
                # int64, (H,), 本项目 free 路径不执行 flexible 修正, 仅保留原接口要求的全0距离约束.
                fixed_halfdist_flex = torch.zeros_like(data['halfedge_type'])
            else:
                # (N, N), 原路径的扭转不变距离矩阵; 旧配置和真实 flexible 任务继续读取原资产.
                fixed_distmat_flex = torch.LongTensor(data['fixed_dist_torsion'])
                # (H,), 每条完全图半边在原始扭转约束下是否保持端点距离.
                fixed_halfdist_flex = fixed_distmat_flex[data['halfedge_index'][0], data['halfedge_index'][1]]
            data.update({
                # (H,), 原入口为扭转距离约束; 本项目 free 无几何入口仅保存兼容全0张量.
                'fixed_halfdist_flex': fixed_halfdist_flex,
            })
        
        return data

    def prepare_sample(self, data, setting):
        """
        为采样保留变换时的离散图与坐标真值，不在此处制造初始噪声。

        输入参数:
            - data: ``PocketMolData``，至少含 ``node_type(N,)``、``node_pos(N,3)`` 与 ``halfedge_type(H,)``。
            - setting: str, 调用方传入的 setting；当前函数随后从 ``data['task_setting']`` 重新读取但不据此分支。

        输出字段:
            - gt_node_type: int64, (N,), ``node_type`` 的独立张量副本。
            - gt_node_pos: (N, 3), ``node_pos`` 的独立张量副本，单位 Å。
            - gt_halfedge_type: int64, (H,), ``halfedge_type`` 的独立张量副本。
        """
        # # make gt 
        # ``data.gt_node_type``：LongTensor，形状为 (N,)；原子类别独立副本，供采样约束和评测使用。
        data['gt_node_type'] = data['node_type'].clone()
        # ``data.gt_node_pos``：FloatTensor，形状为 (N, 3)；局部真值坐标独立副本，列顺序为 XYZ，单位 Å。
        data['gt_node_pos'] = data['node_pos'].clone()
        # ``data.gt_halfedge_type``：LongTensor，形状为 (H,)；无向半边真值类别独立副本。
        data['gt_halfedge_type'] = data['halfedge_type'].clone()

        # # make init
        # ``setting``：str，从数据对象重读的当前自由度模式；现实现仅保留给下方注释掉的旧初始化逻辑。
        setting = data['task_setting']
        # if setting == 'free':  #! no need to remove node_pos. controlled by init_step
        #     data['node_pos'] = torch.zeros_like(data['node_pos'])
        # else:  # torsional and flexible. ~~init is from rdkit~~. Not use rdkit, local is not accurate.
        #     # data = add_rdkit_conf(data=data)
        #     pass
        return data
    
    def set_torsional_feat(self, data: PocketMolData, setting):
        """
        以化学图中心为根, 对可旋转键排序并建立刚体/扭转索引字段.

        输入参数:
            - data.node_type: LongTensor, 形状为 (N,), 只用长度确定配体原子数.
            - data.path_mat: Tensor, 形状为 (N, N), 化学图最短路径矩阵, 单位为键数.
            - data.nbh_dict: dict[int, list[int]], 每个原子编号到一跳邻居编号列表的映射.
            - data.tor_bond_mat: Tensor, 形状为 (N, N), 1 表示对应原子对形成可旋转键.
            - data.tor_twisted_pairs: dict[tuple[int, int], list[set[int], set[int]]], 规范方向可旋转键到断键两侧非轴原子的映射.
            - setting: str, 当前约束模式.

        返回字段:
            - data.n_domain: LongTensor 标量, 当前非空分支为 1; free 无几何分支为 0.
            - data.domain_node_index: LongTensor, 形状为 (2, K), 第一行是域编号, 第二行是原子编号; free 无几何分支形状为 (2, 0).
            - data.tor_bonds_anno: LongTensor, 形状为 (T, 3), BFS 从图中心向外发现的层级、远端轴原子和近端轴原子.
            - data.twisted_nodes_anno: LongTensor, 形状为 (W, 2), 可旋转键行号到轴远端随动原子的映射.
            - data.dihedral_pairs_anno: LongTensor, 形状为 (Q, 3), 可旋转键行号到两侧一跳邻居笛卡尔积的映射.

        关键分支:
            - setting=free 且 mode=train 或 free_no_geometry=True: 返回全部空注释, 本项目 free 训练、验证和采样都不需要真实扭转几何.
            - ``setting == 'rigid'``: 保留覆盖全部原子的单刚体域, 但返回空扭转注释.
            - 其他分支含未启用 free_no_geometry 的旧 free 推理; 仍构造扭转注释, 保留旧几何修正入口.
        """
        if setting == 'free' and (self.mode == 'train' or self.config.get('free_no_geometry', False)):
            data.update({
                # ``data.n_domain``: int64 标量 0, free 无几何样本不声明任何刚体域.
                'n_domain': torch.tensor(0, dtype=torch.long),
                # ``data.domain_node_index``: LongTensor, 形状为 (2, 0), 没有域—原子归属对.
                'domain_node_index': torch.empty([2, 0], dtype=torch.long),
                # ``data.tor_bonds_anno``: LongTensor, 形状为 (0, 3), free 无几何分支不构造扭转键行.
                'tor_bonds_anno': torch.empty([0, 3], dtype=torch.long),
                # ``data.twisted_nodes_anno``: LongTensor, 形状为 (0, 2), free 无几何分支不构造扭转键—随动原子对.
                'twisted_nodes_anno': torch.empty([0, 2], dtype=torch.long),
                # ``data.dihedral_pairs_anno``: LongTensor, 形状为 (0, 3), free 无几何分支不构造扭转键—二面角端点行.
                'dihedral_pairs_anno': torch.empty([0, 3], dtype=torch.long)
            })
            return data
        
        # # rigid domain
        # assert setting in ['flexible', 'torsional', 'rigid']
        # ``n_node``: int, 当前分子的配体原子数 N; 对应 ``node_type`` 第一维长度.
        n_node = data['node_type'].shape[0]
        # ``n_domain``: LongTensor 标量, 值为 1; 当前实现把全部原子视为一个候选刚体域.
        n_domain = torch.tensor(1, dtype=torch.long)
        # ``domain_node_index``: LongTensor, 形状为 (2, N); 第一行全为域 0, 第二行依次列出 ``node_type`` 第一维的原子编号.
        domain_node_index = torch.stack([
            torch.zeros(n_node, dtype=torch.long),
            torch.arange(n_node, dtype=torch.long)
        ], dim=0)
        
        
        # # center nodes
        # ``path_mat``: Tensor, 形状为 (N, N); 化学图最短路径长度, 对角为 0, 单位为键数.
        path_mat = data['path_mat']
        # ``nbh_dict``: dict[int, list[int]], 键和值均索引 ``node_type`` 第一维; 值只含一跳化学键邻居.
        nbh_dict = data['nbh_dict']

        # ``margin``: np.ndarray|Tensor, 形状为 (N,); 第 i 个值是原子 i 到最远原子的图距离, 即图论离心率.
        margin = path_mat.max(0)
        # ``node_c0``: int, 离心率最小的图中心原子索引; 并列中心中均匀随机选择一个.
        node_c0 = np.random.choice(np.argwhere(margin == margin.min()).reshape(-1))
        # neigh_c0 = nbh_dict[node_c0]
        # if len(neigh_c0) >= 2:
        #     neigh_margin = margin[neigh_c0]
        #     node_c1, node_c2 = np.random.choice(neigh_c0, 2, replace=False,
        #                                         p=neigh_margin/neigh_margin.sum())
        # else:
        #     raise ValueError('only one node in the domain')
        # domain_center_nodes = torch.tensor([[node_c0, node_c1, node_c2]], dtype=torch.long)

        if setting == 'rigid':
            data.update({
                # ``data.n_domain``: int64 标量 1, 整分子被视为一个刚体域.
                'n_domain': torch.tensor(1, dtype=torch.long),
                # ``data.domain_node_index``: LongTensor, 形状为 (2, N), 第一行全 0, 第二行是全部原子索引.
                'domain_node_index': torch.stack([
                    torch.zeros(data.num_nodes, dtype=torch.long),
                    torch.arange(data.num_nodes, dtype=torch.long)
                ], dim=0),
                # 'domain_center_nodes': domain_center_nodes,
                # ``data.tor_bonds_anno``: LongTensor, 形状为 (0, 3), rigid 模式不允许分子内扭转.
                'tor_bonds_anno': torch.empty([0, 3], dtype=torch.long),
                # ``data.twisted_nodes_anno``: LongTensor, 形状为 (0, 2), 没有扭转键—随动原子对.
                'twisted_nodes_anno': torch.empty([0, 2], dtype=torch.long),
                # ``data.dihedral_pairs_anno``: LongTensor, 形状为 (0, 3), 没有扭转键—二面角端点行.
                'dihedral_pairs_anno': torch.empty([0, 3], dtype=torch.long)
            })
            return data
        
        # ``tor_bond_mat``: Tensor, 形状为 (N, N); 0/1 对称可旋转键邻接矩阵.
        tor_bond_mat = data['tor_bond_mat']
        # ``n_tor_bonds``: Tensor 标量, 可旋转无向键数 T; 对称矩阵求和后除以 2.
        n_tor_bonds = tor_bond_mat.sum() / 2
        if n_tor_bonds == 0:
            # ``tor_bonds_anno``: LongTensor, 形状为 (0, 3); 无可旋转键时的空轴注释.
            tor_bonds_anno = torch.empty([0, 3], dtype=torch.long)
            # ``twisted_nodes_anno``: LongTensor, 形状为 (0, 2); 无可旋转键时的空随动原子注释.
            twisted_nodes_anno = torch.empty([0, 2], dtype=torch.long)
            # ``dihedral_pairs_anno``: LongTensor, 形状为 (0, 3); 无可旋转键时的空二面角邻居注释.
            dihedral_pairs_anno = torch.empty([0, 3], dtype=torch.long)
        else:
            # # torsional bonds
            # BFS+DFS to determine the order of the torsional bonds and the trees
            # ``global_remain``: np.ndarray[bool], 形状为 (N,); True 表示该原子尚未由 BFS 访问, 度为 1 的叶原子随后预置 False.
            global_remain = np.ones(n_node, dtype=bool)
            # ``node``: int, 当前原子的 0-based 索引.
            # ``nbh``: list[int], 当前原子的一跳化学键邻居索引.
            for node, nbh in nbh_dict.items():
                if len(nbh) == 1:
                    # ``global_remain[node]``: bool 标量; 叶原子不作为 BFS 中间查询点, 提前标为已访问.
                    global_remain[node] = False  # trick 1: frontier node no need to visit
            # ``curr_order``: int, 当前 BFS 层级; 写入 ``tor_bonds_anno`` 第一列.
            curr_order = 0
            # ``query_pool``: list[int], 当前层待访问的原子编号; 数值索引 ``node_type`` 第一维.
            query_pool = [node_c0]
            # ``tor_bonds_anno``: list[list[int, int, int]], 按发现顺序暂存 ``[层级, 远端轴原子, 近端轴原子]``.
            tor_bonds_anno = []
            # ``early_stop``: bool; 找到矩阵统计的全部 T 条旋转键后终止嵌套 BFS.
            early_stop = False  # trick 2: if all torsional bonds are found, stop
            while global_remain.any():
                #print('in loop 523')
                # ``next_query``: list[int], 跨当前层可旋转键后进入下一 BFS 层的远端轴原子编号.
                next_query = []
                while len(query_pool) > 0:
                    #print('in loop 526')
                    # mark the node as visited
                    # ``curr_node``: int, 当前访问原子编号; 数值索引 ``node_type`` 第一维.
                    curr_node = query_pool.pop(0)
                    # ``global_remain[curr_node]``: bool 标量; 把当前原子标为已访问, 避免再次加入 BFS.
                    global_remain[curr_node] = False
                    # add neighbors to query pool
                    # ``nbh``: list[int], 当前原子的一跳化学键邻居编号; 每个值索引原子维.
                    nbh = nbh_dict[curr_node]
                    # ``nb_node``: int, 当前检查的一跳邻居原子索引, 作用于 ``global_remain`` 与 ``tor_bond_mat`` 的原子维.
                    for nb_node in nbh:
                        if (not global_remain[nb_node]) or (nb_node in query_pool):
                            pass  # already visited or in query pool
                        elif tor_bond_mat[curr_node, nb_node] == 1: # find a torsional bond
                            tor_bonds_anno.append([curr_order, nb_node, curr_node])
                            next_query.append(nb_node)
                            if len(tor_bonds_anno) == n_tor_bonds:
                                # ``early_stop``: bool, 已发现矩阵统计的全部 T 条可旋转键, 结束两层循环.
                                early_stop = True
                                break
                        else:  # continue to search
                            query_pool.append(nb_node)
                    if early_stop:
                        break
                if early_stop:
                    break
                # ``curr_order``: int, BFS 层级加一; 后续发现的扭转键写入新的层级值.
                curr_order += 1
                # ``query_pool``: list[int], 把跨当前层扭转键到达的远端轴原子作为下一层起点.
                query_pool = next_query

            # # twisted nodes
            # ``tor_twisted_pairs``: dict[tuple[int, int], list[set[int], set[int]]]; 规范化轴端点到断键两侧非轴原子集合的映射.
            tor_twisted_pairs = data['tor_twisted_pairs']
            # ``twisted_nodes_anno``: list[list[int, int]], 暂存 ``[可旋转键行号, 随动原子编号]``.
            twisted_nodes_anno = []
            # ``index_tor``: int, 当前扭转键在 ``tor_bonds_anno`` 中的行号.
            # ``tor_left``: int, 当前有向扭转轴远离中心的端点原子索引.
            # ``tor_right``: int, 当前有向扭转轴靠近中心的端点原子索引.
            # ``_``: int, 未使用的 BFS 层级; 此循环只需两个轴端点.
            for index_tor, (_, tor_left, tor_right) in enumerate(tor_bonds_anno):
                if tor_left < tor_right:
                    # ``all_nodes_left``: set[int], 沿 ``tor_left`` 一侧且不含轴端点的原子; 按构造方向应远离中心 ``node_c0``.
                    all_nodes_left = tor_twisted_pairs[(tor_left, tor_right)][0]
                else:
                    # ``all_nodes_left``: set[int], 规范化字典以较小端点为键时, 取反方向对应的断键侧集合.
                    all_nodes_left = tor_twisted_pairs[(tor_right, tor_left)][1]
                assert (node_c0 not in all_nodes_left), 'center node c0 should not be in the twisted nodes'
                # assert (node_c1 not in all_nodes_left), 'center node c1 should not be in the twisted nodes'
                # assert (node_c2 not in all_nodes_left), 'center node c2 should not be in the twisted nodes'
                
                # ``node``: int, 当前远端随动原子索引; 写入二元组第二列.
                twisted_nodes_anno.extend([index_tor, node] for node in all_nodes_left)
            # ``twisted_nodes_anno``: LongTensor, 形状为 (W, 2); 第一列索引 ``tor_bonds_anno``, 第二列索引 ``node_type`` 第一维.
            twisted_nodes_anno = torch.tensor(twisted_nodes_anno, dtype=torch.long)
            
            # # dihedral pairs
            # ``dihedral_pairs_anno``: list[list[int, int, int]], 暂存 ``[可旋转键行号, 左侧邻居, 右侧邻居]``.
            dihedral_pairs_anno = []
            # ``index_tor``: int, 当前扭转键在 ``tor_bonds_anno`` 中的行号.
            # ``tor_left``: int, 当前有向扭转轴远离中心的端点原子索引.
            # ``tor_right``: int, 当前有向扭转轴靠近中心的端点原子索引.
            # ``_``: int, 未使用的 BFS 层级; 此循环只需两个轴端点.
            for index_tor, (_, tor_left, tor_right) in enumerate(tor_bonds_anno):
                # ``n``: int, 当前左轴端的一跳邻居原子索引; 等于右轴端时过滤.
                # ``nbh_left``: list[int], 左轴端排除右轴端后的一跳邻居; 数值索引原子维.
                nbh_left = [n for n in nbh_dict[tor_left] if n != tor_right]
                # ``n``: int, 当前右轴端的一跳邻居原子索引; 等于左轴端时过滤.
                # ``nbh_right``: list[int], 右轴端排除左轴端后的一跳邻居; 数值索引原子维.
                nbh_right = [n for n in nbh_dict[tor_right] if n != tor_left]
                # ``node_left``: int, 当前左轴端的外侧邻居原子索引.
                # ``node_right``: int, 当前右轴端的外侧邻居原子索引.
                dihedral_pairs_anno.extend([index_tor, node_left, node_right]
                                        for node_left, node_right in product(nbh_left, nbh_right))
            # ``dihedral_pairs_anno``: LongTensor, 形状为 (Q, 3); 第一列索引 ``tor_bonds_anno``, 后两列索引原子维.
            dihedral_pairs_anno = torch.tensor(dihedral_pairs_anno, dtype=torch.long)

            # ``tor_bonds_anno``: LongTensor, 形状为 (T, 3); 列表顺序成为可旋转键行号, 供另外两张注释表引用.
            tor_bonds_anno = torch.tensor(tor_bonds_anno, dtype=torch.long)
            
        # # combine
        data.update({
            # ``data.n_domain``: int64 标量, 当前非 rigid 返回路径为覆盖全部 N 个原子的单域 1.
            'n_domain': n_domain,
            # ``data.domain_node_index``: LongTensor, 形状为 (2, N), 逐原子给出域 0 与原子索引.
            'domain_node_index': domain_node_index,
            # 'domain_center_nodes': domain_center_nodes,
            # ``data.tor_bonds_anno``: LongTensor, 形状为 (T, 3), 逐扭转键保存 BFS 层级与有向轴端点.
            'tor_bonds_anno': tor_bonds_anno,
            # ``data.twisted_nodes_anno``: LongTensor, 形状为 (W, 2), 逐行关联扭转键行号与随动原子索引.
            'twisted_nodes_anno': twisted_nodes_anno,
            # ``data.dihedral_pairs_anno``: LongTensor, 形状为 (Q, 3), 逐行关联扭转键行号与两个外侧原子索引.
            'dihedral_pairs_anno': dihedral_pairs_anno,
        })
            
        return data

    def sample_setting(self):
        """按配置权重采样一个构象或 docking 运动模式。

        输入字段:
            - self.settings[0]: list[str]，非零权重模式名，元素取 ``free``、``flexible``、``torsional`` 或 ``rigid``。
            - self.settings[1]: list[float]，与模式名逐项对齐且总和为 1 的采样概率。

        返回值:
            - setting: str，本次抽中的运动模式名。
        """
        # ``setting``：str，单次 ``np.random.choice`` 结果；候选与概率按 ``self.settings`` 两个列表的位置对齐。
        setting = np.random.choice(self.settings[0], p=self.settings[1])
        return setting

    def set_fixed(self, data, setting):
        """
        为构象/docking 写入固定类别、可变坐标和 setting 对应的固定距离 prompt。变化都在 def __call__ 里。

        输入参数:
            - data.node_type: int64, (N,), 用于确定原子数量。
            - data.halfedge_type: int64, (H,), 用于确定半边数量。
            - data.halfedge_index: int64, (2, H), 从 ``fixed_distmat`` 抽取逐半边距离标记。
            - data.fixed_dist_torsion: 0/1, (N, N), flexible/torsional setting 的分子内距离不变量矩阵。
            - setting: str, ``free``、``flexible``、``torsional`` 或 ``rigid``。

        输出字段:
            - fixed_node: 0/1, (N,), 全 1，固定原子类别。
            - fixed_pos: 0/1, (N,), 默认全 0，允许所有小分子原子坐标去噪。
            - fixed_halfedge: 0/1, (H,), 全 1，固定化学键类别。
            - fixed_halfdist: 0/1, (H,), free 全 0，rigid 全 1，flexible/torsional 从 ``fixed_dist_torsion`` 按半边端点抽取。
        """
        # ``n_node``：int，原子类别使用的配体原子数 N；对应 ``node_type`` 第一维长度。
        # ``n_pos``：int，坐标使用的配体原子数 N；现实现与 ``n_node`` 相等。
        n_node = n_pos = data['node_type'].shape[0]
        # ``n_halfedge``：int，无向完全图半边数 H；对应 ``halfedge_type`` 第一维长度。
        n_halfedge =data['halfedge_type'].shape[0]
        # ``fixed_node``：LongTensor，形状为 (N,)，初始全 1，表示原子类别固定。
        # ``fixed_pos``：LongTensor，形状为 (N,)，初始全 0，表示配体坐标待恢复。
        # ``fixed_halfedge``：LongTensor，形状为 (H,)，初始全 1，表示半边类别固定。
        fixed_node, fixed_pos, fixed_halfedge = get_vector_list(
                [n_node, n_pos, n_halfedge], [1, 0, 1])  # pos is not fixed
        data.update({
            # ``data.fixed_node``：LongTensor，形状为 (N,)，全 1 表示所有原子类别都是条件。
            'fixed_node': fixed_node,
            # ``data.fixed_pos``：LongTensor，形状为 (N,)，默认全 0 表示所有配体坐标都需恢复。
            'fixed_pos': fixed_pos,
            # ``data.fixed_halfedge``：LongTensor，形状为 (H,)，全 1 表示所有化学键类别都是条件。
            'fixed_halfedge': fixed_halfedge,
        })
        
        # for fixed dist
        if setting == 'free':
            # ``fixed_distmat``：LongTensor，形状为 (N, N)；``free`` 模式没有固定的分子内原子对距离。
            fixed_distmat = torch.zeros(n_node, n_node, dtype=torch.long)
        elif (setting == 'flexible') or (setting == 'torsional'):
            # ``fixed_distmat``：LongTensor，形状为 (N, N)；1 只标记绕任意可旋转键仍保持不变的原子对距离。
            fixed_distmat = torch.LongTensor(data['fixed_dist_torsion'])
        elif setting == 'rigid':
            # ``fixed_distmat``：LongTensor，形状为 (N, N)；``rigid`` 模式固定所有分子内原子对距离。
            fixed_distmat = torch.ones(n_node, n_node, dtype=torch.long)
        # ``fixed_halfdist``：LongTensor，形状为 (H,)；对每条上三角半边抽取其两个端点的固定距离标记。
        fixed_halfdist = fixed_distmat[data['halfedge_index'][0], data['halfedge_index'][1]]
        data.update({
            # ``data.fixed_halfdist``：LongTensor，形状为 (H,)，逐半边标出端点距离是否作为条件保持。
            'fixed_halfdist': fixed_halfdist,
        })
        
        if self.fix_some is not None:
            if isinstance(self.fix_some, str):
                if self.fix_some == 'endres':  # fix the end residue, they cannot be moved
                    peptide_res_index = data['peptide_res_index']
                    is_endres = (peptide_res_index == 0) | (peptide_res_index == peptide_res_index.max())
                    fixed_pos[is_endres] = 1
                elif self.fix_some == 'endresbb':  # fix the end residue bb., they cannot be moved
                    peptide_res_index = data['peptide_res_index']
                    is_endres = (peptide_res_index == 0) | (peptide_res_index == peptide_res_index.max())
                    is_backbone = data['peptide_is_backbone']
                    is_endresbb = is_endres & is_backbone
                    fixed_pos[is_endresbb] = 1
                elif self.fix_some == 'bb':
                    is_backbone = data['peptide_is_backbone']
                    fixed_pos[is_backbone] = 1
                elif self.fix_some == 'firstres':
                    peptide_res_index = data['peptide_res_index']
                    is_first = (peptide_res_index == 0)
                    fixed_pos[is_first] = 1
            elif isinstance(self.fix_some, dict):
                fixed_atom_indices = np.array(self.fix_some.get('atom', []))
                if fixed_atom_indices.max() > data['num_nodes']-1:
                    num_nodes = data['num_nodes']
                    raise ValueError(f'Indices in fix_some out of range. There are {num_nodes} atoms. Max allowed index is {num_nodes-1}.')
                res_bb = self.fix_some.get('res_bb', [])
                res_sc = self.fix_some.get('res_sc', [])
                if res_bb or res_sc:
                    peptide_res_index = np.array(data['peptide_res_index'], dtype=np.int32)
                    is_backbone = np.array(data['peptide_is_backbone'], dtype=bool)
                    is_sel_res_bb = ((peptide_res_index[:, None] == np.array(res_bb)[None]).any(-1)
                                        & is_backbone)
                    is_sel_res_sc = ((peptide_res_index[:, None] == np.array(res_sc)[None]).any(-1)
                                        & (~is_backbone))
                    add_atoms_indices = np.nonzero(is_sel_res_bb | is_sel_res_sc)[0]
                    fixed_atom_indices = np.concatenate([fixed_atom_indices, add_atoms_indices])
                    
                fixed_atom_indices = np.unique(fixed_atom_indices)
                print('fix atoms with indices:', fixed_atom_indices)
                fixed_pos[fixed_atom_indices] = 1
            else:
                raise NotImplementedError(f'unknown fix_some {self.fix_some}')
            data.update({
                'fixed_pos': fixed_pos,
            })
        
        return data


def halfedge_index_to_1d(halfedge_index, num_nodes):
    """
    把完全图上三角半边端点映射为行优先的一维半边编号。

    形状符号:
        - N: 当前单图原子数。
        - H_sel: 输入选择的半边数。

    输入参数:
        - halfedge_index: int64, (2, H_sel), 每列为分子内原子编号 ``(i, j)``，要求 ``0 <= i < j < N``。
        - num_nodes: int, 当前分子的原子数 N。

    返回值:
        - id_edge: int64, (H_sel,), 每列端点在 ``torch.triu_indices(N, N, offset=1)`` 中的位置，范围为 ``[0, N(N-1)/2 - 1]``。
    """
    assert (halfedge_index[0] < halfedge_index[1]).all(), (
        'halfedge_index[0] must be smaller than halfedge_index[1]'
    )
    # ``id_edge``：[2, H_sel] -> [H_sel]；先累计端点 i 之前各行的半边数，再加本行内 j 的偏移。
    id_edge = (
        (2 * num_nodes - halfedge_index[0] - 1) * halfedge_index[0] // 2
        + halfedge_index[1] - halfedge_index[0] - 1
    )
    return id_edge

def add_rdkit_conf(data=None, mol=None):
    if mol is None:
        path = os.path.join('data', data.db, 'mols', data.data_id + '.sdf')
        mol = Chem.MolFromMolFile(path)
    else:
        assert data is None, 'data and mol cannot be both not None'
    # get rdkit conformer
    # mol.Compute2DCoords()
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, clearConfs=True)
    try:
        n_tries = 5
        for i_try in range(n_tries):
            notconverge = AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            if 1-notconverge:
                break
            else:
                # AllChem.EmbedMolecule(mol)
                if i_try == n_tries-1:
                    print(f'Failed to optimize molecule after {n_tries} tries.')
    except:
        print('Error when optimizing molecule.')
        pass
    mol = Chem.RemoveHs(mol)
    pos_rdkit = mol.GetConformer().GetPositions()
    # check atom element is not changed
    assert (data.element == 
            torch.LongTensor([atom.GetAtomicNum() for atom in mol.GetAtoms()])).all().item()
    # check bond is not changed
    new_bond_type = torch.LongTensor([mol.GetBondBetweenAtoms(*[a.item() for a in bond]).GetBondType()
                                                for bond in data.bond_index.T])
    new_bond_type = torch.where(new_bond_type==12, 4, new_bond_type)  # aromatic bond
    if not (new_bond_type == data.bond_type).all().item():
        print('Warning: bond type is changed after rdkit optimization')
        raise ValueError('Warning: bond type is changed after rdkit optimization')

    # overwirte atom pos
    node_pos = torch.FloatTensor(pos_rdkit)
    data['node_pos'] = node_pos - node_pos.mean(dim=0)
    data['i_conf'] = -1
    return data

@register_transforms('cut_peptide')  # seems not used
class CutPeptide(object):
    def __init__(self, config, *args, **kwargs) -> None:
        self.config = config
        self.applicable_tasks = config.applicable_tasks
        self.individual = config.individual
        self.exclude_keys = ['is_atom_remain']

    def __call__(self, data: PocketMolData):
        task = data['task']
        if (task in self.applicable_tasks) and ('peptide_res_index' in data): # only for peptide
            # cut peptide
            config_this = self.individual[task]
            prob = config_this.prob
            # n_res = data['peptide_pep_len']
            n_res = data['peptide_res_index'].max().item() + 1
            if np.random.rand() < prob or n_res > config_this.limit_nres:
                min_nres = config_this.min_nres
                max_nres = min(config_this.max_nres, n_res-1)
                n_res_remain = np.random.randint(min_nres, max_nres+1)
                index_res_start = np.random.randint(0, n_res-n_res_remain+1)
                index_res = torch.arange(index_res_start, index_res_start+n_res_remain)
                peptide_res_index = data['peptide_res_index']
                is_atom_remain = (peptide_res_index[:, None] == index_res[None]).any(dim=1)
            
                data['is_atom_remain'] = is_atom_remain
                assert is_atom_remain.sum() > 0, 'no atom remain'
                return data
        return data


@register_transforms('overwrite_start_pos')  # for dock flex. used _mol_start.sdf provided by posebuster. but not really necessary
class OverwriteStartPos(object):
    """
    用外部 SDF 的第一个 conformer 覆盖配体初始坐标，供 flexible docking 起点实验使用。

    配置字段:
        - start_mol_path: str, 起始 SDF 所在目录。
        - appendix: str, 拼在 ``data_id`` 后的文件名后缀；默认空字符串。
        - center_policy: str, ``unknown`` 表示以外部配体质心为原点，``known`` 表示减去当前样本 ``pocket_center``。

    输入样本字段:
        - data_id: str, 与 ``appendix`` 拼成 ``<start_mol_path>/<data_id><appendix>``。
        - node_pos: (N, 3), 仅提供输出 dtype；原数值会被完全覆盖。
        - node_type: int64, (N,), 用于检查外部分子原子数一致。
        - element: int64, (N,), 原子序数；必须与去氢后外部分子的 RDKit 原子顺序完全一致。
        - pocket_center: (1, 3), ``center_policy=known`` 时使用的原始口袋中心，单位 Å。

    输出样本字段:
        - node_pos: (N, 3), 外部 SDF 坐标经指定中心策略平移后的 XYZ 坐标，单位 Å。

    原地更新边界:
        - 本 transform 只执行 ``data.node_pos = node_pos``；输入容器中的既有键集合和其余值对象不被写入。
    """
    def __init__(self, config, *args, **kwargs):
        # ``config.start_mol_path``：str，外部起始 SDF 目录。
        # ``config.appendix``：str|缺省，拼在 ``data_id`` 后的文件名后缀。
        # ``config.center_policy``：Literal[unknown,known]|缺省，选择配体质心或口袋中心作为局部原点。
        # ``self.config``：EasyDict，保留上述外部坐标来源与中心化策略叶。
        self.config = config
        # ``self.start_mol_path``：str，外部起始 SDF 所在目录。
        self.start_mol_path = config['start_mol_path']
        # ``self.appendix``：str，追加到 ``data_id`` 后的文件名后缀；空字符串表示不追加。
        self.appendix = config.get('appendix', '')
        # ``self.center_policy``：Literal['unknown', 'known']；``unknown`` 减配体质心，``known`` 减 ``pocket_center``。
        self.center_policy = config.get('center_policy', 'unknown')
    
    def __call__(self, data):
        """读取外部起始 SDF，并只覆盖当前样本的配体局部坐标。

        输入字段:
            - data.data_id: str，与 ``appendix`` 拼成外部 SDF 文件名。
            - data.node_pos: FloatTensor，形状为 (N, 3)，提供输出 dtype；原数值会被覆盖。
            - data.node_type: LongTensor，形状为 (N,)，用于检查外部分子原子数。
            - data.element: LongTensor，形状为 (N,)，用于检查外部分子原子序数与顺序。
            - data.pocket_center: FloatTensor，形状为 (1, 3)，``center_policy=known`` 时的世界坐标原点，单位 Å。

        输出字段:
            - data.node_pos: FloatTensor，形状为 (N, 3)，外部 conformer 减指定中心后的局部坐标，单位 Å。
        """

        # ``data_id``：str，当前样本标识；用于拼接唯一的外部起始结构路径。
        data_id = data['data_id']
        # ``mol_path``：str，当前样本唯一的外部起始构象文件路径。
        mol_path = os.path.join(self.start_mol_path, data_id + self.appendix)
        # ``mol``：rdkit.Chem.Mol|None，从 SDF 读取的起始分子；默认执行 sanitize。
        mol = Chem.MolFromMolFile(mol_path)
        # ``mol``：rdkit.Chem.Mol，移除显式氢后的起始分子；原子顺序必须与 ``data.element`` 的重原子顺序一致。
        mol = Chem.RemoveAllHs(mol)
        if mol is None:
            print('mol is None: ', data_id, self.i_repeat)
        # ``atom_pos``：np.ndarray，形状为 (N, 3)；RDKit conformer 0 的世界坐标，列顺序为 XYZ，单位 Å。
        atom_pos = mol.GetConformer(0).GetPositions()
        # ``data.node_pos``：FloatTensor，形状为 (N, 3)；外部 conformer 世界坐标，dtype 与原 ``node_pos`` 对齐，单位 Å。
        data['node_pos'] = torch.tensor(atom_pos, dtype=data['node_pos'].dtype)
        if self.center_policy == 'unknown':
            # ``data.node_pos``：[N, 3] -> [N, 3]；减配体几何中心，转入配体质心局部坐标系，单位 Å。
            data['node_pos'] = data['node_pos'] - data['node_pos'].mean(0, keepdim=True)
        elif self.center_policy == 'known':
            # ``data.node_pos``：[N, 3] - [1, 3] -> [N, 3]；减口袋世界坐标中心，转入 docking 口袋局部坐标系，单位 Å。
            data['node_pos'] = data['node_pos']- data.pocket_center
        else:
            raise f'Invalid center_policy {self.center_policy}'
            
        assert len(data['node_pos']) == len(data['node_type']), 'size not match'
        assert [atom.GetAtomicNum() for atom in mol.GetAtoms()] == data['element'].tolist(), 'element not match'
        return data

@register_transforms('overwrite_pos')
class OverwritePos(object):  # for belief to get the gen pos
    """
    按 ``gen_info.csv`` 定位上一轮生成文件，并用其坐标覆盖当前配体位置。

    构造输入:
        - config.gen_path 或 kwargs.gen_path: str, 生成目录；必须含 ``gen_info.csv`` 和 ``SDF/`` 子目录。
        - config.i_repeat 或 kwargs.i_repeat: int, 选择每个 ``data_id`` 的第几次重复生成。

    ``gen_info.csv`` 读取字段:
        - data_id: str, 原输入样本标识；与待覆盖样本 ``data['data_id']`` 等值匹配。
        - i_repeat: int, 重复生成编号；与构造参数共同筛选唯一一行。
        - filename: str, ``SDF/`` 下的 ``.sdf`` 或 ``.pdb`` 文件名。

    输入样本字段:
        - data_id: str, 用于筛选 ``gen_info.csv``。
        - node_pos: (N, 3), 仅提供输出 dtype；找到可读分子时原数值会被覆盖。
        - pocket_center: (1, 3), 原始口袋中心；从生成文件的世界坐标中减去，单位 Å。
        - node_type: LongTensor，形状为 (N,)，用于检查外部分子原子数。
        - element: LongTensor，形状为 (N,)，用于检查外部分子原子序数与顺序。

    输出样本字段:
        - filename: str, 当前选中的生成文件名。
        - i_repeat: int, 当前选中的重复编号。
        - node_pos: (N, 3), 以 ``pocket_center`` 为原点的生成配体坐标，单位 Å；文件读取失败时保留原值。
    """
    def __init__(self, config, *args, **kwargs):
        # ``self.config``：dict|EasyDict|None；非空时从配置读取生成目录和重复号，为空时改从 ``kwargs`` 读取。
        self.config = config
        if config is not None:
            # ``self.gen_path``：str，包含 ``gen_info.csv`` 与 ``SDF/`` 子目录的生成运行目录。
            self.gen_path = config.gen_path
            # ``self.i_repeat``：int，同一 ``data_id`` 的目标重复生成编号。
            self.i_repeat = config.i_repeat
        else:
            # ``self.gen_path``：str，由调用方运行参数提供的生成目录。
            self.gen_path = kwargs['gen_path']
            # ``self.i_repeat``：int，由调用方运行参数提供的重复编号。
            self.i_repeat = kwargs['i_repeat']
        
        # ``self.df_gen``：pd.DataFrame，形状为 (R, C_csv)；每行描述一个生成候选，调用时按 ``data_id`` 与 ``i_repeat`` 筛成唯一记录。
        self.df_gen = pd.read_csv(os.path.join(self.gen_path, 'gen_info.csv'))
        
    def __call__(self, data):
        # i_repeat = i_repeat if i_repeat is not None else self.i_repeat
        # assert i_repeat is not None, 'i_repeat is not specified'

        # ``data_id``：str，当前样本标识；与 ``self.i_repeat`` 联合筛选 CSV 唯一记录。
        data_id = data['data_id']
        # ``line``：pd.DataFrame，形状为 (1, C_csv)；唯一匹配当前输入与重复编号的候选记录。
        line = self.df_gen[(self.df_gen['data_id'] == data_id) & (self.df_gen['i_repeat'] == self.i_repeat)]
        assert len(line) == 1, ('find multiple lines or none with the same data_id and i_repeat', data_id, self.i_repeat, line)
        # ``filename``：str，唯一匹配行的 ``filename`` 叶；取值为 ``SDF/`` 下的结构文件名。
        filename = line['filename'].values[0]
        # ``mol_path``：str，生成候选的实际 SDF/PDB 路径。
        mol_path = os.path.join(self.gen_path, 'SDF', filename)
        # ``data.filename``：str，下游重建与评分使用的候选文件名。
        data['filename'] = filename
        # ``data.i_repeat``：int，当前候选的重复编号；与 ``filename`` 一起写回样本元数据。
        data['i_repeat'] = self.i_repeat
        
        if filename.endswith('.sdf'):
            # ``mol``：rdkit.Chem.Mol|None，从生成 SDF 读取并执行 sanitize。
            mol = Chem.MolFromMolFile(mol_path)
        elif filename.endswith('.pdb'):
            # ``mol``：rdkit.Chem.Mol|None，从生成 PDB 读取且不执行 sanitize，以保留输入原子记录。
            mol = Chem.MolFromPDBFile(mol_path, sanitize=False)
        if mol is None:
            print('mol is None: ', data_id, self.i_repeat)
            # print('Use random pos insteads')
            # data['node_pos'] = torch.
        else:
            # ``atom_pos``：np.ndarray，形状为 (N, 3)；生成文件 conformer 0 的世界坐标，列顺序为 XYZ，单位 Å。
            atom_pos = mol.GetConformer(0).GetPositions()
            # ``data.node_pos``：FloatTensor，形状为 (N, 3)；世界坐标减 ``pocket_center`` 后的口袋局部配体坐标，单位 Å。
            data['node_pos'] = (torch.tensor(atom_pos, dtype=data['node_pos'].dtype)
                                - data.pocket_center)
            assert len(data['node_pos']) == len(data['node_type']), 'size not match'
            assert [atom.GetAtomicNum() for atom in mol.GetAtoms()] == data['element'].tolist(), 'element not match'
        return data


@register_transforms('overwrite_pos_repeat')
class OverwritePosRepeat(object):  # for linking with unknown fragmen pos to get the initial overwrite
    def __init__(self, config, i_repeat):
        self.config = config
        self.i_repeat = i_repeat
        
    def __call__(self, data):
        
        if self.config['starategy'] == 'linking_unfixed':
            file_dir = self.config['file_dir']
            sdf_dirs = os.listdir(file_dir)
            sep_name = data['key'].split(';')[-1].replace('linking/', '').replace('/', '_sep_')
            sdf_name = [name for name in sdf_dirs if name.endswith(sep_name)]
            assert len(sdf_name) == 1, f'find {len(sdf_name)} files with name {sep_name}'
            sdf_name = sdf_name[0]
            
            # add repeat
            filename = os.path.join(sdf_name, f'repeat_{self.i_repeat}.sdf')
            mol_path = os.path.join(file_dir, filename)
            data['filename'] = filename
            data['i_repeat'] = self.i_repeat
        
        
        if filename.endswith('.sdf'):
            mol = Chem.MolFromMolFile(mol_path, sanitize=False)
        elif filename.endswith('.pdb'):
            mol = Chem.MolFromPDBFile(mol_path, sanitize=False)
        if mol is None:
            print('mol is None: ', data['data_id'], self.i_repeat)
            # print('Use random pos insteads')
            # data['node_pos'] = torch.
        else:
            atom_pos = mol.GetConformer(0).GetPositions()
            atom_pos = torch.tensor(atom_pos, dtype=data['node_pos'].dtype)
            if data['pocket_center'].shape[0] > 0:
                atom_pos = atom_pos - data.pocket_center
            else:
                atom_pos = atom_pos - atom_pos.mean(0, keepdim=True)
            data['node_pos'] = atom_pos
            assert len(data['node_pos']) == len(data['node_type']), 'size not match'
            assert [atom.GetAtomicNum() for atom in mol.GetAtoms()] == data['element'].tolist(), 'element not match'
        return data


@register_transforms('overwrite_mol_repeat')
class OverwriteMolRepeat(object):  # for mol optimize to overwrite the init mol (so that no need for re-assemble db)
    def __init__(self, config, i_repeat):
        self.config = config
        self.i_repeat = i_repeat
        
        if self.config['starategy'] == 'mol_opt':
            self.df = pd.read_csv(os.path.join(self.config['file_root'], self.config['df_path']))
        
    def __call__(self, data):
        
        if self.config['starategy'] == 'mol_opt':
            total_files = self.config['total_files']
            sdf_dir = os.path.join(self.config['file_root'], self.config['sdf_dir'])
            
            data_id = data['data_id']
            file_repeat = self.i_repeat % total_files
            
            line = self.df[(self.df['data_id'] == data_id) & (self.df['i_repeat'] == file_repeat)]
            assert len(line) == 1, 'find multiple lines or none with the same data_id and i_repeat'
            filename = line['filename'].values[0]
            
            # add repeat
            mol_path = os.path.join(sdf_dir, filename)
            data['filename'] = filename
            # data['i_repeat'] = self.i_repeat
            
            # overwite
            assert data['task'] == 'sbdd', 'only sbdd supported, otherwise process more mol info'
            new_data = process_raw(data_id=data_id, mol_path=mol_path, modes=['mols'],
                                pdbid=data.get('pdbid', ''))
            # for data_key in new_data.keys:
            #     if data_key in data.keys:
            #         data[data_key] = new_data[data_key]
            for data_key in new_data.keys():
                if data_key in data:
                    data[data_key] = new_data[data_key]
            return data


@register_transforms('variable_sc_size')
class VariableScSize(object):  # for sampling
    def __init__(self, config, *args, **kwargs) -> None:
        self.config = config
        self.applicable_tasks = config.applicable_tasks
        self.num_atoms_distri = config.num_atoms_distri
        self.exclude_keys = ['is_atom_remain', 'added_index', 'removed_index']
        
        self.not_remove = config.get('not_remove', [])

    def __call__(self, data: PocketMolData):
        task = data['task']
        if (task in self.applicable_tasks) and ('peptide_res_index' in data): # only for peptide
            n_atoms_data = data['node_type'].shape[0]
            n_res = data['peptide_pep_len']
            assert n_res == data['peptide_res_index'].max().item() + 1

            
            n_atoms_mean = self.num_atoms_distri['mean'] * n_res
            n_atoms_std = self.num_atoms_distri['std']['coef'] * n_res + self.num_atoms_distri['std']['bias']
            n_atoms_new = int(np.random.normal(n_atoms_mean, n_atoms_std))
            if n_atoms_new == n_atoms_data: # what a coincidence!!!
                pass
            elif n_atoms_new > n_atoms_data: # add atoms
                data = self.add_atoms(data, n_atoms_new, n_atoms_data)
            elif n_atoms_new < n_atoms_data: # remove atoms
                data = self.remove_atoms(data, n_atoms_new, n_atoms_data)
                n_atoms_new = data['node_type'].shape[0]
            # common
            if 'is_peptide' in data:
                is_peptide = data['is_peptide']
                data['is_peptide'] = is_peptide[0] * torch.ones([n_atoms_new], dtype=is_peptide.dtype)
        return data
            
    def add_atoms(self, data, n_atoms_new, n_atoms_data):
        n_add = n_atoms_new - n_atoms_data
        
        # new node
        node_type = data['node_type']
        new_node_type = torch.cat([node_type, torch.zeros([n_add], dtype=node_type.dtype)], dim=0)
        added_index = np.arange(n_atoms_data, n_atoms_new)
        
        # new node pos
        # determine positions
        is_sidechain = (~data['peptide_is_backbone'])
        n_sc = is_sidechain.sum()
        if n_sc != 0:
            node_sc = torch.nonzero(is_sidechain)[:, 0]
            node_sc_center = node_sc[torch.randint(n_sc, size=[n_add])]
        else:
            node_sc = torch.nonzero(~is_sidechain)[:, 0]  # no sc. can only use backbone
            node_sc_center = node_sc[torch.randint(len(node_sc), size=[n_add])]
        len_mu, len_sigma = 3, 0.6
        lengths = torch.randn([n_add]) * len_sigma + len_mu
        relative_pos = torch.randn([n_add, 3])
        relative_pos = relative_pos / (relative_pos.norm(dim=-1, keepdim=True)+1e-5) * lengths[:, None]
        node_pos = data['node_pos']
        node_pos_new = node_pos[node_sc_center] + relative_pos
        new_node_pos = torch.cat([node_pos, node_pos_new], dim=0)
        
        # new edge
        halfedge_index = data['halfedge_index']
        halfedge_type = data['halfedge_type']
        n_add_halfedge = n_add * n_atoms_data + n_add * (n_add - 1) // 2
        new_halfedge_type = torch.cat([halfedge_type, torch.zeros([n_add_halfedge], dtype=halfedge_type.dtype)], dim=0)
        halfedge_index_old_new = torch.stack(
            torch.meshgrid(torch.arange(n_atoms_data), torch.arange(n_atoms_data, n_atoms_new), indexing='ij'),
        dim=0).reshape(2, -1)
        halfedge_index_new_new = torch.triu_indices(n_add, n_add, offset=1) + n_atoms_data
        new_halfedge_index = torch.cat([
            halfedge_index, halfedge_index_old_new, halfedge_index_new_new], dim=1)
        new_halfedge_index, new_halfedge_type = sort_edge_index(new_halfedge_index, new_halfedge_type)

        # peptide feature
        peptide_is_backbone = data['peptide_is_backbone']
        peptide_atom_name = data['peptide_atom_name']
        new_peptide_is_backbone = torch.cat([peptide_is_backbone, torch.zeros([n_add], dtype=peptide_is_backbone.dtype)], dim=0)
        new_peptide_atom_name = peptide_atom_name + ['X'] * n_add

        data.update({
            'num_nodes': n_atoms_new,
            'node_type': new_node_type,
            'node_pos': new_node_pos,
            'halfedge_index': new_halfedge_index,
            'halfedge_type': new_halfedge_type,
            'peptide_is_backbone': new_peptide_is_backbone,
            'peptide_atom_name': new_peptide_atom_name,
            'added_index': added_index,  # for custom task to know the part name of the added nodes
        })
        return data
    
    
    def remove_atoms(self, data, n_atoms_new, n_atoms_data):
        n_remove = n_atoms_data - n_atoms_new
        peptide_is_backbone = data['peptide_is_backbone']
        peptide_is_sc = ~peptide_is_backbone
        
        
        peptide_is_removable = peptide_is_sc
        peptide_is_removable[self.not_remove] = False
        # n_bb, n_sc = peptide_is_backbone.sum(), peptide_is_sc.sum()
        n_removable = peptide_is_removable.sum().item()
        n_remove = min(n_remove, n_removable)

        is_atom_remain = torch.ones([n_atoms_data], dtype=torch.bool)
        index_removable = torch.nonzero(peptide_is_removable)[:, 0]
        index_remove = np.random.choice(index_removable, n_remove, replace=False)
        is_atom_remain[index_remove] = False

        # node 
        data['node_pos'] = data['node_pos'][is_atom_remain]
        data['node_type'] = data['node_type'][is_atom_remain]
        data['num_nodes'] = data['node_type'].shape[0]
        data['removed_index'] = index_remove  # for custom task to rearange the index of remaining nodes
        # edge
        halfedge_index, halfedge_type = subgraph(
            torch.nonzero(is_atom_remain)[:, 0], edge_index=data.halfedge_index,
            edge_attr=data.halfedge_type, relabel_nodes=True, num_nodes=n_atoms_data)
        data['halfedge_index'] = halfedge_index
        data['halfedge_type'] = halfedge_type
        # peptide
        data['peptide_is_backbone'] = data['peptide_is_backbone'][is_atom_remain]
        data['peptide_atom_name'] = [name for is_r, name in zip(is_atom_remain, data['peptide_atom_name'])
                                     if is_r]
        
        return data


@register_transforms('variable_mol_size')
class VariableMolSize(object):  # for sampling
    def __init__(self, config, *args, **kwargs) -> None:
        self.config = config
        # self.applicable_tasks = config.applicable_tasks
        self.num_atoms_distri = config.num_atoms_distri
        self.exclude_keys = ['is_atom_remain', 'added_index', 'removed_index']
        
        self.not_remove = config.get('not_remove', [])

    def __call__(self, data: PocketMolData):
        
        n_atoms_data = data['node_type'].shape[0]
        n_atoms_new = self.sample_n_atoms(data)

        if n_atoms_new == n_atoms_data: # what a coincidence~~~
                pass
        elif n_atoms_new > n_atoms_data: # add atoms
            data = self.add_atoms(data, n_atoms_new, n_atoms_data)
        elif n_atoms_new < n_atoms_data: # remove atoms
            data = self.remove_atoms(data, n_atoms_new, n_atoms_data)
            n_atoms_new = data['node_type'].shape[0]
        # common
        if 'is_peptide' in data:
            is_peptide = data['is_peptide']
            data['is_peptide'] = is_peptide[0] * torch.ones([n_atoms_new], dtype=is_peptide.dtype)
        return data

    def sample_n_atoms(self, data):
        strategy = self.num_atoms_distri['strategy']
        if strategy == 'pocket_atoms_based':
            num_atoms_pocket = len(data['pocket_pos'])
            n_atoms_mean = self.num_atoms_distri['mean']['coef'] * num_atoms_pocket + self.num_atoms_distri['mean']['bias']
            n_atoms_std = self.num_atoms_distri['std']['coef'] * num_atoms_pocket + self.num_atoms_distri['std']['bias']
            sample_func = lambda: int(np.round(np.random.normal(n_atoms_mean, n_atoms_std)))
        elif strategy == 'mol_atoms_based':
            num_atoms = len(data['node_pos'])
            n_atoms_mean = self.num_atoms_distri['mean']['coef'] * num_atoms + self.num_atoms_distri['mean']['bias']
            n_atoms_std = self.num_atoms_distri['std']['coef'] * num_atoms + self.num_atoms_distri['std']['bias']
            sample_func = lambda: int(np.round(np.random.normal(n_atoms_mean, n_atoms_std)))
        elif strategy == 'multinomial':
            sizes = self.num_atoms_distri['values']
            probs = self.num_atoms_distri['probs']
            sample_func = lambda: int(np.random.choice(sizes, p=probs))
        else:
            raise NotImplementedError(f'num_atoms_distri strategy {self.num_atoms_distri["name"]} not implemented')
        
        n_atoms_min = self.num_atoms_distri.get('min', 2)
        n_atoms_max = self.num_atoms_distri.get('max', 1000000)
        # sample n
        count_try = 0
        while True:
            n_atoms_new = sample_func()
            count_try += 1
            if n_atoms_new >= n_atoms_min and n_atoms_new <= n_atoms_max:
                break
            if count_try > 1000:
                print('Warning: too many tries to sample n_atoms for variable_mol_size')
                n_atoms_new = n_atoms_min
                break
        return n_atoms_new
            
            
    def add_atoms(self, data, n_atoms_new, n_atoms_data):
        n_add = n_atoms_new - n_atoms_data
        
        # new node
        node_type = data['node_type']
        new_node_type = torch.cat([node_type, torch.zeros([n_add], dtype=node_type.dtype)], dim=0)
        added_index = np.arange(n_atoms_data, n_atoms_new)
        
        # new node pos
        # determine positions
        len_mu, len_sigma = 2, 0.5
        node_pos = data['node_pos']
        node_pos_center = node_pos[torch.randint(n_atoms_data, size=[n_add])]

        lengths = torch.randn([n_add]) * len_sigma + len_mu
        relative_pos = torch.randn([n_add, 3])
        relative_pos = relative_pos / (relative_pos.norm(dim=-1, keepdim=True)+1e-5) * lengths[:, None]
        node_pos_new = node_pos_center + relative_pos
        new_node_pos = torch.cat([node_pos, node_pos_new], dim=0)
        
        # new edge
        halfedge_index = data['halfedge_index']
        halfedge_type = data['halfedge_type']
        n_add_halfedge = n_add * n_atoms_data + n_add * (n_add - 1) // 2
        new_halfedge_type = torch.cat([halfedge_type, torch.zeros([n_add_halfedge], dtype=halfedge_type.dtype)], dim=0)
        halfedge_index_old_new = torch.stack(
            torch.meshgrid(torch.arange(n_atoms_data), torch.arange(n_atoms_data, n_atoms_new), indexing='ij'),
        dim=0).reshape(2, -1)
        halfedge_index_new_new = torch.triu_indices(n_add, n_add, offset=1) + n_atoms_data
        new_halfedge_index = torch.cat([
            halfedge_index, halfedge_index_old_new, halfedge_index_new_new], dim=1)
        new_halfedge_index, new_halfedge_type = sort_edge_index(new_halfedge_index, new_halfedge_type)

        # peptide feature
        # peptide_is_backbone = data['peptide_is_backbone']
        # peptide_atom_name = data['peptide_atom_name']
        # new_peptide_is_backbone = torch.cat([peptide_is_backbone, torch.zeros([n_add], dtype=peptide_is_backbone.dtype)], dim=0)
        # new_peptide_atom_name = peptide_atom_name + ['X'] * n_add

        data.update({
            'num_nodes': n_atoms_new,
            'node_type': new_node_type,
            'node_pos': new_node_pos,
            'halfedge_index': new_halfedge_index,
            'halfedge_type': new_halfedge_type,
            # 'peptide_is_backbone': new_peptide_is_backbone,
            # 'peptide_atom_name': new_peptide_atom_name,
            'added_index': added_index,  # for custom task to know the part name of the added nodes
        })
        return data
    
    
    def remove_atoms(self, data, n_atoms_new, n_atoms_data):
        n_remove = n_atoms_data - n_atoms_new

        is_removable = torch.ones([n_atoms_data], dtype=torch.bool)
        is_removable[self.not_remove] = False
        n_removable = is_removable.sum().item()
        n_remove = min(n_remove, n_removable)
        if n_remove == 0:
            return data

        is_atom_remain = torch.ones([n_atoms_data], dtype=torch.bool)
        index_removable = torch.nonzero(is_removable)[:, 0]
        index_remove = np.random.choice(index_removable, n_remove, replace=False)
        is_atom_remain[index_remove] = False

        # node 
        data['node_pos'] = data['node_pos'][is_atom_remain]
        data['node_type'] = data['node_type'][is_atom_remain]
        data['num_nodes'] = data['node_type'].shape[0]
        data['removed_index'] = index_remove  # for custom task to rearange the index of remaining nodes
        # edge
        halfedge_index, halfedge_type = subgraph(
            torch.nonzero(is_atom_remain)[:, 0], edge_index=data.halfedge_index,
            edge_attr=data.halfedge_type, relabel_nodes=True, num_nodes=n_atoms_data)
        data['halfedge_index'] = halfedge_index
        data['halfedge_type'] = halfedge_type
        # peptide
        # data['peptide_is_backbone'] = data['peptide_is_backbone'][is_atom_remain]
        # data['peptide_atom_name'] = [name for is_r, name in zip(is_atom_remain, data['peptide_atom_name'])
        #                              if is_r]
        
        return data




def make_data_placeholder(n_graphs, device=None, max_size=None):
    # n_nodes_list = np.random.randint(15, 50, n_graphs)
    if max_size is None:
        n_nodes_list = np.random.normal(24.923464980477522, 5.516291901819105, size=n_graphs)
    else:
        n_nodes_list = np.array([max_size] * n_graphs)
    n_nodes_list = n_nodes_list.astype('int64')
    batch_node = np.concatenate([np.full(n_nodes, i) for i, n_nodes in enumerate(n_nodes_list)])
    halfedge_index = []
    batch_halfedge = []
    idx_start = 0
    for i_mol, n_nodes in enumerate(n_nodes_list):
        halfedge_index_this_mol = torch.triu_indices(n_nodes, n_nodes, offset=1)
        halfedge_index.append(halfedge_index_this_mol + idx_start)
        n_edges_this_mol = len(halfedge_index_this_mol[0])
        batch_halfedge.append(np.full(n_edges_this_mol, i_mol))
        idx_start += n_nodes
    
    batch_node = torch.LongTensor(batch_node)
    batch_halfedge = torch.LongTensor(np.concatenate(batch_halfedge))
    halfedge_index = torch.cat(halfedge_index, dim=1)
    
    if device is not None:
        batch_node = batch_node.to(device)
        batch_halfedge = batch_halfedge.to(device)
        halfedge_index = halfedge_index.to(device)
    return {
        # 'n_graphs': n_graphs,
        'batch_node': batch_node,
        'halfedge_index': halfedge_index,
        'batch_halfedge': batch_halfedge,
    }



@register_transforms('sbdd')
@register_transforms('denovo')
class DenovoTransform:
    def __init__(self, config, **kwargs):
        self.config = config

        self.mode = mode = kwargs.get('mode', 'test')
        
        
    def __call__(self, data: Mol3DData):
        # set fixed
        data = self.set_fixed(data)

        # fake flexible features
        data = self.set_fake_features(data)

        if self.mode == 'test':
            data = self.prepare_sample(data)
        return data

    def set_fake_features(self, data):
        data.update({
            'n_domain': torch.tensor(0, dtype=torch.long),
            'domain_node_index': torch.empty([2, 0], dtype=torch.long),
            # 'domain_center_nodes': torch.empty([0, 3], dtype=torch.long),
            'tor_bonds_anno': torch.empty([0, 3], dtype=torch.long),
            'twisted_nodes_anno': torch.empty([0, 2], dtype=torch.long),
            'dihedral_pairs_anno': torch.empty([0, 3], dtype=torch.long)
        })
        return data

    def set_fixed(self, data):
        n_node = data['node_type'].shape[0]
        n_halfedge =data['halfedge_type'].shape[0]
        fixed_node, fixed_pos, fixed_halfedge = get_vector_list(
                [n_node, n_node, n_halfedge], [0, 0, 0])
        fixed_distmat = torch.zeros(n_node, n_node, dtype=torch.long)
        fixed_halfdist = fixed_distmat[data['halfedge_index'][0], data['halfedge_index'][1]]
        
        
        data.update({
            'fixed_node': fixed_node,
            'fixed_pos': fixed_pos,
            'fixed_halfedge': fixed_halfedge,
            'fixed_halfdist': fixed_halfdist,
        })
        return data
    
    def prepare_sample(self, data):
        # # make gt : NOTE: when use varible_mol_size, these are not gt
        data['gt_node_type'] = data['node_type'].clone()
        data['gt_node_pos'] = data['node_pos'].clone()
        data['gt_halfedge_type'] = data['halfedge_type'].clone()

        # # make init
        # use init_level to control this
        # data['node_type'] = torch.zeros_like(data['node_type'])
        # data['node_pos'] = torch.zeros_like(data['node_pos'])
        # data['halfedge_type'] = torch.zeros_like(data['halfedge_type'])
        return data



@register_transforms('fbdd')
@register_transforms('maskfill')
class MaskfillTransform:
    def __init__(self, config, **kwargs) -> None:
        self.config = config
        self.mode = mode = kwargs.get('mode', 'test')
        self.exclude_keys = ['linkers', 'anchors_linkers', 'frags', 'anchors_frags',
                             'is_known_halfedge_p1p2', 
                             'bond_rotatable', 'tor_twisted_pairs', 'fixed_dist_torsion',
                             'path_mat', 'nbh_dict', 'tor_bond_mat', 'mmpa', 'brics', 'matches_graph', 'matches_iso']
        if mode == 'train':
            self.exclude_keys.extend([
                'task_setting',
                'node_p1', 'node_p2', 'halfedge_p1', 'halfedge_p2', 'halfedge_p1p2', 'groupof_node_p1',
                'grouped_node_p1', 'grouped_anchor_p1'])
            
        self.preset_partition = config.get('preset_partition', None)
        
        self.settings_dict = {}
        if 'settings' in config:
            for key, value_dict in config.settings.items():
                self.settings_dict[key] = {'options': list(value_dict.keys()), 'weights': list(value_dict.values())}
                assert all(op in MASKFILL_SETTINGS[key] for op in self.settings_dict[key]['options']),\
                        f"unknown maskfill setting {self.settings_dict[key]} for setting {key}"
    
    def __call__(self, data: Mol3DData):
        setting = self.sample_setting()
        data.update({'task_setting': setting})
        
        # make partiotion
        data = self.pre_make_partition(data)
        data, num_part_dict = self.make_partition_features(data)
        # set features
        data = self.set_fixed(data, setting, num_part_dict)
        data = self.set_torsional_feat(data, fake=(setting['part1_pert'] != 'flexible'))

        if self.mode == 'test':
            data = self.prepare_sample(data, setting)
        data = self.remedy_anchor_nodes(data)
        return data
    
    def remedy_anchor_nodes(self, data):
        # break the bonds between p1 and p2 (mainly for setting connecting atoms)
        halfedge_p1p2 = data["halfedge_p1p2"]
        data['halfedge_type'][halfedge_p1p2] = 0
        return data

    def prepare_sample(self, data, setting):
        # # make gt
        data['gt_node_type'] = data['node_type'].clone()
        data['gt_node_pos'] = data['node_pos'].clone()
        data['gt_halfedge_type'] = data['halfedge_type'].clone()
        
        # # make gt for part 1
        node_type = data['node_type']
        node_pos = data['node_pos']
        halfege_type = data['halfedge_type']
        
        for i_part in [1, 2]:
            node_type_part = - torch.ones_like(node_type)
            node_type_part[data[f'node_p{i_part}']] = node_type[data[f'node_p{i_part}']].clone()
            node_pos_part = node_pos.clone()
            halfedge_type_part = - torch.ones_like(halfege_type)
            halfedge_type_part[data[f'halfedge_p{i_part}']] = halfege_type[data[f'halfedge_p{i_part}']].clone()
            data.update({
                f'gt_node_type_p{i_part}': node_type_part,
                f'gt_node_pos_p{i_part}': node_pos_part,
                f'gt_halfedge_type_p{i_part}': halfedge_type_part,
            })
        
        # # make init
        if setting['part1_pert'] in ['fixed', 'free', 'small', 'rigid', 'flexible']:
            data['node_type'][data['node_p2']] = 0
            # for p1 fixed (not has pocket), influence center before add noise
            if data['node_p1'].shape[0] > 0:
                data['node_pos'][data['node_p2']] = (data['node_pos'][data['node_p1']]).mean(dim=0)
            else:
                data['node_pos'] = torch.zeros_like(data['node_pos'])
            data['halfedge_type'][data['halfedge_p2']] = 0
            data['halfedge_type'][data['halfedge_p1p2']] = 0
        else:
            raise NotImplementedError
        return data

    def sample_setting(self):
        setting_dict = {}
        for setting, opt_dict in self.settings_dict.items():
            setting_dict[setting] = np.random.choice(opt_dict['options'], p=opt_dict['weights'])
        return setting_dict
    
    def set_fixed(self, data: Mol3DData, setting, num_part_dict):
        n_node_p1, n_node_p2 = num_part_dict['n_node_p1'], num_part_dict['n_node_p2']
        n_halfedge_p1, n_halfedge_p2 = num_part_dict['n_halfedge_p1'], num_part_dict['n_halfedge_p2']
        n_halfedge_p1p2 = num_part_dict['n_halfedge_p1p2']
        
        # # for part 1
        if setting['part1_pert'] == 'fixed':
            fixed_node_p1, fixed_pos_p1, fixed_halfedge_p1 = get_vector_list(
                [n_node_p1, n_node_p1, n_halfedge_p1], [1, 1, 1])
        elif setting['part1_pert'] == 'small':
            fixed_node_p1, fixed_pos_p1, fixed_halfedge_p1 = get_vector_list(
                [n_node_p1, n_node_p1, n_halfedge_p1], [0, 0, 0])
        elif setting['part1_pert'] in ['free', 'rigid', 'flexible']:
            fixed_node_p1, fixed_pos_p1, fixed_halfedge_p1 = get_vector_list( # pos is not fixed
                [n_node_p1, n_node_p1, n_halfedge_p1], [1, 0, 1])
        else:
            raise ValueError(f"Unknown part1_pert: {setting['part1_pert']}")
        
        # # for connections, use anchor or not
        is_known_halfedge_p1p2 = data['is_known_halfedge_p1p2']
        fixed_halfedge_p1p2 = get_vector([n_halfedge_p1p2], 0)
        fixed_halfedge_p1p2[is_known_halfedge_p1p2] = 1
        
        # # part 2
        fixed_node_p2, fixed_pos_p2, fixed_halfedge_p2 = get_vector_list(
                [n_node_p2, n_node_p2, n_halfedge_p2], [0, 0, 0])

        fixed_dict = {
            'node_p1': fixed_node_p1,
            'pos_p1': fixed_pos_p1,
            'halfedge_p1': fixed_halfedge_p1,
            'node_p2': fixed_node_p2,
            'pos_p2': fixed_pos_p2,
            'halfedge_p2': fixed_halfedge_p2,
            'halfedge_p1p2': fixed_halfedge_p1p2,
        }
        
        # # combine p1 and p2
        fixed_node = combine_vectors_indexed(
            [fixed_dict[f'node_p1'], fixed_dict[f'node_p2']],
            [data['node_p1'], data['node_p2']],
        )
        fixed_pos = combine_vectors_indexed(
            [fixed_dict[f'pos_p1'], fixed_dict[f'pos_p2']],
            [data['node_p1'], data['node_p2']],
        )
        fixed_halfedge = combine_vectors_indexed(
            [fixed_dict[f'halfedge_p1'], fixed_dict[f'halfedge_p2'], fixed_dict[f'halfedge_p1p2']],
            [data['halfedge_p1'], data['halfedge_p2'], data['halfedge_p1p2']],
        )
        data.update({
            'fixed_node': fixed_node,
            'fixed_pos': fixed_pos,
            'fixed_halfedge': fixed_halfedge,
        })
        
        # # for fixed_dist
        if setting['part1_pert'] == 'rigid':
            domain_node_index = torch.stack([
                data['groupof_node_p1'], data['node_p1']
            ], dim=0)
            n_domain = domain_node_index[0].max() + 1 if domain_node_index.shape[1] > 0 else torch.tensor(0, dtype=torch.long)
            fixed_distmat = get_rigid_distmat(domain_node_index[0], domain_node_index[1],
                                              n_domain, data['node_type'].shape[0])
            fixed_halfdist = fixed_distmat[
                data['halfedge_index'][0], data['halfedge_index'][1]]
        elif setting['part1_pert'] == 'flexible':
            domain_node_index = torch.stack([
                data['groupof_node_p1'], data['node_p1']
            ], dim=0)
            n_domain = domain_node_index[0].max() + 1 if domain_node_index.shape[1] > 0 else torch.tensor(0, dtype=torch.long)
            # n_domain = domain_node_index[0].max() + 1
            fixed_distmat = get_rigid_distmat(domain_node_index[0], domain_node_index[1],
                                              n_domain, data['node_type'].shape[0])
            fixed_dist_torsion = torch.LongTensor(data['fixed_dist_torsion'])
            fixed_distmat = torch.where(fixed_dist_torsion==0, fixed_dist_torsion, fixed_distmat)
            fixed_halfdist = fixed_distmat[
                data['halfedge_index'][0], data['halfedge_index'][1]]
        else:  # free, small, fixed
            domain_node_index = torch.empty([2, 0], dtype=torch.long)  # set rigid domain to empty
            n_domain = torch.tensor(0, dtype=torch.long)
            fixed_halfdist = torch.zeros_like(data['halfedge_type'], dtype=torch.long)  # default not fixed distances
            if setting['part1_pert'] == 'fixed':  # fixed distances among part1 nodes
                fixed_halfdist[data['halfedge_p1']] = 1
        data.update({
            'fixed_halfdist': fixed_halfdist,
            'n_domain': n_domain,
            'domain_node_index': domain_node_index,
        })

        return data
    
    def set_torsional_feat(self, data: Mol3DData, fake=False):
        domain_node_index = data['domain_node_index']
        domain_index = domain_node_index[0]
        
        if fake or (domain_index.shape[0] == 0):
            data.update({
                'tor_bonds_anno': torch.empty([0, 3], dtype=torch.long),
                'twisted_nodes_anno': torch.empty([0, 2], dtype=torch.long),
                'dihedral_pairs_anno': torch.empty([0, 3], dtype=torch.long),
            })
            return data
        
        # # rigid domain
        n_domain = domain_index.max() + 1
        n_node = data['node_type'].shape[0]
        domain_index_of_node = np.full((n_node,), -1)
        domain_index_of_node[domain_node_index[1].numpy()] = domain_node_index[0].numpy()
        
        # # center nodes
        path_mat = data['path_mat']
        nbh_dict = data['nbh_dict']
        
        fixed_distmat = get_rigid_distmat(domain_node_index[0], domain_node_index[1],
                                              n_domain, n_node)
        path_mat_domain = path_mat * fixed_distmat.numpy()
        margin = path_mat_domain.max(0)
        node_c0_list = []
        for i_domain in range(n_domain):
            margin_domain = margin[domain_index_of_node == i_domain]
            idx_domain = np.argmin(margin_domain)
            node_c0 = np.argwhere(domain_index_of_node == i_domain).flatten()[idx_domain]
            node_c0_list.append(node_c0)
        
        tor_bond_mat = data['tor_bond_mat']
        n_tor_bonds = tor_bond_mat.sum() / 2

        # # torsional bonds
        node_p1, node_p2 = data['node_p1'].numpy(), data['node_p2'].numpy()
        global_remain = np.ones(n_node, dtype=bool)
        global_remain[node_p2] = False
        for node in node_p1:
            domain_this = domain_index_of_node[node]
            nbh = nbh_dict[node]
            if len([n for n in nbh if domain_index_of_node[n] == domain_this]) == 1:
                global_remain[node] = False  # trick 1: frontier node no need to visit
            if len([n for n in nbh if domain_index_of_node[n] == domain_this]) == 0:
                global_remain[node] = False  # node in part1 has no neighbor in part1
        curr_order = 0
        query_pool = node_c0_list
        tor_bonds_anno = []
        early_stop = False
        n_tor_bonds = (tor_bond_mat[global_remain][:, global_remain]).sum() / 2
        while global_remain.any():
            #print('in loop 844')
            next_query = []
            while len(query_pool) > 0:
                #print('in loop 847')
                # mark the node as visited
                curr_node = query_pool.pop(0)
                global_remain[curr_node] = False
                domain_curr = domain_index_of_node[curr_node]
                # add neighbors IN p1 to query pool
                nbh = nbh_dict[curr_node]
                for nb_node in nbh:
                    if domain_index_of_node[nb_node] != domain_curr:
                        continue  # not in the same domain
                    if (not global_remain[nb_node]) or (nb_node in query_pool):
                        pass # already visited or in query pool
                    elif tor_bond_mat[curr_node, nb_node] == 1: # find a torsional bond
                        tor_bonds_anno.append([curr_order, nb_node, curr_node])
                        next_query.append(nb_node)
                        if len(tor_bonds_anno) == n_tor_bonds:
                            early_stop = True
                            break
                    else:  # continue to search
                        query_pool.append(nb_node)
                if early_stop:
                    break
            if early_stop:
                break
            curr_order += 1
            query_pool = next_query
            
        if len(tor_bonds_anno) == 0:
            # no tor bond
            data.update({
                'tor_bonds_anno': torch.empty([0, 3], dtype=torch.long),
                'twisted_nodes_anno': torch.empty([0, 2], dtype=torch.long),
                'dihedral_pairs_anno': torch.empty([0, 3], dtype=torch.long),
            })
            return data
        # # twisted nodes
        tor_twisted_pairs = data['tor_twisted_pairs']
        twisted_nodes_anno = []
        for index_tor, (_, tor_left, tor_right) in enumerate(tor_bonds_anno):
            if tor_left < tor_right:
                all_nodes_left = tor_twisted_pairs[(tor_left, tor_right)][0]
            else:
                all_nodes_left = tor_twisted_pairs[(tor_right, tor_left)][1]
            
            # select from the same domain
            domain_tor = domain_index_of_node[tor_left]
            assert (domain_index_of_node[tor_right] == domain_tor), 'tor_bond node should be in the same domain'
            all_nodes_left = np.array(list(all_nodes_left))
            all_nodes_left = all_nodes_left[domain_index_of_node[all_nodes_left] == domain_tor]
            
            twisted_nodes_anno.extend([index_tor, node] for node in all_nodes_left)
        twisted_nodes_anno = torch.tensor(twisted_nodes_anno, dtype=torch.long)  # (n_twisted, 2)

        # # dihedral pairs
        dihedral_pairs_anno = []
        for index_tor, (_, tor_left, tor_right) in enumerate(tor_bonds_anno):
            domain_tor = domain_index_of_node[tor_left]
            nbh_left = [n for n in nbh_dict[tor_left] if (n != tor_right) and (domain_index_of_node[n] == domain_tor)]
            nbh_right = [n for n in nbh_dict[tor_right] if (n != tor_left) and (domain_index_of_node[n] == domain_tor)]
            dihedral_pairs_anno.extend([index_tor, node_left, node_right]
                    for node_left, node_right in product(nbh_left, nbh_right))
        dihedral_pairs_anno = torch.tensor(dihedral_pairs_anno, dtype=torch.long)  # (n_dihedral, 3)

        tor_bonds_anno = torch.tensor(tor_bonds_anno, dtype=torch.long)

        data.update({
            'tor_bonds_anno': tor_bonds_anno,
            'twisted_nodes_anno': twisted_nodes_anno,
            'dihedral_pairs_anno': dihedral_pairs_anno,
        })
        return data

    def graph_to_tree_order(self, nbh_list, size_tree):
        n = len(nbh_list)
        tree_order = []
        pool = [random.randint(0, n-1)] # 0, 1, ..., n-1
        while len(tree_order) < size_tree:  # only explore size_tree nodes
            #print('in loop 924')
            curr = np.random.choice(pool)
            tree_order.append(curr)
            pool.remove(curr)
            pool.extend([nb for nb in nbh_list[curr] if
                         (nb not in tree_order) and (nb not in pool)])
        assert len(tree_order) == size_tree, 'not enough nodes in the tree order'
        assert len(set(tree_order)) == size_tree, 'tree order has duplicate nodes'
        return np.array(tree_order)
    
    def partition_from_preset(self, data: Mol3DData):
        preset_partition = self.preset_partition.copy()
        n_nodes = data['node_type'].shape[0]
        
        # re-index if some nodes are removed
        if 'removed_index' in data:
            removed_index = data['removed_index']
            is_removed_node = np.zeros([n_nodes + len(removed_index)], dtype=bool)
            is_removed_node[removed_index] = True
            index_changes = np.cumsum(is_removed_node)
            if 'grouped_node_p1' in preset_partition:
                new_values = []
                for group in preset_partition['grouped_node_p1']:
                    new_group = [n - index_changes[n] for n in group if n not in removed_index]
                    new_values.append(new_group)
                preset_partition['grouped_node_p1'] = new_values
            if 'node_p2' in preset_partition:
                preset_partition['node_p2'] = [n - index_changes[n] for n in preset_partition['node_p2']
                                                if n not in removed_index]
            if 'grouped_anchor_p1' in preset_partition:
                new_values = []
                for group in preset_partition['grouped_anchor_p1']:
                    new_group = [n - index_changes[n] for n in group if n not in removed_index]
                    new_values.append(new_group)
                preset_partition['grouped_anchor_p1'] = new_values
        
        # prepare partition
        index_nodes = np.arange(n_nodes)
        grouped_node_p1 = preset_partition.get('grouped_node_p1', None)
        node_p2 = preset_partition.get('node_p2', None)
        assert (grouped_node_p1 is not None) or (node_p2 is not None), 'grouped_node_p1 or node_p2 should be set'
        
        if grouped_node_p1 is None:
            assert node_p2 is not None, 'Neither grouped_node_p1 nor node_p2 is set'
            node_p1 = [n for n in index_nodes if n not in node_p2]
            grouped_node_p1 = [node_p1]
        if node_p2 is None:
            assert grouped_node_p1 is not None, 'Neither grouped_node_p1 nor node_p2 is set'
            node_p1 = sum(grouped_node_p1, [])
            node_p2 = [n for n in index_nodes if n not in node_p1]
        #TODO: print partition
        grouped_anchor_p1 = preset_partition.get('grouped_anchor_p1', None)
        if grouped_anchor_p1 is None:
            grouped_anchor_p1 = [[] for _ in grouped_node_p1]
            
        data.update({
            'grouped_node_p1': grouped_node_p1,
            'grouped_anchor_p1': grouped_anchor_p1,
            'node_p2': node_p2,
        })
        return data
        
        
    def pre_make_partition(self, data: Mol3DData):
        if self.preset_partition is not None:  # set from config file. for use
            return self.partition_from_preset(data)

        n_nodes = data['node_type'].shape[0]
        setting = data['task_setting']
        setting_decom = setting['decomposition']
        setting_order = setting['order']
        
        # # subgraph neighborhood
        if setting_decom in ['brics', 'mmpa']:
            decom_dict = data[setting_decom]
            nbh_subgraphs = decom_dict['nbh_subgraphs']
        else:  # atoms
            nbh_subgraphs = data['nbh_dict']  # fake. an atom is a subgraph
            assert n_nodes == len(nbh_subgraphs), 'atom nbh info is not complete'
        n_subgraphs = len(nbh_subgraphs)
        
        # # determine subgraphs of p1 and p2
        if n_subgraphs <= 1:
            n_p1 = 0
        else:
            n_p1 = np.random.randint(1, n_subgraphs)  # [0, 1, ..., n_subgraphs-1]
        n_p2 = n_subgraphs - n_p1
        assert n_p2 > 0, 'n_p2 should be positive'
        
        # # make orders
        if setting_order != 'random':  # tree or inv_tree
            if setting_order == 'tree':
                subgraph_p1 = self.graph_to_tree_order(nbh_subgraphs, size_tree=n_p1)
            elif  setting_order == 'inv_tree':
                subgraph_p2 = self.graph_to_tree_order(nbh_subgraphs, size_tree=n_p2)
                subgraph_p1 = np.array([i for i in range(n_subgraphs) if i not in subgraph_p2])
        else:  # random
            subgraph_p1 = np.random.permutation(n_subgraphs)[:n_p1]

        # # combine subgraphs in p1 and p2 separately
        subgraph_part = subgraph_p1.tolist()
        domain_list = []
        while len(subgraph_part) > 0:
            #print('in loop 967')
            root = subgraph_part.pop(0)
            to_explore = [root]
            curr_domain = [root]
            while len(to_explore) > 0:
                #print('in loop 972')
                curr_sb = to_explore.pop(0)
                nbh_sbs = nbh_subgraphs[curr_sb]
                for nbh in nbh_sbs:
                    #print('in loop 981')
                    if (nbh in subgraph_part) and (nbh not in curr_domain) and (nbh not in to_explore):
                        curr_domain.append(nbh)
                        subgraph_part.remove(nbh)
                        if len(nbh_subgraphs[nbh]) > 1:  # not a leaf
                            to_explore.append(nbh)
            domain_list.append(curr_domain)
        
        # for curr_subgraph in subgraph_part:
        #     in_domain = [(curr_subgraph in domain) for domain in domain_list]
        #     if any(in_domain):
        #         domain = domain_list[in_domain.index(True)]
        #     else:
        #         domain = []
        #         domain_list.append(domain)
        #     domain.append(curr_subgraph)
        #     # nbh_curr = nbh_subgraphs[curr_subgraph]
        #     # domain.extend([nb for nb in nbh_curr if
        #     #         (nb not in domain) and (nb in subgraph_part)])
        # subgraph to nodes
        if setting_decom != 'atom':
            subgraphs_to_nodes = decom_dict['subgraphs']
            domain_list = [sum([subgraphs_to_nodes[sb] for sb in subgraphs], [])
                                for subgraphs in domain_list]
        grouped_node_p1 = domain_list
        node_p1 = sum(grouped_node_p1, [])
        node_p2 = [n for n in range(n_nodes) if n not in node_p1]
        assert len(node_p1) + len(node_p2) == n_nodes, 'nodes are not partitioned'
        
        # # get anchors
        in_p1 = np.zeros(n_nodes, dtype=bool)
        in_p1[node_p1] = True
        bond_index = data['bond_index']
        inter_domain = (in_p1[bond_index[0]] != in_p1[bond_index[1]])
        anchors = set(bond_index[:, inter_domain].numpy().flatten())
        grouped_anchor_p1 = [list(anchors & set(grouped_node_p1[i])) for i in range(len(grouped_node_p1))]
        
        data.update({
            'grouped_node_p1': grouped_node_p1,
            'grouped_anchor_p1': grouped_anchor_p1,
            'node_p2': node_p2,
        })
        return data


    def make_partition_features(self, data: Mol3DData):
        """
        Get the partition for fragment_linking task.
        Output is a dict containing:
        - node_p1
        - node_p2
        - halfedge_p1
        - halfedge_p2
        - halfedge_p1p2
        - grouped_halfedge_nonanchors_in_p1p2:
                List: the index of halfedge related to non-anchor nodes of p1 in halfedge_p1p2
                }
        """
        # # sample a separation from all possible separations of frag-linker
        seperation = data  # ['linking']
        
        # # get frags (p1) and linkers (p2)
        grouped_node_p1 = seperation['grouped_node_p1']
        grouped_anchor_p1 = seperation['grouped_anchor_p1']
        node_p1 = sum(grouped_node_p1, [])
        node_p2 = seperation['node_p2']
        # anchor_p2 = data['anchors_linkers'][idx_sep]
        n_frags = len(grouped_node_p1)
        
        assert len(node_p1) == len(np.unique(node_p1)), 'node_p1 has duplicate nodes'
        assert len(node_p2) == len(np.unique(node_p2)), 'node_p1 has duplicate nodes'
        assert len(set(node_p1) & set(node_p2)) == 0, 'node_p1 and node_p2 have common nodes'
        assert len(set(node_p1) | set(node_p2)) == data['node_type'].shape[0], 'node_p1 and node_p2 are not partitioned'
        
        node_p1 = torch.LongTensor(node_p1)
        groupof_node_p1 = torch.LongTensor([i for i, frag in enumerate(grouped_node_p1) for _ in frag])
        node_p2 = torch.LongTensor(node_p2)
        halfedge_index = data.halfedge_index
        if halfedge_index.shape[1] != 0:
            i_all_halfedge = torch.arange(halfedge_index.shape[1], dtype=torch.long)
            halfedge_p1 = subgraph(node_p1, halfedge_index, i_all_halfedge)[1]
            halfedge_p2 = subgraph(node_p2, halfedge_index, i_all_halfedge)[1]
        
            i_all_halfedge[halfedge_p1] = -1
            i_all_halfedge[halfedge_p2] = -1
            halfedge_p1p2 = torch.nonzero(i_all_halfedge >= 0, as_tuple=False).squeeze()
            # another way
            edge_index, i_all_edge = to_undirected(halfedge_index, i_all_halfedge)
            halfedge_p1p2_way2 = bipartite_subgraph([node_p1, node_p2], edge_index, i_all_edge)[1]
            assert (halfedge_p1p2.sort()[0] == halfedge_p1p2_way2.sort()[0]).all(), 'two ways of getting halfedge_p1p2 are different'
        else:
            halfedge_p1 = torch.empty([0], dtype=torch.long)
            halfedge_p2 = torch.empty([0], dtype=torch.long)
            halfedge_p1p2 = torch.empty([0], dtype=torch.long)
        
        # # get halfedge_nonanchors_in_p1p2_grouped
        setting = data['task_setting']
        known_anchor = setting['known_anchor']
        if known_anchor == 'none':  # all edges between p1 and p2 are unknown
            is_known_halfedge_p1p2 = torch.zeros(halfedge_p1p2.shape[0], dtype=torch.bool)
        elif known_anchor == 'partial':
            if n_frags <= 1:
                n_known_frag = 0
            else:
                n_known_frag = np.random.randint(1, n_frags)
            node_known_outer_edge = []
            for i_known_frag in np.random.choice(n_frags, n_known_frag, replace=False):
                node_this_frag = grouped_node_p1[i_known_frag]
                anchor_this_frag = grouped_anchor_p1[i_known_frag]
                node_known_outer_edge.extend([n for n in node_this_frag if n not in anchor_this_frag])
            node_known_outer_edge = torch.tensor(node_known_outer_edge, dtype=torch.long)
            halfedge_index_p1p2 = halfedge_index[:, halfedge_p1p2]
            is_known_halfedge_p1p2 = (halfedge_index_p1p2[..., None] == node_known_outer_edge).any(-1).any(0)
        elif known_anchor == 'all':
            # is_known_halfedge_p1p2 = torch.ones(halfedge_p1p2.shape[0], dtype=torch.bool)
            node_anchor = torch.tensor(sum(grouped_anchor_p1, []))
            halfedge_index_p1p2 = halfedge_index[:, halfedge_p1p2]
            is_known_halfedge_p1p2 = (halfedge_index_p1p2[..., None] != node_anchor).all(-1).all(0)
        else:
            raise ValueError(f'unknown known_anchor value: {known_anchor}')
        
        partition = {
            'node_p1': node_p1,
            'node_p2': node_p2,
            'halfedge_p1': halfedge_p1,
            'halfedge_p2': halfedge_p2,
            'halfedge_p1p2': halfedge_p1p2,
            'groupof_node_p1': groupof_node_p1,
            'is_known_halfedge_p1p2': is_known_halfedge_p1p2,
        }
        
        # # summary partition
        num_part_dict = {f'n_{key}': partition[key].shape[-1] for key in 
            ['node_p1', 'node_p2', 'halfedge_p1', 'halfedge_p2', 'halfedge_p1p2']}
        # santiy check
        assert num_part_dict['n_node_p1'] + num_part_dict['n_node_p2'] == data.node_type.shape[0]
        assert (num_part_dict['n_halfedge_p1'] + num_part_dict['n_halfedge_p2']
                + num_part_dict['n_halfedge_p1p2'] == data.halfedge_type.shape[0])

        data.update(partition)
        return data, num_part_dict


@register_transforms('pepdesign')
class PepdesignTransform:
    def __init__(self, config, **kwargs) -> None:
        self.config = config
        self.mode = mode = kwargs.get('mode', 'test')
        self.exclude_keys = ['peptide_pos', 'peptide_atom_name', 'peptide_is_backbone',
                            'peptide_res_id', 'peptide_atom_to_aa_type', 'peptide_res_index',
                            'peptide_seq', 'peptide_pep_len']
        if mode == 'train':
            self.exclude_keys.extend([
                'task_setting',
                'node_bb', 'node_sc', 'halfedge_bb', 'halfedge_sc', 'halfedge_bbsc', 'is_known_halfedge_bbsc'
            ])
            
        # variable size during training
        self.add_mask_atoms = config.get('add_mask_atoms', False)
        self.num_node_types = kwargs.get('num_node_types', None)
        
        self.settings_dict = {}
        if 'settings' in config:
            for key, value_dict in config.settings.items():
                self.settings_dict[key] = {'options': list(value_dict.keys()), 'weights': list(value_dict.values())}
                assert all(op in PEPDESIGN_SETTINGS[key] for op in self.settings_dict[key]['options']),\
                        f"unknown maskfill setting {self.settings_dict[key]} for setting {key}"

        self.fix_pos = config.get('fix_pos', None)
        self.fix_type_only = config.get('fix_type_only', None)
    
    def __call__(self, data: Mol3DData):
        setting = self.sample_setting()
        if data['db'] == 'pepbdb' and 'X' in data['peptide_seq'] and self.mode != 'test':
            setting['mode'] = 'packing'  # nonstd peptide not for training generation
        data.update({'task_setting': setting})
        # change the peptide indicator
        if setting['mode'] in ['full', 'sc']:
            data['is_peptide'] = torch.ones_like(data['is_peptide'])
        
            if self.add_mask_atoms and self.mode != 'test':
                data = self.add_atoms(data)
        
        # make partiotion
        data = self.pre_make_partition(data)
        data, num_part_dict = self.make_partition_features(data)
        # set features
        data = self.set_fixed(data, setting, num_part_dict)
        data = self.set_torsional_feat(data)

        if self.mode == 'test':
            data = self.prepare_sample(data, setting)
        return data

    def prepare_sample(self, data, setting):
        # # make gt: actually is not used, for pepdesign, directly copy the gt file. just for compatibility
        data['gt_node_type'] = data['node_type'].clone()  
        data['gt_node_pos'] = data['node_pos'].clone()
        data['gt_halfedge_type'] = data['halfedge_type'].clone()
        
        # # make gt for parts
        node_type = data['node_type']
        node_pos = data['node_pos']
        halfege_type = data['halfedge_type']
        
        for i_part in ['bb', 'sc']:
            node_type_part = - torch.ones_like(node_type)
            node_type_part[data[f'node_{i_part}']] = node_type[data[f'node_{i_part}']].clone()
            node_pos_part = node_pos.clone()
            halfedge_type_part = - torch.ones_like(halfege_type)
            halfedge_type_part[data[f'halfedge_{i_part}']] = halfege_type[data[f'halfedge_{i_part}']].clone()
            data.update({
                f'gt_node_type_{i_part}': node_type_part,
                f'gt_node_pos_{i_part}': node_pos_part,
                f'gt_halfedge_type_{i_part}': halfedge_type_part,
            })
        
        # # make init
        data['node_pos'][data['node_sc']] = 0
        if setting['mode'] == 'packing':
            pass
        elif setting['mode'] in ['sc', 'full']:
            data['node_type'][data['node_sc']] = 0
            data['halfedge_type'][data['halfedge_sc']] = 0
            data['halfedge_type'][data['halfedge_bbsc']] = 0
            if setting['mode'] == 'full':
                data['node_pos'][data['node_bb']] = 0
            
        return data

    def sample_setting(self):
        setting_dict = {}
        for setting, opt_dict in self.settings_dict.items():
            setting_dict[setting] = np.random.choice(opt_dict['options'], p=opt_dict['weights'])
        return setting_dict
    
    def set_fixed(self, data: Mol3DData, setting, num_part_dict):
        n_node_bb, n_node_sc = num_part_dict['n_node_bb'], num_part_dict['n_node_sc']
        n_halfedge_bb, n_halfedge_sc = num_part_dict['n_halfedge_bb'], num_part_dict['n_halfedge_sc']
        n_halfedge_bbsc = num_part_dict['n_halfedge_bbsc']
        
        # # bb graph is always fixed, default all others are not fixed
        fixed_node_bb, fixed_pos_bb, fixed_halfedge_bb = get_vector_list(
            [n_node_bb, n_node_bb, n_halfedge_bb], [1, 0, 1])
        fixed_node_sc, fixed_pos_sc, fixed_halfedge_sc, fixed_halfedge_bbsc = get_vector_list(
            [n_node_sc, n_node_sc, n_halfedge_sc, n_halfedge_bbsc], [0, 0, 0, 0])
        # for connections, CA as anchors, other bb atoms' edges with sc are known
        is_known_halfedge_bbsc = data['is_known_halfedge_bbsc']
        fixed_halfedge_bbsc[is_known_halfedge_bbsc] = 1

        # # mode-specific
        if setting['mode'] == 'full':
            pass
        else:
            fixed_pos_bb = get_vector([n_node_bb], 1)  # bb pos is fixed
            if setting['mode'] == 'sc':  
                pass
            elif setting['mode'] == 'packing':  # sc graph are fixed
                fixed_node_sc, fixed_halfedge_sc, fixed_halfedge_bbsc = get_vector_list(
                    [n_node_sc, n_halfedge_sc, n_halfedge_bbsc], [1, 1, 1])
            else:
                raise ValueError(f"Unknown mode: {setting['mode']}")

        fixed_dict = {
            'node_bb': fixed_node_bb,
            'pos_bb': fixed_pos_bb,
            'halfedge_bb': fixed_halfedge_bb,
            'node_sc': fixed_node_sc,
            'pos_sc': fixed_pos_sc,
            'halfedge_sc': fixed_halfedge_sc,
            'halfedge_bbsc': fixed_halfedge_bbsc,
        }
        
        # # combine p1 and p2
        fixed_node = combine_vectors_indexed(
            [fixed_dict[f'node_bb'], fixed_dict[f'node_sc']],
            [data['node_bb'], data['node_sc']],
        )
        fixed_pos = combine_vectors_indexed(
            [fixed_dict[f'pos_bb'], fixed_dict[f'pos_sc']],
            [data['node_bb'], data['node_sc']],
        )
        fixed_halfedge = combine_vectors_indexed(
            [fixed_dict[f'halfedge_bb'], fixed_dict[f'halfedge_sc'], fixed_dict[f'halfedge_bbsc']],
            [data['halfedge_bb'], data['halfedge_sc'], data['halfedge_bbsc']],
        )
        
        self.add_simple_fix_modify(data, fixed_node, fixed_pos, fixed_halfedge)  # for easy use. not important for training/sampling
        
        data.update({
            'fixed_node': fixed_node,
            'fixed_pos': fixed_pos,
            'fixed_halfedge': fixed_halfedge,
        })
        
        # # for fixed_dist: always free pos
        domain_node_index = torch.empty([2, 0], dtype=torch.long)  # set rigid domain to empty
        n_domain = torch.tensor(0, dtype=torch.long)
        fixed_halfdist = torch.zeros_like(data['halfedge_type'], dtype=torch.long)  # default not fixed distances
        if setting['mode'] in ['sc', 'packing']:  # fixed distances of bb
            fixed_halfdist[data['halfedge_bb']] = 1
            
        self.add_simple_fix_dist_modify(data, fixed_halfdist, fixed_pos)  # for easy use. not important for training/sampling
        
        data.update({
            'fixed_halfdist': fixed_halfdist,
            'n_domain': n_domain,
            'domain_node_index': domain_node_index,
        })
        return data
    
    def add_simple_fix_modify(self, data, fixed_node, fixed_pos, fixed_halfedge):
        if (self.fix_pos is None) and (self.fix_type_only is None):
            return  # no modification needed
        
        # re-index if some nodes are removed
        n_nodes = data['node_type'].shape[0] # N
        if 'removed_index' in data:
            removed_index = data['removed_index'] # M removed atoms
            is_removed_node = np.zeros([n_nodes + len(removed_index)], dtype=bool)  # N+M
            is_removed_node[removed_index] = True
            index_changes = np.cumsum(is_removed_node)  # N+M
            def index_mapper(orig_indices):
                return [n - index_changes[n] for n in orig_indices if n not in removed_index]
            peptide_res_index = np.array(data['peptide_res_index'], dtype=np.int32)  # N+M
            peptide_res_index = peptide_res_index[~is_removed_node]  # res_index of remaining atoms; N
        else:  # add
            def index_mapper(orig_indices):
                return orig_indices
            peptide_res_index = np.concatenate([
                np.array(data['peptide_res_index'], dtype=np.int32),
                np.ones(n_nodes-len(data['peptide_res_index']), dtype=np.int32)*(-10000)
            ])
            
        def get_new_atom_indices(fix_dict):
            fixed_atom_indices = np.array(fix_dict.get('atom', []))
            fixed_atom_indices = index_mapper(fixed_atom_indices)
            res_bb = fix_dict.get('res_bb', [])
            res_sc = fix_dict.get('res_sc', [])
            
            fixed_unconnect_indices = np.array([], dtype=np.int64)
            if res_bb or res_sc:
                is_backbone = np.array(data['peptide_is_backbone'], dtype=bool)  # had already been updated in VariableSC transform
                
                is_sel_res_bb = ((peptide_res_index[:, None] == np.array(res_bb)[None]).any(-1)
                                    & is_backbone)
                is_sel_res_sc = ((peptide_res_index[:, None] == np.array(res_sc)[None]).any(-1)
                                    & (~is_backbone))
                add_atoms_indices = np.nonzero(is_sel_res_bb | is_sel_res_sc)[0]
                fixed_atom_indices = np.concatenate([fixed_atom_indices, add_atoms_indices])
                
                # These cannot connect to newly generated atoms: all of res_sc and CA/N of res_bb (C/O of res_bb can connect to new atoms by default, so no more fix needed)
                # therefore the edge_type between these atoms and new atoms should be fixed
                peptide_atom_name = np.array(data['peptide_atom_name'])
                is_sel_res_bb_ca_or_n = ((peptide_atom_name == 'CA') | (peptide_atom_name == 'N')) & is_sel_res_bb
                fixed_unconnect_indices = np.nonzero(is_sel_res_sc | is_sel_res_bb_ca_or_n)[0]
            return np.unique(fixed_atom_indices), np.unique(fixed_unconnect_indices)
        
        fixed_unconnect_indices = np.array([], dtype=np.int64)
        if self.fix_pos is not None:
            assert isinstance(self.fix_pos, dict), 'fix_pos should be a dict'
            fixed_pos_indices, add_unconnect_indices = get_new_atom_indices(self.fix_pos.copy())
            fixed_unconnect_indices = np.concatenate([fixed_unconnect_indices, add_unconnect_indices])
        else:
            fixed_pos_indices = np.array([], dtype=np.int64)
        if self.fix_type_only is not None:
            assert isinstance(self.fix_type_only, dict), 'fix_type_only should be a dict'
            fixed_type_indices, add_unconnect_indices = get_new_atom_indices(self.fix_type_only.copy())
            fixed_unconnect_indices = np.concatenate([fixed_unconnect_indices, add_unconnect_indices])
        else:
            fixed_type_indices = np.array([], dtype=np.int64)
        fixed_type_indices = np.unique(np.concatenate([fixed_type_indices, fixed_pos_indices]))  # fix_pos must also fix type

        # pos
        if len(fixed_pos_indices) > 0:
            fixed_pos[fixed_pos_indices] = 1
        # node
        if len(fixed_type_indices) > 0:
            fixed_node[fixed_type_indices] = 1
        # fix inner halfedges
        if len(fixed_type_indices) > 1:
            fixed_node_indices = torch.tensor(fixed_type_indices, dtype=torch.long)
            halfedge_index = data.halfedge_index
            i_all_halfedge = torch.arange(halfedge_index.shape[1], dtype=torch.long)
            halfedge_inner_fixed_type = subgraph(fixed_node_indices, halfedge_index, i_all_halfedge)[1]
            fixed_halfedge[halfedge_inner_fixed_type] = 1
        if len(fixed_unconnect_indices) > 0:
            fixed_unconnect_indices = torch.tensor(fixed_unconnect_indices, dtype=torch.long)
            halfedge_index = data.halfedge_index  # (2, n_halfedge)
            is_related = (halfedge_index[..., None] == fixed_unconnect_indices).any(-1).any(0)
            fixed_halfedge[is_related] = 1
            
        return 
    
    def add_simple_fix_dist_modify(self, data, fixed_halfdist, fixed_pos):
        # fix halfdist if both nodes' positions are fixed
        n_nodes = fixed_pos.shape[0]
        halfedge_index = data.halfedge_index
        is_ends_fixed_pos = (fixed_pos[halfedge_index] == 1).all(0)  # (2, n_halfedge) -> (n_halfedge,)
        fixed_halfdist[is_ends_fixed_pos] = 1
        return
        
    


    def set_torsional_feat(self, data: Mol3DData):
        # no torsion or flex mode for pepdesign
        data.update({
            'tor_bonds_anno': torch.empty([0, 3], dtype=torch.long),
            'twisted_nodes_anno': torch.empty([0, 2], dtype=torch.long),
            'dihedral_pairs_anno': torch.empty([0, 3], dtype=torch.long),
        })
        return data


    def pre_make_partition(self, data: Mol3DData):
        
        node_bb = torch.nonzero(data['peptide_is_backbone'])[:, 0]
        node_sc = torch.nonzero(~data['peptide_is_backbone'])[:, 0]
        

        data.update({
            'node_bb': node_bb,
            'node_sc': node_sc,
        })
        return data


    def make_partition_features(self, data: Mol3DData):

        # # get backbone and sidechain
        node_bb = data['node_bb']
        node_sc = data['node_sc']
        
        assert len(node_bb) == len(np.unique(node_bb)), 'node_bb has duplicate nodes'
        assert len(node_sc) == len(np.unique(node_sc)), 'node_sc has duplicate nodes'
        assert len(set(node_bb) & set(node_sc)) == 0, 'node_bb and node_sc have common nodes'
        assert len(set(node_bb) | set(node_sc)) == data['node_type'].shape[0], 'node_bb and node_sc are not partitioned'
        
        # # make partition for halfedge
        halfedge_index = data.halfedge_index
        i_all_halfedge = torch.arange(halfedge_index.shape[1], dtype=torch.long)
        halfedge_bb = subgraph(node_bb, halfedge_index, i_all_halfedge)[1]
        halfedge_sc = subgraph(node_sc, halfedge_index, i_all_halfedge)[1]
        i_all_halfedge[halfedge_bb] = -1
        i_all_halfedge[halfedge_sc] = -1
        halfedge_bbsc = torch.nonzero(i_all_halfedge >= 0, as_tuple=False).squeeze()
        # anchor, i.e., CA atoms
        is_known_halfedge_bbsc = torch.zeros(halfedge_bbsc.shape[0], dtype=torch.bool)
        peptide_atom_name = np.array(data['peptide_atom_name'])
        n_or_ca_atoms = np.nonzero(
            (peptide_atom_name == 'CA') | (peptide_atom_name == 'N')
        )[0]  # proline also connects to N
        node_anchor = torch.tensor([n for n in n_or_ca_atoms if n in node_bb])
        halfedge_index_bbsc = halfedge_index[:, halfedge_bbsc]
        is_known_halfedge_bbsc = (halfedge_index_bbsc[..., None] != node_anchor).all(-1).all(0)
        
        partition = {
            'node_bb': node_bb,
            'node_sc': node_sc,
            'halfedge_bb': halfedge_bb,
            'halfedge_sc': halfedge_sc,
            'halfedge_bbsc': halfedge_bbsc,
            'is_known_halfedge_bbsc': is_known_halfedge_bbsc,
        }
        
        # # summary partition
        num_part_dict = {f'n_{key}': partition[key].shape[-1] for key in 
            ['node_bb', 'node_sc', 'halfedge_bb', 'halfedge_sc', 'halfedge_bbsc']}
        # santiy check
        assert num_part_dict['n_node_bb'] + num_part_dict['n_node_sc'] == data.node_type.shape[0]
        assert (num_part_dict['n_halfedge_bb'] + num_part_dict['n_halfedge_sc']
                + num_part_dict['n_halfedge_bbsc'] == data.halfedge_type.shape[0])

        data.update(partition)
        return data, num_part_dict
    
    def add_atoms(self, data):  # ask the model to predict mask-atom type
        # num_per_res = self.add_mask_atoms.num_per_res
        # n_res = data['peptide_res_index'].max().item() + 1
        is_backbone = data['peptide_is_backbone']
        is_sidechain = ~is_backbone
        n_sc = is_sidechain.sum().item()
        
        ratio = self.add_mask_atoms.ratio
        n_add_max = np.clip(int(n_sc * ratio), a_min=0, a_max=n_sc)
        n_add = np.clip(np.random.randint(-n_add_max, n_add_max + 1), a_min=0, a_max=n_add_max)
        if n_add == 0:
            return data
        
        # determine positions
        len_mu, len_sigma = self.add_mask_atoms.len_mu, self.add_mask_atoms.len_sigma
        node_sc = torch.nonzero(is_sidechain)[:, 0]
        node_sc_center = node_sc[torch.randint(n_sc, size=[n_add])]
        lengths = torch.randn([n_add]) * len_sigma + len_mu
        relative_pos = torch.randn([n_add, 3])
        relative_pos = relative_pos / (relative_pos.norm(dim=-1, keepdim=True)+1e-5) * lengths[:, None]
        node_pos = data['node_pos']
        node_pos_new = node_pos[node_sc_center] + relative_pos
        
        # new node
        node_pos = data['node_pos']
        node_type = data['node_type']
        n_atoms_data = node_type.shape[0]
        n_atoms_new = n_atoms_data + n_add
        new_node_type = torch.cat([node_type, (self.num_node_types-1) * torch.ones([n_add], dtype=node_type.dtype)], dim=0)
        new_node_pos = torch.cat([node_pos, node_pos_new], dim=0)
        
        # new edge
        halfedge_index = data['halfedge_index']
        halfedge_type = data['halfedge_type']
        n_add_halfedge = n_add * n_atoms_data + n_add * (n_add - 1) // 2
        new_halfedge_type = torch.cat([halfedge_type, torch.zeros([n_add_halfedge], dtype=halfedge_type.dtype)], dim=0)
        halfedge_index_old_new = torch.stack(
            torch.meshgrid(torch.arange(n_atoms_data), torch.arange(n_atoms_data, n_atoms_new), indexing='ij'),
        dim=0).reshape(2, -1)
        halfedge_index_new_new = torch.triu_indices(n_add, n_add, offset=1) + n_atoms_data
        new_halfedge_index = torch.cat([
            halfedge_index, halfedge_index_old_new, halfedge_index_new_new], dim=1)
        new_halfedge_index, new_halfedge_type = sort_edge_index(new_halfedge_index, new_halfedge_type)

        # peptide feature
        peptide_is_backbone = data['peptide_is_backbone']
        peptide_atom_name = data['peptide_atom_name']
        new_peptide_is_backbone = torch.cat([peptide_is_backbone, torch.zeros([n_add], dtype=peptide_is_backbone.dtype)], dim=0)
        new_peptide_atom_name = peptide_atom_name + ['X'] * n_add
        
        data.update({
            'num_nodes': n_atoms_new,
            'node_type': new_node_type,
            'node_pos': new_node_pos,
            'halfedge_index': new_halfedge_index,
            'halfedge_type': new_halfedge_type,
            'peptide_is_backbone': new_peptide_is_backbone,
            'peptide_atom_name': new_peptide_atom_name,
        })
        if 'is_peptide' in data:
            is_peptide = data['is_peptide']
            data['is_peptide'] = torch.cat([is_peptide, torch.ones([n_add], dtype=is_peptide.dtype)], dim=0)

        return data


    
@register_transforms('linking')
class LinkingTransform(MaskfillTransform):
    def pre_make_partition(self, data: Mol3DData):
        grouped_node_p1 = [list(item) for item in data['frags']]
        grouped_anchor_p1 = [list(item) for item in data['anchors']]
        node_p2 = sum([list(item) for item in data['linkers']], [])

        data.update({
            'grouped_node_p1': grouped_node_p1,
            'grouped_anchor_p1': grouped_anchor_p1,
            'node_p2': node_p2,
        })
        return data
    
    def prepare_sample(self, data, setting):
        # # move center of p1 (anchors) to origin
        if setting['known_anchor'] == 'all':
            anchors = [i for ans in data['anchors'] for i in ans]
            center = data['node_pos'][anchors].mean(dim=0, keepdims=True)
        else:
            center = data['node_pos'][data['node_p1']].mean(dim=0, keepdims=True)
        data['node_pos'] -= center
        data['pocket_pos'] -=center
        data['pocket_center'] += center
        
        # # make gt
        data['gt_node_type'] = data['node_type'].clone()
        data['gt_node_pos'] = data['node_pos'].clone()
        data['gt_halfedge_type'] = data['halfedge_type'].clone()
        
        # # make gt for part 1
        node_type = data['node_type']
        node_pos = data['node_pos']
        halfege_type = data['halfedge_type']
        
        for i_part in [1, 2]:
            node_type_part = - torch.ones_like(node_type)
            node_type_part[data[f'node_p{i_part}']] = node_type[data[f'node_p{i_part}']].clone()
            node_pos_part = node_pos.clone()
            halfedge_type_part = - torch.ones_like(halfege_type)
            halfedge_type_part[data[f'halfedge_p{i_part}']] = halfege_type[data[f'halfedge_p{i_part}']].clone()
            data.update({
                f'gt_node_type_p{i_part}': node_type_part,
                f'gt_node_pos_p{i_part}': node_pos_part,
                f'gt_halfedge_type_p{i_part}': halfedge_type_part,
            })
        
        # # not explicitly remove p2 info. controlled by info_level
            
        return data


@register_transforms('growing')
class GrowingTransform(MaskfillTransform):
    def pre_make_partition(self, data: Mol3DData):
        setting = data['task_setting']
        assert setting['known_anchor'] == 'none', 'Only none known_anchor is supported for frag growing yet.'
        if 'init_frag' in data:
            grouped_node_p1 = [data['init_frag']]
        else:
            grouped_node_p1 = [self.preset_partition['init_frag']]
        grouped_anchor_p1 = [[]]
        if 'add_frag' in data:
            node_p2 = data['add_frag']
        else:  # exclude init_frag
            node_p2 = [n for n in range(data['num_nodes']) if n not in grouped_node_p1[0]]

        data.update({
            'grouped_node_p1': grouped_node_p1,
            'grouped_anchor_p1': grouped_anchor_p1,
            'node_p2': node_p2,
        })
        return data
    



@register_transforms('ar')
class AutoregressiveTransform(MaskfillTransform):
    def __init__(self, config, **kwargs):
        mode = kwargs.get('mode', 'test')
        # assert mode == 'test', 'ar transform only for test (sample). For training, use maskfill transform'
        assert mode != 'train', 'ar transform only for test (sample). For training, use maskfill transform'
        super().__init__(config, **kwargs)

    def sample_setting(self):
        part1_pert = self.config.get('part1_pert', 'small')
        # setting_dict = {'part1_pert': 'small', 'known_anchor': 'none'}
        # setting_dict = {'part1_pert': 'fixed', 'known_anchor': 'none'}
        setting_dict = {'part1_pert': part1_pert, 'known_anchor': 'none'}
        return setting_dict  # to be compatible with maskfill transform

    def pre_make_partition(self, data: Mol3DData):
        # # save gt in advance
        data['gt_node_type'] = data['node_type'].clone()
        data['gt_node_pos'] = data['node_pos'].clone()
        data['gt_halfedge_type'] = data['halfedge_type'].clone()
        data['gt_halfedge_index'] = data['halfedge_index'].clone()
        
        # as the initial state of the ar sampling
        data.update({
            'grouped_node_p1': [],
            'grouped_anchor_p1': [],
            'node_p2': list(range(data['num_nodes'])),
        })
        # if data['pocket_knn_edge_index'].shape[1] > 0:
        #     raise NotImplementedError
        return data
    
    # def prepare_sample(self, data, setting):
    #     return data


def get_rigid_distmat(domain_index, node_index, n_domain, n_total_node):
    fixed_distmat = torch.zeros((n_total_node, n_total_node), dtype=torch.long)
    if n_domain == 0:
        return fixed_distmat
    else:
        for i_domain in range(n_domain):
            node_this_domain = node_index[domain_index == i_domain]
            fixed_distmat[np.ix_(node_this_domain, node_this_domain)] = 1
        return fixed_distmat


@register_transforms('ar2')
class Autoregressive2Transform(MaskfillTransform):
    def __init__(self, config, **kwargs):
        mode = kwargs.get('mode', 'test')
        assert mode == 'test', 'ar2 transform only for test (sample). For training, use maskfill transform'
        super().__init__(config, **kwargs)

    def sample_setting(self):
        setting_dict = {'part1_pert': 'small', 'known_anchor': 'none'}
        return setting_dict  # to be compatible with maskfill transform

    def pre_make_partition(self, data: Mol3DData):
        # # save gt in advance
        data['gt_node_type'] = data['node_type'].clone()
        data['gt_node_pos'] = data['node_pos'].clone()
        data['gt_halfedge_type'] = data['halfedge_type'].clone()
        data['gt_halfedge_index'] = data['halfedge_index'].clone()
        
        # as the initial state of the ar sampling
        n_init = 6
        data.update({
            'grouped_node_p1': [],
            'grouped_anchor_p1': [],
            'node_p2': list(range(n_init)),
        })
        # # reset mol nodes, as in ar2, only n node in initial state
        node_type = data['node_type'][:n_init]
        node_pos = data['node_pos'][:n_init]
        is_peptide = data['is_peptide'][:n_init]
        halfedge_type = data['halfedge_type']
        halfedge_index = data['halfedge_index']
        halfedge_index, halfedge_type = subgraph(
            torch.arange(n_init), halfedge_index, halfedge_type)
        data.update({
            'node_type': node_type,
            'node_pos': node_pos,
            'halfedge_type': halfedge_type,
            'halfedge_index': halfedge_index,
            'is_peptide': is_peptide,
        })
        
        return data
    
    def prepare_sample(self, data, setting):
        return data
    


def get_rigid_distmat(domain_index, node_index, n_domain, n_total_node):
    fixed_distmat = torch.zeros((n_total_node, n_total_node), dtype=torch.long)
    if n_domain == 0:
        return fixed_distmat
    else:
        for i_domain in range(n_domain):
            node_this_domain = node_index[domain_index == i_domain]
            fixed_distmat[np.ix_(node_this_domain, node_this_domain)] = 1
        return fixed_distmat



@register_transforms('custom')
class CustomTransform:
    def __init__(self, config, **kwargs) -> None:
        self.config = config
        self.mode = mode = kwargs.get('mode', 'test')
        
        self.is_peptide = config['is_peptide']
        self.partition = config['partition']
        self.partition_names = [p['name'] for p in self.partition]
        self.fixed = config['fixed']
        
        if 'sc' in self.partition_names:  # 1) last partition; 2) must be others (to auto handle new atoms)
            assert self.partition[-1] == {'name': 'sc', 'nodes': 'others'}, 'sc part is not valid'

    def __call__(self, data: Mol3DData):
        self._check_name(data)
        
        # make partiotion
        data, _ = self.make_partition_features(data)
        
        # set features
        data = self.set_fixed(data)
        data = self.set_torsional_feat(data)
        
        # is_peptide
        is_peptide = self.is_peptide
        data['is_peptide'] = is_peptide * torch.ones_like(data['is_peptide'])

        return data

    def _check_name(self, data):
        for key in self.partition_names:
            assert f'node_part_{key}' not in data, f'find exist key node_part_{key}'
            for key2 in self.partition_names:
                assert f'halfedge_{key}_{key2}' not in data, f'find exist key halfedge_{key}_{key2}'

    def set_fixed(self, data: Mol3DData):
        n_nodes = data['node_type'].shape[0]
        n_halfedges = data['halfedge_type'].shape[0]

        # # set fixed node
        fixed_node = get_vector(n_nodes, 0)  # default not fixed
        for part in self.fixed.get('node', []):
            node_part = data[f'node_part_{part}']
            fixed_node[node_part] = 1
        
        # # set fixed pos
        fixed_pos = get_vector(n_nodes, 0)  # default not fixed
        for part in self.fixed.get('pos', []):
            node_part = data[f'node_part_{part}']
            fixed_pos[node_part] = 1
            
        # # set fixed halfedge
        fixed_halfedge = get_vector(n_halfedges, 0)  # default not fixed
        for parts in self.fixed.get('edge', []):
            halfedge_part = data[f'halfedge_part_{parts[0]}_{parts[1]}']
            fixed_halfedge[halfedge_part] = 1

        data.update({
            'fixed_node': fixed_node,
            'fixed_pos': fixed_pos,
            'fixed_halfedge': fixed_halfedge,
        })
        
        # # for fixed_dist 
        fixed_halfdist = get_vector(n_halfedges, 0)  # default not fixed distances
        # fixed if both ends with fixed_pos == 1
        for part in self.fixed.get('pos', []):
            halfedge_part = data[f'halfedge_part_{part}_{part}']
            fixed_halfdist[halfedge_part] = 1
        
        domain_node_index = torch.empty([2, 0], dtype=torch.long)  # set rigid domain to empty
        n_domain = torch.tensor(0, dtype=torch.long)
        data.update({
            'fixed_halfdist': fixed_halfdist,
            'n_domain': n_domain,
            'domain_node_index': domain_node_index,
        })

        return data
    
    def set_torsional_feat(self, data: Mol3DData):
        data.update({
            'tor_bonds_anno': torch.empty([0, 3], dtype=torch.long),
            'twisted_nodes_anno': torch.empty([0, 2], dtype=torch.long),
            'dihedral_pairs_anno': torch.empty([0, 3], dtype=torch.long),
        })
        return data

    def make_partition_features(self, data: Mol3DData):
        
        num_nodes = data['node_type'].shape[0]
        data_partition = {}
        
        # # make node partition
        # re-index due to the variable pep size
        if 'removed_index' in data:
            removed_index = data['removed_index']
            # is_removed_node = torch.zeros([num_nodes + len(removed_index)], dtype=torch.bool)
            # is_removed_node[removed_index] = True
            # index_changes = torch.cumsum(is_removed_node, dim=0)
            is_removed_node = np.zeros([num_nodes + len(removed_index)], dtype=bool)
            is_removed_node[removed_index] = True
            index_changes = np.cumsum(is_removed_node)
        else:
            # index_changes = torch.zeros([num_nodes], dtype=torch.long)
            index_changes = np.zeros([num_nodes], dtype=np.int64)
            removed_index = []
        if 'added_index' in data:
            pass # nothing to do because nodes of `sc` must be `others`, so automatically added the new atoms
        
        all_atoms = set()
        for i_order, this_part in enumerate(self.partition):
            key = this_part['name']
            value = this_part['nodes']
            if isinstance(value, list):
                value = [atom for atom in value if atom not in removed_index]
                # re-index 
                value = np.array(value) - index_changes[value]
                value_set = set(value)
                assert len(all_atoms.intersection(value_set)) == 0, 'parition has overlap'
            elif isinstance(value, str):
                assert value == 'others', f'invalid value of partition: {value}'
                assert i_order == len(self.partition) - 1, 'others should be the last partition' 
                value = np.array([atom for atom in  np.arange(num_nodes) if atom not in all_atoms])
                value_set = set(value)
            else:
                raise ValueError(f'invalid value of partition: {value}')
            data_partition.update({
                'node_part_' + key: torch.LongTensor(value)})
            all_atoms.update(value_set)
        assert len(all_atoms) == num_nodes, 'partition does not cover all atoms'
        
        # # make halfedge partition
        halfedge_index = data['halfedge_index']
        i_all_halfedge = torch.arange(halfedge_index.shape[1], dtype=torch.long)
        edge_index, i_all_edge = to_undirected(halfedge_index, i_all_halfedge)

        num_partitions = len(self.partition)
        for i_part in range(num_partitions):
            name_pi = self.partition_names[i_part]
            node_pi = data_partition[f'node_part_{name_pi}']
            for j_part in range(i_part, num_partitions):
                name_pj = self.partition_names[j_part]
                node_pj = data_partition[f'node_part_{name_pj}']
                if i_part == j_part:
                    halfedge_pij = subgraph(node_pi, halfedge_index, i_all_halfedge)[1]
                else:
                    halfedge_pij = bipartite_subgraph([node_pi, node_pj], edge_index, i_all_edge)[1]
                name = 'halfedge_part_' + name_pi + '_' + name_pj
                data_partition.update({name: halfedge_pij})

        
        # # summary partition
        num_part_dict = {f'n_{key}': data_partition[key].shape[-1] for key in data_partition.keys()}
        # santiy check
        assert sum(value for key, value in num_part_dict.items() if key.startswith('n_node_part_')) == data.node_type.shape[0]
        assert sum(value for key, value in num_part_dict.items() if key.startswith('n_halfedge_part_')) == data.halfedge_type.shape[0]

        data.update(data_partition)
        return data, num_part_dict
