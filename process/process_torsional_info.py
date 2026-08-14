from itertools import product
import os
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import networkx as nx
import argparse
import torch

import sys
sys.path.append('.')
from utils.fragment import find_rotatable_bond_mat
from process.unmi.process_mols import get_unmi_raw_db, unmi_data_to_rdmol


def get_mol_from_data(data_dict, mols_dir=None, train_txn=None, val_txn=None, root_dir=None):
    """根据数据定位字段恢复固定二维拓扑的原始 RDKit 分子。

    输入参数:
        - data_dict: Mapping，至少含 ``db`` 与 ``data_id`` 两个叶字段；``unmi`` 分支不读取独立 ``key`` 叶，而是从 ``data_id[10:]`` 派生 LMDB 键。
        - data_dict.db: str，数据源名；在 ``mols_dir is None`` 时决定 ``root_dir`` 下的分子子目录。
        - data_dict.data_id: str，样本标识；前缀决定文件命名或 UNMI 事务选择，完整值进入文件名。
        - mols_dir: str|None，显式分子目录；非空时跳过依据 ``db`` 的目录推导。
        - train_txn: LMDB Transaction|None，UNMI 训练样本的只读事务；``data_id`` 含 ``train`` 时必需。
        - val_txn: LMDB Transaction|None，UNMI 验证样本的只读事务；``data_id`` 含 ``valid`` 时必需。
        - root_dir: str|None，数据根目录；``mols_dir`` 为空时必需，构象/docking 推理通常传 ``data``。

    返回值:
        - mol: RDKit Mol|None，按源文件或 UNMI 记录恢复的固定原子顺序、键与 conformer；文件解析失败时 RDKit 可返回 None。

    异常:
        - AssertionError: ``mols_dir`` 与 ``root_dir`` 同时为空，或 ``db`` 不在已知目录映射且不是 ``unmi``。
        - NotImplementedError: ``data_id`` 前缀不在已知文件名规则内。

    注意:
        - ``db`` 只控制目录，``data_id`` 的首个下划线片段控制文件后缀；两者应由上游数据契约保证一致。
    """
    if mols_dir is None:
        assert root_dir is not None, 'either mols_dir or root_dir should not be None'
        # ``db``：str，当前样本的数据源名，用于选择 ``root_dir`` 下的目录布局。
        db = data_dict['db']
        if db in ['geom', 'qm9', 'cremp']:
            # ``mols_dir``：str，构象数据源的 ``<root>/<db>/mols`` 目录。
            mols_dir = os.path.join(root_dir, db, 'mols')
        elif db in ['moad', 'pbdock', 'csd', 'apep', 'pepbdb', 'poseb', 'poseboff']:
            # ``mols_dir``：str，复合物/docking 数据源的 ``<root>/<db>/files/mols`` 目录。
            mols_dir = os.path.join(root_dir, db, 'files/mols')
        else:
            assert db == 'unmi', f'Unknown db {db} to get gt mols'

    # ``data_id``：str，完整样本标识，同时用于推导数据源前缀和磁盘文件名。
    data_id = data_dict['data_id']
    # ``db_name``：str，``data_id`` 首个下划线前的前缀，决定读取文件还是 UNMI 事务。
    db_name = data_id.split('_')[0]
    
    if db_name in ['geom', 'qm9', 'cremp']:
        # ``mol_fn``：str，构象数据源文件名 ``<data_id>.sdf``。
        mol_fn = data_id + '.sdf'
    elif db_name in ['moad', 'pbdock', 'csd', 'apep', 'pepbdb', 'poseb', 'poseboff']:
        # ``mol_fn``：str，复合物/docking 数据源文件名 ``<data_id>_mol.sdf``。
        mol_fn = data_id + '_mol.sdf'
    elif db_name == 'unmi':
        # ``key``：bytes，去掉 ``data_id`` 前 10 个字符后编码得到的 UNMI LMDB 键；切片规则来自既有数据命名约定。
        key = data_id[10:].encode()
        if 'train' in data_id:
            # ``data``：bytes|None，从训练事务按派生键读取的 pickle 载荷。
            data = train_txn.get(key)
        elif 'valid' in data_id:
            # ``data``：bytes|None，从验证事务按派生键读取的 pickle 载荷。
            data = val_txn.get(key)
        # ``data``：bytes -> Python 映射，反序列化后的 UNMI 分子记录。
        data = pickle.loads(data)
        # ``mol``：RDKit Mol，由 UNMI 记录恢复并附加 conformer。
        mol = unmi_data_to_rdmol(data, add_confs=True)
        return mol
    else:
        raise NotImplementedError(f'Unknown db_name: {db_name}')
    # ``mol``：RDKit Mol|None，从 ``mols_dir/mol_fn`` 读取；原子顺序和键用于固定拓扑重建。
    mol = Chem.MolFromMolFile(os.path.join(mols_dir, mol_fn))
    return mol


def get_torsional_info_mol(mol, bond_index, data_id=None):
    """从固定二维分子图预计算柔性构象与 docking 所需的扭转、距离和对称性字段。

    输入参数:
        - mol: RDKit Mol，已去显式氢的小分子；原子顺序必须与 ``bond_index`` 的索引空间一致。
        - bond_index: LongTensor|ndarray，形状为 (2, E)，分子键的 0-based 有向端点；当前数据通常为每条无向键保存两个方向。
        - data_id: str|None，仅用于异常消息定位样本，不参与字段计算。

    返回值:
        - result: dict，单分子的扭转约束与原子置换字段。
        - result.bond_rotatable: int64 ndarray，形状为 (E,)，与 ``bond_index`` 列对齐；1 表示该有向键对应可旋转键。
        - result.tor_twisted_pairs: dict[tuple[int, int], list[set[int], set[int]]]，每个规范方向可旋转键 ``(left, right)`` 映射到断键后 left 侧与 right 侧的可转原子集合；两个轴端点已分别移除。
        - result.fixed_dist_torsion: float64 ndarray，形状为 (N, N)；跨任一可旋转键两侧的原子对为 0，其余原子对为 1。
        - result.tor_bond_mat: ndarray，形状为 (N, N)，RDKit 原子对可旋转键标记矩阵。
        - result.path_mat: float64 ndarray，形状为 (N, N)，RDKit 分子图最短路径长度矩阵，单位为键数。
        - result.nbh_dict: dict[int, list[int]]，每个 0-based 原子索引映射到与其图距离恰为 1 的直接邻居索引列表。
        - result.matches_graph: int64 ndarray，形状为 (M_g, S_g)，忽略手性时的自同构映射；只保留在候选映射之间发生变化的 S_g 个原子列。
        - result.matches_iso: int64 ndarray，形状为 (M_i, S_i)，考虑手性时的自同构映射；训练 ``reassign_in`` 用它在等价原子编号中选择输入排列。

    异常:
        - ValueError: 删除某条候选可旋转键后没有恰好得到两个连通分量；异常消息携带 ``data_id``。

    注意:
        - ``tor_twisted_pairs`` 只为 ``left < right`` 的一个键方向建表；下游需按实际方向选择对应的集合侧。
        - ``fixed_dist_torsion`` 的 0/1 是“该距离是否受扭转保持”的标志，不是以 Å 为单位的实际距离。
    """
    if isinstance(bond_index, torch.Tensor):
        # ``bond_index``：LongTensor -> ndarray，形状保持 (2, E)；预处理全程在 CPU/NumPy 与 NetworkX 上执行。
        bond_index = bond_index.numpy()

    # ``rot_mat``：ndarray，形状为 (N, N)，逐原子对标记 RDKit 规则识别出的可旋转键。
    rot_mat = find_rotatable_bond_mat(mol)
    # ``bond_rotatable``：ndarray，形状为 (E,)，按每列端点从 ``rot_mat`` 抽取的有向键可旋转标志。
    bond_rotatable = rot_mat[bond_index[0], bond_index[1]]
    # ``rotatable_bond_index``：[2, E] -> [2, R]，只保留可旋转的有向键列；双向输入会使每条无向键出现两次。
    rotatable_bond_index = bond_index[:, bond_rotatable==1]
    
    # ``G_base``：NetworkX Graph，以 0-based 原子索引为节点、以 ``bond_index`` 列为无向边的分子图。
    G_base = nx.from_edgelist(bond_index.T)
    # ``tor_twist_pairs``：dict，逐规范方向可旋转键保存断键后的两侧可转原子集合。
    tor_twist_pairs = {}
    # ``rot_bond``：ndarray，形状为 (2,)，当前可旋转有向键的 ``(left, right)`` 原子索引。
    for rot_bond in rotatable_bond_index.T:
        if rot_bond[0] > rot_bond[1]: # note: bond is symmetric
            continue
        # ``G_break``：NetworkX Graph，当前可旋转键被删除后的独立图副本。
        G_break = G_base.copy()
        G_break.remove_edge(*rot_bond)
        # ``connected_components``：list[set[int]]，断键后的全部原子连通分量；合法桥边应恰有两项。
        connected_components = list(nx.connected_components(G_break))
        if len(connected_components) == 2:
            # ``component_0``：set[int]，断键后的第一个原子连通分量；随后会原地移除其所属轴端点。
            # ``component_1``：set[int]，断键后的第二个原子连通分量；随后会原地移除其所属轴端点。
            component_0, component_1 = connected_components
            if rot_bond[0] in component_0:
                component_0.remove(rot_bond[0])
                component_1.remove(rot_bond[1])
                # ``tor_twist_pairs[(left, right)]``：list[set[int], set[int]]，依次为 left 侧与 right 侧的非轴原子集合。
                tor_twist_pairs[rot_bond[0], rot_bond[1]] = [
                    component_0, component_1]
            else:
                component_1.remove(rot_bond[0])
                component_0.remove(rot_bond[1])
                # ``tor_twist_pairs[(left, right)]``：list[set[int], set[int]]，重排两个连通分量以保持键端点方向与列表侧一致。
                tor_twist_pairs[rot_bond[0], rot_bond[1]] = [
                    component_1, component_0]
        else:
            raise ValueError(f'Skip: {data_id} does not have two connected components.')
        
    # # make fixed_dist
    # way 1: initial all fixed (not right if there are multiple components)
    # ``n_atoms``：int，当前去氢分子的原子数 N，也是所有原子对矩阵的边长。
    n_atoms = mol.GetNumAtoms()
    # ``fixed_dist``：float64 ndarray，形状为 (N, N)，初始全 1；0 将标记会受至少一条扭转键影响的跨侧原子对。
    fixed_dist = np.ones((n_atoms, n_atoms))
    # ``tor_edge``：tuple[int, int]，当前规范方向扭转键的两个轴端原子编号。
    # ``twisted_edges``：list[set[int], set[int]]，当前扭转键两侧的非轴原子集合。
    for tor_edge, twisted_edges in tor_twist_pairs.items():
        # ``not_fixed_pair``：tuple[int, int]，当前分别取自两侧集合的 0-based 原子索引对，其欧氏距离可随该扭转变化。
        for not_fixed_pair in product(twisted_edges[0], twisted_edges[1]):
            fixed_dist[not_fixed_pair[0], not_fixed_pair[1]] = 0
            fixed_dist[not_fixed_pair[1], not_fixed_pair[0]] = 0
            
    # # make nbh info
    # ``path_mat``：float64 ndarray，形状为 (N, N)，任意原子对在分子图上的最短路径长度，单位为键数。
    path_mat = Chem.GetDistanceMatrix(mol)
    # ``nbh_dict``：dict[int, list[int]]，逐原子收集图距离为 1 的直接邻居索引。
    nbh_dict = {}
    # ``i``：int，当前中心原子的 0-based RDKit 索引，取值范围为 ``[0, N)``。
    for i in range(n_atoms):
        # ``nbh_dict[i]``：list[int]，``path_mat`` 第 i 行中值为 1 的列索引，顺序按 NumPy 升序索引产生。
        nbh_dict[i] = np.where(path_mat[i] == 1)[0].tolist()
    
    # # add mol symmetries
    # ``matches_list``：list[ndarray]，依次保存忽略手性与考虑手性的压缩自同构映射矩阵。
    matches_list = []
    # ``isomeric``：bool，False/True 分别控制 RDKit 自子结构匹配是否使用手性约束。
    for isomeric in [False, True]:
        # ``matches``：int64 ndarray，初始形状为 (M, N)，每行是 RDKit 返回的一种原子自同构置换，最多保留 10000 行。
        matches = np.array(mol.GetSubstructMatches(mol, uniquify=False, useChirality=isomeric, maxMatches=10000))
        # ``natural_order``：int64 ndarray，形状为 (N,)，恒等原子排列 ``[0, ..., N-1]``。
        natural_order = np.arange(n_atoms)
        # ``is_natural``：Bool ndarray，形状为 (M,)，标出哪些自同构行等于恒等排列。
        is_natural = (matches == natural_order).all(-1)
        if is_natural.sum() == 0:
            # ``is_natural``：Bool ndarray，形状从 (M,) 更新为 (M + 1,)，新增首行被指定为恒等排列。
            is_natural = np.array([True] + [False] * len(matches))
            # ``matches``：[M, N] -> [M + 1, N]，在首行显式补入 RDKit 未返回的恒等排列。
            matches = np.concatenate([natural_order[None], matches], axis=0)
        # ``inconsistent``：Bool ndarray，形状为 (N,)，标出至少一个候选排列与恒等排列不同的原子列。
        inconsistent = (matches != matches[is_natural]).any(axis=0)
        # ``matches``：[M, N] -> [M, S]，只保留发生置换的对称原子列，S 为当前手性设置下的可变列数。
        matches = matches[:, inconsistent]
        matches_list.append(matches)
    # ``matches_graph``：int64 ndarray，形状为 (M_g, S_g)，忽略手性的压缩自同构映射。
    matches_graph = matches_list[0]
    # ``matches_isom``：int64 ndarray，形状为 (M_i, S_i)，考虑手性的压缩自同构映射。
    matches_isom = matches_list[1]
    
    # ``result``：dict，把扭转、距离、邻接和对称原子置换字段按下游 ``PocketMolData`` 使用的叶名汇总。
    result = {
            # ``result.bond_rotatable``：int64 ndarray，形状为 (E,)，逐有向键的可旋转标志。
            'bond_rotatable': np.array(bond_rotatable, dtype=np.int64),
            # ``result.tor_twisted_pairs``：dict[tuple[int, int], list[set[int], set[int]]]，逐规范方向扭转键的两侧非轴原子集合。
            'tor_twisted_pairs': tor_twist_pairs,
            # ``result.fixed_dist_torsion``：float64 ndarray，形状为 (N, N)，扭转下保持距离为 1、允许变化为 0。
            'fixed_dist_torsion': fixed_dist,
            # ``result.tor_bond_mat``：ndarray，形状为 (N, N)，逐原子对的可旋转键矩阵。
            'tor_bond_mat': rot_mat,
            # ``result.path_mat``：float64 ndarray，形状为 (N, N)，分子图最短路径长度，单位为键数。
            'path_mat': path_mat,
            # ``result.nbh_dict``：dict[int, list[int]]，逐原子的直接邻居索引列表。
            'nbh_dict': nbh_dict,
            # ``result.matches_graph``：int64 ndarray，形状为 (M_g, S_g)，忽略手性的压缩自同构映射。
            'matches_graph': matches_graph,
            # ``result.matches_iso``：int64 ndarray，形状为 (M_i, S_i)，考虑手性的压缩自同构映射。
            'matches_iso': matches_isom,
        }
        
    return result


def get_torsional_info(df, mol_path, save_path, mols_dir):
    """批量读取分子记录、计算扭转字段并按 ``data_id`` 写入独立 LMDB。

    输入参数:
        - df: DataFrame，每行至少含叶字段 ``data_id``。
        - df.data_id: str，分子记录、原始 RDKit 分子和输出扭转记录共同使用的样本键。
        - mol_path: str，主分子 ``LMDBDatabase`` 路径；每条记录至少含 ``bond_index`` 及 ``get_mol_from_data`` 所需定位叶。
        - save_path: str，待写入的扭转信息 LMDB 路径；每个键的值为 ``get_torsional_info_mol`` 返回映射。
        - mols_dir: str，原始 SDF 目录；路径含 ``unmi`` 时改用 UNMI 训练/验证事务。

    返回值:
        - None: 所有成功样本被写入 ``save_path``，缺少主分子记录的样本只打印并跳过。
    """
    from utils.dataset import LMDBDatabase
    # ``mol_lmdb``：LMDBDatabase，只读主分子记录库，按 ``data_id`` 返回 ``PocketMolData`` 风格映射。
    mol_lmdb = LMDBDatabase(mol_path, readonly=True)
    # ``tor_lmdb``：LMDBDatabase，可写扭转字段库，键与主分子库保持一致。
    tor_lmdb = LMDBDatabase(save_path, readonly=False)
    
    if 'unmi' in mols_dir:
        # ``train_txn``：LMDB Transaction，UNMI 原始分子训练库的只读事务。
        # ``val_txn``：LMDB Transaction，UNMI 原始分子验证库的只读事务。
        train_txn, val_txn = get_unmi_raw_db()
    else:
        # ``train_txn``：None，文件型数据源不需要 UNMI 训练事务。
        # ``val_txn``：None，文件型数据源不需要 UNMI 验证事务。
        train_txn, val_txn = None, None
    
    # ``_``：int，未使用的 DataFrame 原始行索引。
    # ``line``：Series，当前元数据行，至少提供 ``data_id``。
    for _, line in tqdm(df.iterrows(), total=len(df)):
        # ``data_id``：str，当前样本在输入/输出 LMDB 与原始分子文件中的共同定位键。
        data_id = line['data_id']
        # ``result``：dict，预置为空；读取和计算成功后替换为完整扭转字段映射。
        result = {}

        # ``mol_data``：Mapping|None，主 LMDB 中当前样本的分子字段记录。
        mol_data = mol_lmdb[data_id]
        if mol_data is None:
            print(f'Skip: {data_id} does not have mol data.')
            continue
        # ``bond_index``：LongTensor -> ndarray，形状为 (2, E)，当前分子的有向键端点索引。
        bond_index = mol_data['bond_index'].numpy()
        
        # # find rotatable bonds
        # ``mol``：RDKit Mol|None，按记录定位字段加载的固定二维拓扑分子。
        mol = get_mol_from_data(mol_data, mols_dir, train_txn, val_txn)
        # ``mol``：RDKit Mol，移除显式氢后其原子索引必须与 ``bond_index`` 保持一致。
        mol = Chem.RemoveAllHs(mol)
        
        # ``result``：dict，当前样本完整的扭转、距离、邻接与自同构叶字段。
        result = get_torsional_info_mol(mol, bond_index, data_id)
        
        tor_lmdb.add_one(data_id, result)
    tor_lmdb.close()
    # train_txn.close()
    # val_txn.close()


def get_db_config(db_name, save_name='torsion', root='data_train'):
    """按数据源名展开扭转预处理脚本使用的元数据、分子库和输出库路径。

    输入参数:
        - db_name: str，数据源目录名；``pbdock/csd/moad/apep`` 使用含口袋的目录布局。
        - save_name: str，输出 LMDB 文件名主体，不含 ``.lmdb``。
        - root: str，训练数据根目录。

    返回值:
        - df_path: str，输入样本元数据 CSV 路径。
        - mol_path: str，主分子 LMDB 路径。
        - save_path: str，扭转字段 LMDB 路径。
        - mols_dir: str，固定拓扑 SDF 目录。
    """
    # ``data_dir``：str，当前数据源根目录 ``<root>/<db_name>``。
    data_dir = f'{root}/{db_name}'
    # ``df_path``：str，默认元数据表 ``dfs/meta_uni.csv``。
    df_path = os.path.join(data_dir, 'dfs/meta_uni.csv')
    # ``mols_dir``：str，默认固定拓扑分子目录 ``mols``。
    mols_dir = os.path.join(data_dir, 'mols')
    # ``mol_path``：str，默认主分子记录库 ``lmdb/mols.lmdb``。
    mol_path = os.path.join(data_dir, 'lmdb/mols.lmdb')
    # ``save_path``：str，输出扭转记录库 ``lmdb/<save_name>.lmdb``。
    save_path = os.path.join(data_dir, f'lmdb/{save_name}.lmdb')

    if db_name in ['pbdock', 'csd']:
        # ``df_path``：str，pbdock/csd 改用已过滤且含口袋的元数据表。
        df_path = df_path.replace('meta_uni.csv', 'meta_filter_w_pocket.csv')
    if db_name in ['pbdock', 'csd', 'moad', 'apep']:  # with pocket
        # ``mols_dir``：str，含口袋数据源的固定拓扑分子目录 ``files/mols``。
        mols_dir = os.path.join(data_dir, 'files/mols')
        # ``mol_path``：str，含口袋数据源的主复合物记录库 ``lmdb/pocmol10.lmdb``。
        mol_path = os.path.join(data_dir, 'lmdb/pocmol10.lmdb')
    
    return df_path, mol_path, save_path, mols_dir
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_preset', type=bool, default=True)
    parser.add_argument('--db_name', type=str, default='geom')
    args = parser.parse_args()

    if args.from_preset:
        # ``df_path``：str，当前数据库的样本元数据 CSV 路径。
        # ``mol_path``：str，当前数据库的主分子或复合物 LMDB 路径。
        # ``save_path``：str，待写入的扭转信息 LMDB 路径。
        # ``mols_dir``：str，当前数据库原始 RDKit 分子的文件目录或 UNMI 标识路径。
        df_path, mol_path, save_path, mols_dir = get_db_config(args.db_name)
    else:
        raise NotImplementedError
    df_use = pd.read_csv(df_path)

    get_torsional_info(df_use, mol_path, save_path, mols_dir, )
    print('Done processing torsional info.')
