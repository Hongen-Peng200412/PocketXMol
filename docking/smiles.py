"""按精确 prepared SMILES 读取公共重原子图、对称排列和实例沉积坐标.

read_smiles_graph 为 Dataset、采样与评价提供相同原子顺序的化学图; read_smiles_coords 读取已迁移的世界 XYZ 坐标, 单位 Å. 本模块只读现有 NPZ, 不解析 ligand_object、不重新枚举自同构, 也不写源资产.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem

from docking.assets import BOND_TYPES, LIGAND_ELEMENTS


class UnsupportedSmilesError(ValueError):
    """冻结实例被迁移检查明确标记为不支持, 测试仍须保留失败分母."""


@lru_cache(maxsize=1)
def read_smiles_assets(root):
    """每进程读取一份公共 NPZ, 返回图、排列及精确字符串索引.

    输入 root 是包含两份公共 NPZ 的目录, 如 smiles_assets. 返回 tuple(graphs, symmetries, graph_index, symmetry_index), 不假定两包的字符串排序一致.
        - graphs: dict, smiles_graphs_v1.npz 的只读内存字段.
            - smiles: Unicode, (K,), K 个精确字符串, 如 CCO.
            - atom_offsets: int64, (K+1,), 切分 element、charge、atom_in_ring 的原子轴; 首值0, 末值为总原子数N_total.
            - element/charge: int16/int8, (N_total,), 原子序数与形式电荷, 如6与0.
            - atom_in_ring: bool, (N_total,4), 是否属于3/4/5/6元环; 原dock输入不读取该字段.
            - bond_offsets: int64, (K+1,), 切分 bond_index 第二维与 bond_type、bond_in_ring 第一维; 首值0, 末值为总键数E_total.
            - bond_index: int32, (2,E_total), 每个分子内部零起始原子端点; 不加 atom_offsets.
            - bond_type: uint8, (E_total,), 0/1/2/3/4对应单、双、三、配位、芳香, 如4表示芳香.
            - bond_in_ring: bool, (E_total,4), 逐键3/4/5/6元环标记; 原dock输入不读取.
        - symmetries: dict, smiles_symmetries_v1.npz 的只读内存字段.
            - smiles/atom_count: Unicode/int32, (K,), 精确字符串与对应重原子数, 如CCO与3.
            - matches_offsets: int64, (K+1,), 切分压平的 matches_iso; 首值0, 末值为总排列元素数.
            - matches_shape: int32, (K,2), 每图恢复的(M,S), M为排列数, S为可置换原子数; 无可置换原子时为(1,0).
            - matches_iso: int32, 一维压平排列; 例如[[0,2],[2,0]]表示该图原子0和2可交换.
        - graph_index/symmetry_index: dict[str,int], 精确字符串到各自包的图编号, 如CCO对应0.
    两个包还保留 schema_version 标量1, 不参与模型计算.
    """
    root = Path(root)
    with np.load(root / "smiles_graphs_v1.npz", allow_pickle=False) as archive:
        graphs = {key: archive[key] for key in archive.files}
    with np.load(root / "smiles_symmetries_v1.npz", allow_pickle=False) as archive:
        symmetries = {key: archive[key] for key in archive.files}
    return graphs, symmetries, {str(value): index for index, value in enumerate(graphs["smiles"])}, {str(value): index for index, value in enumerate(symmetries["smiles"])}


@lru_cache(maxsize=16)
def read_coordinate_archive(root, pdb_id):
    """读取一个 PDB 的 SMILES 顺序坐标包; 16 份进程缓存限制重复解压.

    输入 root 是坐标包目录, pdb_id 如5irx. 返回 dict, 字段如下.
        - schema_version: int64标量1, 当前坐标包格式.
        - candidate_ids: int64, (K,), 当前PDB内K个实例编号, 严格升序, 如[0,2].
        - prepared_smiles: Unicode, (K,), 与 candidate_ids 对齐的精确字符串, 如CCO.
        - coord_offsets: int64, (K+1,), 切分 coords 第一维; 首值0, 末值N_total, 第j个区间属于 candidate_ids[j].
        - coords: float32, (N_total,3), 完整沉积重原子的世界XYZ坐标, 单位Å; 每个区间与对应公共SMILES图原子顺序一致.
    """
    with np.load(Path(root) / f"{pdb_id}.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


# ================================================================================================
@lru_cache(maxsize=512)
def read_smiles_graph(root, prepared_smiles):
    """读取一个精确字符串对应的完整公共图, 返回原模型需要的属性和 RDKit Mol.

    输入 root 为公共包目录, prepared_smiles 为精确身份字符串, 如[nH]1cccc1; 不重新规范化身份. 缓存对象只读, 增加构象前须复制 mol.
    返回 dict, N为该图重原子数, E为无向键数, M为排列数, S为可置换原子数.
        - element: int64, (N,), 按公共SMILES顺序排列的原子序数, 如6表示碳.
        - charge: int8, (N,), 逐原子的形式电荷, 如-1; 供分子身份检查, 原模型不另加电荷特征.
        - bond_index: int64, (2,E), 无向键的两个原子端点, 值索引 element.
        - bond_type: int64, (E,), 原模型单/双/三/芳香类别1/2/3/4; 芳香类别与官方解析一致.
        - matches_iso: int64, (M,S), 原子自同构排列, 数值索引该图原子; 原噪声器用于同构重分配.
        - mol: RDKit Mol, 相同原子顺序的完整重原子图, 含精确SMILES的手性与显式氢语义, 不含沉积构象.
    """
    graphs, symmetries, graph_index, symmetry_index = read_smiles_assets(root)
    index = graph_index[prepared_smiles]
    # 原子与键分别从公共连接数组切片, 端点仍为本图原子编号.
    start, stop = graphs["atom_offsets"][index:index + 2]
    bond_start, bond_stop = graphs["bond_offsets"][index:index + 2]
    element = graphs["element"][start:stop].astype(np.int64)
    charge = graphs["charge"][start:stop]
    bond_index = graphs["bond_index"][:, bond_start:bond_stop].astype(np.int64)
    bond_kind = graphs["bond_type"][bond_start:bond_stop]
    if not np.isin(element, LIGAND_ELEMENTS).all() or not np.any(element == 6) or np.any(bond_kind == 3):
        raise ValueError("unsupported_smiles_element_or_bond")
    # 只靠元素/电荷/芳香键不能恢复 [nH] 的显式氢计数; 精确字符串在缓存首次命中时构建 SDF/RMSD 所需 Mol.
    molecule = Chem.RemoveAllHs(Chem.MolFromSmiles(prepared_smiles, sanitize=True), sanitize=True)
    parsed_element = np.array([atom.GetAtomicNum() for atom in molecule.GetAtoms()])
    parsed_charge = np.array([atom.GetFormalCharge() for atom in molecule.GetAtoms()])
    parsed_kind = np.array([BOND_TYPES.index(bond.GetBondType()) for bond in molecule.GetBonds()])
    parsed_bonds = np.array([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in molecule.GetBonds()], dtype=np.int64).reshape(-1, 2).T
    if not np.array_equal(parsed_element, element) or not np.array_equal(parsed_charge, charge) or not np.array_equal(parsed_bonds, bond_index) or not np.array_equal(parsed_kind, bond_kind):
        raise ValueError("runtime_smiles_atom_or_bond_order_differs_from_asset")
    # (M,S), 每图独立的压缩排列切片; 不重新计算或改变公共排列.
    symmetry_id = symmetry_index[prepared_smiles]
    match_start, match_stop = symmetries["matches_offsets"][symmetry_id:symmetry_id + 2]
    matches = symmetries["matches_iso"][match_start:match_stop].reshape(tuple(symmetries["matches_shape"][symmetry_id])).astype(np.int64)
    return dict(element=element, charge=charge, bond_index=bond_index, bond_type=np.array([1, 2, 3, 0, 4], dtype=np.int64)[bond_kind], matches_iso=matches, mol=molecule)


def read_smiles_coords(root, pdb_id, candidate_id, prepared_smiles):
    """按实例编号读取公共 SMILES 顺序的 float32 (N, 3) 沉积世界 XYZ 坐标, 单位 Å.

    candidate_ids 升序保存, searchsorted 返回对应实例位置. 精确字符串同时核对, 防止坐标与不同化学身份的公共图静默组合. 返回只读缓存中的切片, 调用方不得原地改写.
    """
    archive = read_coordinate_archive(root, pdb_id)
    index = int(np.searchsorted(archive["candidate_ids"], candidate_id))
    if index == len(archive["candidate_ids"]) or archive["candidate_ids"][index] != candidate_id or str(archive["prepared_smiles"][index]) != prepared_smiles:
        raise ValueError(f"smiles_coordinate_identity_mismatch: {pdb_id}/{candidate_id}")
    # (N,3), 当前实例区间的世界XYZ重原子坐标; 与精确字符串的原子顺序一致.
    start, stop = archive["coord_offsets"][index:index + 2]
    return archive["coords"][start:stop]
