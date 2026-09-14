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

    root 下 smiles_graphs_v1.npz 的 atom_offsets 和 bond_offsets 分别切分逐原子属性和逐键属性; bond_index 为每个分子内部的零起始编号. smiles_symmetries_v1.npz 的 matches_offsets 切分压平排列, matches_shape 恢复 (M, S), M 为排列数, S 为可置换原子数. 两份包各用自己的 smiles 索引, 不假定排序一致.
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

    返回 NPZ 字段字典: candidate_ids 为 int64 (K,), prepared_smiles 为字符串 (K,), coord_offsets 为 int64 (K+1,), coords 为 float32 (N_total, 3). 第 j 个实例的坐标是 coords[coord_offsets[j]:coord_offsets[j+1]], 原子顺序与该 prepared_smiles 的公共图一致.
    """
    with np.load(Path(root) / f"{pdb_id}.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


# ================================================================================================
@lru_cache(maxsize=512)
def read_smiles_graph(root, prepared_smiles):
    """读取一个精确字符串对应的完整公共图, 返回原模型需要的属性和 RDKit Mol.

    返回字典: element 为 int64 (N,) 原子序数, charge 为 int8 (N,) 形式电荷, bond_index 为 int64 (2, E) 无向键端点, bond_type 为 int64 (E,) 原模型单/双/三/芳香类别 1/2/3/4, matches_iso 为 int64 (M, S) 原子置换, mol 为无构象的完整重原子分子. 缓存对象只读, 增加构象前须复制 mol.
    """
    graphs, symmetries, graph_index, symmetry_index = read_smiles_assets(root)
    index = graph_index[prepared_smiles]
    start, stop = graphs["atom_offsets"][index:index + 2]
    bond_start, bond_stop = graphs["bond_offsets"][index:index + 2]
    element = graphs["element"][start:stop].astype(np.int64)
    charge = graphs["charge"][start:stop]
    bond_index = graphs["bond_index"][:, bond_start:bond_stop].astype(np.int64)
    bond_kind = graphs["bond_type"][bond_start:bond_stop]
    if not np.isin(element, LIGAND_ELEMENTS).all() or not np.any(element == 6) or np.any(bond_kind == 3):
        raise UnsupportedSmilesError("unsupported_smiles_element_or_bond")
    # 只靠元素/电荷/芳香键不能恢复 [nH] 的显式氢计数; 精确字符串在缓存首次命中时构建 SDF/RMSD 所需 Mol.
    molecule = Chem.RemoveHs(Chem.MolFromSmiles(prepared_smiles, sanitize=True), sanitize=True)
    parsed_element = np.array([atom.GetAtomicNum() for atom in molecule.GetAtoms()])
    parsed_charge = np.array([atom.GetFormalCharge() for atom in molecule.GetAtoms()])
    parsed_kind = np.array([BOND_TYPES.index(bond.GetBondType()) for bond in molecule.GetBonds()])
    parsed_bonds = np.array([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in molecule.GetBonds()], dtype=np.int64).reshape(-1, 2).T
    if not np.array_equal(parsed_element, element) or not np.array_equal(parsed_charge, charge) or not np.array_equal(parsed_bonds, bond_index) or not np.array_equal(parsed_kind, bond_kind):
        raise UnsupportedSmilesError("runtime_smiles_atom_or_bond_order_differs_from_asset")
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
        raise UnsupportedSmilesError(f"smiles_coordinate_identity_mismatch: {pdb_id}/{candidate_id}")
    start, stop = archive["coord_offsets"][index:index + 2]
    return archive["coords"][start:stop]
