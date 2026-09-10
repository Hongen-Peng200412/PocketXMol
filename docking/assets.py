"""读取完整配体模板和标准受体, 并按 docking 定位条件选择口袋.

先阅读 read_template, 再阅读 read_receptor 和 select_pocket. 本模块只返回内存中的数组和 RDKit Mol, 不写源资产或派生文件.
配体原子顺序始终对应 ligand_objects 的 atoms; 受体坐标保持源 receptor_tokens 的世界 XYZ 坐标, 单位 Å.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.spatial.distance import cdist


# 原配体元素顺序决定模型类别编号; 最后一类 mask 由原 FeaturizeMol 添加.
LIGAND_ELEMENTS = (6, 7, 8, 9, 15, 16, 17, 5, 35, 53, 34)
# AdaLigand 的五维键编码; 配位键不属于本轮原模型支持的化学键类别.
BOND_TYPES = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.DATIVE, Chem.BondType.AROMATIC)
# 与 atoms.chirality 的七列逐项对应, 不根据沉积坐标猜测或重建模板手性.
CHIRAL_TYPES = (Chem.ChiralType.CHI_OTHER, Chem.ChiralType.CHI_OCTAHEDRAL, Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL, Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.ChiralType.CHI_SQUAREPLANAR)


@lru_cache(maxsize=256)
def read_template(path):
    """以源模板的原子和键顺序构建完整重原子分子, 不修电荷或键级, 过滤无碳原子的配体, 并排除配位键.

    输入参数:
        - path: Path, 如 ligand_objects/CCD_ATP.npz, 读取 atoms、bonds、atom_names 和 smiles; 不读取 object 类型的 blobs.

    返回值:
        - atoms: 结构化数组, (N,), N 为完整模板重原子数; 本链读取以下源字段, 其它字段原样保留但不参与计算.
            - atoms.element: int8, (N,), 原子序数, 如6表示C; 模型词表顺序见 LIGAND_ELEMENTS.
            - atoms.charge: int8, (N,), 每个原子的形式电荷, 如-1; 原样写入 RDKit, 不为 sanitize 修改.
            - atoms.chirality: bool, (N, 7), 源手性类别 one-hot; 列序为 OTHER、OCTAHEDRAL、TETRAHEDRAL_CW、TRIGONALBIPYRAMIDAL、UNSPECIFIED、TETRAHEDRAL_CCW、SQUAREPLANAR.
        - bonds: 结构化数组, (M,), M 为无向化学键数; 本链读取以下源字段.
            - bonds.atom_1/bonds.atom_2: int32, (M,), 键的两个完整模板原子编号, 如0和1, 索引 atoms 第一维.
            - bonds.type: bool, (M, 5), 列序为单、双、三、配位、芳香; 配位键被排除, 源索引4(第五列)的芳香键显式映射到原模型类别4.
        - mol: RDKit Mol, N 个原子按 atoms 排列; 含模板形式电荷和已有手性, 没有目标沉积坐标.
        - smiles: str, 源模板字符串, 如 CCO; 原样返回, 是否能作为有效输入由准备阶段检查.

    本进程最多复用 256 份只读模板. 调用方若需添加构象, 必须先复制 mol.
    """
    with np.load(Path(path), allow_pickle=False) as archive:
        # (N,) 与 (M,), 源结构化属性; 不访问无用途的 blobs、coords 或图派生占位项.
        atoms, bonds = archive["atoms"], archive["bonds"]
        atom_names = archive["atom_names"]
        smiles = str(archive["smiles"].item())
    if not np.isin(atoms["element"], LIGAND_ELEMENTS).all():
        raise ValueError("unsupported_ligand_element")
    if not np.any(atoms["element"] == 6):
        raise ValueError("ligand_without_carbon")
    # int64, (M,), 五维 one-hot 中的类别列; 3 是配位键, 4 是芳香键.
    bond_kind = bonds["type"].argmax(axis=1)
    if np.any(bond_kind == 3):
        raise ValueError("unsupported_ligand_bond")
    molecule = Chem.RWMol()
    for properties, atom_name in zip(atoms, atom_names):
        atom = Chem.Atom(int(properties["element"]))
        atom.SetFormalCharge(int(properties["charge"]))
        atom.SetChiralTag(CHIRAL_TYPES[int(properties["chirality"].argmax())])
        atom.SetProp("_TriposAtomName", str(atom_name))
        molecule.AddAtom(atom)
    for bond, kind in zip(bonds, bond_kind):
        molecule.AddBond(int(bond["atom_1"]), int(bond["atom_2"]), BOND_TYPES[int(kind)])
    mol = molecule.GetMol()
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return atoms, bonds, mol, smiles


@lru_cache(maxsize=16)
def read_receptor(path):
    """读取一个 PDB 的完整标准蛋白和 RNA/DNA 重原子表, 在内存排除 UNK.

    输入参数:
        - path: Path, 如 parse/9v7o/receptor_tokens.npz; 源 res_type 0..19 是蛋白, 20..27 是 RNA/DNA, 28 是 UNK.

    返回值:
        - receptor.coords: float32, (P, 3), 保留的 P 个重原子世界 XYZ 坐标, 单位 Å.
        - receptor.element: uint8, (P,), 原子序数, 与 coords 第一维对齐.
        - receptor.res_type: uint8, (P,), 原 AdaLigand 残基编号, 不使用 feat 中的修饰母体类别.
        - receptor.res_index: int32, (P,), PDB 内原残基编号, 如 315; 筛选后允许不连续.
        - receptor.is_backbone: bool, (P,), 原主链标记; 模型只在蛋白25维编码中读取.
        - receptor.atom_name: S4, (P,), 原子 ASCII 名称, 如 b"OP1".

    本进程最多保留 16 份只读受体表; 不加载未使用的49维 feat 或受体共价键数组.
    """
    with np.load(Path(path), allow_pickle=False) as archive:
        # bool, (P_source,), True 对应标准蛋白或八种标准核苷酸; 同时切分六个逐原子数组.
        keep = archive["res_type"] < 28
        receptor = {key: archive[key][keep] for key in ("coords", "element", "res_type", "res_index", "is_backbone", "atom_name")}
    return receptor


# ================================================================================================
def select_pocket(receptor, ligand_coords, given_center, pocket_mode):
    """按残基重原子质量中心选择完整残基, 返回与原受体第一维对齐的口袋掩码.

    输入参数:
        - receptor: read_receptor 返回的标准受体表; P 个原子按 res_index 分为 R 个残基.
        - ligand_coords: (N, 3), 当前 occurrence 的完整沉积世界 XYZ 坐标, 单位 Å; 只在 envelope 模式用于定位条件.
        - given_center: (3,), 本次给定的世界 XYZ 中心, 单位 Å, center 模式可带 C5 偏移.
        - pocket_mode: str, center 使用距离中心 <15 Å, envelope 使用距离任一配体重原子 <10 Å.

    返回值:
        - atom_keep: bool, (P,), True 表示所属残基满足严格距离阈值; 用于切分 receptor 的所有逐原子数组.
    """
    # int64, (P,), 把源中不连续的残基编号压为 0..R-1, 每个值索引 residue_centers 第一维.
    _, residue_index = np.unique(receptor["res_index"], return_inverse=True)
    # (P,), 按实际元素读取原子质量, 用于残基 COM; 不用原子算术均值替代.
    periodic_table = Chem.GetPeriodicTable()
    masses = np.array([periodic_table.GetAtomicWeight(int(element)) for element in receptor["element"]])
    # (R,), 每个残基全部标准重原子的总质量.
    residue_mass = np.bincount(residue_index, weights=masses)
    # (R, 3), 三个世界坐标分量分别按原子质量加权, 单位 Å.
    residue_centers = np.stack([np.bincount(residue_index, weights=masses * receptor["coords"][:, axis]) for axis in range(3)], axis=1) / residue_mass[:, None]
    if pocket_mode == "center":
        residue_keep = np.linalg.norm(residue_centers - given_center, axis=1) < 15.0
    elif pocket_mode == "envelope":
        residue_keep = cdist(residue_centers, ligand_coords).min(axis=1) < 10.0
    else:
        raise ValueError(f"unknown pocket_mode: {pocket_mode}")
    return residue_keep[residue_index]
