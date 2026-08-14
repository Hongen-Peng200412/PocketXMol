"""为小分子 docking 复用的 PoseBusters 距离碰撞与 InChI 身份检查实现。

代码源自 PoseBusters，并在本仓库中作为 ``docking_aux_scores.py`` 的实现层使用。
距离均来自 RDKit conformer，单位为 Å；半径比例与 InChI 分层比较结果均为无量纲量。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import logging
from copy import deepcopy
from re import findall

from rdkit.Chem.rdchem import GetPeriodicTable, Mol
from rdkit.Chem.rdchem import Mol
from rdkit.rdBase import LogToPythonLogger
from rdkit import RDLogger
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate
from rdkit.Chem.Lipinski import HAcceptorSmarts, HDonorSmarts
from rdkit.Chem.rdchem import AtomValenceException, Bond, Conformer, GetPeriodicTable, Mol, RWMol
from rdkit.Chem.rdMolAlign import GetBestAlignmentTransform
from rdkit.Chem.rdmolfiles import MolFromSmarts
from rdkit.Chem.rdmolops import AddHs, RemoveHs, RemoveStereochemistry, RenumberAtoms, SanitizeMol, AssignStereochemistryFrom3D
from rdkit.Chem.rdMolTransforms import TransformConformer
from rdkit.Chem.inchi import MolFromInchi, MolToInchi
from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger

# ``_periodic_table``：RDKit PeriodicTable，全模块复用的元素范德华/共价半径查询器。
_periodic_table = GetPeriodicTable()

"""Protein related functions."""


from typing import Iterable

from rdkit.Chem.rdchem import Atom, Mol

# ``logger``：logging.Logger，记录缺失 PDB 元数据与标准化回退警告。
logger = logging.getLogger(__name__)
# ``_inorganic_cofactor_elements``：set[str]，即使缺少残基元数据也按元素符号识别的常见无机辅因子集合。
_inorganic_cofactor_elements = {
    "Li",
    "Be",
    "Na",
    "Mg",
    "Cl",
    "K",
    "Ca",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Br",
    "Rb",
    "Mo",
    "Cd",
}
# ``_inorganic_cofactor_ccd_codes``：set[str]，按 PDB CCD 残基名识别的无机辅因子集合。
_inorganic_cofactor_ccd_codes = {
    "FES",
    "MOS",
    "PO3",
    "PO4",
    "PPK",
    "SO3",
    "SO4",
    "VO4",
}
# ``_water_ccd_codes``：set[str]，按 PDB CCD 残基名识别的水集合。
_water_ccd_codes = {
    "HOH",
}


def get_atom_type_mask(mol: Mol, ignore_types: Iterable[str]):
    """根据 PDB 原子/残基元数据生成条件分子的保留掩码。

    输入参数:
        - mol: RDKit Mol，含 A 个原子；若第 0 个原子没有 PDB 残基信息，则全部非氢原子按普通有机分子处理。
        - ignore_types: Iterable[str]，可包含 ``hydrogens``、``protein``、``organic_cofactors``、``inorganic_cofactors``、``waters``。

    返回值:
        - keep_mask: list[bool]，长度为 A，第 a 项为真表示后续距离检查保留 RDKit 原子索引 a。

    异常:
        - ValueError: ``ignore_types`` 含不支持的类别名。
    """
    # ``ignore_types``：set[str]，把任意可迭代输入规范为集合，供后续 O(1) 成员判断。
    ignore_types = set(ignore_types)
    # ``unsupported``：set[str]，调用方传入但本实现没有过滤规则的类别名。
    unsupported = ignore_types - {"hydrogens", "protein", "organic_cofactors", "inorganic_cofactors", "waters"}
    if unsupported:
        raise ValueError(f"Ignore types {unsupported} not supported.")

    if mol.GetAtomWithIdx(0).GetPDBResidueInfo() is None:
        logger.warning("No PDB information found. Assuming organic molecule.")

    # ``ignore_h``：bool，是否排除元素符号为 H 的原子。
    ignore_h = "hydrogens" in ignore_types
    # ``ignore_protein``：bool，是否排除 PDB 元数据中非 HETATM 的蛋白原子。
    ignore_protein = "protein" in ignore_types
    # ``ignore_org_cof``：bool，是否在缺失 PDB 残基信息时把该普通有机分子原子全部排除。
    ignore_org_cof = "organic_cofactors" in ignore_types
    # ``ignore_inorg_cof``：bool，是否按元素或 CCD 残基名排除无机辅因子。
    ignore_inorg_cof = "inorganic_cofactors" in ignore_types
    # ``ignore_water``：bool，是否排除 CCD 残基名为水的原子。
    ignore_water = "waters" in ignore_types

    # 返回列表与 ``mol.GetAtoms()`` 的 RDKit 原子索引顺序严格对齐。
    return [
        _keep_atom(a, ignore_h, ignore_protein, ignore_org_cof, ignore_inorg_cof, ignore_water) for a in mol.GetAtoms()
    ]


def _keep_atom(  # noqa: PLR0913, PLR0911
    atom: Atom, ignore_h: bool, ignore_protein: bool, ignore_org_cof: bool, ignore_inorg_cof: bool, ignore_water: bool
):
    """判断一个 RDKit 原子是否应进入分子间距离矩阵。

    输入参数:
        - atom: RDKit Atom，索引空间属于传入条件分子。
        - ignore_h: bool，是否排除氢原子。
        - ignore_protein: bool，是否排除 PDB 非异源原子记录。
        - ignore_org_cof: bool，是否排除无法从 PDB 元数据分类的普通有机分子原子。
        - ignore_inorg_cof: bool，是否按元素符号或 CCD 残基名排除无机辅因子。
        - ignore_water: bool，是否按 CCD 残基名排除水。

    返回值:
        - keep: bool，为真表示保留该原子参与距离与碰撞计算。
    """
    # ``symbol``：str，当前原子的元素符号，用于氢与无机元素过滤。
    symbol = atom.GetSymbol()
    if ignore_h and symbol == "H":
        return False

    if ignore_inorg_cof and symbol in _inorganic_cofactor_elements:
        return False

    # if loaded from PDB file, we can use the residue names and the hetero flag
    # ``info``：AtomPDBResidueInfo|None，承载 HETATM 标志和三字母残基名。
    info = atom.GetPDBResidueInfo()
    if info is None:
        if ignore_org_cof:
            return False
        return True

    # ``is_hetero``：bool，PDB 记录是否为 HETATM；普通蛋白 ATOM 记录为假。
    is_hetero = info.GetIsHeteroAtom()
    if ignore_protein and not is_hetero:
        return False

    # ``residue_name``：str，PDB/CCD 残基名，用于识别水和常见无机辅因子。
    residue_name = info.GetResidueName()
    if ignore_water and residue_name in _water_ccd_codes:
        return False

    if ignore_inorg_cof and residue_name in _inorganic_cofactor_ccd_codes:
        return False

    return True


# inter distance 
def check_intermolecular_distance(  # noqa: PLR0913
    mol_pred: Mol,
    mol_cond: Mol,
    radius_type: str = "vdw",
    radius_scale: float = 1.0,
    clash_cutoff: float = 0.75,
    ignore_types: set[str] = {"hydrogens"},
    max_distance: float = 5.0,
    search_distance: float = 6.0,
):
    """检查预测配体与条件蛋白是否过远，并统计按原子半径归一化的碰撞原子对。

    输入参数:
        - mol_pred: RDKit Mol，预测配体，必须含一个 conformer；配体氢原子固定被排除。
        - mol_cond: RDKit Mol，条件蛋白/受体，必须含一个与配体同坐标系的 conformer。
        - radius_type: str，原子半径类型；``vdw`` 使用范德华半径，``covalent`` 使用共价半径。
        - radius_scale: float，半径和的缩放因子，默认 1.0，无量纲；调用方未显式传值时详情比例为 ``distance/(sum_radii*radius_scale)``。
        - clash_cutoff: float，详情比例小于该值即记为碰撞，默认 0.75，无量纲；不是 0.75 Å。
        - ignore_types: set[str]，条件分子中需要排除的原子类型，默认只排除氢；候选集合见 ``get_atom_type_mask``。
        - max_distance: float，最近配体—条件原子距离允许的上界，默认 5.0 Å。
        - search_distance: float，只为距离任一配体原子不超过该值的条件原子建立详情，默认 6.0 Å。

    返回值:
        - output: dict，包含 ``results`` 与 ``details`` 两个顶层字段。
        - output.results.smallest_distance: float，候选详情中最小配体—条件原子距离，单位 Å；无详情时为 NaN。
        - output.results.not_too_far_away: bool，``smallest_distance <= max_distance`` 是否成立。
        - output.results.num_pairwise_clashes: int，详情中 ``clash=True`` 的配体—条件原子对数量。
        - output.results.no_clashes: bool，详情中是否不存在 ``clash=True`` 的原子对。
        - output.results.most_extreme_ligand_atom_id: int|NA，详情比例最小原子对的配体 RDKit 原子索引。
        - output.results.most_extreme_protein_atom_id: int|NA，详情比例最小原子对的条件分子 RDKit 原子索引。
        - output.results.most_extreme_ligand_element: str|NA，详情比例最小原子对的配体元素符号。
        - output.results.most_extreme_protein_element: str|NA，详情比例最小原子对的条件原子元素符号。
        - output.results.most_extreme_ligand_vdw: float|NA，详情比例最小原子对的配体原子半径，单位 Å；列名保留为 ``vdw``，即使 ``radius_type=covalent``。
        - output.results.most_extreme_protein_vdw: float|NA，详情比例最小原子对的条件原子半径，单位 Å；列名保留为 ``vdw``，即使 ``radius_type=covalent``。
        - output.results.most_extreme_sum_radii: float|NA，两原子未缩放半径和，单位 Å。
        - output.results.most_extreme_distance: float|NA，两原子实际距离，单位 Å。
        - output.results.most_extreme_sum_radii_scaled: float|NA，``sum_radii*radius_scale``，单位 Å。
        - output.results.most_extreme_relative_distance: float|NA，``distance/sum_radii_scaled``，无量纲。
        - output.results.most_extreme_clash: bool|NA，该原子对的详情比例是否小于 ``clash_cutoff``。
        - output.details: DataFrame，每行是预筛选出的一个配体—条件原子对。
        - output.details.ligand_atom_id: int，配体 RDKit 原子索引。
        - output.details.protein_atom_id: int，条件分子 RDKit 原子索引。
        - output.details.ligand_element: str，配体元素符号。
        - output.details.protein_element: str，条件原子元素符号。
        - output.details.ligand_vdw: float，配体原子半径，单位 Å；列名不随 ``radius_type`` 改变。
        - output.details.protein_vdw: float，条件原子半径，单位 Å；列名不随 ``radius_type`` 改变。
        - output.details.sum_radii: float，两原子未缩放半径和，单位 Å。
        - output.details.distance: float，两原子实际距离，单位 Å。
        - output.details.sum_radii_scaled: float，``sum_radii*radius_scale``，单位 Å。
        - output.details.relative_distance: float，``distance/sum_radii_scaled``，无量纲。
        - output.details.clash: bool，``relative_distance < clash_cutoff`` 是否成立。

    注意:
        - 预筛选掩码使用未缩放比例 ``distance/sum_radii < 1/radius_scale``，并强制纳入实际距离最小与未缩放比例最小的原子对；这里记录既有实现，不把预筛选阈值解释为最终碰撞阈值。
    """
    # ``coords_ligand``：ndarray，形状为 (L_all, 3)，预测配体全部原子的 conformer 坐标，单位 Å，按 RDKit 原子索引排序。
    coords_ligand = mol_pred.GetConformer().GetPositions()
    # ``coords_protein``：ndarray，形状为 (P_all, 3)，条件分子全部原子的 conformer 坐标，单位 Å，按 RDKit 原子索引排序。
    coords_protein = mol_cond.GetConformer().GetPositions()

    # ``atoms_ligand``：ndarray，dtype 为 str，形状为 (L_all,)，预测配体逐原子元素符号。
    atoms_ligand = np.array([a.GetSymbol() for a in mol_pred.GetAtoms()])
    # ``atoms_protein_all``：ndarray，dtype 为 str，形状为 (P_all,)，条件分子逐原子元素符号。
    atoms_protein_all = np.array([a.GetSymbol() for a in mol_cond.GetAtoms()])

    # ``idxs_ligand``：ndarray，dtype 为 int，形状为 (L_all,)，预测配体原始 RDKit 原子索引。
    idxs_ligand = np.array([a.GetIdx() for a in mol_pred.GetAtoms()])
    # ``idxs_protein``：ndarray，dtype 为 int，形状为 (P_all,)，条件分子原始 RDKit 原子索引。
    idxs_protein = np.array([a.GetIdx() for a in mol_cond.GetAtoms()])

    # ``mask``：list[bool]，长度为 L_all，配体侧固定排除氢原子。
    mask = [a.GetSymbol() != "H" for a in mol_pred.GetAtoms()]
    # ``coords_ligand``：[L_all, 3] -> [L, 3]，只保留 L 个配体重原子，单位 Å。
    coords_ligand = coords_ligand[mask, :]
    # ``atoms_ligand``：[L_all] -> [L]，与过滤后的配体坐标逐行对齐。
    atoms_ligand = atoms_ligand[mask]
    # ``mask_ligand_idxs``：ndarray，dtype 为 int，形状为 (L,)，把过滤后行号映射回配体 RDKit 原子索引。
    mask_ligand_idxs = idxs_ligand[mask]
    if ignore_types:
        # ``mask``：list[bool]，长度为 P_all，按 ``ignore_types`` 标记条件原子是否保留。
        mask = get_atom_type_mask(mol_cond, ignore_types)
        # ``coords_protein``：[P_all, 3] -> [P_keep, 3]，保留的条件原子坐标，单位 Å。
        coords_protein = coords_protein[mask, :]
        # ``atoms_protein_all``：[P_all] -> [P_keep]，与过滤后的条件坐标逐行对齐。
        atoms_protein_all = atoms_protein_all[mask]
        # ``mask_protein_idxs``：ndarray，dtype 为 int，形状为 (P_keep,)，把过滤后行号映射回条件分子 RDKit 原子索引。
        mask_protein_idxs = idxs_protein[mask]

    # ``radius_ligand``：ndarray，dtype 为 float，形状为 (L,)，逐配体重原子半径，单位 Å。
    radius_ligand = _get_radii(atoms_ligand, radius_type)
    # ``radius_protein_all``：ndarray，dtype 为 float，形状为 (P_keep,)，逐保留条件原子半径，单位 Å。
    radius_protein_all = _get_radii(atoms_protein_all, radius_type)

    # ``distances_all``：ndarray，dtype 为 float，形状为 (L, P_keep)，全部配体重原子与保留条件原子的两两欧氏距离，单位 Å。
    distances_all = _pairwise_distance(coords_ligand, coords_protein)
    # ``mask_protein``：ndarray，dtype 为 bool，形状为 (P_keep,)，保留距离任一配体重原子不超过 ``search_distance`` 的条件原子。
    mask_protein = distances_all.min(axis=0) <= search_distance
    # ``distances``：[L, P_keep] -> [L, P]，只保留搜索半径内 P 个条件原子的距离列，单位 Å。
    distances = distances_all[:, mask_protein]
    # ``radius_protein``：ndarray，dtype 为 float，形状为 (P,)，搜索半径内条件原子的半径，单位 Å。
    radius_protein = radius_protein_all[mask_protein]
    # ``atoms_protein``：ndarray，dtype 为 str，形状为 (P,)，搜索半径内条件原子的元素符号。
    atoms_protein = atoms_protein_all[mask_protein]
    # ``mask_protein_idxs``：[P_keep] -> [P]，搜索半径内行号到原始条件分子 RDKit 原子索引的映射。
    mask_protein_idxs = mask_protein_idxs[mask_protein]

    # ``radius_sum``：ndarray，dtype 为 float，形状为 (L, P)，每个原子对的未缩放半径和，单位 Å。
    radius_sum = radius_ligand[:, None] + radius_protein[None, :]
    # ``relative_distance``：ndarray，dtype 为 float，形状为 (L, P)，预筛选使用的 ``distance/sum_radii``，无量纲。
    relative_distance = distances / radius_sum
    # ``violations``：ndarray，dtype 为 bool，形状为 (L, P)，选择未缩放距离比例小于 ``1/radius_scale`` 的详情候选对。
    violations = relative_distance < 1 / radius_scale

    if distances.size > 0:
        # ``violations``：强制纳入实际距离最小原子对，保证详情包含用于 ``smallest_distance`` 的条目。
        violations[np.unravel_index(distances.argmin(), distances.shape)] = True  # add smallest distances as info
        # ``violations``：强制纳入未缩放半径比例最小原子对，保证详情包含最紧密的相对接触。
        violations[np.unravel_index(relative_distance.argmin(), relative_distance.shape)] = True
    # ``violation_ligand``：ndarray，dtype 为 int，形状为 (V,)，V 个详情原子对的过滤后配体行号。
    # ``violation_protein``：ndarray，dtype 为 int，形状为 (V,)，与 ``violation_ligand`` 逐项配对的搜索后条件原子列号。
    violation_ligand, violation_protein = np.where(violations)
    # ``reverse_ligand_idxs``：ndarray，dtype 为 int，形状为 (V,)，详情原子对映射回配体 RDKit 原子索引。
    reverse_ligand_idxs = mask_ligand_idxs[violation_ligand]
    # ``reverse_protein_idxs``：ndarray，dtype 为 int，形状为 (V,)，详情原子对映射回条件分子 RDKit 原子索引。
    reverse_protein_idxs = mask_protein_idxs[violation_protein]

    # ``details``：DataFrame，V 行，每行对应一个由 ``violations`` 选中的配体—条件原子对。
    details = pd.DataFrame()
    # ``details.ligand_atom_id``：int 列，V 个详情原子对的配体 RDKit 原子索引。
    details["ligand_atom_id"] = reverse_ligand_idxs
    # ``details.protein_atom_id``：int 列，V 个详情原子对的条件分子 RDKit 原子索引。
    details["protein_atom_id"] = reverse_protein_idxs
    # ``details.ligand_element``：str 列，V 个详情配体原子的元素符号。
    details["ligand_element"] = [atoms_ligand[i] for i in violation_ligand]
    # ``details.protein_element``：str 列，V 个详情条件原子的元素符号。
    details["protein_element"] = [atoms_protein[i] for i in violation_protein]
    # ``details.ligand_vdw``：float 列，V 个配体原子半径，单位 Å；列名沿用上游，即使查询共价半径也不变。
    details["ligand_vdw"] = [radius_ligand[i] for i in violation_ligand]
    # ``details.protein_vdw``：float 列，V 个条件原子半径，单位 Å；列名沿用上游，即使查询共价半径也不变。
    details["protein_vdw"] = [radius_protein[i] for i in violation_protein]
    # ``details.sum_radii``：float 列，原子对未缩放半径和，单位 Å。
    details["sum_radii"] = details["ligand_vdw"] + details["protein_vdw"]
    # ``details.distance``：float 列，原子对实际欧氏距离，单位 Å。
    details["distance"] = distances[violation_ligand, violation_protein]
    # ``details.sum_radii_scaled``：float 列，未缩放半径和乘 ``radius_scale``，单位 Å。
    details["sum_radii_scaled"] = details["sum_radii"] * radius_scale
    # ``details.relative_distance``：float 列，实际距离除以缩放半径和，无量纲。
    details["relative_distance"] = details["distance"] / details["sum_radii_scaled"]
    # ``details.clash``：bool 列，详情距离比例是否小于 ``clash_cutoff``。
    details["clash"] = details["relative_distance"] < clash_cutoff

    # ``results.smallest_distance``：float，配体—条件原子对最小欧氏距离，单位 Å。
    # ``results.not_too_far_away``：bool，最小距离是否不大于 ``max_distance``。
    # ``results.num_pairwise_clashes``：int，距离比例小于 ``clash_cutoff`` 的原子对数。
    # ``results.no_clashes``：bool，是否不存在任何碰撞原子对。
    # ``results``：dict，先汇总上述四个候选级叶。
    results = {
        "smallest_distance": details["distance"].min(),
        "not_too_far_away": details["distance"].min() <= max_distance,
        "num_pairwise_clashes": details["clash"].sum(),
        "no_clashes": not details["clash"].any(),
    }

    # ``i``：int|None，详情中 ``relative_distance`` 最小行的 DataFrame 索引；无详情时为 None。
    i = np.argmin(details["relative_distance"]) if len(details) > 0 else None
    # ``most_extreme``：dict，把最紧密详情行的 11 个叶字段逐一加 ``most_extreme_`` 前缀；无详情时各值为 ``pd.NA``。
    most_extreme = {"most_extreme_" + c: details.loc[i][str(c)] if i is not None else pd.NA for c in details.columns}
    # ``results``：合并基本候选结论与最紧密原子对详情后的最终叶字段映射。
    results = {**results, **most_extreme}

    # 顶层 ``results`` 用于候选级汇总，``details`` 保留原子对级审计证据。
    return {"results": results, "details": details}



def _pairwise_distance(x: np.ndarray, y: np.ndarray):
    """计算两组同坐标系三维点之间的完整欧氏距离矩阵。

    输入参数:
        - x: ndarray，形状为 (L, 3)，第一组点坐标，通常单位为 Å。
        - y: ndarray，形状为 (P, 3)，第二组点坐标，与 ``x`` 使用相同单位和坐标系。

    返回值:
        - distances: ndarray，形状为 (L, P)，第 ``(l, p)`` 项是 ``x[l]`` 与 ``y[p]`` 的欧氏距离。
    """
    # 广播变换为 ``[L, 1, 3] - [1, P, 3] -> [L, P, 3]``，再沿 xyz 维取二范数。
    return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)


def _get_radii(atoms: np.ndarray, radius_type: str):
    """按元素符号查询 RDKit 周期表中的逐原子半径。

    输入参数:
        - atoms: ndarray，dtype 为 str，形状为 (A,)，每项是一个 RDKit 可识别的元素符号。
        - radius_type: str，``vdw`` 表示范德华半径，``covalent`` 表示共价半径。

    返回值:
        - radii: ndarray，dtype 为 float，形状为 (A,)，与 ``atoms`` 逐项对齐的半径，单位 Å。

    异常:
        - ValueError: ``radius_type`` 不是 ``vdw`` 或 ``covalent``。
    """
    if radius_type == "vdw":
        return np.array([_periodic_table.GetRvdw(a) for a in atoms])
    elif radius_type == "covalent":
        return np.array([_periodic_table.GetRcovalent(a) for a in atoms])
    else:
        raise ValueError(f"Unknown radius type {radius_type}. Valid values are 'vdw' and 'covalent'.")


# 下行是原源码位于模块中段的惰性字符串表达式，不是模块 Docstring；为保持可执行 AST 完全等价而原样保留。
"""Module to check identity of docked and crystal ligand."""


LogToPythonLogger()
logger = logging.getLogger(__name__)


def check_identity(mol_pred: Mol, mol_true: Mol, inchi_options: str = ""):
    """标准化预测与参考配体，并按标准 InChI 各层比较 docking 所需的二维身份。

    输入参数:
        - mol_pred: RDKit Mol，预测/docked 配体；若含 conformer，会先从三维坐标重新指定立体化学。
        - mol_true: RDKit Mol，参考/晶体配体；若含 conformer，会先从三维坐标重新指定立体化学。
        - inchi_options: str，原样传给 RDKit ``MolToInchi`` 的选项字符串，默认空字符串。

    返回值:
        - output: dict，顶层只含 ``results`` 映射。
        - output.results.inchi_crystal_valid: bool，参考配体标准化后的 InChI 能否重新解析并 sanitize。
        - output.results.inchi_docked_valid: bool，预测配体标准化后的 InChI 能否重新解析并 sanitize。
        - output.results.inchi_crystal: str，参考配体标准化后生成的 InChI 字符串。
        - output.results.inchi_docked: str，预测配体标准化后生成的 InChI 字符串。
        - output.results.inchi_overall: bool，可选叶；两个有效 InChI 是否完全相同。
        - output.results.inchi_version: bool，可选叶；InChI 版本层是否一致。
        - output.results.formula: bool，可选叶；分子式层是否一致。
        - output.results.connections: bool，可选叶；原子连接层是否一致。
        - output.results.hydrogens: bool，可选叶；氢原子层是否一致。
        - output.results.net_charge: bool，可选叶；净电荷层是否一致。
        - output.results.protons: bool，可选叶；质子层是否一致。
        - output.results.stereo_dbond: bool，可选叶；双键立体层是否一致。
        - output.results.stereo_sp3: bool，可选叶；四面体立体层是否一致。
        - output.results.stereo_sp3_inverted: bool，可选叶；四面体反转标记层是否一致。
        - output.results.stereo_type: bool，可选叶；立体类型层是否一致。
        - output.results.stereo_tetrahedral: bool，可选叶；全部四面体相关层的合取结果。
        - output.results.stereo: bool，可选叶；全部双键与四面体立体层的合取结果。

    注意:
        - 两个 InChI 任一无效时不生成分层比较叶字段；调用方读取 ``stereo`` 时会进入其异常回退路径。
    """
    # ``inchi_crystal``：str，参考配体标准化后生成的 InChI。
    inchi_crystal = standardize_and_get_inchi(mol_true, options=inchi_options)
    # ``inchi_docked``：str，预测配体标准化后生成的 InChI。
    inchi_docked = standardize_and_get_inchi(mol_pred, options=inchi_options)

    # ``inchi_crystal_valid``：bool，参考 InChI 是否可被 RDKit 重新解析并 sanitize。
    inchi_crystal_valid = is_valid_inchi(inchi_crystal)
    # ``inchi_docked_valid``：bool，预测 InChI 是否可被 RDKit 重新解析并 sanitize。
    inchi_docked_valid = is_valid_inchi(inchi_docked)

    if inchi_crystal_valid and inchi_docked_valid:
        # ``inchi_comparison``：dict，两个有效 InChI 的逐层布尔比较结果；参数顺序不影响相等判断。
        inchi_comparison = _compare_inchis(inchi_docked, inchi_crystal)
    else:
        # ``inchi_comparison``：空 dict，无效 InChI 不产生可能误导的分层布尔字段。
        inchi_comparison = {}

    # ``results``：dict，固定包含有效性与原始 InChI 字符串，并在两者有效时展开逐层比较叶。
    results = {
        "inchi_crystal_valid": inchi_crystal_valid,
        "inchi_docked_valid": inchi_docked_valid,
        "inchi_crystal": inchi_crystal,
        "inchi_docked": inchi_docked,
        **inchi_comparison,
    }

    return {"results": results}


# ``standard_layers``：list[str]，按 InChI 层前缀顺序定义本实现参与比较的版本、分子式、连接、电荷与立体字段。
# 同位素 ``/i`` 在无同位素信息时不存在，固定氢 ``/f`` 与重连金属 ``/r`` 不属于本实现比较的标准层。
standard_layers = ["=", "/", "/c", "/h", "/q", "/p", "/t", "/b", "/m", "/s"]
# ``layer_names``：dict[str, str]，把 InChI 层前缀映射到返回 ``results`` 中稳定的英文叶字段名。
layer_names = {
    "=": "inchi_version",
    "/": "formula",
    "/c": "connections",
    "/h": "hydrogens",
    "/q": "net_charge",
    "/p": "protons",
    "/b": "stereo_dbond",  # double bond (Z/E) stereochemistry
    "/t": "stereo_sp3",  # tetrahderal stereochemistry
    "/m": "stereo_sp3_inverted",
    "/s": "stereo_type",
    "/i": "isotopic",
}
# ``stereo_all_layers``：list[str]，综合立体化学结论需要合取的双键与四面体结果叶名。
stereo_all_layers = ["stereo_dbond", "stereo_sp3", "stereo_sp3_inverted", "stereo_type"]
# ``stereo_tetrahedral_layers``：list[str]，四面体立体结论需要合取的结果叶名。
stereo_tetrahedral_layers = ["stereo_sp3", "stereo_sp3_inverted", "stereo_type"]


def _compare_inchis(inchi_true: str, inchi_pred: str, layers: list[str] = standard_layers):
    """把两个有效标准 InChI 拆层并返回逐层相等性。

    输入参数:
        - inchi_true: str，第一个有效标准 InChI；名称沿用上游实现，比较本身对参数顺序对称。
        - inchi_pred: str，第二个有效标准 InChI。
        - layers: list[str]，需要比较的 InChI 层前缀，默认使用 ``standard_layers``。

    返回值:
        - results: dict[str, bool]，固定含 ``inchi_overall``、``stereo_tetrahedral``、``stereo``，并按 ``layers`` 含 ``layer_names`` 映射后的逐层布尔叶。

    异常:
        - AssertionError: 第一个 InChI 缺少分子式层 ``/``。
    """
    # ``results``：dict[str, bool]，逐步累积整体、逐层和综合立体比较结果。
    results = {}

    # fast return when overall InChI is the same
    if inchi_true == inchi_pred:
        # ``results.inchi_overall``：bool，完整 InChI 字符串相同，直接记为真。
        results["inchi_overall"] = True
        # ``layer``：str，逐次取需要比较的 InChI 层前缀。
        for layer in layers:
            # ``results[layer_names[layer]]``：bool，完整字符串相同时当前请求层必然相同。
            results[layer_names[layer]] = True
            # ``results.stereo_tetrahedral``：bool，完整字符串相同时四面体立体层综合结论为真；循环内重复写入不改变结果。
            results["stereo_tetrahedral"] = True
            # ``results.stereo``：bool，完整字符串相同时全部立体层综合结论为真；循环内重复写入不改变结果。
            results["stereo"] = True
        return results

    # otherwise comparison by layer
    # ``results.inchi_overall``：bool，完整 InChI 字符串不相同，先记为假，再计算逐层结论。
    results["inchi_overall"] = False
    # ``layers_true``：dict[str, str]，第一个 InChI 的层前缀到层内容映射。
    layers_true = split_inchi(inchi_true)
    # ``layers_pred``：dict[str, str]，第二个 InChI 的层前缀到层内容映射。
    layers_pred = split_inchi(inchi_pred)
    assert "/" in layers_true, "Molecular formula layer missing from InChI string"
    # ``layer``：str，逐次取需要比较的 InChI 层前缀。
    for layer in layers:
        # ``name``：str，当前层在返回结果中的稳定叶字段名。
        name = layer_names[layer]
        if (layer not in layers_true) or (layer not in layers_pred):
            # ``results[name]``：bool，任一 InChI 缺层时仅在两者都缺该层的情况下记为相同。
            results[name] = (layer not in layers_true) and (layer not in layers_pred)
        else:
            # ``results[name]``：bool，两者都含当前层时直接比较不含前缀的层正文。
            results[name] = layers_true[layer] == layers_pred[layer]

    # ``results.stereo_tetrahedral``：bool，四面体相关比较叶的合取；缺失叶按真处理。
    results["stereo_tetrahedral"] = all(results.get(name, True) for name in stereo_tetrahedral_layers)
    # ``results.stereo``：bool，双键与四面体全部立体比较叶的合取；缺失叶按真处理。
    results["stereo"] = all(results.get(name, True) for name in stereo_all_layers)

    return results


def standardize_and_get_inchi(mol: Mol, options: str = "", log_level=None, warnings_as_errors=False):
    """标准化分子，并在有三维坐标时从坐标重建立体标记后生成 InChI。

    输入参数:
        - mol: RDKit Mol，待标准化分子；函数内部深拷贝，不修改调用方对象。
        - options: str，原样传给 RDKit ``MolToInchi`` 的选项字符串。
        - log_level: 可选日志级别，原样传给 RDKit InChI 生成器。
        - warnings_as_errors: bool，是否把 RDKit InChI 警告视为错误。

    返回值:
        - inchi: str，标准化分子的 InChI 字符串。
    """
    # ``mol``：RDKit Mol，输入对象的独立副本，后续去同位素、氢、电荷与立体修改不会污染调用方。
    mol = deepcopy(mol)
    # ``mol``：完成首次 RDKit sanitize 的同一副本；失败时由 ``assert_sanity`` 抛出异常。
    mol = assert_sanity(mol)

    # ``mol``：逐原子清零同位素编号后的副本。
    mol = remove_isotopic_info(mol)

    # ``has_pose``：bool，是否至少存在一个 conformer，从而可以由三维坐标重新指定立体化学。
    has_pose = mol.GetNumConformers() > 0
    if has_pose:
        RemoveStereochemistry(mol)

    # ``mol``：删除显式氢后的新 RDKit Mol，重原子索引由 RDKit 重建。
    mol = RemoveHs(mol)
    try:
        # ``mol``：按局部形式电荷增减氢并中和后的同一对象。
        mol = neutralize_atoms(mol)
    except AtomValenceException:
        logger.warning("Failed to neutralize molecule. Using uncharger. InChI check might fail.")
        # ``mol``：局部中和失败时由 RDKit Uncharger 回退生成的分子。
        mol = Uncharger().uncharge(mol) 
    # ``mol``：补回立体判定所需氢后的分子。
    mol = add_stereo_hydrogens(mol)

    if has_pose:
        AssignStereochemistryFrom3D(mol, replaceExistingTags=True)

    with CaptureLogger():
        # ``inchi``：str，RDKit 根据标准化分子与 ``options`` 生成的 InChI。
        inchi = MolToInchi(mol, options=options, logLevel=log_level, treatWarningAsError=warnings_as_errors)

    return inchi


def is_valid_inchi(inchi: str) -> bool:
    """检查 InChI 能否被 RDKit 重新解析并通过 sanitize。

    输入参数:
        - inchi: str，待验证的 InChI 字符串。

    返回值:
        - valid: bool，解析结果非空且 RDKit sanitize 成功时为真；任意异常均返回假。
    """
    try:
        # ``mol``：RDKit Mol|None，由 InChI 重新解析的分子。
        mol = MolFromInchi(inchi)
        assert_sanity(mol)
        assert mol is not None
        return True
    except Exception:
        return False


def split_inchi(inchi: str):
    """把不含同位素层的标准 InChI 拆成前缀到层内容的映射。

    输入参数:
        - inchi: str，必须以 ``InChI=`` 开头的标准 InChI。

    返回值:
        - inchi_parts: dict[str, str]，``=`` 叶保存版本，``/``、``/c`` 等叶保存对应层正文且不含前缀。

    异常:
        - ValueError: 字符串不以 ``InChI=`` 开头。
        - AssertionError: 同一个层前缀在字符串中重复出现。
    """
    if not inchi.startswith("InChI="):
        raise ValueError("InChI string must start with 'InChI='")

    # inchi always InChi=1S/...formula.../...layer.../...layer.../...layer...
    # ``version``：str，去掉 ``InChI=`` 后、首个斜杠前的版本标识，例如 ``1S``。
    version = inchi[6:].split(r"/", 1)[0]
    # ``layers``：list[tuple[str, str]]，正则提取的 ``(层前缀, 层正文)``，按原字符串顺序排列。
    layers = findall(r"(?=.*)(\/[a-z]{0,1})(.*?)(?=\/.*|$)", inchi[6:])

    # ``inchi_parts``：dict[str, str]，先以特殊键 ``=`` 保存版本，再逐层写入。
    inchi_parts = {"=": version}
    # ``prefix``：str，逐次取当前 InChI 层前缀。
    # ``layer``：str，与当前 ``prefix`` 配对的层正文。
    for prefix, layer in layers:
        # standard inchi strings without isotopic info have each layer no more than once
        assert prefix not in inchi_parts, f"Layer {prefix} more than once!"
        # ``inchi_parts[prefix]``：str，当前 InChI 层正文，不含 ``/`` 或 ``/x`` 前缀。
        inchi_parts[prefix] = layer

    return inchi_parts


###### mol tools

def assert_sanity(mol: Mol):
    """断言 RDKit 分子能够完成 sanitize，并返回同一对象。

    输入参数:
        - mol: RDKit Mol，待执行 ``SanitizeMol`` 的分子对象。

    返回值:
        - mol: RDKit Mol，sanitize 成功后的同一对象。

    异常:
        - AssertionError: RDKit 返回的 sanitize flags 不为 0。
    """
    # ``flags``：SanitizeFlags，RDKit sanitize 返回的位掩码；0 表示全部检查成功。
    flags = SanitizeMol(mol)
    assert flags == 0, f"Sanitization failed with flags {flags}"
    return mol


def remove_isotopic_info(mol: Mol):
    """逐原子清除同位素编号，并返回同一 RDKit 分子对象。

    输入参数:
        - mol: RDKit Mol，待清除同位素标记的分子。

    返回值:
        - mol: RDKit Mol，所有原子的 isotope 已原地设为 0。
    """
    # ``atom``：RDKit Atom，逐次取当前分子的一个原子并原地清零同位素编号。
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return mol


def neutralize_atoms(mol: Mol):
    """通过修改形式电荷与显式氢数逐原子中和可匹配的局部电荷。

    输入参数:
        - mol: RDKit Mol，待中和分子；函数忽略整体电荷守恒并原地修改匹配原子。

    返回值:
        - mol: RDKit Mol，完成局部中和后的同一对象。
    """
    # https://www.rdkit.org/docs/Cookbook.html#neutralizing-charged-molecules
    # stronger than rdkit.Chem.MolStandardize.rdMolStandardize.Uncharger
    # ``pattern``：RDKit Mol，由 SMARTS 构造的可局部中和正/负一价原子查询。
    pattern = MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    # ``at_matches``：tuple[tuple[int]]，每项是一个匹配原子索引元组。
    at_matches = mol.GetSubstructMatches(pattern)
    # ``at_matches_list``：list[int]，展开为待中和的 RDKit 原子索引。
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        # ``at_idx``：int，逐次取当前待中和的 RDKit 原子索引。
        for at_idx in at_matches_list:
            # ``atom``：RDKit Atom，索引为 ``at_idx`` 的可变原子对象。
            atom = mol.GetAtomWithIdx(at_idx)
            # ``chg``：int，修改前的形式电荷；正值会减少显式氢，负值会增加显式氢。
            chg = atom.GetFormalCharge()
            # ``hcount``：int，修改前的总氢数。
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol

def add_stereo_hydrogens(mol: Mol):
    """为立体化学判定补氢，但排除伯酮亚胺氮上的补氢。

    输入参数:
        - mol: RDKit Mol，待补充显式氢及其坐标的分子。

    返回值:
        - mol: RDKit Mol，由 ``AddHs`` 返回的新分子，只在选中的非氢原子上补氢。
    """
    # ``exclude``：set[int]，匹配 ``[CX3]=[NH1]`` 的氮原子索引，避免在伯酮亚胺氮上补氢。
    exclude = {match[1] for match in mol.GetSubstructMatches(MolFromSmarts("[CX3]=[NH1]"))}
    # ``atoms``：list[int]，允许 ``AddHs`` 补氢的非氢原子索引，并排除 ``exclude``。
    atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1 if a.GetIdx() not in exclude]
    # ``mol``：RDKit Mol，补充显式氢并生成相应坐标后的新对象。
    mol = AddHs(mol, onlyOnAtoms=atoms, addCoords=True)
    return mol




"""Logging utilities."""

import os
import sys

import rdkit
from rdkit import rdBase

# redirect logs to Python logger
rdBase.LogToPythonLogger()


# https://github.com/rdkit/rdkit/discussions/5435
class CaptureLogger(logging.Handler):
    """Helper class that captures Python logger output."""

    def __init__(self, module=None):
        """Initialize logger."""
        super().__init__(level=logging.NOTSET)
        self.logs = {}
        self.devnull = open(os.devnull, "w")
        rdkit.log_handler.setStream(self.devnull)
        rdkit.logger.addHandler(self)

    def __enter__(self):
        """Enter context manager."""
        return self.logs

    def __exit__(self, *args):
        """Exit context manager."""
        self.release()

    def handle(self, record):
        """Handle log record."""
        key = record.levelname
        val = self.format(record)
        self.logs[key] = self.logs.get(key, "") + val
        return False

    def release(self):
        """Release logger."""
        rdkit.log_handler.setStream(sys.stderr)
        rdkit.logger.removeHandler(self)
        self.devnull.close()
        return self.logs
