"""把外部小分子与蛋白结构解析为 PocketXMol 的原始口袋—配体字段。

本轮关注 ``scripts/sample_use.py`` 的真实 docking 输入链：``extract_pocket`` 先在受体世界坐标系
裁剪口袋，``get_pocmol_data`` 再解析小分子的全部 conformer 与口袋原子，
``get_input_from_file`` 最后补齐扭转、同构置换和片段分解字段。这里不执行模型中心化；
``utils.transforms.FeaturizePocket`` 随后才写入 ``pocket_center``，并把配体与口袋平移到共同局部坐标系。

同文件还包含肽构造与离线数据库预处理入口；这些分支不是本轮研究对象，只保留原行为。
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import torch
from rdkit.Chem import AllChem
import pickle
from Bio.PDB import PDBIO, PDBParser
from io import StringIO
from copy import deepcopy
from Bio.PDB.internal_coords import IC_Chain

from PeptideBuilder import Geometry
import PeptideBuilder

sys.path.append('.')
# from utils.dataset import LMDBDatabase
from utils.parser import parse_conf_list, PDBProtein, parse_pdb_peptide
from utils.data import torchify_dict, PocketMolData, Mol3DData
from process.process_torsional_info import get_torsional_info_mol
from process.process_decompose_info import decompose_brics, decompose_mmpa


def get_pdb_angles(stru, select_chain=None, angle_list=None, max_peptide_bond=None):
    """
    Input: Bio.PDB structure
    Output: dict of angles
    """
    if angle_list is None:
        angle_list = ["omg", "psi", "phi", "chi1", "chi2", "chi3", "chi4", 'chirality']
    

    if isinstance(stru, str):
        parser = PDBParser()
        new_stru = parser.get_structure(None, stru)[0]
    else:
        new_stru = deepcopy(stru)
    # print(new_stru)
    
    if max_peptide_bond is not None:
        IC_Chain.MaxPeptideBond = max_peptide_bond
        new_stru.internal_coord = None
    new_stru.atom_to_internal_coordinates(verbose=True)
    result = {}
    # for chain in new_stru:
    #     for residue in chain:
    for residue in new_stru.get_residues():
        chainid = residue.get_parent().id
        if select_chain is not None and chainid not in select_chain:
            continue
        curr_key = (chainid, residue.id, residue.resname)
        curr_result = {}
        if residue.id[0] != " ":
            print('Not support for pdb containing hetero residues yet.')
            return None
        for key in angle_list:
            tmp_v = residue.internal_coord.pick_angle(key)
            if tmp_v is not None:
                tmp_v = tmp_v.angle
            curr_result[key] = tmp_v
        result[curr_key] = curr_result
    # print(result)
    return result

def get_pdb_chirality(stru):
    if isinstance(stru, str):
        parser = PDBParser()
        new_stru = parser.get_structure(None, stru)[0]
    else:
        new_stru = deepcopy(stru)
    
    ch_list = []
    for residue in new_stru.get_residues():
        ch = calculate_chirality(residue)
        ch_list.append(ch)
    return ch_list

def vector_from_two_atoms(atom1, atom2):
    """Return the vector from atom1 to atom2."""
    return np.array(atom2) - np.array(atom1)

def calculate_chirality(residue):
    """
    Calculate the chirality of the residue.
    """
    if 'CB' not in residue: # Glycine
        return None
    N = residue['N'].get_vector().get_array()
    CA = residue['CA'].get_vector().get_array()
    CB = residue['CB'].get_vector().get_array()
    C = residue['C'].get_vector().get_array()
    
    # Calculate the cross product of the vectors
    cross = np.cross(vector_from_two_atoms(CA, N), vector_from_two_atoms(CA, C))
    # Calculate the dot product of the cross product and the vector from CA to CB
    angle = np.dot(cross, vector_from_two_atoms(CA, CB))
    return angle


def build_peptide(pep, return_pdbblock=True):
    for i, aa in enumerate(pep):
        geo = Geometry.geometry(aa)
        if i == 0:
            structure = PeptideBuilder.initialize_res(geo)
        else:
            PeptideBuilder.add_residue(structure, geo)
    if return_pdbblock:
        out = PDBIO()
        out.set_structure(structure)
        string = StringIO()
        out.save(string)
        pdb = string.getvalue()
        return pdb
    else:
        return structure


def add_pep_bb_data(data):
    num_atoms = data['num_atoms']
    num_res = num_atoms // 4
    peptide_data = {
        'pos': np.zeros([num_atoms, 3]),
        'atom_name': ['N', 'CA', 'C', 'O'] * num_res,
        'res_index': np.repeat(np.arange(num_res), 4),
        # 'atom_to_aa_type': ['X'] * num_atoms,
        'is_backbone': np.ones([num_atoms], dtype=bool),
        'pep_len': num_res,
    }
    peptide_data = {'peptide_'+k: v for k, v in peptide_data.items()}
    peptide_data = torchify_dict(peptide_data)
    return peptide_data


def get_make_mol_from_smiles(smiles, add_3D=True, center=None):
    mol = Chem.MolFromSmiles(smiles)
    if add_3D: # add 3D conformer
        mol = Chem.AddHs(mol)
        confid = AllChem.EmbedMolecule(mol, maxAttempts=5000)
        if confid != 0:
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
        AllChem.UFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
        conf = mol.GetConformer(0).GetPositions()
        if center is not None: # move to center
            conf = conf - np.mean(conf, axis=0) + center
        conf_new = mol.GetConformer()
        for i in range(conf.shape[0]):
            conf_new.SetAtomPosition(i, conf[i])
    return mol


def make_dummy_mol_with_coordinate(pos):
    return get_make_mol_from_smiles('C', add_3D=True, center=[pos])


def get_peptide_info(pep):
    used_keys = ['peptide_pos', 'peptide_atom_name', 'peptide_res_index',
                 'peptide_is_backbone', 'peptide_pep_len']
    if isinstance(pep, str):  # is pep path
        peptide_dict = parse_pdb_peptide(pep)
        peptide_dict = {'peptide_'+k: v for k, v in peptide_dict.items()}
        peptide_dict = torchify_dict(peptide_dict)
    else: # is pocmol data
        peptide_dict = add_pep_bb_data(pep)
    peptide_dict = {k: peptide_dict[k] for k in used_keys}
    return peptide_dict


def get_input_from_file(mol, pdb, data_id='', pdbid='', return_mol=False):
    """构造 use/docking 原始联合图，并补齐柔性几何与分解信息。

    输入参数:
        - mol: RDKit Mol|str，小分子对象、SDF/PDB 路径或 SMILES；SDF 可提供多个 conformer。
        - pdb: PDBProtein|str，已裁剪口袋解析器、PDB 文件路径或 PDB block；坐标保留受体世界坐标系。
        - data_id: str，样本标识；写入 ``PocketMolData.data_id``，也用于扭转异常定位。
        - pdbid: str，受体结构标识；写入 ``PocketMolData.pdbid``。
        - return_mol: bool，为真时同时返回去显式氢、原子顺序与图字段一致的 RDKit Mol。

    返回字段 ``pocmol_data``:
        - element: LongTensor，形状为 (N,)，配体原子序数。
        - pos_all_confs: FloatTensor，形状为 (C, N, 3)，配体全部合法 conformer 的世界坐标，单位 Å。
        - i_conf_list: list[int]，长度为 C，第 c 项是保留 conformer 在输入列表中的编号。
        - num_confs: int 标量 C，合法 conformer 数。
        - bond_index: LongTensor，形状为 (2, 2M)，按端点排序的双向化学键，数值索引 ``element`` 第一维。
        - bond_type: LongTensor，形状为 (2M,)，与 ``bond_index`` 列对齐；1/2/3/4 表示单/双/三/芳香键。
        - num_atoms: int 标量 N，配体原子数。
        - num_bonds: int 标量 M，只计每条无向化学键一次。
        - pocket_element: LongTensor，形状为 (P,)，口袋原子序数。
        - pocket_pos: FloatTensor，形状为 (P, 3)，口袋原子世界坐标，单位 Å。
        - pocket_is_backbone: BoolTensor，形状为 (P,)，真值表示口袋原子属于蛋白主链。
        - pocket_atom_name: list[str]，长度为 P，PDB 原子名。
        - pocket_atom_to_aa_type: LongTensor，形状为 (P,)，口袋原子所属氨基酸的 0-based 类别编号。
        - pocket_molecule_name: str|None，口袋 PDB ``HEADER`` 解析出的名称。
        - pdbid: str，调用方给定的受体标识。
        - data_id: str，调用方给定的样本标识。
        - smiles: str，去显式氢后 RDKit Mol 的规范 SMILES。
        - bond_rotatable: LongTensor，形状为 (2M,)，与双向 ``bond_index`` 对齐的可旋转键标记。
        - tor_twisted_pairs: dict[tuple[int, int], list[set[int], set[int]]]，规范方向可旋转键到断键两侧非轴原子的映射。
        - fixed_dist_torsion: FloatTensor，形状为 (N, N)，1 表示原子对距离不随内部扭转改变。
        - tor_bond_mat: Tensor，形状为 (N, N)，逐原子对可旋转键标记。
        - path_mat: FloatTensor，形状为 (N, N)，化学图最短路径长度，单位为键数。
        - nbh_dict: dict[int, list[int]]，每个原子编号到一跳邻居编号列表的映射。
        - matches_graph: LongTensor，形状为 (M_g, S_g)，忽略手性的压缩自同构映射。
        - matches_iso: LongTensor，形状为 (M_i, S_i)，考虑手性的压缩自同构映射。
        - brics.subgraphs: list[list[int]]，按原子数降序排列的 BRICS 片段原子编号。
        - brics.anchors_list: list[set[int]]，每个 BRICS 片段连接被切断键的锚原子编号。
        - brics.nbh_subgraphs: list[list[int]]，每个 BRICS 片段的相邻片段编号。
        - brics.connections: dict[tuple[int, int], tuple[int, int]]，有向片段对到两侧锚原子编号的映射。
        - mmpa.subgraphs: list[list[int]]，按原子数降序排列的 MMPA 片段原子编号。
        - mmpa.anchors_list: list[set[int]]，每个 MMPA 片段连接被切断键的锚原子编号。
        - mmpa.nbh_subgraphs: list[list[int]]，每个 MMPA 片段的相邻片段编号。
        - mmpa.connections: dict[tuple[int, int], tuple[int, int]]，有向片段对到两侧锚原子编号的映射。

    其他返回值:
        - mol: RDKit Mol，仅在 ``return_mol=True`` 时返回；已去显式氢，原子顺序与 ``element`` 对齐。
    """
    # ``pocmol_data``：PocketMolData，先含配体、口袋和身份叶；上述返回字段在函数末完整成立。
    # ``mol``：RDKit Mol，已去显式氢，原子顺序与 ``pocmol_data.element`` 第一维一致。
    pocmol_data, mol = get_pocmol_data(mol, pdb, data_id, pdbid, return_mol=True)
    
    # if 'torsional' in modes:
    # ``bond_index``：LongTensor，形状为 (2, 2M)，逐列端点索引配体 N 个原子。
    bond_index = pocmol_data['bond_index']
    # ``torsional_info.bond_rotatable``：ndarray，形状为 (2M,)，逐双向键可旋转标记。
    # ``torsional_info.tor_twisted_pairs``：dict，逐规范方向扭转键的两侧非轴原子集合。
    # ``torsional_info.fixed_dist_torsion``：ndarray，形状为 (N, N)，扭转下保持距离的 0/1 矩阵。
    # ``torsional_info.tor_bond_mat``：ndarray，形状为 (N, N)，可旋转键矩阵。
    # ``torsional_info.path_mat``：ndarray，形状为 (N, N)，化学图最短路径，单位为键数。
    # ``torsional_info.nbh_dict``：dict[int, list[int]]，逐原子一跳邻居。
    # ``torsional_info.matches_graph``：ndarray，形状为 (M_g, S_g)，忽略手性的压缩自同构映射。
    # ``torsional_info.matches_iso``：ndarray，形状为 (M_i, S_i)，考虑手性的压缩自同构映射。
    torsional_info = get_torsional_info_mol(mol, bond_index, data_id)
    pocmol_data.update(torsional_info)

    # if 'decompose' in modes:
    # ``decom_info.brics.subgraphs``：list[list[int]]，按原子数降序保存 BRICS 片段原子编号。
    # ``decom_info.brics.anchors_list``：list[set[int]]，逐 BRICS 片段保存被切断键的锚原子编号。
    # ``decom_info.brics.nbh_subgraphs``：list[list[int]]，逐 BRICS 片段保存相邻片段编号。
    # ``decom_info.brics.connections``：dict[tuple[int, int], tuple[int, int]]，有向 BRICS 片段对到两侧锚原子的映射。
    # ``decom_info.mmpa.subgraphs``：list[list[int]]，按原子数降序保存 MMPA 片段原子编号。
    # ``decom_info.mmpa.anchors_list``：list[set[int]]，逐 MMPA 片段保存被切断键的锚原子编号。
    # ``decom_info.mmpa.nbh_subgraphs``：list[list[int]]，逐 MMPA 片段保存相邻片段编号。
    # ``decom_info.mmpa.connections``：dict[tuple[int, int], tuple[int, int]]，有向 MMPA 片段对到两侧锚原子的映射。
    # ``decom_info``：dict，保存以上八个分解叶。
    decom_info = {
        # ``brics``：dict，保存以上四个 BRICS 叶。
        'brics': decompose_brics(mol),
        # ``mmpa``：dict，保存以上四个 MMPA 叶。
        'mmpa': decompose_mmpa(mol),
    }
    pocmol_data.update(decom_info)
    
    if not return_mol:
        return pocmol_data
    else:
        return pocmol_data, mol


def get_pocmol_data(mol, pdb, data_id='', pdbid='', return_mol=False):
    """解析小分子与口袋原子，并在同一世界坐标系中建立 ``PocketMolData``。

    输入参数:
        - mol: RDKit Mol|str，小分子对象、SDF/PDB 路径或 SMILES；本轮小分子 docking 不使用 ``peplen_*`` 等肽占位语法。
        - pdb: PDBProtein|str，口袋解析器、PDB 文件路径或 PDB block。
        - data_id: str，写入返回对象的样本标识。
        - pdbid: str，写入返回对象的受体标识。
        - return_mol: bool，为真时把解析后 RDKit Mol 作为第二返回值。

    返回字段 ``data``:
        - element: LongTensor，形状为 (N,)，配体原子序数。
        - pos_all_confs: FloatTensor，形状为 (C, N, 3)，配体 conformer 世界坐标，单位 Å。
        - i_conf_list: list[int]，长度为 C，保留 conformer 的输入编号。
        - num_confs: int 标量 C，合法 conformer 数。
        - bond_index: LongTensor，形状为 (2, 2M)，双向化学键端点。
        - bond_type: LongTensor，形状为 (2M,)，与 ``bond_index`` 列对齐的键类别。
        - num_atoms: int 标量 N，配体原子数。
        - num_bonds: int 标量 M，无向化学键数。
        - pocket_element: LongTensor，形状为 (P,)，口袋原子序数。
        - pocket_pos: FloatTensor，形状为 (P, 3)，口袋世界坐标，单位 Å。
        - pocket_is_backbone: BoolTensor，形状为 (P,)，口袋原子主链标记。
        - pocket_atom_name: list[str]，长度为 P，口袋 PDB 原子名。
        - pocket_atom_to_aa_type: LongTensor，形状为 (P,)，氨基酸类别编号。
        - pocket_molecule_name: str|None，口袋 PDB ``HEADER`` 名称。
        - pdbid: str，受体标识。
        - data_id: str，样本标识。
        - smiles: str，解析后小分子的规范 SMILES。

    其他返回值:
        - mol: RDKit Mol，仅在 ``return_mol=True`` 时返回；已去显式氢并与上述配体字段使用同一原子顺序。

    坐标边界:
        - 本函数不做中心化；``pos_all_confs`` 与 ``pocket_pos`` 都保留输入文件的世界坐标。
    """
    # load mol
    # ``mol_list``：list[RDKit Mol]|None，SDF 中全部 conformer 分子；非 SDF 输入稍后补成单元素列表。
    mol_list = None
    if isinstance(mol, str):
        if mol.endswith('.sdf'):
            # ``sd``：SDMolSupplier，按 SDF 记录顺序惰性读取候选 conformer。
            sd = Chem.SDMolSupplier(mol)
            # ``m``：RDKit Mol|None，``sd`` 当前惰性解析出的一个 SDF 记录。
            # ``mol_list``：list[RDKit Mol|None]，SDF 每条记录；后续 ``parse_conf_list`` 跳过不能解析的项。
            mol_list = [m for m in sd]
            # ``mol``：RDKit Mol|None，首条 SDF 记录，提供固定二维图与规范 SMILES。
            mol = mol_list[0]
            # mol = Chem.MolFromMolFile(mol)
        elif mol.endswith('.pdb'):
            # ``mol``：RDKit Mol|None，从配体 PDB 读取的单个三维小分子。
            mol = Chem.MolFromPDBFile(mol)
        # elif mol.endswith('peptide'): # pep design. make dummy pep with len
        elif mol.startswith('peplen_'): # pep design. make dummy pep with len
            n_res = int(mol.split('_')[1])
            mol = 'NCC(=O)' * n_res
            mol = get_make_mol_from_smiles(mol)
        # elif mol.startswith('peptide'):  # peptide sequence
        elif mol.startswith('cycpeplen_'): # pep design. make dummy cyclic pep with len
            n_res = int(mol.split('_')[1])
            mol = 'N1CC(=O)' + ('NCC(=O)' * (n_res-2)) + 'NCC1(=O)'
            mol = get_make_mol_from_smiles(mol)
        elif mol.startswith('pepseq_'):  # peptide sequence
            seq = mol.split('_')[1]
            pep_pdb = build_peptide(seq)
            mol = Chem.MolFromPDBBlock(pep_pdb)
        else: # smiles
            # ``mol``：RDKit Mol，由 SMILES 建图、嵌入三维 conformer、UFF 优化并去显式氢。
            mol = get_make_mol_from_smiles(mol)
    else:
        # ``mol``：RDKit Mol；既有非字符串分支会用单碳三维占位分子替换传入对象，本轮只记录该事实。
        mol = get_make_mol_from_smiles('C')
    # ``mol``：RDKit Mol，删除全部显式氢后作为配体固定二维图；原子顺序决定后续所有配体叶对齐。
    mol = Chem.RemoveAllHs(mol)
    if mol_list is None:
        # ``mol_list``：list[RDKit Mol]，非 SDF 输入只含当前去氢分子。
        mol_list = [mol]
    else:
        # ``m``：RDKit Mol，``mol_list`` 中当前待删除显式氢的一个 conformer 记录。
        # ``mol_list``：list[RDKit Mol]，逐条 SDF 记录删除显式氢；列表顺序保持不变。
        mol_list = [Chem.RemoveAllHs(m) for m in mol_list]
        
    # ``ligand_dict.element``：ndarray，形状为 (N,)，配体原子序数。
    # ``ligand_dict.bond_index``：ndarray，形状为 (2, 2M)，双向键端点。
    # ``ligand_dict.bond_type``：ndarray，形状为 (2M,)，逐双向键类别。
    # ``ligand_dict.pos_all_confs``：float32 ndarray，形状为 (C, N, 3)，世界坐标，单位 Å。
    # ``ligand_dict.num_atoms``：int 标量 N，配体原子数。
    # ``ligand_dict.num_bonds``：int 标量 M，无向键数。
    # ``ligand_dict.i_conf_list``：list[int]，长度为 C，合法输入 conformer 编号。
    # ``ligand_dict.num_confs``：int 标量 C，合法 conformer 数。
    ligand_dict = parse_conf_list(mol_list)
    if ligand_dict['num_confs'] == 0:
        raise ValueError('No conformers found')
    # ``smiles``：str，当前去氢固定二维图的规范 SMILES。
    smiles = Chem.MolToSmiles(mol)
    # load pdb
    if isinstance(pdb, str):
        # ``pdb``：PDBProtein，从文件路径或 PDB block 解析口袋原子与残基属性。
        pdb = PDBProtein(pdb)
    # ``pocket_dict.element``：int64 ndarray，形状为 (P,)，口袋原子序数。
    # ``pocket_dict.molecule_name``：str|None，PDB ``HEADER`` 解析出的名称。
    # ``pocket_dict.pos``：float32 ndarray，形状为 (P, 3)，口袋世界坐标，单位 Å。
    # ``pocket_dict.is_backbone``：Bool ndarray，形状为 (P,)，逐口袋原子主链标记。
    # ``pocket_dict.atom_name``：list[str]，长度为 P，PDB 原子名。
    # ``pocket_dict.atom_to_aa_type``：int64 ndarray，形状为 (P,)，逐原子氨基酸类别编号。
    pocket_dict = pdb.to_dict_atom()
    # make data
    # ``data``：PocketMolData，口袋叶添加 ``pocket_`` 前缀，配体叶保留原名；NumPy 数组先转为 Tensor。
    data = PocketMolData.from_pocket_mol_dicts(
        pocket_dict=torchify_dict(pocket_dict),
        mol_dict=torchify_dict(ligand_dict),
    )
    # ``data.pdbid``：str，调用方给定的受体标识。
    data.pdbid = pdbid
    # ``data.data_id``：str，调用方给定的样本标识。
    data.data_id = data_id
    # ``data.smiles``：str，固定二维配体图的规范 SMILES。
    data.smiles = smiles
    if not return_mol:
        return data
    else:
        return data, mol

def extract_pocket(protein_path, mol_path, radius=10, save_path=None, criterion='center_of_mass'):
    """按参考配体邻域从完整受体中裁剪口袋 PDB block。

    输入参数:
        - protein_path: str|None，完整受体 PDB 文件路径；None 时直接返回空字符串。
        - mol_path: RDKit Mol|str，参考配体对象、SDF 路径或 PDB 路径；必须含三维 conformer。
        - radius: float，任一配体原子邻域的残基选择半径，单位 Å。
        - save_path: str|None，可选口袋 PDB 落盘路径；None 时只返回文本。
        - criterion: str，残基距离代表；``center_of_mass`` 使用残基质心，``min`` 使用残基原子最小距离。

    返回值:
        - pocket_block: str，只含被选残基的 PDB 文本；坐标保持 ``protein_path`` 的世界坐标系。

    异常:
        - ValueError: 参考配体无法解析，或半径内没有任何残基。
    """
    if protein_path is None:
        return ''
    if isinstance(mol_path, Chem.Mol):
        # ``mol``：RDKit Mol，直接使用调用方参考配体及其 conformer。
        mol = mol_path
    elif mol_path.endswith('.sdf'):
        # ``mol``：RDKit Mol|None，从 SDF 首条记录读取参考配体。
        mol = Chem.MolFromMolFile(mol_path)
    elif mol_path.endswith('.pdb'):
        # ``mol``：RDKit Mol|None，从配体 PDB 读取参考配体。
        mol = Chem.MolFromPDBFile(mol_path)
    else:
        # ``mol``：None，不支持的字符串类型在下一分支抛出明确错误。
        mol = None
    if mol is None:
        raise ValueError('Invalid mol file for extracting pocket:', mol_path)
    # ``pdb``：PDBProtein，完整受体的原子、残基、质心和原始 PDB 行解析结果。
    pdb = PDBProtein(protein_path)
    # ``selected_pocket``：list[dict]，与任一配体原子距离小于 radius 的去重残基，保持受体解析顺序。
    # ``selected_pocket[r].name``：str，第 r 个口袋残基的三字母氨基酸名。
    # ``selected_pocket[r].atoms``：list[int]，第 r 个残基的原子编号，索引 ``pdb.pos`` 与 ``pdb.atoms`` 第一维。
    # ``selected_pocket[r].chain``：str，第 r 个残基的 PDB 链标识。
    # ``selected_pocket[r].segment``：str，第 r 个残基的 PDB segment 标识。
    # ``selected_pocket[r].chain_res_id``：str，第 r 个残基由链、segment、残基号与插入码拼成的唯一键。
    # ``selected_pocket[r].center_of_mass``：float32 ndarray，形状为 (3,)，第 r 个残基的质量中心世界坐标，单位 Å。
    # ``selected_pocket[r].pos_CA``：float32 ndarray，可选，形状为 (3,)，第 r 个残基 CA 原子的世界坐标，单位 Å。
    # ``selected_pocket[r].pos_C``：float32 ndarray，可选，形状为 (3,)，第 r 个残基 C 原子的世界坐标，单位 Å。
    # ``selected_pocket[r].pos_N``：float32 ndarray，可选，形状为 (3,)，第 r 个残基 N 原子的世界坐标，单位 Å。
    # ``selected_pocket[r].pos_O``：float32 ndarray，可选，形状为 (3,)，第 r 个残基 O 原子的世界坐标，单位 Å。
    selected_pocket = pdb.query_residues_ligand(mol, radius=radius, criterion=criterion)
    if len(selected_pocket) == 0:
        raise ValueError('Empty pocket within the radius. Please check your pocket_args configuration.')
    # ``pocket_block``：str，把所选残基的原始 ATOM 行串成独立 PDB 文本；坐标未平移，单位 Å。
    pocket_block = pdb.residues_to_pdb_block(selected_pocket)
    # pdb = PDBProtein(pocket_block)
    # save pocket
    if save_path is not None:
        # ``f``：文本文件句柄，接收与返回值完全相同的口袋 PDB block。
        with open(save_path, 'w') as f:
            f.write(pocket_block)
    return pocket_block


def process_raw(data_id='', mol_path=None, protein_path=None, pdbid='',
                modes=None, return_pocket=False, save_pocket=True, **kwargs):
    
    # load data
    if mol_path is None:  # used in denovo gen. only need pocket_center to define pocket
        mol = make_dummy_mol_with_coordinate(kwargs['pocket_center'])
    elif mol_path.endswith('.sdf'):
        mol = Chem.MolFromMolFile(mol_path)
    else:
        mol = Chem.MolFromSmiles(mol_path)
        # add 3D conformer
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)  # bug for large mol. see func: get_make_mol_from_smiles
        AllChem.UFFOptimizeMolecule(mol)
        # move to pocket center
        pocket_center = np.array(kwargs['pocket_center']).reshape([1, 3])
        mol = Chem.RemoveHs(mol)
        conf = mol.GetConformer(0).GetPositions()
        conf = conf - np.mean(conf, axis=0) + pocket_center
        conf_new = mol.GetConformer()
        for i in range(conf.shape[0]):
            conf_new.SetAtomPosition(i, conf[i])
        
    mol = Chem.RemoveAllHs(mol)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False)

    if modes is None:
        modes = ['extract_pocket', 'pocmol', 'torsional', 'decompose']
    
    if 'extract_pocket' in modes:
        pdb = PDBProtein(protein_path)
        if 'pocket_center' in kwargs:
            ref_mol = make_dummy_mol_with_coordinate(kwargs['pocket_center'])
        else:
            ref_mol = mol
        radius = int(kwargs.get('radius', 10))
        selected_pocket = pdb.query_residues_ligand(ref_mol,
                radius=radius,
                criterion=kwargs.get('criterion', 'center_of_mass'))
        if len(selected_pocket) == 0:
            raise ValueError('Empty pocket within the radius')
        pocket_block = pdb.residues_to_pdb_block(selected_pocket)
        pdb = PDBProtein(pocket_block)
        if save_pocket:
            # save pocket
            pocket_dir = os.path.join(os.path.dirname(os.path.dirname(protein_path)), f'pockets{radius}')
            os.makedirs(pocket_dir, exist_ok=True)
            pocket_path = os.path.join(pocket_dir, os.path.basename(protein_path).replace('_pro.pdb', '_pocket.pdb'))
            with open(pocket_path, 'w') as f:
                f.write(pocket_block)
    
    data = {}
    if 'pocmol' in modes:
        ligand_dict = parse_conf_list([mol])
        if ligand_dict['num_confs'] == 0:
            raise ValueError('No conformers found')
        pocket_dict = pdb.to_dict_atom()

        data = PocketMolData.from_pocket_mol_dicts(
            pocket_dict=torchify_dict(pocket_dict),
            mol_dict=torchify_dict(ligand_dict),
        )
        data.pdbid = pdbid
        data.data_id = data_id
        data.smiles = smiles
        
    if 'mols' in modes:
        # load mol with multiple conformers
        suppl = Chem.SDMolSupplier(mol_path)
        confs_list = []
        for i_conf in range(len(suppl)):
            conf = Chem.MolFromMolBlock(suppl.GetItemText(i_conf).replace(
                "RDKit          3D", "RDKit          2D"
            ))  # removeHs=True is default
            conf = Chem.RemoveAllHs(conf)
            confs_list.append(conf)
            
        ligand_dict = parse_conf_list(confs_list)  #NOTE: multiple conformation
        if ligand_dict['num_confs'] == 0:
            raise ValueError('No conformers found')
        ligand_dict = torchify_dict(ligand_dict)
        data = Mol3DData.from_3dmol_dicts(ligand_dict)
        data.smiles = smiles
        data.data_id = data_id
    
    if 'torsional' in modes:
        bond_index = data['bond_index']
        torsional_info = get_torsional_info_mol(mol, bond_index, data_id)
        data.update(torsional_info)

    if 'decompose' in modes:
        decom_info = {
            'brics': decompose_brics(mol),
            'mmpa': decompose_mmpa(mol),
        }
        data.update(decom_info)
    
    if return_pocket:
        return data, pocket_block
    return data


def process_raw_pep(protein_path, input_ligand_path,
                    input_pep_path=None,  # for pep info
                    ref_ligand_path=None,  # for pocket extraction
                   pocket_args={}, pocmol_args={}, return_pocket=False):
    # get pocket
    if ref_ligand_path is None:
        ref_ligand_path = input_ligand_path
    pocket_pdb = extract_pocket(protein_path, ref_ligand_path, **pocket_args)
    
    # get input ligand
    data_id = pocmol_args.get('data_id', '')
    pocmol_data, mol = get_pocmol_data(input_ligand_path, pocket_pdb, **pocmol_args,
                                  return_mol=True)
    # torsional
    bond_index = pocmol_data['bond_index']
    torsional_info = get_torsional_info_mol(mol, bond_index, data_id)
    pocmol_data.update(torsional_info)
    # decompose
    decom_info = {
        'brics': decompose_brics(mol),
        'mmpa': decompose_mmpa(mol),
    }
    pocmol_data.update(decom_info)
    
    # peptide info
    if input_pep_path is None:
        input_pep_path = pocmol_data
    pep_info = get_peptide_info(input_pep_path)
    pocmol_data.update(pep_info)
    assert torch.isclose(pocmol_data['pos_all_confs'][0], pep_info['peptide_pos'], 1e-2).all(), 'mol and pep atoms may not match'

    if return_pocket:
        return pocmol_data, pocket_pdb
    return pocmol_data
