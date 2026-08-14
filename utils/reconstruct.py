"""PocketXMol Molecule Reconstruction Module.

This module handles reconstruction of 3D molecular structures from generated
graph representations. It converts predicted atom types, positions, and bonds
into valid RDKit molecules or PDB structures.

Key functions:
    - reconstruct_from_generated_with_edges: Main reconstruction for small molecules
    - reconstruct_pdb_from_generated_fold: Reconstruction for peptides to PDB format
    - set_rdmol_positions: Update RDKit molecule coordinates
    - add_bond_from_obmol: Bond order assignment using OpenBabel

The reconstruction process:
    1. Create base molecule from atom types and connectivity
    2. Set 3D coordinates
    3. Assign bond orders (using template or OpenBabel)
    4. Sanitize and validate final structure
"""

# Standard library imports
import itertools
import os
import re
import tempfile
from copy import deepcopy
from io import StringIO

# Third-party imports
import numpy as np
from Bio.PDB import Chain, Model, PDBParser, PDBIO, Structure
from openbabel import openbabel as ob
from rdkit import Geometry
from rdkit.Chem import AllChem as Chem
from scipy.spatial.distance import pdist, squareform

# Local imports
from process.process_torsional_info import get_mol_from_data
from utils.misc import TimeoutException, time_limit

# ``ptable``：Module-level constants
ptable = Chem.GetPeriodicTable()

def reconstruct_pdb_from_generated_fold(mol_info, check_atom=True, gt_path=''):
    """
    Reconstruct peptide PDB structure from generated for structure prediction tasks.
    
    Takes generated 3D coordinates and updates a ground truth PDB template.
    Used for peptide design and conformation generation tasks.
    
    Args:
        mol_info: Dictionary containing:
            - atom_pos: Generated 3D coordinates [N, 3]
            - element: Atomic numbers [N]
            - task: Task type ('conf' or 'dock')
            - db, data_id: Dataset identifiers for ground truth lookup
        check_atom: Whether to verify atom types match ground truth
        gt_path: Optional path to ground truth PDB file
        
    Returns:
        PDB structure object with updated coordinates
    """
    assert mol_info['task'] in ['conf', 'dock'], 'this func only recon pos'
    parser = PDBParser(QUIET=True)

    # get gt pdb file
    db = mol_info['db']
    data_id = mol_info['data_id']
    if os.path.exists(gt_path):
        pdb_struc = parser.get_structure(data_id, gt_path)
    else:
        gt_pdb_file = f'data/{db}/files/peptides/{data_id}_pep.pdb'
        if not os.path.exists(gt_pdb_file):
            raise Exception(f'gt pdb file not found: {gt_pdb_file}')
        pdb_struc = parser.get_structure(data_id, gt_pdb_file)
    
    # get generated pos
    xyz = mol_info['atom_pos'].tolist()
    atomic_nums = mol_info['element'].tolist()

    # change coordinates
    for i_atom, atom in enumerate(pdb_struc.get_atoms()):
        if check_atom:
            # check element
            assert ptable.GetAtomicNumber(atom.element) == atomic_nums[i_atom],\
                f'element not match: {atom.element} vs {atomic_nums[i_atom]}'
        atom.set_coord(xyz[i_atom])

    # rdmol
    pdb_block = StringIO()
    io = PDBIO()
    io.set_structure(pdb_struc)
    io.save(pdb_block)
    pdb_block = pdb_block.getvalue()
    rdmol = Chem.MolFromPDBBlock(pdb_block, )
    
    return pdb_struc, rdmol


def reconstruct_pdb_from_generated(mol_info, check_atom=True, gt_path=''):
    
    if mol_info['task'] in ['conf', 'dock']:
        try:
            return reconstruct_pdb_from_generated_fold(mol_info, check_atom=check_atom, gt_path=gt_path)
        except Exception as e:
            print('recon from pdb file failed:', e)
            pass
    # get rdmol first
    try:
        with time_limit(300):
            rdmol = reconstruct_from_generated_with_edges(mol_info, is_pep=True)
    except TimeoutException as e:
        print('Timeout for reconstructing mol')
        raise MolReconsError()
    
    # convert rdmol to pdb
    tmpfile = tempfile.NamedTemporaryFile(suffix='.sdf')
    Chem.MolToMolFile(rdmol, tmpfile.name)
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "pdb")
    obmol = ob.OBMol()
    obConversion.ReadFile(obmol, tmpfile.name)
    
    tmpfile = tempfile.NamedTemporaryFile(suffix='.pdb')
    obConversion.WriteFile(obmol, tmpfile.name)
    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    pdb_struc = parser.get_structure('tmp', tmpfile.name)
    
    pdb_struc = sort_residues(pdb_struc)

    return pdb_struc, rdmol


def sort_residues(pdb_struc):
    
    new_pdb_struc = Structure.Structure('tmp')
    for model in pdb_struc.get_models():
        new_model = Model.Model(model.id)
        for chain in model.get_chains():
            new_chain = Chain.Chain(chain.id)
            # sort residues
            all_residues = list(chain.get_residues())
            all_residues.sort(key=lambda x: x.id[1])
            for residue in all_residues:
                new_chain.add(residue)
            new_model.add(new_chain)
        new_pdb_struc.add(new_model)
    return new_pdb_struc


class MolReconsError(Exception):
    pass


def reachable_r(a,b, seenbonds):
    '''Recursive helper.'''

    for nbr in ob.OBAtomAtomIter(a):
        bond = a.GetBond(nbr).GetIdx()
        if bond not in seenbonds:
            seenbonds.add(bond)
            if nbr == b:
                return True
            elif reachable_r(nbr,b,seenbonds):
                return True
    return False


def reachable(a,b):
    '''Return true if atom b is reachable from a without using the bond between them.'''
    if a.GetExplicitDegree() == 1 or b.GetExplicitDegree() == 1:
        return False #this is the _only_ bond for one atom
    #otherwise do recursive traversal
    seenbonds = set([a.GetBond(b).GetIdx()])
    return reachable_r(a,b,seenbonds)


def forms_small_angle(a,b,cutoff=45):
    '''Return true if bond between a and b is part of a small angle
    with a neighbor of a only.'''

    for nbr in ob.OBAtomAtomIter(a):
        if nbr != b:
            degrees = b.GetAngle(a,nbr)
            if degrees < cutoff:
                return True
    return False


def make_obmol(xyz, atomic_numbers):
    mol = ob.OBMol()
    mol.BeginModify()
    atoms = []
    for xyz,t in zip(xyz, atomic_numbers):
        x,y,z = xyz
        # ch = struct.channels[t]
        atom = mol.NewAtom()
        atom.SetAtomicNum(t)
        atom.SetVector(x,y,z)
        atoms.append(atom)
    return mol, atoms


def connect_the_dots(mol, atoms, maxbond=4):
    '''Custom implementation of ConnectTheDots.  This is similar to
    OpenBabel's version, but is more willing to make long bonds 
    (up to maxbond long) to keep the molecule connected.  It also 
    attempts to respect atom type information from struct.
    atoms and struct need to correspond in their order
    Assumes no hydrogens or existing bonds.
    '''
    pt = Chem.GetPeriodicTable()

    if len(atoms) == 0:
        return

    mol.BeginModify()

    #just going to to do n^2 comparisons, can worry about efficiency later
    coords = np.array([(a.GetX(),a.GetY(),a.GetZ()) for a in atoms])
    dists = squareform(pdist(coords))
    # types = [struct.channels[t].name for t in struct.c]

    for (i,a) in enumerate(atoms):
        for (j,b) in enumerate(atoms):
            if a == b:
                break
            if dists[i,j] < 0.01:  #reduce from 0.4
                continue #don't bond too close atoms
            if dists[i,j] < maxbond:
                flag = 0
                # if indicators[i][ATOM_FAMILIES_ID['Aromatic']] and indicators[j][ATOM_FAMILIES_ID['Aromatic']]:
                    # print('Aromatic', ATOM_FAMILIES_ID['Aromatic'], indicators[i])
                    # flag = ob.OB_AROMATIC_BOND
                # if 'Aromatic' in types[i] and 'Aromatic' in types[j]:
                #     flag = ob.OB_AROMATIC_BOND
                mol.AddBond(a.GetIdx(),b.GetIdx(),1,flag)

    atom_maxb = {}
    for (i,a) in enumerate(atoms):
        #set max valance to the smallest max allowed by openbabel or rdkit
        #since we want the molecule to be valid for both (rdkit is usually lower)
        maxb = ob.GetMaxBonds(a.GetAtomicNum())
        maxb = min(maxb,pt.GetDefaultValence(a.GetAtomicNum())) 

        if a.GetAtomicNum() == 16: # sulfone check
            if count_nbrs_of_elem(a, 8) >= 2:
                maxb = 6

        # if indicators[i][ATOM_FAMILIES_ID['Donor']]:
        #     maxb -= 1 #leave room for hydrogen
        # if 'Donor' in types[i]:
        #     maxb -= 1 #leave room for hydrogen
        atom_maxb[a.GetIdx()] = maxb
    
    #remove any impossible bonds between halogens
    for bond in ob.OBMolBondIter(mol):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if atom_maxb[a1.GetIdx()] == 1 and atom_maxb[a2.GetIdx()] == 1:
            mol.DeleteBond(bond)

    def get_bond_info(biter):
        '''Return bonds sorted by their distortion'''
        bonds = [b for b in biter]
        binfo = []
        for bond in bonds:
            bdist = bond.GetLength()
            #compute how far away from optimal we are
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            ideal = ob.GetCovalentRad(a1.GetAtomicNum()) + ob.GetCovalentRad(a2.GetAtomicNum()) 
            stretch = bdist-ideal
            binfo.append((stretch,bdist,bond))
        binfo.sort(reverse=True, key=lambda t: t[:2]) #most stretched bonds first
        return binfo

    #prioritize removing hypervalency causing bonds, do more valent 
    #constrained atoms first since their bonds introduce the most problems
    #with reachability (e.g. oxygen)
    # hypers = sorted([(atom_maxb[a.GetIdx()],a.GetExplicitValence() - atom_maxb[a.GetIdx()], a) for a in atoms],key=lambda aa: (aa[0],-aa[1]))
    # for mb,diff,a in hypers:
    #     if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
    #         continue
    #     binfo = get_bond_info(ob.OBAtomBondIter(a))
    #     for stretch,bdist,bond in binfo:
    #         #can we remove this bond without disconnecting the molecule?
    #         a1 = bond.GetBeginAtom()
    #         a2 = bond.GetEndAtom()

    #         #get right valence
    #         if a1.GetExplicitValence() > atom_maxb[a1.GetIdx()] or \
    #             a2.GetExplicitValence() > atom_maxb[a2.GetIdx()]:
    #             #don't fragment the molecule
    #             if not reachable(a1,a2):
    #                 continue
    #             mol.DeleteBond(bond)
    #             if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
    #                 break #let nbr atoms choose what bonds to throw out


    binfo = get_bond_info(ob.OBMolBondIter(mol))
    #now eliminate geometrically poor bonds
    for stretch,bdist,bond in binfo:
        #can we remove this bond without disconnecting the molecule?
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        #as long as we aren't disconnecting, let's remove things
        #that are excessively far away (0.45 from ConnectTheDots)
        #get bonds to be less than max allowed
        #also remove tight angles, because that is what ConnectTheDots does
        if stretch > 0.45 or forms_small_angle(a1,a2) or forms_small_angle(a2,a1):
            #don't fragment the molecule
            if not reachable(a1,a2):
                continue
            mol.DeleteBond(bond)

    mol.EndModify()

def add_context(data):
    data.ligand_context_pos = data.ligand_pos
    data.ligand_context_element = data.ligand_element
    data.ligand_context_bond_index = data.ligand_bond_index
    data.ligand_context_bond_type = data.ligand_bond_type
    return data

periodic_table = Chem.GetPeriodicTable()
def create_sdf_string(mol_info):
    """把解码后的原子、坐标和键叶字段直接序列化为 V2000 mol block 字符串。

    输入参数:
        - mol_info: Mapping，解码后的单分子字段映射。
        - mol_info.atom_pos: ndarray，形状为 (N, 3)，N 个生成原子的坐标，单位 Å，顺序与 ``element`` 对齐。
        - mol_info.element: ndarray，形状为 (N,)，N 个生成原子的原子序数。
        - mol_info.bond_index: ndarray，形状为 (2, E)，0-based 原子索引；允许同时含 ``i->j`` 与 ``j->i``。
        - mol_info.bond_type: ndarray，形状为 (E,)，与 ``bond_index`` 列对齐的 V2000 键型整数。
        - mol_info.atom_pos_masked: ndarray|None，可选，形状为 (M, 3)，用于轨迹占位的额外坐标；每个占位原子写成 He 且不添加键。

    返回值:
        - sdf_string: str，一个不含 ``$$$$`` 记录分隔符的 V2000 mol block；坐标保留四位小数，键端点转换为 1-based 索引。

    注意:
        - 该函数不执行 RDKit sanitize；采样重建失败和轨迹导出仍可用它保留模型的原始离散预测。
    """
    # ``xyz``：ndarray，形状为 (N, 3)，待写入 atom block 的生成坐标，单位 Å。
    xyz = mol_info['atom_pos']
    # ``atomic_nums``：ndarray，形状为 (N,)，逐生成原子的原子序数。
    atomic_nums = mol_info['element']
    # ``elements``：list[str]，长度为 N，把每个原子序数映射为 V2000 使用的元素符号。
    elements = [periodic_table.GetElementSymbol(int(z)) for z in atomic_nums]
    # ``bond_index``：ndarray，形状为 (2, E)，可能含双向重复的 0-based 生成原子索引。
    bond_index = mol_info['bond_index']
    # ``bond_type``：ndarray，形状为 (E,)，与 ``bond_index`` 每列对齐的键型整数。
    bond_type = mol_info['bond_type']
    # ``n_atoms``：int，当前 atom block 的原子行数，初始为生成原子数 N。
    n_atoms = len(atomic_nums)
    
    if 'atom_pos_masked' in mol_info:
        # ``pos_masked``：ndarray，形状为 (M, 3)，仅为可视化保留的被遮蔽原子位置，单位 Å。
        pos_masked = mol_info['atom_pos_masked']
        # ``xyz``：[N, 3] -> [N + M, 3]，在生成原子后追加遮蔽占位坐标。
        xyz = np.concatenate([xyz, pos_masked], axis=0)
        # ``elements``：list[str]，长度从 N 扩为 N+M，追加 M 个 He 占位符。
        elements = elements + ['He']*len(pos_masked)
        # ``n_atoms``：int，从 N 更新为 N+M，与扩展后的 atom block 对齐。
        n_atoms = n_atoms + len(pos_masked)
    
    # ``non_symmetric``：ndarray，dtype 为 bool，形状为 (E,)，只保留端点满足 ``i<j`` 的一个方向以去除双向重复键。
    non_symmetric = bond_index[0] < bond_index[1]
    # ``bond_index``：[2, E] -> [2, U]，U 条去重无向键的 0-based 端点索引。
    bond_index = bond_index[:, non_symmetric]
    # ``bond_type``：[E] -> [U]，按相同掩码保留的无向键类型。
    bond_type = bond_type[non_symmetric]
    # ``n_bonds``：int，V2000 bond block 中的去重无向键行数 U。
    n_bonds = len(bond_type)
    
    # ``header``：str，V2000 mol block 的三行头部；第二行记录生成函数名。
    header = '\n Created by Python create_sdf_string function\n\n'
    # ``counts_line``：str，固定 V2000 格式的原子数、键数与其余零值计数字段。
    counts_line = f'{n_atoms:3}{n_bonds:3}  0  0  0  0  0  0  0  0999 V2000\n'
    # ``atoms_block``：str，逐原子累积的 V2000 atom block，初始为空。
    atoms_block = ""
    # ``i``：int，当前 atom block 的 0-based 行号，同时索引 ``elements``。
    # ``x``：float，当前原子 X 坐标，单位 Å。
    # ``y``：float，当前原子 Y 坐标，单位 Å。
    # ``z``：float，当前原子 Z 坐标，单位 Å。
    for i, (x, y, z) in enumerate(xyz):
        atoms_block += f'{x:10.4f}{y:10.4f}{z:10.4f} {elements[i]:2}  ' + '  '.join(['0']*12) + '\n'
    # ``bonds_block``：str，逐键累积的 V2000 bond block，初始为空。
    bonds_block = ""
    # ``i``：int，当前键第一个端点的 0-based 原子索引；写出时加 1 转为 V2000 编号。
    # ``j``：int，当前键第二个端点的 0-based 原子索引；写出时加 1 转为 V2000 编号。
    # ``b``：int，当前键的 V2000 键型整数。
    for (i, j), b in zip(bond_index.T, bond_type):
        bonds_block += f'{i+1:3d}{j+1:3d}  {b}  0\n'
    # ``sdf_string``：str，拼接头部、计数、原子、键与 ``M  END`` 终止行后的完整 mol block。
    sdf_string = header + counts_line + atoms_block + bonds_block + 'M  END\n'
    return sdf_string


def reconstruct_pos(mol_info, in_mol=None):
    """在固定二维拓扑的 RDKit 分子副本上仅覆盖生成坐标。

    输入参数:
        - mol_info: Mapping，解码后的单分子字段映射。
        - mol_info.atom_pos: ndarray，形状为 (N, 3)，生成坐标，单位 Å；``FeaturizeMol.decode_output`` 已加回 ``pocket_center``，可直接写入 SDF 原始坐标系。
        - mol_info.db: str，可选定位叶；``in_mol is None`` 时供 ``get_mol_from_data`` 选择数据目录。
        - mol_info.data_id: str，可选定位叶；``in_mol is None`` 时供 ``get_mol_from_data`` 选择文件名或事务记录。
        - in_mol: RDKit Mol|None；提供时直接深拷贝，原对象及其 conformer 不被修改；为空时依据定位叶从 ``data`` 根目录加载。

    返回值:
        - mol: RDKit Mol，原子顺序、键、形式电荷和立体化学沿用输入二维图，第 0 个 conformer 的 N 个坐标被逐原子替换。

    异常:
        - AssertionError: 原分子原子数与 ``atom_pos`` 第一维不相等。
    """

    # data_id = mol_info['data_id']
    # db = mol_info['db']
    if in_mol is None:
        # ``mol``：RDKit Mol，由数据定位字段恢复，含固定拓扑和至少一个 conformer。
        mol = get_mol_from_data(mol_info, root_dir='data')
    else:
        # ``mol``：调用方输入分子的独立副本，后续坐标写入不会污染原对象。
        mol = deepcopy(in_mol)
    # ``pos``：list[list[float]]，形状为 (N, 3)，单位 Å，与 RDKit 原子索引顺序对齐。
    pos = mol_info['atom_pos'].tolist()
    assert mol.GetNumAtoms() == len(pos), 'num atoms do not match gen pos'
    # ``conf``：RDKit Conformer 0；原地持有当前 mol 的三维坐标。
    conf = mol.GetConformer(0)
    # ``i``：int，当前 0-based RDKit 原子索引。
    # ``xyz``：list[float]，长度为 3，当前原子的生成坐标，单位 Å。
    for i, xyz in enumerate(pos):
        conf.SetAtomPosition(i, Geometry.Point3D(*xyz))
    
    return mol


def reconstruct_from_generated_with_edges(mol_info, check_validity=True, add_edge=None, in_mol=None, is_pep=False):
    """把解码字段恢复为 RDKit 分子，优先复用构象/docking 的固定二维图。

    输入参数:
        - mol_info: Mapping，解码后的单分子字段映射。
        - mol_info.task: str；``conf`` 或 ``dock`` 首先走只替换坐标的固定拓扑短路分支。
        - mol_info.atom_pos: ndarray，形状为 (N, 3)，已回到原始输出坐标系的生成坐标，单位 Å。
        - mol_info.element: ndarray，形状为 (N,)，通用预测拓扑分支使用的原子序数。
        - mol_info.bond_index: ndarray|None，形状为 (2, E)，通用分支的 0-based 预测键端点；可含双向边，只有 ``i<j`` 一侧写入 RDKit。
        - mol_info.bond_type: ndarray|None，形状为 (E,)，与 ``bond_index`` 列对齐；1、2、3、4 分别表示单、双、三、芳香键。
        - mol_info.db: str，可选定位叶；固定拓扑且 ``in_mol is None`` 时用于恢复原始 RDKit 分子。
        - mol_info.data_id: str，可选定位叶；固定拓扑且 ``in_mol is None`` 时用于恢复原始 RDKit 分子。
        - check_validity: bool，通用预测拓扑分支是否运行 RDKit sanitize 与修复流程。
        - add_edge: str|None，缺少预测键时的补键后端；当前只有 ``openbabel`` 分支被实现。
        - in_mol: RDKit Mol|None，构象/docking 推理入口传入的固定拓扑分子。
        - is_pep: bool，小分子构象与 docking 路径为假。

    返回值:
        - mol: RDKit Mol；构象与 docking 路径等价于 ``reconstruct_pos``，二维图完全来自输入分子，模型只决定三维坐标。

    注意:
        - ``conf/dock`` 固定拓扑短路被宽泛 ``except Exception`` 包围；加载或原子数校验失败时会打印错误并退回通用预测拓扑重建，这里只记录既有行为。
    """

    if mol_info['task'] in ['conf', 'dock']:
        try:
            return reconstruct_pos(mol_info, in_mol)
        except Exception as e:
            print('Failed to use reconstruct_pos:', e)
            pass
    
    # ``xyz``：list[list[float]]，形状为 (N, 3)，通用建图分支的原子坐标，单位 Å。
    xyz = mol_info['atom_pos'].tolist()
    # ``atomic_nums``：list[int] 长度 N，逐原子的元素原子序数。
    atomic_nums = mol_info['element'].tolist()
    if 'bond_index' not in mol_info:
        if add_edge == 'openbabel':
            try:
                return reconstruct_from_generated(mol_info)
            except:
                raise MolReconsError()
        # elif add_edge == 'edm':
        #     bond_index, bond_type = predict_bonds(atomic_nums, np.array(xyz))
        else:
            raise ValueError('add_edge must be openbabel or edm')
    else:
        # ``bond_index``：list[list[int]]，形状为 (2, E)，可能同时含两个方向。
        bond_index = mol_info['bond_index'].tolist()
        # ``bond_type``：list[int] 长度 E，与 bond_index 的列顺序对齐。
        bond_type = mol_info['bond_type'].tolist()
    # ``n_atoms``：int N，用于初始化 RDKit conformer 的固定原子容量。
    n_atoms = len(atomic_nums)

    # ``rd_mol``：可编辑 RWMol，逐原子/逐键构建预测拓扑。
    rd_mol = Chem.RWMol()
    # ``rd_conf``：包含 N 个位置槽的 RDKit Conformer。
    rd_conf = Chem.Conformer(n_atoms)
    
    # ``i``：int，当前新建原子的 0-based RDKit 索引，同时索引 ``xyz`` 第一维。
    # ``atom``：int，当前新建原子的原子序数。
    for i, atom in enumerate(atomic_nums):
        # ``rd_atom``：以原子序数构造的 RDKit Atom；未在此恢复额外原子属性。
        rd_atom = Chem.Atom(atom)
        rd_mol.AddAtom(rd_atom)
        # ``rd_coords``：第 i 个原子的 Geometry.Point3D，单位 Å。
        rd_coords = Geometry.Point3D(*xyz[i])
        rd_conf.SetAtomPosition(i, rd_coords)
    rd_mol.AddConformer(rd_conf)
    
    # ``i``：int，当前预测键在 ``bond_type`` 和 ``bond_index`` 中的对齐索引。
    # ``type_this``：int，当前预测键类别；1/2/3/4 对应单/双/三/芳香键。
    for i, type_this in enumerate(bond_type):
        # ``node_i``：int，当前预测键第一个端点的 0-based RDKit 原子索引。
        # ``node_j``：int，当前预测键第二个端点的 0-based RDKit 原子索引。
        node_i, node_j = bond_index[0][i], bond_index[1][i]
        if node_i < node_j:
            if type_this == 1:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.SINGLE)
            elif type_this == 2:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.DOUBLE)
            elif type_this == 3:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.TRIPLE)
            elif type_this == 4:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.AROMATIC)
            else:
                raise Exception('unknown bond order {}'.format(type_this))
    
    
    # ``mol``：从 RWMol 固化出的 RDKit Mol，尚未保证 sanitize 成功。
    mol = rd_mol.GetMol()
    if check_validity:
        try:
            Chem.SanitizeMol(deepcopy(mol))
            # ``fixed``：bool；深拷贝 sanitize 成功表示当前预测拓扑已可接受。
            fixed = True
            Chem.SanitizeMol(mol)
        except Exception as e:
            # ``fixed``：``fixed=False`` 触发后续 kekulize、价态和芳香性修复。
            fixed = False
        # TODO: ok but not good solution. https://github.com/rdkit/rdkit/wiki/FrequentlyAskedQuestions
        if not fixed:
            try:
                Chem.Kekulize(deepcopy(mol))
            except Chem.rdchem.KekulizeException as e:
                # ``err``：RDKit KekulizeException，仅检查消息是否为未 kekulize 原子。
                err = e
                if 'Unkekulized' in err.args[0]:
                    try:
                        with time_limit(300):
                            # ``mol``：RDKit Mol，宽松芳香性修复后的分子对象。
                            # ``fixed``：bool，宽松芳香性修复是否成功。
                            mol, fixed = fix_aromatic(mol)
                    except TimeoutException as e:
                        print('Timeout for reconstructing mol')
                        raise MolReconsError()

        if not is_pep:  # only for small mols, skip further fixing for peptides
            # valence error for N 
            if not fixed:
                try:
                    with time_limit(300):
                        # ``mol``：RDKit Mol，小分子价态修复后的分子对象。
                        # ``fixed``：bool，小分子价态修复是否成功。
                        mol, fixed = fix_valence(mol)
                except TimeoutException as e:
                    print('Timeout for reconstructing mol')
                    raise MolReconsError()
                
            # print('s2')
            if not fixed:
                try:
                    with time_limit(300):
                        # ``mol``：RDKit Mol，严格芳香性修复后的分子对象。
                        # ``fixed``：bool，严格芳香性修复是否成功。
                        mol, fixed = fix_aromatic(mol, True)
                except TimeoutException as e:
                    print('Timeout for reconstructing mol')
                    raise MolReconsError()
            
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            raise MolReconsError()
            # return None
        
    # check valid
    # rd_mol_check = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    # if (rd_mol_check is None) and check_validity:
    #     raise MolReconsError()
    return mol


def get_ring_sys(mol):
    all_rings = Chem.GetSymmSSSR(mol)
    if len(all_rings) == 0:
        ring_sys_list = []
    else:
        ring_sys_list = [all_rings[0]]
        for ring in all_rings[1:]:
            form_prev = False
            for prev_ring in ring_sys_list:
                if set(ring).intersection(set(prev_ring)):
                    prev_ring.extend(ring)
                    form_prev = True
                    break
            if not form_prev:
                ring_sys_list.append(ring)
    ring_sys_list = [list(set(x)) for x in ring_sys_list]
    return ring_sys_list

def fix_valence(mol):
    mol = deepcopy(mol)
    fixed = False
    cnt_loop = 0
    while True:
        try:
            Chem.SanitizeMol(deepcopy(mol))
            fixed = True
            Chem.SanitizeMol(mol)
            break
        except Chem.rdchem.AtomValenceException as e:
            err = e
        except Exception as e:
            return mol, False # from HERE: rerun sample
        cnt_loop += 1
        if cnt_loop > 100:
            break
        N4_valence = re.compile(u"Explicit valence for atom # ([0-9]{1,}) N, 4, is greater than permitted")
        index = N4_valence.findall(err.args[0])
        if len(index) > 0:
            mol.GetAtomWithIdx(int(index[0])).SetFormalCharge(1)
        else:
            break
    return mol, fixed


def get_all_subsets(ring_list):
    all_sub_list = []
    for n_sub in range(len(ring_list)+1):
        all_sub_list.extend(itertools.combinations(ring_list, n_sub))
    return all_sub_list

def fix_aromatic(mol, strict=False):
    mol_orig = mol
    atomatic_list = [a.GetIdx() for a in mol.GetAromaticAtoms()]
    N_ring_list = []
    S_ring_list = []
    for ring_sys in get_ring_sys(mol):
        if set(ring_sys).intersection(set(atomatic_list)):
            idx_N = [atom for atom in ring_sys if mol.GetAtomWithIdx(atom).GetSymbol() == 'N']
            if len(idx_N) > 0:
                idx_N.append(-1) # -1 for not add to this loop
                N_ring_list.append(idx_N)
            idx_S = [atom for atom in ring_sys if mol.GetAtomWithIdx(atom).GetSymbol() == 'S']
            if len(idx_S) > 0:
                idx_S.append(-1) # -1 for not add to this loop
                S_ring_list.append(idx_S)
    # enumerate S
    fixed = False
    if strict:
        S_ring_list = [s for ring in S_ring_list for s in ring if s != -1]
        permutation = get_all_subsets(S_ring_list)
    else:
        permutation = list(itertools.product(*S_ring_list))
    for perm in permutation:
        mol = deepcopy(mol_orig)
        perm = [x for x in perm if x != -1]
        for idx in perm:
            mol.GetAtomWithIdx(idx).SetFormalCharge(1)
        try:
            if strict:
                mol, fixed = fix_valence(mol)
            Chem.SanitizeMol(deepcopy(mol))
            fixed = True
            Chem.SanitizeMol(mol)
            break
        except:
            continue
    # enumerate N
    if not fixed:
        if strict:
            N_ring_list = [s for ring in N_ring_list for s in ring if s != -1]
            permutation = get_all_subsets(N_ring_list)
        else:
            permutation = list(itertools.product(*N_ring_list))
        for perm in permutation:  # each ring select one atom
            perm = [x for x in perm if x != -1]
            # print(perm)
            actions = itertools.product([0, 1], repeat=len(perm))
            for action in actions: # add H or charge
                mol = deepcopy(mol_orig)
                for idx, act_atom in zip(perm, action):
                    if act_atom == 0:
                        mol.GetAtomWithIdx(idx).SetNumExplicitHs(1)
                    else:
                        mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                try:
                    if strict:
                        mol, fixed = fix_valence(mol)
                    Chem.SanitizeMol(deepcopy(mol))
                    fixed = True
                    Chem.SanitizeMol(mol)
                    break
                except:
                    continue
            if fixed:
                break
    return mol, fixed


########## the following for reconstruct_from_generated (auto add bond) ##########

def reconstruct_from_generated(mol_info):
    xyz = mol_info['atom_pos'].tolist()
    atomic_nums = mol_info['element'].tolist()
    # xyz = data.ligand_context_pos.clone().cpu().tolist()
    # atomic_nums = data.ligand_context_element.clone().cpu().tolist()
    # indicators = data.ligand_context_feature_full[:, -len(ATOM_FAMILIES_ID):].clone().cpu().bool().tolist()

    mol, atoms = make_obmol(xyz, atomic_nums)
    fixup(atoms, mol, )

    connect_the_dots(mol, atoms, 2)
    fixup(atoms, mol, )
    mol.EndModify()

    fixup(atoms, mol, )

    mol.AddPolarHydrogens()
    mol.PerceiveBondOrders()
    fixup(atoms, mol, )

    for (i,a) in enumerate(atoms):
        ob.OBAtomAssignTypicalImplicitHydrogens(a)
    fixup(atoms, mol, )

    mol.AddHydrogens()
    fixup(atoms, mol, )

    #make rings all aromatic if majority of carbons are aromatic
    for ring in ob.OBMolRingIter(mol):
        if 5 <= ring.Size() <= 6:
            carbon_cnt = 0
            aromatic_ccnt = 0
            for ai in ring._path:
                a = mol.GetAtom(ai)
                if a.GetAtomicNum() == 6:
                    carbon_cnt += 1
                    if a.IsAromatic():
                        aromatic_ccnt += 1
            if aromatic_ccnt >= carbon_cnt/2 and aromatic_ccnt != ring.Size():
                #set all ring atoms to be aromatic
                for ai in ring._path:
                    a = mol.GetAtom(ai)
                    a.SetAromatic(True)

    #bonds must be marked aromatic for smiles to match
    for bond in ob.OBMolBondIter(mol):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if a1.IsAromatic() and a2.IsAromatic():
            bond.SetAromatic(True)
            
    mol.PerceiveBondOrders()

    rd_mol = convert_ob_mol_to_rd_mol(mol)

    # Post-processing
    rd_mol = postprocess_rd_mol_1(rd_mol)
    rd_mol = postprocess_rd_mol_2(rd_mol)

    return rd_mol


def fixup(atoms, mol, ):
    '''Set atom properties to match channel.  Keep doing this
    to beat openbabel over the head with what we want to happen.'''

    mol.SetAromaticPerceived(True)  #avoid perception
    for i, atom in enumerate(atoms):
        # ch = struct.channels[t]
        # ind = indicators[i]

        # if ind[ATOM_FAMILIES_ID['Aromatic']]:
        #     atom.SetAromatic(True)
        #     atom.SetHyb(2)

        # if ind[ATOM_FAMILIES_ID['Donor']]:
        #     if atom.GetExplicitDegree() == atom.GetHvyDegree():
        #         if atom.GetHvyDegree() == 1 and atom.GetAtomicNum() == 7:
        #             atom.SetImplicitHCount(2)
        #         else:
        #             atom.SetImplicitHCount(1) 


        # elif ind[ATOM_FAMILIES_ID['Acceptor']]: # NOT AcceptorDonor because of else
        #     atom.SetImplicitHCount(0)   

        if (atom.GetAtomicNum() in (7, 8)) and atom.IsInRing():     # Nitrogen, Oxygen
            #this is a little iffy, ommitting until there is more evidence it is a net positive
            #we don't have aromatic types for nitrogen, but if it
            #is in a ring with aromatic carbon mark it aromatic as well
            acnt = 0
            for nbr in ob.OBAtomAtomIter(atom):
                if nbr.IsAromatic():
                    acnt += 1
            if acnt > 1:
                atom.SetAromatic(True)


def convert_ob_mol_to_rd_mol(ob_mol,struct=None):
    '''Convert OBMol to RDKit mol, fixing up issues'''
    ob_mol.DeleteHydrogens()
    n_atoms = ob_mol.NumAtoms()
    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)

    for ob_atom in ob.OBMolAtomIter(ob_mol):
        rd_atom = Chem.Atom(ob_atom.GetAtomicNum())
        #TODO copy format charge
        if ob_atom.IsAromatic() and ob_atom.IsInRing() and ob_atom.MemberOfRingSize() <= 6:
            #don't commit to being aromatic unless rdkit will be okay with the ring status
            #(this can happen if the atoms aren't fit well enough)
            rd_atom.SetIsAromatic(True)
        i = rd_mol.AddAtom(rd_atom)
        ob_coords = ob_atom.GetVector()
        x = ob_coords.GetX()
        y = ob_coords.GetY()
        z = ob_coords.GetZ()
        rd_coords = Geometry.Point3D(x, y, z)
        rd_conf.SetAtomPosition(i, rd_coords)

    rd_mol.AddConformer(rd_conf)

    for ob_bond in ob.OBMolBondIter(ob_mol):
        i = ob_bond.GetBeginAtomIdx()-1
        j = ob_bond.GetEndAtomIdx()-1
        bond_order = ob_bond.GetBondOrder()
        if bond_order == 1:
            rd_mol.AddBond(i, j, Chem.BondType.SINGLE)
        elif bond_order == 2:
            rd_mol.AddBond(i, j, Chem.BondType.DOUBLE)
        elif bond_order == 3:
            rd_mol.AddBond(i, j, Chem.BondType.TRIPLE)
        else:
            raise Exception('unknown bond order {}'.format(bond_order))

        if ob_bond.IsAromatic():
            bond = rd_mol.GetBondBetweenAtoms (i,j)
            bond.SetIsAromatic(True)

    rd_mol = Chem.RemoveHs(rd_mol, sanitize=False)

    pt = Chem.GetPeriodicTable()
    #if double/triple bonds are connected to hypervalent atoms, decrement the order

    positions = rd_mol.GetConformer().GetPositions()
    nonsingles = []
    for bond in rd_mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE or bond.GetBondType() == Chem.BondType.TRIPLE:
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            dist = np.linalg.norm(positions[i]-positions[j])
            nonsingles.append((dist,bond))
    nonsingles.sort(reverse=True, key=lambda t: t[0])

    for (d,bond) in nonsingles:
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        if calc_valence(a1) > pt.GetDefaultValence(a1.GetAtomicNum()) or \
           calc_valence(a2) > pt.GetDefaultValence(a2.GetAtomicNum()):
            btype = Chem.BondType.SINGLE
            if bond.GetBondType() == Chem.BondType.TRIPLE:
                btype = Chem.BondType.DOUBLE
            bond.SetBondType(btype)

    for atom in rd_mol.GetAtoms():
        #set nitrogens with 4 neighbors to have a charge
        if atom.GetAtomicNum() == 7 and atom.GetDegree() == 4:
            atom.SetFormalCharge(1)

    rd_mol = Chem.AddHs(rd_mol,addCoords=True)

    positions = rd_mol.GetConformer().GetPositions()
    center = np.mean(positions[np.all(np.isfinite(positions),axis=1)],axis=0)
    for atom in rd_mol.GetAtoms():
        i = atom.GetIdx()
        pos = positions[i]
        if not np.all(np.isfinite(pos)):
            #hydrogens on C fragment get set to nan (shouldn't, but they do)
            rd_mol.GetConformer().SetAtomPosition(i,center)

    try:
        Chem.SanitizeMol(rd_mol,Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
    except:
        raise MolReconsError()
    # try:
    #     Chem.SanitizeMol(rd_mol,Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
    # except: # mtr22 - don't assume mols will pass this
    #     pass
    #     # dkoes - but we want to make failures as rare as possible and should debug them
    #     m = pybel.Molecule(ob_mol)
    #     i = np.random.randint(1000000)
    #     outname = 'bad%d.sdf'%i
    #     print("WRITING",outname)
    #     m.write('sdf',outname,overwrite=True)
    #     pickle.dump(struct,open('bad%d.pkl'%i,'wb'))

    #but at some point stop trying to enforce our aromaticity -
    #openbabel and rdkit have different aromaticity models so they
    #won't always agree.  Remove any aromatic bonds to non-aromatic atoms
    for bond in rd_mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if bond.GetIsAromatic():
            if not a1.GetIsAromatic() or not a2.GetIsAromatic():
                bond.SetIsAromatic(False)
        elif a1.GetIsAromatic() and a2.GetIsAromatic():
            bond.SetIsAromatic(True)

    return rd_mol

UPGRADE_BOND_ORDER = {Chem.BondType.SINGLE:Chem.BondType.DOUBLE, Chem.BondType.DOUBLE:Chem.BondType.TRIPLE}

def postprocess_rd_mol_1(rdmol):

    rdmol = Chem.RemoveHs(rdmol)

    # Construct bond nbh list
    nbh_list = {}
    for bond in rdmol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx() 
        if begin not in nbh_list: nbh_list[begin] = [end]
        else: nbh_list[begin].append(end)
            
        if end not in nbh_list: nbh_list[end] = [begin]
        else: nbh_list[end].append(begin)

    # Fix missing bond-order
    for atom in rdmol.GetAtoms():
        idx = atom.GetIdx()
        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            for j in nbh_list[idx]:
                if j <= idx: continue
                nb_atom = rdmol.GetAtomWithIdx(j)
                nb_radical = nb_atom.GetNumRadicalElectrons()
                if nb_radical > 0:
                    bond = rdmol.GetBondBetweenAtoms(idx, j)
                    bond.SetBondType(UPGRADE_BOND_ORDER[bond.GetBondType()])
                    nb_atom.SetNumRadicalElectrons(nb_radical - 1)
                    num_radical -= 1
            atom.SetNumRadicalElectrons(num_radical)

        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            atom.SetNumRadicalElectrons(0)
            num_hs = atom.GetNumExplicitHs()
            atom.SetNumExplicitHs(num_hs + num_radical)
            
    return rdmol


def postprocess_rd_mol_2(rdmol):
    rdmol_edit = Chem.RWMol(rdmol)

    ring_info = rdmol.GetRingInfo()
    ring_info.AtomRings()
    rings = [set(r) for r in ring_info.AtomRings()]
    for i, ring_a in enumerate(rings):
        if len(ring_a) == 3:
            non_carbon = []
            atom_by_symb = {}
            for atom_idx in ring_a:
                symb = rdmol.GetAtomWithIdx(atom_idx).GetSymbol()
                if symb != 'C':
                    non_carbon.append(atom_idx)
                if symb not in atom_by_symb:
                    atom_by_symb[symb] = [atom_idx]
                else:
                    atom_by_symb[symb].append(atom_idx)
            if len(non_carbon) == 2:
                rdmol_edit.RemoveBond(*non_carbon)
            if 'O' in atom_by_symb and len(atom_by_symb['O']) == 2:
                rdmol_edit.RemoveBond(*atom_by_symb['O'])
                rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).SetNumExplicitHs(
                    rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).GetNumExplicitHs() + 1
                )
                rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).SetNumExplicitHs(
                    rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).GetNumExplicitHs() + 1
                )
    rdmol = rdmol_edit.GetMol()

    for atom in rdmol.GetAtoms():
        if atom.GetFormalCharge() > 0:
            atom.SetFormalCharge(0)

    return rdmol
    

def calc_valence(rdatom):
    '''Can call GetExplicitValence before sanitize, but need to
    know this to fix up the molecule to prevent sanitization failures'''
    cnt = 0.0
    for bond in rdatom.GetBonds():
        cnt += bond.GetBondTypeAsDouble()
    return cnt


def count_nbrs_of_elem(atom, atomic_num):
    '''
    Count the number of neighbors atoms
    of atom with the given atomic_num.
    '''
    count = 0
    for nbr in ob.OBAtomAtomIter(atom):
        if nbr.GetAtomicNum() == atomic_num:
            count += 1
    return count
