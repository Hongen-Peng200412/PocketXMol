import os
import numpy as np
from rdkit import Chem
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from utils.fragment import find_rotatable_bond_mat
# from rdkit.Chem.rdchem import BondType
# from rdkit.Chem import ChemicalFeatures
# from rdkit import RDConfig

# from utils.mol2frag import mol2frag

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
# BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
# BOND_NAMES = {i: t for i, t in enumerate(BondType.names.keys())}


class PDBLigand(object):

    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, mode='auto', removeHs=True):
        super().__init__()
        # # Read PDB file
        self.removeHs = removeHs
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if (line[0:6].strip() == 'ATOM') or (line[0:6].strip() == 'HETATM'):
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': line[0:6].strip(),  # ATOM or HETATM
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break   # Some PDBs have more than 1 model.

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            if self.removeHs and atom['element_symb'] == 'H':
                continue
            
            nonstd = (atom['res_name'] not in self.AA_NAME_NUMBER)
            
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            next_ptr = len(self.element)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                    'chain_res_id': chain_res_id
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)

        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass
        
        # Process backbone atoms of residues
        for residue in self.residues:
            self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name   # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

    def to_dict_atom(self):
        return {
            'element': np.array(self.element, dtype=np.int64),
            'molecule_name': self.title,
            'pos': np.array(self.pos, dtype=np.float32),
            'is_backbone': np.array(self.is_backbone, dtype=bool),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)
        }

    def to_dict_residue(self):
        return {
            'amino_acid': np.array(self.amino_acid, dtype=np.int64),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(self, mol, radius, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        mol_pos = mol.GetConformer().GetPositions()
        for center in mol_pos:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected
    
    def get_pocket_info(self, selected):
        num_res = len(selected)
        num_atoms = sum([len(residue['atoms']) for residue in selected])
        res_ids = [residue['chain_res_id'] for residue in selected]
        cover_chains = sorted(list(set([residue['chain'] for residue in selected])))
        pocket_info = {
            'num_res': num_res,
            'num_atoms': num_atoms,
            'res_ids': ';'.join(res_ids),
            'num_chains': len(cover_chains),
            'cover_chain_ids': ';'.join(cover_chains),
        }
        return pocket_info

    def get_chain_seqs(self, chains):
        if isinstance(chains, str):
            chains = [chains]
        seqs = []
        for chain in chains:
            seq = []
            for residue in self.residues:
                if residue['chain'] == chain:
                    seq.append(self.AA_NAME_SYM[residue['name']])
            seqs.append(''.join(seq))
            
        return ';'.join(seqs)


    def residues_to_pdb_block(self, residues, name='POCKET'):
        block =  "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block
    
    
def parse_mol_with_confs(mol, smiles=None, confs=None):
    if smiles is not None: # check smiles
        if smiles != Chem.MolToSmiles(mol, isomericSmiles=False):
            return None
    pos_all_confs = []
    i_conf_list = []

    data = parse_3d_mol(mol, smiles=smiles, not_pos=True)
    if data is None:
        return None
    
    # get all confs
    if confs is None:
        for i_conf in range(mol.GetNumConformers()):
            pos = mol.GetConformer(i_conf).GetPositions()
            pos_all_confs.append(pos)
            i_conf_list.append(i_conf)
            # check bond length
            # bond_lengths = np.linalg.norm(pos[data['bond_index'][0]] - pos[data['bond_index'][1]], ord=2, axis=-1)
            # if (bond_lengths > 3.5).any():
            #     print('Skipping conformer with too long bond length: %s' % bond_lengths.max())
            #     return None
    else:
        pos_all_confs = np.array(confs)[:, :len(data['element']), :]
        i_conf_list = list(range(len(confs)))

    return {
        'element': np.array(data['element']),
        'bond_index': np.array(data['bond_index']),
        'bond_type': np.array(data['bond_type']),
        'pos_all_confs': np.array(pos_all_confs, dtype=np.float32),
        'num_atoms': data['num_atoms'],
        'num_bonds': data['num_bonds'],
        'i_conf_list': i_conf_list,
        'num_confs': len(i_conf_list),
    }

class PDBProtein(object):

    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, mode='auto', removeHs=True):
        super().__init__()
        self.removeHs = removeHs
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break   # Some PDBs have more than 1 model.

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            if self.removeHs and atom['element_symb'] == 'H':
                continue
            if atom['res_name'] not in self.AA_NAME_NUMBER:
                continue
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            next_ptr = len(self.element)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                    'chain_res_id': chain_res_id
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)

        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass
        
        # Process backbone atoms of residues
        for residue in self.residues:
            self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name   # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

    def to_dict_atom(self):
        """把已解析的蛋白重原子属性导出为逐原子对齐的字段映射。

        返回值:
            - atom_dict: dict，保留 PDB 世界坐标、元素和残基类别的蛋白原子字段。
            - atom_dict.element: int64 ndarray，形状为 (P,)；逐蛋白重原子的原子序数。
            - atom_dict.molecule_name: str|None，PDB ``HEADER`` 文本的小写形式；源文本没有 ``HEADER`` 时为 None。
            - atom_dict.pos: float32 ndarray，形状为 (P, 3)；逐蛋白重原子的 PDB 世界坐标，最后一维按 XYZ 排列，单位 Å。
            - atom_dict.is_backbone: bool ndarray，形状为 (P,)；True 表示对应原子名属于 ``CA/C/N/O``，与 ``pos`` 第一维逐原子对齐。
            - atom_dict.atom_name: list[str]，长度为 P；逐蛋白重原子的 PDB 原子名，与 ``pos`` 第一维逐原子对齐。
            - atom_dict.atom_to_aa_type: int64 ndarray，形状为 (P,)；逐蛋白重原子所属氨基酸的 0-based 类别编号，编号映射由 ``AA_NAME_NUMBER`` 定义。
        """
        return {
            # ``atom_dict.element``：int64 ndarray，形状为 (P,)；逐蛋白重原子的原子序数。
            'element': np.array(self.element, dtype=np.int64),
            # ``atom_dict.molecule_name``：str|None，PDB ``HEADER`` 文本的小写形式；源文本无 ``HEADER`` 时为 None。
            'molecule_name': self.title,
            # ``atom_dict.pos``：float32 ndarray，形状为 (P, 3)；逐蛋白重原子的 PDB 世界坐标，最后一维按 XYZ 排列，单位 Å。
            'pos': np.array(self.pos, dtype=np.float32),
            # ``atom_dict.is_backbone``：bool ndarray，形状为 (P,)；True 表示对应原子名属于 ``CA/C/N/O``。
            'is_backbone': np.array(self.is_backbone, dtype=bool),
            # ``atom_dict.atom_name``：list[str]，长度为 P；与 ``pos`` 第一维逐蛋白原子对齐的 PDB 原子名。
            'atom_name': self.atom_name,
            # ``atom_dict.atom_to_aa_type``：int64 ndarray，形状为 (P,)；逐蛋白原子的氨基酸类别编号，映射由 ``AA_NAME_NUMBER`` 定义。
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)
        }

    def to_dict_residue(self):
        return {
            'amino_acid': np.array(self.amino_acid, dtype=np.int64),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(self, mol, radius, criterion='center_of_mass'):
        """选择距参考配体任一原子小于给定半径的蛋白残基。

        输入参数:
            - mol: RDKit Mol，含一个三维 conformer；配体坐标必须与本蛋白的 PDB 世界坐标处于同一坐标系，单位 Å。
            - radius: float，残基代表坐标或残基原子到任一配体原子的严格距离上限，单位 Å。
            - criterion: str，``min`` 使用残基全部原子的最小距离；其他值作为残基坐标字段名读取，默认 ``center_of_mass``。

        返回值:
            - selected: list[dict]，满足半径条件的去重残基，顺序与 ``self.residues`` 一致。
            - selected[*].name: str，三字母氨基酸名。
            - selected[*].atoms: list[int]，残基重原子编号；数值索引 ``self.pos`` 与 ``self.atoms`` 第一维。
            - selected[*].chain: str，PDB 链标识。
            - selected[*].segment: str，PDB segment 标识。
            - selected[*].chain_res_id: str，由链、segment、残基号与插入码拼成的残基唯一键。
            - selected[*].center_of_mass: float32 ndarray，形状为 (3,)；残基质量中心的 PDB 世界坐标，按 XYZ 排列，单位 Å。
            - selected[*].pos_CA: float32 ndarray，形状为 (3,)；CA 原子的 PDB 世界坐标，缺失时回退到残基质量中心，单位 Å。
            - selected[*].pos_C: float32 ndarray，形状为 (3,)；C 原子的 PDB 世界坐标，缺失时回退到残基质量中心，单位 Å。
            - selected[*].pos_N: float32 ndarray，形状为 (3,)；N 原子的 PDB 世界坐标，缺失时回退到残基质量中心，单位 Å。
            - selected[*].pos_O: float32 ndarray，形状为 (3,)；O 原子的 PDB 世界坐标，缺失时回退到残基质量中心，单位 Å。
        """
        # ``selected``：list[dict]，按蛋白解析顺序收集首次落入配体邻域的残基。
        selected = []
        # ``sel_idx``：set[int]，已经写入 ``selected`` 的 ``self.residues`` 第一维编号，防止一个残基被多个配体原子重复加入。
        sel_idx = set()
        # The time-complexity is O(mn).
        # ``mol_pos``：float64 ndarray，形状为 (L, 3)；参考配体各原子的世界坐标，最后一维按 XYZ 排列，单位 Å。
        mol_pos = mol.GetConformer().GetPositions()
        # ``center``：float64 ndarray，形状为 (3,)；当前参考配体原子的世界坐标，单位 Å。
        for center in mol_pos:
            # ``i``：int，当前残基在 ``self.residues`` 中的 0-based 编号。
            # ``residue``：dict，当前候选蛋白残基；字段契约与返回值 ``selected[*]`` 相同。
            for i, residue in enumerate(self.residues):
                if criterion == 'min':
                    # ``res_pos``：float32 ndarray，形状为 (P_r, 3)；当前残基 P_r 个重原子的 PDB 世界坐标，单位 Å。
                    res_pos = np.array([self.pos[atom] for atom in residue['atoms']])
                    # ``distance_all``：ndarray，形状为 (P_r,)；当前配体原子到残基每个重原子的欧氏距离，单位 Å。
                    distance_all = np.linalg.norm(res_pos - center, ord=2, axis=-1)
                    # ``distance``：float，当前配体原子到该残基任一重原子的最小距离，单位 Å。
                    distance = distance_all.min()
                else:
                    # ``distance``：float，当前配体原子到 ``residue[criterion]`` 代表坐标的欧氏距离，单位 Å。
                    distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected
    
    def get_pocket_info(self, selected):
        num_res = len(selected)
        num_atoms = sum([len(residue['atoms']) for residue in selected])
        res_ids = [residue['chain_res_id'] for residue in selected]
        cover_chains = sorted(list(set([residue['chain'] for residue in selected])))
        pocket_info = {
            'num_res': num_res,
            'num_atoms': num_atoms,
            'res_ids': ';'.join(res_ids),
            'num_chains': len(cover_chains),
            'cover_chain_ids': ';'.join(cover_chains),
        }
        return pocket_info

    def get_chain_seqs(self, chains):
        if isinstance(chains, str):
            chains = [chains]
        seqs = []
        for chain in chains:
            seq = []
            for residue in self.residues:
                if residue['chain'] == chain:
                    seq.append(self.AA_NAME_SYM[residue['name']])
            seqs.append(''.join(seq))
            
        return ';'.join(seqs)


    def residues_to_pdb_block(self, residues, name='POCKET'):
        """把选中残基的原始 PDB 原子行拼成独立口袋文本。

        输入参数:
            - residues: Sequence[dict]，待写出的蛋白残基；每个 ``residues[*].atoms`` 是索引 ``self.atoms`` 第一维的重原子编号列表。
            - name: str，写入 ``HEADER`` 与 ``COMPND`` 记录的口袋名称。

        返回值:
            - block: str，以 ``HEADER``、``COMPND`` 开始并以 ``END`` 结束的 PDB block；ATOM 行及其 XYZ 世界坐标保持输入 PDB 原文不变。
        """
        # ``block``：str，逐步累积口袋 PDB 记录；初始两行保存 ``name`` 指定的标题与组分名。
        block =  "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        # ``residue``：dict，当前待写残基；``atoms`` 叶给出其重原子在 ``self.atoms`` 中的编号。
        for residue in residues:
            # ``atom_idx``：int，当前残基重原子的 0-based 编号，索引 ``self.atoms`` 第一维。
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block
    
    
def parse_pdb_peptide(pdb_path):
    parser = PDBParser(PERMISSIVE=0)
    structure = parser.get_structure('structure', pdb_path)
    
    pep_feature_dict = {'pos': [], 'atom_name': [], 'is_backbone': [], 'res_id': [], 'atom_to_aa_type': []}
    for atom in structure.get_atoms():
        element = atom.element
        if element == 'H':
            continue
        element = element if len(element) == 1 else element[0] + element[1].lower()
        pep_feature_dict['pos'].append(atom.get_coord())
        pep_feature_dict['atom_name'].append(atom.get_name())
        res = atom.get_parent()
        pep_feature_dict['res_id'].append(res.get_id()[1])
        resname = res.get_resname()
        aa_type = PDBProtein.AA_NAME_NUMBER[resname] if resname in PDBProtein.AA_NAME_NUMBER else len(PDBProtein.AA_NAME_NUMBER)
        pep_feature_dict['atom_to_aa_type'].append(aa_type)
        pep_feature_dict['is_backbone'].append(
            (atom.get_name() in PDBProtein.BACKBONE_NAMES) and (aa_type < len(PDBProtein.AA_NAME_NUMBER)))

    unique_res_id, res_index = np.unique(pep_feature_dict['res_id'], return_inverse=True)
    pep_feature_dict['res_index'] = res_index
    assert (np.diff(unique_res_id) == 1).all(), 'residue id is not continuous'

    pep_feature_dict = {k: np.array(v) for k, v in pep_feature_dict.items()}
    
    pep_feature_dict['seq'] = ''.join([seq1(residue.get_resname()) or 'X' for residue in structure.get_residues()])
    pep_feature_dict['pep_len'] = len(pep_feature_dict['seq'])
    pep_feature_dict['pep_path'] = pdb_path
    pep_feature_dict['atom_name'] = list(pep_feature_dict['atom_name'])
    
    if 'X' in pep_feature_dict['seq']:
        print('Warning: X in peptide sequence:', pdb_path)
    
    return pep_feature_dict


def parse_mol_with_confs(mol, smiles=None, confs=None):
    if smiles is not None: # check smiles
        if smiles != Chem.MolToSmiles(mol, isomericSmiles=False):
            return None
    pos_all_confs = []
    i_conf_list = []

    data = parse_3d_mol(mol, smiles=smiles, not_pos=True)
    if data is None:
        return None
    
    # get all confs
    if confs is None:
        for i_conf in range(mol.GetNumConformers()):
            pos = mol.GetConformer(i_conf).GetPositions()
            pos_all_confs.append(pos)
            i_conf_list.append(i_conf)
            # check bond length
            # bond_lengths = np.linalg.norm(pos[data['bond_index'][0]] - pos[data['bond_index'][1]], ord=2, axis=-1)
            # if (bond_lengths > 3.5).any():
            #     print('Skipping conformer with too long bond length: %s' % bond_lengths.max())
            #     return None
    else:
        pos_all_confs = np.array(confs)[:, :len(data['element']), :]
        i_conf_list = list(range(len(confs)))

    return {
        'element': np.array(data['element']),
        'bond_index': np.array(data['bond_index']),
        'bond_type': np.array(data['bond_type']),
        'pos_all_confs': np.array(pos_all_confs, dtype=np.float32),
        'num_atoms': data['num_atoms'],
        'num_bonds': data['num_bonds'],
        'i_conf_list': i_conf_list,
        'num_confs': len(i_conf_list),
    }


def parse_conf_list(conf_list, smiles=None):
    """筛选拓扑一致的 RDKit conformer，并汇总为构象生成或 docking 的配体字段。

    输入参数:
        - conf_list: Sequence[RDKit Mol]，候选构象列表；每个分子应含一个三维 conformer，与 ``smiles`` 或首个保留构象拓扑不一致的项会被跳过。
        - smiles: str|None，可选非手性 SMILES 约束；非空时 ``parse_3d_mol`` 只接受规范非手性 SMILES 完全相同的分子。

    返回值:
        - ligand_dict: dict，首个有效构象的二维图与全部通过筛选的三维坐标。
        - ligand_dict.element: int64 ndarray，形状为 (N,)；逐配体原子的原子序数，原子顺序以首个保留构象为准；C=0 时为 float64 空数组 (0,)。
        - ligand_dict.bond_index: int64 ndarray，形状为 (2, 2M)；排序后的双向化学键端点，数值索引 ``element`` 第一维；C=0 时为 float64 空数组 (0,)。
        - ligand_dict.bond_type: int64 ndarray，形状为 (2M,)；与 ``bond_index`` 列对齐的键类别，1/2/3/4 表示单/双/三/芳香键；C=0 时为 float64 空数组 (0,)。
        - ligand_dict.pos_all_confs: float32 ndarray，形状为 (C, N, 3)；C 个保留构象的世界坐标，最后一维按 XYZ 排列，单位 Å；C=0 时实际形状为 (0,)。
        - ligand_dict.num_atoms: int 标量 N，首个有效构象的配体原子数。
        - ligand_dict.num_bonds: int 标量 M，首个有效构象的无向化学键数。
        - ligand_dict.i_conf_list: list[int]，长度为 C；逐保留构象指向 ``conf_list`` 第一维的原始编号。
        - ligand_dict.num_confs: int 标量 C，保留构象数量。

    注意:
        - 没有有效构象时，本函数仍返回由空列表和零计数组成的字段，不抛出专用异常。
    """
    # data_list = [parse_drug3d_mol(conf) for conf in conf_list]
    # ``element``：list，尚未找到有效构象时的空哨兵；找到首个有效构象后更新为形状 (N,) 的 int64 ndarray。
    element = []
    # ``bond_index``：list，尚未找到有效构象时的空哨兵；找到首个有效构象后更新为形状 (2, 2M) 的 int64 ndarray。
    bond_index = []
    # ``bond_type``：list，尚未找到有效构象时的空哨兵；找到首个有效构象后更新为形状 (2M,) 的 int64 ndarray。
    bond_type = []
    # ``pos_all_confs``：list[ndarray]，逐项保存一个通过拓扑筛选的 (N, 3) 配体世界坐标，单位 Å。
    pos_all_confs = []
    # ``i_conf_list``：list[int]，逐保留构象保存其在 ``conf_list`` 第一维中的原始编号。
    i_conf_list = []
    # ``num_atoms``：int 标量，首个有效构象的原子数 N；0 是尚未找到有效构象的哨兵值。
    num_atoms = 0
    # ``num_bonds``：int 标量，首个有效构象的无向键数 M；0 是尚未找到有效构象的哨兵值。
    num_bonds = 0  # NOTE: the num of bonds is not symtric
    # ``i_conf``：int，当前候选在 ``conf_list`` 第一维中的 0-based 编号。
    # ``conf``：RDKit Mol，当前待解析候选构象；必须含一个三维 conformer。
    for i_conf,  conf in enumerate(conf_list):
        # ``data``：dict|None，``parse_3d_mol`` 返回的当前构象原子、坐标和双向键字段；SMILES 不匹配时为 None。
        data = parse_3d_mol(conf, smiles=smiles)
        # ``data.element``：int64 ndarray，形状为 (N_i,)；当前构象的逐原子序数。
        # ``data.pos``：float32 ndarray，形状为 (N_i, 3)；当前构象的世界坐标，单位 Å。
        # ``data.bond_index``：int64 ndarray，形状为 (2, 2M_i)；当前构象排序后的双向键端点。
        # ``data.bond_type``：int64 ndarray，形状为 (2M_i,)；与 ``data.bond_index`` 列对齐的键类别。
        # ``data.num_atoms``：int 标量 N_i，当前构象的原子数。
        # ``data.num_bonds``：int 标量 M_i，当前构象的无向键数。
        if data is None:
            continue
        # check element
        if len(element) == 0:
            # ``element``：int64 ndarray，形状为 (N,)；首个有效构象确定的参考原子序数与顺序。
            element = data['element']
            # ``num_atoms``：int 标量 N，首个有效构象确定的参考原子数。
            num_atoms = data['num_atoms']
        else:
            if data['num_atoms'] != num_atoms:
                print('Skipping conformer with different number of atoms')
                continue
            if not np.all(element == data['element']):
                print('Skipping conformer with different element order')
                continue
        # check bond
        if len(bond_index) == 0:
            # ``bond_index``：int64 ndarray，形状为 (2, 2M)；首个有效构象确定的参考双向键端点。
            bond_index = data['bond_index']
            # ``bond_type``：int64 ndarray，形状为 (2M,)；首个有效构象确定的参考键类别。
            bond_type = data['bond_type']
            # ``num_bonds``：int 标量 M，首个有效构象确定的参考无向键数。
            num_bonds = data['num_bonds']
        else:
            if data['num_bonds'] != num_bonds:
                print('Skipping conformer with different number of bonds')
                continue
            if not np.all(bond_index == data['bond_index']):
                print('Skipping conformer with different bond index')
                continue
            if not np.all(bond_type == data['bond_type']):
                print('Skipping conformer with different bond type')
                continue
        pos_all_confs.append(data['pos'])
        i_conf_list.append(i_conf)

    return {
        # ``ligand_dict.element``：int64 ndarray，形状为 (N,)；首个有效构象的逐原子序数与顺序；C=0 时为 float64 空数组 (0,)。
        'element': np.array(element),
        # ``ligand_dict.bond_index``：int64 ndarray，形状为 (2, 2M)；首个有效构象排序后的双向键端点；C=0 时为 float64 空数组 (0,)。
        'bond_index': np.array(bond_index),
        # ``ligand_dict.bond_type``：int64 ndarray，形状为 (2M,)；与 ``bond_index`` 列对齐的键类别；C=0 时为 float64 空数组 (0,)。
        'bond_type': np.array(bond_type),
        # 'bond_rotatable': np.array(data['bond_rotatable']),
        # ``ligand_dict.pos_all_confs``：float32 ndarray，形状为 (C, N, 3)；全部保留构象的世界坐标，单位 Å；C=0 时实际形状为 (0,)。
        'pos_all_confs': np.array(pos_all_confs, dtype=np.float32),
        # ``ligand_dict.num_atoms``：int 标量 N，首个有效构象的原子数。
        'num_atoms': num_atoms,
        # ``ligand_dict.num_bonds``：int 标量 M，首个有效构象的无向键数。
        'num_bonds': num_bonds,
        # ``ligand_dict.i_conf_list``：list[int]，长度为 C；逐保留构象在 ``conf_list`` 中的原始编号。
        'i_conf_list': i_conf_list,
        # ``ligand_dict.num_confs``：int 标量 C，保留构象数量。
        'num_confs': len(i_conf_list),
    }


def parse_3d_mol(mol, smiles=None, not_pos=False):
    if smiles is not None: # check smiles
        if smiles != Chem.MolToSmiles(mol, isomericSmiles=False):
            return None
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    if not not_pos:
        conf = mol.GetConformer()
    ele_list = []
    pos_list = []
    for i, atom in enumerate(mol.GetAtoms()):
        ele = atom.GetAtomicNum()
        if not not_pos:
            pos = conf.GetAtomPosition(i)
            pos_list.append(list(pos))
        ele_list.append(ele)
    
    row, col = [], []
    bond_type = []
    for bond in mol.GetBonds():
        b_type = int(bond.GetBondType())
        assert b_type in [1, 2, 3, 12], 'Bond can only be 1,2,3,12 bond'
        b_type = b_type if b_type != 12 else 4
        b_index = [
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx()
        ]
        bond_type += 2*[b_type]
        row += [b_index[0], b_index[1]]
        col += [b_index[1], b_index[0]]
    
    bond_type = np.array(bond_type, dtype=np.int64)
    bond_index = np.array([row, col],dtype=np.int64)

    perm = (bond_index[0] * num_atoms + bond_index[1]).argsort()
    bond_index = bond_index[:, perm]
    bond_type = bond_type[perm]
    
    # # is rotable bond 
    # rot_mat = find_rotatable_bond_mat(mol)
    # bond_rotatable = rot_mat[bond_index[0], bond_index[1]]

    data = {
        'element': np.array(ele_list, dtype=np.int64),
        'pos': np.array(pos_list, dtype=np.float32),
        'bond_index': np.array(bond_index, dtype=np.int64),
        'bond_type': np.array(bond_type, dtype=np.int64),
        # 'bond_rotatable': np.array(bond_rotatable, dtype=np.int64),
        'num_atoms': num_atoms,
        'num_bonds': num_bonds,
    }
    return data
