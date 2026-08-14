"""
定义 PocketXMol 分子—口袋图的内存字段容器及 PyG 批处理索引偏移规则。

主要入口是 :class:`PocketMolData`：数据集先把 LMDB 中的配体字段和可选口袋字段装入该容器，
随后 ``FeaturizePocket``、``FeaturizeMol``、任务变换和噪声器继续向同一对象写入模型字段。
本模块不落盘；返回的核心数据是逐样本 ``Data`` 或由 PyG 拼接后的 ``Batch``，其中原子、半边、
刚体域、可旋转键和口袋 kNN 边的索引会按照各自实体数量独立偏移。
"""

import copy
from typing import Any
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

# FOLLOW_BATCH = ['protein_element', 'ligand_context_element', 'pos_real', 'pos_fake']



def edge_index_to_index_of_edge(edge_index, batch_node):
    """
    edge_index: (2, E)
    """
    assert (edge_index[0] < edge_index[1]).all()
    num_nodes_batch = batch_node.bincounts()
    index_mol = (edge_index - num_nodes_batch).abs().min()[1]
    bias = num_nodes_batch[:index_mol]
    edge_index_in_mol = edge_index - bias
    id_edge_0, id_edge_1 = edge_index_in_mol
    id_edge = (2*num_nodes_batch[id_edge_0]-id_edge_0-1) * id_edge_0 // 2 + id_edge_1 - id_edge_0 - 1
    return id_edge


def inc_func_mol3d(self, key, value, *args, **kwargs):
    """
    给 PyG 返回分子索引字段在拼接下一个样本前应增加的偏移量。

    输入参数:
        - self: ``Mol3DData`` 或 ``PocketMolData``，至少含 ``node_type``，部分任务还含 ``halfedge_type``、``n_domain`` 和扭转注释字段。
        - key: str, 当前待拼接字段名。
        - value: 当前字段值；本函数只根据 ``key`` 与同一样本中的实体数量计算偏移，不修改该值。

    返回值:
        - 原子索引字段: int 标量 ``N``，其中 ``N=len(node_type)``。
        - 半边索引字段: int 标量 ``H``，其中 ``H=len(halfedge_type)``。
        - domain_node_index: int64, (2, 1), 第一行增加刚体域数 ``D``，第二行增加原子数 ``N``。
        - tor_bonds_anno: int64, (3,), 扭转层级列不偏移，两列原子编号各增加 ``N``。
        - twisted_nodes_anno: int64, (2,), 可旋转键编号增加扭转键数 ``T``，原子编号增加 ``N``。
        - dihedral_pairs_anno: int64, (3,), 可旋转键编号增加 ``T``，两列二面角端点原子编号各增加 ``N``。
        - 未注册字段: ``None``，交由 ``Data.__inc__`` 的默认规则处理。
    """
    # bond_index、edge_index 与 halfedge_index 的数值都索引 ``node_type`` 第一维。
    if key == 'bond_index':
        return len(self['node_type'])
    elif key == 'edge_index':
        return len(self['node_type'])
    elif key == 'halfedge_index':
        return len(self['node_type'])
    elif key in ['node_p1', 'node_p2', 'node_bb', 'node_sc'] or key.startswith('node_part_'):  # index of node of part in mol
        return len(self['node_type'])
    elif key in ['halfedge_p1', 'halfedge_p2', 'halfedge_p1p2',
                 'halfedge_bb', 'halfedge_sc', 'halfedge_bbsc'] or\
                     key.startswith('halfedge_part_'):  # index of halfedge of part in mol
        return len(self['halfedge_type'])
    # (2, K), 第一行索引刚体域，第二行索引分子原子；两行使用不同的批次偏移。
    elif key == 'domain_node_index':
        return torch.tensor([[self['n_domain']], [len(self['node_type'])]]) # [2, 1]
    elif key == 'domain_center_nodes':
        return len(self['node_type'])
    elif key == 'tor_bonds_anno':
        # ``n_node``：int N，当前图的配体原子数；作为后续图 node 索引的批处理偏移。
        n_node = len(self['node_type'])
        return torch.tensor([0, n_node, n_node])
    elif key == 'twisted_nodes_anno':
        # ``n_tor``：int T，当前图的可旋转键注释行数；作为后续图 torsion 行号的批处理偏移。
        n_tor = len(self['tor_bonds_anno'])
        return torch.tensor([n_tor, len(self['node_type'])])
    elif key == 'dihedral_pairs_anno':
        # ``n_node``：int N，当前图配体原子数；用于两列原子索引偏移。
        n_node = len(self['node_type'])
        # ``n_tor``：int T，当前图可旋转键数；用于首列 torsion 行号偏移。
        n_tor = len(self['tor_bonds_anno'])
        return torch.tensor([n_tor, n_node, n_node])
    return None
    
    
def inf_func_pocket(self, key, value, *args, **kwargs):
    """
    给 PyG 返回口袋索引字段在拼接下一个样本前应增加的偏移量。

    输入参数:
        - self: ``PocketMolData``，含 ``pocket_pos``。
        - key: str, 当前待拼接字段名。
        - value: 当前字段值；本函数不修改该值。

    返回值:
        - pocket_knn_edge_index: int 标量 ``P``，其中 ``P=len(pocket_pos)``，使两行端点继续索引拼接后的口袋原子第一维。
        - 未注册 key: ``None``，表示本函数没有定义额外偏移规则。
    """
    if key == 'pocket_knn_edge_index':
        return len(self['pocket_pos'])
    return None


class PocketMolData(Data):
    """
    同时承载一个配体图、可选蛋白口袋图以及构象/docking 任务派生字段。

    构造输入:
        - ``**kwargs``: 直接写入 PyG ``Data`` 的字段；字段可以是张量、NumPy 数组、标量字符串或任务注释容器。

    原始配体核心字段:
        - element: int64, (N,), 每个配体原子的原子序数；与 ``pos_all_confs`` 第二维逐原子对齐。
        - pos_all_confs: (C, N, 3), 同一分子的 C 个候选构象；最后一维按 XYZ 排列，单位 Å。
        - i_conf_list: (C,), 每个候选构象在原数据中的编号；与 ``pos_all_confs`` 第一维对齐。
        - bond_index: int64, (2, 2M), 双向化学键端点；数值索引 ``element`` 第一维。
        - bond_type: int64, (2M,), 化学键类别；与 ``bond_index`` 第二维逐键对齐，1/2/3/4 表示单/双/三/芳香键。
        - num_atoms: int 标量 N；``FeaturizeMol`` 将其复制为 PyG 的 ``num_nodes``。
        - num_bonds: int 标量 M；只计每条无向化学键一次。

    原始口袋核心字段:
        - pocket_element: int64, (P,), 每个口袋原子的原子序数。
        - pocket_pos: (P, 3), 口袋原子坐标；最后一维按 XYZ 排列，单位 Å。
        - pocket_atom_to_aa_type: int64, (P,), 每个口袋原子所属氨基酸的 0 到 19 类编号。
        - pocket_is_backbone: bool 或 0/1, (P,), 每个口袋原子是否属于蛋白主链。
        - pdbid: str, 受体结构标识；缺失或 NaN 会在 ``FeaturizePocket`` 中归一为空字符串。

    变换后的模型核心字段:
        - node_type: int64, (N,), 配体原子类别编号；数值索引 ``FeaturizeMol`` 定义的原子类别表。
        - node_pos: (N, 3), 配体原子在以 ``pocket_center`` 为原点的坐标；无口袋任务则以配体质心为原点，单位 Å。
        - halfedge_index: int64, (2, H), 完全图上三角半边端点，H=N(N-1)/2；每列满足首端点小于末端点。
        - halfedge_type: int64, (H,), 半边类别；0 表示非键，1 到 4 表示化学键，额外 mask 类由配置决定。
        - pocket_atom_feature: (P, 25), 元素 one-hot、氨基酸 one-hot 与主链标记拼接后的口袋节点特征。
        - pocket_knn_edge_index: int64, (2, E_p), 口袋 kNN 有向边端点；数值索引 ``pocket_pos`` 第一维。
        - fixed_node: LongTensor|BoolTensor，形状为 (N,)，1 表示原子类别在当前任务中固定。
        - fixed_pos: LongTensor|BoolTensor，形状为 (N,)，1 表示坐标在当前任务中固定。
        - fixed_halfedge: LongTensor|BoolTensor，形状为 (H,)，1 表示半边类别在当前任务中固定。
        - fixed_halfdist: LongTensor|BoolTensor，形状为 (H,)，1 表示端点距离在当前任务中固定。

    批处理行为:
        - ``__inc__`` 分别按原子数 N、半边数 H、口袋原子数 P、刚体域数 D 和可旋转键数 T 偏移索引字段。
        - 普通张量由 PyG 沿默认拼接维连接；``follow_batch`` 额外生成 ``node_type_batch``、``halfedge_type_batch`` 与 ``pocket_pos_batch``。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_pocket_mol_dicts(pocket_dict=None, mol_dict=None, **kwargs):
        """
        把未加前缀的口袋字段和配体字段合并成一个 ``PocketMolData``。

        输入参数:
            - pocket_dict.element: LongTensor|ndarray，形状为 (P,)，口袋原子序数。
            - pocket_dict.molecule_name: str|None，口袋 PDB ``HEADER`` 名称。
            - pocket_dict.pos: FloatTensor|ndarray，形状为 (P, 3)，口袋世界坐标，单位 Å。
            - pocket_dict.is_backbone: BoolTensor|ndarray，形状为 (P,)，口袋原子主链标记。
            - pocket_dict.atom_name: list[str]，长度为 P，PDB 原子名。
            - pocket_dict.atom_to_aa_type: LongTensor|ndarray，形状为 (P,)，口袋原子氨基酸类别。
            - mol_dict.element: LongTensor|ndarray，形状为 (N,)，配体原子序数。
            - mol_dict.pos_all_confs: FloatTensor|ndarray，形状为 (C, N, 3)，配体 conformer 坐标，单位 Å。
            - mol_dict.i_conf_list: list[int]，长度为 C，合法 conformer 的输入编号。
            - mol_dict.num_confs: int 标量 C，合法 conformer 数。
            - mol_dict.bond_index: LongTensor|ndarray，形状为 (2, 2M)，双向化学键端点。
            - mol_dict.bond_type: LongTensor|ndarray，形状为 (2M,)，逐双向键类别。
            - mol_dict.num_atoms: int 标量 N，配体原子数。
            - mol_dict.num_bonds: int 标量 M，无向化学键数。
            - kwargs.pdbid: str|缺省，受体结构标识。
            - kwargs.data_id: str|缺省，样本标识。
            - kwargs.smiles: str|缺省，固定二维配体图的规范 SMILES。

        返回字段:
            - instance.pocket_element: LongTensor|ndarray，形状为 (P,)，由 ``pocket_dict.element`` 映射。
            - instance.pocket_molecule_name: str|None，由 ``pocket_dict.molecule_name`` 映射。
            - instance.pocket_pos: FloatTensor|ndarray，形状为 (P, 3)，由 ``pocket_dict.pos`` 映射，单位 Å。
            - instance.pocket_is_backbone: BoolTensor|ndarray，形状为 (P,)，由 ``pocket_dict.is_backbone`` 映射。
            - instance.pocket_atom_name: list[str]，长度为 P，由 ``pocket_dict.atom_name`` 映射。
            - instance.pocket_atom_to_aa_type: LongTensor|ndarray，形状为 (P,)，由 ``pocket_dict.atom_to_aa_type`` 映射。
            - instance.element: LongTensor|ndarray，形状为 (N,)，由 ``mol_dict.element`` 原名写入。
            - instance.pos_all_confs: FloatTensor|ndarray，形状为 (C, N, 3)，由 ``mol_dict.pos_all_confs`` 原名写入，单位 Å。
            - instance.i_conf_list: list[int]，长度为 C，由 ``mol_dict.i_conf_list`` 原名写入。
            - instance.num_confs: int 标量 C，由 ``mol_dict.num_confs`` 原名写入。
            - instance.bond_index: LongTensor|ndarray，形状为 (2, 2M)，由 ``mol_dict.bond_index`` 原名写入。
            - instance.bond_type: LongTensor|ndarray，形状为 (2M,)，由 ``mol_dict.bond_type`` 原名写入。
            - instance.num_atoms: int 标量 N，由 ``mol_dict.num_atoms`` 原名写入。
            - instance.num_bonds: int 标量 M，由 ``mol_dict.num_bonds`` 原名写入。
            - instance.pdbid: str|缺省，由 ``kwargs.pdbid`` 预写入；同名后续字典叶可覆盖。
            - instance.data_id: str|缺省，由 ``kwargs.data_id`` 预写入；同名后续字典叶可覆盖。
            - instance.smiles: str|缺省，由 ``kwargs.smiles`` 预写入；同名后续字典叶可覆盖。
        """
        # ``instance``：PocketMolData，先承载调用方附加字段；随后依次合并口袋与配体字典。
        instance = PocketMolData(**kwargs)

        if pocket_dict is not None:
            # ``key``：str，当前口袋叶名。
            # ``item``：当前口袋叶的数组、张量或元数据值。
            for key, item in pocket_dict.items():
                # ``instance``：将 pocket_dict.<key> 映射为 pocket_<key>，防止与配体同名字段冲突。
                instance['pocket_' + key] = item

        if mol_dict is not None:
            # ``key``：str，当前配体叶名。
            # ``item``：当前配体叶的数组、张量或元数据值。
            for key, item in mol_dict.items():
                # ``instance``：保留 mol_dict 原叶子名；同名时覆盖 kwargs 中的旧值。
                instance[key] = item

        return instance
    

    def __inc__(self, key, value, *args, **kwargs):
        """返回当前字段在 PyG 拼接下一个图前使用的索引偏移。

        输入参数:
            - key: str，当前待拼接字段名。
            - value: 当前字段值；只传给偏移规则或 PyG 默认实现，本函数不修改它。
            - ``*args``: PyG 传入的附加位置参数，原样转发。
            - ``**kwargs``: PyG 传入的附加关键字参数，原样转发。

        返回值:
            - inc: int|Tensor|PyG 默认值，命中本模块规则时按原子、半边、刚体域、扭转或口袋实体数偏移，否则由 ``Data.__inc__`` 决定。
        """
        # for defined mol inc
        # ``inc``：int|Tensor|None，按配体原子、半边、刚体域或扭转实体计算的候选偏移。
        inc = inc_func_mol3d(self, key, value, *args, **kwargs)
        if inc is not None:
            return inc
        # for defined pocket inc
        # ``inc``：int|None，配体规则未命中后按口袋原子数计算的候选偏移。
        inc = inf_func_pocket(self, key, value, *args, **kwargs)
        if inc is not None:
            return inc
        # undefined
        return super().__inc__(key, value, *args, **kwargs)


class Mol3DData(Data):
    """
    承载不含独立口袋命名空间的单个三维分子图。

    ``from_3dmol_dicts`` 保留配体字典的原字段名，并用 ``orig_keys`` 记录最初键列表；
    ``__inc__`` 与 ``PocketMolData`` 共用分子、口袋索引偏移规则，供通用变换和批处理复用。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_3dmol_dicts(ligand_dict=None, **kwargs):
        """
        从配体字段映射构造 ``Mol3DData`` 并记录原始字段清单。

        输入参数:
            - ligand_dict: dict|None, 配体字段映射；键名和值原样写入实例。
            - ``**kwargs``: 先写入实例的附加字段；``ligand_dict`` 中的同名键会覆盖它们。

        返回值:
            - instance: ``Mol3DData``；若提供 ``ligand_dict``，``orig_keys`` 为 ``list[str]``，按输入映射迭代顺序保存其全部键名。
        """
        # ``instance``：Mol3DData，先承载调用方附加字段，再合并配体全部叶子。
        instance = Mol3DData(**kwargs)

        if ligand_dict is not None:
            # ``key``：str，当前配体叶名。
            # ``item``：当前配体叶的数组、张量或元数据值。
            for key, item in ligand_dict.items():
                # ``instance``：保留 ligand_dict 原叶子名；同名时覆盖 kwargs 中的旧值。
                instance[key] = item
            # ``instance.orig_keys``：list[str]，记录进入对象前的原始配体键集合，供后续变换区分原字段与派生字段。
            instance['orig_keys'] = list(ligand_dict.keys())

        # instance['nbh_list'] = {i.item():[j.item() for k, j in enumerate(instance.ligand_bond_index[1]) if instance.ligand_bond_index[0, k].item() == i] for i in instance.ligand_bond_index[0]}
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        # ``inc``：int/Tensor|None，先尝试按配体原子、半边、torsion 实体计算批处理偏移。
        inc = inc_func_mol3d(self, key, value, *args, **kwargs)
        if inc is not None:
            return inc
        # ``inc``：int|None，仅当配体规则未命中时尝试口袋点索引偏移。
        inc = inf_func_pocket(self, key, value, *args, **kwargs)
        if inc is not None:
            return inc
        return super().__inc__(key, value, *args, **kwargs)


def torchify_dict(data):
    """
    仅把字段映射中的 NumPy 数组零拷贝转换为 PyTorch 张量。

    输入参数:
        - data: dict, 任意字段映射；值可以是 ``np.ndarray``、张量、标量或 Python 容器。

    返回值:
        - output: dict, 键集合与输入一致；每个 ``np.ndarray`` 通过 ``torch.from_numpy`` 共享底层内存，其他值保持对象身份不变。
    """
    # ``output``：dict[str, object]，保持输入键集合；仅 ndarray 叶子转换为共享数值的 CPU Tensor。
    output = {}
    # ``k``：str，当前字段名。
    # ``v``：当前字段对应的 ndarray 或其他 Python 叶值。
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            # ``output``：Tensor，形状和 dtype 与 ndarray v 保持一致；可能与 v 共享 CPU 内存。
            output[k] = torch.from_numpy(v)
        else:
            # ``output``：非 ndarray 叶子原样保留类型与对象引用。
            output[k] = v
    return output
