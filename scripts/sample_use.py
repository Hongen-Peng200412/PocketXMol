"""对用户提供的蛋白与小分子执行 PocketXMol docking 采样并保存候选 pose。

``configs/sample/examples/dock_smallmol*.yml`` 对应的小分子路径先裁剪
蛋白口袋、从输入 SDF/SMILES 建立固定二维图与起始构象，再复制成候选数据集，调用
``sample_loop3`` 完成迭代去噪，最后用输入 RDKit 分子的拓扑重建每个候选坐标。

命令行参数:
    - config_task: str，任务 YAML 路径，定义输入结构、候选预算、task prompt 与采样噪声。
    - config_model: str|None，模型 YAML 路径，定义 checkpoint；与任务 YAML 合并读取。
    - outdir: str，实验输出根目录；脚本在其下创建带配置名前缀的新日志目录。
    - device: str，PyTorch 设备，例如 ``cuda:0`` 或 ``cpu``。
    - batch_size: int，候选批大小覆盖值；0 表示沿用任务 YAML 的 ``sample.batch_size``。
    - shuffle: bool，是否打乱复制出的候选；现有 argparse ``type=bool`` 语义原样保留。
    - num_workers: int，DataLoader worker 覆盖值；-1 表示沿用训练配置。

主要落盘产物:
    - <log_dir>/<task_yaml_name>.yml: 合并后的运行配置快照。
    - <log_dir>/<experiment>_SDF/0_inputs/pocket_block.pdb: 本次裁剪出的口袋 PDB block。
    - <log_dir>/<experiment>_SDF/0_inputs/input_mol.sdf: 固定二维拓扑和起始构象。
    - <log_dir>/<experiment>_SDF/<index>[-<tag>].sdf: 每个小分子候选 pose。
    - <log_dir>/SDF/<index>-<traj_kind>.sdf: 按概率保存、以 ``$$$$`` 分隔的去噪轨迹。
    - <log_dir>/gen_info.csv: 候选文件名、元数据、SMILES、重建标签和原始 confidence 汇总。
    - <log_dir>/samples_all.pt: 只保存各结果池长度的占位映射，不保存完整分子对象。
"""

# Standard library imports
import argparse
import gc
import os
import shutil
import sys
from itertools import cycle

# Third-party imports
import numpy as np
import torch
from Bio import PDB
from Bio.SeqUtils import seq1
from easydict import EasyDict
from rdkit import Chem
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

# Local imports
sys.path.append('.')
from models.maskfill import PMAsymDenoiser
from models.sample import get_cfd_traj, sample_loop3, seperate_outputs2
from process.utils_process import (
    add_pep_bb_data,
    extract_pocket,
    get_input_from_file,
    get_peptide_info,
    make_dummy_mol_with_coordinate,
)
from scripts.train_pl import DataModule
from utils.dataset import UseDataset
from utils.misc import *
from utils.reconstruct import *
from utils.sample_noise import get_sample_noiser
from utils.transforms import *


def print_pool_status(pool, logger, is_pep: bool = False) -> None:
    """Print statistics of generation results.

    Args:
        pool: Result pool containing successful and failed generations.
        logger: Logger instance.
        is_pep: Whether generating peptides (affects output format).
    """
    if not is_pep:
        logger.info('[Pool] Succ/Incomp/Bad: %d/%d/%d' % (
            len(pool.succ), len(pool.incomp), len(pool.bad)
        ))
    else:
        logger.info('[Pool] Succ/Nonstd/Incomp/Bad: %d/%d/%d/%d' % (
            len(pool.succ), len(pool.nonstd), len(pool.incomp), len(pool.bad)
        ))


def get_input_data(protein_path,
                   input_ligand=None,
                   is_pep=False,
                   pocket_args={},
                   pocmol_args={}):
    """裁剪蛋白口袋，并把输入小分子和口袋解析为联合图的原始字段。

    输入参数:
        - protein_path: str，完整受体蛋白 PDB 路径；输出口袋坐标保留该文件的原始坐标系。
        - input_ligand: str|None，小分子 SDF/PDB 路径或 SMILES；小分子 docking 示例传带三维 conformer 的 SDF。
        - is_pep: bool，小分子 docking 调用传 False。
        - pocket_args: Mapping，口袋裁剪配置。
        - pocket_args.ref_ligand_path: str|None，可选参考配体路径；非空时优先用其原子定位口袋。
        - pocket_args.pocket_coord: Sequence[float]|None，可选长度 3 的裁剪中心，单位 Å；仅在没有参考配体时转成单碳占位分子。
        - pocket_args.radius: float，可选裁剪半径，单位 Å，默认 10。
        - pocket_args.criterion: str，可选残基选择准则，默认 ``center_of_mass``。
        - pocmol_args: Mapping，展开传给 ``get_input_from_file`` 的身份元数据。
        - pocmol_args.data_id: str，可选样本唯一标识，进入联合数据和输出索引。
        - pocmol_args.pdbid: str，可选 PDB 标识，只作为联合数据元字段。

    返回值:
        - pocmol_data.element: LongTensor，形状为 (N,)，配体原子序数。
        - pocmol_data.pos_all_confs: FloatTensor，形状为 (C, N, 3)，输入 conformer 世界坐标，单位 Å。
        - pocmol_data.i_conf_list: list[int]，长度为 C，合法 conformer 的输入编号。
        - pocmol_data.num_confs: int 标量 C，合法 conformer 数。
        - pocmol_data.bond_index: LongTensor，形状为 (2, 2M)，双向化学键端点。
        - pocmol_data.bond_type: LongTensor，形状为 (2M,)，与 ``bond_index`` 列对齐的键类别。
        - pocmol_data.num_atoms: int 标量 N，配体原子数。
        - pocmol_data.num_bonds: int 标量 M，无向化学键数。
        - pocmol_data.pocket_element: LongTensor，形状为 (P,)，口袋原子序数。
        - pocmol_data.pocket_pos: FloatTensor，形状为 (P, 3)，口袋原子世界坐标，单位 Å。
        - pocmol_data.pocket_is_backbone: BoolTensor，形状为 (P,)，口袋原子主链标记。
        - pocmol_data.pocket_atom_name: list[str]，长度为 P，PDB 原子名。
        - pocmol_data.pocket_atom_to_aa_type: LongTensor，形状为 (P,)，口袋原子所属氨基酸类别。
        - pocmol_data.pocket_molecule_name: str|None，口袋 PDB ``HEADER`` 名称。
        - pocmol_data.pdbid: str，调用配置给定的受体标识。
        - pocmol_data.data_id: str，调用配置给定的样本标识。
        - pocmol_data.smiles: str，固定二维配体图的规范 SMILES。
        - pocmol_data.bond_rotatable: LongTensor，形状为 (2M,)，与双向键对齐的可旋转标记。
        - pocmol_data.tor_twisted_pairs: dict[tuple[int, int], list[set[int], set[int]]]，可旋转键两侧非轴原子集合。
        - pocmol_data.fixed_dist_torsion: Tensor，形状为 (N, N)，1 表示距离不随内部扭转改变。
        - pocmol_data.tor_bond_mat: Tensor，形状为 (N, N)，逐原子对可旋转键标记。
        - pocmol_data.path_mat: Tensor，形状为 (N, N)，化学图最短路径，单位为键数。
        - pocmol_data.nbh_dict: dict[int, list[int]]，逐原子一跳邻居编号。
        - pocmol_data.matches_graph: LongTensor，形状为 (M_g, S_g)，忽略手性的压缩自同构映射。
        - pocmol_data.matches_iso: LongTensor，形状为 (M_i, S_i)，考虑手性的压缩自同构映射。
        - pocmol_data.brics.subgraphs: list[list[int]]，BRICS 片段原子编号。
        - pocmol_data.brics.anchors_list: list[set[int]]，BRICS 片段锚原子编号。
        - pocmol_data.brics.nbh_subgraphs: list[list[int]]，BRICS 片段邻接表。
        - pocmol_data.brics.connections: dict[tuple[int, int], tuple[int, int]]，BRICS 片段连接锚原子对。
        - pocmol_data.mmpa.subgraphs: list[list[int]]，MMPA 片段原子编号。
        - pocmol_data.mmpa.anchors_list: list[set[int]]，MMPA 片段锚原子编号。
        - pocmol_data.mmpa.nbh_subgraphs: list[list[int]]，MMPA 片段邻接表。
        - pocmol_data.mmpa.connections: dict[tuple[int, int], tuple[int, int]]，MMPA 片段连接锚原子对。
        - pocket_pdb: str，裁剪后口袋的 PDB block 文本，不是文件对象。
        - mol: RDKit Mol，已去显式氢，原子顺序与 ``pocmol_data.element`` 对齐；输出重建会深拷贝并只覆盖坐标。

    异常:
        - AssertionError: 未提供参考配体或口袋中心，且 ``input_ligand`` 不是可用于裁剪的 SDF/PDB 路径。
    """
    # 下方第二个三引号字符串是原仓库遗留的无效表达式，不是函数 Docstring；学习分支只记录而不删除，以保持可执行 AST 不变。
    """
    Process input protein and ligand files for generation.
    
    Extracts protein pocket around ligand/reference and prepares molecular data.
    
    Args:
        protein_path: Path to protein PDB file
        input_ligand: Ligand specification (SDF/PDB path or special format like 'pepseq_XXX')
        is_pep: Whether processing peptide
        pocket_args: Pocket extraction parameters (radius, ref_ligand_path, etc.)
        pocmol_args: Additional molecule processing parameters
        
    Returns:
        Tuple of (pocmol_data, pocket_pdb, mol):
            - pocmol_data: Processed pocket-molecule data dict
            - pocket_pdb: Extracted pocket PDB file object
            - mol: RDKit molecule object (or None)
    """

    # ``ref_ligand``：str|RDKit Mol|None，优先读取的口袋裁剪参考配体。
    ref_ligand = pocket_args.get('ref_ligand_path', None)
    # ``pocket_coord``：Sequence[float]|None，显式给定的原始蛋白坐标系裁剪中心，单位 Å。
    pocket_coord = pocket_args.get('pocket_coord', None)
    if ref_ligand is not None:
        pass  # 非空参考配体路径原样交给 ``extract_pocket``。
    elif pocket_coord is not None:
        # ``ref_ligand``：RDKit Mol，以给定中心为坐标的单碳占位分子，使现有配体邻域裁剪接口可复用。
        ref_ligand = make_dummy_mol_with_coordinate(pocket_coord)
    else:  # use input_ligand as reference
        print('Neither ref_ligand nor pocket_coord provided for pocket extraction. Using input_ligand as reference.')
        assert input_ligand is not None and (input_ligand.endswith('.sdf') or input_ligand.endswith('.pdb')), \
            'Only SDF/PDB input_ligand can be used for pocket extraction.'
        # ``ref_ligand``：str，回退为输入配体 SDF/PDB 路径，兼作口袋裁剪参考。
        ref_ligand = input_ligand
    
    # ``pocket_pdb``：str，按参考配体、半径和残基判据从完整蛋白裁剪出的 PDB block。
    pocket_pdb = extract_pocket(protein_path, ref_ligand, 
                            radius=pocket_args.get('radius', 10),
                            criterion=pocket_args.get('criterion', 'center_of_mass'))
    
    # ``pocmol_data.element``：LongTensor，形状为 (N,)，配体原子序数。
    # ``pocmol_data.pos_all_confs``：FloatTensor，形状为 (C, N, 3)，配体世界坐标，单位 Å。
    # ``pocmol_data.i_conf_list``：list[int]，长度为 C，合法 conformer 输入编号。
    # ``pocmol_data.num_confs``：int 标量 C，合法 conformer 数。
    # ``pocmol_data.bond_index``：LongTensor，形状为 (2, 2M)，双向化学键端点。
    # ``pocmol_data.bond_type``：LongTensor，形状为 (2M,)，逐双向键类别。
    # ``pocmol_data.num_atoms``：int 标量 N，配体原子数。
    # ``pocmol_data.num_bonds``：int 标量 M，无向化学键数。
    # ``pocmol_data.pocket_element``：LongTensor，形状为 (P,)，口袋原子序数。
    # ``pocmol_data.pocket_pos``：FloatTensor，形状为 (P, 3)，口袋世界坐标，单位 Å。
    # ``pocmol_data.pocket_is_backbone``：BoolTensor，形状为 (P,)，逐口袋原子主链标记。
    # ``pocmol_data.pocket_atom_name``：list[str]，长度为 P，PDB 原子名。
    # ``pocmol_data.pocket_atom_to_aa_type``：LongTensor，形状为 (P,)，逐口袋原子氨基酸类别。
    # ``pocmol_data.pocket_molecule_name``：str|None，口袋 PDB ``HEADER`` 名称。
    # ``pocmol_data.pdbid``：str，受体结构标识。
    # ``pocmol_data.data_id``：str，样本标识。
    # ``pocmol_data.smiles``：str，固定二维配体图的规范 SMILES。
    # ``pocmol_data.bond_rotatable``：LongTensor，形状为 (2M,)，可旋转键标记。
    # ``pocmol_data.fixed_dist_torsion``：Tensor，形状为 (N, N)，扭转下保持距离的 0/1 矩阵。
    # ``pocmol_data.tor_twisted_pairs``：dict，可旋转键到断键两侧非轴原子的映射。
    # ``pocmol_data.tor_bond_mat``：Tensor，形状为 (N, N)，逐原子对可旋转键标记。
    # ``pocmol_data.path_mat``：Tensor，形状为 (N, N)，化学图最短路径，单位为键数。
    # ``pocmol_data.nbh_dict``：dict[int, list[int]]，逐原子一跳邻居编号。
    # ``pocmol_data.matches_graph``：LongTensor，形状为 (M_g, S_g)，忽略手性的压缩自同构映射。
    # ``pocmol_data.matches_iso``：LongTensor，形状为 (M_i, S_i)，考虑手性的压缩自同构映射。
    # ``pocmol_data.brics.subgraphs``：list[list[int]]，BRICS 片段原子编号。
    # ``pocmol_data.brics.anchors_list``：list[set[int]]，BRICS 片段锚原子编号。
    # ``pocmol_data.brics.nbh_subgraphs``：list[list[int]]，BRICS 片段邻接表。
    # ``pocmol_data.brics.connections``：dict[tuple[int, int], tuple[int, int]]，BRICS 片段连接锚原子对。
    # ``pocmol_data.mmpa.subgraphs``：list[list[int]]，MMPA 片段原子编号。
    # ``pocmol_data.mmpa.anchors_list``：list[set[int]]，MMPA 片段锚原子编号。
    # ``pocmol_data.mmpa.nbh_subgraphs``：list[list[int]]，MMPA 片段邻接表。
    # ``pocmol_data.mmpa.connections``：dict[tuple[int, int], tuple[int, int]]，MMPA 片段连接锚原子对。
    # ``mol``：RDKit Mol，去氢后的输入小分子，保留固定二维图和至少一个 conformer。
    pocmol_data, mol = get_input_from_file(input_ligand, pocket_pdb, return_mol=True, **pocmol_args)
    
    # Add peptide-specific information
    if is_pep:
        if input_ligand.endswith('.pdb'):  # Peptide docking from PDB
            pep_info = get_peptide_info(input_ligand)
            # Verify consistency (sanity check)
            assert torch.isclose(pocmol_data['pos_all_confs'][0], pep_info['peptide_pos'], 1e-2).all(), \
                'Molecule and peptide atoms may not match'
        elif 'peplen_' in input_ligand:  # Peptide design
            pep_info = add_pep_bb_data(pocmol_data)
        else:  # pepseq_{xxx} - peptide docking from sequence
            pep_info = {}
        pocmol_data.update(pep_info)
    
    return pocmol_data, pocket_pdb, mol



if __name__ == '__main__':
    # ``parser``：ArgumentParser，定义本模块 Docstring 中逐项列出的 7 个命令行入口字段。
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_task', type=str, default='configs/sample/examples/dock_pep_know_some.yml', help='task config')
    parser.add_argument('--config_model', type=str, default='configs/sample/pxm.yml', help='model config')
    parser.add_argument('--outdir', type=str, default='./outputs_use')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=0, help='batch size; by default use the value in the config file')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=-1, help='num_workers for dataloader; by default use the value in the train config file')
    # ``args.config_task``：str，任务 YAML 路径。
    # ``args.config_model``：str|None，模型与 checkpoint YAML 路径；None 表示只读取任务 YAML。
    # ``args.outdir``：str，本次采样实验的输出根目录。
    # ``args.device``：str，模型、批次和 noiser 使用的 PyTorch 设备字符串。
    # ``args.batch_size``：int，非零时覆盖任务 YAML 的候选批大小；0 表示使用配置值。
    # ``args.shuffle``：bool，是否打乱重复输入候选的 DataLoader 顺序。
    # ``args.num_workers``：int，非负值覆盖训练配置的 DataLoader worker 数；-1 表示继承训练配置。
    # ``args``：Namespace，保存以上七个命令行叶。
    args = parser.parse_args()

    # ``config.sample.seed``：int，基础随机种子。
    # ``config.sample.batch_size``：int，候选批大小。
    # ``config.sample.num_mols``：int|缺省，最多保存候选数。
    # ``config.sample.num_repeats``：int|缺省，完整候选数据集遍历次数。
    # ``config.sample.save_traj_prob``：float，逐候选保存轨迹的概率。
    # ``config.sample.save_output``：list[str]|缺省，额外保存的模型输出叶名。
    # ``config.model.checkpoint``：str，PocketXMol Lightning checkpoint 路径。
    # ``config.data.protein_path``：str，完整受体 PDB 路径。
    # ``config.data.input_ligand``：str，小分子 SDF/PDB 路径或 SMILES。
    # ``config.data.is_pep``：bool|缺省，小分子路径应为 False。
    # ``config.data.pocket_args.ref_ligand_path``：str|缺省，口袋裁剪参考配体路径。
    # ``config.data.pocket_args.pocket_coord``：list[float]|缺省，长度为 3 的世界坐标裁剪中心，单位 Å。
    # ``config.data.pocket_args.radius``：float|缺省，口袋裁剪半径，单位 Å。
    # ``config.data.pocket_args.criterion``：str|缺省，残基距离判据。
    # ``config.data.pocmol_args.data_id``：str|缺省，样本标识。
    # ``config.data.pocmol_args.pdbid``：str|缺省，受体标识。
    # ``config.data.transforms``：list[Mapping]|缺省，任务变换后的追加 transform 配置，按列表顺序执行。
    # ``config.transforms``：Mapping，训练 featurizer 配置；use 任务 YAML 可覆盖其叶。
    # ``config.task.name``：str，图级任务名，小分子 docking 为 ``dock``。
    # ``config.task.transform``：Mapping，传给 ``get_transforms`` 的任务 prompt 配置。
    # ``config.noise``：Mapping，传给 ``get_sample_noiser`` 的采样 noiser 配置。
    # ``config``：EasyDict，以上任务 YAML 与模型 YAML 叶按 ``make_config`` 规则深合并后的结果。
    config = make_config(args.config_task, args.config_model)
    if args.config_model is not None:
        # ``config_name``：str，先取不含扩展名的任务 YAML 文件名。
        config_name = os.path.basename(args.config_task).replace('.yml', '') 
        # ``config_name``：str，追加模型 YAML 文件名，作为日志目录前缀以区分 checkpoint 配置。
        config_name += '_' + os.path.basename(args.config_model).replace('.yml', '')
    else:
        # ``config_name``：str，无独立模型 YAML 时只使用任务配置文件名作为日志目录前缀。
        config_name = os.path.basename(args.config_task)[:os.path.basename(args.config_task).rfind('.')]
    # ``s``：str，当前参与种子派生的输出目录或任务配置路径中的一个字符。
    # ``seed``：NumPy 整数标量，把 YAML 基础种子与输出目录/任务路径字符码相加，使不同调用路径获得稳定但不同的随机流。
    seed = config.sample.seed + np.sum([ord(s) for s in args.outdir]+[ord(s) for s in args.config_task])
    # 同步 Python、NumPy 与 PyTorch 随机源；noiser、DataLoader 和轨迹抽样共享该完整种子。
    seed_all(seed)
    # ``config.sample.complete_seed``：int，把实际使用的完整种子写回配置快照，便于复现实验。
    config.sample.complete_seed = seed.item()
    # ``ckpt.state_dict``：dict[str, Tensor]，Lightning 参数映射；只加载 ``model.`` 前缀叶。
    # ``ckpt``：checkpoint 映射，按 ``args.device`` 加载以避免设备不匹配。
    ckpt = torch.load(config.model.checkpoint, map_location=args.device, weights_only=False)
    # ``cfg_dir``：str，约定把 checkpoint 路径中的 ``checkpoints`` 目录名替换为同级 ``train_config``。
    cfg_dir = os.path.dirname(config.model.checkpoint).replace('checkpoints', 'train_config')
    # ``train_config``：list[str]，训练配置目录中的文件名列表；既有实现假定拼接后唯一定位一个 YAML。
    train_config = os.listdir(cfg_dir)
    # ``train_config.model``：Mapping，恢复 PMAsymDenoiser 结构与输出头。
    # ``train_config.transforms``：Mapping，恢复口袋/配体 featurizer 配置。
    # ``train_config.noise``：Mapping，提供 ``prior=from_train`` 的训练任务先验。
    # ``train_config.train.num_workers``：int，命令行未覆盖时的 DataLoader worker 数。
    # ``train_config.train.pin_memory``：bool，DataLoader 页锁定内存开关。
    # ``train_config``：EasyDict，从 checkpoint 相邻训练 YAML 读取的以上叶字段。
    train_config = make_config(os.path.join(cfg_dir, ''.join(train_config)))

    # ``save_traj_prob``：float，范围应为 ``[0, 1]``，每个候选独立保存完整轨迹的概率。
    save_traj_prob = config.sample.save_traj_prob
    # ``batch_size``：int，命令行非零值优先，否则读取任务 YAML 的候选批大小。
    batch_size = config.sample.batch_size if args.batch_size == 0 else args.batch_size
    # ``num_mols``：int，本次单输入最多保存的候选数，缺省为 100。
    num_mols = config.sample.get('num_mols', 100)
    # ``num_repeats``：int，完整遍历复制候选数据集的轮数，缺省为 1。
    num_repeats = config.sample.get('num_repeats', 1)

    # ``log_root``：str，用户指定的实验输出根目录。
    log_root = args.outdir
    # ``log_dir``：str，本次新建的唯一实验目录，目录名以前述 ``config_name`` 为前缀。
    log_dir = get_new_log_dir(log_root, prefix=config_name)
    # ``logger``：logging.Logger，文本日志写入 ``log_dir``。
    logger = get_logger('sample', log_dir)
    # writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info('Load from %s...' % config.model.checkpoint)
    logger.info(args)
    logger.info(config)
    save_config(config, os.path.join(log_dir, os.path.basename(args.config_task)))
    # for script_dir in ['scripts', 'utils', 'models']:
    #     shutil.copytree(script_dir, os.path.join(log_dir, script_dir))
    # ``sdf_dir``：str，轨迹 SDF 与可选原始输出张量 ``.pt`` 的目录。
    sdf_dir = os.path.join(log_dir, 'SDF')
    # ``pure_sdf_dir``：str，最终候选和输入结构目录，命名为 ``<实验目录名>_SDF``。
    pure_sdf_dir = os.path.join(log_dir, os.path.basename(log_dir) +'_SDF')
    os.makedirs(sdf_dir, exist_ok=True)
    os.makedirs(pure_sdf_dir, exist_ok=True)
    # ``df_path``：str，候选级元数据与 confidence 表 ``gen_info.csv`` 的路径。
    df_path = os.path.join(log_dir, 'gen_info.csv')

    logger.info('Loading data placeholder...')
    # ``samp_trans``：str，任务 YAML 中希望覆盖训练配置的 featurizer/transform 名称。
    for samp_trans in config.get('transforms', {}).keys():  # overwirte transform config from sample.yml to train.yml
        if samp_trans in train_config.transforms.keys():
            # 只更新训练配置中已存在的同名 transform 叶字段，例如 docking 示例的口袋中心。
            train_config.transforms.get(samp_trans).update(
                config.transforms.get(samp_trans)
            )
    # ``dm``：DataModule，仅借用训练配置重建 featurizer 列表和模型输入维度，不创建训练集。
    dm = DataModule(train_config)
    # ``featurizer_list``：list[callable]，训练时的数据特征化链，顺序决定字段依赖。
    featurizer_list = dm.get_featurizers()
    # ``featurizer``：FeaturizeMol，特征化链最后一项；其 ``decode_output`` 把模型类别/坐标恢复为分子叶字段。
    featurizer = featurizer_list[-1]
    # ``in_dims.num_node_types``：int，含可选 mask 的配体原子类别总数 C_n。
    # ``in_dims.num_edge_types``：int，含非键和可选 mask 的完整半边类别总数 C_e。
    # ``in_dims.pocket_in_dim``：int，口袋原子输入特征宽度 D_p_raw；当前标准口袋特征为 25。
    # ``in_dims``：dict[str, int]，模型构造与 noiser 使用的上述输入维度叶。
    in_dims = dm.get_in_dims()
    # ``task_trans``：callable，use 模式的 docking task transform，建立 fixed prompt、任务标签和结构化运动注释。
    task_trans = get_transforms(config.task.transform, mode='use')
    is_ar = config.task.transform.get('name', '')
    # ``noiser``：DockSamplNoiser，读取任务采样配置并用训练配置解析 ``prior: from_train``。
    noiser = get_sample_noiser(config.noise, in_dims['num_node_types'], in_dims['num_edge_types'],
                               mode='sample',device=args.device, ref_config=train_config.noise)
    if 'variable_mol_size' in getattr(config, 'transforms', []):  # mol design
        transforms = featurizer_list + [
            get_transforms(config.transforms.variable_mol_size), task_trans]
    elif 'variable_sc_size' in getattr(config, 'transforms', []):  # pep design
        transforms = featurizer_list + [
            get_transforms(config.transforms.variable_sc_size), task_trans]
    else:
        # ``transforms``：list[callable]，小分子 docking 依次执行训练 featurizer 与 docking task transform。
        transforms = featurizer_list + [task_trans]
    # ``tr``：Mapping，``config.data.transforms`` 中当前待实例化的附加变换配置。
    # ``addition_transforms``：list[callable]，任务 YAML ``data.transforms`` 额外声明的数据变换；示例默认为空。
    addition_transforms = [get_transforms(tr) for tr in config.data.get('transforms', [])]
    # ``transforms``：Compose，按列表顺序串行执行全部字段构造变换。
    transforms = Compose(transforms + addition_transforms)
    # ``t``：callable，Compose 中当前变换；读取其声明的批归属跟踪字段。
    # ``follow_batch``：list[str]，合并各 transform 声明的 PyG 叶字段，为其额外生成 ``<key>_batch`` 图归属索引。
    follow_batch = sum([getattr(t, 'follow_batch', []) for t in transforms.transforms], [])
    # ``t``：callable，Compose 中当前变换；读取其声明的不参与批拼接字段。
    # ``exclude_keys``：list[str]，合并各 transform 声明的非张量/不应拼接字段，传给 PyG DataLoader。
    exclude_keys = sum([getattr(t, 'exclude_keys', []) for t in transforms.transforms], [])
    
    # # Data loader
    logger.info('Loading dataset...')
    # ``data_cfg.protein_path``：str，完整受体 PDB 路径。
    # ``data_cfg.input_ligand``：str，小分子 SDF/PDB 路径或 SMILES。
    # ``data_cfg.is_pep``：bool|缺省，小分子路径应为 False。
    # ``data_cfg.pocket_args.ref_ligand_path``：str|缺省，口袋裁剪参考配体。
    # ``data_cfg.pocket_args.pocket_coord``：list[float]|缺省，长度为 3 的裁剪中心，单位 Å。
    # ``data_cfg.pocket_args.radius``：float|缺省，口袋裁剪半径，单位 Å。
    # ``data_cfg.pocket_args.criterion``：str|缺省，残基距离判据。
    # ``data_cfg.pocmol_args.data_id``：str|缺省，样本标识。
    # ``data_cfg.pocmol_args.pdbid``：str|缺省，受体标识。
    # ``data_cfg.transforms``：list[Mapping]|缺省，任务变换之后追加的 transform 配置。
    # ``data_cfg``：EasyDict，任务 YAML 的上述输入数据叶。
    data_cfg = config.data
    # ``is_pep``：bool|None，优先读取显式任务字段；小分子示例固定为 False。
    is_pep = data_cfg.get('is_pep', None)
    if is_pep is None:
        is_pep = data_cfg.input_ligand.endswith('.pdb') or data_cfg.input_ligand.startswith('pep')
    # ``data.element``：LongTensor，形状为 (N,)，配体原子序数。
    # ``data.pos_all_confs``：FloatTensor，形状为 (C, N, 3)，配体世界坐标，单位 Å。
    # ``data.i_conf_list``：list[int]，长度为 C，合法 conformer 输入编号。
    # ``data.num_confs``：int 标量 C，合法 conformer 数。
    # ``data.bond_index``：LongTensor，形状为 (2, 2M)，双向化学键端点。
    # ``data.bond_type``：LongTensor，形状为 (2M,)，逐双向键类别。
    # ``data.num_atoms``：int 标量 N，配体原子数。
    # ``data.num_bonds``：int 标量 M，无向化学键数。
    # ``data.pocket_element``：LongTensor，形状为 (P,)，口袋原子序数。
    # ``data.pocket_pos``：FloatTensor，形状为 (P, 3)，口袋世界坐标，单位 Å。
    # ``data.pocket_is_backbone``：BoolTensor，形状为 (P,)，口袋原子主链标记。
    # ``data.pocket_atom_name``：list[str]，长度为 P，PDB 原子名。
    # ``data.pocket_atom_to_aa_type``：LongTensor，形状为 (P,)，口袋原子氨基酸类别。
    # ``data.pocket_molecule_name``：str|None，口袋 PDB ``HEADER`` 名称。
    # ``data.pdbid``：str，受体结构标识。
    # ``data.data_id``：str，样本标识。
    # ``data.smiles``：str，固定二维配体图的规范 SMILES。
    # ``data.bond_rotatable``：LongTensor，形状为 (2M,)，可旋转键标记。
    # ``data.fixed_dist_torsion``：Tensor，形状为 (N, N)，扭转下保持距离的 0/1 矩阵。
    # ``data.tor_twisted_pairs``：dict，可旋转键到断键两侧非轴原子的映射。
    # ``data.tor_bond_mat``：Tensor，形状为 (N, N)，逐原子对可旋转键标记。
    # ``data.path_mat``：Tensor，形状为 (N, N)，化学图最短路径，单位为键数。
    # ``data.nbh_dict``：dict[int, list[int]]，逐原子一跳邻居编号。
    # ``data.matches_graph``：LongTensor，形状为 (M_g, S_g)，忽略手性的压缩自同构映射。
    # ``data.matches_iso``：LongTensor，形状为 (M_i, S_i)，考虑手性的压缩自同构映射。
    # ``data.brics.subgraphs``：list[list[int]]，BRICS 片段原子编号。
    # ``data.brics.anchors_list``：list[set[int]]，BRICS 片段锚原子编号。
    # ``data.brics.nbh_subgraphs``：list[list[int]]，BRICS 片段邻接表。
    # ``data.brics.connections``：dict[tuple[int, int], tuple[int, int]]，BRICS 片段连接锚原子对。
    # ``data.mmpa.subgraphs``：list[list[int]]，MMPA 片段原子编号。
    # ``data.mmpa.anchors_list``：list[set[int]]，MMPA 片段锚原子编号。
    # ``data.mmpa.nbh_subgraphs``：list[list[int]]，MMPA 片段邻接表。
    # ``data.mmpa.connections``：dict[tuple[int, int], tuple[int, int]]，MMPA 片段连接锚原子对。
    # ``pocket_block``：str，裁剪口袋的 PDB block。
    # ``in_mol``：RDKit Mol，固定小分子拓扑与起始 conformer，原子顺序与 ``data.element`` 对齐。
    data, pocket_block, in_mol = get_input_data(
        protein_path=data_cfg.protein_path,
        input_ligand=data_cfg.get('input_ligand', None),
        is_pep=is_pep,
        pocket_args=data_cfg.get('pocket_args', {}),
        pocmol_args=data_cfg.get('pocmol_args', {})
    )
    # ``test_set``：UseDataset，逻辑长度为 ``num_mols``，每次深拷贝同一输入记录并应用随机采样变换以形成独立候选。
    test_set = UseDataset(data, n=num_mols, task=config.task.name, transforms=transforms)

    # ``test_loader``：PyG DataLoader，把候选图拼接为 Batch；节点、半边和口袋实体沿各自首维串联。
    test_loader = DataLoader(test_set, batch_size, shuffle=args.shuffle,
                            num_workers = train_config.train.num_workers if args.num_workers == -1 else args.num_workers,
                            pin_memory = train_config.train.pin_memory,
                            follow_batch=follow_batch, exclude_keys=exclude_keys)
    # save pocket and mol
    # ``input_pocmol_dir``：str，保存本次实际使用的裁剪口袋和输入小分子，作为候选坐标系/拓扑证据。
    input_pocmol_dir = os.path.join(pure_sdf_dir, '0_inputs')
    os.makedirs(input_pocmol_dir, exist_ok=True)
    # ``f``：文本文件句柄，写出裁剪后的口袋 PDB block。
    with open(os.path.join(input_pocmol_dir, 'pocket_block.pdb'), 'w') as f:
        f.write(pocket_block)
    Chem.MolToMolFile(in_mol, os.path.join(input_pocmol_dir, 'input_mol.sdf'))

    logger.info('Loading diffusion model...')
    if train_config.model.name == 'pm_asym_denoiser':
        # ``model``：PMAsymDenoiser，按 checkpoint 的训练配置与输入类别维度实例化到目标设备。
        model = PMAsymDenoiser(config=train_config.model, **in_dims).to(args.device)
    # ``k``：str，checkpoint 中当前参数全名；只保留 ``model.`` 前缀并在加载前移除该前缀。
    # ``value``：Tensor，当前 checkpoint 参数值；形状必须与去前缀后的模型参数一致。
    model.load_state_dict({k[6:]:value for k, value in ckpt['state_dict'].items() if k.startswith('model.')}) # prefix is 'model'
    model.eval()

    # ``pool.succ``：list[dict]，成功得到单连通分子的候选记录。
    # ``pool.bad``：list[dict]，触发 ``MolReconsError`` 的候选记录。
    # ``pool.incomp``：list[dict]，重建后含多个连通分量的候选记录。
    # ``pool``：EasyDict，候选级结果池；每个列表叶按生成顺序保存 ``mol_info`` 映射。
    pool = EasyDict({
        # ``pool.succ``：list[dict]，单连通重建成功候选。
        'succ': [],
        # ``pool.bad``：list[dict]，重建失败候选。
        'bad': [],
        # ``pool.incomp``：list[dict]，多连通分量候选。
        'incomp': [],
        **({'nonstd': []} if is_pep else {})
    })
    # ``info_keys[0]``：``data_id``，输入样本标识的字段名。
    # ``info_keys[1]``：``db``，逻辑数据来源的字段名。
    # ``info_keys[2]``：``task``，图级任务名的字段名。
    # ``info_keys[3]``：``key``，use 数据集候选定位键的字段名。
    # ``info_keys``：list[str]，从 Batch 拆回每个候选并写入 ``mol_info/gen_info.csv`` 的定位与任务叶名。
    info_keys = [
        'data_id',
        'db',
        'task',
        'key',
    ]
    # ``i_saved``：int，当前已落盘候选总数，同时作为输出文件名前缀，初始为 0。
    i_saved = 0
    # generating molecules
    logger.info('Start sampling... (Total: n_mols=%d)' % (num_mols))
    
    try:
        # ``i_repeat``：int，当前完整采样轮编号，取值范围为 ``[0, num_repeats)``，写入每个候选索引行。
        for i_repeat in range(num_repeats):
            logger.info(f'Generating molecules.')
            # ``batch``：PyG Batch，逐次从 ``test_loader`` 取得，含 B 个候选图及拼接后的 N 个原子、H 条半边和 P 个口袋节点。
            for batch in test_loader:
                if i_saved >= num_mols:
                    logger.info('Enough molecules. Stop sampling.')
                    break
                
                # ``batch``：同一 PyG Batch 的设备版本；全部张量叶迁移到 ``args.device``。
                batch = batch.to(args.device)
                # ``batch``：PyG Batch，采样结束后的最终状态；配体实体维仍按本批 B 个图拼接，任务条件叶保持不变。
                # ``outputs.pred_node``：FloatTensor，形状为 (N, C_n)，末步原子类别 logits。
                # ``outputs.pred_pos``：FloatTensor，形状为 (N, 3)，末步干净局部坐标预测，单位 Å。
                # ``outputs.pred_halfedge``：FloatTensor，形状为 (H, C_e)，末步完整半边类别 logits。
                # ``outputs.confidence_node``：FloatTensor，形状为 (N, 1)，末步原子类别 confidence 原始输出。
                # ``outputs.confidence_pos``：FloatTensor，形状为 (N, 1)，末步坐标 confidence 原始输出。
                # ``outputs.confidence_halfedge``：FloatTensor，形状为 (H, 1)，末步半边类别 confidence 原始输出。
                # ``outputs.confidence_pos_traj``：FloatTensor，形状为 (N, T_total)，逐原子逐步坐标 confidence 原始输出。
                # ``outputs``：dict[str, Tensor]，保存上述末步预测和 confidence 叶。
                # ``trajs.all.node``：ndarray，形状为 (S_all, N)，干净参照、逐步输入与逐步投影状态的原子类别轨迹。
                # ``trajs.all.pos``：ndarray，形状为 (S_all, N, 3)，干净参照、逐步输入与逐步投影状态的坐标轨迹，单位 Å。
                # ``trajs.all.halfedge``：ndarray，形状为 (S_all, H)，干净参照、逐步输入与逐步投影状态的半边类别轨迹。
                # ``trajs.in.node``：ndarray，形状为 (T_total, N)，逐步带噪原子类别轨迹。
                # ``trajs.in.pos``：ndarray，形状为 (T_total, N, 3)，逐步带噪坐标轨迹，单位 Å。
                # ``trajs.in.halfedge``：ndarray，形状为 (T_total, H)，逐步带噪半边类别轨迹。
                # ``trajs.out.node``：ndarray，形状为 (T_total, N)，逐步投影后原子类别轨迹。
                # ``trajs.out.pos``：ndarray，形状为 (T_total, N, 3)，逐步投影后坐标轨迹，单位 Å。
                # ``trajs.out.halfedge``：ndarray，形状为 (T_total, H)，逐步投影后半边类别轨迹。
                # ``trajs.raw.node``：ndarray，形状为 (T_total, N)，逐步原始原子类别 argmax 轨迹。
                # ``trajs.raw.pos``：ndarray，形状为 (T_total, N, 3)，逐步原始坐标预测轨迹，单位 Å。
                # ``trajs.raw.halfedge``：ndarray，形状为 (T_total, H)，逐步原始半边类别 argmax 轨迹。
                # ``trajs``：dict[str, dict[str, ndarray]]，保存上述四个来源的三个轨迹叶。
                batch, outputs, trajs = sample_loop3(batch, model, noiser, args.device, is_ar=is_ar)
                
                # ``key``：str，当前从 Batch 拆出的元数据叶名，取 ``data_id``、``db``、``task`` 或 ``key``。
                # ``i``：int，源码按 ``range(len(batch))`` 生成；常见 PyG 中 ``len(Batch)`` 是字段数而非图数，疑似与真实候选数错位。
                # ``data_list``：list[dict]，意图逐图收集 ``info_keys`` 的四个元数据叶；本学习分支只记录潜在长度问题。
                # ``data_list[i].data_id``：str，第 i 个候选所属输入样本标识。
                # ``data_list[i].db``：str，第 i 个候选的逻辑数据来源。
                # ``data_list[i].task``：str，第 i 个候选的图级任务名。
                # ``data_list[i].key``：str，第 i 个候选在 use 数据集中的定位键。
                data_list = [{key:batch[key][i] for key in info_keys} for i in range(len(batch))]
                # ``generated_list``：list[dict]，长度为 B；每项是一个候选的解码输入映射。
                # ``generated_list[b].node``：ndarray，形状为 (N_b,)，候选 b 的最终原子类别索引。
                # ``generated_list[b].pos``：ndarray，形状为 (N_b, 3)，候选 b 的最终局部配体坐标，单位 Å。
                # ``generated_list[b].halfedge``：ndarray，形状为 (H_b,)，候选 b 的最终完整半边类别索引。
                # ``generated_list[b].halfedge_index``：int64 ndarray，形状为 (2, H_b)，端点索引候选 b 的 N_b 个原子。
                # ``generated_list[b].pocket_center``：ndarray，形状为 (1, 3)，解码时加回局部坐标的口袋平移中心，单位 Å。
                # ``outputs_list``：list[dict]，长度为 B；每项保存可按实体维拆分的单候选 CPU Tensor 输出。
                # ``outputs_list[b].pred_node``：Tensor|缺省，形状为 (N_b, C_n)，候选 b 的原子类别 logits。
                # ``outputs_list[b].pred_pos``：Tensor|缺省，形状为 (N_b, 3)，候选 b 的局部坐标预测，单位 Å。
                # ``outputs_list[b].pred_halfedge``：Tensor|缺省，形状为 (H_b, C_e)，候选 b 的半边类别 logits。
                # ``outputs_list[b].confidence_node``：Tensor|缺省，形状为 (N_b, 1)，候选 b 的原子类别 confidence 原始输出。
                # ``outputs_list[b].confidence_pos``：Tensor|缺省，形状为 (N_b, 1)，候选 b 的坐标 confidence 原始输出。
                # ``outputs_list[b].confidence_halfedge``：Tensor|缺省，形状为 (H_b, 1)，候选 b 的半边 confidence 原始输出。
                # ``outputs_list[b].confidence_pos_traj``：Tensor|缺省，形状为 (N_b, T_total)，候选 b 的坐标 confidence 轨迹。
                # ``outputs_list[b].halfedge_index``：Tensor，形状为 (2, H_b)，端点索引候选 b 的 N_b 个原子。
                # ``outputs_list[b].pocket_center``：Tensor，通常形状为 (1, 3)，候选 b 的口袋中心，单位 Å。
                # ``traj_list_dict``：dict[str, list[dict]]|list，存在轨迹时按来源映射 B 个单候选轨迹；无轨迹时为空列表。
                # ``traj_list_dict[source][b].node``：ndarray，形状为 (S, N_b)，候选 b 的原子类别时间轨迹。
                # ``traj_list_dict[source][b].pos``：ndarray，形状为 (S, N_b, 3)，候选 b 的局部坐标时间轨迹，单位 Å。
                # ``traj_list_dict[source][b].halfedge``：ndarray，形状为 (S, H_b)，候选 b 的半边类别时间轨迹。
                generated_list, outputs_list, traj_list_dict = seperate_outputs2(batch, outputs, trajs)
                
                # ``mol_info_list``：list[dict]，按候选顺序累积完成解码、固定拓扑重建与可选轨迹序列化的记录。
                mol_info_list = []
                # ``i_mol``：int，当前批内候选编号，同时索引 ``generated_list/outputs_list/traj_list_dict/data_list``。
                for i_mol in tqdm(range(len(generated_list)), desc='Post process generated mols'):
                    # ``mol_info.atom_pos``：ndarray，形状为 (N_b, 3)，已加回口袋中心的候选世界坐标，单位 Å。
                    # ``mol_info.element``：ndarray，形状为 (N_b,)，由原子类别索引反查的原子序数。
                    # ``mol_info.bond_index``：ndarray，形状为 (2, E_b)，预测为真实键的双向端点索引。
                    # ``mol_info.bond_type``：ndarray，形状为 (E_b,)，与 ``bond_index`` 列对齐的 RDKit 键类别编号。
                    # ``mol_info``：dict，先保存以上四个解码分子叶。
                    mol_info = featurizer.decode_output(**generated_list[i_mol]) 
                    # ``mol_info.data_id``：str，当前候选复制自输入记录的样本标识。
                    # ``mol_info.db``：str，固定为 ``use`` 的逻辑数据来源。
                    # ``mol_info.task``：str，小分子 docking 配置写入 ``dock``。
                    # ``mol_info.key``：str，当前候选在 use 数据集中的定位键。
                    # ``mol_info``：dict，追加上述四个定位叶；依赖 ``data_list`` 与候选拆分顺序一致。
                    mol_info.update(data_list[i_mol])  # add data info
                    
                    try:
                        if not is_pep:
                            with CaptureLogger():
                                # ``rdmol``：RDKit Mol，深拷贝 ``in_mol`` 的固定二维图并把 conformer 0 覆盖为当前生成坐标。
                                rdmol = reconstruct_from_generated_with_edges(mol_info, in_mol=in_mol)
                            # ``smiles``：str，由重建分子的二维图生成；固定拓扑成功时应与输入小分子身份一致。
                            smiles = Chem.MolToSmiles(rdmol)
                            if '.' in smiles:
                                # ``tag``：str，``incomp`` 表示重建分子含多个连通分量。
                                tag = 'incomp'
                                pool.incomp.append(mol_info)
                                logger.warning('Incomplete molecule: %s' % smiles)
                            else:
                                # ``tag``：空字符串，表示小分子重建成功且为单连通分量。
                                tag = ''
                                pool.succ.append(mol_info)
                                logger.info('Success: %s' % smiles)
                        else:
                            with CaptureLogger():
                                pdb_struc, rdmol = reconstruct_pdb_from_generated(mol_info, gt_path=data_cfg.input_ligand)
                            aaseq = seq1(''.join(res.resname for res in pdb_struc.get_residues()))
                            if rdmol is None:
                                rdmol = Chem.MolFromSmiles('')
                            smiles = Chem.MolToSmiles(rdmol)
                            if '.' in smiles:
                                tag = 'incomp'
                                pool.incomp.append(mol_info)
                                logger.warning('Incomplete molecule: %s' % aaseq)
                            elif 'X' in aaseq:
                                tag = 'nonstd'
                                pool.nonstd.append(mol_info)
                                logger.warning('Non-standard amino acid: %s' % aaseq)
                            else:  # nb
                                tag = ''
                                pool.succ.append(mol_info)
                                logger.info('Success: %s' % aaseq)
                    except MolReconsError:
                        pool.bad.append(mol_info)
                        logger.warning('Reconstruction error encountered.')
                        # ``smiles``：空字符串，重建失败时没有可用 RDKit 身份表示。
                        smiles = ''
                        # ``tag``：str，``bad`` 表示只能保存未经 sanitize 的原始 V2000 文本。
                        tag = 'bad'
                        # ``rdmol``：str，重建失败时变量类型由 RDKit Mol 变为 ``create_sdf_string`` 生成的 mol block。
                        rdmol = create_sdf_string(mol_info)
                        if is_pep:
                            aaseq = ''
                            pdb_struc = PDB.Structure.Structure('bad')
                    
                    # ``mol_info.rdmol``：RDKit Mol|str，成功时为固定拓扑分子，失败时为原始 V2000 mol block。
                    # ``mol_info.smiles``：str，成功/不完整候选的规范 SMILES，失败时为空。
                    # ``mol_info.tag``：str，取 ``''``、``incomp`` 或 ``bad``。
                    # ``mol_info.output``：dict，单候选主输出、confidence 和 confidence 轨迹叶。
                    # ``mol_info.pdb_struc``：Bio.PDB Structure|缺省，仅肽路径写入；小分子 docking 不存在该叶。
                    # ``mol_info.aaseq``：str|缺省，仅肽路径写入；小分子 docking 不存在该叶。
                    # ``mol_info``：dict，追加上述重建、身份、状态、输出及可选肽叶。
                    mol_info.update({
                        # ``mol_info.rdmol``：RDKit Mol|str，固定拓扑重建结果或回退 V2000 mol block。
                        'rdmol': rdmol,
                        # ``mol_info.smiles``：str，规范分子身份；重建失败时为空字符串。
                        'smiles': smiles,
                        # ``mol_info.tag``：str，候选重建状态。
                        'tag': tag,
                        # ``mol_info.output``：dict[str, Tensor]，单候选模型输出和 confidence 叶。
                        'output': outputs_list[i_mol],
                        **({
                            # ``mol_info.pdb_struc``：Bio.PDB Structure，仅肽路径的重建蛋白结构。
                            'pdb_struc': pdb_struc,
                            # ``mol_info.aaseq``：str，仅肽路径的重建氨基酸序列。
                            'aaseq': aaseq,
                        } if is_pep else {})
                    })
                    
                    # ``p_save_traj``：float，范围为 ``[0, 1)``，当前候选独立的轨迹落盘抽样值。
                    p_save_traj = np.random.rand()
                    if p_save_traj <  save_traj_prob:
                        # ``mol_traj``：dict[str, list[str]]，从轨迹类别映射到逐时间步 V2000 mol block 列表。
                        mol_traj = {}
                        # ``traj_who``：str，逐次取 ``traj_list_dict`` 的轨迹来源键，通常为 ``in/out/raw_out``。
                        for traj_who in traj_list_dict.keys():
                            # ``traj_this_mol.node``：ndarray，形状为 (S, N_b)，当前候选当前来源的原子类别时间轨迹。
                            # ``traj_this_mol.pos``：ndarray，形状为 (S, N_b, 3)，当前候选当前来源的局部坐标时间轨迹，单位 Å。
                            # ``traj_this_mol.halfedge``：ndarray，形状为 (S, H_b)，当前候选当前来源的半边类别时间轨迹。
                            # ``traj_this_mol``：dict，保存以上三个逐时间步叶。
                            traj_this_mol = traj_list_dict[traj_who][i_mol]
                            # ``t``：int，当前轨迹来源内的时间步编号，与三个轨迹叶的第 0 维对齐。
                            for t in range(len(traj_this_mol['node'])):
                                # ``mol_this.atom_pos``：ndarray，形状为 (N_b, 3)，当前时间步的世界坐标，单位 Å。
                                # ``mol_this.element``：ndarray，形状为 (N_b,)，当前时间步的原子序数。
                                # ``mol_this.bond_index``：ndarray，形状为 (2, E_t)，当前时间步预测真实键的双向端点索引。
                                # ``mol_this.bond_type``：ndarray，形状为 (E_t,)，与 ``bond_index`` 列对齐的键类别。
                                # ``mol_this``：dict，保存以上四个当前时间步解码叶。
                                mol_this = featurizer.decode_output(
                                        node=traj_this_mol['node'][t],
                                        pos=traj_this_mol['pos'][t],
                                        halfedge=traj_this_mol['halfedge'][t],
                                        halfedge_index=generated_list[i_mol]['halfedge_index'],
                                        pocket_center=generated_list[i_mol]['pocket_center'],
                                    )
                                # ``mol_this``：dict -> str，把当前时间步直接转为未经 sanitize 的 V2000 mol block，以容纳中间态无效价态。
                                mol_this = create_sdf_string(mol_this)
                                # ``mol_traj[traj_who]``：list[str]，按 t 递增累积当前来源的 mol block。
                                mol_traj.setdefault(traj_who, []).append(mol_this)
                                
                        # ``mol_info.traj``：dict[str, list[str]]，仅在本候选命中保存概率时存在。
                        mol_info['traj'] = mol_traj
                    # ``mol_info_list``：保持与 ``generated_list`` 相同的批内候选顺序。
                    mol_info_list.append(mol_info)

                # ``df_info_list``：list[dict]，当前批次准备追加到 ``gen_info.csv`` 的候选级扁平行。
                df_info_list = []
                # ``data_finished``：dict，逐次取已完成重建与可选轨迹处理的单候选记录。
                for data_finished in mol_info_list:
                    # ``rdmol``：RDKit Mol|str，从候选记录取出的重建结果。
                    rdmol = data_finished['rdmol']
                    # ``tag``：str，候选重建状态，取 ``''``、``incomp`` 或 ``bad``。
                    tag = data_finished['tag']
                    # ``filename_base``：str，全局保存编号加可选状态后缀，例如 ``3`` 或 ``3-bad``。
                    filename_base = str(i_saved) + (f'-{tag}' if tag else '')
                    # save pdb
                    if is_pep:
                        pdb_struc = data_finished['pdb_struc']
                        filename_pdb = filename_base + '.pdb'
                        pdb_io = PDBIO()
                        pdb_io.set_structure(pdb_struc)
                        pdb_io.save(os.path.join(pure_sdf_dir, filename_pdb))
                    # rdmol to sdf
                    # ``filename_sdf``：str，小分子路径为 ``<filename_base>.sdf``，写入最终候选目录。
                    filename_sdf = filename_base + ('.sdf' if not is_pep else '_mol.sdf')
                    if tag != 'bad':
                        Chem.MolToMolFile(rdmol, os.path.join(pure_sdf_dir, filename_sdf))
                    else:
                        # ``f``：文本文件句柄，重建失败时直接写出 ``rdmol`` 中的 V2000 mol block。
                        with open(os.path.join(pure_sdf_dir, filename_sdf), 'w+') as f:
                            f.write(rdmol)
                    # save traj
                    if 'traj' in data_finished:
                        # ``traj_who``：str，逐次取当前候选已保存的轨迹来源键。
                        for traj_who in data_finished['traj'].keys():
                            # ``sdf_file``：str，用 ``$$$$`` 记录分隔符拼接该候选当前来源的全部时间步 mol block。
                            sdf_file = '$$$$\n'.join(data_finished['traj'][traj_who])
                            # ``name_traj``：str，轨迹文件名 ``<filename_base>-<traj_who>.sdf``。
                            name_traj = filename_base + f'-{traj_who}.sdf'
                            # ``f``：文本文件句柄，写出多记录轨迹 SDF。
                            with open(os.path.join(sdf_dir, name_traj), 'w+') as f:
                                f.write(sdf_file)
                    # ``i_saved``：int，当前候选落盘完成后加 1，使下一候选使用新的全局文件编号。
                    i_saved += 1
                    # save output
                    # ``output.pred_node``：Tensor|缺省，形状为 (N_b, C_n)，当前候选原子类别 logits。
                    # ``output.pred_pos``：Tensor|缺省，形状为 (N_b, 3)，当前候选局部坐标预测，单位 Å。
                    # ``output.pred_halfedge``：Tensor|缺省，形状为 (H_b, C_e)，当前候选半边类别 logits。
                    # ``output.confidence_node``：Tensor，形状为 (N_b, 1)，当前候选原子类别 confidence 原始输出。
                    # ``output.confidence_pos``：Tensor，形状为 (N_b, 1)，当前候选坐标 confidence 原始输出。
                    # ``output.confidence_halfedge``：Tensor，形状为 (H_b, 1)，当前候选半边类别 confidence 原始输出。
                    # ``output.confidence_pos_traj``：Tensor，形状为 (N_b, T_total)，当前候选坐标 confidence 轨迹。
                    # ``output.halfedge_index``：Tensor，形状为 (2, H_b)，当前候选半边端点索引。
                    # ``output.pocket_center``：Tensor，通常形状为 (1, 3)，当前候选口袋中心，单位 Å。
                    # ``output``：dict[str, Tensor]，保存上述当前候选拆分后的 CPU Tensor 叶。
                    output = data_finished['output']
                    # ``cfd_traj``：float，对逐原子原始 ``confidence_pos_traj`` 先按原子均值，再按既有轮次选择规则汇总；不是统一 sigmoid 概率。
                    cfd_traj = get_cfd_traj(output['confidence_pos_traj'])  # get cfd
                    # ``cfd_pos``：float，当前候选 N_b 个原子的最终原始坐标 confidence 均值，未统一 sigmoid。
                    cfd_pos = output['confidence_pos'].detach().cpu().numpy().mean()
                    # ``cfd_node``：float，当前候选 N_b 个原子的最终原子类别 confidence 原始输出均值。
                    cfd_node = output['confidence_node'].detach().cpu().numpy().mean()
                    # ``cfd_edge``：float，当前候选 H_b 条半边的最终类别 confidence 原始输出均值。
                    cfd_edge = output['confidence_halfedge'].detach().cpu().numpy().mean()
                    # ``save_output``：list[str]，任务 YAML 可选声明需要额外保存为张量文件的输出叶名。
                    save_output = getattr(config.sample, 'save_output', [])
                    if len(save_output) > 0:
                        # ``key``：str，``save_output`` 中当前需要序列化的模型输出叶名。
                        # ``output``：dict[str, Tensor]，过滤为 ``save_output`` 明确列出的叶；每叶保持单候选 Tensor 形状。
                        output = {key: output[key] for key in save_output}
                        torch.save(output, os.path.join(sdf_dir, filename_base + '.pt'))

                    # log info 
                    # ``info_dict.data_id``：str，候选所属输入样本标识。
                    # ``info_dict.db``：str，候选逻辑数据来源；use 路径固定为 ``use``。
                    # ``info_dict.task``：str，小分子 docking 候选写入 ``dock``。
                    # ``info_dict.key``：str，use 数据集中当前重复候选的定位键。
                    # ``info_dict.aaseq``：str|缺省，仅肽路径写入；小分子 docking 不存在该叶。
                    # ``info_dict.smiles``：str，候选重建后的规范 SMILES；重建失败时为空。
                    # ``info_dict.tag``：str，候选重建状态，取 ``''``、``incomp`` 或 ``bad``。
                    # ``key``：str，当前复制到 CSV 行的候选元数据叶名。
                    # ``info_dict``：dict，先收集以上候选元数据叶。
                    info_dict = {
                        key: data_finished[key] for key in info_keys +
                        (['aaseq'] if is_pep else []) + ['smiles', 'tag']
                    }
                    # ``info_dict.filename``：str，最终候选文件名；小分子路径使用 SDF 文件名。
                    # ``info_dict.i_repeat``：int，当前候选所属的完整采样轮号。
                    # ``info_dict.cfd_traj``：float，坐标 confidence 轨迹的既有两阶段汇总值。
                    # ``info_dict.cfd_pos``：float，最终逐原子坐标 confidence 原始输出均值。
                    # ``info_dict.cfd_node``：float，最终逐原子类别 confidence 原始输出均值。
                    # ``info_dict.cfd_edge``：float，最终逐半边类别 confidence 原始输出均值。
                    # ``info_dict``：dict，追加上述文件定位、轮次和四个 confidence 汇总叶。
                    info_dict.update({
                        # ``info_dict.filename``：str，最终候选结构文件名。
                        'filename': filename_sdf if not is_pep else filename_pdb,
                        # ``info_dict.i_repeat``：int，当前完整采样轮号。
                        'i_repeat': i_repeat,
                        # ``info_dict.cfd_traj``：float，坐标 confidence 轨迹汇总值。
                        'cfd_traj': cfd_traj,
                        # ``info_dict.cfd_pos``：float，最终坐标 confidence 均值。
                        'cfd_pos': cfd_pos,
                        # ``info_dict.cfd_node``：float，最终原子 confidence 均值。
                        'cfd_node': cfd_node,
                        # ``info_dict.cfd_edge``：float，最终半边 confidence 均值。
                        'cfd_edge': cfd_edge,
                    })

                    # ``df_info_list``：保持与本批保存候选和全局 ``i_saved`` 增长顺序一致。
                    df_info_list.append(info_dict)
            
                # ``df_info_batch``：DataFrame，当前批每行一个已保存候选，列为 ``info_dict`` 全部叶的并集。
                df_info_batch = pd.DataFrame(df_info_list)
                if os.path.exists(df_path):
                    # ``df_info``：DataFrame，从已有 ``gen_info.csv`` 读取的前序批次候选行。
                    df_info = pd.read_csv(df_path)
                    # ``df_info``：DataFrame，按保存顺序把当前批候选追加到已有行并重建连续行索引。
                    df_info = pd.concat([df_info, df_info_batch], ignore_index=True)
                else:
                    # ``df_info``：DataFrame，首批直接使用当前候选行。
                    df_info = df_info_batch
                df_info.to_csv(df_path, index=False)
                print_pool_status(pool, logger, is_pep=is_pep)
                
                # clean up
                del batch, outputs, trajs, mol_info_list[0:len(mol_info_list)]
                if args.device != 'cpu':
                    with torch.cuda.device(args.device):
                        torch.cuda.empty_cache()
                gc.collect()


        # make dummy pool  (save disk space)
        # ``key``：str，小分子结果池类别名，取 ``succ``、``bad`` 或 ``incomp``。
        # ``value``：list[dict]，当前类别的候选记录列表；这里只读取长度。
        # ``dummy_pool``：dict[str, list[str]]，每个结果类别只保留与真实池等长的空字符串，避免序列化完整分子占用磁盘。
        dummy_pool = {key: ['']*len(value) for key, value in pool.items()}
        torch.save(dummy_pool, os.path.join(log_dir, 'samples_all.pt'))
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt. Stop sampling.')
