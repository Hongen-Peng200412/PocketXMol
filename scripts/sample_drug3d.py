"""批量执行分子构象生成/小分子 docking，并建立后续评测所需产物契约。

脚本把任务 YAML 与模型 checkpoint 配置合并，复用训练 featurizer，逐批执行
``sample_loop3``，把固定二维图与生成坐标重建为 SDF，并写出 ``gen_info.csv``。
后者是构象集合聚合、docking 辅助评分、置信度排序和 RMSD 评测的共同主索引。

本脚本包含仓库其他分子任务的兼容分支；本轮只解释 ``task.name=conf/dock`` 且
``is_pep=False`` 的路径。数值字段 ``cfd_*`` 都是模型原始置信度输出的均值，未在
落盘前统一 sigmoid；比较分数时必须与训练/排序脚本的同一语义对齐。
"""

from copy import deepcopy
import os

import sys
sys.path.append('.')
import shutil
import argparse
import gc
import torch
import torch.utils.tensorboard
import numpy as np
from itertools import cycle
from easydict import EasyDict
from tqdm.auto import tqdm
from rdkit import Chem
from torch_geometric.loader import DataLoader
from collections import OrderedDict

from scripts.train_pl import DataModule
from models.maskfill import PMAsymDenoiser
from models.sample import seperate_outputs2, sample_loop3, get_cfd_traj
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *
# from utils.chem import *
from utils.sample_noise import get_sample_noiser

def print_pool_status(pool, logger):
    """记录当前累计成功、非连通与重建失败的候选数。

    输入参数:
        - pool: EasyDict，候选结果列表容器。
        - pool.succ: list[dict]，重建为单连通合法分子的候选记录。
        - pool.incomp: list[dict]，重建后含多个连通分量的候选记录。
        - pool.bad: list[dict]，不能由 RDKit 完成重建的候选记录。
        - logger: logging.Logger，接收三类候选当前数量的日志器。

    返回值:
        - None: 只读取三个列表长度并写日志，不修改 ``pool``。
    """

    logger.info('[Pool] Succ/Incomp/Bad: %d/%d/%d' % (
        len(pool.succ), len(pool.incomp), len(pool.bad)
    ))

# ``is_vscode``：bool；只改变调试输出目录与采样预算，不进入模型输入。
is_vscode = False
if os.environ.get("TERM_PROGRAM") == "vscode":
    # ``is_vscode``：VS Code 终端下启用小规模、强制保存轨迹的调试协议。
    is_vscode = True

if __name__ == '__main__':
    # ``parser``：命令行参数解析器，定义任务配置、模型配置、输出和运行设备。
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_task', type=str, default='configs/sample/test/denovo_geom/base.yml', help='task config file')
    parser.add_argument('--config_model', type=str, default='configs/sample/pxm.yml', help='model config file')
    parser.add_argument('--outdir', type=str, default='outputs_test')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=1)
    # ``args.config_task``：str，构象或小分子 docking 任务 YAML 路径。
    # ``args.config_model``：str|None，模型与 checkpoint YAML 路径；None 表示只读取任务 YAML。
    # ``args.outdir``：str，采样实验输出根目录。
    # ``args.device``：str，模型、批次和 noiser 使用的 PyTorch 设备字符串。
    # ``args.batch_size``：int，非零时覆盖任务 YAML 候选批大小；0 表示使用配置值。
    # ``args.shuffle``：bool，是否打乱测试图的 DataLoader 顺序。
    # ``args.num_workers``：int，DataLoader worker 数。
    # ``args``：Namespace，保存以上七个命令行叶。
    args = parser.parse_args()

    # ``config.sample.seed``：int，采样随机种子。
    # ``config.sample.batch_size``：int，命令行未覆盖时的候选批大小。
    # ``config.sample.num_mols``：int|缺省，全实验最多落盘的候选数。
    # ``config.sample.num_repeats``：int|缺省，完整测试集遍历次数。
    # ``config.sample.save_traj_prob``：float，逐候选保存完整轨迹的概率。
    # ``config.sample.save_output``：list[str]|缺省，额外保存为 ``.pt`` 的模型输出叶名。
    # ``config.model.checkpoint``：str，PocketXMol Lightning checkpoint 路径。
    # ``config.data.dataset``：Mapping，测试 assembly 与逻辑 LMDB 布局配置。
    # ``config.data.split``：str|None，可选测试划分名。
    # ``config.data.transforms``：list[Mapping]|缺省，任务变换后追加的数据 transform 配置。
    # ``config.task.name``：str，图级任务名；本轮为 ``conf`` 或 ``dock``。
    # ``config.task.db``：str，测试使用的唯一逻辑数据库名。
    # ``config.task.transform``：Mapping，构象或 docking fixed prompt 与运动模式配置。
    # ``config.noise``：Mapping，构象或 docking 采样 noiser 配置。
    # ``config.transforms``：Mapping|缺省，对训练 featurizer 的采样期覆盖配置。
    # ``config``：EasyDict，按 ``make_config`` 规则深合并上述任务 YAML 与模型 YAML 叶。
    config = make_config(args.config_task, args.config_model)
    if args.config_model is not None:
        # ``config_name``：str，先取任务 YAML 文件干名，作为实验目录前缀的一部分。
        config_name = os.path.basename(args.config_task).replace('.yml', '') 
        # 追加模型 YAML 文件干名，区分同一任务使用的不同 checkpoint。
        config_name += '_' + os.path.basename(args.config_model).replace('.yml', '')
    else:
        # ``config_name``：未提供模型配置时只使用任务 YAML 文件干名。
        config_name = os.path.basename(args.config_task)[:os.path.basename(args.config_task).rfind('.')]
    # ``seed``：int，来自 ``sample.seed``；控制 noiser、DataLoader 与轨迹抽样等随机源。
    seed = config.sample.seed  # + np.sum([ord(s) for s in args.outdir]+[ord(s) for s in args.config_task])
    seed_all(seed)
    # config.sample.complete_seed = seed.item()
    # ``ckpt.state_dict``：dict[str, Tensor]，Lightning 参数映射；只加载 ``model.`` 前缀叶。
    # ``ckpt``：checkpoint 映射，按 ``args.device`` 加载以避免设备不匹配。
    ckpt = torch.load(config.model.checkpoint, map_location=args.device, weights_only=False)
    # ``cfg_dir``：str，把 checkpoint 目录约定中的 ``checkpoints`` 替换为 ``train_config``。
    cfg_dir = os.path.dirname(config.model.checkpoint).replace('checkpoints', 'train_config')
    # ``train_config``：list[str]，训练配置目录下的文件名；源码假定拼接后只有一个有效文件。
    train_config = os.listdir(cfg_dir)
    # ``train_config.model``：Mapping，恢复 PMAsymDenoiser 结构与输出头。
    # ``train_config.transforms``：Mapping，恢复口袋和配体 featurizer 配置。
    # ``train_config.noise``：Mapping，提供采样 ``prior: from_train`` 使用的训练任务先验。
    # ``train_config.train.num_workers``：int，训练期 DataLoader worker 数；当前脚本只在相应调用处读取。
    # ``train_config.train.pin_memory``：bool，测试 DataLoader 页锁定内存开关。
    # ``train_config``：EasyDict，从 checkpoint 相邻训练 YAML 读取的以上叶。
    train_config = make_config(os.path.join(cfg_dir, ''.join(train_config)))

    # ``save_traj_prob``：float [0,1]，每个已重建候选独立保存完整轨迹的概率。
    save_traj_prob = config.sample.save_traj_prob
    # ``batch_size``：int；CLI 非零值覆盖 YAML 的 sample.batch_size。
    batch_size = config.sample.batch_size if args.batch_size == 0 else args.batch_size
    # ``num_mols``：int，总落盘候选上限；缺省为近似无限。
    num_mols = getattr(config.sample, 'num_mols', int(1e10))
    # ``num_repeats``：int，完整遍历测试集的重复次数；不同 repeat 使用随机采样得到不同候选。
    num_repeats = getattr(config.sample, 'num_repeats', 1)
    if is_vscode:  # for debug using vscode
        # ``dir_names``：list[str]，任务 YAML 父目录按斜杠拆分后的层级。
        dir_names= os.path.dirname(args.config_task).split('/')
        # ``is_sample``：int，目录层级中 ``sample`` 的位置。
        is_sample = dir_names.index('sample')
        # ``names``：list[str]，用于复刻任务配置层级的调试目录片段。
        names = dir_names[is_sample+1:] + os.path.dirname(args.config_task).split('/')[1:]
        # ``log_root``：str，VS Code 调试输出根目录。
        log_root = '/'.join(
            [args.outdir.replace('outputs', 'outputs_vscode')] + names
        )
        # ``save_traj_prob``：调试模式对每个候选都保存轨迹。
        save_traj_prob = 1.0
        # ``batch_size``：调试模式固定小批量。
        batch_size = 11
        # ``num_mols``：调试模式最多保存 100 个候选。
        num_mols = 100
        # ``num_repeats``：调试模式只遍历测试集两次。
        num_repeats = 2
    else:
        # ``log_root``：str，正常运行的用户指定输出根目录。
        log_root = args.outdir
        os.makedirs(log_root, exist_ok=True)
        # remove bad result dir with the same name
        # ``file``：str，``log_root`` 下当前实验目录名；只检查与本次配置前缀相同的目录。
        for file in os.listdir(log_root):
            if file.startswith(config_name):
                if not os.path.exists(os.path.join(log_root, file, 'samples_all.pt')):
                    print('Remove bad result dir:', file)
                    # shutil.rmtree(os.path.join(log_root, file))
                else:
                    print('Found existing result dir:', file)
                    # exit()
    # ``log_dir``：str，带时间/递增后缀的新实验目录，避免覆盖已有完整结果。
    log_dir = get_new_log_dir(log_root, prefix=config_name)
    # ``logger``：同时写终端和实验目录日志的 Logger。
    logger = get_logger('sample', log_dir)
    # ``writer``：TensorBoard writer；当前脚本创建但未写标量。
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info('Load from %s...' % config.model.checkpoint)
    logger.info(args)
    logger.info(config)
    save_config(config, os.path.join(log_dir, os.path.basename(args.config_task)))
    # ``script_dir``：str，当前复制到实验目录留档的源码子目录名。
    for script_dir in ['scripts', 'utils', 'models']:
        shutil.copytree(script_dir, os.path.join(log_dir, script_dir))
    # ``sdf_dir``：str，当前实验的标准候选 SDF/可选 pt/轨迹文件目录。
    sdf_dir = os.path.join(log_dir, 'SDF')
    os.makedirs(sdf_dir, exist_ok=True)
    # ``df_path``：str，候选级主索引 ``gen_info.csv`` 的路径。
    df_path = os.path.join(log_dir, 'gen_info.csv')

    logger.info('Loading data placeholder...')
    # ``samp_trans``：str，采样配置中当前覆盖训练配置的 transform 子树名。
    for samp_trans in config.get('transforms', {}).keys():  # overwirte transform config from sample.yml to train.yml
        if samp_trans in train_config.transforms.keys():
            train_config.transforms.get(samp_trans).update(
                config.transforms.get(samp_trans)
            )
    # ``dm``：DataModule；此处只复用其训练时 featurizer 和输入维度构造逻辑。
    dm = DataModule(train_config)
    # ``featurizer_list``：按训练配置顺序实例化的 transform 列表。
    featurizer_list = dm.get_featurizers()
    # ``featurizer``：FeaturizeMol，负责把原子类别、半边类别和局部坐标反解为分子叶。
    featurizer = featurizer_list[-1]  # for mol decoding
    # ``in_dims.num_node_types``：int，含可选 mask 的原子类别总数 C_n。
    # ``in_dims.num_edge_types``：int，含非键和可选 mask 的完整半边类别总数 C_e。
    # ``in_dims``：dict[str, int]，模型构造与 noiser 使用的以上类别维度叶。
    in_dims = dm.get_in_dims()
    # ``task_trans``：构象或 docking transform；写 fixed prompt、运动域和 task 标签。
    task_trans = get_transforms(config.task.transform)
    # ``is_ar``：str transform 名称；conf/dock 不以 ar 开头，因此采样只执行标准轮次。
    is_ar = config.task.transform.name
    # ``noiser``：与 task.name 对应的采样 noiser；``from_train`` prior 从 train_config.noise 解析。
    noiser = get_sample_noiser(config.noise, in_dims['num_node_types'], in_dims['num_edge_types'],
                               mode='sample',device=args.device, ref_config=train_config.noise)
    if 'variable_mol_size' in getattr(config, 'transforms', []):
        # ``transforms``：变长生成兼容路径的 transform 列表；本轮固定图任务通常不进入。
        transforms = featurizer_list + [
            get_transforms(config.transforms.variable_mol_size), task_trans]
    else:
        # ``transforms``：本轮正常路径，先 featurize，再写任务 prompt/运动注释。
        transforms = featurizer_list + [task_trans]
    # ``tr``：Mapping，``config.data.transforms`` 中当前待实例化的附加变换配置。
    # ``addition_transforms``：list[callable]，测试 YAML ``data.transforms`` 中额外实例化的覆盖/起始位姿变换。
    addition_transforms = [get_transforms(tr) for tr in config.data.get('transforms', [])]
    # ``transforms``：可调用 Compose，按上面确定的顺序串联所有变换。
    transforms = Compose(transforms + addition_transforms)
    # ``t``：callable，Compose 中当前变换；读取其声明的批归属跟踪字段。
    # ``follow_batch``：list[str]，要求 PyG 为对应实体字段生成 ``*_batch`` 图归属向量。
    follow_batch = sum([getattr(t, 'follow_batch', []) for t in transforms.transforms], [])
    # ``t``：callable，Compose 中当前变换；读取其声明的不参与批拼接字段。
    # ``exclude_keys``：list[str]，组 Batch 时排除不可拼接或无需送入模型的字段。
    exclude_keys = sum([getattr(t, 'exclude_keys', []) for t in transforms.transforms], [])
    
    logger.info('Loading dataset...')
    # ``data_cfg.dataset.root``：str，测试数据共同根目录。
    # ``data_cfg.dataset.assembly_path``：str，相对根目录的测试 assembly 路径。
    # ``data_cfg.dataset.dbs[*].name``：str，逻辑数据库名。
    # ``data_cfg.dataset.dbs[*].lmdb_root``：str，逻辑数据库的子 LMDB 相对根目录。
    # ``data_cfg.dataset.dbs[*].lmdb_path``：dict[str, str]，子库名到 LMDB 文件名映射。
    # ``data_cfg.split``：str|None，可选测试划分名。
    # ``data_cfg.transforms``：list[Mapping]|缺省，任务变换之后追加的 transform 配置。
    # ``data_cfg``：EasyDict，保存以上测试数据定位与追加变换叶。
    data_cfg = config.data
    # ``num_workers``：int；CLI=-1 时回用训练配置，否则使用本次显式值。
    num_workers = train_config.train.num_workers if args.num_workers == -1 else args.num_workers
    # ``test_set``：固定一个 task/db 的 TestTaskDataset；每个索引是一张待采样分子图。
    test_set = TestTaskDataset(data_cfg.dataset, config.task,
                               mode='test',
                               split=getattr(data_cfg, 'split', None),
                               transforms=transforms)
    # ``test_loader``：PyG DataLoader；follow_batch 维持节点、半边和口袋实体到图的对齐。
    test_loader = DataLoader(test_set, batch_size, shuffle=args.shuffle,
                            num_workers = num_workers,
                            pin_memory = train_config.train.pin_memory,
                            follow_batch=follow_batch, exclude_keys=exclude_keys)

    logger.info('Loading diffusion model...')
    if train_config.model.name == 'pm_asym_denoiser':
        # ``model``：统一 PMAsymDenoiser，空口袋处理构象，非空 pocket_context 处理 docking。
        model = PMAsymDenoiser(config=train_config.model, **in_dims).to(args.device)
    # ``k``：str，checkpoint 中当前参数全名；只保留 ``model.`` 前缀并在加载前移除该前缀。
    # ``value``：Tensor，当前 checkpoint 参数值；形状必须与去前缀后的模型参数一致。
    model.load_state_dict({k[6:]:value for k, value in ckpt['state_dict'].items() if k.startswith('model.')}) # prefix is 'model'
    model.eval()

    # ``pool.succ``：list[dict]，重建为单连通合法分子的候选记录。
    # ``pool.bad``：list[dict]，RDKit 重建失败、只能保留原始 mol block 的候选记录。
    # ``pool.incomp``：list[dict]，重建后含多个连通分量的候选记录。
    # ``pool``：EasyDict，保存以上三类候选列表；结束时只持久化等长占位列表。
    pool = EasyDict({
        # ``succ``：list[dict]，单连通合法候选。
        'succ': [],
        # ``bad``：list[dict]，重建失败候选。
        'bad': [],
        # ``incomp``：list[dict]，多连通分量候选。
        'incomp': [],
    })
    # ``info_keys[0]``：``data_id``，样本标识。
    # ``info_keys[1]``：``db``，逻辑数据库名。
    # ``info_keys[2]``：``task``，图级任务名。
    # ``info_keys[3]``：``key``，assembly 解析出的实际复合 LMDB 键。
    # ``info_keys``：list[str]，按以上顺序从 Batch 拆回每个候选的四个元数据叶。
    info_keys = [
        'data_id',
        'db',
        'task',
        'key',
    ]
    # ``i_saved``：int，全实验单调递增的候选文件编号，也控制 num_mols 上限。
    i_saved = 0
    # generating molecules
    logger.info('Start sampling... (n_repeats=%d, n_mols=%d)' % (num_repeats, num_mols))
    # ``i_repeat``：int，当前完整测试集采样轮号，取值范围为 ``[0, num_repeats)``。
    for i_repeat in range(num_repeats):
        logger.info(f'Generating molecules. Testset repeat {i_repeat}.')
        
        if ('overwrite_pos_repeat' in config.sample or  # overwrite pos. and different for each repeat. for linker desigin with unknown frag pos
            'overwrite_mol_repeat' in config.sample):  # overwrite mol. for mol optimize round >= 1
            if 'overwrite_pos_repeat' in config.sample:
                # ``overwrite_pos_repeat``：每轮起始坐标覆盖配置；本轮常规 conf/dock 不使用。
                overwrite_pos_repeat = config.sample.overwrite_pos_repeat
                # ``overwirter``：按 i_repeat 选择起始坐标的 transform（变量名沿用源码拼写）。
                overwirter = OverwritePosRepeat(config=overwrite_pos_repeat, i_repeat=i_repeat)
                # ``transforms``：插入重复轮次坐标覆盖后的 Compose。
                transforms = Compose(featurizer_list + [overwirter, task_trans] + addition_transforms)
            elif 'overwrite_mol_repeat' in config.sample:
                # ``overwrite_mol_repeat``：每轮替换整个输入分子的兼容配置。
                overwrite_mol_repeat = config.sample.overwrite_mol_repeat
                # ``overwirter``：按 i_repeat 选择输入分子的 transform。
                overwirter = OverwriteMolRepeat(config=overwrite_mol_repeat, i_repeat=i_repeat)
                # ``transforms``：先替换原分子、再 featurize 和写任务 prompt。
                transforms = Compose([overwirter] + featurizer_list + [task_trans] + addition_transforms)
            # ``test_set``：使用本轮动态 transform 重新创建的数据集。
            test_set = TestTaskDataset(data_cfg.dataset, config.task,
                               mode='test',
                               split=getattr(data_cfg, 'split', None),
                               transforms=transforms)
            # ``test_loader``：与本轮 test_set 对齐的新 DataLoader。
            test_loader = DataLoader(test_set, batch_size, shuffle=args.shuffle,
                                    num_workers = num_workers,
                                    pin_memory = train_config.train.pin_memory,
                                    follow_batch=follow_batch, exclude_keys=exclude_keys)
        
        # ``i_batch``：int，当前测试 DataLoader 批次编号。
        # ``batch``：PyG Batch，当前批 B 个待采样图及其拼接后的原子、半边和可选口袋实体。
        for i_batch, batch in enumerate(test_loader):
            if i_saved >= num_mols:
                logger.info('Enough molecules. Stop sampling.')
                break
            
            # ``batch``：移到运行设备后的 PyG Batch；所有实体叶和 batch 向量同步移动。
            batch = batch.to(args.device)
            # outputs, trajs = sample_loop2(batch, model, noiser, args.device)
            # ``batch``：PyG Batch，最后一次 M-Projector 写回后的最终分子状态。
            # ``outputs.pred_node``：FloatTensor，形状为 (N, C_n)，末步原子类别 logits。
            # ``outputs.pred_pos``：FloatTensor，形状为 (N, 3)，末步干净局部坐标预测，单位 Å。
            # ``outputs.pred_halfedge``：FloatTensor，形状为 (H, C_e)，末步半边类别 logits。
            # ``outputs.confidence_node``：FloatTensor，形状为 (N, 1)，末步原子 confidence 原始输出。
            # ``outputs.confidence_pos``：FloatTensor，形状为 (N, 1)，末步坐标 confidence 原始输出。
            # ``outputs.confidence_halfedge``：FloatTensor，形状为 (H, 1)，末步半边 confidence 原始输出。
            # ``outputs.confidence_pos_traj``：FloatTensor，形状为 (N, T_total)，逐原子逐步坐标 confidence 原始输出。
            # ``outputs``：dict[str, Tensor]，保存以上末步预测与 confidence 叶。
            # ``trajs.all.node``：ndarray，形状为 (S_all, N)，干净参照、逐步输入和逐步投影状态的原子类别轨迹。
            # ``trajs.all.pos``：ndarray，形状为 (S_all, N, 3)，干净参照、逐步输入和逐步投影状态的坐标轨迹，单位 Å。
            # ``trajs.all.halfedge``：ndarray，形状为 (S_all, H)，干净参照、逐步输入和逐步投影状态的半边类别轨迹。
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
            # ``i``：int，源码按 ``range(len(batch))`` 生成的索引；常见 PyG 中 ``len(Batch)`` 是字段数而非图数，疑似导致长度错位。
            # ``data_list``：list[dict]，源码把四个图级元数据叶按上述 ``i`` 拆出；本学习分支只记录潜在长度问题。
            # ``data_list[i].data_id``：str，第 i 个候选所属样本标识。
            # ``data_list[i].db``：str，第 i 个候选所属逻辑数据库名。
            # ``data_list[i].task``：str，第 i 个候选的图级任务名。
            # ``data_list[i].key``：str，第 i 个候选实际读取的复合 LMDB 键。
            data_list = [{key:batch[key][i] for key in info_keys} for i in range(len(batch))]
            # try:
            # ``generated_list``：list[dict]，长度为 B；每项是一个候选的解码输入映射。
            # ``generated_list[b].node``：ndarray，形状为 (N_b,)，候选 b 的最终原子类别索引。
            # ``generated_list[b].pos``：ndarray，形状为 (N_b, 3)，候选 b 的最终局部坐标，单位 Å。
            # ``generated_list[b].halfedge``：ndarray，形状为 (H_b,)，候选 b 的最终半边类别索引。
            # ``generated_list[b].halfedge_index``：ndarray，形状为 (2, H_b)，候选 b 的 0-based 半边端点索引。
            # ``generated_list[b].pocket_center``：ndarray，形状为 (1, 3)，解码时加回的口袋中心，单位 Å。
            # ``outputs_list``：list[dict]，长度为 B；每项保存按候选拆分的 CPU Tensor 输出叶。
            # ``outputs_list[b].pred_node``：Tensor|缺省，形状为 (N_b, C_n)，候选 b 的原子类别 logits。
            # ``outputs_list[b].pred_pos``：Tensor|缺省，形状为 (N_b, 3)，候选 b 的局部坐标预测，单位 Å。
            # ``outputs_list[b].pred_halfedge``：Tensor|缺省，形状为 (H_b, C_e)，候选 b 的半边类别 logits。
            # ``outputs_list[b].confidence_node``：Tensor|缺省，形状为 (N_b, 1)，候选 b 的原子 confidence 原始输出。
            # ``outputs_list[b].confidence_pos``：Tensor|缺省，形状为 (N_b, 1)，候选 b 的坐标 confidence 原始输出。
            # ``outputs_list[b].confidence_halfedge``：Tensor|缺省，形状为 (H_b, 1)，候选 b 的半边 confidence 原始输出。
            # ``outputs_list[b].confidence_pos_traj``：Tensor|缺省，形状为 (N_b, T_total)，候选 b 的坐标 confidence 轨迹。
            # ``outputs_list[b].halfedge_index``：Tensor，形状为 (2, H_b)，候选 b 的半边端点索引。
            # ``outputs_list[b].pocket_center``：Tensor，通常形状为 (1, 3)，候选 b 的口袋中心，单位 Å。
            # ``traj_list_dict``：dict[str, list[dict]]|list，按轨迹来源保存 B 个单候选轨迹；无轨迹时为空列表。
            # ``traj_list_dict[source][b].node``：ndarray，形状为 (S, N_b)，候选 b 的原子类别轨迹。
            # ``traj_list_dict[source][b].pos``：ndarray，形状为 (S, N_b, 3)，候选 b 的局部坐标轨迹，单位 Å。
            # ``traj_list_dict[source][b].halfedge``：ndarray，形状为 (S, H_b)，候选 b 的半边类别轨迹。
            generated_list, outputs_list, traj_list_dict = seperate_outputs2(batch, outputs, trajs)
            # except:
            #     continue
            
            # ``mol_info_list``：长度逐步增长到 B 的重建候选记录列表。
            mol_info_list = []
            # ``i_mol``：int，当前批内候选编号，索引解码输入、拆分输出、元数据与轨迹列表。
            for i_mol in tqdm(range(len(generated_list)), desc='Post process generated mols'):
                # ``mol_info.atom_pos``：ndarray，形状为 (N_b, 3)，已加回口袋中心的候选世界坐标，单位 Å。
                # ``mol_info.element``：ndarray，形状为 (N_b,)，由原子类别反查的原子序数。
                # ``mol_info.bond_index``：ndarray，形状为 (2, E_b)，预测真实键的双向端点索引。
                # ``mol_info.bond_type``：ndarray，形状为 (E_b,)，与 ``bond_index`` 列对齐的键类别。
                # ``mol_info``：dict，先保存以上四个解码分子叶。
                mol_info = featurizer.decode_output(**generated_list[i_mol]) 
                # ``mol_info.data_id``：str，当前候选所属样本标识。
                # ``mol_info.db``：str，当前候选所属逻辑数据库名。
                # ``mol_info.task``：str，当前图级任务名；本轮为 ``conf`` 或 ``dock``。
                # ``mol_info.key``：str，当前候选实际读取的复合 LMDB 键。
                # ``mol_info``：dict，追加以上四个与 ``data_list[i_mol]`` 对齐的元数据叶。
                mol_info.update(data_list[i_mol])  # add data info
                
                # reconstruct mols
                try:
                    # ``rdmol``：固定二维拓扑并覆盖生成坐标后的 RDKit Mol。
                    rdmol = reconstruct_from_generated_with_edges(mol_info)
                    # ``smiles``：规范 SMILES，用于检测多片段；不作为构象/docking 评分输入。
                    smiles = Chem.MolToSmiles(rdmol)
                    if '.' in smiles:
                        # ``tag``：``tag='incomp'`` 表示重建成功但规范 SMILES 含多个连通分量。
                        tag = 'incomp'
                        pool.incomp.append(mol_info)
                        logger.warning('Incomplete molecule: %s' % smiles)
                    else:
                        # ``tag``：空 tag 表示可正常作为单分子 SDF 落盘。
                        tag = ''
                        pool.succ.append(mol_info)
                        logger.info('Success: %s' % smiles)
                except MolReconsError:
                    pool.bad.append(mol_info)
                    logger.warning('Reconstruction error encountered.')
                    # ``smiles``：重建失败候选以空字符串占位。
                    smiles = ''
                    # ``tag``：``tag='bad'`` 控制后续把文本 SDF fallback 直接写文件。
                    tag = 'bad'
                    # ``rdmol``：此分支实际为 SDF 文本 str，而非 RDKit Mol。
                    rdmol = create_sdf_string(mol_info)
                
                # ``mol_info.rdmol``：RDKit Mol|str，成功时为重建分子，失败时为 V2000 mol block。
                # ``mol_info.smiles``：str，成功候选的规范 SMILES；重建失败时为空字符串。
                # ``mol_info.tag``：str，取 ``''``、``incomp`` 或 ``bad``。
                # ``mol_info.output``：dict[str, Tensor]，当前候选拆分后的主预测与 confidence 叶。
                # ``mol_info``：dict，追加以上四个重建与模型输出叶。
                mol_info.update({
                    # ``rdmol``：RDKit Mol|str，重建结果或回退 mol block。
                    'rdmol': rdmol,
                    # ``smiles``：str，规范分子身份或失败空字符串。
                    'smiles': smiles,
                    # ``tag``：str，候选重建状态。
                    'tag': tag,
                    # ``output``：dict[str, Tensor]，单候选模型输出。
                    'output': outputs_list[i_mol],
                })
                
                # ``p_save_traj``：float [0,1)，逐候选独立轨迹落盘抽样值。
                p_save_traj = np.random.rand()  # save traj
                if p_save_traj <  save_traj_prob:
                    # ``mol_traj[source]``：list[str]，当前轨迹来源逐时间步的 V2000 mol block。
                    # ``mol_traj``：dict[str, list[str]]，轨迹来源到上述帧序列的映射。
                    mol_traj = {}
                    # ``traj_who``：str，当前轨迹来源名，通常为 ``all``、``in``、``out`` 或 ``raw``。
                    for traj_who in traj_list_dict.keys():
                        # ``traj_this_mol.node``：ndarray，形状为 (S, N_b)，当前候选当前来源的原子类别轨迹。
                        # ``traj_this_mol.pos``：ndarray，形状为 (S, N_b, 3)，当前候选当前来源的局部坐标轨迹，单位 Å。
                        # ``traj_this_mol.halfedge``：ndarray，形状为 (S, H_b)，当前候选当前来源的半边类别轨迹。
                        # ``traj_this_mol``：dict，保存以上三个逐时间步叶。
                        traj_this_mol = traj_list_dict[traj_who][i_mol]
                        # ``t``：int，当前来源中的时间步编号，与三个轨迹叶的第 0 维对齐。
                        for t in range(len(traj_this_mol['node'])):
                            # ``mol_this.atom_pos``：ndarray，形状为 (N_b, 3)，当前时间步世界坐标，单位 Å。
                            # ``mol_this.element``：ndarray，形状为 (N_b,)，当前时间步原子序数。
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
                            # ``mol_this``：覆盖为该帧的 SDF mol block 文本。
                            mol_this = create_sdf_string(mol_this)
                            mol_traj.setdefault(traj_who, []).append(mol_this)
                            
                    # 新叶 ``mol_info['traj']``：dict[str, list[str]]，从轨迹来源映射到 SDF 帧列表，仅对抽中候选存在。
                    mol_info['traj'] = mol_traj
                mol_info_list.append(mol_info)

            # ``df_info_list``：当前批次候选级 CSV 行字典列表。
            df_info_list = []
            # ``data_finished``：dict，当前已重建候选；含分子叶、元数据、重建状态和单候选模型输出。
            for data_finished in mol_info_list:
                # ``rdmol``：正常分支为 RDKit Mol，bad 分支为 SDF 文本。
                rdmol = data_finished['rdmol']
                # ``tag``：''、incomp 或 bad，编码到文件名并写入 CSV。
                tag = data_finished['tag']
                # ``filename``：全实验唯一的候选 SDF 文件名；失败标签附在编号后。
                filename = str(i_saved) + (f'-{tag}' if tag else '') + '.sdf'
                if tag != 'bad':
                    Chem.MolToMolFile(rdmol, os.path.join(sdf_dir, filename))
                else:
                    # ``f``：文本文件句柄，指向当前 bad 候选的回退 SDF 路径。
                    with open(os.path.join(sdf_dir, filename), 'w+') as f:
                        f.write(rdmol)
                # save traj
                if 'traj' in data_finished:
                    # ``traj_who``：str，当前待写多帧 SDF 的轨迹来源名。
                    for traj_who in data_finished['traj'].keys():
                        # ``sdf_file``：多帧 SD 文件文本；帧间用 ``$$$$`` 分隔。
                        sdf_file = '$$$$\n'.join(data_finished['traj'][traj_who])
                        # ``name_traj``：在候选文件干后附加 all/in/out/raw 来源名。
                        name_traj = filename.replace('.sdf', f'-{traj_who}.sdf')
                        # ``f``：文本文件句柄，指向当前来源的多帧轨迹 SDF 路径。
                        with open(os.path.join(sdf_dir, name_traj), 'w+') as f:
                            f.write(sdf_file)
                # 每成功写出一个候选后推进全局文件编号与停止计数。
                i_saved += 1
                
                # ``output.pred_node``：Tensor|缺省，形状为 (N_b, C_n)，当前候选的原子类别 logits。
                # ``output.pred_pos``：Tensor|缺省，形状为 (N_b, 3)，当前候选的局部坐标预测，单位 Å。
                # ``output.pred_halfedge``：Tensor|缺省，形状为 (H_b, C_e)，当前候选的半边类别 logits。
                # ``output.confidence_node``：Tensor，形状为 (N_b, 1)，当前候选的原子 confidence 原始输出。
                # ``output.confidence_pos``：Tensor，形状为 (N_b, 1)，当前候选的坐标 confidence 原始输出。
                # ``output.confidence_halfedge``：Tensor，形状为 (H_b, 1)，当前候选的半边 confidence 原始输出。
                # ``output.confidence_pos_traj``：Tensor，形状为 (N_b, T_total)，当前候选的坐标 confidence 轨迹。
                # ``output.halfedge_index``：Tensor，形状为 (2, H_b)，当前候选的半边端点索引。
                # ``output.pocket_center``：Tensor，通常形状为 (1, 3)，当前候选的口袋中心，单位 Å。
                # ``output``：dict[str, Tensor]，保存以上单候选模型输出与坐标还原叶。
                output = data_finished['output']
                # ``cfd_traj``：Python float，逐原子聚合后取轨迹后半段均值。
                cfd_traj = get_cfd_traj(output['confidence_pos_traj'])  # get cfd
                # ``cfd_pos``：NumPy 标量，末步逐原子坐标置信度原始值均值。
                cfd_pos = output['confidence_pos'].detach().cpu().numpy().mean()
                # ``cfd_node``：NumPy 标量，末步逐原子类别置信度 logit 均值。
                cfd_node = output['confidence_node'].detach().cpu().numpy().mean()
                # ``cfd_edge``：NumPy 标量，末步逐半边类别置信度 logit 均值。
                cfd_edge = output['confidence_halfedge'].detach().cpu().numpy().mean()
                # ``save_output``：list[str]，选择额外序列化到每候选 .pt 的输出叶名。
                save_output = getattr(config.sample, 'save_output', [])
                if len(save_output) > 0:
                    # ``key``：str，``save_output`` 中当前需要序列化的模型输出叶名。
                    # ``output``：dict[str, Tensor]，裁剪为用户指定叶字段的映射；仅影响随后 ``torch.save``。
                    output = {key: output[key] for key in save_output}
                    torch.save(output, os.path.join(sdf_dir, filename.replace('.sdf', '.pt')))

                # ``info_dict.data_id``：str，候选所属样本标识。
                # ``info_dict.db``：str，候选所属逻辑数据库名。
                # ``info_dict.task``：str，候选图级任务名；本轮为 ``conf`` 或 ``dock``。
                # ``info_dict.key``：str，候选实际读取的复合 LMDB 键。
                # ``info_dict.smiles``：str，候选重建后的规范 SMILES；失败时为空字符串。
                # ``info_dict.tag``：str，候选重建状态，取 ``''``、``incomp`` 或 ``bad``。
                # ``key``：str，当前复制到 CSV 行的候选元数据叶名。
                # ``info_dict``：dict，先收集以上六个候选元数据叶。
                info_dict = {
                    key: data_finished[key] for key in info_keys + ['smiles', 'tag']
                }
                # ``info_dict.filename``：str，最终候选 SDF 文件名。
                # ``info_dict.i_repeat``：int，当前候选所属的完整采样轮号。
                # ``info_dict.cfd_traj``：float，坐标 confidence 轨迹的既有两阶段汇总值。
                # ``info_dict.cfd_pos``：NumPy 浮点标量，最终逐原子坐标 confidence 原始输出均值。
                # ``info_dict.cfd_node``：NumPy 浮点标量，最终逐原子类别 confidence 原始输出均值。
                # ``info_dict.cfd_edge``：NumPy 浮点标量，最终逐半边类别 confidence 原始输出均值。
                # ``info_dict``：dict，追加上述文件定位、轮次和四个 confidence 汇总叶。
                info_dict.update({
                    # ``filename``：str，最终候选 SDF 文件名。
                    'filename': filename,
                    # ``i_repeat``：int，完整采样轮号。
                    'i_repeat': i_repeat,
                    # ``cfd_traj``：float，坐标 confidence 轨迹汇总值。
                    'cfd_traj': cfd_traj,
                    # ``cfd_pos``：NumPy 浮点标量，最终坐标 confidence 均值。
                    'cfd_pos': cfd_pos,
                    # ``cfd_node``：NumPy 浮点标量，最终原子 confidence 均值。
                    'cfd_node': cfd_node,
                    # ``cfd_edge``：NumPy 浮点标量，最终半边 confidence 均值。
                    'cfd_edge': cfd_edge,
                })

                df_info_list.append(info_dict)
        
            # ``df_info_batch``：当前批次 DataFrame，每行与一个已写候选文件对应。
            df_info_batch = pd.DataFrame(df_info_list)
            # # save df
            if os.path.exists(df_path):
                # ``df_info``：此前所有批次的候选主索引。
                df_info = pd.read_csv(df_path)
                # ``df_info``：追加当前批并重建连续行索引后的完整主索引。
                df_info = pd.concat([df_info, df_info_batch], ignore_index=True)
            else:
                # ``df_info``：首批直接把当前 DataFrame 作为完整主索引。
                df_info = df_info_batch
            df_info.to_csv(df_path, index=False)
            print_pool_status(pool, logger)
            
            # clean up
            del batch, outputs, trajs, mol_info_list[0:len(mol_info_list)]
            with torch.cuda.device(args.device):
                torch.cuda.empty_cache()
            gc.collect()


    # ``key``：str，当前候选池类别名，取 ``succ``、``bad`` 或 ``incomp``。
    # ``value``：list[dict]，当前类别的候选记录列表；这里只读取长度。
    # ``dummy_pool``：dict[str, list[str]]，保留三个类别的计数但以空字符串代替大型 ``mol_info`` 对象。
    dummy_pool = {key: ['']*len(value) for key, value in pool.items()}
    torch.save(dummy_pool, os.path.join(log_dir, 'samples_all.pt'))
    # torch.save(pool, os.path.join(log_dir, 'samples_all.pt'))
