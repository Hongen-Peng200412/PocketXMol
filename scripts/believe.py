"""用独立 confidence checkpoint 对已经生成的小分子 docking pose 重新评分。

脚本从原生成目录读取任务/数据契约，用 ``OverwritePos`` 把每个已生成 SDF pose
写回对应输入图，再按 confidence YAML 执行一次条件评分前向。当前 YAML 把
``level.min=max=1``，所以 prior 不会扰动该输入 pose；``init_step=0.01`` 仍传入
调度器，但线性缩放后的 level 固定为 1。每个候选
最终以逐原子 ``confidence_pos`` 原始输出均值形成 ``tuned_cfd``；它不使用真实
RMSD，因而可在部署时参与排序。
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
from models.sample import seperate_outputs2, sample_loop3
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *
from utils.sample_noise import get_sample_noiser
from evaluate.evaluate_mols import get_dir_from_prefix


def print_pool_status(pool, logger):
    """记录兼容候选池的三类列表长度；当前重评分主循环未调用。

    输入参数:
        - pool: EasyDict，兼容主采样脚本的候选池。
        - pool.succ: list[object]，单连通重建成功候选。
        - pool.incomp: list[object]，多连通分量候选。
        - pool.bad: list[object]，重建失败候选。
        - logger: logging.Logger，接收三类候选数量的日志器。

    返回值:
        - None: 只读取三个列表长度并写日志。
    """

    logger.info('[Pool] Succ/Incomp/Bad: %d/%d/%d' % (
        len(pool.succ), len(pool.incomp), len(pool.bad)
    ))

# ``is_vscode``：bool 兼容开关；当前脚本后续未读取。
is_vscode = False
if os.environ.get("TERM_PROGRAM") == "vscode":
    # ``is_vscode``：保留与主采样脚本一致的环境检测结果。
    is_vscode = True


if __name__ == '__main__':
    # ``parser``：重评分命令行参数解析器。
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='base_pxm')
    parser.add_argument('--result_root', type=str, default='./outputs_test/dock_posebusters')
    parser.add_argument('--config', type=str, default='configs/sample/confidence/tuned_cfd.yml', help='confidence config')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=1)
    # ``args.exp_name``：str，待重评分生成实验目录的名称前缀。
    # ``args.result_root``：str，生成实验所在的结果根目录。
    # ``args.config``：str，confidence 重评分 YAML 路径。
    # ``args.device``：str，模型与批次使用的 PyTorch 设备字符串。
    # ``args.batch_size``：int，非零时覆盖原生成配置的候选批大小；0 表示继承原配置。
    # ``args.num_workers``：int，重评分 DataLoader worker 数。
    # ``args``：Namespace，保存以上六个命令行叶。
    args = parser.parse_args()
    
    # ``gen_path``：str，由根目录与实验名前缀解析出的唯一生成目录。
    gen_path = get_dir_from_prefix(args.result_root, args.exp_name)
    # ``gen_name``：str，生成目录基名；当前后续未读取。
    gen_name = os.path.basename(gen_path)
    # ``belief_name``：str，confidence YAML 文件干名；也作为输出列名与 CSV 文件名。
    belief_name = os.path.basename(args.config).split('.')[0]
    # ``save_path``：str，例如 ``<gen_path>/tuned_cfd.csv``。
    save_path = os.path.join(gen_path, f'{belief_name}.csv')
    # ``be_config.sample.seed``：int，重评分过程使用的统一随机种子。
    # ``be_config.sample.batch_size``：int|缺省，当前脚本不读取；实际批大小继承原生成配置或由 CLI 覆盖。
    # ``be_config.model.checkpoint``：str|缺省，独立 ranker checkpoint；缺少 ``model`` 组时回退原生成 checkpoint。
    # ``be_config.noise.name``：str，固定为 ``dock``，实例化 DockSamplNoiser。
    # ``be_config.noise.num_steps``：int，单步重评分配置为 1。
    # ``be_config.noise.init_step``：float，唯一采样步的规范初始进度。
    # ``be_config.noise.prior``：Mapping，单步 noiser 的坐标先验配置。
    # ``be_config.noise.level``：Mapping，把单步进度映射为各运动自由度信息等级。
    # ``be_config.task.name``：str，图级任务名，固定为 ``dock``。
    # ``be_config.task.transform``：Mapping，believe 模式 docking fixed prompt 与运动模式配置。
    # ``be_config``：EasyDict，保存以上 confidence 模型、noiser 与任务叶。
    be_config = make_config(args.config)
    seed_all(be_config.sample.seed)
    
    # ``f``：str，``gen_path`` 下当前文件名；只保留 ``.yml`` 任务配置。
    # ``sample_config_file``：list[str]，生成目录内保存的原任务 YAML 文件名；要求恰好一个。
    sample_config_file = [f for f in os.listdir(gen_path) if f.endswith('.yml')]
    assert len(sample_config_file) == 1, 'sample config file is not 1 ge'
    # ``sa_config.data``：Mapping，原生成实验的数据集、划分和追加 transform 配置。
    # ``sa_config.task.db``：str，原生成实验使用的唯一逻辑数据库名。
    # ``sa_config.sample.batch_size``：int，CLI 未覆盖时的重评分批大小。
    # ``sa_config.sample.num_repeats``：int，必须重放的原生成候选轮数。
    # ``sa_config.model.checkpoint``：str，未配置独立 ranker 时的回退 checkpoint。
    # ``sa_config``：EasyDict，从生成目录内唯一任务 YAML 恢复的以上叶。
    sa_config = make_config(os.path.join(gen_path, sample_config_file[0]))
    
    # load ckpt (of sampling) and train config
    if 'model' in be_config:
        # ``model_ckpt_path``：str，独立 tuned ranker checkpoint 路径。
        model_ckpt_path = be_config.model.checkpoint
    else:
        # ``model_ckpt_path``：未指定独立模型时回退原生成 checkpoint，得到 self-model 重评分。
        model_ckpt_path = sa_config.model.checkpoint  # the same model as sampling
    # ``ckpt.state_dict``：dict[str, Tensor]，Lightning 参数映射；后续只加载 ``model.`` 前缀叶。
    # ``ckpt``：checkpoint 映射，按 ``args.device`` 加载。
    ckpt = torch.load(model_ckpt_path, map_location=args.device)
    # ``cfg_dir``：str，与 checkpoint 配套的训练配置目录。
    cfg_dir = os.path.dirname(model_ckpt_path).replace('checkpoints', 'train_config')
    # ``train_config``：list[str]，目录中的训练配置文件名。
    train_config = os.listdir(cfg_dir)
    # ``train_config.model``：Mapping，恢复 PMAsymDenoiser 结构与输出头。
    # ``train_config.transforms``：Mapping，恢复口袋和配体 featurizer 配置。
    # ``train_config.noise``：Mapping，为 ``prior: from_train`` 提供训练任务先验。
    # ``train_config.train.num_workers``：int，CLI 为 -1 时的 worker 回退值。
    # ``train_config.train.pin_memory``：bool，重评分 DataLoader 页锁定内存开关。
    # ``train_config``：EasyDict，从 checkpoint 相邻训练 YAML 读取的以上叶。
    train_config = make_config(os.path.join(cfg_dir, ''.join(train_config)))

    # ``batch_size``：int；CLI 非零值覆盖原采样 batch_size。
    batch_size = sa_config.sample.batch_size if args.batch_size == 0 else args.batch_size
    
    # ``log_dir``：str，重评分日志直接写回原生成目录。
    log_dir = gen_path
    # ``logger``：believe 名称的文件/终端 Logger。
    logger = get_logger('believe', log_dir)
    logger.info('Load from %s...' % model_ckpt_path)

    # df_path = os.path.join(log_dir, 'gen_info.csv')

    logger.info('Loading data placeholder...')
    # ``dm``：仅复用训练时 featurizer 和输入维度构造逻辑的 DataModule。
    dm = DataModule(train_config)
    # ``featurizer_list``：训练配置定义的 transform 实例序列。
    featurizer_list = dm.get_featurizers()
    # ``featurizer``：最后一个分子 featurizer；当前脚本保留引用但不执行解码。
    featurizer = featurizer_list[-1]  # for mol decoding
    # ``in_dims.num_node_types``：int，含可选 mask 的原子类别总数 C_n。
    # ``in_dims.num_edge_types``：int，含非键和可选 mask 的完整半边类别总数 C_e。
    # ``in_dims``：dict[str, int]，模型构造与 noiser 使用的以上类别维度叶。
    in_dims = dm.get_in_dims()
    # ``task_trans``：believe 模式 docking transform，建立 fixed prompt 与运动域注释。
    task_trans = get_transforms(be_config.task.transform, mode='believe')  # use belief task
    # ``noiser``：num_steps=1；现有 YAML min=max=1，使输入 pose 不受 prior 扰动。
    noiser = get_sample_noiser(be_config.noise, in_dims['num_node_types'], in_dims['num_edge_types'], # use belief noser
                               mode='sample',device=args.device, ref_config=train_config.noise)
    # ``transforms_list``：基础 featurizer 后接 belief docking prompt 的 transform 序列。
    transforms_list = featurizer_list + [task_trans]
    # ``be_config.task.db``：对齐原采样数据库名，使 TestTaskDataset 找到相同图记录。
    be_config.task.db = sa_config.task.db

    logger.info('Loading diffusion model...')
    if train_config.model.name == 'pm_asym_denoiser':
        # ``model``：与 checkpoint 训练配置完全一致的 PMAsymDenoiser。
        model = PMAsymDenoiser(config=train_config.model, **in_dims).to(args.device)
    # ``k``：str，checkpoint 中当前参数全名；只保留 ``model.`` 前缀并在加载前移除该前缀。
    # ``value``：Tensor，当前 checkpoint 参数值；形状必须与去前缀后的模型参数一致。
    model.load_state_dict({k[6:]:value for k, value in ckpt['state_dict'].items() if k.startswith('model.')}) # prefix is 'model'
    model.eval()
    
    # ``df_belief_list``：候选级评分行字典列表，跨所有 repeat 累积。
    df_belief_list = []
    # ``num_repeats``：int，必须与原生成候选的 repeat 编号范围一致。
    num_repeats = sa_config.sample.num_repeats
    # ``info_keys[0]``：``data_id``，候选所属复合物标识的字段名。
    # ``info_keys[1]``：``filename``，原生成候选 SDF 文件名的字段名。
    # ``info_keys``：list[str]，当前候选对齐所需的两个元数据键；变量保留但后续未直接使用。
    info_keys = [
        'data_id',
        'filename',
    ]
    # ``i_repeat``：int，当前重放的原生成轮号，取值范围为 ``[0, num_repeats)``。
    for i_repeat in range(num_repeats):
        
        logger.info(f'Loading dataset for repeat {i_repeat}')
        # ``data_cfg``：原采样数据配置，保证读取同一测试分子顺序。
        data_cfg = sa_config.data
        # ``overwriter``：按 data_id/i_repeat 从 gen_path 读取已生成 SDF 并覆盖 node_pos。
        overwriter = OverwritePos(config=None,
                        gen_path=gen_path, i_repeat=i_repeat)
        # ``transforms``：featurize -> belief task prompt -> 已生成 pose 坐标覆盖。
        transforms = Compose(transforms_list + [overwriter])
        # ``test_set``：使用 belief task/noiser 但复用原数据集索引的测试集。
        test_set = TestTaskDataset(data_cfg.dataset, be_config.task,  # use belief task
                                mode='test',
                                split=getattr(data_cfg, 'split', None),
                                transforms=transforms)
        # ``t``：callable，Compose 中当前变换；读取其声明的批归属跟踪字段。
        # ``follow_batch``：list[str]，PyG 需要生成图归属向量的实体字段列表。
        follow_batch = sum([getattr(t, 'follow_batch', []) for t in transforms.transforms], [])
        # ``t``：callable，Compose 中当前变换；读取其声明的不参与批拼接字段。
        # ``exclude_keys``：list[str]，批处理时排除的字段列表。
        exclude_keys = sum([getattr(t, 'exclude_keys', []) for t in transforms.transforms], [])
        # ``num_workers``：int；CLI=-1 才复用训练 worker 数。
        num_workers = train_config.train.num_workers if args.num_workers == -1 else args.num_workers
        # ``test_loader``：顺序固定的 PyG DataLoader，确保候选与 repeat 对齐。
        test_loader = DataLoader(test_set, batch_size, shuffle=False,
                                num_workers = num_workers,
                                pin_memory = train_config.train.pin_memory,
                                follow_batch=follow_batch, exclude_keys=exclude_keys)

        # generating molecules
        logger.info(f'Generating molecules. Testset repeat {i_repeat}.')
        # ``batch``：PyG Batch，当前批 B 个待重评分 pose 及其配体、半边和口袋实体。
        for batch in test_loader:

            # ``data_list``：list[PyG Data]，长度 B，按图拆出的 CPU 单图记录。
            # ``data_list[b].filename``：str，图 b 在原生成目录中的候选 SDF 文件名。
            # ``data_list[b].data_id``：str，图 b 所属复合物标识。
            data_list = batch.to_data_list()
            # ``batch``：移到模型设备后的 PyG Batch。
            batch = batch.to(args.device)
            # outputs, trajs = sample_loop2(batch, model, noiser, args.device)
            # ``batch``：PyG Batch，单步 M-Projector 写回后的候选状态；固定 level=1 时保持既有 pose 语义。
            # ``outputs.pred_node``：FloatTensor，形状为 (N, C_n)，单步原子类别 logits。
            # ``outputs.pred_pos``：FloatTensor，形状为 (N, 3)，单步干净坐标预测，单位 Å。
            # ``outputs.pred_halfedge``：FloatTensor，形状为 (H, C_e)，单步半边类别 logits。
            # ``outputs.confidence_node``：FloatTensor，形状为 (N, 1)，逐原子类别 confidence 原始输出。
            # ``outputs.confidence_pos``：FloatTensor，形状为 (N, 1)，逐原子坐标 confidence 原始输出。
            # ``outputs.confidence_halfedge``：FloatTensor，形状为 (H, 1)，逐半边 confidence 原始输出。
            # ``outputs.confidence_pos_traj``：FloatTensor，形状为 (N, 1)，单步坐标 confidence 轨迹。
            # ``outputs``：dict[str, Tensor]，保存以上预测与 confidence 叶。
            # ``trajs.all.node``：ndarray，形状为 (S_all, N)，参照、单步输入和单步投影状态的原子类别轨迹。
            # ``trajs.all.pos``：ndarray，形状为 (S_all, N, 3)，参照、单步输入和单步投影状态的坐标轨迹，单位 Å。
            # ``trajs.all.halfedge``：ndarray，形状为 (S_all, H)，参照、单步输入和单步投影状态的半边类别轨迹。
            # ``trajs.in.node``：ndarray，形状为 (1, N)，单步带噪原子类别轨迹。
            # ``trajs.in.pos``：ndarray，形状为 (1, N, 3)，单步带噪坐标轨迹，单位 Å。
            # ``trajs.in.halfedge``：ndarray，形状为 (1, H)，单步带噪半边类别轨迹。
            # ``trajs.out.node``：ndarray，形状为 (1, N)，单步投影后原子类别轨迹。
            # ``trajs.out.pos``：ndarray，形状为 (1, N, 3)，单步投影后坐标轨迹，单位 Å。
            # ``trajs.out.halfedge``：ndarray，形状为 (1, H)，单步投影后半边类别轨迹。
            # ``trajs.raw.node``：ndarray，形状为 (1, N)，单步原始原子类别 argmax 轨迹。
            # ``trajs.raw.pos``：ndarray，形状为 (1, N, 3)，单步原始坐标预测轨迹，单位 Å。
            # ``trajs.raw.halfedge``：ndarray，形状为 (1, H)，单步原始半边类别 argmax 轨迹。
            # ``trajs``：dict[str, dict[str, ndarray]]，保存上述四个来源的三个轨迹叶。
            batch, outputs, trajs = sample_loop3(batch, model, noiser, args.device)
            
            # # decode outputs to molecules
            # data_list = [{key:batch[key][i] for key in info_keys} for i in range(len(batch))]
            # try:
            # ``generated_list[b].node``：ndarray，形状为 (N_b,)，候选 b 的最终原子类别索引。
            # ``generated_list[b].pos``：ndarray，形状为 (N_b, 3)，候选 b 的最终局部坐标，单位 Å。
            # ``generated_list[b].halfedge``：ndarray，形状为 (H_b,)，候选 b 的最终半边类别索引。
            # ``generated_list[b].halfedge_index``：ndarray，形状为 (2, H_b)，候选 b 的半边端点索引。
            # ``generated_list[b].pocket_center``：ndarray，形状为 (1, 3)，候选 b 的口袋中心，单位 Å。
            # ``generated_list``：list[dict]，长度为 B；保存以上单候选最终状态叶。
            # ``outputs_list[b].pred_node``：Tensor|缺省，形状为 (N_b, C_n)，候选 b 的原子类别 logits。
            # ``outputs_list[b].pred_pos``：Tensor|缺省，形状为 (N_b, 3)，候选 b 的局部坐标预测，单位 Å。
            # ``outputs_list[b].pred_halfedge``：Tensor|缺省，形状为 (H_b, C_e)，候选 b 的半边类别 logits。
            # ``outputs_list[b].confidence_node``：Tensor，形状为 (N_b, 1)，候选 b 的原子 confidence 原始输出。
            # ``outputs_list[b].confidence_pos``：Tensor，形状为 (N_b, 1)，候选 b 的坐标 confidence 原始输出。
            # ``outputs_list[b].confidence_halfedge``：Tensor，形状为 (H_b, 1)，候选 b 的半边 confidence 原始输出。
            # ``outputs_list[b].confidence_pos_traj``：Tensor，形状为 (N_b, 1)，候选 b 的单步坐标 confidence 轨迹。
            # ``outputs_list[b].halfedge_index``：Tensor，形状为 (2, H_b)，候选 b 的半边端点索引。
            # ``outputs_list[b].pocket_center``：Tensor，通常形状为 (1, 3)，候选 b 的口袋中心，单位 Å。
            # ``outputs_list``：list[dict]，长度为 B；保存以上单候选 CPU Tensor 输出叶。
            # ``traj_list_dict[source][b].node``：ndarray，形状为 (S, N_b)，候选 b 的原子类别轨迹。
            # ``traj_list_dict[source][b].pos``：ndarray，形状为 (S, N_b, 3)，候选 b 的坐标轨迹，单位 Å。
            # ``traj_list_dict[source][b].halfedge``：ndarray，形状为 (S, H_b)，候选 b 的半边类别轨迹。
            # ``traj_list_dict``：dict[str, list[dict]]，按来源保存以上三个单候选轨迹叶。
            generated_list, outputs_list, traj_list_dict = seperate_outputs2(batch, outputs, trajs)
            # except:
            #     continue
            
            # ``mol_info_list``：历史遗留空列表；当前循环直接生成评分行，不向其中追加。
            mol_info_list = []
            # for output, data in zip(outputs_list, data_list):
            # ``i_gen``：int，源码按 ``range(len(batch))`` 生成；常见 PyG 中 ``len(Batch)`` 是字段数而非图数，疑似与真实候选数错位。
            for i_gen in range(len(batch)):
                # ``gen_data.node``：ndarray，形状为 (N_b,)，当前候选最终原子类别索引。
                # ``gen_data.pos``：ndarray，形状为 (N_b, 3)，当前候选最终局部坐标，单位 Å。
                # ``gen_data.halfedge``：ndarray，形状为 (H_b,)，当前候选最终半边类别索引。
                # ``gen_data.halfedge_index``：ndarray，形状为 (2, H_b)，当前候选半边端点索引。
                # ``gen_data.pocket_center``：ndarray，形状为 (1, 3)，当前候选口袋中心，单位 Å。
                # ``gen_data``：dict，保存以上单候选最终状态叶；当前只为接口完整性读取。
                gen_data = generated_list[i_gen]
                # ``output.confidence_pos``：Tensor，形状为 (N_b, 1)，当前候选逐原子坐标 confidence 原始输出。
                # ``output``：dict[str, Tensor]，当前候选拆分后的模型输出；评分只读取上述叶。
                output = outputs_list[i_gen]
                # ``data.filename``：str，当前候选在原生成目录中的 SDF 文件名。
                # ``data.data_id``：str，当前候选所属复合物标识。
                # ``data``：PyG Data，对应原始单图记录；评分定位读取以上两个叶。
                data = data_list[i_gen]
                
                # ``data_id``：str，候选所属复合物标识。
                data_id = data['data_id']
                # ``filename``：str，与原 gen_info.csv 中已生成 pose 文件名一致。
                filename = data['filename']
                # regen rmsd
                # orig_pos = data['node_pos'].detach().cpu().numpy()
                # regen_pos = gen_data['pos']
                # regen_rmsd = np.sqrt(np.mean(np.sum((orig_pos - regen_pos)**2, axis=-1)))
                # ``info_dict.filename``：str，原生成候选 SDF 文件名。
                # ``info_dict[belief_name]``：float，逐原子坐标 confidence 原始输出均值。
                # ``info_dict.data_id``：str，候选所属复合物标识。
                # ``info_dict.i_repeat``：int，候选所属原生成轮号。
                # ``info_dict``：dict，保存以上四个候选级重评分叶。
                info_dict = {
                    # ``filename``：str，原生成候选 SDF 文件名。
                    'filename': filename,
                    # ``belief_name`` 动态列：float，逐原子坐标 confidence 原始输出均值。
                    f'{belief_name}': torch.mean(output['confidence_pos']).item(),
                    # f'{belief_name}_rmsd': regen_rmsd,
                    # ``data_id``：str，候选所属复合物标识。
                    'data_id': data_id,
                    # ``i_repeat``：int，原生成轮号。
                    'i_repeat': i_repeat,
                }
                df_belief_list.append(info_dict)
    # ``df_belief``：DataFrame，每行一个原候选的 tuned confidence 记录。
    # ``df_belief.filename``：str，原生成候选 SDF 文件名。
    # ``df_belief[belief_name]``：float，逐原子坐标 confidence 原始输出均值。
    # ``df_belief.data_id``：str，候选所属复合物标识。
    # ``df_belief.i_repeat``：int，候选所属原生成轮号。
    df_belief = pd.DataFrame(df_belief_list)
    df_belief.to_csv(save_path, index=False)
        
    # print('Done. Results saved at %s' % result_path)
