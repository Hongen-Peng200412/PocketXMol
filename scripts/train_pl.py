"""
组织 PocketXMol 多任务训练的数据变换、PyG 批处理、去噪模型、损失和 Lightning 运行生命周期。

正式入口是文件末尾的命令行主程序；核心跨模块边界是 :class:`DataModule` 与
:class:`ModelLightning`。构象生成和小分子 docking 共用 ``PMAsymDenoiser``，任务差异先由
``ConfTransform``/``DockTransform`` 写成 fixed prompt 与扭转注释，再由训练噪声器生成
``node_in``、``pos_in`` 和 ``halfedge_in``。

运行会在 ``TensorBoardLogger.log_dir`` 下落盘：

- ``checkpoints/{step}.ckpt`` 与 ``checkpoints/last.ckpt``: Lightning checkpoint；顶层含 ``state_dict``，模型参数键以 ``model.`` 开头。
- ``src/``: 当前仓库指定目录中的 ``.py``、``.sh``、``.ipynb`` 源码快照；路径由 ``copy_py_files`` 的 ``dst_dir`` 决定。
- ``train_config/<config-name>.yml``: EasyDict 配置的序列化副本；路径由命令行 ``--config`` 文件名决定。
- TensorBoard 事件文件: 训练/验证损失、学习率和梯度范数；根目录由 ``--logdir``、配置路径层级和 ``--tag`` 共同决定。
"""

# Standard library imports
import argparse
import gc
import os
import shutil
import sys
from typing import Any, Callable, Optional, Union

# Third-party imports
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

# Local imports
sys.path.append('.')
from models.loss import get_loss_func
from models.maskfill import PMAsymDenoiser
from utils.dataset import ForeverTaskDataset
from utils.misc import *
from utils.sample_noise import get_sample_noiser
from utils.train import GradualWarmupScheduler, get_optimizer, get_scheduler
from utils.transforms import Compose, FeaturizeMol, FeaturizePocket, get_transforms

torch.set_float32_matmul_precision('medium')


def copy_py_files(src_dir, dst_dir, base=False):
    """
    递归复制可执行源码与 notebook，形成一次训练运行的代码快照。

    输入参数:
        - src_dir: str, 当前递归层的源目录；主程序传入仓库根目录 ``.``。
        - dst_dir: str, 当前递归层的目标目录；主程序传入 ``<log_dir>/src``。
        - base: bool, True 时只递归 ``scripts``、``models``、``notebooks``、``utils``、``process``、``evaluate`` 六类一级/任意同名目录；False 时递归全部目录。

    落盘产物:
        - ``<dst_dir>/**/*.py``: Python 源文件，目录层级相对 ``src_dir`` 保持不变。
        - ``<dst_dir>/**/*.sh``: Shell 脚本，目录层级相对 ``src_dir`` 保持不变。
        - ``<dst_dir>/**/*.ipynb``: Jupyter notebook，目录层级相对 ``src_dir`` 保持不变。

    副作用与边界:
        - 自动创建 ``dst_dir``；同名文件由 ``shutil.copy`` 覆盖。
        - 不复制 YAML、模型权重、数据集或 Git 元数据；训练配置由主程序单独调用 ``save_config`` 保存。
    """
    os.makedirs(dst_dir, exist_ok=True)
    # ``item``：str，当前递归层 ``src_dir`` 下的文件或目录名。
    for item in os.listdir(src_dir):
        # Get the absolute path of the item
        item_path = os.path.join(src_dir, item)
        if os.path.isdir(item_path):
            if (not base) or (item in ['scripts', 'models', 'notebooks', 'utils', 'process', 'evaluate']):
                # If the item is a directory, recursively call the function on it
                copy_py_files(item_path, os.path.join(dst_dir, item))
        elif (item.endswith('.py') or item.endswith('.sh') or item.endswith('.ipynb')):
            # If the item is a file and ends with .py, copy it to the destination directory
            shutil.copy(item_path, dst_dir)


# XXX
class DataModule(pl.LightningDataModule):
    """
    把训练配置解析为逐样本变换链、无限任务数据集和 PyG 批次加载器。

    构造参数:
        - config.model: 映射式模型配置；本类不构造模型，但 ``get_in_dims`` 为其推导类别数与口袋输入维度。
        - config.transforms.featurizer: 配体特征配置；定义原子序数表、键类别表和是否添加 mask 类。
        - config.transforms.featurizer_pocket: 可选口袋特征配置；存在时必须含 ``knn``，可选 ``center``。
        - config.transforms.task: 任务变换配置；``name=mixed`` 时 ``individual`` 按任务名建立 ``ConfTransform`` 等实例。
        - config.noise: 训练噪声配置；``name=mixed`` 时按样本 ``task`` 选择对应先验和信息等级。
        - config.data.dataset: assembly、数据库根目录与子 LMDB 布局；传给 ``ForeverTaskDataset``。
        - config.data.task_db_weights: 两级任务/数据库采样权重；传给训练与验证数据集。
        - config.train: batch size、worker 数、pin memory 和 persistent worker 等加载器设置。

    单样本变换顺序:
        - ``FeaturizePocket``: 构造口袋 25 维原子特征、口袋 kNN 边和中心化坐标。
        - ``FeaturizeMol``: 构造配体节点类别、中心化坐标、完全图半边类别。
        - 任务变换: 为构象/docking 写入 ``task_setting``、fixed prompt、刚体域和可旋转键注释。
        - 训练噪声器: 采样信息等级并写入 ``node_in``、``pos_in``、``halfedge_in``。

    PyG 批次核心字段:
        - node_type: LongTensor，形状为 (N_all,)，所有分子的干净原子类别。
        - node_in: LongTensor，形状为 (N_all,)，所有分子的带噪原子类别。
        - node_pos: FloatTensor，形状为 (N_all, 3)，所有分子的干净局部坐标，单位 Å。
        - pos_in: FloatTensor，形状为 (N_all, 3)，所有分子的带噪局部坐标，单位 Å。
        - halfedge_index: int64, (2, H_all), 各分子完全图上三角半边端点；拼接后索引 ``node_type`` 第一维。
        - halfedge_type: LongTensor，形状为 (H_all,)，干净半边类别；与 ``halfedge_index`` 第二维对齐。
        - halfedge_in: LongTensor，形状为 (H_all,)，带噪半边类别；与 ``halfedge_index`` 第二维对齐。
        - fixed_node: LongTensor|BoolTensor，形状为 (N_all,)，1 表示原子类别由任务条件固定。
        - fixed_pos: LongTensor|BoolTensor，形状为 (N_all,)，1 表示坐标由任务条件固定。
        - fixed_halfedge: LongTensor|BoolTensor，形状为 (H_all,)，1 表示半边类别由任务条件固定。
        - fixed_halfdist: LongTensor|BoolTensor，形状为 (H_all,)，1 表示半边端点距离由任务条件固定。
        - node_type_batch: int64, (N_all,), 每个配体原子所属图编号；由 ``follow_batch=['node_type']`` 生成。
        - halfedge_type_batch: int64, (H_all,), 每条半边所属图编号；由 ``follow_batch=['halfedge_type']`` 生成。
        - pocket_atom_feature: (P_all, D_p), 所有口袋原子的离散特征，默认 D_p=25。
        - pocket_pos: (P_all, 3), 以各自 ``pocket_center`` 为原点的口袋坐标，单位 Å。
        - pocket_pos_batch: int64, (P_all,), 每个口袋原子所属图编号；由 ``follow_batch=['pocket_pos']`` 生成。
        - task: list[str]，长度 B；第 b 个字符串是图 b 的任务名。
        - task_setting: 训练时被 ``exclude_keys`` 排除，不进入批次；任务差异已经编码进 fixed 与扭转字段。

    公开输出:
        - ``train_dataloader``: 无限 ``ForeverTaskDataset`` 的 PyG ``DataLoader``；训练步数由 Lightning ``max_steps`` 截断。
        - ``val_dataloader``: 每次迭代有限的 ``ForeverTaskDataset``；worker 数在超大 world size 时除以 4。
    """
    
    def __init__(self, config):
        """保存数据、变换、噪声和加载器配置，实际对象延迟到 ``setup`` 构造。

        输入参数:
            - config.transforms.featurizer: Mapping，配体元素/键词表、mask 类与坐标中心化配置。
            - config.transforms.featurizer_pocket: Mapping，可选，口袋 kNN 邻居数与中心化配置。
            - config.transforms.task: Mapping，按任务名分派 fixed prompt 和运动注释的变换配置。
            - config.noise: Mapping，按任务名分派先验、信息等级与噪声步数的训练配置。
            - config.data.dataset: Mapping，assembly 文件和子 LMDB 数据库布局。
            - config.data.task_db_weights: Mapping，任务到数据库的两级采样权重。
            - config.train: Mapping，batch size、worker、pin memory 与 persistent worker 加载器配置。
        """
        super().__init__()
        # ``self.config``：EasyDict，完整训练配置；setup 与 get_in_dims 从不同顶层组读取叶字段。
        self.config = config
        
    def get_featurizers(self):
        """
        按坐标依赖顺序构造口袋与配体特征变换。

        读取配置:
            - self.config.transforms.featurizer: Mapping，传给 ``FeaturizeMol`` 的配体特征配置。
            - self.config.transforms.featurizer_pocket: Mapping，可选，传给 ``FeaturizePocket`` 的口袋特征配置。

        返回值:
            - 仅配体配置: ``[FeaturizeMol]``。
            - 含口袋配置: ``[FeaturizePocket, FeaturizeMol]``；口袋先计算 ``pocket_center``，配体随后减去同一中心。
        """
        # ``featurizer``：FeaturizeMol，配体词表、完全图半边和坐标中心化变换。
        featurizer = FeaturizeMol(self.config.transforms.featurizer)
        if 'featurizer_pocket' in self.config.transforms:
            # ``feat_pocket``：FeaturizePocket，口袋 25 维特征、kNN 边和 pocket_center 变换。
            feat_pocket = FeaturizePocket(self.config.transforms.featurizer_pocket)
            return [feat_pocket, featurizer]  # pocket first because mol need to substract pocket center
        else:
            return [featurizer]

    def get_in_dims(self, featurizers=None):
        """
        从特征化器推导模型离散类别数和口袋连续特征宽度。

        输入参数:
            - featurizers: list|None, ``get_featurizers`` 的返回值；None 时在函数内重新构造。

        返回值:
            - in_dims.num_node_types: int, 配体原子类别数；等于元素表长度加可选 node mask 类。
            - in_dims.num_edge_types: int, 半边类别数；等于化学键类别数、非键类和可选 edge mask 类之和。
            - in_dims.pocket_in_dim: int, 仅含口袋特征化器时存在；元素 4 类、氨基酸 20 类和主链标记共 25 维。
        """
        if featurizers is None:
            # ``featurizers``：list[callable]，按口袋先、配体后的坐标依赖顺序重新构造。
            featurizers = self.get_featurizers()
        # ``num_node_types``：int K_a，配体原子分类词表宽度；直接决定原子 Embedding 和分类头的类别维度。
        num_node_types = featurizers[-1].num_node_types
        # ``num_edge_types``：int K_e，非键+真实键+可选 mask 的半边分类宽度。
        num_edge_types = featurizers[-1].num_edge_types
        # ``in_dims``：dict[str, int]，模型构造所需的离散词表宽度；有口袋时再补口袋输入宽度。
        in_dims = {
            # ``in_dims.num_node_types``：int，配体原子分类词表宽度。
            'num_node_types': num_node_types,
            # ``in_dims.num_edge_types``：int，完整半边分类词表宽度。
            'num_edge_types': num_edge_types,
        }
        if len(featurizers) == 2:
            in_dims.update({
                # ``in_dims.pocket_in_dim``：int，口袋原子的连续输入特征宽度。
                'pocket_in_dim': featurizers[0].feature_dim,
            })
        return in_dims
        
    def setup(self, stage=None):
        """
        为当前 Lightning rank 建立变换链、数据集和训练/验证加载器。

        输入参数:
            - stage: str|None, Lightning 生命周期参数；当前实现不按 stage 分支，每次调用都构造训练与验证对象。

        读取配置:
            - self.config.transforms.featurizer: Mapping，构造配体特征、完全图半边和坐标中心化变换。
            - self.config.transforms.featurizer_pocket: Mapping，可选，构造口袋特征、kNN 边与中心化变换。
            - self.config.transforms.task: Mapping，构造按 ``data.task`` 分派的任务 prompt 变换。
            - self.config.noise: Mapping，构造按 ``data.task`` 分派的训练加噪器。
            - self.config.data.dataset: Mapping，传给 ``ForeverTaskDataset`` 的 assembly 与 LMDB 布局。
            - self.config.data.task_db_weights: Mapping，传给 ``ForeverTaskDataset`` 的任务—数据库采样权重。
            - self.config.train.batch_size: int，训练和验证每批图数；VS Code 调试环境覆盖为 40。
            - self.config.train.num_workers: int，每个 rank 的训练 DataLoader worker 数。
            - self.config.train.pin_memory: bool，是否让 DataLoader 把 CPU Tensor 放入页锁定内存。
            - self.config.train.persistent_workers: bool，是否跨 epoch 保留 DataLoader worker 进程。
            - self.trainer.global_rank: int，当前 DDP 进程的全局编号。
            - self.trainer.world_size: int，参与训练的 DDP 进程总数。

        生成属性:
            - transforms: PyG ``Compose``；元素顺序为可选 cut、口袋特征、配体特征、任务变换、训练噪声器。
            - train_loader: PyG ``DataLoader``，批次大小为配置值；VS Code 调试环境强制为 40。
            - val_loader: PyG ``DataLoader``，批次大小同训练；当 ``world_size > 100`` 时 worker 数为训练的四分之一整数商。
        """

        # ``featurizers``：list[callable]，最后一个始终是 ``FeaturizeMol``；若有口袋则第一个是 ``FeaturizePocket``。
        featurizers = self.get_featurizers()
        # ``in_dims``：dict[str, int]，键集合是模型构造所需的类别数及可选口袋输入宽度。
        in_dims = self.get_in_dims(featurizers)

# [ ] --------------------------------------
        # ``task_trans``：callable，训练时在单个样本上按 ``data['task']`` 构造 fixed prompt 与刚体/扭转注释。
        task_trans = get_transforms(self.config.transforms.task, mode='train',
                                    num_node_types=in_dims['num_node_types'],)
        # ``noiser``：callable，训练时在单个样本上采样信息等级并生成三类带噪模型输入。
        noiser = get_sample_noiser(self.config.noise, in_dims['num_node_types'], in_dims['num_edge_types'],
                                   mode='train')
        # ``transform_list``：list[callable]，严格按数据依赖排列；每项原地补充同一个样本容器。
        transform_list = featurizers + [task_trans, noiser]
        if 'cut_peptide' in self.config.transforms:
            transform_list = [get_transforms(self.config.transforms.cut_peptide)] + transform_list
        # ``self.transforms``：Compose，单样本变换按列表顺序串行执行并共享同一 Data 对象。
        self.transforms = Compose(transform_list)
        # ``t``：callable，Compose 中当前变换；读取其声明的批归属跟踪字段。
        # ``follow_batch``：list[str]，PyG 为每个字段 k 额外生成 ``k_batch``，记录拼接后第一维的图归属。
        follow_batch = sum([getattr(t, 'follow_batch', []) for t in self.transforms.transforms], [])
        # ``t``：callable，Compose 中当前变换；读取其声明的不参与批拼接字段。
        # ``exclude_keys``：list[str]，原始大数组、Python 映射和仅单样本有效字段不会进入 ``Batch``。
        exclude_keys = sum([getattr(t, 'exclude_keys', []) for t in self.transforms.transforms], [])
        # self.num_node_types = in_dims['num_node_types']
        # self.num_edge_types = in_dims['num_edge_types']

        # # Datasets and sampler
        # ``data_cfg``：EasyDict，包含 ``dataset`` 的 LMDB 布局和 ``task_db_weights`` 的两级采样概率。
        data_cfg = self.config.data
        # ``num_samplers_args``：dict[str, int]，共同定义每个 DataLoader worker 在全局 DDP 采样器中的唯一编号。
        num_samplers_args = {
            # ``num_samplers_args.num_workers``：int，当前 rank 创建的训练数据 worker 数。
            'num_workers': self.config.train.num_workers,
            # ``num_samplers_args.global_rank``：int，当前 DDP 进程的全局编号。
            'global_rank': self.trainer.global_rank,
            # ``num_samplers_args.world_size``：int，参与训练的 DDP 进程总数。
            'world_size': self.trainer.world_size,
        }     
# [ ] --------------------------------------

        # ``train_set``：IterableDataset，变换发生在批处理前，因此每次 noiser 只处理一个确定任务的样本。
        train_set = ForeverTaskDataset(data_cfg.dataset, data_cfg.task_db_weights,'train',
                                       transforms=self.transforms, shuffle=True, **num_samplers_args)
        if num_samplers_args['world_size'] > 100:
            # ``divider``：int 4，超大 DDP 规模下验证 worker 数降为四分之一。
            divider = 4
        else:
            # ``divider``：int 1，常规规模下验证与训练使用相同 worker 数。
            divider = 1
        # ``num_samplers_args.num_workers``：int，把实际验证 worker 数同步传给 ForeverTaskDataset 的区间分片逻辑。
        num_samplers_args['num_workers'] = self.config.train.num_workers//divider
        # ``val_set``：IterableDataset，使用同一任务分布但 ``shuffle=False`` 且一轮后停止。
        val_set = ForeverTaskDataset(data_cfg.dataset, data_cfg.task_db_weights, 'val',
                                     transforms=self.transforms, shuffle=False, **num_samplers_args)

        # # Dataloaders
        train_cfg = self.config.train
        # ``self.train_loader``：PyG DataLoader，拼接无限训练单样本流并生成 follow_batch 图归属向量。
        self.train_loader = DataLoader(train_set, batch_size=train_cfg.batch_size if not is_vscode else 40,
                                       num_workers=train_cfg.num_workers, pin_memory=train_cfg.pin_memory,
                                       follow_batch=follow_batch, exclude_keys=exclude_keys,
                                       persistent_workers=train_cfg.persistent_workers,
        )
        # ``self.val_loader``：PyG DataLoader，拼接一次有限验证流；worker 数可能按 divider 缩减。
        self.val_loader = DataLoader(val_set, batch_size=train_cfg.batch_size if not is_vscode else 40,
                                     num_workers=train_cfg.num_workers//divider, pin_memory=train_cfg.pin_memory,
                                     follow_batch=follow_batch, exclude_keys=exclude_keys,
                                     persistent_workers=train_cfg.persistent_workers,
        )
    def train_dataloader(self):
        return self.train_loader
    def val_dataloader(self):
        return self.val_loader


class ModelLightning(pl.LightningModule):
    """
    封装共享去噪网络、逐任务损失、优化器和训练/验证日志。

    构造参数:
        - config.model: ``PMAsymDenoiser`` 配置；``name`` 必须为 ``pm_asym_denoiser`` 才会创建 ``self.model``。
        - config.model.pretrained: str|空字符串；非空时读取 checkpoint ``state_dict`` 中以 ``model.`` 开头的键并去掉该前缀。
        - config.loss: 损失配置；当前 reduced 配置由 ``IndividualTasksLoss`` 分别统计 mixed 与各任务分量。
        - config.train: 优化器、调度器、warmup 与日志相关设置。
        - args.num_gpus: int，当前节点使用的 GPU 数。
        - args.multi_node: bool，是否跨多个计算节点训练。
        - num_node_types: int, 配体原子类别数；传给原子 Embedding 与分类头。
        - num_edge_types: int, 配体半边类别数；传给半边 Embedding 与分类头。
        - ``**kwargs``: 至少含 ``pocket_in_dim``，传给 ``PMAsymDenoiser`` 的口袋线性嵌入层。

    前向输入:
        - batch.node_in: LongTensor，形状为 (N_all,)，带噪原子类别。
        - batch.pos_in: FloatTensor，形状为 (N_all, 3)，带噪配体局部坐标，单位 Å。
        - batch.halfedge_in: LongTensor，形状为 (H_all,)，带噪半边类别。
        - batch.halfedge_index: LongTensor，形状为 (2, H_all)，完全图半边端点。
        - batch.fixed_node: LongTensor|BoolTensor，形状为 (N_all,)，原子类别条件掩码。
        - batch.fixed_pos: LongTensor|BoolTensor，形状为 (N_all,)，坐标条件掩码。
        - batch.fixed_halfedge: LongTensor|BoolTensor，形状为 (H_all,)，半边类别条件掩码。
        - batch.fixed_halfdist: LongTensor|BoolTensor，形状为 (H_all,)，半边距离条件掩码。
        - batch.node_type_batch: LongTensor，形状为 (N_all,)，逐原子图归属编号。
        - batch.pocket_atom_feature: FloatTensor，形状为 (P, D_p_raw)，口袋输入特征。
        - batch.pocket_pos: FloatTensor，形状为 (P, 3)，口袋局部坐标，单位 Å。
        - batch.pocket_knn_edge_index: LongTensor，形状为 (2, E_p)，口袋 kNN 有向边端点。
        - batch.pocket_pos_batch: LongTensor，形状为 (P,)，逐口袋原子图归属编号。
        - batch.is_peptide: LongTensor，形状为 (N_all,)，小分子任务为全 0；仅配置请求时读取。

    前向输出:
        - pred_node: (N_all, K_n), 每个配体原子的真值类别 logits。
        - pred_pos: (N_all, 3), 每个配体原子的去噪坐标，坐标原点与 ``pos_in`` 一致，单位 Å。
        - pred_halfedge: (H_all, K_e), 每条无向半边的真值类别 logits。
        - confidence_node: (N_all, 1), 可选原子类别置信度原始分数；配置 ``add_output`` 含 ``confidence`` 时存在。
        - confidence_pos: (N_all, 1), 可选原子坐标置信度原始分数；与原子第一维对齐。
        - confidence_halfedge: (H_all, 1), 可选半边类别置信度原始分数；与半边第一维对齐。

    损失输出:
        - <task>/node: 当前任务未固定原子类别的标量损失。
        - <task>/pos: 当前任务未固定坐标的标量损失。
        - <task>/edge: 当前任务未固定半边类别的标量损失。
        - <task>/fixed_node: 当前任务固定原子类别的标量损失。
        - <task>/fixed_pos: 当前任务固定坐标的标量损失。
        - <task>/fixed_edge: 当前任务固定半边类别的标量损失。
        - <task>/dist: 当前任务同域待恢复距离的标量损失。
        - <task>/fixed_dist: 当前任务固定距离的标量损失。
        - <task>/dih: 当前任务二面角周期标量损失。
        - <task>/total: 当前任务各叶按 ``config.loss.weights`` 汇总的标量。
        - mixed/total: 混合批次各恢复叶与可选 confidence/口袋距离项的反向传播标量。
    """
    def __init__(self, config, args, num_node_types, num_edge_types, **kwargs):
        super(ModelLightning, self).__init__()
        # ``config.model.name``：str，模型注册名；当前必须为 ``pm_asym_denoiser``。
        # ``config.model.pretrained``：str|缺省，可选预训练 Lightning checkpoint 路径。
        # ``config.model``：Mapping，PMAsymDenoiser 的结构、输出头和附加节点特征叶。
        # ``config.loss.name``：str，损失注册名；当前为 ``individual_tasks``。
        # ``config.loss.weights``：Mapping，各恢复分量的标量权重叶。
        # ``config.loss.tasks``：list[str]，逐任务日志与掩码的有序任务名。
        # ``config.loss.confidence``：Mapping|缺省，置信度目标与分量权重叶。
        # ``config.train.optimizer``：Mapping，优化器类型、学习率、权重衰减和动量叶。
        # ``config.train.scheduler.warmup_step``：int，线性 warmup 步数。
        # ``config.train.scheduler.instance``：Mapping，调度器类型与构造参数叶。
        # ``config.train.scheduler.params``：Mapping，Lightning 调度间隔、频率与监控指标叶。
        # ``self.config``：EasyDict，保留上述模型、损失和优化训练叶。
        self.config = config
        self.save_hyperparameters()
        # ``self.num_gpus``：int，命令行声明的当前节点 GPU 数。
        self.num_gpus = args.num_gpus
        # ``self.multi_node``：bool，是否跨多个计算节点运行。
        self.multi_node = args.multi_node
        # ``self.sync_dist``：bool，多 GPU 或多节点时让 Lightning 日志跨进程同步归约。
        self.sync_dist = (self.num_gpus>1) or self.multi_node

        # Model
        if self.config.model.name == 'pm_asym_denoiser':
            # ``self.model``：PMAsymDenoiser，共享原子交互去噪网络；词表宽度与口袋输入宽度由 DataModule 推导。
            self.model = PMAsymDenoiser(config=self.config.model,
                                  num_node_types=num_node_types,
                                  num_edge_types=num_edge_types, **kwargs)
        
        if getattr(self.config.model, 'pretrained', ''):
            # ``ckpt``：dict，CPU 上读取的 Lightning checkpoint；预期含 ``state_dict`` 叶。
            # ``ckpt.state_dict``：dict[str, Tensor]，Lightning 模块参数全名到权重张量的映射。
            ckpt = torch.load(self.config.model.pretrained, map_location='cpu')
            # ``k``：str，checkpoint 中当前参数全名；只有 ``model.`` 前缀的主网络参数被保留。
            # ``value``：Tensor，当前 checkpoint 参数值；形状必须与去前缀后的模型参数一致。
            self.model.load_state_dict({k[6:]:value for k, value in ckpt['state_dict'].items()
                                        if k.startswith('model.')})
            print('Load pretrained model from', self.config.model.pretrained)
        

        # ``self.loss_func``：nn.Module，当前配置为 IndividualTasksLoss，返回逐任务及 mixed 标量映射。
        self.loss_func = get_loss_func(self.config.loss)
        # self.skip = False
        # self.automatic_optimization = False
        # self.gradient_clip_val = getattr(self.config.train, 'gradient_clip_val', 0)
        # self.gradient_clip_algorithm = getattr(self.config.train, 'gradient_clip_algorithm', 'norm')

    def forward(self, batch):
        """把带噪分子—口袋批次交给共享去噪模型。

        输入字段:
            - batch.node_in: LongTensor，形状为 (N,)，带噪原子类别。
            - batch.pos_in: FloatTensor，形状为 (N, 3)，带噪配体局部坐标，单位 Å。
            - batch.halfedge_in: LongTensor，形状为 (H,)，带噪半边类别。
            - batch.halfedge_index: LongTensor，形状为 (2, H)，完全图半边端点。
            - batch.fixed_node: LongTensor|BoolTensor，形状为 (N,)，原子类别条件掩码。
            - batch.fixed_pos: LongTensor|BoolTensor，形状为 (N,)，坐标条件掩码。
            - batch.fixed_halfedge: LongTensor|BoolTensor，形状为 (H,)，半边类别条件掩码。
            - batch.fixed_halfdist: LongTensor|BoolTensor，形状为 (H,)，半边距离条件掩码。
            - batch.node_type_batch: LongTensor，形状为 (N,)，逐原子图归属编号。
            - batch.pocket_atom_feature: FloatTensor，形状为 (P, D_p_raw)，口袋输入特征。
            - batch.pocket_pos: FloatTensor，形状为 (P, 3)，口袋局部坐标，单位 Å。
            - batch.pocket_knn_edge_index: LongTensor，形状为 (2, E_p)，口袋 kNN 有向边端点。
            - batch.pocket_pos_batch: LongTensor，形状为 (P,)，逐口袋原子图归属编号。
            - batch.is_peptide: LongTensor，形状为 (N,)，小分子任务为全 0；仅配置请求时读取。

        返回字段:
            - pred_node: FloatTensor，形状为 (N, K_n)，干净原子类别 logits。
            - pred_pos: FloatTensor，形状为 (N, 3)，干净配体局部坐标预测，单位 Å。
            - pred_halfedge: FloatTensor，形状为 (H, K_e)，干净半边类别 logits。
            - confidence_node: FloatTensor，形状为 (N, 1)，可选原子 confidence 原始输出。
            - confidence_pos: FloatTensor，形状为 (N, 1)，可选坐标 confidence 原始输出。
            - confidence_halfedge: FloatTensor，形状为 (H, 1)，可选半边 confidence 原始输出。
        """
        return self.model(batch)
    
    def reduce_batch(self, batch):
        """
        OOM 后把当前 PyG 批次裁成前一半图并重新拼接到原设备。

        输入参数:
            - batch: PyG ``Batch``，长度为图数量 B；必须能转回 ``data_list``。

        返回值:
            - batch: PyG ``Batch``，包含原批次前 ``floor(B/2)`` 个图；所有 ``*_batch`` 对应的源字段重新加入 ``follow_batch``。

        副作用:
            - 删除现有参数梯度、把原批次移到 CPU、清空 CUDA cache；样本次序不重排。
            - 当 B=1 时 ``new_bs=0``，当前实现没有在本函数内阻止空批次。
        """
        # 清除 (与各参数同形) 的梯度张量，为重新前向释放显存。
        for p in self.model.parameters():
            if p.grad is not None:
                del p.grad  # free some memory
        # gc.collect()
        torch.cuda.empty_cache()

        # drop last 10 percent
        # ``new_bs``：int，既有实现按 ``len(batch)`` 的一半向下取整；常见 PyG Batch 的 ``len`` 返回字段数而非图数，若本意是图数应读取 ``batch.num_graphs``，本学习分支只记录不修复。
        new_bs = int(len(batch) * 0.5)
        print(f"\nOut of memory error occurred in step. Reduce bs {len(batch)} to {new_bs}")
        # ``device``：torch.device，裁剪完成后新 Batch 返回该设备。
        device = batch.batch.device
        # follow_batch = [k.replace('_batch','') for k in batch.keys if k.endswith('_batch')]
        # ``k``：str，当前 Batch 字段名；仅保留以 ``_batch`` 结尾的图归属叶并去掉后缀。
        # ``follow_batch``：list[str]，从现有 ``<field>_batch`` 键反推重建 Batch 时需要追踪的源字段。
        follow_batch = [k.replace('_batch','') for k in batch.keys() if k.endswith('_batch')]
        # batch_cpu = batch.detach().cpu()
        batch_cpu = batch.cpu()
        del batch
        # gc.collect()
        torch.cuda.empty_cache()
        # ``data_list``：list[Data]，长度 B；每个元素恢复一个裁剪前的单分子字段集合。
        data_list = batch_cpu.to_data_list()
        del data_list[new_bs:]
        # ``batch``：PyG Batch，把保留的前 new_bs 个图重新拼接、重建索引偏移和图归属向量后移回原设备。
        batch = Batch.from_data_list(data_list[:new_bs], follow_batch=follow_batch).to(device)
        # size_list = [data.num_nodes for data in data_list]  # choose large first
        # idx_sort = np.argsort(size_list)[::-1]
        # batch = Batch.from_data_list([data_list[i] for i in idx_sort[:new_bs]],
        #                              follow_batch=follow_batch).to(device)
        return batch

    def training_step2(self, batch, batch_idx):
        """保留的手动优化/OOM 重试实现；名称不是 Lightning 当前 ``training_step`` 钩子，本训练入口不会调用。"""
        # ``opt``：Optimizer，Lightning 管理的当前优化器；仅此停用实验钩子使用。
        opt = self.optimizers()
        # ``sch``：调度器，Lightning 管理的当前学习率调度器。
        sch = self.lr_schedulers()
        while True:
            try:
                opt.zero_grad()
                # forward
                outputs = self.model(batch)
                # ``loss_dict``：dict[str, scalar Tensor]，停用实验路径中的逐任务损失映射。
                loss_dict = self.loss_func(batch, outputs)
                # bachward
                self.manual_backward(loss_dict['loss'])
                if self.gradient_clip_val > 0:
                    self.clip_gradients(opt, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm=self.gradient_clip_algorithm)
                opt.step()
                break
            except Exception as e:
                if isinstance(e, RuntimeError) and "out of memory" in str(e):
                    opt.zero_grad()
                    # del outputs
                    try:
                        del outputs
                    except Exception as e:
                        print(e)
                        pass
                    gc.collect()
                    torch.cuda.empty_cache()
                    # del loss_dict
                    try:
                        del loss_dict
                    except Exception as e:
                        print(e)
                        pass
                    gc.collect()
                    torch.cuda.empty_cache()
                    if len(batch) >= 4:
                        # ``batch``：PyG Batch，OOM 后裁为前一半图并重建索引。
                        batch = self.reduce_batch(batch)
                    else:
                        return None
                else:
                    raise e

        sch.step()
        # ``loss``：scalar Tensor，停用路径假定损失映射存在无前缀 loss 键。
        loss = loss_dict['loss']
        # ``loss_dict``：dict[str, scalar Tensor]，所有键加 train/ 前缀用于日志。
        loss_dict = {'train/'+k: v for k, v in loss_dict.items()}
        self.log_dict({k:v for k,v in loss_dict.items() if '_fixed/' not in k}, batch_size=batch.num_graphs,
                      sync_dist=self.sync_dist, prog_bar=True, logger=True)
        self.log_dict({k:v for k,v in loss_dict.items() if '_fixed/' in k}, batch_size=batch.num_graphs,
                      sync_dist=self.sync_dist, prog_bar=False, logger=True)
        # self.log('train_loss', loss, sync_dist=(self.num_gpus>1))
        # self.log('train/grad', orig_grad_norm)
        self.log('train/lr', opt.param_groups[0]['lr'], sync_dist=self.sync_dist)
        # ``norms``：dict[str, Tensor]，每个可训练参数的梯度二范数。
        norms = grad_norm(self.model, norm_type=2)
        if norms:
            # ``max_norms``：scalar Tensor，所有参数梯度二范数中的最大值。
            max_norms = max(norms.values())
            self.log_dict({'train/max_norm': max_norms}, sync_dist=self.sync_dist)
        return loss

    def on_before_optimizer_step(self, optimizer):
        """
        在优化器更新前执行线性 warmup，并记录当前学习率与最大参数梯度范数。

        输入参数:
            - optimizer: Optimizer, Lightning 即将执行 ``step`` 的优化器。

        状态变化:
            - global_step < warmup_step 时，第 0 步缓存每个参数组的基础学习率，此后按 ``(global_step + 1) / warmup_step`` 线性缩放。
            - ``train/lr`` 为第一个参数组当前学习率；``train/max_norm`` 为 ``grad_norm`` 返回映射中的最大二范数。
        """
        # warmup lr increase
        if self.global_step < self.warmup_step:
            if self.global_step == 0:
                # ``self.base_lr_list``：list[float]，按 optimizer.param_groups 顺序缓存 warmup 前基础学习率。
                self.base_lr_list = []
                # ``param_group``：dict，当前优化器参数组，至少含当前基础学习率叶 ``lr``。
                for param_group in self.optimizers().param_groups:
                    self.base_lr_list.append(param_group['lr'])
            # ``ratio``：float，当前 warmup 完成比例；第一个优化步为 1/warmup_step，最后一个 warmup 步达到 1。
            ratio = float((self.global_step+1) / self.warmup_step)
            # ``i``：int，当前参数组序号，同时索引 ``base_lr_list``。
            # ``param_group``：dict，当前待写入缩放后 ``lr`` 的优化器参数组。
            for i, param_group in enumerate(self.optimizers().param_groups):
                    # ``param_group.lr``：float，把第 i 个参数组学习率设为其基础值乘当前 warmup 比例。
                    param_group['lr'] = self.base_lr_list[i] * ratio
        
        
        self.log('train/lr', optimizer.param_groups[0]['lr'], sync_dist=self.sync_dist, prog_bar=True)
        # ``norms``：dict[str, Tensor]，当前反向后各参数梯度二范数。
        norms = grad_norm(self.model, norm_type=2)
        if norms:
            # ``max_norms``：scalar Tensor，把最大梯度范数移到模块设备后记录。
            max_norms = max(norms.values()).to(self.device)
            self.log_dict({'train/max_norm': max_norms}, sync_dist=self.sync_dist)

    # def on_after_backward(self) -> None:
    #     for name, param in self.model.named_parameters():
    #         if param.grad.isinf().any() or param.grad.isnan().any():
    #             print(f"Inf or NaN in {name}")
    #     return super().on_after_backward()

    def training_step(self, batch, batch_idx):
        """
        对一个多任务 PyG 批次执行前向、损失计算和 OOM 裁剪重试。

        输入参数:
            - batch.node_in: LongTensor，形状为 (N,)，带噪原子类别。
            - batch.pos_in: FloatTensor，形状为 (N, 3)，带噪配体局部坐标，单位 Å。
            - batch.halfedge_in: LongTensor，形状为 (H,)，带噪半边类别。
            - batch.halfedge_index: LongTensor，形状为 (2, H)，完全图半边端点。
            - batch.fixed_node: LongTensor|BoolTensor，形状为 (N,)，原子类别条件掩码。
            - batch.fixed_pos: LongTensor|BoolTensor，形状为 (N,)，坐标条件掩码。
            - batch.fixed_halfedge: LongTensor|BoolTensor，形状为 (H,)，半边类别条件掩码。
            - batch.fixed_halfdist: LongTensor|BoolTensor，形状为 (H,)，半边距离条件掩码。
            - batch.node_type_batch: LongTensor，形状为 (N,)，逐原子图归属编号。
            - batch.halfedge_type_batch: LongTensor，形状为 (H,)，逐半边图归属编号。
            - batch.pocket_atom_feature: FloatTensor，形状为 (P, D_p_raw)，口袋输入特征。
            - batch.pocket_pos: FloatTensor，形状为 (P, 3)，口袋局部坐标，单位 Å。
            - batch.pocket_knn_edge_index: LongTensor，形状为 (2, E_p)，口袋 kNN 有向边端点。
            - batch.pocket_pos_batch: LongTensor，形状为 (P,)，逐口袋原子图归属编号。
            - batch.node_type: LongTensor，形状为 (N,)，干净原子类别监督。
            - batch.node_pos: FloatTensor，形状为 (N, 3)，干净配体局部坐标监督，单位 Å。
            - batch.halfedge_type: LongTensor，形状为 (H,)，干净半边类别监督。
            - batch.task: list[str]，长度为 B，逐图任务名。
            - batch.domain_node_index: LongTensor，形状为 (2, K)，刚体域—原子归属索引。
            - batch.tor_bonds_anno: LongTensor，形状为 (T, 3)，扭转层级与轴端点。
            - batch.dihedral_pairs_anno: LongTensor，形状为 (Q, 3)，扭转行号与二面角外侧端点。
            - batch_idx: int, 当前 epoch 内批次编号；本实现不参与数值计算。

        中间结构:
            - outputs.pred_node: FloatTensor，形状为 (N, C_n)，干净原子类别 logits。
            - outputs.pred_pos: FloatTensor，形状为 (N, 3)，干净配体坐标预测，单位 Å。
            - outputs.pred_halfedge: FloatTensor，形状为 (H, C_e)，干净半边类别 logits。
            - outputs.confidence_node: FloatTensor，形状为 (N, 1)，可选原子 confidence 原始输出。
            - outputs.confidence_pos: FloatTensor，形状为 (N, 1)，可选坐标 confidence 原始输出。
            - outputs.confidence_halfedge: FloatTensor，形状为 (H, 1)，可选半边 confidence 原始输出。
            - loss_dict.<scope>/node: 标量 Tensor，待恢复原子类别损失。
            - loss_dict.<scope>/fixed_node: 标量 Tensor，条件原子类别损失。
            - loss_dict.<scope>/pos: 标量 Tensor，待恢复坐标损失。
            - loss_dict.<scope>/fixed_pos: 标量 Tensor，条件坐标损失。
            - loss_dict.<scope>/edge: 标量 Tensor，待恢复半边类别损失。
            - loss_dict.<scope>/fixed_edge: 标量 Tensor，条件半边类别损失。
            - loss_dict.<scope>/dist: 标量 Tensor，同域待恢复距离损失。
            - loss_dict.<scope>/fixed_dist: 标量 Tensor，条件距离损失。
            - loss_dict.<scope>/dih: 标量 Tensor，二面角周期损失。
            - loss_dict.<scope>/total: 标量 Tensor，当前 scope 加权损失和。
            - loss_dict.mixed/cfd_total: 标量 Tensor，可选 confidence 总损失。
            - loss_dict.mixed/cfd_node: 标量 Tensor，可选原子 confidence BCE。
            - loss_dict.mixed/cfd_pos: 标量 Tensor，可选坐标 confidence MSE。
            - loss_dict.mixed/cfd_edge: 标量 Tensor，可选半边 confidence BCE。
            - loss_dict.mixed/p_dist: 标量 Tensor，可选配体—口袋距离损失。

        返回值:
            - loss: 标量 Tensor；优先取 ``loss_dict['loss']``，当前 ``IndividualTasksLoss`` 路径取 ``loss_dict['mixed/total']``，由 Lightning 自动反向传播。

        OOM 分支:
            - 仅捕获消息含 ``out of memory`` 的 ``RuntimeError``，调用 ``reduce_batch`` 后重试；其他异常原样抛出。
        """

        while True:
            try:
                # ``outputs.pred_node``：FloatTensor，形状为 (N, C_n)，原子类别 logits。
                # ``outputs.pred_pos``：FloatTensor，形状为 (N, 3)，配体局部坐标预测，单位 Å。
                # ``outputs.pred_halfedge``：FloatTensor，形状为 (H, C_e)，半边类别 logits。
                # ``outputs.confidence_node``：FloatTensor，形状为 (N, 1)，可选原子 confidence 原始输出。
                # ``outputs.confidence_pos``：FloatTensor，形状为 (N, 1)，可选坐标 confidence 原始输出。
                # ``outputs.confidence_halfedge``：FloatTensor，形状为 (H, 1)，可选半边 confidence 原始输出。
                outputs = self.model(batch)
                # ``loss_dict.<scope>/node``：标量 Tensor，待恢复原子类别损失。
                # ``loss_dict.<scope>/fixed_node``：标量 Tensor，条件原子类别损失。
                # ``loss_dict.<scope>/pos``：标量 Tensor，待恢复坐标损失。
                # ``loss_dict.<scope>/fixed_pos``：标量 Tensor，条件坐标损失。
                # ``loss_dict.<scope>/edge``：标量 Tensor，待恢复半边类别损失。
                # ``loss_dict.<scope>/fixed_edge``：标量 Tensor，条件半边类别损失。
                # ``loss_dict.<scope>/dist``：标量 Tensor，同域待恢复距离损失。
                # ``loss_dict.<scope>/fixed_dist``：标量 Tensor，条件距离损失。
                # ``loss_dict.<scope>/dih``：标量 Tensor，二面角周期损失。
                # ``loss_dict.<scope>/total``：标量 Tensor，当前 scope 加权损失和。
                # ``loss_dict.mixed/cfd_total``：标量 Tensor，可选 confidence 总损失。
                # ``loss_dict.mixed/cfd_node``：标量 Tensor，可选原子 confidence BCE。
                # ``loss_dict.mixed/cfd_pos``：标量 Tensor，可选坐标 confidence MSE。
                # ``loss_dict.mixed/cfd_edge``：标量 Tensor，可选半边 confidence BCE。
                # ``loss_dict.mixed/p_dist``：标量 Tensor，可选配体—口袋距离损失。
                loss_dict = self.loss_func(batch, outputs)
                # print('\n', len(batch.node_type_batch), len(batch.pocket_pos_batch))
                break
            except Exception as e:
                if isinstance(e, RuntimeError) and "out of memory" in str(e):
                    # zhale = 1.
                    print('\nOOM', len(batch.node_type_batch), len(batch.pocket_pos_batch))
                    # if len(batch) >= 4:
                    batch = self.reduce_batch(batch)
                    # else:
                    #     return None
                else:
                    raise e

        if 'loss' in loss_dict:
            # ``loss``：scalar Tensor，兼容旧损失实现直接返回的总损失键。
            loss = loss_dict['loss']
        else:
            # ``loss``：标量 Tensor，当前 reduced 配置的实际优化目标。
            loss = loss_dict['mixed/total']

        # ``k``：str，当前未加训练阶段前缀的损失键。
        # ``v``：标量 Tensor，当前损失键对应的数值。
        # ``loss_dict``：dict[str, scalar Tensor]，所有原损失键加 ``train_`` 前缀；记录时再把 ``train_mixed`` 替换为 ``train``。
        loss_dict = {'train_'+k: v for k, v in loss_dict.items()}
        # if self.global_step != 73:
        # if True:
        # ``k``：str，当前训练损失键；此日志映射只保留 ``mixed/`` 叶并把展示前缀改为 ``train``。
        # ``v``：标量 Tensor，当前 mixed 损失叶值。
        self.log_dict({k.replace('train_mixed', 'train'):v for k, v in loss_dict.items() if 'mixed/' in k}, batch_size=batch.num_graphs,
                    sync_dist=self.sync_dist, prog_bar=True, logger=True)
        # ``k``：str，当前训练损失键；此日志映射排除 ``mixed/`` 叶并保留命名任务前缀。
        # ``v``：标量 Tensor，当前命名任务损失叶值。
        self.log_dict({k:v for k, v in loss_dict.items() if 'mixed/' not in k}, batch_size=batch.num_graphs,
                    sync_dist=self.sync_dist, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        在验证批次上执行与训练相同的前向和损失分解，不更新参数。

        输入参数:
            - batch.node_in: LongTensor，形状为 (N,)，带噪原子类别。
            - batch.pos_in: FloatTensor，形状为 (N, 3)，带噪配体局部坐标，单位 Å。
            - batch.halfedge_in: LongTensor，形状为 (H,)，带噪半边类别。
            - batch.halfedge_index: LongTensor，形状为 (2, H)，完整图半边端点，索引配体原子维。
            - batch.fixed_node: LongTensor|BoolTensor，形状为 (N,)，0 表示原子类别待恢复，1 表示条件。
            - batch.fixed_pos: LongTensor|BoolTensor，形状为 (N,)，0 表示坐标待恢复，1 表示条件。
            - batch.fixed_halfedge: LongTensor|BoolTensor，形状为 (H,)，0 表示半边类别待恢复，1 表示条件。
            - batch.fixed_halfdist: LongTensor|BoolTensor，形状为 (H,)，0 表示端点距离待恢复，1 表示条件。
            - batch.node_type: LongTensor，形状为 (N,)，干净原子类别监督。
            - batch.node_pos: FloatTensor，形状为 (N, 3)，干净配体局部坐标监督，单位 Å。
            - batch.halfedge_type: LongTensor，形状为 (H,)，干净半边类别监督。
            - batch.node_type_batch: LongTensor，形状为 (N,)，逐原子图归属编号，取值范围为 ``[0, B)``。
            - batch.halfedge_type_batch: LongTensor，形状为 (H,)，逐半边图归属编号，取值范围为 ``[0, B)``。
            - batch.task: list[str]，长度为 B，第 b 项是图 b 的任务名。
            - batch.domain_node_index: LongTensor，形状为 (2, K)，第一行是刚体域号，第二行是批内原子编号。
            - batch.tor_bonds_anno: LongTensor，形状为 (T, 3)，每行是执行层级和两个扭转轴端点。
            - batch.dihedral_pairs_anno: LongTensor，形状为 (Q, 3)，每行是扭转行号和两个二面角外侧端点。
            - batch_idx: int, 当前验证轮内批次编号；本实现不参与数值计算。

        返回字段:
            - loss_dict.val_<scope>/node: 标量 Tensor，待恢复原子类别损失。
            - loss_dict.val_<scope>/fixed_node: 标量 Tensor，条件原子类别损失。
            - loss_dict.val_<scope>/pos: 标量 Tensor，待恢复坐标损失。
            - loss_dict.val_<scope>/fixed_pos: 标量 Tensor，条件坐标损失。
            - loss_dict.val_<scope>/edge: 标量 Tensor，待恢复半边类别损失。
            - loss_dict.val_<scope>/fixed_edge: 标量 Tensor，条件半边类别损失。
            - loss_dict.val_<scope>/dist: 标量 Tensor，同域待恢复距离损失。
            - loss_dict.val_<scope>/fixed_dist: 标量 Tensor，条件距离损失。
            - loss_dict.val_<scope>/dih: 标量 Tensor，二面角周期损失。
            - loss_dict.val_<scope>/total: 标量 Tensor，当前 scope 加权损失和。
            - loss_dict.val_mixed/cfd_total: 标量 Tensor，可选 confidence 总损失。
            - loss_dict.val_mixed/cfd_node: 标量 Tensor，可选原子 confidence BCE。
            - loss_dict.val_mixed/cfd_pos: 标量 Tensor，可选坐标 confidence MSE。
            - loss_dict.val_mixed/cfd_edge: 标量 Tensor，可选半边 confidence BCE。
            - loss_dict.val_mixed/p_dist: 标量 Tensor，可选配体—口袋距离损失。
        """
        while True:
            try:
                # ``outputs``：dict[str, Tensor]，验证批次模型输出；原子/半边第一维与 batch 对齐。
                outputs = self.model(batch)
                # ``loss_dict``：dict[str, scalar Tensor]，验证批次逐任务和 mixed 损失。
                loss_dict = self.loss_func(batch, outputs)
                break
            except Exception as e:
                if isinstance(e, RuntimeError) and "out of memory" in str(e):
                    # try:
                    #     del outputs
                    #     del loss_dict
                    # except:
                    #     pass
                    # gc.collect()
                    # torch.cuda.empty_cache()
                    # ``batch``：PyG Batch，验证 OOM 时同样裁为前一半图后重试。
                    batch = self.reduce_batch(batch)
                else:
                    raise e
        
        # ``k``：str，当前未加验证阶段前缀的损失键。
        # ``v``：标量 Tensor，当前损失键对应的数值。
        # ``loss_dict``：dict[str, scalar Tensor]，所有损失键加 ``val_`` 前缀用于区分训练日志。
        loss_dict = {'val_'+k: v for k, v in loss_dict.items()}
        # ``k``：str，当前验证损失键；此日志映射只保留 mixed 且非 fixed 的叶，并把展示前缀改为 ``val``。
        # ``v``：标量 Tensor，当前 mixed 非 fixed 损失叶值。
        self.log_dict({k.replace('val_mixed', 'val'):v for k,v in loss_dict.items() if ('mixed/' in k) and ('fixed_' not in k)}, batch_size=batch.num_graphs,
                      sync_dist=self.sync_dist, prog_bar=True, logger=True)
        # ``k``：str，当前验证损失键；此日志映射只保留 mixed 的 fixed 叶，并把展示前缀改为 ``val``。
        # ``v``：标量 Tensor，当前 mixed fixed 损失叶值。
        self.log_dict({k.replace('val_mixed', 'val'):v for k,v in loss_dict.items() if ('mixed/' in k) and ('fixed_' in k)}, batch_size=batch.num_graphs,
                      sync_dist=self.sync_dist, prog_bar=False, logger=True)
        # ``k``：str，当前验证损失键；此日志映射只保留命名任务叶。
        # ``v``：标量 Tensor，当前命名任务损失叶值。
        self.log_dict({k:v for k,v in loss_dict.items() if ('mixed/' not in k)}, batch_size=batch.num_graphs,
                      sync_dist=self.sync_dist, prog_bar=False, logger=True)

        self.log('val/loss', loss_dict['val_mixed/total'], batch_size=batch.num_graphs,
                      sync_dist=self.sync_dist, prog_bar=False, logger=True)
        return loss_dict

    def configure_optimizers(self):
        """
        从 ``config.train`` 构造优化器、调度器和 Lightning 调度元数据。

        读取配置:
            - self.config.train.optimizer: Mapping，优化器名称、学习率、权重衰减及实现特有参数。
            - self.config.train.scheduler.warmup_step: int，可选，线性 warmup 的优化步数；缺省为 0。
            - self.config.train.scheduler.instance: Mapping，调度器名称及其实例化参数。
            - self.config.train.scheduler.params.interval: str，Lightning 调度时间单位，取 ``step`` 或 ``epoch``。
            - self.config.train.scheduler.params.frequency: int，每隔多少个 interval 调用一次调度器。
            - self.config.train.scheduler.params.monitor: str，需要指标驱动时读取的 Lightning 日志键。

        返回值:
            - optimizer: ``torch.optim.Optimizer``，参数只来自 ``self.model``；类型与超参数由 ``config.train.optimizer`` 决定。
            - lr_scheduler.scheduler: 调度器实例；由 ``config.train.scheduler.instance`` 构造。
            - lr_scheduler.interval: str，Lightning 按 step 或 epoch 调度的时间单位。
            - lr_scheduler.frequency: int，每隔多少个 interval 调用一次调度器。
            - lr_scheduler.monitor: str，ReduceLROnPlateau 读取的日志标量名。

        状态变化:
            - warmup_step: int, 从 scheduler 配置读取；只由 ``on_before_optimizer_step`` 使用，不包装调度器实例。
        """
        # ``optimizer``：torch.optim.Optimizer，只接收 ``self.model`` 参数和 optimizer 配置叶。
        optimizer = get_optimizer(self.config.train.optimizer, self.model)
        # ``scheduler_config``：EasyDict，包含 ``warmup_step``、``instance`` 与 ``params`` 三个子对象。
        # ``scheduler_config.warmup_step``：int|缺省，线性 warmup 步数。
        # ``scheduler_config.instance``：Mapping，调度器名称及实例化参数。
        # ``scheduler_config.params.interval``：str，Lightning 调度时间单位。
        # ``scheduler_config.params.frequency``：int，Lightning 调度调用频率。
        # ``scheduler_config.params.monitor``：str，需要指标驱动时读取的日志键。
        scheduler_config = self.config.train.scheduler
        # ``scheduler``：lr_scheduler，按 ``instance`` 配置绑定上述 optimizer。
        scheduler = get_scheduler(scheduler_config.instance, optimizer)
        # ``self.warmup_step``：int，线性 warmup 步数；缺省 0 表示不缩放学习率。
        self.warmup_step = getattr(scheduler_config, "warmup_step", 0)
        if self.warmup_step > 0:
            print('Warmup step is', self.warmup_step)
        return {
            # ``optimizer``：torch.optim.Optimizer，Lightning 执行反向更新的优化器实例。
            'optimizer': optimizer,
            # ``lr_scheduler``：dict，Lightning 调度器实例及其调度元数据。
            'lr_scheduler': {
                # ``lr_scheduler.scheduler``：lr_scheduler，绑定 ``optimizer`` 的学习率调度器实例。
                'scheduler': scheduler,
                # ``lr_scheduler.interval``：str，由 ``scheduler_config.params`` 展开的调度时间单位。
                # ``lr_scheduler.frequency``：int，由 ``scheduler_config.params`` 展开的调度调用频率。
                # ``lr_scheduler.monitor``：str，由 ``scheduler_config.params`` 展开的指标日志键。
                **scheduler_config.params,
            },
        }


    def optimizer_step2(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        """保留的优化器 OOM 处理实验；名称不是 Lightning 当前 ``optimizer_step`` 钩子，本训练入口不会调用。"""
        try:
            return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        except Exception as e:
            if isinstance(e, RuntimeError) and "out of memory" in str(e):
                print(f"\nOut of memory error occurred in optimizer_step. Skip.")
                # ``self.skip``：bool，停用优化器钩子的 OOM 标记；当前正式训练路径不读取。
                self.skip = True
                torch.cuda.empty_cache()
                super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
            else:
                raise e


# ``is_vscode``：bool，模块导入时的调试环境标记；VS Code 终端中 DataLoader batch size 强制为 40。
is_vscode = False
if os.environ.get("TERM_PROGRAM") == "vscode":
    # ``is_vscode``：bool，TERM_PROGRAM 精确为 vscode 时启用调试路径。
    is_vscode = True

if __name__ == '__main__':
    from pytorch_lightning.loggers import TensorBoardLogger  # only training script need this
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
        default='configs/train/train_pxm.yml')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--multi_node', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='GPU device id. Only for single GPU training.')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--logdir', type=str, default='lightning_logs_tasked')
    parser.add_argument('--profile', type=bool, default=False)
    parser.add_argument('--resume', type=str, default='')
    # ``args``：argparse.Namespace，保存配置路径、设备拓扑、日志目录、resume 等命令行叶子。
    args = parser.parse_args()
    # ``args.logdir`` 最终为 TensorBoard 根目录；若配置位于 ``configs/train/<subdirs>``，追加这些子目录保持实验分组。
    if is_vscode:
        # ``args.logdir``：path，在调试运行中把日志根移到 vscode 子目录，避免与正式实验混放。
        args.logdir = os.path.join('vscode', args.logdir)
    # ``dir_names``：list[str]，配置父目录按 ``/`` 切分后的路径段。
    dir_names = os.path.dirname(args.config).split('/')
    # ``is_train``：int，路径段 ``train`` 的位置；不存在会直接抛 ValueError。
    is_train = dir_names.index('train')
    # ``names``：list[str]，train 后的配置子目录层级，用于保持日志实验分组。
    names = dir_names[is_train+1:]
    # ``args.logdir``：path，把配置子目录逐层追加到命令行日志根。
    args.logdir = '/'.join([args.logdir] + names)

    # Load configs
    # ``config``：EasyDict，递归合并/解析规则由 ``utils.misc.make_config`` 定义；后续所有模型、数据和训练字段都从此对象读取。
    config = make_config(args.config)
    # ``config_name``：str，去掉最后一个扩展名的配置文件名；用作 TensorBoardLogger 的实验名称。
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    # data and model
    dm = DataModule(config)
    # ``in_dims``：dict[str, int]，把特征词表大小和口袋输入宽度同时传入 Lightning 模型包装。
    in_dims = dm.get_in_dims()
    # ``model``：ModelLightning，绑定模型、损失和优化过程；输入维度来自同一 DataModule 的特征词表。
    model = ModelLightning(config, args, **in_dims)

    # callbacks
    # ``checkpoint_callback``：每 ``ckpt_every_n_steps`` 保存一次且不淘汰旧步数 checkpoint；另维护 last.ckpt。
    checkpoint_callback = ModelCheckpoint(
        filename='{step}',
        monitor='val/loss',
        mode='min',
        # save_top_k=3,
        save_top_k=-1,
        every_n_train_steps=config.train.ckpt_every_n_steps,
        save_last=True,
        verbose=True,
        # every_n_epochs=config.train.ckpt_every_n_epochs,
    )
    # ``logger``：logger.log_dir 由 save_dir/name/version 三层组成，也是源码快照与训练配置的共同输出根目录。
    logger = TensorBoardLogger(
        save_dir=args.logdir,
        name=config_name,
        version=args.tag if args.tag else None,
    )

    if not args.multi_node:
        # ``devices``：int|list[int]；多 GPU 时取设备数量，单 GPU 时显式指定命令行 device 编号。
        devices = args.num_gpus if args.num_gpus > 1 else [args.device]
        # ``num_nodes``：int，单节点或非 multi_node 模式沿用命令行节点数。
        num_nodes = args.num_nodes
    else:
        # ``devices``：int，multi_node 模式每节点只暴露 1 个设备给当前训练进程配置。
        devices = 1
        # ``num_nodes``：int，multi_node 模式从环境变量 NUM_NODES 读取节点总数，缺省 1。
        num_nodes = int(os.environ.get("NUM_NODES", 1))
    # print(args.num_nodes, args.num_gpus, devices)
    trainer = pl.Trainer(
        devices=devices,
        num_nodes=num_nodes,
        max_epochs=1,
        max_steps=config.train.max_steps,
        callbacks=[checkpoint_callback],
        precision=config.train.precision,
        check_val_every_n_epoch=None,
        log_every_n_steps=config.train.val_check_interval,
        val_check_interval=config.train.val_check_interval,
        gradient_clip_val=getattr(config.train, 'gradient_clip_val', None),
        logger=logger,
        num_sanity_val_steps=0,
        profiler='simple' if args.profile else None,
        strategy='ddp',
        # strategy='ddp_find_unused_parameters_true',
        accelerator='gpu',
        # detect_anomaly=True,
        # detect_anomaly=True
        # limit_train_batches=1.0 if not args.profile else 100,
        # limit_val_batches=1.0 if not args.profile else 50,
    )
    
    # resume
    # ``log_dir``：str，本次 Lightning 版本目录；checkpoint 由 callback 在其子目录管理。
    log_dir = trainer.logger.log_dir
    if args.resume:
        # ``ckpt_path``：str，``--resume`` 被解释为当前 experiment name 目录下的版本名，而非任意 checkpoint 路径。
        ckpt_path = os.path.join(os.path.dirname(log_dir),
                        args.resume, 'checkpoints/last.ckpt')
        print('Resume from', ckpt_path)
        # ``config.resume``：path，把实际解析出的 last.ckpt 路径回写配置，随训练配置快照一起落盘。
        config['resume'] = ckpt_path
    else:
        # ``ckpt_path``：None，未请求续训时 Trainer.fit 从新初始化状态开始。
        ckpt_path = None
    
    # save source code (only for rank 0 if use multiple GPUs)
    if (not args.multi_node and ((args.num_gpus == 1) or (trainer.global_rank == 0))) or \
        (args.multi_node and trainer.global_rank == 0):
        # ``curr_dir``：仓库根目录与 ``<log_dir>/src``；只有 global rank 0 执行文件写入。
        curr_dir = '.' # os.path.dirname(os.path.realpath(__file__))
        # ``save_dir``：path，本次实验版本目录下的源码快照根。
        save_dir = os.path.join(trainer.logger.log_dir, "src")
        copy_py_files(curr_dir, save_dir, base=True)
        # ``config_dir``：``<log_dir>/train_config``，只保存本次实际解析后的一个 YAML 配置副本。
        config_dir = os.path.join(log_dir, 'train_config')
        os.makedirs(config_dir, exist_ok=True)
        save_config(config, os.path.join(config_dir, os.path.basename(args.config)))
        # shutil.copyfile(args.config, os.path.join(config_dir, os.path.basename(args.config)))

    trainer.fit(model, dm, ckpt_path=ckpt_path)
    print('Training finished!')
