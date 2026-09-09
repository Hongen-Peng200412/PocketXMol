"""
沿原 PocketXMol 训练框架组织数据、自动优化、验证与检查点.

先读 DataModule、ModelLightning 与 DockingCheckpoint, 再读命令行装配. adaligand 分支使用完整 occurrence 数据集、蛋白/核酸受体和 W&B; 旧 LMDB 分支保留原训练入口.

adaligand 的 --logdir 直接指定单个实验目录, 主要产物如下:
    - checkpoints/step=<更新步>.ckpt: Lightning 字典; state_dict 含 model.* 参数, optimizer_states 与 loops 保存优化器和更新进度, hyper_parameters.config/args 记录实际配置及 manifest_root 等资产来源.
    - checkpoints/last.ckpt: 最近一次完成验证的完整检查点, 原版本保护可追加 vN 后缀; callbacks 中 DockingCheckpoint 的 pocketxmol 字段保存原调度器、下降次数、停止原因、W&B run_id 和主进程随机状态.
    - src/ 与 train_config/<配置名>.yml: 首次运行的源码快照与实际配置; 续训另写 src_resume_<时间>/ 和 train_config_resume_<时间>/.
    - wandb/: W&B 自身的运行记录; online 模式上传 train/val 损失、学习率与梯度范数, 凭据只从运行环境读取.

旧 LMDB 分支的上述目录仍位于 TensorBoardLogger.log_dir, 并保留 TensorBoard 事件文件. 本文件不生成新科学数据、不改变 loss 或原 reduce_batch 行为.
"""

# Standard library imports
import argparse
import gc
import os
import random
from copy import deepcopy
from datetime import datetime
import shutil
import sys
from typing import Any, Callable, Optional, Union

# Third-party imports
import numpy as np
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
from docking.dataset import OccurrenceDataset
from models.loss import get_loss_func
from models.maskfill import PMAsymDenoiser
from utils.dataset import ForeverTaskDataset
from utils.misc import *
from utils.sample_noise import get_sample_noiser
from utils.train import GradualWarmupScheduler, get_optimizer, get_scheduler
from utils.transforms import Compose, FeaturizeMol, FeaturizePocket, get_transforms

torch.set_float32_matmul_precision('medium')


def copy_py_files(src_dir, dst_dir, base=False):
    """递归保存一次运行的 Python、Shell 和 notebook 源码快照.

    src_dir、dst_dir 是当前递归层的源目录与目标目录. base=True 时仅进入 scripts、models、notebooks、utils、process、evaluate、docking 这些一级目录; 后续层保留全部子目录.
    自动建立 dst_dir, 同名源文件由 shutil.copy 覆盖. YAML 配置由主程序单独保存, 本函数不复制权重、资产或 Git 元数据.
    """
    os.makedirs(dst_dir, exist_ok=True)
    for item in os.listdir(src_dir):
        item_path = os.path.join(src_dir, item)
        if os.path.isdir(item_path):
            if (not base) or item in ['scripts', 'models', 'notebooks', 'utils', 'process', 'evaluate', 'docking']:
                copy_py_files(item_path, os.path.join(dst_dir, item))
        elif item.endswith(('.py', '.sh', '.ipynb')):
            shutil.copy(item_path, dst_dir)


# ================================================================================================
class DataModule(pl.LightningDataModule):
    """按数据来源建立原特征变换链和 PyG 训练/验证加载器.

    构造参数:
        - config.data.dataset: Mapping; name=adaligand 时含 root、derived_root、manifest_root、pocket_mode、knn, 由 OccurrenceDataset 读取冻结实例清单和只读资产.
        - config.model.nucleic_branch: str, adaligand 模型的 RA/RB 核酸分支.
        - config.transforms: Mapping, 配体特征及 task 变换配置; 旧 LMDB 路径还可包含 featurizer_pocket 与 cut_peptide.
        - config.noise: Mapping, 原任务噪声器配置; adaligand 仅选择 dock.
        - config.train: Mapping, batch_size、num_workers、pin_memory、persistent_workers 决定加载器资源.

    批次字段:
        - node_type: int64, (N,), 拼接后N个配体原子的干净类别.
        - node_in: int64, (N,), 噪声器写入的带噪原子类别.
        - node_pos: float32, (N, 3), 配体干净局部XYZ坐标, 单位Å; 原点为该实例的给定中心或包络口袋均值.
        - pos_in: float32, (N, 3), 噪声器写入的带噪局部XYZ坐标, 单位Å, 与node_pos使用同一原点.
        - halfedge_type: int64, (H,), H条完全图半边的干净类别.
        - halfedge_in: int64, (H,), 噪声器写入的带噪半边类别.
        - halfedge_index: int64, (2, H), 每条半边两端的配体原子编号, 索引node_type第一维.
        - node_type_batch: int64, (N,), 每个配体原子所属的批内样本编号.
        - halfedge_type_batch: int64, (H,), 每条半边所属的批内样本编号.
        - pocket_atom_feature: (P, 25), P 个受体原子的原蛋白特征; adaligand 的核酸原子对应全零.
        - pocket_nucleic_feature: (P, 15), adaligand 标准 RNA/DNA 原子特征; 蛋白原子对应全零, 与 pocket_pos 第一维对齐.
        - pocket_is_nucleic: bool, (P,), True 标识核酸原子; 仅 adaligand 分支提供.
        - pocket_pos: float32, (P, 3), 受体局部XYZ坐标, 单位Å, 原点与对应配体一致.
        - pocket_pos_batch: int64, (P,), 每个受体原子所属的批内样本编号.
        - task: list[str], 长度为批内样本数 B; adaligand 全部为 dock.

    adaligand 的训练清单按 occurrence 均匀有放回抽样, 验证完整遍历 validation.jsonl; 中心验证使用已保存 C5 偏移, 包络验证使用 E. 数据集先完成受体编码与定位, 本类只接原 FeaturizeMol、任务变换与训练噪声器.
    """

    def __init__(self, config):
        """保存完整配置, 实际数据集在 Lightning 调用 setup 时建立."""
        super().__init__()
        self.config = config
        self.is_docking = getattr(config.data.dataset, 'name', '') == 'adaligand'

    def get_featurizers(self):
        """返回按坐标依赖排序的特征变换; adaligand 已定位口袋, 只需 FeaturizeMol."""
        featurizer = FeaturizeMol(self.config.transforms.featurizer)
        if not self.is_docking and 'featurizer_pocket' in self.config.transforms:
            return [FeaturizePocket(self.config.transforms.featurizer_pocket), featurizer]
        return [featurizer]

    def get_in_dims(self, featurizers=None):
        """返回模型输入宽度: num_node_types/num_edge_types 为配体词表大小, pocket_in_dim 为原蛋白特征宽度25.

        featurizers 是 get_featurizers 的有序列表; None 兼容主程序在 setup 之前查询模型维度. adaligand 的15维核酸特征由模型独立嵌入, 不改变 pocket_in_dim.
        """
        if featurizers is None:
            featurizers = self.get_featurizers()
        in_dims = {
            'num_node_types': featurizers[-1].num_node_types,
            'num_edge_types': featurizers[-1].num_edge_types,
        }
        if self.is_docking:
            in_dims['pocket_in_dim'] = 25
        elif len(featurizers) == 2:
            in_dims['pocket_in_dim'] = featurizers[0].feature_dim
        return in_dims

    def setup(self, stage=None):
        """构造单样本变换链及无限训练、有限验证加载器.

        stage 是 Lightning 生命周期参数, 当前两条来源都建立训练和验证对象. adaligand 使用单GPU独立实验, 每个 worker 的训练均匀有放回抽样与验证互斥分片均由 OccurrenceDataset 实现; task_db_weights 不参与其抽样.
        self.transforms 按配体特征、任务、噪声排列; 旧来源在之前增加口袋特征及可选肽裁剪. follow_batch 让 PyG 生成原子、半边和口袋原子的样本归属向量, exclude_keys 去掉只适用于单样本的原始字段.
        """
        featurizers = self.get_featurizers()
        in_dims = self.get_in_dims(featurizers)
        task_trans = get_transforms(self.config.transforms.task, mode='train', num_node_types=in_dims['num_node_types'])
        noiser = get_sample_noiser(self.config.noise, in_dims['num_node_types'], in_dims['num_edge_types'], mode='train')
        transform_list = featurizers + [task_trans, noiser]
        if not self.is_docking and 'cut_peptide' in self.config.transforms:
            transform_list = [get_transforms(self.config.transforms.cut_peptide)] + transform_list
        self.transforms = Compose(transform_list)
        # list[str], 每个名字生成对应的 <字段>_batch, 指明拼接后实体所属的样本.
        follow_batch = sum([getattr(t, 'follow_batch', []) for t in self.transforms.transforms], [])
        exclude_keys = sum([getattr(t, 'exclude_keys', []) for t in self.transforms.transforms], [])
        data_cfg = self.config.data
        train_cfg = self.config.train
        if self.is_docking:
            follow_batch = list(dict.fromkeys(follow_batch + ['pocket_pos']))
            protocol = 'C5' if data_cfg.dataset.pocket_mode == 'center' else 'E'
            train_set = OccurrenceDataset(data_cfg.dataset, 'train', self.transforms, self.config.model.nucleic_branch, protocol, True)
            val_set = OccurrenceDataset(data_cfg.dataset, 'validation', self.transforms, self.config.model.nucleic_branch, protocol, False)
            batch_size = train_cfg.batch_size
            val_workers = train_cfg.num_workers
        else:
            num_samplers_args = {
                'num_workers': train_cfg.num_workers,
                'global_rank': self.trainer.global_rank,
                'world_size': self.trainer.world_size,
            }
            train_set = ForeverTaskDataset(data_cfg.dataset, data_cfg.task_db_weights, 'train', transforms=self.transforms, shuffle=True, **num_samplers_args)
            divider = 4 if self.trainer.world_size > 100 else 1
            val_workers = train_cfg.num_workers // divider
            num_samplers_args['num_workers'] = val_workers
            val_set = ForeverTaskDataset(data_cfg.dataset, data_cfg.task_db_weights, 'val', transforms=self.transforms, shuffle=False, **num_samplers_args)
            batch_size = 40 if is_vscode else train_cfg.batch_size
        self.train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=train_cfg.num_workers, pin_memory=train_cfg.pin_memory, follow_batch=follow_batch, exclude_keys=exclude_keys, persistent_workers=train_cfg.persistent_workers)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=val_workers, pin_memory=train_cfg.pin_memory, follow_batch=follow_batch, exclude_keys=exclude_keys, persistent_workers=train_cfg.persistent_workers)

    def train_dataloader(self):
        """向 Lightning 提供无限训练实例流, 由优化器更新数或 Plateau 停止条件结束."""
        return self.train_loader

    def val_dataloader(self):
        """向 Lightning 提供一次完整验证实例遍历, 沿原 validation_step 计算 val/loss."""
        return self.val_loader


class ModelLightning(pl.LightningModule):
    """
    封装共享去噪网络、逐任务损失、优化器和训练/验证日志.

    构造参数:
        - config.model: ``PMAsymDenoiser`` 配置; ``name`` 必须为 ``pm_asym_denoiser`` 才会创建 ``self.model``.
        - config.model.pretrained: str|空字符串, 旧 LMDB 分支的预训练模型路径.
        - config.train.initial_checkpoint: str, adaligand 首次训练的官方模型路径; --resume 时跳过官方初始化.
        - config.loss: 损失配置; 当前 reduced 配置由 ``IndividualTasksLoss`` 分别统计 mixed 与各任务分量.
        - config.train: 优化器、调度器、warmup 与日志相关设置.
        - args.num_gpus: int, 当前节点使用的 GPU 数.
        - args.multi_node: bool, 是否跨多个计算节点训练.
        - num_node_types: int, 配体原子类别数; 传给原子 Embedding 与分类头.
        - num_edge_types: int, 配体半边类别数; 传给半边 Embedding 与分类头.
        - ``**kwargs``: 至少含 ``pocket_in_dim``, 传给 ``PMAsymDenoiser`` 的口袋线性嵌入层.

    形状符号: N_all 为批内配体原子数, H_all 为配体完全图半边数, P 为受体原子数, E_p 为口袋有向边数, K_n/K_e 为模型输出类别数.

    前向输入:
        - batch.node_in: LongTensor, 形状为 (N_all,), 带噪原子类别.
        - batch.pos_in: FloatTensor, 形状为 (N_all, 3), 带噪配体局部坐标, 单位 Å.
        - batch.halfedge_in: LongTensor, 形状为 (H_all,), 带噪半边类别.
        - batch.halfedge_index: LongTensor, 形状为 (2, H_all), 完全图半边端点.
        - batch.fixed_node: LongTensor|BoolTensor, 形状为 (N_all,), 原子类别条件掩码.
        - batch.fixed_pos: LongTensor|BoolTensor, 形状为 (N_all,), 坐标条件掩码.
        - batch.fixed_halfedge: LongTensor|BoolTensor, 形状为 (H_all,), 半边类别条件掩码.
        - batch.fixed_halfdist: LongTensor|BoolTensor, 形状为 (H_all,), 半边距离条件掩码.
        - batch.node_type_batch: LongTensor, 形状为 (N_all,), 逐原子图归属编号.
        - batch.pocket_atom_feature: FloatTensor, (P, 25), P 个受体原子的原蛋白输入特征; 核酸位置为0.
        - batch.pocket_nucleic_feature: FloatTensor, (P, 15), adaligand 核酸输入特征; 蛋白位置为0, 与 pocket_pos 第一维对齐.
        - batch.pocket_is_nucleic: bool, (P,), adaligand 标准 RNA/DNA 原子标记, True 为核酸.
        - batch.pocket_pos: FloatTensor, 形状为 (P, 3), 口袋局部坐标, 单位 Å.
        - batch.pocket_knn_edge_index: LongTensor, 形状为 (2, E_p), 口袋 kNN 有向边端点.
        - batch.pocket_pos_batch: LongTensor, 形状为 (P,), 逐口袋原子图归属编号.
        - batch.is_peptide: LongTensor, 形状为 (N_all,), 小分子任务为全 0; 仅配置请求时读取.

    前向输出:
        - pred_node: (N_all, K_n), 每个配体原子的真值类别 logits.
        - pred_pos: (N_all, 3), 每个配体原子的去噪坐标, 坐标原点与 ``pos_in`` 一致, 单位 Å.
        - pred_halfedge: (H_all, K_e), 每条无向半边的真值类别 logits.
        - confidence_node: (N_all, 1), 可选原子类别置信度原始分数; 配置 ``add_output`` 含 ``confidence`` 时存在.
        - confidence_pos: (N_all, 1), 可选原子坐标置信度原始分数; 与原子第一维对齐.
        - confidence_halfedge: (H_all, 1), 可选半边类别置信度原始分数; 与半边第一维对齐.

    损失输出:
        - <task>/node: 当前任务未固定原子类别的标量损失.
        - <task>/pos: 当前任务未固定坐标的标量损失.
        - <task>/edge: 当前任务未固定半边类别的标量损失.
        - <task>/fixed_node: 当前任务固定原子类别的标量损失.
        - <task>/fixed_pos: 当前任务固定坐标的标量损失.
        - <task>/fixed_edge: 当前任务固定半边类别的标量损失.
        - <task>/dist: 当前任务同域待恢复距离的标量损失.
        - <task>/fixed_dist: 当前任务固定距离的标量损失.
        - <task>/dih: 当前任务二面角周期标量损失.
        - <task>/total: 当前任务各叶按 ``config.loss.weights`` 汇总的标量.
        - mixed/total: 混合批次各恢复叶与可选 confidence/口袋距离项的反向传播标量.
    """
    def __init__(self, config, args, num_node_types, num_edge_types, **kwargs):
        """构造原主干及 loss, 新实验只加载官方 model 权重, 自己续训交给 Lightning 恢复.

        config.train.initial_checkpoint 是 adaligand 首次训练的官方检查点路径; args.resume 非空时跳过该路径. num_node_types/num_edge_types 为配体类别数, kwargs.pocket_in_dim 保持蛋白25维.
        官方参数去掉 model. 前缀后严格匹配旧主干, 只允许新 nucleic_embedder/nucleic_encoder 参数缺失. RB 随后从已加载的 pocket_encoder 复制核酸编码器初值; 恢复自己检查点时不执行这一步.
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.num_gpus = args.num_gpus
        self.multi_node = args.multi_node
        self.sync_dist = self.num_gpus > 1 or self.multi_node
        self.is_docking = getattr(config.data.dataset, 'name', '') == 'adaligand'
        if self.config.model.name == 'pm_asym_denoiser':
            self.model = PMAsymDenoiser(config=self.config.model, num_node_types=num_node_types, num_edge_types=num_edge_types, **kwargs)

        initial_checkpoint = self.config.train.initial_checkpoint if self.is_docking else getattr(self.config.model, 'pretrained', '')
        if initial_checkpoint and not (self.is_docking and args.resume):
            # dict[str, Tensor], 只取主网络参数; 不载入官方 optimizer、scheduler 或 best 分数.
            ckpt = torch.load(initial_checkpoint, map_location='cpu', weights_only=False)
            model_state = {key[6:]: value for key, value in ckpt['state_dict'].items() if key.startswith('model.')}
            if self.is_docking:
                incompatible = self.model.load_state_dict(model_state, strict=False)
                missing = [key for key in incompatible.missing_keys if not key.startswith(('nucleic_embedder.', 'nucleic_encoder.'))]
                if missing or incompatible.unexpected_keys:
                    raise RuntimeError(f'官方权重与原模型不匹配: missing={missing}, unexpected={incompatible.unexpected_keys}')
                if self.config.model.nucleic_branch == 'RB':
                    self.model.nucleic_encoder.load_state_dict(self.model.pocket_encoder.state_dict())
            else:
                self.model.load_state_dict(model_state)
            print('Load pretrained model from', initial_checkpoint)
        self.loss_func = get_loss_func(self.config.loss)

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
        """沿原工厂创建 optimizer 与 scheduler, 保留 Lightning 自动反向传播和梯度累积.

        config.train.optimizer 决定全部 model.parameters() 的单一参数组. scheduler.instance 决定原调度器参数, warmup_step 仍由 on_before_optimizer_step 读取.
        adaligand 返回 optimizer, 并将原 ReduceLROnPlateau 保存为 docking_scheduler; 仅 DockingCheckpoint 在完整验证结束后调用一次 step 并先更新后保存. 旧来源返回原 Lightning lr_scheduler 元数据, 调度行为不变.
        """
        optimizer = get_optimizer(self.config.train.optimizer, self.model)
        scheduler_config = self.config.train.scheduler
        scheduler = get_scheduler(scheduler_config.instance, optimizer)
        self.warmup_step = getattr(scheduler_config, 'warmup_step', 0)
        if self.warmup_step > 0:
            print('Warmup step is', self.warmup_step)
        if self.is_docking:
            self.docking_scheduler = scheduler
            return optimizer
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, **scheduler_config.params},
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


class DockingCheckpoint(ModelCheckpoint):
    """按优化器更新数触发原验证, 调度一次并保存全部验证检查点.

    构造参数:
        - dirpath: str, 单个实验的 checkpoints 目录; 原 ModelCheckpoint 保存 step=<更新步>.ckpt 和 last.ckpt.
        - run_id: str, W&B 运行标识, 如 h3k82abc; 新训练由 W&B 生成, 续训从检查点取回.

    callbacks[state_key].pocketxmol 字段, state_key 是 Lightning 根据回调类名与监控配置生成的字符串:
        - scheduler: dict, 原 ReduceLROnPlateau.state_dict(), 下列状态与配置一并恢复.
            - factor: float, 实际下降时的学习率乘数.
            - min_lrs: list[float], 与 optimizer.param_groups 对齐的学习率下限.
            - patience: int, 允许连续未达到改善要求的验证次数; 超过该次数才下降.
            - cooldown: int, 配置的冷却验证次数, 正式值0.
            - cooldown_counter: int, 当前剩余冷却验证次数.
            - mode: str, 指标优化方向, 正式值min表示最小化val/loss.
            - threshold_mode: str, 改善阈值模式, 正式值rel表示相对改善.
            - threshold: float, 改善阈值, 正式值0.01.
            - eps: float, 新旧学习率差必须超过此值才实际修改.
            - last_epoch: int, 已调用 scheduler.step 的验证次数, 与数据 epoch 不同.
            - _last_lr: list[float], 最近一次调度后各参数组的学习率.
            - mode_worse: float, 未见首个验证值时的最差基准, min 模式为正无穷.
            - best: float, 最近达到相对改善要求的验证基准, 与原始 loss 的最低值可能不同.
            - num_bad_epochs: int, 当前连续未达到改善要求的验证次数.
        - decline_count: int, 各参数组学习率确实减小的累计次数; 第三次减小时停止.
        - last_validation_step: int, 最近已处理的优化器更新编号, 用于防止累积批次或续训重复验证.
        - stop_reason: str|None, plateau 表示第三次下降, max_steps 表示到达更新上限, None 表示仍可续训.
        - run_id: str, 上述 W&B 运行标识, 不含账号凭据.
        - rng: dict, 训练主进程的随机生成器状态, 不含 DataLoader worker 的私有 Generator.
            - python: tuple(version, internal_state, gauss_next), int版本号、内部整数状态元组、float|None高斯缓存; 由random.setstate整体恢复.
            - numpy: tuple(algorithm, keys, position, has_gauss, cached_gaussian), str算法名、uint32状态数组、int当前索引、int是否持有高斯缓存、float缓存值; 由np.random.set_state整体恢复.
            - torch: CPU uint8 向量, torch.get_rng_state() 的主进程 CPU 随机状态.
            - cuda: uint8 Tensor|None, 当前训练设备的 CUDA 随机状态; CPU 测试为 None, 不初始化其它可见GPU.

    callbacks[state_key] 还保留原 ModelCheckpoint 选择字段:
        - dirpath: str, 实际checkpoints目录; 普通resume必须沿用此目录.
        - best_model_path: str, 原始val/loss最低者路径.
        - best_model_score: scalar Tensor|None, 最低原始val/loss; 尚未验证为None.
        - best_k_models: dict[str,scalar Tensor], 每个保存路径及对应val/loss; save_top_k=-1保留全部.
        - kth_best_model_path: str, 原ModelCheckpoint记录的当前最差保留路径; 不触发本项目删除.
        - kth_value: scalar Tensor, 上述最差保留检查点分数.
        - last_model_path: str, 最近保存的完整检查点路径, 原版本保护可以增加vN后缀.
        - monitor: str, 监控字段val/loss.

    根字段 lr_schedulers 为 []: 本分支未向Lightning注册自动调度器, 调度状态实际在 callbacks[state_key].pocketxmol.scheduler. optimizer_states、global_step、loops 和完整配置仍由Lightning保存. RNG只覆盖训练主进程; 多worker预取队列不保存, 恢复后仍按同一occurrence均匀有放回分布抽样.
    """

    def __init__(self, dirpath, run_id):
        """建立只在验证结束保存的原 ModelCheckpoint, 不启用检查点淘汰."""
        super().__init__(dirpath=dirpath, filename='{step}', monitor='val/loss', mode='min', save_top_k=-1, save_last=True, every_n_train_steps=0, every_n_epochs=1, save_on_train_epoch_end=False, verbose=True)
        self.run_id = run_id
        self.scheduler_state = None
        self.decline_count = 0
        self.last_validation_step = 0
        self.stop_reason = None
        self.rng_state = None
        self.cuda_device = None

    def on_fit_start(self, trainer, pl_module):
        """在优化器构造后恢复原调度器; 原生 optimizer/loops 随后仍由 Lightning 恢复."""
        if self.scheduler_state is not None:
            pl_module.docking_scheduler.load_state_dict(self.scheduler_state)
        else:
            self.scheduler_state = pl_module.docking_scheduler.state_dict()

    def on_train_start(self, trainer, pl_module):
        """训练循环恢复后重建验证门控并恢复主进程随机状态; 已停止的检查点不再更新参数."""
        trainer.val_check_batch = float('inf')
        self.cuda_device = pl_module.device if pl_module.device.type == 'cuda' else None
        if self.rng_state is not None:
            random.setstate(self.rng_state['python'])
            np.random.set_state(self.rng_state['numpy'])
            torch.set_rng_state(self.rng_state['torch'].cpu())
            if self.cuda_device is not None and self.rng_state['cuda'] is not None:
                torch.cuda.set_rng_state(self.rng_state['cuda'].cpu(), self.cuda_device)
        if self.stop_reason is not None:
            trainer.should_stop = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """仅在新的完整优化器更新到达配置间隔时开放本次原验证循环.

        trainer.global_step 在梯度累积期间保持不变. last_validation_step 排除同一步的后续 microbatch; val_check_batch=1 让原 Lightning 验证条件通过, inf 则等待下一次更新. 因此续训调整梯度累积数也不改变每800次更新验证一次的口径.
        """
        step = trainer.global_step
        due = step > self.last_validation_step and step % pl_module.config.train.val_check_interval == 0
        trainer.val_check_batch = 1 if due else float('inf')
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_end(self, trainer, pl_module):
        """使用完整原 val/loss 调度一次, 先记录下降/停止状态再调用原检查点保存.

        Lightning 原自动 Plateau 调度发生在 on_validation_end 之后. 本分支不向 Lightning 注册 scheduler, 从而避免保存旧调度状态或重复 step. 原 ModelCheckpoint 按原始 loss 的最小值选 best, 与 Plateau 的相对改善阈值无关.
        """
        step = trainer.global_step
        if trainer.sanity_checking or trainer.state.fn != 'fit' or step <= self.last_validation_step:
            return
        # list[float], 对齐 optimizer.param_groups; 只有真实学习率下降才计数, eps 导致的不变不计入.
        old_lrs = [group['lr'] for group in trainer.optimizers[0].param_groups]
        pl_module.docking_scheduler.step(trainer.callback_metrics['val/loss'])
        if any(group['lr'] < before for group, before in zip(trainer.optimizers[0].param_groups, old_lrs)):
            self.decline_count += 1
        self.last_validation_step = step
        self.scheduler_state = pl_module.docking_scheduler.state_dict()
        if self.decline_count >= 3:
            self.stop_reason = 'plateau'
        elif step >= trainer.max_steps:
            self.stop_reason = 'max_steps'
        if self.stop_reason is not None:
            trainer.should_stop = True
        super().on_validation_end(trainer, pl_module)

    def state_dict(self):
        """返回原检查点选择状态及类文档定义的 pocketxmol 调度、停止、W&B 与主进程随机状态."""
        state = super().state_dict()
        state['pocketxmol'] = {
            'scheduler': self.scheduler_state,
            'decline_count': self.decline_count,
            'last_validation_step': self.last_validation_step,
            'stop_reason': self.stop_reason,
            'run_id': self.run_id,
            'rng': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state(self.cuda_device) if self.cuda_device is not None else None,
            },
        }
        return state

    def load_state_dict(self, state_dict):
        """恢复自己检查点的完整选择与训练控制状态; 实验目录须沿用原路径, 以保留原 best/last 记录."""
        super().load_state_dict(state_dict)
        state = state_dict['pocketxmol']
        self.scheduler_state = state['scheduler']
        self.decline_count = state['decline_count']
        self.last_validation_step = state['last_validation_step']
        self.stop_reason = state['stop_reason']
        self.run_id = state['run_id']
        self.rng_state = state['rng']


# 旧 LMDB 调试入口保持 VS Code 终端的原行为; adaligand 始终使用显式批量与实验路径.
is_vscode = os.environ.get('TERM_PROGRAM') == 'vscode'

if __name__ == '__main__':
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', nargs='?', help='训练 YAML; 与原 --config 二选一.')
    parser.add_argument('--config', type=str, help='兼容原入口的训练 YAML 参数.')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--multi_node', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='单GPU训练的可见设备编号.')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--logdir', type=str, default='lightning_logs_tasked')
    parser.add_argument('--profile', type=bool, default=False)
    parser.add_argument('--resume', type=str, default='', help='adaligand 接收完整 ckpt 路径; 旧来源保留 TensorBoard 版本目录名.')
    args = parser.parse_args()
    if args.config_path and args.config:
        parser.error('位置参数与 --config 不能同时指定.')
    args.config = args.config_path or args.config or 'configs/train/train_pxm.yml'
    config = make_config(args.config)
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    is_docking = getattr(config.data.dataset, 'name', '') == 'adaligand'
    seed_all(config.train.seed)

    if is_docking:
        import wandb

        # 单卡各跑一个实验, 名义全局批量只由单卡 batch 与累积相乘; 原 reduce_batch 例外不改.
        if args.num_gpus != 1 or args.num_nodes != 1 or args.multi_node:
            parser.error('adaligand 当前阶段使用单GPU独立实验, 每张授权卡分别启动一个配置.')
        if config.train.batch_size * config.train.accumulate_grad_batches != config.train.global_batch_size:
            parser.error('batch_size × accumulate_grad_batches 必须等于 global_batch_size.')
        log_dir = os.path.abspath(args.logdir)
        checkpoint_callback = DockingCheckpoint(os.path.join(log_dir, 'checkpoints'), '')
        ckpt_path = os.path.abspath(args.resume) if args.resume else None
        if ckpt_path:
            # 预读自家callback状态和hyper_parameters.config以核对恢复条件; 此处不向模型装载权重, 实际恢复交给Trainer.
            resume_checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            callback_state = resume_checkpoint['callbacks'][checkpoint_callback.state_key]
            run_state = callback_state['pocketxmol']
            if run_state['stop_reason'] is not None:
                parser.error(f"该检查点已按 {run_state['stop_reason']} 结束, 不能作为普通续训再次更新参数.")
            if os.path.realpath(callback_state['dirpath']) != os.path.realpath(checkpoint_callback.dirpath):
                parser.error('续训 --logdir 必须使用原实验目录, 以保持原 best/last 和 W&B 运行记录.')
            # 两份科学配置必须相同; 仅移除恢复路径、单卡批量/累积、加载资源、日志频率及W&B在线/离线偏好.
            saved_config = deepcopy(resume_checkpoint['hyper_parameters']['config'])
            requested_config = deepcopy(config)
            for compared_config in (saved_config, requested_config):
                if 'resume' in compared_config:
                    compared_config.pop('resume')
                for resource_key in ('batch_size', 'accumulate_grad_batches', 'num_workers', 'pin_memory', 'persistent_workers', 'log_every_n_steps'):
                    compared_config.train.pop(resource_key, None)
                compared_config.train.wandb.pop('mode', None)
            if saved_config != requested_config:
                parser.error('续训科学配置与检查点不同; 普通 resume 不能改变模型、口袋、噪声、数据清单或优化目标.')
            checkpoint_callback.run_id = run_state['run_id']
            config['resume'] = ckpt_path
            del resume_checkpoint
        else:
            if os.path.isdir(checkpoint_callback.dirpath) and os.listdir(checkpoint_callback.dirpath):
                parser.error('该实验已有检查点; 请用 --resume 续训或指定新的 --logdir.')
            checkpoint_callback.run_id = wandb.util.generate_id()
        os.makedirs(log_dir, exist_ok=True)
        # W&B 自身写运行目录和日志; mode 明确传递, 不把 API key 写入配置或打印.
        logger = WandbLogger(name=config.train.wandb.name, save_dir=log_dir, entity=config.train.wandb.entity, project=config.train.wandb.project, id=checkpoint_callback.run_id, mode=config.train.wandb.mode, offline=config.train.wandb.mode == 'offline', log_model=False)
        logger.experiment
    else:
        if is_vscode:
            args.logdir = os.path.join('vscode', args.logdir)
        dir_names = os.path.dirname(args.config).replace('\\', '/').split('/')
        names = dir_names[dir_names.index('train') + 1:]
        args.logdir = os.path.join(args.logdir, *names)
        checkpoint_callback = ModelCheckpoint(filename='{step}', monitor='val/loss', mode='min', save_top_k=-1, every_n_train_steps=config.train.ckpt_every_n_steps, save_last=True, verbose=True)
        logger = TensorBoardLogger(save_dir=args.logdir, name=config_name, version=args.tag if args.tag else None)
        log_dir = logger.log_dir
        ckpt_path = os.path.join(os.path.dirname(log_dir), args.resume, 'checkpoints/last.ckpt') if args.resume else None
        if ckpt_path:
            config['resume'] = ckpt_path

    dm = DataModule(config)
    model = ModelLightning(config, args, **dm.get_in_dims())
    if not args.multi_node:
        devices = args.num_gpus if args.num_gpus > 1 else [args.device]
        num_nodes = args.num_nodes
    else:
        devices = 1
        num_nodes = int(os.environ.get('NUM_NODES', 1))
    trainer = pl.Trainer(
        devices=devices,
        num_nodes=num_nodes,
        max_epochs=-1 if is_docking else 1,
        max_steps=config.train.max_steps,
        callbacks=[checkpoint_callback],
        precision=config.train.precision,
        check_val_every_n_epoch=None,
        log_every_n_steps=config.train.log_every_n_steps if is_docking else config.train.val_check_interval,
        val_check_interval=config.train.val_check_interval,
        accumulate_grad_batches=config.train.accumulate_grad_batches if is_docking else 1,
        gradient_clip_val=getattr(config.train, 'gradient_clip_val', None),
        logger=logger,
        num_sanity_val_steps=0,
        profiler='simple' if args.profile else None,
        strategy='auto' if is_docking else 'ddp',
        accelerator='gpu',
    )
    if trainer.global_rank == 0:
        # 首次快照保持 src/train_config 原目录; 续训新增带时间的目录, 不覆盖原实验来源.
        suffix = '_resume_' + datetime.now().strftime('%Y%m%d-%H%M%S') if is_docking and ckpt_path else ''
        copy_py_files('.', os.path.join(log_dir, 'src' + suffix), base=True)
        config_dir = os.path.join(log_dir, 'train_config' + suffix)
        os.makedirs(config_dir, exist_ok=True)
        save_config(config, os.path.join(config_dir, os.path.basename(args.config)))
    fit_args = {'weights_only': False} if is_docking else {}
    trainer.fit(model, dm, ckpt_path=ckpt_path, **fit_args)
    if is_docking:
        print(f'Training result: updates={trainer.global_step}, stop_reason={checkpoint_callback.stop_reason}, best={checkpoint_callback.best_model_path}, val/loss={checkpoint_callback.best_model_score}', flush=True)
    print('Training finished!')
