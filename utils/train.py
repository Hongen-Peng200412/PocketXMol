"""提供原训练入口共用的优化器、学习率调度器与批次迭代辅助对象.

训练参数接线先读 get_optimizer 和 get_scheduler: 前者返回绑定模型全部参数的原 Adam/AdamW, 后者返回原调度器实例. 两者不执行训练更新、调度 step 或写文件, 检查点与运行日志由 scripts/train_pl.py 负责.
"""

import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from .warmup import GradualWarmupScheduler, GradualWarmupConstantScheduler


#customize exp lr scheduler with min lr
class ExponentialLR_with_minLr(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, min_lr=1e-4, last_epoch=-1, verbose=False):
        self.gamma = gamma    
        self.min_lr = min_lr
        super(ExponentialLR_with_minLr, self).__init__(optimizer, gamma, last_epoch, verbose)    
        
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs
        return [max(group['lr'] * self.gamma, self.min_lr)
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
                for base_lr in self.base_lrs]


def repeat_data(data: Data, num_repeat) -> Batch:
    datas = [copy.deepcopy(data) for i in range(num_repeat)]
    return Batch.from_data_list(datas)


def repeat_batch(batch: Batch, num_repeat) -> Batch:
    datas = batch.to_data_list()
    new_data = []
    for i in range(num_repeat):
        new_data += copy.deepcopy(datas)
    return Batch.from_data_list(new_data)


def shuffled_cyclic_iterator(N, shuffle=True):
    if isinstance(N, int):
        elements = list(range(N))
    elif isinstance(N, tuple):
        elements = list(range(*N))
    
    while True:
        if shuffle:
            np.random.shuffle(elements)
        for i in elements:
            yield i

class InfIterator:
    def __init__(self, data, batch_size, shuffle):
        self.data = np.array(data)
        self.size_data = len(self.data)
        self.index = 0

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.data)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index + self.batch_size > self.size_data:
            value_0 = self.data[self.index:]
            self.reset()
            size_addition = self.batch_size - len(value_0)
            value_1 = self.data[:size_addition]
            self.index += size_addition
            value = np.concatenate([value_0, value_1], axis=0)
        else:
            value = self.data[self.index:self.index + self.batch_size]
            self.index += self.batch_size
        return value


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def get_optimizer(cfg, model):
    """沿原优化器类型构造全部 model.parameters() 的单一参数组.

    输入参数:
        - cfg: Mapping, type 选择 adam/adamw, lr 为初始学习率, weight_decay 为权重衰减, beta1/beta2 为一阶/二阶矩系数, 可选 eps 缺省保持 PyTorch 原值1e-8.
        - model: nn.Module, parameters() 给出同一参数组内的全部模型参数.
    返回原 torch.optim.Adam/AdamW 实例, 不恢复 checkpoint 中的状态, 不区分旧主干和新增核酸参数的学习率.
    """
    if cfg.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2, ),
            eps=getattr(cfg, "eps", 1e-8),
        )
    elif cfg.type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2, ),
            eps=getattr(cfg, "eps", 1e-8),
        )
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(cfg, optimizer):
    """按配置构造原学习率调度器, 不在本函数调用 step.

    输入参数:
        - cfg: Mapping, type 选择原调度类型; plateau 的 factor/patience/min_lr/cooldown 控制倍率、等待次数、学习率下限与冷却期, mode/threshold_mode/threshold/eps 控制改善判断及实际下降门槛.
        - optimizer: torch.optim.Optimizer, 返回调度器修改其 param_groups 中的 lr, 不新建或拆分参数组.
    返回 cfg.type 对应的原调度器实例. plateau 新接入字段缺失时保留 PyTorch 原默认值, 兼容旧配置; 其余调度入口不变, step 与 checkpoint 的先后次序由训练入口负责.
    """
    if cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr,
            cooldown=cfg.cooldown,
            mode=getattr(cfg, "mode", "min"),
            threshold_mode=getattr(cfg, "threshold_mode", "rel"),
            threshold=getattr(cfg, "threshold", 1e-4),
            eps=getattr(cfg, "eps", 1e-8),
        )
    elif cfg.type == 'warmup_plateau':
        # raise NotImplementedError('Not used anymore.')
        return GradualWarmupScheduler(
            optimizer,
            warmup_steps = cfg.warmup_steps,
            check_frequency=cfg.check_frequency,
            after_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=cfg.factor,
                patience=cfg.patience,
                min_lr=cfg.min_lr
            )
        )
    elif cfg.type == 'warmup_constant':
        return GradualWarmupConstantScheduler(
            optimizer,
            multiplier = cfg.multiplier,
            total_epoch = cfg.total_epoch,
        )
    elif cfg.type == 'expmin':
        return ExponentialLR_with_minLr(
            optimizer,
            gamma=cfg.factor,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == 'expmin_milestone':
        gamma = np.exp(np.log(cfg.factor) / cfg.milestone)
        return ExponentialLR_with_minLr(
            optimizer,
            gamma=gamma,
            min_lr=cfg.min_lr,
        )
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)

