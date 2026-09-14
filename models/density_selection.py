"""从密度和冻结语言表征预测 D3 配体区域, 选择固定数量的密度体素.

入口 DensitySelection 只读取完整 U-Net 的 48 维特征与 768 维 SMI-TED 向量, 不读取坐标或标签.
返回全 48³ 裁块的两个分类 logits 和预测概率最高的 4096 个 ZYX 展平索引.
density_segmentation_loss 单独用当前实例的 ligand-area 标签计算辅助监督; 不参与 top-K 选择, 不写文件.
"""

import torch
from torch import nn
from torch.nn import functional as F


def density_segmentation_loss(logits, target):
    """计算固定的 D3 全块 focal 与前景 Dice-Tversky 辅助损失.

    输入参数:
        - logits: float Tensor, (B, 2, Z, Y, X), B 个裁块的背景/当前配体区域 logits, 正式 Z=Y=X=48.
        - target: int64 或 bool Tensor, (B, Z, Y, X), 当前 occurrence 的伪 ligand-area 标签, 0 是背景、1 是前景; 不聚合其他配体标签.
    返回字段:
        - focal: 标量 Tensor, 两类权重均为 0.5、gamma=2、概率裁至 [1e-6, 1-1e-6], 对批次全部体素平均.
        - dice: 标量 Tensor, 前景 TP/FP/FN 在批次和空间维共同求和, 两误差系数均为 0.5, smooth=1.
        - weighted: 标量 Tensor, 0.1*(0.7*focal+0.3*dice), 仅由训练入口加至实际反传目标; 原 dock 的 val/loss 保持独立.
    使用 FP32 归约, float64 验收输入保留 float64; 不增加 MSE、有效体素掩码或基于标签的采样.
    数值定义对应 Pocket_Plus AdaptiveClassificationCompositeLoss 的二 logit 多分类分支.
    """
    with torch.autocast(device_type=logits.device.type, enabled=False):
        prediction = logits if logits.dtype == torch.float64 else logits.float()
        labels = target.long()  # int64, (B, Z, Y, X), 索引 logits 的两个类别通道.
        log_probability = F.log_softmax(prediction, dim=1)
        probability = log_probability.exp()  # (B, 2, Z, Y, X), 两类概率逐体素相加为 1.
        true_probability = probability.gather(1, labels[:, None]).squeeze(1).clamp(1e-6, 1-1e-6)
        cross_entropy = F.nll_loss(log_probability, labels, reduction='none')
        focal = (0.5 * (1-true_probability).square() * cross_entropy).mean()
        foreground = probability[:, 1]  # (B, Z, Y, X), 仅前景通道参加 Dice-Tversky.
        foreground_target = labels.to(prediction.dtype)
        true_positive = (foreground * foreground_target).sum()
        false_positive = (foreground * (1-foreground_target)).sum()
        false_negative = ((1-foreground) * foreground_target).sum()
        dice = 1 - (true_positive+1) / (true_positive+0.5*false_positive+0.5*false_negative+1)
        return dict(focal=focal, dice=dice, weighted=0.1*(0.7*focal+0.3*dice))


# ================================================================================================
class DensitySelection(nn.Module):
    """以密度 48 维和冻结 SMI-TED 768 维预测背景/配体区域, 不接受 XYZ 或 GT 标签.

    第一层等价于拼接后的 816→64 线性层, 拆成两次投影并相加避免复制 768 维语言向量.
    只在密度投影保留偏置; 隐藏层 ReLU, 最后 64→2 输出 logits. 参数随机初始化并随密度主体训练.
    前向输入输出见 forward, 训练与推理始终用预测 top4096, 不将概率作为体素特征乘数.
    """

    def __init__(self):
        """构造固定 48+768→64→2 概率头, 宽度由已批准 D3 契约决定."""
        super().__init__()
        self.density_projection = nn.Linear(48, 64)
        self.language_projection = nn.Linear(768, 64, bias=False)
        self.output = nn.Linear(64, 2)

    def forward(self, feature, language):
        """返回两个分类 logits 与每个实例的预测 top4096 体素索引.

        输入参数:
            - feature: (B, 48, 48, 48, 48), B 个实例的完整 U-Net 特征, 后三轴为 ZYX.
            - language: float32, (B, 768), 与 feature 首维逐实例对应的冻结 SMI-TED 向量.
        返回值:
            - logits: (B, 2, 48, 48, 48), 第 0 类背景、第 1 类当前配体区域, 供独立全块辅助监督.
            - indices: int64, (B, 4096), 预测前景概率降序选出的源裁块 ZYX 展平索引, X 变化最快; 并列时取较小展平索引, 如相同概率下 0 先于 1.
        不对离散排序求导; logits 的辅助监督训练概率头, 选中体素特征仍接收去噪损失梯度.
        """
        batch = feature.shape[0]
        voxels = feature.flatten(2).transpose(1, 2)  # (B, 48³, 48), 按 ZYX 展平, 与返回 indices 对齐.
        hidden = self.density_projection(voxels) + self.language_projection(language.detach())[:, None]  # (B, 48³, 64), 一个语言投影在本实例全部体素广播, 不混合实例.
        logits = self.output(F.relu(hidden)).transpose(1, 2).reshape(batch, 2, 48, 48, 48)  # [B, 48³, 2] -> [B, 2, 48, 48, 48], 分类通道移到空间轴之前.
        # FP32 softmax 避免 bf16 概率舍入制造不必要并列; stable 保留原 ZYX 索引的升序.
        probability = logits.float().softmax(dim=1)[:, 1].flatten(1)  # float32, (B, 48³), 仅预测前景概率用于选择, 不附加坐标条件.
        indices = torch.argsort(probability, dim=1, descending=True, stable=True)[:, :4096]
        return logits, indices
