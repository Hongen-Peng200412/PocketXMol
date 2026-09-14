"""把 56 通道裁块编码为 D1 的 6³ 特征或完整 U-Net 的 48³ 特征.

入口 DensityEncoder 复用 Pocket_Plus 卷积、三维旋转位置编码 RoPE 和解码门控, 单次执行, 不保存循环状态或写文件.
输入为 (B, 56, 48, 48, 48), B 为裁块数; D1 返回 (B, 256, 6, 6, 6), D2/D3/D4 返回 (B, 48, 48, 48, 48), 后三轴均为 ZYX.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import SDPBackend, sdpa_kernel

from models.density_blocks import Bottleneck, ShortConv, ShortConvAdd, Res2NetBlock, AttentionGate


class VolumeAttention(nn.Module):
    """对一个体素层施加八头自注意力与前馈残差, 保持 256 通道和空间尺寸.

    构造参数 backend 为 reference、sdpa 或 flash, 分别表示显式分数验收、PyTorch 自动选择内核、强制 Flash.
    flash 遇到 FP32 推理输入时仅在注意力内核入口把 q、k、v 转为 bf16, 输出恢复 value 原类型; 卷积与投影精度不变.
    前向输入与输出均为 (B, 256, D, H, W), B 为裁块数, D、H、W 对应数组 ZYX; 当前每头 192 维.
    三条数组索引轴分别生成旋转相位, 不使用世界 XYZ 或新增物理位置通道; DensityEncoder 连续组合四层.
    """
    def __init__(self, backend):
        """建立固定八头的投影、RoPE频率与前馈层; backend 的取值见类说明."""
        super().__init__()
        self.backend = backend
        self.q = nn.Linear(256,1536)
        self.k = nn.Linear(256,1536,bias=False)
        self.v = nn.Linear(256,1536,bias=False)
        self.back = nn.Linear(1536,256,bias=False)
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        self.w1 = nn.Linear(256,768,bias=False)
        self.w2 = nn.Linear(256,768,bias=False)
        self.w3 = nn.Linear(768,256,bias=False)
        self.register_buffer('inv_freq',1/(100**(torch.arange(0,64,2).float()/64)),persistent=False)

    def forward(self, feature):
        """沿同一裁块的全部 D*H*W 个体素做注意力, 保持各裁块相互独立."""
        batch,channels,depth,height,width = feature.shape
        vector = feature.flatten(2).transpose(1,2)  # [B, 256, D, H, W] -> [B, D*H*W, 256], 展平空间轴, X 索引变化最快.
        # 原实现按数组的三条轴依次编码并乘 1.5; 物理 XYZ 读出的几何另由 density_basis 定义.
        grid = torch.stack(torch.meshgrid(*(torch.arange(n,device=feature.device,dtype=torch.float32) for n in (depth,height,width)),indexing='ij'),dim=-1).reshape(-1,3)*1.5
        phase = (grid[...,None]*self.inv_freq).flatten(1).repeat_interleave(2,dim=-1)  # [D*H*W, 3, 32] -> [D*H*W, 192], 每条轴 32 个频率, 相邻两分量共用相位.
        cosine,sine = phase.cos()[None,:,None],phase.sin()[None,:,None]  # (1, D*H*W, 1, 192), 在批次和注意力头上广播.
        query = self.q(vector).reshape(batch,-1,8,192)
        key = self.k(vector).reshape(batch,-1,8,192)
        value = self.v(vector).reshape(batch,-1,8,192)
        query = query*cosine + torch.stack((-query[...,1::2],query[...,::2]),dim=-1).flatten(-2)*sine
        key = key*cosine + torch.stack((-key[...,1::2],key[...,::2]),dim=-1).flatten(-2)*sine
        query,key = query.to(value.dtype),key.to(value.dtype)  # RoPE 相位运算后与 value 对齐精度, 供同一注意力内核读取.
        if self.backend == 'reference':
            weights = (torch.einsum('blai,bkai->blka',query,key)/math.sqrt(192)).softmax(dim=-2)  # (B, L, K, 8), 每个查询体素沿 K 个键体素归一化; L=K=D*H*W.
            result = torch.einsum('blka,bkai->blai',weights,value)  # (B, L, 8, 192), 按权重对 K 个 value 求和.
        else:
            original_dtype = value.dtype
            if self.backend == 'flash' and value.dtype not in (torch.float16,torch.bfloat16):
                query,key,value = query.bfloat16(),key.bfloat16(),value.bfloat16()
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION) if self.backend == 'flash' else sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
                result = F.scaled_dot_product_attention(query.transpose(1,2),key.transpose(1,2),value.transpose(1,2),dropout_p=0,scale=1/math.sqrt(192)).transpose(1,2)
            result = result.to(original_dtype)  # (B, L, 8, 192), 恢复投影类型, FP32 推理不把后续线性层整体切为 bf16.
        result = self.norm1(vector+self.back(result.reshape(batch,-1,1536)))
        result = self.norm2(result+self.w3(F.silu(self.w1(result))*self.w2(result)))
        return result.transpose(1,2).reshape(batch,channels,depth,height,width)


# ================================================================================================
class DensityEncoder(nn.Module):
    """按 D1 或 D2/D3/D4 构造单次密度编码器, 不加载 Pocket_Plus 已训练参数.

    构造参数 config 的字段:
        - mode: str, D1 在三次下采样与瓶颈注意力后返回 6³×256; D2/D3/D4 包含第四次下采样和完整解码, 返回 48³×48.
        - checkpoint: bool, True 时在训练反向中重算重型块的激活以节省显存, 不跨优化器更新缓存特征.
        - attention_backend: str, reference、sdpa 或 flash, 传入四层 VolumeAttention.
    前向 voxel 为 (B, 56, 48, 48, 48), 不接收上一轮特征; 返回形状见 mode 定义.
    """
    def __init__(self, config):
        """按 config.mode 建立所需编码与解码层; 参数和输出形状见类说明."""
        super().__init__()
        self.mode = config['mode']
        if self.mode not in ('D1','D2','D3','D4'):
            raise ValueError(f'未知密度模式 {self.mode}, 仅支持 D1/D2/D3/D4.')
        self.use_checkpoint = config['checkpoint']
        self.input_projection = ShortConvAdd(56,64)
        self.stem = ShortConv(64,256)
        self.down = nn.ModuleList([Bottleneck(256,64,stride=2,affine=True) for _ in range(3 if self.mode=='D1' else 4)])
        self.attention = nn.Sequential(*[VolumeAttention(config['attention_backend']) for _ in range(4)])
        if self.mode != 'D1':
            self.decoder3 = nn.Sequential(*[Res2NetBlock(256,256,scale=3) for _ in range(4)])
            self.gates = nn.ModuleList([AttentionGate(256,256,128),AttentionGate(256,128,64),AttentionGate(256,64,64)])
            self.decoders = nn.ModuleList([nn.Sequential(*[Res2NetBlock(ch,ch,scale=4) for _ in range(4)]) for ch in (128,64,64)])
            self.output_convs = nn.ModuleList([nn.Conv3d(64,32,kernel_size=k,padding=k//2) for k in (3,5,7)])
            self.output = nn.Conv3d(96,48,kernel_size=3,padding=1)

    def run_block(self, module, *args):
        """执行一个无随机操作的网络块, 仅训练且需要梯度时按配置重算激活.

        module 是被调用的卷积、注意力或门控模块, args 是其原始张量参数; 返回该模块的原始输出.
        本辅助方法由多次下采样、瓶颈及多级解码共同调用, 不改变各模块的张量接口.
        """
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(module,*args,use_reentrant=False,preserve_rng_state=False)
        return module(*args)

    def forward(self, voxel):
        """把固定 48³ 裁块经过成熟输入投影和 U-Net, D1 在 6³ 瓶颈立即返回."""
        # 每次传入全零附加特征, 不接受或返回上一轮状态; 保留成熟输入层的可学习偏置.
        initial = self.input_projection(voxel,voxel.new_zeros((voxel.shape[0],64,*voxel.shape[2:])))
        feature = self.stem(initial)
        skips = [feature]  # 列表元素为 (B, 256, D, H, W), 按 48³、24³、12³、6³、3³ 顺序保存当前模式已计算层.
        for downsample in self.down:
            feature = self.run_block(downsample,feature)
            skips.append(feature)
        feature = self.run_block(self.attention,feature)
        if self.mode == 'D1':
            return feature
        feature = self.run_block(self.decoder3,F.interpolate(feature,size=skips[3].shape[2:],mode='trilinear',align_corners=True)+skips[3])  # [B, 256, 3, 3, 3] -> [B, 256, 6, 6, 6], 插值后加入同分辨率跳跃特征.
        for gate,decoder,skip in zip(self.gates,self.decoders,reversed(skips[:3])):
            feature = self.run_block(decoder,self.run_block(gate,feature,skip))
        return self.output(F.relu(torch.cat([conv(feature) for conv in self.output_convs],dim=1)))
