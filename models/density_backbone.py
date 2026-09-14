"""把56通道裁块编码为D1的6³特征或完整U-Net的48³特征。

DensityEncoder复用Pocket_Plus卷积、三维RoPE和解码门控；单次执行，无循环状态。
输入(B,56,48,48,48)，D1返回(B,256,6,6,6)，D4返回(B,48,48,48,48)。
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import SDPBackend, sdpa_kernel

from models.density_blocks import Bottleneck, ShortConv, ShortConvAdd, Res2NetBlock, AttentionGate


class VolumeAttention(nn.Module):
    """保持原四层瓶颈中每层的256输入、8头×192维和3D RoPE。

    reference显式构造分数；sdpa让PyTorch选择内核；flash强制Flash内核，不能静默回退。
    """
    def __init__(self, backend):
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
        batch,channels,depth,height,width = feature.shape
        vector = feature.flatten(2).transpose(1,2)
        # 原实现按数组的三条轴依次编码，并乘1.5；物理XYZ读出另由basis定义。
        grid = torch.stack(torch.meshgrid(*(torch.arange(n,device=feature.device,dtype=torch.float32) for n in (depth,height,width)),indexing='ij'),dim=-1).reshape(-1,3)*1.5
        phase = (grid[...,None]*self.inv_freq).flatten(1).repeat_interleave(2,dim=-1)
        cosine,sine = phase.cos()[None,:,None],phase.sin()[None,:,None]
        query = self.q(vector).reshape(batch,-1,8,192)
        key = self.k(vector).reshape(batch,-1,8,192)
        value = self.v(vector).reshape(batch,-1,8,192)
        query = query*cosine + torch.stack((-query[...,1::2],query[...,::2]),dim=-1).flatten(-2)*sine
        key = key*cosine + torch.stack((-key[...,1::2],key[...,::2]),dim=-1).flatten(-2)*sine
        query,key = query.to(value.dtype),key.to(value.dtype)
        if self.backend == 'reference':
            weights = (torch.einsum('blai,bkai->blka',query,key)/math.sqrt(192)).softmax(dim=-2)
            result = torch.einsum('blka,bkai->blai',weights,value)
        else:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION) if self.backend == 'flash' else sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
                result = F.scaled_dot_product_attention(query.transpose(1,2),key.transpose(1,2),value.transpose(1,2),dropout_p=0,scale=1/math.sqrt(192)).transpose(1,2)
        result = self.norm1(vector+self.back(result.reshape(batch,-1,1536)))
        result = self.norm2(result+self.w3(F.silu(self.w1(result))*self.w2(result)))
        return result.transpose(1,2).reshape(batch,channels,depth,height,width)


class DensityEncoder(nn.Module):
    """只建立当前模式使用的编码/解码层；D1不构建或执行第四次下采样和解码器。"""
    def __init__(self, config):
        super().__init__()
        self.mode = config['mode']
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
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(module,*args,use_reentrant=False,preserve_rng_state=False)
        return module(*args)

    def forward(self, voxel):
        # 只使用零初始化输入，不接受/返回上一轮特征；保留成熟输入层的可学习偏置。
        initial = self.input_projection(voxel,voxel.new_zeros((voxel.shape[0],64,*voxel.shape[2:])))
        feature = self.stem(initial)
        skips = [feature]
        for downsample in self.down:
            feature = self.run_block(downsample,feature)
            skips.append(feature)
        feature = self.run_block(self.attention,feature)
        if self.mode == 'D1':
            return feature
        feature = self.run_block(self.decoder3,F.interpolate(feature,size=skips[3].shape[2:],mode='trilinear',align_corners=True)+skips[3])
        for gate,decoder,skip in zip(self.gates,self.decoders,reversed(skips[:3])):
            feature = self.run_block(decoder,self.run_block(gate,feature,skip))
        return self.output(F.relu(torch.cat([conv(feature) for conv in self.output_convs],dim=1)))
