"""从编码密度向当前配体原子产生320维残差，保持原坐标更新器。

DensityReadout的D1读取全部216个粗特征；D4按当前坐标读取home体素±3邻域。
几何以裁块角点和三条源XYZ体素基向量表示，原数组始终按ZYX存放。
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def density_attention(query,key,value,query_pos,key_pos,beta,distance_bias,backend):
    """计算4头读出；输入(B,heads,N,64)，位置(B,N,3)，输出同query形状。

    分数=qk/8-softplus(beta)*||(x-p)/10Å||²；增广实现省去仅依赖query的常数。
    增广Q/K/V补至72维并显式scale=1/8，输出只取原64个value分量。
    """
    if backend == 'reference':
        scores = query @ key.transpose(-1,-2) / 8
        if distance_bias:
            squared = (query_pos[:,:,None]-key_pos[:,None]).square().sum(-1)/100
            scores = scores - F.softplus(beta)[None,:,None,None]*squared[:,None]
        return scores.softmax(-1) @ value
    output_dim = value.shape[-1]
    if distance_bias:
        coefficient = F.softplus(beta)[None,:,None,None]
        query_position = query_pos[:,None].expand(-1,query.shape[1],-1,-1)
        key_position = key_pos[:,None].expand(-1,key.shape[1],-1,-1)
        extra_query = torch.cat((query_position*coefficient*0.16,-coefficient.expand(query.shape[0],-1,query.shape[2],-1)*0.08),dim=-1)
        extra_key = torch.cat((key_position,key_position.square().sum(-1,keepdim=True)),dim=-1)
        query = F.pad(torch.cat((query,extra_query.to(query.dtype)),dim=-1),(0,4))
        key = F.pad(torch.cat((key,extra_key.to(key.dtype)),dim=-1),(0,4))
        value = F.pad(value,(0,8))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION) if backend == 'flash' else sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
        result = F.scaled_dot_product_attention(query,key,value,dropout_p=0,scale=1/8)
    return result[...,:output_dim]


class DensityReadout(nn.Module):
    """单个去噪块的独立4头密度读出；alpha可学习，初始0.1，额外门控恒为1。"""
    def __init__(self,node_dim,config):
        super().__init__()
        self.mode = config['mode']
        self.backend = config['attention_backend']
        self.distance_bias = config['distance_bias']
        feature_dim = 256 if self.mode=='D1' else 48
        self.query = nn.Linear(node_dim,256)
        self.key = nn.Linear(feature_dim,256)
        self.value = nn.Linear(feature_dim,256)
        self.output = nn.Linear(256,node_dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.full((4,),math.log(math.expm1(1.0))))
        axis = torch.arange(6 if self.mode=='D1' else 48)
        grid = torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),dim=-1).reshape(-1,3)[:,[2,1,0]]
        self.register_buffer('voxel_xyz',grid*(8 if self.mode=='D1' else 1)+0.5,persistent=False)
        offsets = torch.arange(-3,4)
        self.register_buffer('offsets',torch.stack(torch.meshgrid(offsets,offsets,offsets,indexing='ij'),dim=-1).reshape(-1,3),persistent=False)

    def forward(self,h_node,pos_node,batch_node,feature,origin,basis):
        result = torch.zeros_like(h_node)
        # 按分子切分，保证任何读出都不混用其它分子的密度或原子。
        for molecule in range(feature.shape[0]):
            atom_ids = torch.nonzero(batch_node==molecule,as_tuple=False).flatten()
            if atom_ids.numel()==0:
                continue
            atom_pos = pos_node[atom_ids]
            voxel_features = feature[molecule].flatten(1).transpose(0,1)
            query = self.query(h_node[atom_ids]).reshape(-1,4,64)
            if self.mode == 'D1':
                with torch.autocast(device_type=pos_node.device.type,enabled=False):
                    voxel_pos = origin[molecule].float()+self.voxel_xyz.float()@basis[molecule].float()
                key = self.key(voxel_features).reshape(1,-1,4,64).transpose(1,2)
                value = self.value(voxel_features).reshape(1,-1,4,64).transpose(1,2)
                attended = density_attention(query.transpose(0,1)[None],key,value,atom_pos[None],voxel_pos[None],self.beta,self.distance_bias,self.backend)[0].transpose(0,1).reshape(-1,256)
                result[atom_ids] = self.alpha*self.output(attended)
            else:
                # floor定义home，保留越界索引后取交集；禁止把原子夹到边缘。
                with torch.autocast(device_type=pos_node.device.type,enabled=False):
                    home = torch.floor((atom_pos.float()-origin[molecule].float())@torch.linalg.inv(basis[molecule].float())).long()
                neighborhood = home[:,None]+self.offsets[None]
                inside = ((neighborhood>=0)&(neighborhood<48)).all(-1)
                linear = neighborhood[...,2]*48*48+neighborhood[...,1]*48+neighborhood[...,0]
                selected = voxel_features[linear.masked_fill(~inside,0)]
                key = self.key(selected).reshape(len(atom_ids),343,4,64).permute(0,2,1,3)
                value = self.value(selected).reshape(len(atom_ids),343,4,64).permute(0,2,1,3)
                scores = (query.permute(0,1,2)[:,:,None]*key).sum(-1)/8
                if self.distance_bias:
                    with torch.autocast(device_type=pos_node.device.type,enabled=False):
                        voxel_pos = origin[molecule].float()+(neighborhood.float()+0.5)@basis[molecule].float()
                    squared = (atom_pos[:,None]-voxel_pos).square().sum(-1)/100
                    scores = scores-F.softplus(self.beta)[None,:,None]*squared[:,None]
                scores = scores.masked_fill(~inside[:,None],float('-inf'))
                nonempty = inside.any(-1)
                scores = torch.where(nonempty[:,None,None],scores,torch.zeros_like(scores))
                weights = scores.softmax(-1)*inside[:,None]
                attended = (weights[...,None]*value).sum(-2).reshape(-1,256)
                residual = self.alpha*self.output(attended)
                result[atom_ids] = residual*nonempty[:,None]
        return result
