"""从编码密度向当前配体原子产生320维残差，保持原坐标更新器。

DensityReadout的D1读取全部216个粗特征；D4按当前坐标读取home体素±3邻域。
几何以裁块角点和三条源XYZ体素基向量表示，原数组始终按ZYX存放。
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch_geometric.utils import to_dense_batch


def density_attention(query,key,value,query_pos,key_pos,beta,distance_bias,backend):
    """计算4头读出；输入(B,heads,N,64)，位置(B,N,3)，输出同query形状。

    分数=qk/8-softplus(beta)*||(x-p)/10Å||²。距离分数明确使用FP32几何，
    再交给SDPA数学内核；普通注意力可强制Flash。低精度QK增广未通过数值门控。
    """
    if backend == 'reference':
        scores = query @ key.transpose(-1,-2) / 8
        if distance_bias:
            squared = (query_pos[:,:,None]-key_pos[:,None]).square().sum(-1)/100
            scores = scores - F.softplus(beta)[None,:,None,None]*squared[:,None]
        return scores.softmax(-1) @ value
    if distance_bias:
        with torch.autocast(device_type=query.device.type,enabled=False):
            geometry_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
            squared = (query_pos.to(geometry_dtype)[:,:,None]-key_pos.to(geometry_dtype)[:,None]).square().sum(-1)/100
            bias = -F.softplus(beta.to(geometry_dtype))[None,:,None,None]*squared[:,None]
            with sdpa_kernel(SDPBackend.MATH):
                return F.scaled_dot_product_attention(query,key,value,attn_mask=bias,dropout_p=0,scale=1/8)
    original_dtype = value.dtype
    if backend == 'flash' and query.dtype not in (torch.float16,torch.bfloat16):
        query,key,value = query.bfloat16(),key.bfloat16(),value.bfloat16()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION) if backend == 'flash' else sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
        result = F.scaled_dot_product_attention(query,key,value,dropout_p=0,scale=1/8)
    return result.to(original_dtype)


class DensityReadout(nn.Module):
    """单个去噪块的独立4头密度读出；alpha可学习，初始0.1，额外门控恒为1。"""
    def __init__(self,node_dim,config):
        """建立一个块的独立投影；node_dim当前为320，config固定D1/D4与数值内核。"""
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
        """读取当前位置；返回(N,320)残差，PyG节点按分子编号排列。

        h_node为(N,320)，pos_node为局部XYZ(N,3)，batch_node为(N,)；
        feature为(B,C,Z,Y,X)，origin为(B,3)，basis为(B,3,3)源体素基向量。
        """
        original_dtype = h_node.dtype
        with torch.autocast(device_type=h_node.device.type,enabled=False):
            h_node = h_node.to(self.query.weight.dtype)
            if self.mode == 'D1':
                # 只对变长原子查询暂时补齐；所有分子的216个真实密度体素保持原样。
                query,valid = to_dense_batch(self.query(h_node),batch_node,batch_size=feature.shape[0])
                atom_pos,_ = to_dense_batch(pos_node,batch_node,batch_size=feature.shape[0])
                query = query.reshape(feature.shape[0],-1,4,64).transpose(1,2)
                voxel_features = feature.flatten(2).transpose(1,2).to(self.key.weight.dtype)
                key = self.key(voxel_features).reshape(feature.shape[0],-1,4,64).transpose(1,2)
                value = self.value(voxel_features).reshape(feature.shape[0],-1,4,64).transpose(1,2)
                with torch.autocast(device_type=pos_node.device.type,enabled=False):
                    voxel_pos = origin[:,None].float()+(self.voxel_xyz[None,:,:,None].float()*basis[:,None].float()).sum(-2)
                attended = density_attention(query,key,value,atom_pos,voxel_pos,self.beta,self.distance_bias,self.backend)
                attended = attended.transpose(1,2).reshape(feature.shape[0],-1,256)[valid]
                # alpha在混合精度下仍为FP32；先提升投影再乘，避免标量梯度在bf16中归约。
                return (self.alpha*self.output(attended).to(self.alpha.dtype)).to(original_dtype)
            # 全批次原子一次gather；全局索引包含分子偏移，不会跨分子读取密度。
            with torch.autocast(device_type=pos_node.device.type,enabled=False):
                atom_origin = origin.float()[batch_node]
                atom_basis = basis.float()[batch_node]
                inverse = torch.linalg.inv(basis.float())[batch_node]
                home = torch.floor(((pos_node.float()-atom_origin)[:,:,None]*inverse).sum(-2)).long()
            neighborhood = home[:,None]+self.offsets[None]
            inside = ((neighborhood>=0)&(neighborhood<48)).all(-1)
            linear = neighborhood[...,2]*48*48+neighborhood[...,1]*48+neighborhood[...,0]
            linear = linear.masked_fill(~inside,0)+batch_node[:,None]*(48**3)
            voxel_features = feature.permute(0,2,3,4,1).reshape(-1,48)
            selected = voxel_features[linear].to(self.key.weight.dtype)
            query = self.query(h_node).reshape(-1,4,64)
            key = self.key(selected).reshape(len(h_node),343,4,64).permute(0,2,1,3)
            value = self.value(selected).reshape(len(h_node),343,4,64).permute(0,2,1,3)
            scores = (query[:,:,None]*key).sum(-1)/8
            if self.distance_bias:
                with torch.autocast(device_type=pos_node.device.type,enabled=False):
                    voxel_pos = atom_origin[:,None]+((neighborhood.float()+0.5)[...,None]*atom_basis[:,None]).sum(-2)
                squared = (pos_node[:,None]-voxel_pos).square().sum(-1)/100
                scores = scores-F.softplus(self.beta)[None,:,None]*squared[:,None]
            scores = scores.masked_fill(~inside[:,None],float('-inf'))
            nonempty = inside.any(-1)
            scores = torch.where(nonempty[:,None,None],scores,torch.zeros_like(scores))
            weights = scores.softmax(-1)*inside[:,None]
            attended = (weights[...,None]*value).sum(-2).reshape(-1,256)
            residual = self.alpha*self.output(attended).to(self.alpha.dtype)
            return (residual*nonempty[:,None]).to(original_dtype)
