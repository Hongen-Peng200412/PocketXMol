"""从编码密度向当前配体原子产生特征残差, 由原坐标更新器继续预测坐标.

入口 DensityReadout 的 D1 模式读取全部 216 个粗体素, D4 模式按当前原子所在体素的各轴 ±3 邻域读取.
返回 (N, node_dim) 残差, N 为此批全部配体原子数, 当前 node_dim=320; 本模块不写文件.
几何以局部裁块角点和三条源 XYZ 体素基向量表示, 密度特征数组始终按 ZYX 存放.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch_geometric.utils import to_dense_batch


def density_attention(query,key,value,query_pos,key_pos,beta,distance_bias,backend):
    """按原子到体素的注意力分数聚合 value, 返回与 query 同形的特征.

    输入参数:
        - query: (B, heads, N, 64), B 个独立分子的 N 个查询原子, 当前 heads=4.
        - key: (B, heads, K, 64), 每个分子的 K 个密度体素键特征.
        - value: (B, heads, K, 64), 与 key 逐体素对齐的待聚合特征.
        - query_pos: (B, N, 3), 原子局部 XYZ 坐标, 单位 Å.
        - key_pos: (B, K, 3), 体素中心局部 XYZ 坐标, 与 query_pos 共用模型原点, 单位 Å.
        - beta: (heads,), 可学习距离参数; softplus(beta) 为非负系数, 每头独立.
        - distance_bias: bool, True 时分数为 qk/8-softplus(beta)*||(x-p)/10 Å||², False 时仅为 qk/8.
        - backend: str, reference 显式构造分数用于验收; 无距离偏置时 flash 强制 Flash, sdpa 允许 PyTorch 选择内核; flash 接收 FP32 投影时仅在内核入口转为 bf16, 返回时恢复原 value 类型.
    输出形状为 (B, heads, N, 64), softmax 及求和沿 K 个体素进行.
    距离偏置在禁用 autocast 的几何段计算, 正式输入使用 FP32, float64 验收输入保持 float64; 该分支显式使用 SDPA 数学内核.
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
            bias = -F.softplus(beta.to(geometry_dtype))[None,:,None,None]*squared[:,None]  # (B, heads, N, K), 逐头系数广播到同分子的原子-体素距离.
            with sdpa_kernel(SDPBackend.MATH):
                return F.scaled_dot_product_attention(query,key,value,attn_mask=bias,dropout_p=0,scale=1/8)
    original_dtype = value.dtype
    if backend == 'flash' and query.dtype not in (torch.float16,torch.bfloat16):
        query,key,value = query.bfloat16(),key.bfloat16(),value.bfloat16()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION) if backend == 'flash' else sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
        result = F.scaled_dot_product_attention(query,key,value,dropout_p=0,scale=1/8)
    return result.to(original_dtype)


# ================================================================================================
class DensityReadout(nn.Module):
    """为一个去噪块产生独立的四头密度残差, alpha 初始 0.1 且可学习.

    构造参数:
        - node_dim: int, 原子隐藏特征宽度及残差输出宽度, 当前为 320.
        - config: Mapping, mode 为 D1 或 D4, attention_backend 为注意力实现, distance_bias 决定是否加入平方距离项.
    前向输入输出见 forward; 六个去噪块各持有一套参数, 不共享 alpha、beta 或投影层, 额外门控恒为 1.
    """
    def __init__(self,node_dim,config):
        """建立一个块的独立投影; node_dim 当前为320, config 固定D1/D4与数值内核."""
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
        self.register_buffer('voxel_xyz',grid*(8 if self.mode=='D1' else 1)+0.5,persistent=False)  # (K, 3), 相对源裁块角点的 XYZ 连续体素索引; D1 为 8*j+0.5, K=216.
        offsets = torch.arange(-3,4)
        self.register_buffer('offsets',torch.stack(torch.meshgrid(offsets,offsets,offsets,indexing='ij'),dim=-1).reshape(-1,3),persistent=False)

    def forward(self,h_node,pos_node,batch_node,feature,origin,basis):
        """批量读取每个原子的当前密度邻域, 返回 (N, node_dim) 残差.

        输入参数:
            - h_node: (N, node_dim), 此批全部 N 个配体原子的当前隐藏特征; 输出恢复其原始类型.
            - pos_node: (N, 3), 与 h_node 逐原子对齐的当前局部 XYZ 坐标, 单位 Å; 每个去噪块重新读取.
            - batch_node: int64, (N,), 按 PyG 拼批顺序排列的分子编号, 如 [0, 0, 1], 索引 feature、origin、basis 首维.
            - feature: D1 为 (B, 256, 6, 6, 6), D4 为 (B, 48, 48, 48, 48), B 个分子的编码密度, 空间轴 ZYX.
            - origin: float32, (B, 3), 实际裁块角点的模型局部 XYZ 坐标, 单位 Å.
            - basis: float32, (B, 3, 3), 三行依次为源 X、Y、Z 一个体素步长在局部坐标系的向量, 单位 Å.
        投影和残差乘法使用参数类型, 正式混合精度训练的参数为 FP32; 普通 D1 Flash 的内核入口例外见 density_attention.
        几何使用 FP32 逐元素乘加, 避免全局 medium 矩阵精度改变 floor 边界; D4 邻域与裁块取交集, 空交集返回零残差.
        """
        original_dtype = h_node.dtype
        with torch.autocast(device_type=h_node.device.type,enabled=False):
            h_node = h_node.to(self.query.weight.dtype)
            if self.mode == 'D1':
                # query: (B, N_max, 256), N_max 为此批最大原子数; valid 为 (B, N_max), True 是真实原子, False 是补齐查询.
                # 只补齐变长查询, 不增加密度体素或跨分子混合.
                query,valid = to_dense_batch(self.query(h_node),batch_node,batch_size=feature.shape[0])
                atom_pos,_ = to_dense_batch(pos_node,batch_node,batch_size=feature.shape[0])
                query = query.reshape(feature.shape[0],-1,4,64).transpose(1,2)
                # (B, 256, 6, 6, 6) -> (B, 216, 256), X 索引变化最快; 全批量共享投影运算而不共享分子特征.
                voxel_features = feature.flatten(2).transpose(1,2).to(self.key.weight.dtype)
                key = self.key(voxel_features).reshape(feature.shape[0],-1,4,64).transpose(1,2)
                value = self.value(voxel_features).reshape(feature.shape[0],-1,4,64).transpose(1,2)
                with torch.autocast(device_type=pos_node.device.type,enabled=False):
                    voxel_pos = origin[:,None].float()+(self.voxel_xyz[None,:,:,None].float()*basis[:,None].float()).sum(-2)
                attended = density_attention(query,key,value,atom_pos,voxel_pos,self.beta,self.distance_bias,self.backend)
                attended = attended.transpose(1,2).reshape(feature.shape[0],-1,256)[valid]
                # (B, N_max, 256) 按 valid 恢复原 N 个原子; alpha 在 FP32 中参与乘法和标量梯度归约.
                return (self.alpha*self.output(attended).to(self.alpha.dtype)).to(original_dtype)
            # 全批次原子一次索引; 每个原子使用自己分子的角点和体素基向量.
            with torch.autocast(device_type=pos_node.device.type,enabled=False):
                atom_origin = origin.float()[batch_node]
                atom_basis = basis.float()[batch_node]
                inverse = torch.linalg.inv(basis.float())[batch_node]
                home = torch.floor(((pos_node.float()-atom_origin)[:,:,None]*inverse).sum(-2)).long()
            neighborhood = home[:,None]+self.offsets[None]  # int64, (N, 343, 3), 各原子周围 7³ 体素的 XYZ 索引, 可越界.
            inside = ((neighborhood>=0)&(neighborhood<48)).all(-1)  # bool, (N, 343), True 表示落在此分子裁块内.
            linear = neighborhood[...,2]*48*48+neighborhood[...,1]*48+neighborhood[...,0]
            linear = linear.masked_fill(~inside,0)+batch_node[:,None]*(48**3)  # (N, 343), 加分子偏移后索引全批 ZYX 展平体素; 无效占位随后屏蔽.
            voxel_features = feature.permute(0,2,3,4,1).reshape(-1,48)  # (B*48³, 48), 先分子再 ZYX 展平, 与 linear 的分子偏移对齐.
            selected = voxel_features[linear].to(self.key.weight.dtype)  # (N, 343, 48), 每个原子的邻域只来自其所属分子.
            query = self.query(h_node).reshape(-1,4,64)  # (N, 4, 64), 原子查询拆为四头.
            key = self.key(selected).reshape(len(h_node),343,4,64).permute(0,2,1,3)  # (N, 4, 343, 64), 邻域轴保留 linear 的顺序.
            value = self.value(selected).reshape(len(h_node),343,4,64).permute(0,2,1,3)  # (N, 4, 343, 64), 与 key 逐邻域体素对齐.
            scores = (query[:,:,None]*key).sum(-1)/8
            if self.distance_bias:
                with torch.autocast(device_type=pos_node.device.type,enabled=False):
                    voxel_pos = atom_origin[:,None]+((neighborhood.float()+0.5)[...,None]*atom_basis[:,None]).sum(-2)
                squared = (pos_node[:,None]-voxel_pos).square().sum(-1)/100
                scores = scores-F.softplus(self.beta)[None,:,None]*squared[:,None]
            scores = scores.masked_fill(~inside[:,None],float('-inf'))
            nonempty = inside.any(-1)
            scores = torch.where(nonempty[:,None,None],scores,torch.zeros_like(scores))
            weights = scores.softmax(-1)*inside[:,None]  # (N, 4, 343), 沿各原子的有效邻域归一化; 空交集权重全零.
            attended = (weights[...,None]*value).sum(-2).reshape(-1,256)  # 沿 343 个邻域求和得到 (N, 4, 64), 再合并四头为 (N, 256).
            residual = self.alpha*self.output(attended).to(self.alpha.dtype)
            return (residual*nonempty[:,None]).to(original_dtype)
