"""
实现 PocketXMol 默认去噪骨干中的节点、边和三维坐标联合更新，并注入口袋原子上下文。

主要入口是 :class:`ContextNodeEdgeNet`。``PMAsymDenoiser`` 用 ``node_only=True`` 的实例编码固定口袋图，
再用含 context 的实例反复更新配体节点特征、双向边特征和原子坐标。每个 block 依次调用
:class:`ContextNodeBlock`、:class:`EdgeBlock` 与 :class:`PosUpdate`；坐标更新只用网络预测的标量乘
相对坐标方向，从而保持对整体平移与旋转的等变性。

本模块不落盘；默认骨干返回 ``h_node(N,D_n)``、``pos_node(N,3)`` 和 ``h_edge(E,D_e)``。
"""

from itertools import permutations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, Linear, Conv1d, ModuleList
from torch_scatter import scatter_mean, scatter_sum, scatter_softmax
from torch_geometric.nn import radius_graph, knn_graph, knn, radius
from models.common import GaussianSmearing, MLP, NONLINEARITIES
from utils.motion import apply_axis_angle_rotation, apply_torsional_rotation_multiple_domains
from utils.data import edge_index_to_index_of_edge


class ContextNodeBlock(Module):
    """
    汇总分子内有向边消息与可选口袋—分子消息，更新每个目标节点的标量特征。

    形状符号:
        - N: 待更新节点数；分子去噪时为配体原子数，口袋编码时为口袋原子数。
        - E: 节点图有向边数。
        - P: 可选上下文节点数，即口袋原子数。
        - C: 可选上下文到节点的有向边数。
        - D_n: 输入节点特征宽度 ``node_dim``。
        - D_e: 输入边特征宽度 ``edge_dim``。
        - D_g: 节点 prompt 宽度 ``gate_dim``；默认配体去噪为 2，口袋编码为 0。
        - D_c: 上下文节点特征宽度 ``context_dim``。
        - D_ce: 上下文边特征宽度 ``context_edge_dim``。
        - D_h: 消息隐藏宽度 ``hidden_dim``。

    构造参数:
        - node_dim: int, 输入与输出节点特征宽度 D_n。
        - edge_dim: int, 分子内边特征宽度 D_e。
        - hidden_dim: int, 节点、边和消息投影的共同宽度 D_h。
        - gate_dim: int, 每个节点的 prompt 宽度 D_g；大于 0 时启用分子内边门控和上下文门控。
        - context_dim: int, 上下文节点特征宽度 D_c；0 表示不构造上下文消息网络。
        - context_edge_dim: int, 上下文边特征宽度 D_ce。
        - layernorm_before: bool, False 使用 ``LayerNorm(update + residual)``，True 使用 ``LayerNorm(update) + residual``。

    前向输入:
        - x: (N, D_n), 当前节点特征。
        - edge_index: int64, (2, E), 分子内有向边；每列 ``[目标节点, 来源节点]``，两行数值均索引 x 第一维。
        - edge_attr: (E, D_e), 分子内有向边特征；与 ``edge_index`` 第二维逐边对齐。
        - node_extra: (N, D_g)|None, fixed prompt；D_g>0 时必需，第 i 行与 x 第 i 个节点对齐。
        - ctx_x: (P, D_c)|None, 可选上下文节点特征。
        - ctx_edge_index: int64, (2, C)|None, 上下文边；每列 ``[待更新目标节点, 上下文来源节点]``，两行分别索引 x 与 ctx_x 第一维。
        - ctx_edge_attr: (C, D_ce)|None, 上下文边的距离特征；与 ``ctx_edge_index`` 第二维对齐。

    前向输出:
        - out: (N, D_n), 融合分子内邻居、可选上下文与残差后的节点特征；节点顺序保持不变。
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, gate_dim,
                 context_dim=0, context_edge_dim=0, layernorm_before=False):
        super().__init__()
        # ``self.node_dim``：int D_n，输入/输出节点特征宽度。
        self.node_dim = node_dim
        # ``self.edge_dim``：int D_e，分子内有向边特征宽度。
        self.edge_dim = edge_dim
        # ``self.gate_dim``：int D_g，节点 fixed prompt 宽度；0 关闭门控。
        self.gate_dim = gate_dim
        # ``self.context_dim``：int D_c，上下文节点特征宽度；0 表示没有上下文网络。
        self.context_dim = context_dim
        # ``self.context_edge_dim``：int D_ce，上下文边特征宽度。
        self.context_edge_dim = context_edge_dim
        # ``self.layernorm_before``：bool，控制 LayerNorm 放在更新+残差之前还是之后。
        self.layernorm_before = layernorm_before
        
        # ``self.node_net``：MLP；[N, D_n] -> [N, D_h] 的节点消息投影。
        self.node_net = MLP(node_dim, hidden_dim, hidden_dim)
        # ``self.edge_net``：MLP；[E, D_e] -> [E, D_h] 的边消息投影。
        self.edge_net = MLP(edge_dim, hidden_dim, hidden_dim)
        # ``self.msg_net``：Linear；[E, D_h] -> [E, D_h] 的边—端点融合消息变换。
        self.msg_net = Linear(hidden_dim, hidden_dim)

        if self.gate_dim > 0:
            # ``self.gate``：MLP，按边和双端点/双 prompt 拼接特征预测 (E, D_h) 门控 logits。
            self.gate = MLP(edge_dim+(node_dim+gate_dim)*2, hidden_dim, hidden_dim)

        # ``self.centroid_lin``：Linear；[N, D_n] -> [N, D_h] 的节点自身残差前投影。
        self.centroid_lin = Linear(node_dim, hidden_dim)
        # ``self.layer_norm``：LayerNorm，对最后 D_h/D_n 维归一；当前配置 hidden_dim 与 node_dim 相等。
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # self.act = nn.ReLU()
        # self.out_transform = Linear(hidden_dim, node_dim)
        # ``self.out_layer``：MLP；[N, D_h] -> [N, D_n]，把聚合消息映射回节点残差宽度。
        self.out_layer = MLP(hidden_dim, node_dim, hidden_dim)
        
        if self.context_dim > 0:
            # ``self.ctx_node_net``：MLP；[P, D_c] -> [P, D_h] 的上下文节点投影。
            self.ctx_node_net = MLP(context_dim, hidden_dim, hidden_dim)
            # ``self.ctx_edge_net``：MLP；[C, D_ce] -> [C, D_h] 的上下文边投影。
            self.ctx_edge_net = MLP(context_edge_dim, hidden_dim, hidden_dim)
            # ``self.ctx_msg_net``：Linear；[C, D_h] -> [C, D_h] 的上下文逐边消息变换。
            self.ctx_msg_net = Linear(hidden_dim, hidden_dim)
            # ``self.ctx_gate``：MLP；按口袋来源、上下文边、配体目标及 prompt 拼接特征预测形状为 (C, D_h) 的门控 logits。
            self.ctx_gate = MLP(context_dim+context_edge_dim+(node_dim+gate_dim), hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr, node_extra,
                ctx_x=None, ctx_edge_index=None, ctx_edge_attr=None):
        """将分子内边与可选口袋上下文消息归约到目标节点。

        输入参数:
            - x: FloatTensor，形状为 (N, D_n)，当前待更新节点特征。
            - edge_index: LongTensor，形状为 (2, E)，每列为目标节点、来源节点，均索引 ``x`` 第一维。
            - edge_attr: FloatTensor，形状为 (E, D_e)，与 ``edge_index`` 列逐边对齐的分子内边特征。
            - node_extra: FloatTensor|None，形状为 (N, D_g)，逐节点 fixed prompt；``gate_dim>0`` 时必需。
            - ctx_x: FloatTensor|None，形状为 (P, D_c)，可选口袋上下文节点特征。
            - ctx_edge_index: LongTensor|None，形状为 (2, C)，第一行索引 ``x``，第二行索引 ``ctx_x``。
            - ctx_edge_attr: FloatTensor|None，形状为 (C, D_ce)，与 ``ctx_edge_index`` 列逐边对齐的上下文边特征。

        返回值:
            - out: FloatTensor，形状为 (N, D_n)，融合分子内邻居、可选口袋上下文与残差后的节点特征。
        """
        # ``N``：int，待更新节点数 N；作为 scatter_sum 的显式输出第一维，保留没有入边的节点。
        N = x.size(0)
        # ``row``：LongTensor，形状为 (E,)，每条分子内有向边的目标节点编号，索引 ``x`` 第一维。
        # ``col``：LongTensor，形状为 (E,)，每条分子内有向边的来源节点编号，索引 ``x`` 第一维。
        row, col = edge_index   # (E,) , (E,)

        # ``h_node``：FloatTensor，形状为 (N, D_h)；逐节点把输入特征投影到消息隐藏空间。
        h_node = self.node_net(x)  # (N, H)

        # ``h_edge``：FloatTensor，形状为 (E, D_h)；逐边把当前边特征投影到消息隐藏空间。
        h_edge = self.edge_net(edge_attr)  # (E, H_per_head)
        # ``msg_j``：FloatTensor，形状为 (E, D_h)；每条边联合自身、来源节点和目标节点特征形成尚未门控的消息。
        msg_j = self.msg_net(h_edge + h_node[col] + h_node[row])

        if self.gate_dim > 0:
            # ``gate``：FloatTensor，形状为 (E, D_h)；门控输入按边、来源节点、来源 prompt、目标节点、目标 prompt 顺序拼接。
            gate = self.gate(torch.cat([edge_attr, x[col], node_extra[col], x[row], node_extra[row]], dim=-1))
            # ``msg_j``：FloatTensor，形状为 (E, D_h)；sigmoid 将每条边每个隐藏通道的消息缩放到 0..1。
            msg_j = msg_j * torch.sigmoid(gate)

        # ``aggr_msg``：FloatTensor，形状为 (N, D_h)；按 ``row`` 对 E 条消息求和，第 i 行汇总所有指向节点 i 的分子内边。
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=N)
        # ``out``：FloatTensor，形状为 (N, D_h)；节点自身线性投影与入边消息之和。
        out = self.centroid_lin(x) + aggr_msg
        
        # context messages
        if ctx_x is not None:
            # ``row``：LongTensor，形状为 (C,)，每条上下文边的目标配体节点编号，索引 ``x`` 第一维。
            # ``col``：LongTensor，形状为 (C,)，每条上下文边的来源口袋节点编号，索引 ``ctx_x`` 第一维。
            row, col = ctx_edge_index
            # ``h_ctx``：FloatTensor，形状为 (P, D_h)；逐口袋节点投影到上下文消息空间。
            h_ctx = self.ctx_node_net(ctx_x)
            # ``h_ctx_edge``：FloatTensor，形状为 (C, D_h)；逐上下文边投影距离基特征。
            h_ctx_edge = self.ctx_edge_net(ctx_edge_attr)
            # ``msg_ctx``：FloatTensor，形状为 (C, D_h)；上下文边特征与来源口袋节点特征逐通道相乘，再经线性层形成消息。
            msg_ctx = self.ctx_msg_net(h_ctx_edge * h_ctx[col])
            if self.gate_dim > 0:
                # ``gate``：FloatTensor，形状为 (C, D_h)；门控输入依次为上下文边、口袋来源节点、分子目标节点和目标 fixed prompt。
                gate = self.ctx_gate(torch.cat([ctx_edge_attr, ctx_x[col], x[row], node_extra[row]], dim=-1))
                # ``msg_ctx``：FloatTensor，形状为 (C, D_h)；逐上下文边、逐通道乘 0..1 门控。
                msg_ctx = msg_ctx * torch.sigmoid(gate)
            # ``aggred_ctx_msg``：FloatTensor，形状为 (N, D_h)；按配体目标节点编号汇总所有相邻口袋原子的上下文消息。
            aggred_ctx_msg = scatter_sum(msg_ctx, row, dim=0, dim_size=N)
            # ``out``：FloatTensor，形状为 (N, D_h)；把口袋上下文归约结果加到分子内消息结果。
            out = out + aggred_ctx_msg

        # ``out``：FloatTensor，形状为 (N, D_n)；把隐藏消息宽度映射回节点宽度，随后按配置放置 LayerNorm 与残差。
        out = self.out_layer(out)
        if not self.layernorm_before:
            # ``out``：FloatTensor，形状为 (N, D_n)；更新与输入残差相加后再归一化。
            out = self.layer_norm(out + x)
        else:
            # ``out``：FloatTensor，形状为 (N, D_n)；先归一化更新分量，再加未归一化输入残差。
            out = self.layer_norm(out) + x
        return out


class NodeEncoder(Module):
    
    def __init__(self, node_dim=256, edge_dim=64, key_dim=128, num_heads=4, 
                    num_blocks=6, k=48, cutoff=10.0, use_atten=True, use_gate=True,
                    dist_version='new'):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.k = k
        self.cutoff = cutoff
        self.use_atten = use_atten
        self.use_gate = use_gate

        if dist_version == 'new':
            self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=20)
            self.edge_emb = Linear(self.additional_edge_feat+20, edge_dim)
        elif dist_version == 'old':
            self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_dim-self.additional_edge_feat)
            self.edge_emb = Linear(edge_dim, edge_dim)
        else:
            raise NotImplementedError('dist_version notimplemented')
        self.node_blocks = ModuleList()
        for _ in range(num_blocks):
            block = NodeBlock(
                node_dim=node_dim,
                edge_dim=edge_dim,
                key_dim=key_dim,
                num_heads=num_heads,
                use_atten=use_atten,
                use_gate=use_gate,
            )
            self.node_blocks.append(block)

    @property
    def out_channels(self):
        return self.node_dim

    def forward(self, h, pos, edge_index, is_mol):
        #NOTE in the encoder, the edge dose not change since the position of mol and protein is fixed
        # edge_index = radius_graph(pos, self.cutoff, batch=batch, loop=False)
        edge_attr = self._add_edge_features(pos, edge_index, is_mol)
        for interaction in self.node_blocks:
            h = h + interaction(h, edge_index, edge_attr)
        return h

    @property
    def additional_edge_feat(self,):
        return 2

    def _add_edge_features(self, pos, edge_index, is_mol):
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)
        # 2-vector represent the two node types (atoms of protein or mol)
        edge_src_feat = is_mol[edge_index[0]].float().view(-1, 1)
        edge_dst_feat = is_mol[edge_index[1]].float().view(-1, 1)
        edge_attr = torch.cat([edge_attr, edge_src_feat, edge_dst_feat], dim=1)
        edge_attr = self.edge_emb(edge_attr)
        return edge_attr


class BondFFN(Module):
    """
    将一条边的特征、一个端点的节点特征和可选 prompt 融合为定宽边消息。

    形状符号:
        - E: 并行处理的边数。
        - D_b: 输入边特征宽度。
        - D_n: 端点节点特征宽度。
        - D_i: 中间特征宽度。
        - D_o: 输出边特征宽度。
        - D_g: 边 prompt 特征宽度。

    构造参数:
        - bond_dim: int, 输入边特征宽度 D_b。
        - node_dim: int, 与每条边对齐的端点/端点组合特征宽度 D_n。
        - inter_dim: int, 两路投影相加后的中间宽度 D_i。
        - gate_dim: int, ``extra`` prompt 宽度 D_g；大于 0 时启用输出门控。
        - out_dim: int|None, 输出宽度 D_o；None 时取 ``bond_dim``。

    前向输入:
        - bond_feat_input: (E, D_b), 每条有向边当前特征。
        - node_feat_input: (E, D_n), 与同一边逐行对齐的端点或双端点特征。
        - extra: (E, D_g)|None, 与边逐行对齐的 fixed prompt；仅 ``gate_dim>0`` 时读取。

    前向输出:
        - inter_feat: (E, D_o), 边—节点加性融合后的消息；启用门控时逐通道乘 ``sigmoid(gate)``。
    """
    def __init__(self, bond_dim, node_dim, inter_dim, gate_dim, out_dim=None):
        super().__init__()
        # ``out_dim``：int D_o，未显式提供时保持输入边宽度 D_b。
        out_dim = bond_dim if out_dim is None else out_dim
        # ``self.gate_dim``：int D_g，额外 prompt 宽度；0 关闭门控网络。
        self.gate_dim = gate_dim
        # ``self.bond_linear``：Linear；[E, D_b] -> [E, D_i] 的无偏置边投影。
        self.bond_linear = Linear(bond_dim, inter_dim, bias=False)
        # ``self.node_linear``：Linear；[E, D_n] -> [E, D_i] 的无偏置节点投影。
        self.node_linear = Linear(node_dim, inter_dim, bias=False)
        # ``self.inter_module``：MLP；[E, D_i] -> [E, D_o] 的融合变换。
        self.inter_module = MLP(inter_dim, out_dim, inter_dim)
        if self.gate_dim > 0:
            # ``self.gate``：MLP；[E, D_b+D_n+D_g] -> [E, D_o] 的门控 logits。
            self.gate = MLP(bond_dim+node_dim+gate_dim, out_dim, 32)

    def forward(self, bond_feat_input, node_feat_input, extra):
        """保持边顺序，融合边特征、对齐端点特征与可选 prompt。

        输入参数:
            - bond_feat_input: FloatTensor，形状为 (E, D_b)，当前有向边特征。
            - node_feat_input: FloatTensor，形状为 (E, D_n)，与每条边逐行对齐的端点或双端点特征。
            - extra: FloatTensor|None，形状为 (E, D_g)，与每条边逐行对齐的 fixed prompt；``gate_dim>0`` 时必需。

        返回值:
            - inter_feat: FloatTensor，形状为 (E, D_o)，加性融合并可选逐通道门控后的边消息。
        """
        # ``bond_feat``：Tensor，形状为 (E, D_i)；无偏置线性投影后的边分量。
        bond_feat = self.bond_linear(bond_feat_input)
        # ``node_feat``：Tensor，形状为 (E, D_i)；无偏置线性投影后的节点分量。
        node_feat = self.node_linear(node_feat_input)
        # ``inter_feat``：Tensor，形状为 (E, D_i)；逐边逐通道相加，不在 E 维发生归约。
        inter_feat = bond_feat + node_feat
        # ``inter_feat``：Tensor，形状为 (E, D_o)；共享 MLP 只改变最后一维。
        inter_feat = self.inter_module(inter_feat)
        if self.gate_dim > 0:
            # ``gate``：Tensor，形状为 (E, D_o)；每条边的门控由原始边、原始对齐节点和 prompt 拼接得到。
            gate = self.gate(torch.cat([bond_feat_input, node_feat_input, extra], dim=-1))
            # ``inter_feat``：Tensor，形状为 (E, D_o)；逐边/通道乘 0..1 门控。
            inter_feat = inter_feat * torch.sigmoid(gate)
        return inter_feat


class EdgeBlock(Module):
    """
    通过共享端点的邻边消息、双端点节点特征和边残差更新每条有向边。

    形状符号:
        - N: 节点数。
        - E: 有向边数。
        - D_e: 边特征宽度。
        - D_n: 节点特征宽度。
        - D_g: 边 prompt 特征宽度。

    构造参数:
        - edge_dim: int, 输入与输出边特征宽度 D_e。
        - node_dim: int, 节点特征宽度 D_n。
        - hidden_dim: int|None, ``BondFFN`` 中间宽度；None 时取 ``2*D_e``。
        - gate_dim: int, ``bond_extra`` 宽度 D_g；大于 0 时端点邻边消息启用门控。
        - layernorm_before: bool, False 使用 ``LayerNorm(update+residual)``，True 使用 ``LayerNorm(update)+residual``。

    前向输入:
        - h_bond: (E, D_e), 当前有向边特征。
        - bond_index: int64, (2, E), 每列 ``[左/目标节点, 右/来源节点]``；两行索引 h_node 第一维。
        - h_node: (N, D_n), 已在当前 block 更新后的节点特征。
        - bond_extra: (E, D_g)|None, 与有向边逐行对齐的 fixed edge/distance prompt。

    前向输出:
        - h_bond: (E, D_e), 融合两端邻边、两端节点和自身残差的有向边特征；边顺序不变。
    """
    def __init__(self, edge_dim, node_dim, hidden_dim=None, gate_dim=0, layernorm_before=False):
        super().__init__()
        # ``self.gate_dim``：int D_g，逐边 prompt 宽度；0 关闭两个 BondFFN 的门控。
        self.gate_dim = gate_dim
        # ``inter_dim``：int D_i，未配置时取 2D_e，作为边—端点融合中间宽度。
        inter_dim = edge_dim * 2 if hidden_dim is None else hidden_dim
        # ``self.layernorm_before``：bool，控制 LayerNorm 与边残差的先后顺序。
        self.layernorm_before = layernorm_before

        # ``self.bond_ffn_left``：BondFFN，融合当前边与左/目标端点节点特征。
        self.bond_ffn_left = BondFFN(edge_dim, node_dim, inter_dim=inter_dim, gate_dim=gate_dim)
        # ``self.bond_ffn_right``：BondFFN，融合当前边与右/来源端点节点特征。
        self.bond_ffn_right = BondFFN(edge_dim, node_dim, inter_dim=inter_dim, gate_dim=gate_dim)

        # ``self.msg_left``：Linear，把按端点归约后的左侧邻边消息保持为 D_e。
        self.msg_left = Linear(edge_dim, edge_dim)
        # ``self.msg_right``：Linear，把按端点归约后的右侧邻边消息保持为 D_e。
        self.msg_right = Linear(edge_dim, edge_dim)

        # ``self.node_ffn_left``：Linear；[E, D_n] -> [E, D_e] 的左端点节点投影。
        self.node_ffn_left = Linear(node_dim, edge_dim)
        # ``self.node_ffn_right``：Linear；[E, D_n] -> [E, D_e] 的右端点节点投影。
        self.node_ffn_right = Linear(node_dim, edge_dim)

        # ``self.self_ffn``：Linear；[E, D_e] -> [E, D_e] 的当前边自身投影。
        self.self_ffn = Linear(edge_dim, edge_dim)
        # ``self.layer_norm``：LayerNorm，对每条边最后 D_e 维归一。
        self.layer_norm = nn.LayerNorm(edge_dim)
        # ``self.out_layer``：MLP，五项相加后的 (E, D_e) 更新变换。
        self.out_layer = MLP(edge_dim, edge_dim, edge_dim)

    def forward(self, h_bond, bond_index, h_node, bond_extra):
        """按共享端点归约邻边消息，再更新每条有向边。

        输入参数:
            - h_bond: FloatTensor，形状为 (E, D_e)，当前有向边特征。
            - bond_index: LongTensor，形状为 (2, E)，每列为左端点、右端点，均索引 ``h_node`` 第一维。
            - h_node: FloatTensor，形状为 (N, D_n)，当前节点特征。
            - bond_extra: FloatTensor|None，形状为 (E, D_g)，与有向边逐行对齐的 fixed edge/distance prompt。

        返回值:
            - h_bond: FloatTensor，形状为 (E, D_e)，融合双端邻边、双端节点和边残差后的特征。
        """
        # ``N``：int，节点数 N；用于两个 scatter_sum 显式保留无邻边节点槽位。
        N = h_node.size(0)
        # ``left_node``：LongTensor，形状为 (E,)，每条边的左端点编号，索引 ``h_node`` 第一维。
        # ``right_node``：LongTensor，形状为 (E,)，每条边的右端点编号，索引 ``h_node`` 第一维。
        left_node, right_node = bond_index

        # ``msg_bond_left``：Tensor，形状为 (E, D_e)；每条边融合自身与左/目标端点，再准备按右/来源端点归约。
        msg_bond_left = self.bond_ffn_left(h_bond, h_node[left_node], bond_extra)
        # ``msg_bond_left``：[E,D_e] -> [N,D_e]，第 j 行汇总所有右端点为 j 的边消息。
        msg_bond_left = scatter_sum(msg_bond_left, right_node, dim=0, dim_size=N)
        # ``msg_bond_left``：[N,D_e] -> [E,D_e]，为当前边取其左端点处聚合的邻边消息。
        msg_bond_left = msg_bond_left[left_node]

        # ``msg_bond_right``：Tensor，形状为 (E, D_e)；每条边融合自身与右/来源端点，再准备按左/目标端点归约。
        msg_bond_right = self.bond_ffn_right(h_bond, h_node[right_node], bond_extra)
        # ``msg_bond_right``：[E,D_e] -> [N,D_e]，第 i 行汇总所有左端点为 i 的边消息。
        msg_bond_right = scatter_sum(msg_bond_right, left_node, dim=0, dim_size=N)
        # ``msg_bond_right``：[N,D_e] -> [E,D_e]，为当前边取其右端点处聚合的邻边消息。
        msg_bond_right = msg_bond_right[right_node]
        
        # ``h_bond_update``：Tensor，形状为 (E, D_e)；五项逐边相加：两侧邻边消息、两端节点投影和当前边自身投影。
        h_bond_update = (
            self.msg_left(msg_bond_left)
            + self.msg_right(msg_bond_right)
            + self.node_ffn_left(h_node[left_node])
            + self.node_ffn_right(h_node[right_node])
            + self.self_ffn(h_bond)
        )
        # ``h_bond_update``：Tensor，形状为 (E, D_e)；共享 MLP 细化五项加和后的逐边更新。
        h_bond_update = self.out_layer(h_bond_update)

        # skip connection
        if not self.layernorm_before:
            # ``h_bond``：Tensor，形状为 (E, D_e)；更新与旧边残差相加后归一化。
            h_bond = self.layer_norm(h_bond_update + h_bond)
        else:
            # ``h_bond``：Tensor，形状为 (E, D_e)；先归一化更新，再加旧边残差。
            h_bond = self.layer_norm(h_bond_update) + h_bond
        return h_bond


class ContextNodeEdgeNet(Module):
    """
    堆叠多层节点—边—坐标更新，并在每层重建分子—口袋近邻关系。

    形状符号:
        - N: 待更新图的节点数。
        - E: 待更新图的有向边数；配体图中 E=2H，口袋图中 E=E_p。
        - P: 可选口袋上下文节点数。
        - C: 分子—口袋有向边数。
        - D_n: 节点特征宽度。
        - D_e: 分子内边特征宽度。
        - D_c: 上下文节点特征宽度。
        - D_gn: 节点 prompt 宽度。
        - D_ge: 边 prompt 宽度。
        - R: 分子内距离高斯基通道数。
        - R_c: 上下文距离高斯基通道数。

    构造参数:
        - node_dim: int, 输入、隐藏和输出节点宽度 D_n。
        - edge_dim: int, block 内有向边宽度 D_e；``node_only=True`` 时只是距离嵌入后的临时宽度。
        - hidden_dim: int, ``ContextNodeBlock`` 消息隐藏宽度。
        - num_blocks: int, 节点/边/坐标联合更新层数。
        - dist_cfg: 映射，传给 ``GaussianSmearing``；``num_gaussians`` 定义 R，``start/stop/type_`` 定义距离基区间。
        - gate_dim: int, 节点 prompt 宽度 D_gn；默认配体为 2、口袋为 0。
        - context_dim: int, 口袋上下文节点宽度 D_c；无上下文时为 0。
        - context_cfg.edge_dim: int, 分子—口袋边隐藏宽度。
        - context_cfg.knn: int, 每个配体原子选择的口袋近邻数；小于 100 时使用 ``torch_geometric.nn.knn``。
        - context_cfg.dist_cfg: 映射，定义分子—口袋距离的 R_c 个高斯基。
        - node_only: bool, True 时只更新节点特征，不更新边和坐标；用于固定口袋编码。
        - downsample_context: bool, True 时仅在选择口袋近邻时给口袋坐标加标准差 5 Å 的高斯扰动，距离仍用原坐标计算。
        - layernorm_before: bool, 传给节点/边 block 的残差归一化顺序。

    前向输入:
        - h_node: (N, D_n), 当前节点特征。
        - pos_node: (N, 3), 当前节点局部坐标，最后一维按 XYZ 排列，单位 Å。
        - h_edge: (E, D_e)|None, 当前有向边特征；``node_only=True`` 的口袋编码传 None。
        - edge_index: int64, (2, E), 每列 ``[目标节点, 来源节点]``，两行索引 h_node/pos_node 第一维。
        - node_extra: (N, D_gn)|None, 节点 fixed prompt；口袋编码传 None。
        - edge_extra: (E, D_ge)|None, 有向边 fixed prompt；口袋编码传 None。
        - batch_node: int64, (N,)|None, 每个节点所属图编号；构造上下文 kNN 时必需。
        - h_ctx: (P, D_c)|None, 已编码口袋节点特征；None 表示不注入上下文。
        - pos_ctx: (P, 3)|None, 与 pos_node 使用同一局部原点的口袋坐标，单位 Å。
        - batch_ctx: int64, (P,)|None, 每个口袋原子所属图编号。

    前向输出:
        - ``node_only=True``: h_node, (N, D_n), num_blocks 层更新后的节点特征。
        - ``node_only=False``: tuple ``(h_node, pos_node, h_edge)``，形状依次为 ``(N,D_n)``、``(N,3)``、``(E,D_e)``。

    每层数据流:
        - 由当前 pos_node 计算 E 条分子内边的相对向量、距离和 RBF；位置更新后下一层会重新计算。
        - 若有口袋，每层按当前配体位置重建 C 条近邻边并计算上下文 RBF。
        - 依次更新节点、边，再把分子内与口袋相对向量产生的两个坐标增量加到 pos_node。
        - 口袋编码 ``node_only=True`` 时固定使用首次计算的距离 RBF，只执行节点更新。
    """
    def __init__(self, node_dim, edge_dim, hidden_dim,
                 num_blocks, dist_cfg, gate_dim=0,
                 context_dim=0, context_cfg=None,
                 node_only=False, **kwargs):
        super().__init__()
        # ``self.node_dim``：int D_n，所有 block 的节点输入/输出宽度。
        self.node_dim = node_dim
        # ``self.edge_dim``：int D_e，所有 block 的有向边隐藏宽度。
        self.edge_dim = edge_dim
        # ``self.num_blocks``：int L，节点—边—坐标联合更新层数。
        self.num_blocks = num_blocks
        # ``dist_cfg.start``：float|缺省，分子内距离高斯中心下界，单位 Å，缺省 0。
        # ``dist_cfg.stop``：float|缺省，分子内距离高斯中心上界，单位 Å，缺省 10。
        # ``dist_cfg.num_gaussians``：int，分子内距离高斯基通道数 R。
        # ``dist_cfg.type_``：str|缺省，高斯中心排布类型。
        # ``self.dist_cfg``：dict，保留上述分子内距离展开叶。
        self.dist_cfg = dist_cfg
        # ``self.gate_dim``：int D_gn，节点 fixed prompt 宽度。
        self.gate_dim = gate_dim
        # ``self.node_only``：bool，True 只编码节点，不建立边/位置更新模块。
        self.node_only = node_only
        # ``kwargs.downsample_context``：bool|缺省，只在上下文 kNN 搜索时扰动口袋坐标。
        # ``kwargs.layernorm_before``：bool|缺省，节点/边残差块的归一化顺序。
        # ``self.kwargs``：dict，保留上述额外结构开关。
        self.kwargs = kwargs
        # ``self.downsample_context``：bool，是否只在上下文近邻搜索时给口袋坐标加 5 Å 噪声。
        self.downsample_context = kwargs.get('downsample_context', False)
        # ``self.layernorm_before``：bool，节点/边 block 中 LayerNorm 与残差的先后顺序。
        self.layernorm_before = kwargs.get("layernorm_before", False)

        # ``self.distance_expansion``：GaussianSmearing；[E] -> [E, R] 的分子内距离基展开。
        self.distance_expansion = GaussianSmearing(**dist_cfg)
        # ``num_gaussians``：int，分子内标量距离展开后的通道数 R。
        num_gaussians = dist_cfg['num_gaussians']
        # ``input_edge_dim``：int，第一层边嵌入输入宽度；口袋 node-only 只有 RBF，配体则拼接当前 D_e 边特征与 RBF。
        input_edge_dim = num_gaussians + (0 if node_only else edge_dim)
            
        # for context
        # ``context_cfg.edge_dim``：int，分子—口袋有向边隐藏宽度 D_ce。
        # ``context_cfg.knn``：int，每个配体目标节点连接的同图口袋近邻数上限。
        # ``context_cfg.dist_cfg.start``：float|缺省，上下文距离高斯中心下界，单位 Å。
        # ``context_cfg.dist_cfg.stop``：float|缺省，上下文距离高斯中心上界，单位 Å。
        # ``context_cfg.dist_cfg.num_gaussians``：int，上下文距离基通道数 R_c。
        # ``context_cfg.dist_cfg.type_``：str|缺省，上下文高斯中心排布类型。
        # ``self.context_cfg``：dict|None，保留上述分子—口袋边与距离基叶。
        self.context_cfg = context_cfg
        if context_cfg is not None:
            # ``context_edge_dim``：int，分子—口袋边在 node/position block 内使用的隐藏宽度 D_ce。
            context_edge_dim = context_cfg['edge_dim']
            # ``self.knn``：int k，每个配体目标选择的同图口袋来源近邻数；>=100 切换显式全连接。
            self.knn = context_cfg['knn']
            # ``self.dist_exp_ctx``：GaussianSmearing；[C] -> [C, R_c] 的上下文距离基展开。
            self.dist_exp_ctx = GaussianSmearing(**context_cfg['dist_cfg'])
            # ``input_context_edge_dim``：int，上下文边线性嵌入前的 RBF 通道数 R_c。
            input_context_edge_dim = context_cfg['dist_cfg']['num_gaussians']
            assert context_dim > 0, 'context_dim should be larger than 0 if context_cfg is not None'
            assert not node_only, 'not support node_only with context'
        else:
            # ``context_edge_dim``：int 0，无上下文时传给 ContextNodeBlock 的上下文边宽度占位。
            context_edge_dim = 0
        
        # node network
        # ``self.edge_embs``：ModuleList 长度 L，每层把“旧边特征+当前距离基”映射回 D_e。
        self.edge_embs = ModuleList()
        # ``self.node_blocks_with_edge``：ModuleList 长度 L，每层一个 ContextNodeBlock。
        self.node_blocks_with_edge = ModuleList()
        if not node_only:
            # ``self.edge_blocks``：ModuleList 长度 L，每层一个有向边更新块。
            self.edge_blocks = ModuleList()
            # ``self.pos_blocks``：ModuleList 长度 L，每层一个分子内坐标更新块。
            self.pos_blocks = ModuleList()
            if self.context_cfg is not None:
                # ``self.ctx_edge_embs``：ModuleList 长度 L，每层把上下文 RBF 映射为 D_ce。
                self.ctx_edge_embs = ModuleList()
                # ``self.ctx_pos_blocks``：ModuleList 长度 L，每层一个配体—口袋坐标更新块。
                self.ctx_pos_blocks = ModuleList()
        # ``_``：int，当前 block 的 0-based 构造序号；网络层存入 ModuleList 后无需保留该编号。
        for _ in range(num_blocks):
            # edge emb
            self.edge_embs.append(Linear(input_edge_dim, edge_dim))
            # node update
            self.node_blocks_with_edge.append(ContextNodeBlock(
                node_dim, edge_dim, hidden_dim, gate_dim,
                context_dim, context_edge_dim, layernorm_before=self.layernorm_before
            ))
            if node_only:
                continue
            # edge update
            self.edge_blocks.append(EdgeBlock(
                edge_dim=edge_dim, node_dim=node_dim, gate_dim=gate_dim, layernorm_before=self.layernorm_before
            ))
            # pos update
            self.pos_blocks.append(PosUpdate(
                node_dim, edge_dim, hidden_dim=edge_dim, gate_dim=gate_dim*2,
            ))
            if self.context_cfg is not None:
                self.ctx_edge_embs.append(Linear(input_context_edge_dim, context_edge_dim))
                self.ctx_pos_blocks.append(PosUpdate(
                    node_dim, context_edge_dim, hidden_dim=edge_dim, gate_dim=gate_dim,
                    node_dim_right=context_dim,
                ))
        
        self.local_update = kwargs.get('local_update', False)
        if self.local_update:
            self.local_net = LocalPosUpdate(node_dim, edge_dim, node_dim, cutoff=3.8)
        
                
    def forward(self, h_node, pos_node, h_edge, edge_index,
                node_extra, edge_extra, batch_node=None,
                h_ctx=None, pos_ctx=None, batch_ctx=None):
        """逐层更新节点、边与坐标，并在每层重建配体—口袋上下文边。

        输入参数:
            - h_node: FloatTensor，形状为 (N, D_n)，当前配体或口袋节点特征。
            - pos_node: FloatTensor，形状为 (N, 3)，当前节点局部坐标，单位 Å。
            - h_edge: FloatTensor|None，形状为 (E, D_e)，当前分子内有向边特征；口袋 node-only 编码传 None。
            - edge_index: LongTensor，形状为 (2, E)，每列为目标节点与来源节点编号，数值索引 ``h_node`` 第一维。
            - node_extra: FloatTensor|None，形状为 (N, D_gn)，逐节点 fixed prompt；口袋编码传 None。
            - edge_extra: FloatTensor|None，形状为 (E, D_ge)，逐有向边 fixed prompt；口袋编码传 None。
            - batch_node: LongTensor|None，形状为 (N,)，每个目标节点的图归属编号；上下文 kNN 使用。
            - h_ctx: FloatTensor|None，形状为 (P, D_c)，编码后的口袋上下文特征。
            - pos_ctx: FloatTensor|None，形状为 (P, 3)，与 ``pos_node`` 同原点的口袋局部坐标，单位 Å。
            - batch_ctx: LongTensor|None，形状为 (P,)，每个口袋上下文节点的图归属编号。

        返回值:
            - h_node: FloatTensor，形状为 (N, D_n)，``num_blocks`` 层后的节点表示。
            - pos_node: FloatTensor，形状为 (N, 3)，``node_only=False`` 时更新后的节点局部坐标，单位 Å。
            - h_edge: FloatTensor，形状为 (E, D_e)，``node_only=False`` 时更新后的分子内有向边表示。

        返回分支:
            - ``node_only=True`` 只返回 ``h_node``。
            - ``node_only=False`` 返回三元组 ``(h_node, pos_node, h_edge)``。
        """

        # ``i``：int，当前联合更新 block 的 0-based 层号，索引各个长度为 L 的 ModuleList。
        for i in range(self.num_blocks):
            # # remake edge fetures (distance have been changed in each iteration)
            if (i==0) or (not self.node_only):
                # ``h_dist``：FloatTensor，形状为 (E, R)，逐分子内有向边的距离径向基特征。
                # ``relative_vec``：FloatTensor，形状为 (E, 3)，逐边从来源端指向目标端的相对向量，单位 Å。
                # ``distance``：FloatTensor，形状为 (E,)，逐边欧氏距离，单位 Å。
                h_dist, relative_vec, distance = self._build_edges_dist(pos_node, edge_index)
            if not self.node_only:
                # ``h_edge``：[E,D_e] + [E,R] -> [E,D_e+R]，只拼接特征维，保持有向边次序。
                h_edge = torch.cat([h_edge, h_dist], dim=-1)
            else:
                # ``h_edge``：Tensor，形状为 (E, R)；node-only 口袋编码没有旧边特征，只使用首次距离基。
                h_edge = h_dist
            # ``h_edge``：Tensor，形状为 (E, D_e)；第 i 层专属线性层把当前边表示映射回统一边宽度。
            h_edge = self.edge_embs[i](h_edge)
            
            # # edge with context
            if h_ctx is not None:
                # ``h_ctx_edge``：FloatTensor，形状为 (C, R_c)，逐配体—口袋边的距离径向基特征。
                # ``vec_ctx``：FloatTensor，形状为 (C, 3)，逐上下文边从口袋端指向配体端的相对向量，单位 Å。
                # ``dist_ctx``：FloatTensor，形状为 (C,)，逐上下文边欧氏距离，单位 Å。
                # ``ctx_knn_edge_index``：LongTensor，形状为 (2, C)，第一行索引配体节点，第二行索引口袋节点。
                h_ctx_edge, vec_ctx, dist_ctx, ctx_knn_edge_index = self._build_context_edges_dist(
                    pos_node, pos_ctx, batch_node, batch_ctx)
                # ``h_ctx_edge``：Tensor，形状为 (C, D_ce)；将上下文距离 RBF 映射到 ContextNodeBlock/PosUpdate 使用的边宽度。
                h_ctx_edge = self.ctx_edge_embs[i](h_ctx_edge)
            else:
                # ``ctx_knn_edge_index``：None，无口袋上下文时显式传给 ContextNodeBlock 的边端点占位。
                ctx_knn_edge_index = None
                # ``h_ctx_edge``：None，无口袋上下文时显式传入的上下文边特征占位。
                h_ctx_edge = None

            # # node feature updates
            h_node = self.node_blocks_with_edge[i](h_node, edge_index, h_edge, node_extra,
                                        h_ctx, ctx_knn_edge_index, h_ctx_edge)
            if self.node_only:
                continue
            
            # # edge feature updates
            h_edge = self.edge_blocks[i](h_edge, edge_index, h_node, edge_extra)

            # # pos updates
            # ``pos_node``：[N,3] + [N,3] -> [N,3]；内部边坐标增量由配体相对向量生成，单位 Å。
            pos_node = pos_node + self.pos_blocks[i](h_node, h_edge, edge_index, relative_vec, distance, node_extra, edge_extra)
            if h_ctx is not None:
                # ``pos_node``：[N,3] + [N,3] -> [N,3]；上下文增量由“配体目标 - 口袋来源”相对向量生成。
                pos_node = pos_node + self.ctx_pos_blocks[i](
                    h_node, h_ctx_edge, ctx_knn_edge_index, vec_ctx, dist_ctx, node_extra,
                    edge_extra=None, h_node_right=h_ctx)

        if self.local_update:
            pos_node = pos_node + self.local_net(h_node, pos_node, h_edge, edge_index, batch_node)

        if self.node_only:
            return h_node
        else:
            return h_node, pos_node, h_edge

    def _build_edges_dist(self, pos, edge_index):
        """
        为固定有向边计算相对坐标、欧氏距离和径向基特征。

        输入参数:
            - pos: (N, 3), 节点局部坐标，最后一维按 XYZ 排列，单位 Å。
            - edge_index: int64, (2, E), 每列 ``[目标节点, 来源节点]``。

        返回值:
            - h_dist: (E, R), 每条边距离的高斯径向基响应。
            - relative_vec: (E, 3), ``pos[目标] - pos[来源]``，单位 Å；整体平移不改变该向量。
            - distance: (E,), 相对向量的欧氏长度，单位 Å。
        """
        # ``relative_vec``：Tensor，形状为 (E, 3)；每条边从来源节点指向目标节点的相对坐标向量，单位 Å。
        relative_vec = pos[edge_index[0]] - pos[edge_index[1]]
        # ``distance``：Tensor，形状为 (E,)；沿 XYZ 维求二范数得到标量边长，单位 Å。
        distance = torch.norm(relative_vec, dim=-1, p=2)
        # ``h_dist``：Tensor，形状为 (E, R)；只从旋转/平移不变量 distance 生成的径向基特征。
        h_dist = self.distance_expansion(distance)
        return h_dist, relative_vec, distance
    
    def _build_context_edges_dist(self, pos, pos_ctx, batch_node, batch_ctx):
        """
        在同一图内部连接每个配体原子的口袋近邻，并计算上下文边几何量。

        输入参数:
            - pos: (N, 3), 配体原子局部坐标，单位 Å。
            - pos_ctx: (P, 3), 口袋原子局部坐标，与 pos 使用同一原点和 XYZ 轴，单位 Å。
            - batch_node: int64, (N,), 每个配体原子所属图编号。
            - batch_ctx: int64, (P,), 每个口袋原子所属图编号。

        返回值:
            - h_dist: (C, R_c), 分子—口袋距离的高斯径向基响应。
            - relative_vec: (C, 3), ``配体目标坐标 - 口袋来源坐标``，单位 Å。
            - distance: (C,), 分子—口袋欧氏距离，单位 Å。
            - ctx_knn_edge_index: int64, (2, C), 每列 ``[配体目标原子编号, 口袋来源原子编号]``，两行分别索引 pos 与 pos_ctx 第一维。

        关键分支:
            - ``knn < 100``: 每个配体原子在同图口袋中取 k 个近邻；``downsample_context`` 只扰动近邻搜索坐标，不扰动返回距离。
            - ``knn >= 100``: 对每个图显式构造配体—口袋全连接笛卡尔积。
            - P=0: 构象生成依赖当前 PyG/torch-cluster 后端为 kNN 返回形状 ``(2,0)`` 的空边；本函数没有单独跳过空上下文。
        """
        if self.knn < 100:
            if self.downsample_context:
                # ``pos_ctx_noised``：Tensor，形状为 (P, 3)；仅用于选择近邻的口袋坐标；逐分量加入标准差 5 Å 的高斯噪声。
                pos_ctx_noised = pos_ctx + torch.randn_like(pos_ctx) * 5  # works like masked position information
            else:
                # ``pos_ctx_noised``：Tensor，形状为 (P, 3)；不扰动时与 pos_ctx 共享同一张量对象。
                pos_ctx_noised = pos_ctx
            # ``ctx_knn_edge_index``：LongTensor，形状为 (2, C)；knn(y=配体,x=口袋) 的第一行索引配体目标，第二行索引口袋近邻来源。
            ctx_knn_edge_index = knn(y=pos, x=pos_ctx_noised, k=self.knn,
                                    batch_x=batch_ctx, batch_y=batch_node)
        else: # fully connected x-yf
            # ``device``：torch.device，显式全连接端点张量创建在配体坐标设备上。
            device = pos.device
            # ``ctx_knn_edge_index``：list[Tensor]，第 b 项为图 b 的 ``(2, N_b*P_b)`` 配体—口袋端点。
            ctx_knn_edge_index = []
            # ``cum_node``：int，当前图在拼接配体原子数组中的起始偏移。
            cum_node = 0
            # ``cum_ctx``：int，当前图在拼接口袋坐标数组中的起始偏移。
            cum_ctx = 0
            # ``i_batch``：int，当前图号，取值范围为 ``[0, max(batch_ctx)]``，同时筛选配体与口袋实体。
            for i_batch in range(batch_ctx.max()+1):
                # ``num_ctx``：scalar int64 Tensor，图 i_batch 的口袋原子数 P_b。
                num_ctx = (batch_ctx==i_batch).sum()
                # ``num_node``：scalar int64 Tensor，图 i_batch 的配体原子数 N_b。
                num_node = (batch_node==i_batch).sum()
                # ``ctx_knn_edge_index_this``：LongTensor，形状为 (2, N_b*P_b)；meshgrid 枚举当前图所有配体目标—口袋来源组合并加批次偏移。
                ctx_knn_edge_index_this = torch.stack(
                    torch.meshgrid(
                        torch.arange(num_node, device=device) + cum_node,
                        torch.arange(num_ctx, device=device) + cum_ctx,
                    )).view(2, -1)
                # scalar，累加本图配体数，得到下一图配体端点全局偏移。
                cum_node += num_node
                # scalar，累加本图口袋数，得到下一图口袋端点全局偏移。
                cum_ctx += num_ctx
                ctx_knn_edge_index.append(ctx_knn_edge_index_this)
            # ``ctx_knn_edge_index``：list[(2, C_b)] -> (2, C)，沿边维拼接各图端点，图间没有交叉边。
            ctx_knn_edge_index = torch.cat(ctx_knn_edge_index, dim=-1)

        # ``relative_vec``：Tensor，形状为 (C, 3)；每条上下文边从口袋来源指向配体目标的相对坐标，使用未扰动口袋位置，单位 Å。
        relative_vec = pos[ctx_knn_edge_index[0]] - pos_ctx[ctx_knn_edge_index[1]]
        # ``distance``：Tensor，形状为 (C,)；上下文边欧氏距离，单位 Å。
        distance = torch.norm(relative_vec, dim=-1, p=2)
        # ``h_dist``：Tensor，形状为 (C, R_c)；上下文距离的高斯径向基响应。
        h_dist = self.dist_exp_ctx(distance)
        return h_dist, relative_vec, distance, ctx_knn_edge_index
        


class PosUpdate(Module):
    """
    用每条边预测的标量权重缩放相对坐标方向，并归约为目标节点的 E(3) 等变位移。

    形状符号:
        - N: 待更新左节点数。
        - E: 有向边数。
        - D_n: 左节点特征宽度。
        - D_r: 右节点特征宽度。
        - D_e: 边特征宽度。
        - D_ge: 边 prompt 特征宽度；当前节点 prompt 宽度被实现固定为 2。

    构造参数:
        - node_dim: int, 待更新左/目标节点特征宽度 D_n。
        - edge_dim: int, 有向边特征宽度 D_e。
        - hidden_dim: int, MLP 隐藏宽度。
        - gate_dim: int, ``node_extra`` 与可选 ``edge_extra`` 拼接后的实际宽度；分子内边为 4，上下文边为 2。
        - node_dim_right: int|None, 右/来源节点特征宽度 D_r；None 时等于 node_dim，口袋上下文更新时取 context_dim。

    前向输入:
        - h_node: (N, D_n), 待更新目标节点特征。
        - h_edge: (E, D_e), 有向边特征。
        - edge_index: int64, (2, E), 每列 ``[左/目标节点, 右/来源节点]``。
        - relative_vec: (E, 3), ``目标坐标 - 来源坐标``，单位 Å。
        - distance: (E,), ``relative_vec`` 的欧氏长度，单位 Å。
        - node_extra: (N, 2), 目标节点 fixed prompt；宽度 2 既由 ``pos_scale_net`` 输入层硬编码，也必须与上游 prompt 对齐。
        - edge_extra: (E, D_ge)|None, 有向边 fixed prompt；上下文边更新为 None。
        - h_node_right: (P, D_r)|None, 可选另一实体集合的来源节点特征；None 时来源也从 h_node 读取。

    前向输出:
        - delta_pos: (N, 3), 每个目标节点的坐标增量，单位 Å；节点顺序与 h_node 第一维一致。

    等变性:
        - MLP 只从节点/边标量特征产生 ``weight_edge`` 和 ``pos_scale`` 标量。
        - 唯一方向量是相对坐标 ``relative_vec``；整体旋转会同步旋转输出位移，整体平移不会改变位移。

    实现约束:
        - ``pos_scale_net`` 的输入宽度固定为 ``node_dim + 1 + 2``，因此 ``node_extra`` 不能是任意 ``D_gn``；当前实现要求恰有 2 个 fixed prompt 通道。
    """
    def __init__(self, node_dim, edge_dim, hidden_dim, gate_dim, node_dim_right=None):
        super().__init__()
        # ``self.left_lin_edge``：MLP；[E, D_n] -> [E, D_n] 的目标节点逐边投影。
        self.left_lin_edge = MLP(node_dim, node_dim, hidden_dim)
        # ``node_dim_right``：int D_r，来源节点宽度；内部边缺省与目标节点宽度相同。
        node_dim_right = node_dim if node_dim_right is None else node_dim_right
        # ``self.right_lin_edge``：MLP；[E, D_r] -> [E, D_n] 的来源节点逐边投影。
        self.right_lin_edge = MLP(node_dim_right, node_dim, hidden_dim)
        # ``self.edge_lin``：BondFFN，融合边、双端点与 prompt，输出每条边一个有符号标量权重。
        self.edge_lin = BondFFN(edge_dim, node_dim*2, node_dim, gate_dim, out_dim=1)
        # ``self.pos_scale_net``：Sequential，按 D_n 维节点特征、固定 2 维节点 prompt 与 1 维位移范数预测形状为 (N, 1) 的 0..1 节点缩放。
        self.pos_scale_net = nn.Sequential(MLP(node_dim+1+2, 1, hidden_dim), nn.Sigmoid())

    def forward(self, h_node, h_edge, edge_index, relative_vec, distance, node_extra, edge_extra=None, h_node_right=None):
        """从标量特征与相对方向构造 E(3) 等变节点位移。

        输入参数:
            - h_node: FloatTensor，形状为 (N, D_n)，待更新目标节点特征。
            - h_edge: FloatTensor，形状为 (E, D_e)，与有向边逐行对齐的边特征。
            - edge_index: LongTensor，形状为 (2, E)，每列为目标节点、来源节点。
            - relative_vec: FloatTensor，形状为 (E, 3)，目标坐标减来源坐标的相对向量，单位 Å。
            - distance: FloatTensor，形状为 (E,)，``relative_vec`` 的欧氏长度，单位 Å。
            - node_extra: FloatTensor，形状为 (N, 2)，逐目标节点 fixed prompt。
            - edge_extra: FloatTensor|None，形状为 (E, D_ge)，逐有向边 fixed prompt。
            - h_node_right: FloatTensor|None，形状为 (P, D_r)，另一来源实体集合的节点特征；None 时来源也读取 ``h_node``。

        返回值:
            - delta_pos: FloatTensor，形状为 (N, 3)，按目标节点归约并缩放后的坐标增量，单位 Å。
        """
        # ``edge_index_left``：LongTensor，形状为 (E,)，逐边目标节点编号，索引 ``h_node`` 第一维。
        # ``edge_index_right``：LongTensor，形状为 (E,)，逐边来源节点编号，索引 ``h_node_right`` 第一维。
        edge_index_left, edge_index_right = edge_index

        # ``left_feat``：Tensor，形状为 (E, D_n)；目标节点特征经共享 MLP 投影，保持边顺序。
        left_feat = self.left_lin_edge(h_node[edge_index_left])
        # ``h_node_right``：(N, D_n) 或 (P, D_r)，None 时内部边两端共用同一节点特征表。
        h_node_right = h_node if h_node_right is None else h_node_right
        # ``right_feat``：Tensor，形状为 (E, D_n)；来源节点特征投影到与目标相同的宽度。
        right_feat = self.right_lin_edge(h_node_right[edge_index_right])
        # ``both_extra``：Tensor，形状为 (E, 2)；为每条边选择其目标节点的固定 2 维 prompt。
        both_extra = node_extra[edge_index_left]
        if edge_extra is not None:
            # ``both_extra``：[E,D_gn] + [E,D_ge] -> [E,D_gn+D_ge]，只拼接 prompt 通道。
            both_extra = torch.cat([both_extra, edge_extra], dim=-1)
        # ``weight_edge``：Tensor，形状为 (E, 1)；每条边的有符号标量作用强度；节点双端特征在末维拼成 2D_n。
        weight_edge = self.edge_lin(h_edge,
                            torch.cat([left_feat, right_feat], dim=-1),
                            both_extra)
        
        # ``force_edge``：Tensor，形状为 (E, 3)；标量权重乘单位相对方向，再乘 5/(distance+5) 抑制远距离作用；数值按坐标增量解释，常数 5 使用 Å 距离尺度。
        force_edge = weight_edge * relative_vec / (distance.unsqueeze(-1) + 1e-6) / (distance.unsqueeze(-1) + 5.) * 5
        # ``delta_pos``：[E,3] -> [N,3]，按 edge_index_left 对所有入边作用求和；没有入边的节点得到零向量。
        delta_pos = scatter_sum(force_edge, edge_index_left, dim=0, dim_size=h_node.shape[0])
        # ``delta_pos``：Tensor，形状为 (N, 1)；节点级缩放输入依次为节点特征、fixed prompt 和当前位移范数；sigmoid 输出 0..1。
        delta_pos = delta_pos * self.pos_scale_net(torch.cat([h_node, node_extra,
                                        torch.norm(delta_pos, dim=-1, keepdim=True)], dim=-1))
        return delta_pos

class LocalPosUpdate(Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, cutoff=3.5):
        super().__init__()
        self.cutoff = cutoff
        self.dist_exp = GaussianSmearing(stop=cutoff, num_gaussians=16)
        self.dist_lin = MLP(edge_dim+16, edge_dim, edge_dim)
        self.node_lin = MLP(node_dim*3, node_dim, hidden_dim)
        self.edge_lin = MLP(edge_dim*3, edge_dim, edge_dim)
        self.weight_lin = MLP(node_dim+edge_dim, 3, node_dim)
    def forward(self, h_node, node_pos, h_edge, edge_index, batch_node):
        # find only edges within a threshold
        is_short_edge, vector, distance = self._find_short_edges(node_pos, edge_index)
        # drop nodes with no short edges
        node_degree = scatter_sum(is_short_edge.long(), edge_index[0], dim=-1)
        is_short_node = (node_degree > 1)
        is_short_edge = is_short_edge & (is_short_node[edge_index[0]]) & (is_short_node[edge_index[1]])
        short_edge_index = edge_index[:, is_short_edge]

        node_triples = []
        for node in torch.argwhere(is_short_node).flatten():
            node_neighs = short_edge_index[1, short_edge_index[0] == node]
            node_pairs = permutations(node_neighs, 2)
            node_triples.extend([torch.stack([node, pair[0], pair[1]]) for pair in node_pairs])
        node_triples_tensor = torch.tensor(node_triples)
        node_triples_stack = torch.stack(node_triples)
        
        edge_ids = [
            edge_index_to_index_of_edge(node_triples[:, 0:2].T, batch_node),
            edge_index_to_index_of_edge(node_triples[:, [0,2]].T, batch_node),
            edge_index_to_index_of_edge(node_triples[:, 1:3].T, batch_node)
        ]
        
        # feats
        # h_node = self.node_lin(h_node)
        dist_feat = self.dist_exp(distance)
        h_edge = self.dist_lin(torch.cat([h_edge, dist_feat], dim=-1))
        
        h_node_triplets = self.node_lin(torch.cat(h_node[node_triples], dim=-1))
        h_edge_triplets = self.edge_lin(torch.cat(h_edge[edge_ids], dim=-1))
        weight = self.weight_lin(torch.cat([h_node_triplets, h_edge_triplets], dim=-1))
        
        vector_triplets = vector[edge_ids[:2]]
        vector_cross = torch.linalg.cross(vector_triplets[0], vector_triplets[1], dim=-1)
        vector_triplets = torch.cat([vector_triplets, vector_cross], dim=-1)
        delta_pos = (weight * vector_triplets).sum(dim=1)
        delta_pos = scatter_sum(delta_pos, node_triples[:, 0], dim=0, dim_size=h_node.shape[0])
        
        return delta_pos

    def _find_short_edges(self, node_pos, edge_index):
        vec = node_pos[edge_index[0]] - node_pos[edge_index[1]]
        distance = torch.norm(vec, dim=-1)
        unit = vec / (distance.unsqueeze(-1) + 1e-8)
        is_short_edge = distance < self.cutoff
        return is_short_edge, unit, distance



class RigidNet(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, cutoff, use_gate):
        super().__init__()
        edge_dim = edge_dim // 2
        hidden_dim = hidden_dim // 2
        self.distance_expansion = GaussianSmearing(start=0, stop=cutoff, num_gaussians=edge_dim)
        self.domain_net = NodeBlock(node_dim, edge_dim, hidden_dim, use_gate, is_heter=True)
        self.translation_net = MLP(node_dim*2+edge_dim, 1, hidden_dim, 2)
        self.torque_net = MLP(node_dim*2+3, 1, hidden_dim, 2)
        self.angle_net = nn.Sequential(
            MLP(node_dim+1, 1, hidden_dim//2, 2),
            nn.Sigmoid(),
        )
    
    def forward(self, h_node, pos_node, delta_pos,
                domain_node_index_0, domain_node_index_1):
        
        h_node_in_domain = h_node[domain_node_index_1]
        pos_node_in_domain = pos_node[domain_node_index_1]
        delta_pos_in_domain = delta_pos[domain_node_index_1]
        
        # # domain feature
        n_domain = domain_node_index_0.max() + 1
        pos_domain = scatter_mean(pos_node_in_domain, index=domain_node_index_0,
                                  dim=0, dim_size=n_domain)[domain_node_index_0]
        radius_vec = pos_node_in_domain - pos_domain
        dist_domain = torch.norm(radius_vec, dim=-1, p=2)
        h_edge = self.distance_expansion(dist_domain)
        h_domain = self.domain_net(h_node, torch.stack([domain_node_index_0, domain_node_index_1], dim=0),
                                   h_edge, torch.zeros_like(h_node[..., 0:1]))
        
        # # domain translation
        translation_weight = self.translation_net(torch.cat([
            h_domain[domain_node_index_0], h_node_in_domain, h_edge], dim=-1))
        force_edge = translation_weight * delta_pos_in_domain
        translation_domain = scatter_mean(force_edge, index=domain_node_index_0, dim=0, dim_size=n_domain)
        translation_domain = translation_domain[domain_node_index_0]

        # # torque for each node (for node in domain)
        torque = torch.linalg.cross(radius_vec, delta_pos_in_domain, dim=-1)
        h_torque = torch.cat([h_node_in_domain, h_domain[domain_node_index_0], 
                              dist_domain[..., None],
                              torch.norm(delta_pos_in_domain, dim=-1, p=2, keepdim=True),
                              torch.norm(torque, dim=-1, p=2, keepdim=True),
                              ], dim=-1)
        scalar = self.torque_net(h_torque)
        scaled_torque = torque * scalar
        
        # torque for each domain
        torque_domain = scatter_mean(scaled_torque, index=domain_node_index_0, dim=0, dim_size=n_domain)

        # get rotatio axis (unit vector of torque) and rotation angle
        torque_norm = torch.norm(torque_domain, dim=-1, p=2, keepdim=True)
        rot_axis = torque_domain / torque_norm
        rot_angle = self.angle_net(torch.cat([h_domain, torque_norm], dim=-1)) * torch.pi
        # NOTE: can prdict cos and sin for stability in the feature (like AF2)
        # apply rotation
        pos_update = pos_domain + translation_domain + apply_axis_angle_rotation(
            radius_vec, rot_axis[domain_node_index_0], rot_angle[domain_node_index_0])
        
        pos_out = pos_node + delta_pos
        pos_out[domain_node_index_1] = pos_update
        return pos_out, rot_axis, rot_angle
        

class TorsionNet(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, cutoff, use_gate):
        super().__init__()
        self.radius_net = GaussianSmearing(start=0, stop=cutoff, num_gaussians=edge_dim)
        self.torque_net = MLP(node_dim*2+edge_dim*3+3, 1, hidden_dim, 2)

        self.torsional_left_net = NodeBlock(node_dim, edge_dim, hidden_dim//2, use_gate, )
        self.angle_net = nn.Sequential(
            MLP(node_dim+1, 1, hidden_dim//2, 2),
            nn.Sigmoid(),
        )
    
    def forward(self, h_node, pos_node, force,
                h_edge, edge_index, 
                torsional_edge_anno, twisted_edge_anno):
        """
        h_node: (N, node_dim)
        pos_node: (N, 3)
        force [delta_pos]: (N, 3)
        h_edge: (E, edge_dim)
        edge_index: (2, E)
        torsional_edge_anno: (2, n_tor)
        twisted_edge_anno: (2, T)
        """
        
        i_bond_for_tor_edge, tor_edge =torsional_edge_anno
        i_tor_edge_for_twisted_edge, twisted_edge = twisted_edge_anno
        
        # # fetch torsional bond/node features and vec
        tor_left, tor_right = edge_index[:, tor_edge]
        h_tor_edge = h_edge[tor_edge]  # (n_tor, edge_dim)
        h_tor_left = h_node[tor_left]  # (n_tor, node_dim)
        h_tor_right = h_node[tor_right]  # (n_tor, node_dim)
        vec_tor_bond = pos_node[tor_left] - pos_node[tor_right]  # (n_tor, 3)
        len_tor_bond = torch.norm(vec_tor_bond, dim=-1, p=2, keepdim=True)  # (n_tor, 1)
        unit_tor_bond = vec_tor_bond / (len_tor_bond + 1e-6)  # (n_tor, 3)
        unit_tor_bond_expand = unit_tor_bond[i_tor_edge_for_twisted_edge]  #  (T, 3)
        
        # # fetch twisted node and edge feature
        twisted_node, tor_end = edge_index[:, twisted_edge]
        assert (tor_end == tor_left[i_tor_edge_for_twisted_edge]).all(), "torsional end must be the same as torsional left"
        h_twisted_edge = h_edge[twisted_edge]  # (T, edge_dim)
        h_twisted_node = h_node[twisted_node]  # (T, node_dim)
        force_twisted_node = force[twisted_node]  # (T, 3)
        vec_twisted_edge = pos_node[twisted_node] - pos_node[tor_end]  # (T, 3)
        
        # # calculate torque
        vec_radius = vec_twisted_edge - torch.sum(vec_twisted_edge * unit_tor_bond_expand, dim=-1, keepdim=True) * unit_tor_bond_expand
        len_radius = torch.norm(vec_radius, dim=-1, p=2)
        h_radius = self.radius_net(len_radius)
        force_tangent = force_twisted_node - torch.sum(force_twisted_node * unit_tor_bond_expand, dim=-1, keepdim=True) * unit_tor_bond_expand
        torque = torch.linalg.cross(vec_radius, force_tangent, dim=-1)  # (T, 3)
        
        # # calculate torque weight
        h_torque = torch.cat([h_twisted_node, h_twisted_edge, h_tor_left[i_tor_edge_for_twisted_edge],
                              h_tor_edge[i_tor_edge_for_twisted_edge],
                              h_radius, 
                              torch.norm(force[twisted_node], dim=-1, p=2, keepdim=True),
                              torch.norm(force_tangent, dim=-1, p=2, keepdim=True),
                              torch.norm(torque, dim=-1, p=2, keepdim=True),], dim=-1)
        torque_weight = self.torque_net(h_torque)  # (T, 1)
        torque = torque * torque_weight
        
        # # aggregate torque to calculate angles
        torque_tor = scatter_mean(torque, index=i_tor_edge_for_twisted_edge, dim=0)  # (n_tor, 3)
        len_torque = torch.norm(torque_tor, dim=-1, p=2, keepdim=True)  # (n_tor, 1)
        assert torch.linalg.cross(torque_tor, unit_tor_bond, dim=-1).abs().max() < 1e-2, "torque must be in parallel with torsional bond"
        h_node = self.torsional_left_net(h_node, edge_index[:, twisted_edge].flip(0),
                                        h_edge[twisted_edge], torch.zeros_like(h_node[..., 0:1]))
        h_node_tor = h_node[tor_left]
        angles = self.angle_net(torch.cat([
            len_torque, h_node_tor,
        ], dim=-1)) * torch.pi
        direction = torch.sum(torque_tor * unit_tor_bond, dim=-1, keepdim=True)
        angles = angles * torch.sign(direction) # (n_tor, 1)
        # angles = angles[tor_left]
        
        # # apply rotation
        pos_update = apply_torsional_rotation_multiple_domains(
            pos_node, edge_index, 
            tor_edge, angles, i_bond_for_tor_edge,
            twisted_edge, i_tor_edge_for_twisted_edge
        )
        return pos_update, angles

