"""
实现 PocketXMol 构象生成与小分子 docking 共用的配体去噪模型.

从 PMAsymDenoiser.forward 阅读主流程: PyG Batch 的带噪配体、fixed 条件标记和口袋图先成为隐藏特征, 再由原去噪网络更新配体原子、相互作用边及坐标.
口袋编码保持官方蛋白入口; 可选 RA 在蛋白与核酸联合图中共享编码器.
配置 model.density 时, 另将固定密度裁块编码为体素特征, 在配体去噪网络的各层读取该特征.
本模块不落盘, 返回原子和半边的分类原始分数 logits、去噪坐标以及可选的原子类别、位置和半边置信度分数, 各字段形状见类 Docstring.
本模块不读取时间步或任务名称, 构象与 docking 的任务条件由 fixed_*、带噪输入和口袋字段表达.
"""

# Third-party imports

import torch
from easydict import EasyDict
from torch.nn import Module
from torch.nn import functional as F
from tqdm import tqdm

# Local imports
from models.graph import NodeEdgeNet
from models.graph_context import ContextNodeEdgeNet
from models.graph_gvp import ContextNodeEdgeNetGVP
from models.ipa import ContextGAEdgeNet, GAEncoder

from models.common import *
from models.corrector import correct_pos, get_dihedral_batch
from models.diffusion import *


class PMAsymDenoiser(Module):
    """
    用同一非时间条件网络预测干净原子类别、坐标和半边类别.

    当前 reduced 配置的 docking + free 路径:
        - ``model.name=pm_asym_denoiser`` 只实例化本类; ``gvp``、``pocket.name`` 和 ``denoiser.name`` 均缺省, 因而口袋编码器与配体去噪骨干都选择 ``ContextNodeEdgeNet``.
        - 口袋实例以 ``node_only=True`` 执行 4 个节点更新块; 配体实例执行 6 个节点-边-坐标联合更新块, 每层通过 32-NN 口袋上下文更新 pose.
        - ``free`` 不选择另一神经网络; 它只让 noiser 对坐标加入逐原子 Gaussian 噪声, 并传入 ``fixed_node=1, fixed_pos=0, fixed_halfedge=1, fixed_halfdist=0`` prompt. 网络仍输出原子、坐标、半边和 confidence 预测, 采样器只把 ``pred_pos`` 写回下一步.

    形状符号:
        - B: PyG 批次中的分子图数量.
        - N: 批次中配体原子总数.
        - H: 批次中无向完全图半边总数.
        - E=2H: 为消息传递复制正反方向后的有向边总数.
        - P: 批次中口袋原子总数.
        - E_p: 批次中口袋 kNN 有向边总数.

    构造参数:
        - config.pocket_dim: int, 编码后口袋节点宽度.
        - config.nucleic_branch: None|str, None 保持官方蛋白编码; RA 在联合口袋图中共享编码器.
        - config.density: 可选映射, 存在时构造密度编码器和逐层原子读出; 字段由 DensityEncoder、DensityReadout 读取, 缺省不增加密度参数或依赖.
        - config.node_dim: int, 拼接原子 Embedding、两维 fixed prompt 和附加节点特征后的总宽度.
        - config.edge_dim: int, 拼接半边 Embedding 与两维 fixed prompt 后的总宽度.
        - config.addition_node_features: list[str], 追加到节点表示的逐原子标量字段名; reduced 配置只含 ``is_peptide``.
        - config.add_output: list[str], 额外输出头名称; 含 ``confidence`` 时建立三种置信度头.
        - config.gvp: bool, True 时改用 GVP 口袋/去噪骨干; 缺省 False 使用 ``ContextNodeEdgeNet``.
        - config.pocket.name: str, ``default`` 选择 ContextNodeEdgeNet, ``ipa`` 选择 GAEncoder; 缺省 ``default``.
        - config.pocket: 映射, 完整展开传给口袋编码器; ``node_only=True`` 由本类额外传入.
        - config.denoiser.name: str, ``default`` 选择 ContextNodeEdgeNet, ``ipa`` 选择 ContextGAEdgeNet; 缺省 ``default``.
        - config.denoiser: 映射, 完整展开传给配体去噪骨干, 并额外传 ``context_dim=pocket_dim``.
        - num_node_types: int, 原子词表大小; 当前包括配置元素类别和可选 mask 类.
        - num_edge_types: int, 半边词表大小; 0 为非键、1..4 为真实键, 末端可含 mask 类.
        - pocket_in_dim: int, ``pocket_atom_feature`` 宽度; 默认元素/氨基酸/主链编码共 25.

    前向批次字段:
        - node_in: int64, (N,), 带噪原子类别; 数值索引 ``nodetype_embedder`` 词表.
        - pos_in: (N, 3), 带噪配体局部坐标; 最后一维按 XYZ 排列, 单位 Å.
        - halfedge_in: int64, (H,), 带噪无向半边类别; 与 ``halfedge_index`` 第二维对齐.
        - halfedge_index: int64, (2, H), 完全图上三角端点; 每列满足 ``i<j``, 数值索引配体原子第一维.
        - fixed_node: LongTensor|BoolTensor, 形状为 (N,), 1 表示原子类别是当前任务给定条件.
        - fixed_pos: LongTensor|BoolTensor, 形状为 (N,), 1 表示坐标是当前任务给定条件.
        - fixed_halfedge: LongTensor|BoolTensor, 形状为 (H,), 1 表示半边类别是当前任务给定条件.
        - fixed_halfdist: LongTensor|BoolTensor, 形状为 (H,), 1 表示端点距离是当前任务给定条件.
        - node_type_batch: int64, (N,), 每个配体原子所属图编号, 范围 ``[0,B-1]``.
        - pocket_atom_feature: (P, pocket_in_dim), 原 25 维蛋白元素、氨基酸与主链特征; RA 中核酸原子的全部通道为零.
        - pocket_is_nucleic: bool, (P,), 仅 RA 读取; True 是标准 RNA/DNA 原子, False 是标准蛋白原子, 与 pocket_pos 第一维对齐.
        - pocket_nucleic_feature: (P, 15), 仅 RA 读取; 0:4 是 C/N/O/P 元素编码, 4:12 是 A/C/G/U/DA/DC/DG/DT 核苷酸编码, 12:15 依次表示碱基 base、糖基 sugar、磷酸基 phosphate; 蛋白原子全零.
        - pocket_pos: (P, 3), 与 pos_in 使用同一局部原点的口袋坐标, 单位 Å.
        - pocket_knn_edge_index: int64, (2, E_p), 有向 kNN 边端点索引 pocket_pos 第一维; RA 使用蛋白与核酸联合图, 批内不同实例由 PyG 图编号隔离; 同一口袋内不同链及蛋白与核酸之间均可连边.
        - pocket_pos_batch: int64, (P,), 每个口袋原子所属图编号.
        - density_input: float32, (B, 56, 48, 48, 48), 仅密度模型读取; 固定裁块的全部56通道, 空间轴 ZYX.
        - density_origin: float32, (B, 3), 实际裁块角点的局部 XYZ 坐标, 单位 Å, 与 pos_in 共用原点.
        - density_basis: float32, (B, 3, 3), 三行依次为源 X、Y、Z 一个体素步长的局部向量, 单位 Å.
        - is_peptide: 0/1, (N,), 小分子构象/docking 为全 0; 若配置不请求该附加特征则不读取.

    前向输出字段:
        - pred_node: (N, num_node_types), 每个配体原子的干净类别 logits; 未做 softmax.
        - pred_pos: (N, 3), 每个配体原子的去噪局部坐标, 原点与 pos_in 相同, 单位 Å.
        - pred_halfedge: (H, num_edge_types), 每条无向半边的干净类别 logits; 正反向隐藏特征先求和再解码.
        - confidence_node: (N, 1), 原子主分类预测是否正确的二分类 logit, 未做 sigmoid.
        - confidence_pos: (N, 1), 原始位置置信度输出, 与配体原子第一维对齐; 当前训练配置由 ConfidenceLoss 以 sigmoid 后的数值拟合 0.2 ** 原子坐标误差_Å, 不是误差小于1 Å的分类概率.
        - confidence_halfedge: (H, 1), 半边主分类预测是否正确的二分类 logit, 与无向半边第一维对齐.

    位置置信度的目标由外部 ConfidenceLoss 决定; 旧配置 prob_1A 不在(0,1)时直接回归负坐标误差, 本模型输出端不做 sigmoid.

    条件边界:
        - 本网络没有 ``task``、``task_setting``、扩散时间步或连续噪声等级输入; 条件来自 fixed prompt、带噪状态、口袋和明确配置的密度裁块.
        - 无口袋构象生成传入 P=0 的空张量; 口袋编码器和上下文 kNN 后端必须支持空边路径.
    """
    
    def __init__(self,
        config,
        num_node_types,
        num_edge_types,
        pocket_in_dim,
        **kwargs
    ):
        """建立原配体去噪网络, 按配置增加共享口袋编码的 RA 核酸投影和密度条件模块.

        输入参数及前向字段见类 Docstring. 原 pocket_embedder、pocket_encoder 和配体参数名保持不变.
        """
        super().__init__()
        # ``config.pocket_dim``: int, 编码后口袋节点宽度.
        # ``config.node_dim``: int, 拼接全部节点输入后的总宽度.
        # ``config.edge_dim``: int, 拼接全部半边输入后的总宽度.
        # ``config.addition_node_features``: list[str]|缺省, 按顺序追加的逐原子标量字段名.
        # ``config.add_output``: list[str]|缺省, 额外输出头名称; ``confidence`` 建立三个置信度字段.
        # ``config.gvp``: bool|缺省, 是否使用 GVP 版本骨干.
        # ``config.pocket.name``: str|缺省, 口袋编码器注册名.
        # ``config.pocket.edge_dim``: int, 口袋内部有向边隐藏宽度.
        # ``config.pocket.hidden_dim``: int, 口袋编码器节点隐藏宽度.
        # ``config.pocket.num_blocks``: int, 口袋内部消息传递层数.
        # ``config.pocket.dist_cfg.num_gaussians``: int, 口袋边距离基通道数.
        # ``config.denoiser.name``: str|缺省, 配体去噪骨干注册名.
        # ``config.denoiser.hidden_dim``: int, 配体节点隐藏宽度.
        # ``config.denoiser.num_blocks``: int, 配体-口袋联合更新层数.
        # ``config.denoiser.dist_cfg.stop``: float, 配体内部距离基最大中心, 单位 Å.
        # ``config.denoiser.dist_cfg.num_gaussians``: int, 配体内部距离基通道数.
        # ``config.denoiser.gate_dim``: int, 节点 fixed prompt/门控宽度.
        # ``config.denoiser.context_cfg.edge_dim``: int, 配体-口袋边隐藏宽度.
        # ``config.denoiser.context_cfg.knn``: int, 每个配体原子的口袋近邻数上限.
        # ``config.denoiser.context_cfg.dist_cfg.stop``: float, 上下文距离基最大中心, 单位 Å.
        # ``config.denoiser.context_cfg.dist_cfg.num_gaussians``: int, 上下文距离基通道数.
        # ``config.denoiser.context_cfg.dist_cfg.type_``: str|缺省, 距离基中心排布类型.
        # ``self.config``: EasyDict, 保留上述模型结构、输出头与附加节点字段.
        self.config = config
        # ``self.num_node_types``: int, 原子类别嵌入词表及分类头宽度.
        self.num_node_types = num_node_types
        # ``self.num_edge_types``: int, 半边类别嵌入词表及分类头宽度.
        self.num_edge_types = num_edge_types
        # ``gvp``: bool, 决定 default 名称解析到标量 ContextNodeEdgeNet 还是 GVP 版本.
        gvp = getattr(config, 'gvp', False)
        
        # Pocket encoder: processes protein context
        # ``pocket_dim``: int, 原始口袋 ``pocket_in_dim`` 维离散特征经线性层后的节点宽度.
        pocket_dim = config.pocket_dim
        # ``self.pocket_embedder``: Linear; [P, pocket_in_dim] -> [P, pocket_dim] 的口袋原始特征投影.
        self.pocket_embedder = nn.Linear(pocket_in_dim, pocket_dim)
        # ``pocket_name``: str, 口袋骨干类型; 当前 reduced 配置缺省为 default.
        pocket_name = getattr(config.pocket, 'name', 'default')
        if pocket_name == 'default':
            # ``pocket_encoder_bb``: type, 默认在标量 ContextNodeEdgeNet 与 GVP 版本之间按 gvp 开关选择.
            pocket_encoder_bb = ContextNodeEdgeNet if not gvp else ContextNodeEdgeNetGVP
        elif pocket_name == 'ipa':
            # ``pocket_encoder_bb``: type, IPA/几何注意力口袋编码器.
            pocket_encoder_bb = GAEncoder
        # ``self.pocket_encoder``: Module, 以 node_only=True 实例化; 返回 (P, pocket_dim) 且不更新口袋坐标.
        self.pocket_encoder = pocket_encoder_bb(pocket_dim, node_only=True, **config.pocket)
        # str|None, 缺省 None 是原模型兼容入口, 不增加核酸参数或读取核酸字段.
        self.nucleic_branch = config.get('nucleic_branch', None)
        if self.nucleic_branch not in (None, 'RA'):
            raise ValueError(f'Unknown nucleic_branch: {self.nucleic_branch}')
        if self.nucleic_branch is not None:
            # Linear, [P_na, 15] -> [P_na, pocket_dim]; P_na 是当前批次标准核酸原子数.
            self.nucleic_embedder = nn.Linear(15, pocket_dim)
        
        # Molecule embedding layers
        # ``self.addition_node_features``: list[str], 每个名称对应一个随后拼接的逐原子标量通道.
        self.addition_node_features = getattr(config, 'addition_node_features', [])
        # ``node_dim``: int, fixed prompt 与可选附加通道拼接完成后的节点总宽度.
        node_dim = config.node_dim
        # ``edge_dim``: int, 半边 Embedding 与两维 fixed prompt 拼接后的总宽度.
        edge_dim = config.edge_dim
        # ``node_emb_dim``: int, 原子类别 Embedding 宽度; 预留 2 个 fixed 通道及每个附加字段的 1 个通道.
        node_emb_dim = node_dim - 2 - len(self.addition_node_features)
        # ``self.nodetype_embedder``: Embedding, int64 (N,) -> float (N, node_emb_dim) 的原子类别查表.
        self.nodetype_embedder = nn.Embedding(num_node_types, node_emb_dim)
        # ``self.edgetype_embedder``: 半边类别 Embedding 宽度为 ``edge_dim - 2``; 两维分别预留给 fixed_halfedge 与 fixed_halfdist.
        self.edgetype_embedder = nn.Embedding(num_edge_types, edge_dim-2)
        
        # Denoiser network: remove noise from molecule representations
        # ``denoiser_name``: str, 配体去噪骨干类型; 当前 reduced 配置缺省为 default.
        denoiser_name = getattr(config.denoiser, 'name', 'default')
        if denoiser_name == 'default':
            # ``denoiser_bb``: type, 默认在标量 ContextNodeEdgeNet 与 GVP 版本之间按 gvp 开关选择.
            denoiser_bb = ContextNodeEdgeNet if not gvp else ContextNodeEdgeNetGVP
        elif denoiser_name == 'ipa':
            # ``denoiser_bb``: type, IPA/几何注意力配体-口袋去噪骨干.
            denoiser_bb = ContextGAEdgeNet
        # ``self.denoiser``: Module, 输入节点/边总宽度分别为 ``node_dim``、``edge_dim``, 并用 ``pocket_dim`` 维口袋节点作为上下文.
        self.denoiser = denoiser_bb(node_dim, edge_dim,
                            context_dim=pocket_dim, **config.denoiser)
        self.density_encoder = None
        if 'density' in config:
            # 仅密度模型加载其卷积依赖, 既有无密度配置不要求安装 einops.
            from models.density_backbone import DensityEncoder
            from models.density_readout import DensityReadout
            self.density_encoder = DensityEncoder(config.density)
            self.denoiser.density_readers = nn.ModuleList([
                DensityReadout(node_dim,config.density) for _ in range(config.denoiser.num_blocks)
            ])

        # Output decoders
        # ``self.node_decoder``: MLP; [N, node_dim] -> [N, num_node_types] 的原子类别 logits 解码头.
        self.node_decoder = MLP(node_dim, num_node_types, node_dim)
        # ``self.edge_decoder``: MLP; [H, edge_dim] -> [H, num_edge_types] 的无向半边类别 logits 解码头.
        self.edge_decoder = MLP(edge_dim, num_edge_types, edge_dim)
        
        # Additional outputs (e.g., confidence scores)
        # ``self.add_output``: list[str], 控制除三项主预测之外需要实例化和返回的输出头.
        self.add_output = getattr(config, 'add_output', [])
        if 'confidence' in self.add_output:
            # ``self.node_cfd``: MLP; [N, node_dim] -> [N, 1] 的原子类别置信度 logit 头.
            self.node_cfd = MLP(node_dim, 1, node_dim//2)
            # ``self.pos_cfd``: MLP; [N, node_dim] -> [N, 1] 的原始位置置信度头; 当前训练配置在loss内经sigmoid拟合0.2 ** 坐标误差_Å.
            self.pos_cfd = MLP(node_dim, 1, node_dim//2)
            # ``self.edge_cfd``: MLP; [H, edge_dim] -> [H, 1] 的无向半边置信度 logit 头.
            self.edge_cfd = MLP(edge_dim, 1, edge_dim//2)
            

    def forward(self, batch, **kwargs):
        """编码带噪分子与口袋条件, 并返回干净变量及可选置信度预测.

        输入字段:
            - batch.node_in: LongTensor, 形状为 (N,), 带噪原子类别编号.
            - batch.pos_in: FloatTensor, 形状为 (N, 3), 带噪配体局部坐标, 单位 Å.
            - batch.halfedge_in: LongTensor, 形状为 (H,), 带噪无向半边类别.
            - batch.halfedge_index: LongTensor, 形状为 (2, H), 完全图上三角端点, 数值索引配体原子.
            - batch.fixed_node: LongTensor|BoolTensor, 形状为 (N,), 1 表示原子类别是条件.
            - batch.fixed_pos: LongTensor|BoolTensor, 形状为 (N,), 1 表示坐标是条件.
            - batch.fixed_halfedge: LongTensor|BoolTensor, 形状为 (H,), 1 表示半边类别是条件.
            - batch.fixed_halfdist: LongTensor|BoolTensor, 形状为 (H,), 1 表示半边端点距离是条件.
            - batch.node_type_batch: LongTensor, 形状为 (N,), 每个配体原子的图归属编号.
            - batch.pocket_atom_feature: float32, (P, pocket_in_dim), 原 25 维蛋白输入特征, RA 中核酸原子全零.
            - batch.pocket_is_nucleic: bool, (P,), RA 专用; True 标记标准核酸原子, 同时对应 pocket_atom_feature 中全零的原子特征.
            - batch.pocket_nucleic_feature: float32, (P, 15), RA 专用; 核酸的元素、核苷酸与组分编码按类 Docstring 的顺序拼接, 蛋白原子全零.
            - batch.pocket_pos: FloatTensor, 形状为 (P, 3), 与配体同原点的口袋局部坐标, 单位 Å.
            - batch.pocket_knn_edge_index: LongTensor, 形状为 (2, E_p), 口袋内部有向 kNN 边端点.
            - batch.pocket_pos_batch: LongTensor, 形状为 (P,), 每个口袋原子的图归属编号.
            - batch.density_input: float32, (B, 56, 48, 48, 48), 仅密度模型读取; B 个分子的固定裁块, 空间轴 ZYX.
            - batch.density_origin: float32, (B, 3), 实际裁块角点的局部 XYZ 坐标, 单位 Å, 与 pos_in 共用模型原点.
            - batch.density_basis: float32, (B, 3, 3), 三行依次为源 X、Y、Z 一个体素步长在局部坐标系的向量, 单位 Å.
            - batch.is_peptide: LongTensor, 形状为 (N,), 小分子构象/docking 为全 0; 仅配置请求时读取.
            - kwargs: Mapping, 本实现不读取其中任何字段, 保留给统一调用接口.

        返回字段:
            - pred_node: FloatTensor, 形状为 (N, num_node_types), 干净原子类别 logits.
            - pred_pos: FloatTensor, 形状为 (N, 3), 干净配体局部坐标预测, 单位 Å.
            - pred_halfedge: FloatTensor, 形状为 (H, num_edge_types), 干净半边类别 logits.
            - confidence_node: FloatTensor, 形状为 (N, 1), 可选原子类别置信度 logit.
            - confidence_pos: FloatTensor, 形状为 (N, 1), 可选坐标置信度原始输出.
            - confidence_halfedge: FloatTensor, 形状为 (H, 1), 可选半边类别置信度 logit.
        """

        # ``pos_in``: FloatTensor, 形状为 (N, 3); 带噪配体局部坐标, 后续坐标更新保持相同原点, 单位 Å.
        pos_in = batch['pos_in']
        # ``h_node_in``: FloatTensor, 形状为 (N, self.nodetype_embedder.embedding_dim); 按 ``node_in`` 的 ``num_node_types`` 类编号查表得到原子类别嵌入.
        h_node_in = self.nodetype_embedder(batch['node_in'])
        # ``h_halfedge_in``: FloatTensor, 形状为 (H, self.config.edge_dim - 2); 按 ``halfedge_in`` 的 ``num_edge_types`` 类编号查表得到无向半边类别嵌入.
        h_halfedge_in = self.edgetype_embedder(batch['halfedge_in'])
        
        # ``node_extra``: FloatTensor, 形状为 (N, 2); dtype 与 ``pos_in`` 相同, 两列依次为 ``fixed_node`` 与 ``fixed_pos`` 的 0/1 prompt.
        node_extra = torch.stack([batch['fixed_node'], batch['fixed_pos']], dim=1).to(pos_in.dtype)
        # ``halfedge_extra``: FloatTensor, 形状为 (H, 2); dtype 与 ``pos_in`` 相同, 两列依次为 ``fixed_halfedge`` 与 ``fixed_halfdist`` 的 0/1 prompt.
        halfedge_extra = torch.stack([batch['fixed_halfedge'], batch['fixed_halfdist']], dim=1).to(pos_in.dtype)
        # ``h_node_in``: [N, self.nodetype_embedder.embedding_dim] + [N, 2] -> [N, self.nodetype_embedder.embedding_dim + 2]; 只拼接节点特征维.
        h_node_in = torch.cat([h_node_in, node_extra], dim=-1)
        # ``h_halfedge_in``: [H, self.config.edge_dim - 2] + [H, 2] -> [H, self.config.edge_dim]; 只拼接半边特征维.
        h_halfedge_in = torch.cat([h_halfedge_in, halfedge_extra], dim=-1)

        # ``n_halfedges``: int, 无向完全图半边总数; 用于把去噪后的双向边重新配对.
        n_halfedges = h_halfedge_in.shape[0]
        # ``halfedge_index``: LongTensor, 形状为 (2, H); 上三角无向半边端点, 每列满足 i<j, 数值索引配体原子维.
        halfedge_index = batch['halfedge_index']
        # ``edge_index``: [2, n_halfedges] + [2, n_halfedges] -> [2, 2 * n_halfedges]; 后半通过交换两行生成反向边, 每列在 GNN 中解释为目标、来源端点.
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        # ``h_edge_in``: [n_halfedges, self.config.edge_dim] -> [2 * n_halfedges, self.config.edge_dim]; 正反向边初始特征相同, 前后两段逐边配对.
        h_edge_in = torch.cat([h_halfedge_in, h_halfedge_in], dim=0)
        # ``edge_extra``: [n_halfedges, 2] -> [2 * n_halfedges, 2]; 正反向边共享同一 fixed edge/distance prompt.
        edge_extra = torch.cat([halfedge_extra, halfedge_extra], dim=0)
        
        # Add additional node features (e.g., peptide indicator)
        if 'is_peptide' in self.addition_node_features:
            # ``is_peptide``: FloatTensor, 形状为 (N, 1); dtype 与 ``pos_in`` 相同, 小分子构象/docking 全为 0, 与 ``h_node_in`` 原子维对齐.
            is_peptide = batch['is_peptide'].unsqueeze(-1).to(pos_in.dtype)
            # ``h_node_in``: [N, self.nodetype_embedder.embedding_dim + 2] + [N, 1] -> [N, self.nodetype_embedder.embedding_dim + 3]; 追加 is_peptide 通道.
            h_node_in = torch.cat([h_node_in, is_peptide], dim=-1)
        
        # (P, self.config.pocket_dim), 原 25 维蛋白特征的投影; RA 随后用核酸投影替换核酸原子对应的位置.
        h_pocket = self.pocket_embedder(batch['pocket_atom_feature'])
        if self.nucleic_branch is not None:
            # bool, (P,), 同时切分口袋特征和坐标的原子维; 例如 [False, True] 表示蛋白原子后跟核酸原子.
            is_nucleic = batch['pocket_is_nucleic']
            # [P_na, 15] -> [P_na, self.config.pocket_dim], 按原口袋原子编号写回, 不改变 pocket_pos_batch 的分子分组.
            h_pocket[is_nucleic] = self.nucleic_embedder(batch['pocket_nucleic_feature'][is_nucleic])

        # (P, self.config.pocket_dim), 官方只编码蛋白图; RA 沿 Dataset 给出的联合 kNN 图共享同一编码器.
        h_pocket = self.pocket_encoder(
            h_node=h_pocket,
            pos_node=batch['pocket_pos'],
            edge_index=batch['pocket_knn_edge_index'],
            h_edge=None,
            node_extra=None,
            edge_extra=None,
        )

        # ``h_node``: FloatTensor, 形状为 (N, self.config.node_dim), 完成 ``self.config.denoiser.num_blocks`` 个联合 block 后的配体节点隐藏特征.
        # ``pos_node``: FloatTensor, 形状为 (N, 3), 完成 ``self.config.denoiser.num_blocks`` 个坐标增量后的配体局部坐标, 单位 Å.
        # ``h_edge``: FloatTensor, 形状为 (2 * n_halfedges, self.config.edge_dim), 与双向 ``edge_index`` 列对齐的最终边隐藏特征.
        density_arguments = {}  # 无密度时不传新增字段, 保持原去噪调用路径.
        if self.density_encoder is not None:
            # 同一前向内只编码一次, 六个去噪块共享该特征; 训练的不同更新不复用旧编码结果.
            density_arguments = dict(
                density_feature=self.density_encoder(batch['density_input']),
                density_origin=batch['density_origin'],
                density_basis=batch['density_basis'],
            )
        h_node, pos_node, h_edge = self.denoiser(
            h_node=h_node_in,
            pos_node=pos_in, 
            h_edge=h_edge_in, 
            edge_index=edge_index,
            node_extra=node_extra,
            edge_extra=edge_extra,
            batch_node=batch['node_type_batch'],
            # Pocket context
            h_ctx=h_pocket,
            pos_ctx=batch['pocket_pos'],
            batch_ctx=batch['pocket_pos_batch'],
            **density_arguments,
        )
        
        # ``pred_node``: FloatTensor, 形状为 (N, self.num_node_types); 逐原子干净类别 logits, 未归一化.
        pred_node = self.node_decoder(h_node)
        # ``pred_halfedge``: [2 * n_halfedges, self.config.edge_dim] -> [n_halfedges, self.config.edge_dim] -> [n_halfedges, self.num_edge_types]; 一一配对的正反向隐藏特征求和而不取平均, 再解码无向半边 logits.
        pred_halfedge = self.edge_decoder(h_edge[:n_halfedges] + h_edge[n_halfedges:])
        # ``pred_pos``: FloatTensor, 形状为 (N, 3); GNN 最后一层更新后的配体局部坐标, 单位 Å.
        pred_pos = pos_node
        
        # Optional: predict self-confidence scores
        # ``additional_outputs.confidence_node``: FloatTensor, 形状为 (N, 1), 可选原子 confidence logit.
        # ``additional_outputs.confidence_pos``: FloatTensor, 形状为 (N, 1), 可选坐标 confidence 原始输出.
        # ``additional_outputs.confidence_halfedge``: FloatTensor, 形状为 (H, 1), 可选半边 confidence logit.
        # ``additional_outputs``: dict[str, Tensor], 默认空; 启用 confidence 时包含上述三个字段.
        additional_outputs = {}
        if 'confidence' in self.add_output:
            # ``pred_node_cfd``: FloatTensor, 形状为 (N, 1); 由最终节点特征预测的原子类别置信度 logit.
            pred_node_cfd = self.node_cfd(h_node)
            # ``pred_pos_cfd``: FloatTensor, 形状为 (N, 1); 由同一最终节点特征预测的坐标置信度原始输出.
            pred_pos_cfd = self.pos_cfd(h_node)
            # ``pred_edge_cfd``: FloatTensor, 形状为 (H, 1); 由正反向边隐藏特征之和预测的无向半边置信度 logit.
            pred_edge_cfd = self.edge_cfd(h_edge[:n_halfedges] + h_edge[n_halfedges:])
            additional_outputs = {
                # ``additional_outputs.confidence_node``: FloatTensor, 形状为 (N, 1); 原子类别 confidence logit.
                'confidence_node': pred_node_cfd,
                # ``additional_outputs.confidence_pos``: FloatTensor, 形状为 (N, 1); 坐标 confidence 原始输出.
                'confidence_pos': pred_pos_cfd,
                # ``additional_outputs.confidence_halfedge``: FloatTensor, 形状为 (H, 1); 半边 confidence logit.
                'confidence_halfedge': pred_edge_cfd,
            }
        
        # ``return.pred_node``: FloatTensor, 形状为 (N, self.num_node_types), 原子类别 logits.
        # ``return.pred_pos``: FloatTensor, 形状为 (N, 3), 去噪局部坐标, 单位 Å.
        # ``return.pred_halfedge``: FloatTensor, 形状为 (n_halfedges, self.num_edge_types), 半边类别 logits.
        # ``return.confidence_node``: FloatTensor, 形状为 (N, 1), 仅启用 confidence 时返回.
        # ``return.confidence_pos``: FloatTensor, 形状为 (N, 1), 仅启用 confidence 时返回.
        # ``return.confidence_halfedge``: FloatTensor, 形状为 (H, 1), 仅启用 confidence 时返回.
        return {
            # ``return.pred_node``: FloatTensor, 形状为 (N, self.num_node_types); 原子类别 logits.
            'pred_node': pred_node,
            # ``return.pred_pos``: FloatTensor, 形状为 (N, 3); 去噪配体局部坐标, 单位 Å.
            'pred_pos': pred_pos,
            # ``return.pred_halfedge``: FloatTensor, 形状为 (n_halfedges, self.num_edge_types); 无向半边类别 logits.
            'pred_halfedge': pred_halfedge,
            **additional_outputs,
        }
