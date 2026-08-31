"""
实现 PocketXMol 将分子坐标按信息等级混合到高斯、平移、旋转和扭转先验的几何噪声。

构象生成与小分子 docking 的主入口是 :class:`MolPrior` 和 :class:`AllPosPrior`。``free`` setting 使用
:class:`GaussianExplodePrior` 逐原子加噪；``rigid``、``torsional``、``flexible`` 分别组合
:class:`TranslationPrior`、:class:`RotationPrior` 和 :class:`TorsionalPrior`。所有坐标输入输出形状均为
``(N,3)``，最后一维按 XYZ 排列，数值单位与输入坐标一致，当前管线为 Å；旋转和扭转角单位为弧度。

本模块不落盘。构象/docking 的 ``MolPrior(pos_only=True)`` 不实例化离散节点/半边先验；
``GaussianPrior``、``CategoricalPrior`` 和已弃用 ``RigidPrior`` 不在这两个任务的当前位置噪声路径中。
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter_mean
from utils.motion import RobustAngleSO3Distribution, apply_axis_angle_rotation,\
        apply_torsional_rotation_multiple_domains, robust_sample_angle, sample_uniform_angle
from models.corrector import kabsch_flatten

PRIOR_DICT = {}
def register_prior(name):
    """把先验类登记到 ``PRIOR_DICT[name]``；同名注册会覆盖先前类。"""
    def decorator(cls):
        # ``PRIOR_DICT``：type，先验类本身；工厂按配置 name 延迟实例化。
        PRIOR_DICT[name] = cls
        return cls
    return decorator

# XXX 总————————>class MolPrior(nn.Module)
def get_prior(config, *args, **kwargs):
    """
    从映射式配置实例化一个先验，或解析 ``name=from_train`` 的配置引用。

    输入参数:
        - config: 映射式配置|None；None 直接返回 None，否则必须支持 ``config.name``。
        - ``*args``/``**kwargs``: 原样传给先验构造器。
        - train_config: ``config.name == 'from_train'`` 时必须位于 kwargs；其值替代当前 config 后递归解析。

    返回值:
        - prior: ``nn.Module``|None, ``PRIOR_DICT[最终名称]`` 的实例或 None。

    状态边界:
        - ``from_train`` 分支会从 kwargs 中弹出 ``train_config``；构象/docking 的 ``ConfSampleNoiser`` 通常在调用 ``MolPrior`` 前已先解析该引用。
    """
    if config is None:
        return None
    # ``name``：str，PRIOR_DICT 注册键或特殊引用名 from_train。
    name = config.name
    if name != 'from_train':
        return PRIOR_DICT[name](config, *args, **kwargs) #.to(device)
    else:
        # ``train_config``：映射式训练先验配置；pop 会从当前 kwargs 副本移除该键后递归解析。
        train_config = kwargs.pop('train_config')
        return get_prior(train_config, *args, **kwargs)

# XXX 总————————>class AllPosPrior(nn.Module)
class MolPrior(nn.Module):
    """
    组合一个位置先验与可选离散原子/半边先验，统一返回带噪模型输入。

    构造参数:
        - config.pos: 映射式位置先验配置；构象/docking 使用 ``name=allpos``。
        - config.node: 映射式原子类别先验配置；仅 ``pos_only=False`` 时读取。
        - config.edge: 映射式半边类别先验配置；仅 ``pos_only=False`` 时读取。
        - config.pos_only: bool, 配置级位置专用开关；与构造参数 ``pos_only`` 做逻辑或。
        - num_node_types: int, 离散原子类别数 K_n；仅构造 node prior 时使用。
        - num_edge_types: int, 离散半边类别数 K_e；仅构造 edge prior 时使用。
        - pos_only: bool, True 时不实例化也不调用离散先验；构象/docking 固定为 True。

    ``add_noise`` 输入:
        - node_type: int64, (N,), 真值原子类别；pos_only 路径原样保留但不传给先验。
        - node_pos: (N, 3), 真值/当前配体局部坐标，单位 Å。
        - halfedge_type: int64, (H,), 真值半边类别；pos_only 路径原样保留但不传给先验。
        - level_dict: dict[str, Tensor], setting 对应的信息等级；叶字段由 ``AllPosPrior`` 或离散先验消费。
        - from_prior: bool, True 时 level=0 的实体必须完全独立于真值，避免采样首步数据泄漏。
        - pos_only: bool, 调用级临时位置专用开关；与实例开关做逻辑或。
        - ``**kwargs``: ``mol_size`` 或刚体/扭转索引字段，原样传给位置先验。

    返回值:
        - 位置专用: pos_pert, (N, 3), 带噪坐标。
        - 联合离散: tuple ``(node_pert, pos_pert, halfedge_pert)``，形状依次为 ``(N,)``、``(N,3)``、``(H,)``。
    """
    def __init__(self, config, num_node_types, num_edge_types, pos_only=False):
        super().__init__()
        # ``config.pos``：Mapping，构象/docking 的 ``AllPosPrior`` 配置。
        # ``config.node``：Mapping|缺省，非 pos_only 路径的原子类别先验配置。
        # ``config.edge``：Mapping|缺省，非 pos_only 路径的半边类别先验配置。
        # ``config.pos_only``：bool|缺省，配置级关闭离散先验的开关。
        # ``self.config``：EasyDict，保留上述位置与可选离散先验叶。
        self.config = config
        # ``self.num_node_types``：int K_n，原子离散类别数；仅非 pos_only 路径使用。
        self.num_node_types = num_node_types
        # ``self.num_edge_types``：int K_e，半边离散类别数；仅非 pos_only 路径使用。
        self.num_edge_types = num_edge_types
        # ``self.pos_only``：bool，构造参数或配置任一为 True 即关闭离散 node/edge 先验。
        self.pos_only = pos_only or getattr(config, 'pos_only', False)
        
        # ``self.pos``：nn.Module，构象/docking 为 AllPosPrior，统一分派自由或结构化坐标噪声。
        self.pos = get_prior(config.pos)
        if not self.pos_only:
            # ``self.node``：nn.Module，逐原子离散类别先验；构象/docking 不实例化。
            self.node = get_prior(config.node, num_classes=num_node_types)
            # ``self.halfedge``：nn.Module，逐半边离散类别先验；构象/docking 不实例化。
            self.halfedge = get_prior(config.edge, num_classes=num_edge_types)
    
    @torch.no_grad()
    def add_noise(self, node_type, node_pos, halfedge_type,
                  level_dict, from_prior, pos_only=False, **kwargs):
        """按实例/调用的 pos_only 边界调用位置先验，并可选调用原子与半边离散先验。"""
        # ``pos_pert``：Tensor，形状为 (N, 3)；AllPosPrior 根据 level_dict 的互斥叶字段选择自由高斯或刚体/扭转组合。
        pos_pert = self.pos.add_noise(node_pos, level_dict, from_prior, **kwargs)
        if not (self.pos_only or pos_only):
            # ``node_pert``：LongTensor，形状为 (N,)；逐原子类别从真值 one-hot 与类别先验混合分布采样。
            node_pert = self.node.add_noise(node_type, level_dict['node'], from_prior)
            # ``halfedge_pert``：LongTensor，形状为 (H,)；逐半边类别从真值 one-hot 与类别先验混合分布采样。
            halfedge_pert = self.halfedge.add_noise(halfedge_type, level_dict['halfedge'], from_prior)
            return node_pert, pos_pert, halfedge_pert
        else:
            return pos_pert

# XXX  总————> class GaussianExplodePrior(nn.Module)
@register_prior('allpos')
# @register_prior('flexible')
class AllPosPrior(nn.Module):
    """
    按 ``info_level`` 的叶字段选择逐原子自由噪声，或顺序组合刚体平移、旋转和内部扭转。

    配置字段:
        - pos: 映射式逐原子位置先验|None；``free`` setting 的 ``info_level['pos']`` 使用。
        - translation: 映射式刚体平移先验|None；``info_level`` 含 ``trans`` 时使用。
        - rotation: 映射式刚体旋转先验|None；``info_level`` 含 ``rot`` 时使用。
        - torsional: 映射式可旋转键先验|None；``info_level`` 含 ``tor`` 时使用。

    ``add_noise`` 输入:
        - pos: (N, 3), 当前分子局部坐标，单位 Å。
        - info_level.pos: (N,), 逐原子自由位置保留比例；存在时立即走 free 分支并返回。
        - info_level.trans: (D,), 每个刚体域的平移保留比例。
        - info_level.rot: (D,), 每个刚体域的旋转保留比例。
        - info_level.tor: (T,), 每条 ``tor_bonds_anno`` 可旋转键的扭转保留比例。
        - from_prior: bool, True 时各先验确保 level=0 不依赖原坐标的相应自由度。
        - kwargs.mol_size: (N,), free Gaussian 可选逐原子分子大小，用于确定噪声标准差。
        - kwargs.domain_node_index: int64, (2,K), 第一行刚体域编号，第二行原子编号。
        - kwargs.tor_bonds_anno: int64, (T,3), 扭转层级和两轴端点。
        - kwargs.twisted_nodes_anno: int64, (W,2), 扭转键行号到随动原子编号。

    返回值:
        - pos: (N, 3), 按 ``trans -> rot -> tor`` 顺序应用所含自由度后的坐标，单位 Å；未点名自由度保持不变。

    互斥边界:
        - 设计意图是 ``pos`` 与 ``trans/rot/tor`` 互斥；当前代码的 assert 接收生成器对象，实际不会验证该互斥条件，调用方必须保证 level_dict 合法。
    """
    def __init__(self, config):
        super().__init__()
        # ``config.pos``：Mapping|None，free 模式逐原子坐标先验。
        # ``config.translation``：Mapping|None，flexible/rigid 模式逐域整体平移先验。
        # ``config.rotation``：Mapping|None，flexible/rigid 模式逐域整体旋转先验。
        # ``config.torsional``：Mapping|None，flexible/torsional 模式逐可旋转键先验。
        # ``self.config``：EasyDict，保留上述四个互斥或组合使用的坐标先验叶。
        self.config = config
        # ``self.pos_prior``：nn.Module|None，free 模式逐原子位置先验。
        self.pos_prior = get_prior(getattr(config, 'pos', None))
        # ``self.translation_prior``：nn.Module|None，结构化模式逐域整体平移先验。
        self.translation_prior = get_prior(getattr(config, 'translation', None))
        # ``self.rotation_prior``：nn.Module|None，结构化模式逐域整体旋转先验。
        self.rotation_prior = get_prior(getattr(config, 'rotation', None))
        # ``self.torsional_prior``：nn.Module|None，结构化模式逐可旋转键扭转先验。
        self.torsional_prior = get_prior(getattr(config, 'torsional', None))
        
    @torch.no_grad()
    def add_noise(self, pos, info_level, from_prior, **kwargs):
        """按 info_level 叶键分派位置先验；free 分支提前返回，结构化分支按平移、旋转、扭转顺序执行。"""
        if 'pos' in info_level:
            # ``pos``：[N, 3] -> [N, 3]；``free`` 逐原子高斯加噪，``mol_size`` 可控制逐原子标准差。
            pos = self.pos_prior.add_noise(pos, info_level['pos'], from_prior,
                                           kwargs.get('mol_size', None))
            assert ((key not in info_level) for key in ['trans', 'rot', 'tor']), 'pos noise is not compatiable with flexible noise'
            return pos
        # domain_index = kwargs['domain_index']
        # n_domain = domain_index.max() + 1
        if 'trans' in info_level:
            # ``pos``：[N, 3] -> [N, 3]；先按 ``domain_node_index`` 对每域施加整体平移。
            pos = self.translation_prior.add_noise(pos, info_level['trans'],
                    from_prior, kwargs['domain_node_index'])
        if 'rot' in info_level:
            # ``pos``：[N, 3] -> [N, 3]；再绕各域当前质心施加整体旋转。
            pos = self.rotation_prior.add_noise(pos, info_level['rot'],
                    from_prior, kwargs['domain_node_index'])
        if 'tor' in info_level:
            # ``pos``：[N, 3] -> [N, 3]；最后按 ``tor_bonds_anno`` 层级施加内部扭转。
            pos = self.torsional_prior.add_noise(pos, info_level['tor'], from_prior,
                            kwargs['tor_bonds_anno'], kwargs['twisted_nodes_anno'],
                            kwargs['domain_node_index'])
        return pos

# XXX
@register_prior('gaussian_simple')
class GaussianExplodePrior(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        sigma_max = getattr(config, 'sigma_max', 1) 
        # self.sigma_max = nn.Parameter(torch.tensor(sigma_max), requires_grad=False)
        self.register_buffer('sigma_max', torch.tensor(sigma_max))
        self.sigma_func = getattr(config, 'sigma_func', None)
        
    @torch.no_grad()
    def add_noise(self, x, info_level, from_prior, mol_size=None):

        if info_level.dim() < x.dim():
            info_level_exp = info_level[:, None].expand_as(x)
        else:
            info_level_exp = info_level

        # NOTE: when info_level == 0, the prior mean is 0. DIFFERENT from GaussianPrior
        # x = torch.where(info_level_exp == 0, torch.zeros_like(x), x)

        if mol_size is None or self.sigma_func is None:
            noise = torch.zeros_like(x)
            noise.normal_(mean=0, std=self.sigma_max)
        else:
            assert len(mol_size) == len(x), 'Error: mol_size and x have different dim'
            if self.sigma_func == 'sqrt': # 
                sigma = self.sigma_max * mol_size.sqrt()
                noise = torch.randn_like(x) * sigma[:, None].clamp(min=1)
            elif self.sigma_func == 'linbias':
                sigma = (0.08 * mol_size + 1).clamp(min=5)
                noise = torch.randn_like(x) * sigma[:, None]
            elif self.sigma_func == 'sqrtbias':
                sigma = ((mol_size - 40).clamp(min=0).sqrt() + 2).clamp(min=5)
                noise = torch.randn_like(x) * sigma[:, None]
            elif self.sigma_func == 'seg59':
                sigma = torch.clamp(0.08 * mol_size + 1, min=5, max=9)
                noise = torch.randn_like(x) * sigma[:, None]
            else:
                raise NotImplementedError(f'Error: sigma_func {self.sigma_func} not implemented')
            
        pert = x + (1 - info_level_exp) * noise
        if  from_prior:
            pert = torch.where(info_level_exp == 0, noise, pert)
            
        pert = torch.where(info_level_exp == 1, x, pert)
        return pert
   

@register_prior('torsional')
class TorsionalPrior(nn.Module):
    """
    为每条可旋转键采样轴角，并只旋转该键远离图中心一侧的随动原子。

    配置字段:
        - sigma_max: float, 最大角噪声占 π 的倍数；构造后 ``self.sigma_max=config.sigma_max*π``，单位弧度。
        - decouple: bool, True 时对每个刚体域执行 Kabsch 对齐，去除扭转附带的整体旋转和平移；缺省 True。

    前向输入:
        - pos: (N, 3), 全部分子原子的当前局部坐标，单位 Å。
        - info_level: (T,), 每条可旋转键的信息保留比例；与 ``tor_bonds_anno`` 第一维对齐。
        - from_prior: bool, True 时 level=0 的键改用 ``[-π,π]`` 均匀角，避免依赖原二面角。
        - tor_bonds_anno: int64, (T, 3), 每行 ``[执行层级, 远端轴原子, 近端轴原子]``。
        - twisted_nodes_anno: int64, (W, 2), 每行 ``[可旋转键行号, 随该键旋转的原子编号]``。
        - domain_node_index: int64, (2, K)|None, 第一行为刚体域编号、第二行为参与去耦对齐的原子编号；``decouple=True`` 时必需。

    前向输出:
        - pos_tor: (N, 3), 依 ``tor_order`` 应用全部扭转并可选域内 Kabsch 去耦后的坐标，单位 Å。

    采样规则:
        - sigmas: (T,), ``(1-info_level)*sigma_max``，单位弧度。
        - from_prior=False: 所有键从 ``robust_sample_angle(sigmas)`` 采样。
        - from_prior=True: 只有 level=0 的键替换为均匀角，其他键仍使用 robust 角样本。
    """
    def __init__(self, config):
        super().__init__()
        # ``config.sigma_max``：float|缺省，最大扭转角尺度占 π 的倍数，缺省 1。
        # ``config.decouple``：bool|缺省，是否用 Kabsch 去除扭转附带的整域刚体运动，缺省 True。
        # ``self.config``：EasyDict，保留上述扭转尺度与去耦开关叶。
        self.config = config
        # ``self.sigma_max``：scalar Tensor/float，配置 π 倍率转换后的最大扭转角尺度，单位 rad。
        self.sigma_max = getattr(config, 'sigma_max', 1) * torch.pi
        # ``self.decouple``：bool；True 时扭转后按 domain_node_index 做 Kabsch，去除附带的整域刚体运动。
        self.decouple = getattr(config, 'decouple', True)  # decouple from rototranslation

    @torch.no_grad()
    def add_noise(self, pos, info_level, from_prior,
                  tor_bonds_anno, twisted_nodes_anno,
                  domain_node_index=None):
        """拆分三张扭转注释，采样 T 个角度并返回同形坐标。"""
        # ``tor_order``：LongTensor，形状为 (T,)；每条可旋转键的 BFS 执行层级；同层可并行，层间由运动函数按序处理。
        tor_order = tor_bonds_anno[:, 0]
        # ``tor_bonds``：LongTensor，形状为 (T, 2)；每行两个数值索引 pos 第一维，定义旋转轴方向。
        tor_bonds = tor_bonds_anno[:, 1:]

        # ``index_tor``：LongTensor，形状为 (W,)；每个随动原子引用 tor_bonds 第一维中的哪条可旋转键。
        index_tor = twisted_nodes_anno[:, 0]
        # ``twisted_nodes``：LongTensor，形状为 (W,)；随动原子编号；数值索引 pos 第一维。
        twisted_nodes = twisted_nodes_anno[:, 1]

        if len(tor_order) == 0:
            return pos

        # prepare sigma and angles
        # ``sigmas``：Tensor，形状为 (T,)；每条键的角噪声尺度，单位弧度；level=1 为 0，level=0 为 sigma_max。
        sigmas = (1 - info_level) * self.sigma_max
        if not from_prior:
            # ``angles``：Tensor，形状为 (T,)；按各自 sigma 采样的有符号扭转角，单位弧度。
            angles = robust_sample_angle(sigmas)
        else:  # in [-pi, pi], x + uniform is uniform, thus no data leakage
            # ``angles_not_prior``：Tensor，形状为 (T,)；按各键 sigma 采样、用于非零信息等级的角候选，单位弧度。
            angles_not_prior = robust_sample_angle(sigmas)
            # ``angles_prior``：Tensor，形状为 (T,)；[-π,π] 均匀角候选；函数只借 sigmas 获取形状/设备。
            angles_prior = sample_uniform_angle(sigmas)
            # ``angles``：Tensor，形状为 (T,)；布尔条件逐键广播；仅 level 恰为 0 的键选择均匀角。
            angles = torch.where(info_level == 0, angles_prior, angles_not_prior)

        # apply torsional rotation
        # ``pos_tor``：Tensor，形状为 (N, 3)；按层级、轴端点和随动原子集合应用 T 个扭转后的坐标，单位 Å。
        pos_tor = apply_torsional_rotation_multiple_domains(
            pos, tor_order, tor_bonds, angles,
            twisted_nodes, index_tor,
        )

        # decouple from rototranslation by minimizing rmsd
        if self.decouple:
            # ``domain_index``：LongTensor，形状为 (K,)，每个域内条目的刚体域编号，取值范围为 [0, D)。
            # ``node_index``：LongTensor，形状为 (K,)，每个域内条目对应的配体原子编号，索引 ``pos`` 第一维。
            domain_index, node_index = domain_node_index
            # ``global_rot``：FloatTensor，形状为 (D, 3, 3)，把扭转后域坐标对齐回扭转前域坐标的无反射旋转矩阵。
            # ``global_trans``：FloatTensor，形状为 (D, 1, 3)，同一刚体对齐的行向量平移，单位 Å。
            global_rot, global_trans = kabsch_flatten(pos_tor[node_index], pos[node_index], domain_index)
    
            # apply global rotation and translation
            # ``pos_corrected_expand``：[K,3] -> [K,1,3]；按每个域的旋转矩阵右乘并加平移，输出 (K, 1, 3)。
            pos_corrected_expand = torch.matmul(
                pos_tor[node_index, None, :],
                global_rot.transpose(1, 2)[domain_index]
            ) + global_trans[domain_index]

            # ``pos_tor``：(K, 3) 写回，只替换 domain_node_index 登记原子；未登记原子保持扭转后原值。
            pos_tor[node_index] = pos_corrected_expand.squeeze(1)
        return pos_tor


@register_prior('rigid')
class RigidPrior(nn.Module):
    def __init__(self, config):
        raise NotImplementedError('Deprecated! Use flexible or allpos prior instead.')
        super().__init__()
        self.config = config
        self.translation_prior = get_prior(config.translation) if config.translation else None
        self.rotation_prior = get_prior(config.rotation) if config.rotation else None
    
    @torch.no_grad()
    def add_noise(self, pos, info_level, **kwargs):
        """
        pos: positions, [n_node, 3]
        info_level: [n_domain]
        domain_index: [n_node]
        Return:
            positions with rigid perturbation (translation & rotation) of each domain, [n_node, 3]
        """
        domain_index = kwargs['domain_index']
        n_domain = domain_index.max() + 1
        # translation
        if self.translation_prior:
            pos = self.translation_prior.add_noise(pos, info_level[:n_domain], **kwargs)
        # rotation
        if self.rotation_prior:
            pos = self.rotation_prior.add_noise(pos, info_level[n_domain:n_domain*2], **kwargs)
        return pos


@register_prior('rotation')
class RotationPrior(nn.Module):
    """
    绕每个刚体域自身质心施加随机 SO(3) 旋转，不改变域内任意原子对距离。

    配置字段:
        - sigma_max: float, 最大旋转角尺度占 π 的倍数；构造后单位为弧度。

    前向输入:
        - pos: (N, 3), 全部分子原子的当前局部坐标，单位 Å。
        - info_level: (D,), 每个刚体域的旋转信息保留比例。
        - from_prior: bool, True 时 level=0 的域从均匀 SO(3) 角分布采样，其他域使用 sigma 控制分布。
        - domain_node_index: int64, (2, K), 第一行是 K 个域内原子的域编号，第二行是其在 pos 第一维的原子编号。

    前向输出:
        - pos: (N, 3), 域内原子绕各自质心旋转后的坐标；不属于任何域的原子保持原值，单位 Å。
    """
    def __init__(self, config):
        super().__init__()
        # ``config.sigma_max``：float|缺省，最大整体旋转角尺度占 π 的倍数，缺省 1。
        # ``self.config``：EasyDict，保留整体旋转角尺度叶。
        self.config = config
        # ``self.sigma_max``：scalar Tensor/float，最大整体旋转角尺度，单位 rad。
        self.sigma_max = getattr(config, 'sigma_max', 1) * torch.pi
        # ``self.angle_distr``：RobustAngleSO3Distribution，按 sigma 或 uniform 标志采样 SO(3) 旋转角。
        self.angle_distr = RobustAngleSO3Distribution()
        
    @torch.no_grad()
    def add_noise(self, pos, info_level, from_prior,
                  domain_node_index):
        """为 D 个域采样轴角，并按 domain_node_index 将旋转后的 K 个域内原子写回坐标副本。"""
        # ``device``：torch.device，随机轴与角分布张量创建在坐标设备上。
        device = pos.device
        # ``n_domain``：int，刚体域数量 D；由 level 向量长度定义，不从域索引最大值推断。
        n_domain = info_level.shape[0]
        # ``domain_index``：LongTensor，形状为 (K,)，每个域内条目的刚体域编号，取值范围为 [0, D)。
        # ``node_index``：LongTensor，形状为 (K,)，每个域内条目对应的配体原子编号，索引 ``pos`` 第一维。
        domain_index, node_index = domain_node_index
        # ``sigmas``：Tensor，形状为 (D,)；每个域的旋转角尺度，单位弧度。
        sigmas = self.sigma_max * (1 - info_level)
        
        # sample
        # ``axes``：Tensor，形状为 (D, 3)；从三维标准高斯归一化得到的随机单位旋转轴。
        axes = self._sample_axis(n_domain, device)
        # ``angles``：Tensor，形状为 (D,)；按各域 sigma 从稳健 SO(3) 角分布采样，单位弧度。
        angles = self._sample_anlge(sigmas, is_unfiform=False)
        if from_prior:
            # ``angles_uniform``：Tensor，形状为 (D,)；均匀 SO(3) 角候选；仅替换 info_level 恰为 0 的域。
            angles_uniform = self._sample_anlge(sigmas, is_unfiform=True)
            # ``angles``：Tensor，形状为 (D,)；仅 level 恰为 0 的域替换为均匀 SO(3) 角。
            angles = torch.where(info_level == 0, angles_uniform, angles)

        # apply rotate around com instead of center
        # pos_domain = scatter_mean(pos, domain_index, dim=0)  # center of each domain
        # pos_center = pos[domain_center_nodes].mean(dim=1)
        # ``pos_center``：Tensor，形状为 (D, 3)；每个域内已登记原子坐标的算术平均，即旋转中心，单位 Å。
        pos_center = scatter_mean(pos[node_index], domain_index, dim=0)
        # ``pos_rel``：Tensor，形状为 (K, 3)；每个域内原子相对所属域质心的坐标，单位 Å。
        pos_rel = pos[node_index] - pos_center[domain_index]
        # ``pos_update``：Tensor，形状为 (K, 3)；按所属域轴角旋转相对坐标后加回域质心，单位 Å。
        pos_update = (apply_axis_angle_rotation(pos_rel, axes[domain_index], angles[domain_index])
                        + pos_center[domain_index])

        # ``pos``：Tensor，形状为 (N, 3)；复制原坐标，避免域内写回修改调用方输入张量。
        pos = pos.clone()
        # ``pos``：(K, 3) 写回，只替换 domain_node_index 登记的域内原子。
        pos[node_index] = pos_update
        return pos
    
    def _sample_axis(self, n, device):
        """返回 ``(n,3)`` 随机单位向量；每行由独立三维标准高斯向量沿最后一维归一化。"""
        # ``axes``：Tensor，形状为 (n, 3)；三维标准高斯向量沿 XYZ 维归一化得到随机单位轴。
        axes = F.normalize(torch.randn(n, 3, device=device), dim=-1)
        return axes

    def _sample_anlge(self, sigmas, is_unfiform):
        """返回与 ``sigmas(D,)`` 对齐的旋转角；``is_unfiform`` 原样传给 SO(3) 角分布采样器。"""
        # ``angles``：与 sigmas 同形，按 is_uniform 选择稳健尺度分布或均匀 SO(3) 角分布。
        angles = self.angle_distr.sample(sigmas, is_uniform=is_unfiform)
        return angles


@register_prior('translation')
class TranslationPrior(nn.Module):
    """
    平移每个刚体域的质心，同时保持域内原子相对坐标完全不变。

    配置字段:
        - ve: bool, True 使用 variance-exploding 加性平移，False 使用质心的 variance-preserving 混合。
        - sigma_max: float, VE 模式每个 XYZ 分量的最大高斯标准差，单位 Å。
        - mean: float, VP 模式先验质心每个 XYZ 分量的高斯均值，单位 Å。
        - std: float, VP 模式先验质心每个 XYZ 分量的高斯标准差，单位 Å。

    前向输入:
        - x: (N, 3), 全部分子原子的当前局部坐标，单位 Å。
        - info_level: (D,), 每个刚体域的平移信息保留比例。
        - from_prior: bool, 控制 VE 的 level=0 域是否显式去掉原质心；VP 在 level=0 时公式已自动消除原质心。
        - domain_node_index: int64, (2, K), 第一行域编号、第二行 x 第一维原子编号。

    前向输出:
        - x: (N, 3), 域内原子整体平移后的坐标副本；域内相对坐标及未登记原子保持不变，单位 Å。

    VE 规则:
        - 每域采样 ``noise_domain(D,3) ~ Normal(0,sigma_max)``，域内每个原子加 ``(1-level)*noise_domain``。
        - ``from_prior`` 且 level=0 时再减原域质心，使新域质心等于 noise_domain，不泄露原平移。

    VP 规则:
        - 每域采样先验质心 ``noise_domain ~ Normal(mean,std)``。
        - 新质心为 ``sqrt(level)*old_center + sqrt(1-level)*noise_domain``，再平移全部域内原子。
    """
    def __init__(self, config):
        super().__init__()
        # ``config.ve``：bool|缺省，True 选择 VE 加性平移，False 选择 VP 质心混合，缺省 True。
        # ``config.sigma_max``：float|缺省，VE 模式每个 XYZ 分量的最大标准差，单位 Å。
        # ``config.mean``：float|缺省，VP 模式先验质心每个 XYZ 分量均值，单位 Å。
        # ``config.std``：float|缺省，VP 模式先验质心每个 XYZ 分量标准差，单位 Å。
        # ``self.config``：EasyDict，保留上述平移分支与尺度叶。
        self.config = config
        # ``self.ve``：bool，True 使用加性 variance-exploding，False 使用质心 variance-preserving 混合。
        self.ve = getattr(config, 've', True)
        if self.ve:  # variance explode
            # ``sigma_max``：float，VE 每个 XYZ 分量的最大平移噪声标准差，单位 Å。
            sigma_max = getattr(config, 'sigma_max', 1)
            self.register_buffer('sigma_max', torch.tensor(sigma_max))
        else:  # variance perserve
            # ``mean``：float，VP 先验质心各 XYZ 分量均值，单位 Å。
            mean = getattr(config, 'mean', 0)
            # ``std``：float，VP 先验质心各 XYZ 分量标准差，单位 Å。
            std = getattr(config, 'std', 1)
            self.register_buffer('mean', torch.tensor(mean))
            self.register_buffer('std', torch.tensor(std))

    
    def add_noise(self, x, info_level, from_prior, domain_node_index):
        """按 D 个域的信息等级采样新质心，并把域内 K 个原子整体平移到该质心。"""
        # ``n_domain``：int，刚体域数量 D；决定每域噪声向量数量。
        n_domain = info_level.shape[0]
        # ``domain_index``：LongTensor，形状为 (K,)，为每个域内条目选择逐域信息等级与平移噪声。
        # ``node_index``：LongTensor，形状为 (K,)，为每个域内条目选择 ``x`` 第一维的配体原子。
        domain_index, node_index = domain_node_index
        
        if info_level.dim() < x.dim():
            # ``info_level_exp``：[D] -> [D,1] -> [D,3]，把每域标量 level 广播到 XYZ 三个分量。
            info_level_exp = info_level[:, None].expand(n_domain, x.shape[1])
        else:
            # ``info_level_exp``：(D,) 或 (D, 3)，输入已包含与坐标相同维数时不再广播。
            info_level_exp = info_level
        
        # ``pos_domain``：Tensor，形状为 (K, 3)；按 domain_node_index 第二行抽取的全部域内原子坐标，单位 Å。
        pos_domain = x[node_index]
        if self.ve:
            # ``noise_domain``：Tensor，形状为 (D, 3)；每个域一个零均值高斯平移向量，标准差 sigma_max Å。
            noise_domain = torch.randn_like(pos_domain[:n_domain]) * self.sigma_max
            # ``pert``：Tensor，形状为 (K, 3)；将所属域的平移向量按 1-level 缩放后加到每个域内原子。
            pert = pos_domain + (1 - info_level_exp[domain_index]) * noise_domain[domain_index]
            if from_prior:
                # ``center_domain``：Tensor，形状为 (D, 3)；加噪前每个域的质心，单位 Å。
                center_domain = scatter_mean(pos_domain, domain_index, dim=0)
                # ``pert``：level=0 的域从 pert 中减旧质心，使输出质心只由 noise_domain 决定。
                pert = torch.where(info_level_exp[domain_index] == 0, 
                            pert - center_domain[domain_index], pert)
        else:
            # ``noise_domain``：Tensor，形状为 (D, 3)；VP 模式每个域独立采样的先验质心，单位 Å。
            noise_domain = torch.randn_like(pos_domain[:n_domain]) * self.std + self.mean
            # ``pos_center_before``：Tensor，形状为 (D, 3)；加噪前每个域的质心，单位 Å。
            pos_center_before = scatter_mean(pos_domain, domain_index, dim=0)
            # ``pos_center_after``：Tensor，形状为 (D, 3)；按 sqrt(level) 与 sqrt(1-level) 混合旧质心和先验质心。
            pos_center_after = (info_level_exp).sqrt() * pos_center_before +\
                    (1 - info_level_exp).sqrt() * noise_domain
            # ``pert``：Tensor，形状为 (K, 3)；保留每个原子的域内相对坐标，只把域质心替换为 pos_center_after。
            pert = pos_domain - pos_center_before[domain_index] + pos_center_after[domain_index]
            if from_prior:
                pass  # when from_prior, info_level_exp==0. auto cancelling out x
        # ``pert``：Tensor，形状为 (K, 3)；level=1 的域精确恢复原坐标，消除采样数值误差。
        pert = torch.where(info_level_exp[domain_index] == 1, pos_domain, pert)
        
        # ``x``：Tensor，形状为 (N, 3)；复制全坐标，避免域内写回修改调用方输入。
        x = x.clone()
        # ``x``：(K, 3) 写回，只替换 domain_node_index 登记原子。
        x[node_index] = pert
        return x

 

@register_prior('gaussian')
class GaussianPrior(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        mean = getattr(config, 'mean', 0)
        std = getattr(config, 'std', 1)
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))
        
    @torch.no_grad()
    def add_noise(self, x, info_level, from_prior):
        if info_level.dim() < x.dim():
            info_level_exp = info_level[:, None].expand_as(x)
        else:
            info_level_exp = info_level
        noise = torch.zeros_like(x)
        noise.normal_(mean=self.mean, std=self.std)
        pert = info_level_exp.sqrt() * x + (1 - info_level_exp).sqrt() * noise
        if from_prior:
            pass  # when from_prior, info_level_exp==0. auto cancelling out x
            # pert = torch.where(info_level_exp == 0, noise, pert)
        pert = torch.where(info_level_exp == 1, x, pert)
        return pert

    # def sample_prior(self, x):
    #     return self.add_noise(x, info_level=torch.zeros_like(x))

@register_prior('categorical')
class CategoricalPrior(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        prior_type = config.prior_type
        prior_probs = getattr(config, 'prior_probs', None)
        probs = self.get_prior(prior_type, prior_probs)
        self.prior_probs = nn.Parameter(probs, requires_grad=False)
        
    def get_prior(self, prior_type, prior_probs):
        if prior_type == 'uniform':
            probs = 1. / self.num_classes * torch.ones(self.num_classes)
        elif prior_type == 'tomask':
            probs = torch.zeros(self.num_classes)
            probs[-1] = 1.
        elif prior_type == 'tomask_half':
            probs = torch.zeros(self.num_classes)
            probs[-1] = 0.5
            probs[:-1] = 0.5 / (self.num_classes - 1)
        elif prior_type == 'predefined':
            assert prior_probs is not None, 'Error: prior_probs is None and prior_type is predefined'
            assert len(prior_probs) == self.num_classes, f'Error: len(prior_probs) != {self.num_classes}'
            prior_probs = np.array([float(p) for p in prior_probs])
            prior_probs = prior_probs / sum(prior_probs)
            probs = torch.tensor(prior_probs)
        else:
            raise NotImplementedError(f'Error: prior_type {prior_type} not implemented')
        return probs
        # self.sampler = torch.distributions.Categorical(probs=probs)

    @torch.no_grad()
    def add_noise(self, x, info_level, from_prior, return_posterior=False):
        """
        x: (N,)
        info_level: (N,)
        """

        
        x_onehot = F.one_hot(x, self.num_classes).float()
        info_level_exp = info_level[:, None].expand_as(x_onehot)

        prior = self.prior_probs[None, :].expand_as(x_onehot)
        prob = info_level_exp * x_onehot + (1 - info_level_exp) * prior  # (N, K)
        if from_prior:
            pass  # when from_prior, info_level_exp==0. auto cancelling out x_onehot
        # sample from prob
        pert = torch.multinomial(prob, num_samples=1).squeeze(-1) # (N,)
        pert = torch.where(info_level == 1, x, pert)
        prob = torch.where(info_level_exp == 1, x_onehot, prob)
        if return_posterior:
            return pert, prob
        return pert
        
