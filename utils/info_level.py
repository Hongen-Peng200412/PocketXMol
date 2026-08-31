"""
Define and sample info_level
"""
import torch
from torch import nn
import numpy as np

INFO_LEVEL_DICT = {}
def register_info_level(name):
    def decorator(cls):
        INFO_LEVEL_DICT[name] = cls
        return cls
    return decorator

# def get_level(config, *args, **kwargs):
#     name = config.name
#     return INFO_LEVEL_DICT[name](config, *args, **kwargs)

def get_level(name, *args, **kwargs):
    return INFO_LEVEL_DICT[name](*args, **kwargs)

# XXX
class MolInfoLevel:
    """
    为一个分子采样共享进度，并按自由度实体数广播信息保留比例。

    配置字段:
        - name: str, 进度映射名称；``uniform`` 使用 ``1-step``，``advance``/``power``/``exp`` 使用对应 scaler。
        - min: float, 映射后 level 的线性缩放下界。
        - max: float, 映射后 level 的线性缩放上界。
        - asym: str|None, None 时所有自由度共享 level；``pos_first``/``pos_last`` 时几何自由度与离散类型使用不同进度。
        - step2level: 映射配置，仅 ``advance``、``power``、``exp`` 读取。

    构造参数:
        - device: torch.device|str|None, 随机 step、level 向量和 scaler buffer 所在设备。
        - mode: str, ``train`` 要求调用 ``sample_for_mol(step=None)``，``sample`` 要求显式传入标量 step。

    进度与 level:
        - step: 标量 Tensor，规范区间为 0..1；训练时从 ``Uniform[0,1)`` 采样，采样时由迭代器给出。
        - level: 标量 Tensor；先把 step 反向映射，再缩放为 ``level*(max-min)+min``。
        - step==1: 无论映射名称都先设 level=0，再执行 min/max 缩放；因此实际值为 min。
        - ``uniform``: step=0 映射到 max，step 接近 1 映射到 min，level 与加噪进度负相关。

    ``sample_for_mol`` 计数字段:
        - n_node: int, 原子类别 level 数量；返回 (N,) 并与 ``node_type`` 第一维对齐。
        - n_pos: int, 原子坐标 level 数量；返回 (N,) 并与 ``node_pos`` 第一维对齐。
        - n_edge: int, 半边类别 level 数量；返回 (H,) 并与 ``halfedge_type`` 第一维对齐。
        - n_trans: int, 刚体平移 level 数量；返回 (D,) 并与刚体域编号对齐。
        - n_rot: int, 刚体旋转 level 数量；返回 (D,) 并与刚体域编号对齐。
        - n_tor: int, 扭转 level 数量；返回 (T,) 并与 ``tor_bonds_anno`` 第一维对齐。

    返回结构:
        - 传一个计数字段时直接返回一个一维 Tensor。
        - 传多个计数字段时返回 ``list[Tensor]``，元素顺序严格等于调用 kwargs 的插入顺序。
        - asym=None 时全部向量填同一标量；asym 非 None 时位置/平移/旋转/扭转使用位置标量，节点/边类别使用类型标量。
    """
    def __init__(self, config, device=None, mode='train'):
        # ``config.name``：Literal[uniform,advance,power,exp]，step 到规范 level 的映射名。
        # ``config.min``：float，最终 level 线性缩放下界。
        # ``config.max``：float，最终 level 线性缩放上界。
        # ``config.asym``：Literal[pos_first,pos_last]|None，几何与离散类型的非对称调度策略。
        # ``config.step2level.scale_start``：float，advance 映射高保留端参考值。
        # ``config.step2level.scale_end``：float，advance 映射低保留端参考值。
        # ``config.step2level.width``：float，advance 映射过渡宽度。
        # ``config.step2level.k``：float，power/exp 映射的曲率参数。
        # ``self.config``：EasyDict，保留上述 level 区间、非对称策略和映射参数叶。
        self.config = config
        # ``self.name``：str，uniform/advance/power/exp 映射名称。
        self.name = config.name
        # ``self.min``：float，规范 level 线性缩放后的下界。
        self.min = config.min
        # ``self.max``：float，规范 level 线性缩放后的上界。
        self.max = config.max
        # ``self.asym``：str|None，几何与离散类型进度的非对称调度策略。
        self.asym = getattr(config, 'asym', None)
        # NOTE: 训练时 configs\train\train_pxm_reduced.yml 是 uniform 没用到, 但推理时 /C:/Users/15919/Desktop/PocketXMol/configs/sample/examples/dock_smallmol.yml 用的 free 指定了 advance 
        if self.name == 'advance':
            # ``self.step2level``：AdvanceScaler，端点校准 sigmoid step->level 映射，并迁移到 device。
            self.step2level = AdvanceScaler(config.step2level).to(device)
        elif self.name == 'power':
            # ``self.step2level``：PowerScaler，1-step^k 映射，并迁移 buffer 到 device。
            self.step2level = PowerScaler(config.step2level).to(device)
        elif self.name == 'exp':
            # ``self.step2level``：ExpScaler，端点归一化指数映射，并迁移 buffer 到 device。
            self.step2level = ExpScaler(config.step2level).to(device)

        # ``self.device``：torch.device|str|None，随机 step 与广播 level 张量的创建设备。
        self.device = device
        # ``self.mode``：str，train 要求 step=None，sample 要求显式 step。
        self.mode = mode
        # ``self.allowed_keys``：list[str]，限定 sample_for_mol 可接受的实体计数名，并防止调用方把未知自由度静默广播。
        self.allowed_keys = ['n_node', 'n_pos', 'n_edge', 'n_trans', 'n_rot', 'n_tor']
        
    def _sample(self, step):
        """
        取得一个共享 step，并返回单标量 level 或 ``(level_pos, level_type)``。

        输入参数:
            - step: 标量 Tensor|None；None 时在 ``self.device`` 上采样 ``Uniform[0,1)``。

        返回值:
            - asym=None: 标量 Tensor，全部自由度共享的信息保留比例。
            - asym 非 None: tuple[scalar Tensor, scalar Tensor]，依次为几何自由度与离散类型的信息保留比例。
        """
        if step is None:
            # ``step``：标量 Tensor，训练样本共享的均匀随机加噪进度。
            step = torch.rand((), device=self.device)
        
        if self.asym is None:
            # ``level``：scalar Tensor，共享给当前分子的所有请求自由度。
            level = self._step2level(step)
            return level
        else:
            if step < 0.5:
                # ``step_pos``：scalar Tensor，前半区间把几何位置进度拉伸为 1.8*step。
                step_pos = 9 / 5 *step
                # ``step_type``：scalar Tensor，把前半区间的离散类型进度压缩到 [0,0.1)。
                step_type = 1 / 5 * step
            else:
                # ``step_pos``：scalar Tensor，后半区间把几何位置进度映射到 0.9..1。
                step_pos = 1 / 5 * step + 0.8
                # ``step_type``：scalar Tensor，把后半区间的离散类型进度扩展到 [0.1,1]。
                step_type = 9 / 5 * step - 0.8
            if self.asym == 'pos_first':
                pass
            elif self.asym == 'pos_last':
                # ``step_pos``：scalar Tensor，交换后使用原离散类型进度，使几何位置后去噪。
                # ``step_type``：scalar Tensor，交换后使用原几何位置进度，使离散类型先去噪。
                step_pos, step_type = step_type, step_pos
            # ``level_pos``：scalar Tensor，由裁剪到 0..1 的几何进度反向映射得到。
            level_pos = self._step2level(step_pos.clamp(min=0, max=1))
            # ``level_type``：scalar Tensor，离散原子/边类别的信息保留比例。
            level_type = self._step2level(step_type.clamp(min=0, max=1))
            return (level_pos, level_type)
        
    def _step2level(self, step):
        """
        将一个规范化加噪进度映射并缩放为信息保留比例。

        输入参数:
            - step: 标量 Tensor，预期位于 0..1。

        返回值:
            - level: 标量 Tensor，先由配置映射得到规范值，再线性缩放到 ``[min,max]`` 对应区间。
        """
        if step == 1:  # step = 1 -> level = 0
            # ``level``：scalar Tensor，最大采样进度显式映射为规范 level 0。
            level = torch.zeros((), device=self.device)
        else:
            # transform. NOTE: negative correlation
            if self.name == 'uniform':  # uniformly sample from [min, max)
                # ``level``：scalar Tensor，线性反相关映射，step 0/1 对应规范 level 1/0。
                level = 1 - step 
            elif self.name in ['advance', 'power', 'exp']:
                # ``level``：scalar Tensor，由构造时选定的可调用 scaler 映射。
                level = self.step2level(step)
            else:
                raise ValueError(f'Unknown info_level name: {self.name}')
        
        # scale 
        # ``level``：标量 Tensor，把规范 level=0/1 分别映射到配置 min/max。
        level = level * (self.max - self.min) + self.min
        return level
        
    def set_value(self, shape, value):
        """
        把一个标量 level 广播为指定实体数/形状的浮点张量。

        输入参数:
            - shape: int|tuple[int,...], ``torch.ones`` 接受的输出形状。
            - value: 标量 Tensor|float, 填入全部实体位置的信息保留比例。

        返回值:
            - levels: shape, 位于 ``self.device``；每个元素数值都等于 value。
        """
        return torch.ones(shape, device=self.device) * value

    # NOTE: 此类的核心函数        
    def sample_for_mol(self, step, **kwargs):
        """
        为调用方点名的分子自由度生成逐实体 level 向量。

        输入参数:
            - step: 标量 Tensor|None；训练模式必须为 None，采样模式必须非 None。
            - kwargs.n_node: int，可选原子类别实体数；返回形状为 (N,) 的 level。
            - kwargs.n_pos: int，可选原子坐标实体数；返回形状为 (N,) 的 level。
            - kwargs.n_edge: int，可选半边类别实体数；返回形状为 (H,) 的 level。
            - kwargs.n_trans: int，可选整体平移实体数；返回形状为 (D,) 或 (B,) 的 level。
            - kwargs.n_rot: int，可选整体旋转实体数；返回形状为 (D,) 或 (B,) 的 level。
            - kwargs.n_tor: int，可选可旋转键实体数；返回形状为 (T,) 的 level。

        返回字段:
            - n_node 对应项: FloatTensor，形状为 (N,)，逐原子类别信息等级。
            - n_pos 对应项: FloatTensor，形状为 (N,)，逐原子坐标信息等级。
            - n_edge 对应项: FloatTensor，形状为 (H,)，逐半边类别信息等级。
            - n_trans 对应项: FloatTensor，形状为 (D,) 或 (B,)，逐刚体域/图平移信息等级。
            - n_rot 对应项: FloatTensor，形状为 (D,) 或 (B,)，逐刚体域/图旋转信息等级。
            - n_tor 对应项: FloatTensor，形状为 (T,)，逐可旋转键扭转信息等级。
            - 单键调用: 直接返回该键对应的一个 Tensor。
            - 多键调用: 返回 list[Tensor]，元素顺序严格等于 kwargs 插入顺序。
        """
        if self.mode == 'train':
            assert step is None, 'step should not be given in train mode'
        elif self.mode == 'sample':
            assert step is not None, 'step should be given in sample mode'
        else:
            raise ValueError(f'Unknown mode: {self.mode}')
        # ``value``：标量 Tensor 或二元标量 tuple；同一分子内由所有请求自由度共享。
        value = self._sample(step)

        # ``value_list``：list[Tensor]，按 kwargs 插入顺序累积各自由度的逐实体 level。
        value_list = []
        # ``key``：str，当前自由度计数叶名；其值 ``kwargs[key]`` 是要生成的 level 元素数量。
        for key in kwargs:
            assert key in self.allowed_keys, f'Unknown key: {key}'
            if self.asym is None:
                value_list.append(self.set_value(kwargs[key], value))
            else:
                if key in ['n_pos', 'n_trans', 'n_rot', 'n_tor']:
                    value_list.append(self.set_value(kwargs[key], value[0]))
                elif key in ['n_node', 'n_edge']:
                    value_list.append(self.set_value(kwargs[key], value[1]))
                else:
                    raise ValueError(f'Unknown key {key}')

        if len(value_list) == 1:
            # ``value_list``：Tensor，单自由度调用去掉 list 外壳，直接返回逐实体向量。
            value_list = value_list[0]
        return value_list


@register_info_level('individual')
class IndividualInfoLevel:
    # def __init__(self, config):
    #     self.config = config
    #     pass

    def sample_from_shape(self, shape):
        """
        Directly sample from [0, 1)
        """
        info = torch.rand(shape)
        return info
    

@register_info_level('preset')
class PresetInfoLevel:
    # def __init__(self, config):
    #     self.config = config
    #     pass

    def sample_from_shape(self, shape, value):
        info = torch.ones(shape) * value
        return info
    
@register_info_level('whole')
class WholeInfoLevel:
    # def __init__(self, config):
    #     self.config = config

    def sample_from_shape(self, shape, *args, **kwargs):
        value = torch.rand(1)
        info = torch.ones(shape) * value
        return info
    

@register_info_level('diff')
class DiffInfoTraj(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.num_steps = config['num_steps']
        self.traj_func = TrajFunction(config)
        # steps_list = np.linspace(0, 1, self.num_steps)
        # self.level_list = np.array([traj_func(step) for step in steps_list])
        
    def sample_from_shape(self, shape, step):
        level = self.traj_func(step)
        info = torch.ones(shape, device=level.device) * level
        return info
        


class AdvanceScaler(nn.Module):
    """
    用端点校准的 sigmoid 曲线把规范化 step 映射为规范化 level。

    配置字段:
        - scale_start: float, 输入 step=0 时的输出值。
        - scale_end: float, 输入 step=1 时的输出值。
        - width: float, sigmoid 陡峭度 k；绝对值越大，变化越集中在 step=0.5 附近。

    内部参数:
        - k: 标量不可训练 Parameter，等于 width。
        - a: 标量 Tensor，端点校准 sigmoid 的幅度系数。
        - b: 标量 Tensor，端点校准 sigmoid 的平移系数。

    输入输出:
        - x: 标量或任意形状 Tensor，通常位于 0..1。
        - return: 与 x 同形 Tensor；先把 x 线性映射到 -1..1，再应用校准 sigmoid。
    """
    def __init__(self, config):
        super().__init__()
        # ``self.config``：dict/EasyDict，保存 scale_start、scale_end 与 width 三个曲线参数。
        self.config = config
        # ``scale_start``：scalar Parameter，step=0 时目标输出端点，不参与梯度。
        scale_start = nn.Parameter(torch.tensor(config['scale_start']), requires_grad=False)
        # ``scale_end``：scalar Parameter，step=1 时目标输出端点，不参与梯度。
        scale_end = nn.Parameter(torch.tensor(config['scale_end']), requires_grad=False)
        # ``width``：scalar Parameter，sigmoid 曲线宽度/陡峭度，不参与梯度。
        width = nn.Parameter(torch.tensor(config['width']), requires_grad=False)
        self.setup(scale_start, scale_end, width)
        
    def setup(self, scale_start, scale_end, width):
        """由两个端点值和宽度求出 sigmoid 仿射系数 a、b。"""
        # ``self.k``：scalar Parameter，保存 sigmoid 宽度 k。
        self.k = width
        # ``A0``：scalar Tensor，x=1 时要求达到的终点输出。
        A0 = scale_end
        # ``A1``：scalar Tensor，x=0 时要求达到的起点输出。
        A1 = scale_start

        # ``self.a``：scalar Tensor，校准 sigmoid 幅度，使两个端点精确通过 A0/A1。
        self.a = (A0-A1)/(torch.sigmoid(-self.k) - torch.sigmoid(self.k))
        # ``self.b``：scalar Tensor，校准 sigmoid 纵向偏移。
        self.b = 0.5 * (A0 + A1 - self.a)
        
    def __call__(self, x):
        """将 ``x`` 从 0..1 映射到 -1..1，并返回端点校准的 sigmoid 值。"""
        # ``x``：与输入同形，规范进度从 [0,1] 线性变换为 [-1,1]。
        x = 2 * x - 1
        return self.a * torch.sigmoid(- self.k * x) + self.b


class PowerScaler(nn.Module):
    """
    使用 ``1-step^k`` 把加噪进度映射为信息保留比例。

    配置字段:
        - k: float, 幂指数；作为 buffer 随设备迁移且不参与训练。

    输入输出:
        - x: 标量或任意形状 Tensor。
        - return: 与 x 同形的 ``1-x^k``。
    """
    def __init__(self, config):
        super().__init__()
        # ``self.config``：dict/EasyDict，保存幂指数 k 配置。
        self.config = config
        self.register_buffer('k', torch.tensor(config['k']))
        # assert self.k >= 1, 'k should be >= 1'
        
    def __call__(self, x):
        return 1 - torch.pow(x, self.k)


class ExpScaler(nn.Module):
    """
    使用端点归一化指数曲线把 step=0/1 分别映射为 level=1/0。

    配置字段:
        - k: float, 指数曲率；作为 buffer 随设备迁移且不参与训练。

    输入输出:
        - x: 标量或任意形状 Tensor。
        - return: 与 x 同形的 ``(exp(kx)-exp(k))/(1-exp(k))``。
    """
    def __init__(self, config):
        super().__init__()
        # ``self.config``：dict/EasyDict，保存指数曲率 k 配置。
        self.config = config
        self.register_buffer('k', torch.tensor(config['k']))
        # assert self.k > 0, 'k should be > 0'
        
    def __call__(self, x):
        # ``value``：与 x 同形；分子和分母共享 k，使 x=0 得 1、x=1 得 0。
        value = (torch.exp(self.k * x) - torch.exp(self.k)) / (1 - torch.exp(self.k))
        return value


