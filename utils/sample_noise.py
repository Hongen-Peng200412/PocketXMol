"""
控制 PocketXMol 训练加噪与迭代采样的状态转换，并把模型输出写回下一步批次。

构象生成和小分子 docking 共用 :class:`ConfSampleNoiser`；:class:`DockSamplNoiser` 只把任务名改为
``dock``，从而选择 docking 的训练先验并保留配体相对口袋的整体平移。:class:`BaseSampleNoiser`
统一执行 ``当前 node/pos/halfedge -> 信息等级 -> 先验加噪 -> fixed 覆盖 -> *_in 字段`` 生命周期，
``models.sample.sample_loop3`` 则在每一步模型前向后调用 ``outputs2batch`` 推进 ``node_pos``。

本模块不落盘；返回的核心对象是原 PyG ``Data/Batch``，新增或更新 ``node_in``、``pos_in``、
``halfedge_in``、``step`` 和当前预测状态。
"""

from easydict import EasyDict
from torch_scatter import scatter_mean, scatter_min
from tqdm import tqdm
import torch
from copy import deepcopy
from scipy.optimize import linear_sum_assignment

from torch_geometric.data import Batch
from torch_geometric.nn import radius

from models.diffusion import *
from models.corrector import correct_pos_batch, kabsch_flatten, correct_pos_batch_no_tor, grad_len_to_pos, correct_pos_by_fixed_dist_batch
from utils.data import Mol3DData
from utils.prior import MolPrior
from utils.info_level import MolInfoLevel
from utils.shape import get_points_from_letter

SAMPLE_NOISE_DICT = {}
def register_sample_noise(name):
    """把噪声控制器类登记到 ``SAMPLE_NOISE_DICT[name]``；同名注册会覆盖先前类。"""
    def add_to_registery(cls):
        # ``SAMPLE_NOISE_DICT``：type，noiser 类本身；get_sample_noiser 按配置 name 延迟实例化。
        SAMPLE_NOISE_DICT[name] = cls
        return cls
    return add_to_registery


def get_sample_noiser(config, num_node_types, num_edge_types, *args, **kwargs):
    """
    按 ``config['name']`` 实例化已注册噪声控制器。

    输入参数:
        - config: 映射式噪声配置；嵌套 ``prior``、``level``、``num_steps`` 等字段由选中类继续读取。
        - num_node_types: int, 原子离散类别数 K_n。
        - num_edge_types: int, 半边离散类别数 K_e。
        - ``*args``/``**kwargs``: 原样传给控制器；采样入口会提供 ``mode``、``device`` 和训练噪声 ``ref_config``。

    返回值:
        - noiser: callable, ``SAMPLE_NOISE_DICT[config.name]`` 的实例。
    """
    # ``name``：str，SAMPLE_NOISE_DICT 注册键，如 mixed、conf 或 dock。
    name = config['name']
    return SAMPLE_NOISE_DICT[name](config, num_node_types, num_edge_types, *args, **kwargs)

# XXX tool
def dict_list2item(inputs):
    """
    递归把 PyG 批处理产生的列表叶子压缩为第一个元素。

    输入参数:
        - inputs: 任意嵌套 dict/list/标量；``task_setting`` 在单样本时是 str，在批次还原时可能是 ``list[str]`` 或嵌套映射。

    返回值:
        - dict: 保留全部键并递归压缩每个值。
        - list: 返回 ``inputs[0]``。
        - 其他类型: 原样返回。

    已知校验边界:
        - 当前 ``assert len(set(inputs))`` 只验证列表至少含一个不同值；它不会验证所有元素相等，与错误消息声明不一致。
        - 正式采样依赖同一批次全部图共享 setting；调用方必须自行满足此前提。
    """
    if isinstance(inputs, dict):
        # ``key``：str，当前嵌套映射叶名。
        # ``value``：object，当前叶值；可以继续是 dict、PyG 收集的 list 或终端标量。
        return {key: dict_list2item(value) for key, value in inputs.items()}
    elif isinstance(inputs, list):
        assert len(set(inputs)), 'all task_setting should be the same for sampling'
        return inputs[0]
    else:
        return inputs

class BaseSampleNoiser:
    """
    定义所有任务噪声器共享的训练加噪和采样步生命周期。

    构造参数:
        - task_name: str, 当前子噪声器对应的训练任务名；用于从 mixed ``ref_config`` 中选择同名先验。
        - config: 映射式当前噪声配置。
        - config.num_steps: int, 仅 sample 模式读取的离散迭代步数 S。
        - config.init_step: float, 仅 sample 模式读取的最大规范进度；缺省 1。
        - config.spring_in: 映射|False, 可选带噪键长弹簧修正配置。
        - config.reassign_in: bool, 是否在加噪后按分子对称置换重新分配坐标。
        - num_node_types: int，原子类别数；fixed 检查与通用子类使用。
        - num_edge_types: int，半边类别数；fixed 检查与可选键长弹簧使用。
        - mode: str, ``train`` 在每个单样本上随机采 level，``sample`` 根据显式 step 迭代。
        - device: torch.device|str, 先验、level 和随机张量设备。
        - ref_config: 映射|None, checkpoint 的训练噪声配置；sample 配置 ``prior=from_train`` 时提供实际先验。
        - pos_only: bool, 当前任务是否只改变坐标；构象/docking 为 True。

    sample 模式派生配置:
        - num_steps: int S, ``steps_loop`` 的离散步数。
        - init_step: float, 第一迭代步的规范进度。
        - ref_prior_config: 映射|None, 若 ref_config.name=mixed，则从 ``individual`` 中选 ``name==task_name`` 的第一项并取其 ``prior``。

    ``__call__`` 输入批次字段:
        - node_type: int64, (N,), 当前干净/上一步原子类别状态。
        - node_pos: (N, 3), 当前干净/上一步配体局部坐标，单位 Å。
        - halfedge_type: int64, (H,), 当前干净/上一步无向半边类别。
        - fixed_node: LongTensor|BoolTensor，形状为 (N,)，1 表示原子类别必须在加噪后恢复。
        - fixed_pos: LongTensor|BoolTensor，形状为 (N,)，1 表示坐标必须在加噪后恢复。
        - fixed_halfedge: 0/1, (H,), 1 表示相应半边类别必须在加噪后恢复。
        - fixed_halfdist: 0/1, (H,), 可选 spring 和几何修正使用的固定距离标记。
        - task: str|list[str], 当前任务名；训练变换前为 str，采样批次通常是同值列表。
        - task_setting: str|list|dict, 当前自由度模式；由 ``dict_list2item`` 压缩。

    ``__call__`` 输出批次字段:
        - node_in: int64, (N,), 当前步模型输入原子类别；fixed_node=1 的位置精确等于 node_type。
        - pos_in: (N, 3), 当前步模型输入坐标；fixed_pos=1 的原子精确等于 node_pos，单位 Å。
        - halfedge_in: int64, (H,), 当前步模型输入半边类别；fixed_halfedge=1 的位置精确等于 halfedge_type。

    首步边界:
        - ``step == 1`` 时调用任务先验的 ``from_prior=True``，要求 level=0 自由度完全不依赖真值。
        - 其他 step 和训练 ``step=None`` 使用 ``from_prior=False``，在当前状态周围按 level 添加噪声。
    """
    def __init__(self, task_name, config, num_node_types, num_edge_types,
                mode, device, ref_config=None, pos_only=False, *args, **kwargs):
        super().__init__()
        # ``config.prior``：Mapping|str，当前任务先验配置；sample 可用 ``from_train`` 继承训练先验。
        # ``config.level``：Mapping，信息等级采样器及 step-to-level 调度配置。
        # ``config.num_steps``：int|缺省，sample 模式离散去噪步数。
        # ``config.init_step``：float|缺省，sample 首步规范进度，缺省 1。
        # ``config.spring_in``：Mapping|False|缺省，加噪输入的可选键长弹簧配置。
        # ``config.reassign_in``：bool|缺省，训练时是否按 ``matches_iso`` 重排等价原子坐标。
        # ``config.pre_process``：str|None，fixed 坐标在先验后的输入恢复策略。
        # ``config.post_process``：str|Mapping|None，网络坐标写回前的几何投影策略。
        # ``self.config``：EasyDict，保留上述当前任务 noiser 叶。
        self.config = config
        # ``self.num_node_types``：int K_n，原子类别数；fixed 检查和通用子类使用。
        self.num_node_types = num_node_types
        # ``self.num_edge_types``：int K_e，半边类别数；spring_in 用其界定真实键类别区间。
        self.num_edge_types = num_edge_types
        # ``self.mode``：str，train 随机 level，sample 使用显式 step。
        self.mode = mode
        # ``self.device``：torch.device|str，先验、level 与随机张量所在设备。
        self.device = device
        # ``self.pos_only``：bool，当前任务是否只改变坐标；构象/docking 为 True。
        self.pos_only = pos_only
        
        # see me: free docking 在推理时实际只使用其中的逐原子 Gaussian prior。
        if mode == 'sample':
            # ``self.num_steps``：int，反向去噪循环的迭代步数。
            self.num_steps = config.num_steps
            # ``self.init_step``：float，采样第一步的最大规范进度；缺省 1 才会触发 from_prior 首步。
            self.init_step = config.get('init_step', 1)
            if ref_config is not None:
                if ref_config.name == 'mixed':
                    # ``ref_config``：list[config]，从训练 mixed noiser 中筛选与当前 task_name 完全同名的子配置。
                    ref_config = [c for c in ref_config.individual if c.name==task_name]
                    if len(ref_config) > 0:
                        # ``ref_config.name``：str，与 ``task_name`` 完全相同的训练子 noiser 名。
                        # ``ref_config.prior``：Mapping，当前任务训练坐标先验，供 sample 的 ``from_train`` 解析。
                        # ``ref_config.level``：Mapping，训练随机信息等级配置；这里只保留但不作为采样调度使用。
                        # ``ref_config.reassign_in``：bool|缺省，训练输入同构重排开关；sample 继承先验时不读取。
                        # ``ref_config``：EasyDict，mixed ``individual`` 中首个同名训练子配置。
                        ref_config = ref_config[0]
                    else:
                        # ``ref_config``：None，训练 mixed 配置中没有当前任务子项。
                        ref_config = None
                # ``self.ref_prior_config.pos``：Mapping|None，free 逐原子坐标先验。
                # ``self.ref_prior_config.translation``：Mapping|None，逐图整体平移先验。
                # ``self.ref_prior_config.rotation``：Mapping|None，逐图整体旋转先验。
                # ``self.ref_prior_config.torsional``：Mapping|None，逐可旋转键扭转先验。
                # ``self.ref_prior_config``：EasyDict|None，训练子配置的上述 ``prior`` 叶；sample ``from_train`` 使用。
                self.ref_prior_config = getattr(ref_config, 'prior', None)
            else:
                # ``self.ref_prior_config``：None，调用方未提供 checkpoint 训练噪声配置。
                self.ref_prior_config = None

        
    def add_noise(self, *args, **kwargs):
        """
        按任务自由度把当前分子状态与先验噪声混合；具体字段契约由子类实现。

        位置参数约定:
            - node_type: int64, (N,), 当前原子类别；N 为本样本或批次中的配体原子总数。
            - node_pos: float, (N, 3), 当前配体坐标，单位 Å；构象任务使用分子中心坐标系，docking 使用口袋局部坐标系。
            - halfedge_type: int64, (H,), 当前无向半边类别；与 batch.halfedge_index 的 H 列一一对齐。
            - batch: PyG Data/Batch, 提供任务 setting、图归属索引及结构化运动注释。

        关键字参数约定:
            - from_prior: bool, True 表示完全从 level=0 的任务先验初始化；False 表示在当前状态周围加噪。
            - level_dict: dict[str, Tensor], 每个自由度的信息保留量；叶子字段由 ``sample_level`` 给出。

        返回值:
            - in_dict.node: int64, (N,), 加噪后的原子类别。
            - in_dict.pos: float, (N, 3), 加噪后的配体坐标，单位 Å。
            - in_dict.halfedge: int64, (H,), 加噪后的无向半边类别。
        """
        raise NotImplementedError('Please implement this for each task.')

    def sample_level(self, *args, **kwargs):
        """
        为当前任务自由度生成信息等级；具体键集合由子类和 ``task_setting`` 决定。

        输入参数:
            - step: float|None, 采样时为规范进度，训练时为 None 并触发随机采样。
            - batch: PyG Data/Batch, 提供原子数、图数、可旋转键数等自由度计数。

        返回值:
            - level_dict: dict[str, Tensor], 值域通常为 [0, 1]；1 保留当前状态，0 采用任务先验。
        """
        raise NotImplementedError('Please implement this for each task.')

    def outputs2batch(self, *args, **kwargs):
        """
        把模型当前步预测转换为下一步批次状态；具体几何约束由子类实现。

        输入参数:
            - batch: PyG Batch, 当前采样状态及 fixed/几何注释字段。
            - outputs: dict[str, Tensor], 模型输出；至少含 ``pred_pos``，形状 (N, 3)，单位 Å。

        返回值:
            - batch: 同一 PyG Batch 对象；原地更新下一步所读取的 ``node_pos`` 等状态字段。
        """
        raise NotImplementedError('Please implement this for each task.')
    
    def _get_task(self, batch):
        """
        从单样本或同任务批次中读取任务名。

        输入字段:
            - batch.task: str|list[str], 如 ``conf`` 或 ``dock``；采样批次预期所有元素相同。

        返回值:
            - task: str, 若输入为列表则取第一个元素。

        已知校验边界:
            - 当前 ``assert len(set(task))`` 仅保证列表的不同值集合非空，不会验证列表元素全部相等。
        """
        # ``task``：str|list[str]，列表长度通常等于图数 B；PyG 会把每图的字符串属性收集成列表。
        task = batch['task']
        if isinstance(task, list):
            assert len(set(task)), 'all task should be the same for sampling'
            # ``task``：str，依赖“同一采样批次只含一种任务”的调用方约束选择首项。
            task = task[0]
        return task
    
    def _get_setting(self, batch):
        """
        读取并压缩当前批次共享的自由度模式。

        输入字段:
            - batch.task_setting: str|list|dict, 叶子可能被 PyG 批处理为同值列表。

        返回值:
            - setting: str|dict, 构象/docking 常见字符串为 ``free``、``flexible``、``torsional`` 或 ``rigid``。
        """
        # ``setting``：str|dict，递归移除 PyG 为每图保留的列表维；嵌套键保持不变。
        setting = dict_list2item(batch['task_setting'])
        return setting
    
    def _fetch_data(self, batch):
        """
        从批次复制当前分子状态，切断返回张量与原字段的存储和梯度关系。

        输入字段:
            - batch.node_type: int64, (N,), 当前原子类别。
            - batch.node_pos: float, (N, 3), 当前配体坐标，单位 Å。
            - batch.halfedge_type: int64, (H,), 当前无向半边类别。

        返回值:
            - item[0]: int64, (N,), ``node_type`` 的 detach+clone 副本。
            - item[1]: float, (N, 3), ``node_pos`` 的 detach+clone 副本，单位 Å。
            - item[2]: int64, (H,), ``halfedge_type`` 的 detach+clone 副本。

        调用边界:
            - 训练时字段表示真值；采样首步表示初始化模板，后续步表示上一步 ``outputs2batch`` 写回的状态。
        """
        # ``node_type``：LongTensor，形状为 (N,)；与批次中的全局配体原子顺序对齐。
        node_type = batch.node_type 
        # ``node_pos``：FloatTensor，形状为 (N, 3)；单位 Å；坐标系由任务决定。
        node_pos = batch.node_pos
        # ``halfedge_type``：LongTensor，形状为 (H,)；与 halfedge_index 的列顺序对齐。
        halfedge_type = batch.halfedge_type
        
        return [node_type.detach().clone(),
                node_pos.detach().clone(),
                halfedge_type.detach().clone()]

    def reassign_in_node(self, batch, in_dict):
        """返回未改动的 ``in_dict``；需要处理同构原子置换的子类覆盖此钩子。"""
        return in_dict

    # see me: 下面函数无用
    def spring_in_pos(self, batch, in_dict):
        """
        对预测为化学键的半边施加区间键长弹簧，同时保持 fixed 原子不动。

        输入字段:
            - batch.fixed_halfdist: 0/1, (H,), 1 表示该半边距离已有硬约束；只要批次存在任一硬约束，本函数整体跳过。
            - batch.halfedge_index: int64, (2, H), 无向半边端点索引，每列只保存一次原子对。
            - batch.halfedge_type: int64, (H,), 参考/真值半边类别；仅 ``inout_bond=True`` 时参与二次筛选。
            - batch.fixed_pos: 0/1, (N,), 1 表示该原子坐标不能被弹簧更新。
            - in_dict.halfedge: int64, (H,), 当前模型输入半边类别。
            - in_dict.pos: float, (N, 3), 待修正配体坐标，单位 Å。

        配置字段 ``config.spring_in``:
            - iters: int, 固定步数的坐标修正迭代次数。
            - lr: float, 每次从非 fixed 原子坐标减去梯度时的步长。
            - min: float, 不惩罚的最小键长，单位 Å。
            - max: float, 不惩罚的最大键长，单位 Å。
            - std: float, 仅诊断变量 ``loss`` 使用的长度归一化尺度，单位 Å。
            - inout_bond: bool, 可选；True 时要求输入类别和参考类别都属于化学键类别区间。

        返回字段:
            - in_dict.pos: float, (N, 3), 原地写回弹簧修正坐标；fixed_pos=1 的行保持不变。

        类别边界:
            - ``0`` 是无键；``1 <= type < num_edge_types`` 被视为真实化学键；等于 ``num_edge_types`` 的额外掩码类不属于键。
        """
        if batch['fixed_halfdist'].sum() > 0:  # if has fixed_dist, no need to spring bonds since lens are not perturbed
            return in_dict
        
        # ``spring_in``：dict/EasyDict，当前弹簧后处理的完整配置。
        spring_in = self.config['spring_in']
        # ``iters``：int，坐标更新次数。
        iters = spring_in['iters']
        # ``lr``：float，每次显式梯度更新的无量纲乘子。
        lr = spring_in['lr']

        # ``in_halfedge``：LongTensor，形状为 (H,)；与无向 halfedge_index 的 H 列对齐的当前输入类别。
        in_halfedge = in_dict['halfedge']
        # ``halfedge_index``：LongTensor，形状为 (2, H)；每列是一个只出现一次的无向原子对端点。
        halfedge_index = batch['halfedge_index']
        # ``is_bond``：BoolTensor，形状为 (H,)；标出当前输入中类别位于真实化学键区间的半边。
        is_bond = (in_halfedge > 0) & (in_halfedge < self.num_edge_types)
        if getattr(spring_in, 'inout_bond', False):
            # ``halfedge_type``：LongTensor，形状为 (H,)；参考半边类别，与 in_halfedge 逐半边对齐。
            halfedge_type = batch['halfedge_type']
            # ``is_bond``：BoolTensor，形状为 (H,)；进一步要求参考类别也位于真实化学键区间。
            is_bond = is_bond & (halfedge_type > 0) & (halfedge_type < self.num_edge_types)
        # ``halfbond_index``：LongTensor，形状为 (2, E_b)；从 H 条半边筛出的预测键；每个无向键当前只出现一次。
        halfbond_index = halfedge_index[:, is_bond]
        # ``bond_index``：LongTensor，形状为 (2, 2E_b)；把端点行翻转后拼接，使每个键按两个方向参与梯度累计。
        bond_index = torch.cat([halfbond_index, halfbond_index.flip(0)], dim=-1)

        # ``in_pos``：FloatTensor，形状为 (N, 3)；待原地更新的当前输入坐标，单位 Å。
        in_pos = in_dict['pos']
        # ``fixed_pos``：BoolTensor，形状为 (N,)；True 的原子在每次更新时被排除。
        fixed_pos = batch['fixed_pos'].bool()
        

        # spring the pos
        with torch.enable_grad():
            # in_pos.requires_grad_(True)
            for i in range(iters):
                # ``value_min``：float，不惩罚键长区间的下界，单位 Å。
                # ``value_max``：float，不惩罚键长区间的上界，单位 Å。
                value_min, value_max = spring_in['min'], spring_in['max']
                # ``std``：float，诊断损失的长度归一化尺度，单位 Å。
                std = spring_in['std']
                # ``lens``：FloatTensor，形状为 (2E_b,)；每条有向键当前端点距离，单位 Å；同一无向键的两个值相同。
                lens = torch.linalg.norm(in_pos[bond_index[0]] - in_pos[bond_index[1]], dim=-1)
                # ``square``：FloatTensor，形状为 (2E_b,)；仅短于下界的键保留平方超限量，单位 Å²。
                square = torch.where(lens < value_min, (lens - value_min) ** 2, torch.zeros_like(lens))
                # ``square``：FloatTensor，形状为 (2E_b,)；进一步把长于上界的键替换为平方超限量，其余保持前一步结果。
                square = torch.where(lens > value_max, (lens - value_max) ** 2, square)
                # ``loss``：float 标量，平均绝对超限量/std；当前值不参与坐标更新，仅保留为诊断计算。
                loss = torch.mean( square.sqrt() / std)
                
                # ``grad``：FloatTensor，形状为 (N, 3)；区间键长目标对每个原子坐标的聚合修正方向。
                grad = grad_len_to_pos(in_pos, bond_index, spring_in)
                # grad = torch.autograd.grad(loss, in_pos)[0]
                in_pos[~fixed_pos] = in_pos[~fixed_pos] - grad[~fixed_pos] * lr

        # ``in_dict.pos``：dict.pos: float, (N, 3)，写回完成 iters 次更新后的坐标引用，单位 Å。
        in_dict['pos'] = in_pos
        return in_dict
    
    def additional_process(self, batch, in_dict):
        """返回未改动的 ``in_dict``；子类可在 fixed 硬覆盖前增加任务特定预处理。"""
        return in_dict

    @torch.no_grad()
    def __call__(self, batch, step=None):
        """
        生成当前训练样本或采样步的模型输入，并在最后强制恢复所有 fixed 字段。

        输入参数:
            - batch: PyG Data/Batch, 原地承载当前状态、结构注释、fixed 掩码及本步新增输入。
            - step: float|None, 规范采样进度；``1`` 表示先验初始化，训练传 None。

        输入字段:
            - node_type: int64, (N,), 当前原子类别。
            - node_pos: float, (N, 3), 当前配体坐标，单位 Å。
            - halfedge_type: int64, (H,), 当前无向半边类别。
            - fixed_node: 0/1, (N,), 原子类别硬固定掩码。
            - fixed_pos: 0/1, (N,), 原子坐标硬固定掩码。
            - fixed_halfedge: 0/1, (H,), 半边类别硬固定掩码。

        输出字段:
            - batch.node_in: int64, (N,), 经过先验、可选处理及 fixed_node 恢复后的模型输入类别。
            - batch.pos_in: float, (N, 3), 经过先验、可选处理及 fixed_pos 恢复后的模型输入坐标，单位 Å。
            - batch.halfedge_in: int64, (H,), 经过先验、可选处理及 fixed_halfedge 恢复后的模型输入类别。

        # see me 处理顺序:
            - 复制当前状态 -> 采信息等级 -> 按 step 选择先验初始化或局部加噪 -> (在 docking&free不会触发) spring -> [在 docking&free下会触发]同构重排 -> (在 docking&free不会触发) 任务预处理 -> fixed 硬恢复。
        """
        # node_pos_protect = deepcopy(batch.node_pos.detach().clone())
        # # check mode and inputs consistency
        # from_batch, step = self._check_input(batch, outputs, step)
        # device = batch.node_type.device
        
        # ``node_type``：LongTensor，形状为 (N,)，当前原子类别的 detach+clone 副本。
        # ``node_pos``：FloatTensor，形状为 (N, 3)，当前配体坐标的 detach+clone 副本，单位 Å。
        # ``halfedge_type``：LongTensor，形状为 (H,)，当前半边类别的 detach+clone 副本。
        node_type, node_pos, halfedge_type = self._fetch_data(batch)

        # ``level_dict``：dict[str, Tensor]，每个叶子与一种运动自由度的实体轴对齐，值域通常为 [0, 1]。
        level_dict = self.sample_level(step, batch)
        
        # # adding noise after moveing all nodes to their origins
        if step == 1:
            # ``in_dict``：dict，首步从任务先验初始化 level=0 自由度；类别和坐标叶子的形状由任务子类保证。
            in_dict = self.add_noise(node_type, node_pos, halfedge_type, batch,
                from_prior=True, level_dict=level_dict)  # here only the key of level_dict=level_dict is useful
        else:
            # level_dict = {k:torch.ones_like(v) for k,v in level_dict.items()}
            # ``in_dict``：dict，训练或非首采样步在当前状态周围按 level 混合先验噪声。
            in_dict = self.add_noise(node_type, node_pos, halfedge_type, batch,
                                        from_prior=False, level_dict=level_dict)
        
        if self.config.get('spring_in', False):
            # ``in_dict``：三叶 dict，pos 可能被键长区间弹簧更新，node/halfedge 保持原叶子。
            in_dict = self.spring_in_pos(batch, in_dict)

        if self.config.get('reassign_in', False):
            # ``in_dict``：三叶 dict，训练构象/docking 可按同构原子排列重排 pos 行。
            in_dict = self.reassign_in_node(batch, in_dict)
            
        # ``in_dict``：三叶 dict，执行任务子类的已知坐标预处理。
        in_dict = self.additional_process(batch, in_dict)
            
        # see me: 这些逻辑也有些别扭。类似的很多事实意味着新版代码可能需要大重构————即便不改变科学逻辑
        # fixed_node 对齐 N 个原子；先比较后覆盖，避免 fixed 原子类别被先验或预处理改变。
        if not (batch.node_type[batch['fixed_node']==1] ==  in_dict['node'][batch['fixed_node']==1]).all():
            # print('Force fixed_node to be fixed')
            # ``is_fixed``：BoolTensor，形状为 (N,)；True 行从原 node_type 恢复到 in_dict.node。
            is_fixed = (batch['fixed_node']==1)
            # ``in_dict.node``：(N,) 掩码写入，恢复 fixed 原子类别。
            in_dict['node'][is_fixed] = batch.node_type[is_fixed]
        # assert (batch.node_pos[(batch['fixed_pos']==1)]
        #         - in_dict['pos'][batch['fixed_pos']==1]).abs().sum() < 1e-4
        # fixed_pos 对齐 N 个原子；阈值 1e-4 比较所有 fixed 坐标绝对误差之和，坐标单位 Å。
        if not (batch.node_pos[(batch['fixed_pos']==1)]
                - in_dict['pos'][batch['fixed_pos']==1]).abs().sum() < 1e-4:
            # ``in_dict.pos``：(N, 3) 掩码写入，恢复 fixed 原子坐标，单位 Å。
            in_dict['pos'][batch['fixed_pos']==1] = batch.node_pos[batch['fixed_pos']==1]
        # assert (batch.halfedge_type[batch['fixed_halfedge']==1]
        #         ==  in_dict['halfedge'][batch['fixed_halfedge']==1]).all()
        # fixed_halfedge 对齐 H 条无向半边；逐元素近似比较后恢复原类别。
        if not torch.isclose(batch.halfedge_type[batch['fixed_halfedge']==1],
                in_dict['halfedge'][batch['fixed_halfedge']==1]).all():
            # print('Force fixed_halfedge to be fixed')
            # ``is_fixed``：BoolTensor，形状为 (H,)；True 半边从原 halfedge_type 恢复到 in_dict.halfedge。
            is_fixed = (batch['fixed_halfedge']==1)
            # ``in_dict.halfedge``：(H,) 掩码写入，恢复 fixed 无向半边类别。
            in_dict['halfedge'][is_fixed] = batch.halfedge_type[is_fixed]

        # ``key``：str，当前 noiser 短叶名，取 ``node``、``pos`` 或 ``halfedge``。
        # ``value``：Tensor，当前短叶对应的加噪输入值。
        # 将每个短叶名追加 ``_in`` 后缀，并把对应值原地挂到同一 Batch。
        batch.update({f'{key}_in':value for key, value in in_dict.items()})
        
        return batch

    
    def steps_loop(self, add_last=False):
        """
        按从高噪声到低噪声的顺序产生规范采样进度。

        输入参数:
            - add_last: bool, False 产生 S 个正进度；True 额外在末尾产生 0。

        产生值:
            - step: float, ``init_step * k / S``，k 从 S(=self.num_step) 递减到 1；``add_last=True`` 时再产生 k=0。

        边界:
            - 默认序列包含 ``init_step``，不包含 0；当 ``init_step=1`` 时首值会触发 ``from_prior=True``。
        """
        # for step in np.linspace(0, 1, self.num_steps)[::-1]:
        if add_last:
            # ``steps``：LongTensor，形状为 (S+1,)；升序整数 0..S；随后反转为 S..0。
            steps = np.arange(0, self.num_steps+1) # 0, 1, ..., num_steps
        else:
            # ``steps``：LongTensor，形状为 (S,)；升序整数 1..S；随后反转为 S..1。
            steps = np.arange(1, self.num_steps+1)  # 1, 2, ..., num_steps
        # ``step``：NumPy 整数，按 S 到 1（或到 0）递减的离散步号，随后归一化到 ``[0, init_step]``。
        for step in steps[::-1]:
            yield (step / self.num_steps) * self.init_step # include 1 but not 0

# XXX
# TODO: 目前的原版实现中, 有太多关于类的嵌套。但是在后续微调中，我们会：不使用多数据集、多任务，关注docking任务;  只使用free版本的噪声，所以在这些配置中，代码的嵌套次数和复杂度都理应得到优化。
@register_sample_noise('conf')
class ConfSampleNoiser(BaseSampleNoiser):
    """
    为构象生成与小分子 docking 生成仅坐标噪声, 并把坐标预测投影回所选运动自由度.

    形状中 N 是当前样本或批次的配体原子总数, H 是无向半边数, B 是分子数, T 是可旋转键数, W 是随扭转运动的原子注释数, N_d 是刚性域登记的原子数.

    构造参数 ``config``:
        - prior: dict|``from_train``, 坐标先验配置; 后者读取 ``ref_prior_config``.
        - level: dict, ``MolInfoLevel`` 的训练分布或采样 step 映射配置.
        - pre_process: None|str, 可选输入硬约束; 小分子锚点先验使用 ``fix_closest``.
        - post_process: None|str|dict, 模型输出到下一步坐标的约束策略.
        - recenter: str, free 构象加噪后的整体刚体处理; ``norotate`` 额外用 Kabsch 移除全局旋转.
        - spring_in: dict|False, 继承自基类的可选键长弹簧配置.
        - reassign_in: bool, 是否在训练加噪后选择最接近真值的同构原子排列.

    运动模式 ``batch.task_setting``:
        - free: 每个原子独立三维位移; level 字段 ``pos`` 与 N 个原子对齐.
        - flexible: 每图整体平移/旋转加每条可旋转键扭转; level 字段为 ``trans``、``rot``、``tor``.
        - torsional: 只改变可旋转键扭转角; level 字段只有 ``tor``.
        - rigid: 每图只做整体平移和旋转; level 字段为 ``trans``、``rot``.

    构象与 docking 的关键差异:
        - ``task=conf`` 在先验前后按图减去配体几何中心, 消除无意义的整体平移.
        - ``task=dock`` 不重居中, 因而保留配体相对口袋的整体平移与旋转作为待预测自由度.

    模型输出契约:
        - outputs.pred_pos: float, (N, 3), 网络直接预测坐标, 单位 Å.
        - batch.node_pos: float, (N, 3), ``outputs2batch`` 写回的下一迭代坐标, 单位 Å.
        - 原子类别 ``node_type`` 和半边类别 ``halfedge_type`` 在本类中保持不变.
    """
    def __init__(self,
        config, num_node_types, num_edge_types,
        mode='sample', device='cpu', ref_config=None, task_name='conf',
        **kwargs
    ):
        """装配原坐标先验与信息等级, 预处理、同构重分配和固定字段恢复保持原配置.

        config 字段及自由度含义见类 Docstring; num_node_types、num_edge_types 是原离散词表大小, 坐标任务不增加类别噪声.
        mode 决定训练或采样的信息等级来源, device 决定先验张量设备; ref_config 仅用于原有 prior='from_train' 配置读取.
        """
        super().__init__(task_name, config, num_node_types, num_edge_types,
            mode, device, ref_config, pos_only=True, **kwargs)
        
        # ``prior_config.pos.name``: str, allpos 将四种坐标自由度交给 AllPosPrior, 本项目free只使用其中的pos.
        # ``prior_config.pos.pos``: Mapping|None, free 模式逐原子高斯先验.
        # ``prior_config.pos.translation``: Mapping|None, flexible/rigid 模式整体平移先验.
        # ``prior_config.pos.rotation``: Mapping|None, flexible/rigid 模式整体旋转先验.
        # ``prior_config.pos.torsional``: Mapping|None, flexible/torsional 模式内部扭转先验.
        # ``prior_config``: EasyDict, 直接取 ``config.prior``, 或在其为 ``from_train`` 时取训练同名任务的上述先验字段.
        prior_config = config.prior if config.prior != 'from_train' else self.ref_prior_config
        # ``self.prior``: MolPrior, pos_only=True 关闭 node/halfedge 离散先验, 只实例化坐标自由度先验.
        self.prior = MolPrior(prior_config, num_node_types, num_edge_types, 
                              pos_only=True).to(device)

        # ``self.level``: MolInfoLevel, 将训练随机采样或采样 step 转为各自由度的 [0, 1] 信息保留量.
        self.level = MolInfoLevel(config.level, device=device, mode=mode)
        
        # ``self.pre_process``: str|None, 在先验之后、fixed 硬恢复之前约束输入坐标; 锚点先验使用 ``fix_closest``.
        self.pre_process = config.get('pre_process', None)
        # ``self.post_process``: None 时直接写回 ``pred_pos``.
        # ``self.post_process``: str 时可选择 correct_pos/correct_dist/correct_center/correct_closest/flex_to_free/transparency.
        # ``self.post_process.name``: str, Mapping 模式的 use 约束名称.
        # ``self.post_process.atom_space[*].atom``: int, Mapping 模式约束的图内原子编号.
        # ``self.post_process.atom_space[*].coord``: list[float]|缺省, 长度为 3 的已知世界坐标, 单位 Å.
        # ``self.post_process.atom_space[*].radius``: float|缺省, 允许位置偏差半径, 单位 Å.
        # ``self.post_process``: None|str|EasyDict, 把网络坐标投影或修正成下一步 ``node_pos``; 标准构象/docking 配置为 None.
        self.post_process = config.get('post_process', None)

    def sample_level(self, step, batch):
        """
        按运动模式为每个实际自由度采样信息保留量。

        输入参数:
            - step: float|None, sample 模式的规范进度或 train 模式的 None。
            - batch.node_type: int64, (N,), 仅用长度 N 计数 free 原子位移自由度。
            - batch.num_graphs: int B, PyG Batch 中的分子图数；单个 Data 缺省为 1。
            - batch.tor_bonds_anno: int64, (T, 3), 每行一条可旋转键注释；仅用行数 T。

        返回值 ``level_dict``:
            - pos: float, (N,), free 模式每个原子位移的信息等级。
            - trans: float, (B,), flexible/rigid 模式每个图整体平移的信息等级。
            - rot: float, (B,), flexible/rigid 模式每个图整体旋转的信息等级。
            - tor: float, (T,), flexible/torsional 模式每条可旋转键的信息等级。

        返回键顺序:
            - flexible 固定为 ``trans -> rot -> tor``；rigid 固定为 ``trans -> rot``，与 ``MolInfoLevel.sample_for_mol`` 的 kwargs 插入顺序一致。
        """
        # ``setting``：str|dict，本类预期四个字符串模式；未知值会使 level_dict 未定义并在返回时报错。
        setting = self._get_setting(batch)
        # setting = 'free'
        if setting == 'free':
            # ``level_pos``：FloatTensor，形状为 (N,)；逐原子独立位移的信息保留量。
            level_pos = self.level.sample_for_mol(step,
                n_pos=batch['node_type'].shape[0],
            )
            # ``level_dict``：dict.pos: float, (N,)，free 逐原子位移的信息等级。
            level_dict = {'pos': level_pos}
        
        # see me: 下面不需要关注, 因为我们只使用 free 噪声
        elif setting == 'flexible':
            # ``n_trans``：int B，每个图对应一个整体平移向量。
            n_trans = getattr(batch, 'num_graphs', 1)
            # ``n_rot``：int B，每个图对应一个整体旋转。
            n_rot = getattr(batch, 'num_graphs', 1)
            # ``n_tor``：int T，可旋转键注释行数，也是独立扭转自由度数。
            n_tor = batch['tor_bonds_anno'].shape[0]
            # ``L_trans``：FloatTensor，形状为 (B,)，逐图整体平移信息等级。
            # ``L_rot``：FloatTensor，形状为 (B,)，逐图整体旋转信息等级。
            # ``L_tor``：FloatTensor，形状为 (T,)，逐可旋转键扭转信息等级。
            L_trans, L_rot, L_tor = self.level.sample_for_mol(step,
                n_trans=n_trans,
                n_rot=n_rot,
                n_tor=n_tor,
            )
            # ``level_dict.trans``：FloatTensor，形状为 (B,)，逐图整体平移信息等级。
            # ``level_dict.rot``：FloatTensor，形状为 (B,)，逐图整体旋转信息等级。
            # ``level_dict.tor``：FloatTensor，形状为 (T,)，逐可旋转键扭转信息等级。
            level_dict = {
                'trans': L_trans,
                'rot': L_rot,
                'tor': L_tor,
            }
        elif setting == 'torsional':
            # ``n_tor``：int T，只统计扭转自由度，不创建整体刚体自由度。
            n_tor = batch['tor_bonds_anno'].shape[0]
            # ``level_tor``：FloatTensor，形状为 (T,)；逐可旋转键的信息保留量。
            level_tor = self.level.sample_for_mol(step,
                n_tor=n_tor,
            )
            # ``level_dict``：dict.tor: float, (T,)，torsional 模式唯一信息等级叶子。
            level_dict = {'tor': level_tor}
        elif setting == 'rigid':
            # ``n_trans``：int B，每图一个整体平移向量。
            n_trans = getattr(batch, 'num_graphs', 1)
            # ``n_rot``：int B，每图一个整体旋转。
            n_rot = getattr(batch, 'num_graphs', 1)
            # ``L_trans``：FloatTensor，形状为 (B,)，逐图整体平移信息等级。
            # ``L_rot``：FloatTensor，形状为 (B,)，逐图整体旋转信息等级。
            L_trans, L_rot = self.level.sample_for_mol(step,
                n_trans=n_trans,
                n_rot=n_rot,
            )
            # ``level_dict.trans``：FloatTensor，形状为 (B,)，逐图整体平移信息等级。
            # ``level_dict.rot``：FloatTensor，形状为 (B,)，逐图整体旋转信息等级。
            level_dict = {
                'trans': L_trans,
                'rot': L_rot,
            }
        return level_dict

    def add_noise(self, node_type, node_pos, halfedge_type, batch,
                   from_prior=False, level_dict=None):
        """
        按 ``task_setting`` 装配坐标先验参数, 并生成当前步配体坐标输入.

        输入参数:
            - node_type: int64, (N,), 当前原子类别; 本任务不加类别噪声, 返回时原样引用.
            - node_pos: float, (N, 3), 当前配体坐标, 单位 Å.
            - halfedge_type: int64, (H,), 当前半边类别; 本任务不加类别噪声, 返回时原样引用.
            - batch.task: str|list[str], ``conf`` 或 ``dock``.
            - batch.task_setting: str|list[str], ``free``、``flexible``、``torsional`` 或 ``rigid``.
            - batch.node_type_batch: int64, (N,), 可选; 每个原子所属图号, 范围 0..B-1.
            - batch.tor_bonds_anno: int64, (T, 3), ``[BFS 序号, 远端轴原子, 近端轴原子]``.
            - batch.twisted_nodes_anno: int64, (W, 2), ``[扭转行号, 随该键转动的原子]``.
            - batch.domain_node_index: int64, (2, N_d), ``[刚性域号, 域内原子]``; 结构化先验据此按域展开变换.
            - batch.num_nodes: int, 单个 Data 不含 node_type_batch 时使用的原子数.
            - from_prior: bool, True 从先验初始化; False 在当前坐标周围按 level 加噪.
            - level_dict: dict[str, Tensor], 键和形状由 ``sample_level`` 定义.
            - level_dict.pos: float, (N,), free 模式的真实信息保留量; 同一分子的原子共享数值, 实际噪声强度为s=1-level_dict.pos, 不用采样步号代替.

        传给 ``MolPrior`` 的附加字段:
            - node_type: None, 明确关闭原子类别先验输入.
            - halfedge_type: None, 明确关闭半边类别先验输入.
            - mol_size: int64, (N,), free 模式中每个原子位置重复其所属分子的原子数.
            - tor_bonds_anno: int64, (T, 3), flexible/torsional 模式使用.
            - twisted_nodes_anno: int64, (W, 2), flexible/torsional 模式使用.
            - domain_node_index: int64, (2, N_d), flexible/torsional/rigid 模式使用.

        返回值 ``in_dict``:
            - node: int64, (N,), 与输入 node_type 相同.
            - pos: float, (N, 3), 加噪、必要时重居中并可选去全局旋转后的坐标, 单位 Å.
            - halfedge: int64, (H,), 与输入 halfedge_type 相同.

        本函数返回后, BaseSampleNoiser.__call__再执行原同构重分配与fixed恢复; dock不对候选质心作额外调整.
        """
        # ``task``: str, 决定是否在加噪前后移除配体整体平移; conf 会, dock 不会.
        task = self._get_task(batch)
        # ``setting``: str, 决定自由原子噪声或结构化 trans/rot/tor 先验.
        setting = self._get_setting(batch)  # return dict_list2item(batch['task_setting'])
        # ``additional_kwargs.node_type``: None, pos_only 路径明确不向原子类别先验传标签.
        # ``additional_kwargs.halfedge_type``: None, pos_only 路径明确不向半边类别先验传标签.
        # ``additional_kwargs``: dict[str, object], 随后按 setting 追加坐标先验需要的字段.
        additional_kwargs = {
            'node_type': None,
            'halfedge_type': None,
        }
        if setting == 'free':
            if 'node_type_batch' in batch:
                # ``mol_size``: LongTensor, 形状为 (B,); 每个图的原子计数; 由 N 个图号直方图得到.
                mol_size = torch.bincount(batch['node_type_batch'])
                # ``mol_size``: LongTensor, 形状为 (N,); 原子所在图(配体)的原子个数
                mol_size = mol_size[batch['node_type_batch']]
            else:
                # ``mol_size``: LongTensor, 形状为 (N,); 单图时每个原子位置都填相同的 num_nodes=N.
                mol_size = batch['num_nodes'] * torch.ones(node_type.shape[0], dtype=torch.long, device=node_type.device)
            additional_kwargs.update({
                'mol_size': mol_size,
            })

        elif setting in ['flexible', 'torsional']:
            additional_kwargs.update({
                'tor_bonds_anno': batch['tor_bonds_anno'],
                'twisted_nodes_anno': batch['twisted_nodes_anno'],
                'domain_node_index': batch['domain_node_index'],
            })
        elif setting == 'rigid':
            additional_kwargs.update({
                'domain_node_index': batch['domain_node_index'],
            })
        # # recenter before add_noise
        if (task == 'conf'):
            # ``batch_node``: LongTensor, 形状为 (N,); 每个原子所属图号; 单图 Data 缺省为全 0.
            batch_node = getattr(batch, 'node_type_batch',
                        torch.zeros(node_type.shape[0], dtype=torch.long, device=node_pos.device))
            # ``node_pos_center``: FloatTensor, 形状为 (N, 3); 每个原子位置重复其所属图的几何中心, 单位 Å.
            node_pos_center = scatter_mean(node_pos, batch_node, dim=0, dim_size=batch_node.max()+1)[batch_node]
            # ``node_pos``: [N, 3] -> [N, 3]; 逐图减中心, 转入各分子质心位于原点的局部坐标系.
            node_pos = node_pos - node_pos_center
            if self.mode == 'train':
                batch.update({'node_pos': node_pos.clone()})  # can be ommited since featurizer has done this
            

        # ``pos_in``: FloatTensor, 形状为 (N, 3); 按 free 或 trans/rot/tor 先验生成的坐标, 单位 Å.
        pos_in = self.prior.add_noise(node_pos=node_pos.clone(), level_dict=level_dict,
                                      from_prior=from_prior, **additional_kwargs,)

        # # recenter after add_noise
        if (task == 'conf'):
            # ``batch_node``: LongTensor, 形状为 (N,); 与上面的分图中心计算相同; 此处重新取得以保持分支局部完整.
            batch_node = getattr(batch, 'node_type_batch',
                        torch.zeros(node_type.shape[0], dtype=torch.long, device=node_pos.device))
            # ``pos_in_center``: FloatTensor, 形状为 (N, 3); 每个原子位置重复其所属图的加噪后几何中心, 单位 Å.
            pos_in_center = scatter_mean(pos_in, batch_node, dim=0, dim_size=batch_node.max()+1)[batch_node]
            # ``pos_in``: [N, 3] -> [N, 3]; 再次逐图减中心, 消除随机先验产生的数值整体平移.
            pos_in = pos_in - pos_in_center
            if (setting == 'free') and (self.config.get('recenter', 'default') == 'norotate'):
                # ``domain_index``: LongTensor, 形状为 (N,); 把每个图当作一个 Kabsch 对齐域.
                domain_index = batch_node
                # ``global_rot``: FloatTensor, 形状为 (B, 3, 3), 把 ``pos_in`` 对齐到 ``node_pos`` 的逐图无反射旋转矩阵.
                # ``global_trans``: FloatTensor, 形状为 (B, 1, 3), 同一逐图刚体对齐的行向量平移, 单位 Å.
                global_rot, global_trans = kabsch_flatten(pos_in, node_pos, domain_index)
        
                # ``pos_corrected_expand``: FloatTensor, 形状为 (N, 1, 3); [N, 3] -> [N, 1, 3], 逐原子右乘所属图旋转矩阵并加平移.
                pos_corrected_expand = torch.matmul(
                    pos_in[:, None, :],
                    global_rot.transpose(1, 2)[domain_index]
                ) + global_trans[domain_index]

                # ``pos_in``: FloatTensor, 形状为 (N, 3); 移除 free 噪声相对真值的整体刚体旋转后, 保留内部构象变化.
                pos_in = pos_corrected_expand.squeeze(1)
                
        
        # ``in_dict.node``: LongTensor, 形状为 (N,), 与输入 ``node_type`` 相同.
        # ``in_dict.pos``: FloatTensor, 形状为 (N, 3), 加噪并完成任务级预处理的配体坐标, 单位 Å.
        # ``in_dict.halfedge``: LongTensor, 形状为 (H,), 与输入 ``halfedge_type`` 相同.
        in_dict = {
            'node': node_type,
            'pos': pos_in,
            'halfedge': halfedge_type,
        }
        
        return in_dict

    def reassign_in_node(self, batch, in_dict):
        """
        按照预先计算好的自同构映射, 在训练加噪后选择与真值最接近的原子排列。

        输入字段:
            - batch.matches_iso: int64/array-like, (M, S), M 种同构映射；每行给出同一组 S 个对称原子在该映射下的全局索引。
            - batch.node_pos: float, (N, 3), 当前训练真值坐标，单位 Å。
            - in_dict.node: int64, (N,), 加噪后的原子类别。
            - in_dict.pos: float, (N, 3), 加噪后的坐标，单位 Å。

        返回字段:
            - in_dict.pos: float, (N, 3), ``base_order`` 行被最优同构排列的坐标覆盖。

        评分定义:
            - 对每个排列先计算 S 个原子的欧氏距离，再对原子取算术平均；变量沿用 ``rmsd`` 名称，但没有平方后平均再开根，并非严格 RMSD。
        """
        if (self.mode == 'train') and ('is_atom_remain' not in batch):  # not applicable for cutt peptide
            # ``matches_iso``：int64/array-like, (M, S)，同一对称原子集合的 M 种全局索引排列。
            matches_iso = batch['matches_iso']
            # ``base_order``：int64/ndarray, (S,)，取首个排列并升序排序，作为写回坐标的规范目标顺序。
            base_order = np.sort(matches_iso[0])
            # assert (np.diff(base_order)>0).sum(), 'base_order should be in ascending order'
            # ``node_pos``：FloatTensor，形状为 (S, 3)；规范顺序下的真值对称原子坐标，单位 Å。
            node_pos = batch['node_pos'][base_order]
            # ``in_pos_orig``：FloatTensor，形状为 (N, 3)；全部加噪输入坐标，单位 Å。
            in_pos_orig = in_dict['pos']
            # ``in_pos_syms``：FloatTensor，形状为 (M, S, 3)；按每个同构映射重排出的候选对称原子坐标，单位 Å。
            in_pos_syms = in_pos_orig[matches_iso, ...]
            # ``rmsd``：FloatTensor，形状为 (M, S)；各候选排列中逐原子到规范真值位置的欧氏距离，单位 Å。
            rmsd = torch.norm(node_pos[None] - in_pos_syms, p=2, dim=-1)
            # ``rmsd_mean``：FloatTensor，形状为 (M,)；每个排列的平均逐原子欧氏距离，单位 Å。
            rmsd_mean = torch.mean(rmsd, dim=1)
            # ``min_index``：int64 标量，平均距离最小的候选排列行号。
            min_index = torch.argmin(rmsd_mean, dim=0)
            # ``in_dict.pos``：(S, 3) 写回，把规范 base_order 行替换为最优同构排列的加噪坐标。
            in_dict['pos'][base_order] = in_pos_syms[min_index]
            assert (in_dict['node'][base_order] == in_dict['node'][matches_iso[min_index]]).all(), 'node type not symmetry'

        return in_dict
    
    # see me: 对于 docking + free 配置, 默认配置下不会触发
    def additional_process(self, batch, in_dict):
        """
        在基类 fixed 恢复前，为锚点先验选取并固定最靠近口袋的配体原子。

        输入字段:
            - batch.node_type_batch: int64, (N,), 配体原子所属图号。
            - batch.pocket_pos_batch: int64, (P,), 口袋点所属图号。
            - batch.gt_node_pos: float, (N, 3), 已知真值配体坐标，单位 Å。
            - batch.pocket_pos: float, (P, 3), 口袋坐标，单位 Å，和 gt_node_pos 位于同一口袋局部坐标系。
            - batch.node_closest: int64, (B,), 可选缓存；每图离任一口袋点最近的配体原子全局索引(node_closest_this + (batch['node_type_batch']<i_mol).sum())
            - batch.fixed_pos: 0/1, (N,), 坐标硬固定掩码。
            - batch.node_pos: float, (N, 3), 当前参考坐标，单位 Å。
            - in_dict.pos: float, (N, 3), 待约束的加噪输入坐标，单位 Å。

        输出副作用:
            - ``pre_process=fix_closest`` 时缓存 ``node_closest``，把这些原子的 ``fixed_pos`` 置 1，并把 ``node_pos/in_dict.pos`` 恢复为 ``gt_node_pos``。
        """
        if self.pre_process is not None:
            if self.pre_process == 'fix_closest':   # fixed, cannot be changed
                if 'node_closest' in batch:
                    # ``node_closest``：LongTensor，形状为 (B,)；缓存的每图最近配体原子全局索引。
                    node_closest = batch['node_closest']
                else:
                    # ``node_closest``：list[scalar Tensor]，循环中暂存 B 个批次全局原子索引。
                    node_closest = []
                    # ``i_mol``：int，当前图号，同时筛选配体原子与口袋原子的 batch 归属向量。
                    for i_mol in range(batch['node_type_batch'].max().item()+1):
                        # ``dist_mat``：FloatTensor，形状为 (N_i, P_i)；第 i 图每个配体原子到每个口袋点的欧氏距离，单位 Å。
                        dist_mat = torch.cdist(batch['gt_node_pos'][batch['node_type_batch']==i_mol],
                                                  batch['pocket_pos'][batch['pocket_pos_batch']==i_mol])
                        # ``node_closest_this``：int64 标量，先对每个配体原子取最近口袋距离，再取距离最小的图内配体原子索引。
                        node_closest_this = torch.argmin(dist_mat.min(dim=1)[0], dim=0)
                        # ``node_closest_this``：int64 标量，累加前序图原子数，把图内索引转换为批次全局索引。
                        node_closest_this = node_closest_this + (batch['node_type_batch']<i_mol).sum()
                        node_closest.append(node_closest_this)
                    # ``node_closest``：LongTensor，形状为 (B,)；每图一个最近配体原子全局索引。
                    node_closest = torch.stack(node_closest)
                    # ``batch.node_closest``：LongTensor，形状为 (B,)；缓存最近原子全局索引，后续步骤避免重复 cdist。
                    batch['node_closest'] = node_closest
                # ``batch.fixed_pos``：(N,) 索引写入，把每图最近原子设为坐标 fixed。
                batch['fixed_pos'][node_closest] = 1
                # ``batch.node_pos``：(B, 3) 索引写入，先把当前参考坐标恢复为真值，满足基类 fixed 一致性检查。
                batch['node_pos'][node_closest] = batch['gt_node_pos'][node_closest].clone()  # for following check of node_pos == in_pos
                # ``in_dict.pos``：(B, 3) 索引写入，把最近原子输入坐标恢复为真值，单位 Å。
                in_dict['pos'][node_closest] = batch['gt_node_pos'][node_closest].clone()
            elif self.pre_process == 'fix_some':  # fix some atoms, cannot be changed
                # ``fixed_pos``：BoolTensor，形状为 (N,)；True 的原子输入坐标将在此处恢复为已知参考值。
                fixed_pos = batch['fixed_pos'].bool()
                if 'gt_node_pos' in  batch:
                    # ``in_dict.pos``：(N, 3) 掩码写入，从真值坐标恢复全部 fixed 原子。
                    in_dict['pos'][fixed_pos] = batch['gt_node_pos'][fixed_pos].clone()
                else:
                    # ``in_dict.pos``：(N, 3) 掩码写入，无真值字段时从当前 node_pos 恢复 fixed 原子。
                    in_dict['pos'][fixed_pos] = batch['node_pos'][fixed_pos].clone()
            else:
                raise NotImplementedError('not implemented for pre_process:', self.pre_process)
        return in_dict

    # see me: 对于 docking & free, 这个原本专门用于推理阶段的函数没有用, 也就是退化为: pred_pos = outputs['pred_pos']  batch['node_pos'] = pred_pos.clone()
    def outputs2batch(self, batch, outputs):
        """
        把网络坐标预测转换为下一采样步的 ``batch.node_pos``，并最后恢复 fixed 原子坐标。
        outputs2batch() 只用于推理迭代；训练阶段直接计算 pred_pos 对真值的损失，不经过这里。

        输入 ``outputs``:
            - pred_pos: float, (N, 3), 网络直接预测的配体坐标，单位 Å。
            - pred_pos.grad: 不使用；本函数在采样 ``torch.no_grad`` 路径中执行。

        公共输入字段:
            - batch.task_setting: str|list[str], 当前运动模式。
            - batch.step: float, 当前规范采样进度；某些后处理据此启停。
            - batch.node_pos: float, (N, 3), 当前步参考坐标，单位 Å。
            - batch.fixed_pos: 0/1, (N,), 最终必须保持当前 node_pos 的原子掩码。
            - batch.node_type_batch: int64, (N,), 每个配体原子所属图号。
            - batch.gt_node_pos: float, (N, 3), 可选真值/已知坐标，单位 Å。

        ``post_process=None`` 分支:
            - free: 直接采用 outputs.pred_pos。
            - flexible/torsional: 调用 ``correct_pos_batch``，把网络预测投影为保持刚性域并允许扭转的坐标。
            - rigid: 调用 ``correct_pos_batch_no_tor``，只保留每域整体旋转和平移。

        ``post_process=know_some`` 嵌套字段:
            - name: str, 必须为 ``know_some``。
            - atom_space: list[dict], 每项描述一个需要固定或限制的模板原子。
            - atom_space[*].atom: int, 单分子模板中的局部原子索引。
            - atom_space[*].coord: list[float] 长度 3, 可选绝对坐标；减 pocket_center 后转为模型局部坐标，单位 Å。
            - atom_space[*].radius: float, 低 step 阶段允许预测偏离目标的最大半径，单位 Å。

        字符串 ``post_process`` 支持:
            - correct_pos: 在配置的 step 闭区间内按真值刚性/扭转约束投影。
            - correct_dist: 在阈值 step 前按 fixed_halfdist 修正指定原子对距离。
            - correct_center: 当预测中心与真值中心距离超过半径时，把整图平移到真值中心。
            - correct_closest: 当已知最近原子的偏差超过半径时，用其位移平移整图。
            - flex_to_free: 高 step 用 flexible 投影，低 step 切换 task_setting 到 free 并清空 fixed_halfdist。
            - transparency: 忽略网络输出，直接把当前 pos_in 写成下一状态。

        输出副作用:
            - batch.node_pos: float, (N, 3), 下一步坐标，单位 Å；fixed_pos=True 的行最终恢复为调用前 node_pos。
            - batch.task_setting: list[str]，部分 ``flex_to_free`` 分支会原地改为 ``free``。
            - batch.fixed_halfdist: LongTensor|BoolTensor，形状为 (H,)，部分分支会清空或保持该距离约束。
            - batch.fixed_pos: LongTensor|BoolTensor，形状为 (N,)，部分已知原子分支会读取该坐标条件掩码。
            - batch.node_type: LongTensor，形状为 (N,)，本函数不修改。
            - batch.halfedge_type: LongTensor，形状为 (H,)，本函数不修改。
        """
        if self.post_process is None:
            # ``setting``：str，自由度模式决定是否把直接坐标预测投影回结构化运动流形。
            setting = self._get_setting(batch)
            if setting in ['flexible', 'torsional']:
                # ``pred_pos``：FloatTensor，形状为 (N, 3)；保留刚性域内部几何并允许 tor_bonds_anno 扭转后的坐标，单位 Å。
                pred_pos = correct_pos_batch(batch, outputs)
                # pred_pos = correct_pos_by_fixed_dist_batch(batch, outputs)
            elif setting == 'rigid':
                # ``pred_pos``：FloatTensor，形状为 (N, 3)；仅按 domain_node_index 应用每域 Kabsch 刚体变换后的坐标，单位 Å。
                pred_pos = correct_pos_batch_no_tor(batch, outputs)
            else:
                # ``pred_pos``：FloatTensor，形状为 (N, 3)；free 模式直接采用网络坐标预测，单位 Å。
                pred_pos = outputs['pred_pos']
        elif isinstance(self.post_process, dict):  # for use
            # ``pred_pos``：FloatTensor，形状为 (N, 3)；独立副本；后续对已知原子位置的写入不会改 outputs.pred_pos。
            pred_pos = outputs['pred_pos'].clone()
            # ``name``：str，结构化后处理注册名；本分支仅接受 know_some。
            name = self.post_process.name
            assert name == 'know_some', 'Only know_some post_process is supported.'
            # ``num_mols``：int64 标量 B，由最大图号+1得到；默认假定图号连续且批次非空。
            num_mols = batch['node_type_batch'].max() + 1
            # ``num_nodes``：int64 标量 N_each，假定批次中所有图具有相同原子数，并用总 num_nodes 整除 B。
            num_nodes = batch['num_nodes'] // num_mols

            # ``step``：float 标量，当前规范采样进度。
            step = batch['step']
            if step > 0.2:  # directly fixed
                if 'orig_fixed_pos' not in batch:
                    self.pre_process = 'fix_some'  # fix the some atoms
                    # ``batch.orig_fixed_pos``：0/1, (N,)，首次启用已知原子硬固定前保存原 fixed_pos，供低 step 阶段恢复。
                    batch['orig_fixed_pos'] = batch['fixed_pos'].clone()
                    # ``i_batch``：int，当前图号；把模板局部原子索引平移到该图在批次中的全局原子区间。
                    for i_batch in range(num_mols):
                        # ``one_setting.atom``：int，当前已知原子在单图内的 0-based 索引。
                        # ``one_setting.coord``：list[float]|缺省，长度为 3 的已知世界坐标，单位 Å。
                        # ``one_setting.radius``：float|缺省，当前硬固定阶段不读取的软约束半径，单位 Å。
                        # ``one_setting``：Mapping，保存以上已知原子约束叶。
                        for one_setting in self.post_process.atom_space:
                            # ``atom_index``：int64 标量，把模板局部 atom 索引平移为第 i_batch 图的批次全局索引。
                            atom_index = one_setting['atom'] + num_nodes * i_batch
                            # ``batch.fixed_pos``：(N,) 单索引写入，把该已知原子的坐标设为 fixed。
                            batch['fixed_pos'][atom_index] = 1
                            # modify the coord
                            if 'coord' in one_setting:
                                # ``coord``：FloatTensor，形状为 (3,)；用户给定的绝对已知坐标，单位 Å。
                                coord = torch.tensor(one_setting['coord'], dtype=batch['node_pos'].dtype, device=batch['node_pos'].device)
                                # (3,) -> (3,)，减去该图口袋中心，转入模型使用的口袋局部坐标系。
                                coord -= batch['pocket_center'][i_batch]
                                # ``batch.node_pos``：(N, 3) 单行写入，把当前参考位置替换为用户已知局部坐标。
                                batch['node_pos'][atom_index] = coord
            else:
                # reset fixed_pos
                if 'orig_fixed_pos' in batch:
                    self.pre_process = None
                    # ``batch.fixed_pos``：0/1, (N,)，恢复启用 know_some 前保存的原 fixed_pos 掩码。
                    batch['fixed_pos'] = batch['orig_fixed_pos']
                # ``i_batch``：int，当前图号；把模板局部原子索引平移到该图在批次中的全局原子区间。
                for i_batch in range(num_mols):
                    # ``one_setting.atom``：int，当前软约束原子在单图内的 0-based 索引。
                    # ``one_setting.radius``：float，允许预测坐标偏离目标的球半径，单位 Å。
                    # ``one_setting.coord``：list[float]|缺省，长度为 3 的目标世界坐标；缺省时使用真值局部坐标。
                    # ``one_setting``：Mapping，保存以上软约束叶。
                    for one_setting in self.post_process.atom_space:
                        # ``atom_index``：int64 标量，第 i_batch 图中该模板原子的批次全局索引。
                        atom_index = one_setting['atom'] + num_nodes * i_batch
                        # ``radius``：float 标量，允许该原子预测偏离目标位置的球半径，单位 Å。
                        radius = one_setting['radius']
                        if 'coord' in one_setting: # check equal to gt_node_pos
                            # ``coord``：FloatTensor，形状为 (3,)；用户绝对坐标，单位 Å。
                            coord = torch.tensor(one_setting['coord'], dtype=batch['node_pos'].dtype, device=batch['node_pos'].device)
                            # (3,) -> (3,)，减 pocket_center 转为模型口袋局部坐标。
                            coord -= batch['pocket_center'][i_batch]
                        else:
                            # ``coord``：FloatTensor，形状为 (3,)；未显式给坐标时使用该原子的真值局部坐标，单位 Å。
                            coord = batch['gt_node_pos'][atom_index]
                        # ``dist``：float 标量，该原子预测位置到已知目标的欧氏距离，单位 Å。
                        dist = torch.norm(pred_pos[atom_index] - coord, dim=-1)
                        if dist > radius:  # move
                            # ``pred_pos``：(N, 3) 单行写入，超出允许球时把该原子直接放到目标坐标。
                            pred_pos[atom_index] = coord
        else:  # str
            if self.post_process == 'correct_pos':  # correct pos just like flex mode
                # ``corr_config``：dict/EasyDict，刚性/扭转坐标投影配置。
                corr_config = self.config.get('correct_pos', None)
                # ``interval_steps``：sequence[float] 长度 2，启用投影的规范 step 闭区间 [lower, upper]。
                interval_steps = corr_config['interval_steps']
                if batch['step'] >= interval_steps[0] and batch['step'] <= interval_steps[1]:
                    # ``pred_pos``：FloatTensor，形状为 (N, 3)；以 gt_node_pos 为几何参考投影后的坐标，单位 Å。
                    pred_pos = correct_pos_batch(batch, outputs, use_pos='gt')
                else:
                    # ``pred_pos``：FloatTensor，形状为 (N, 3)；区间外直接采用网络预测，单位 Å。
                    pred_pos = outputs['pred_pos']
            elif self.post_process == 'correct_dist':  # correct the interval pairwise distances using gradient descent
                # ``corr_config``：dict|None，fixed_halfdist 原子对距离修正的迭代参数。
                corr_config = self.config.get('correct_dist', None)
                if corr_config is not None and 'threshold_step' in corr_config:
                    # ``threshold_step``：float，只有当前 step 不大于该阈值时才执行距离修正。
                    threshold_step = corr_config['threshold_step']
                else:
                    # ``threshold_step``：int，缺少阈值时使用极大值，使常规 [0,1] step 全部进入修正分支。
                    threshold_step = 1000000000000  # boring
                if batch['step'] <= threshold_step:
                    # ``pred_pos``：FloatTensor，形状为 (N, 3)；按真值固定距离约束迭代修正后的坐标，单位 Å。
                    pred_pos = correct_pos_by_fixed_dist_batch(batch, outputs, use_pos='gt', config=corr_config)
                else:
                    # ``pred_pos``：FloatTensor，形状为 (N, 3)；阈值外直接采用网络预测，单位 Å。
                    pred_pos = outputs['pred_pos']
            elif self.post_process == 'correct_center':
                # ``corr_config``：dict/EasyDict，读取 ``radius``：允许预测中心偏离真值中心的阈值，单位 Å。
                corr_config = self.config.get('correct_center', None)
                # ``pred_pos``：FloatTensor，形状为 (N, 3)；待按整图中心修正的网络预测坐标，单位 Å。
                pred_pos = outputs['pred_pos']
                # ``batch_index``：LongTensor，形状为 (N,)；原子所属图号，scatter 归约轴实体为 B 个图。
                batch_index = batch['node_type_batch']
                # ``gt_center``：FloatTensor，形状为 (B, 3)；每图真值配体几何中心，单位 Å。
                gt_center = scatter_mean(batch['gt_node_pos'], index=batch_index, dim=0)
                # ``pred_center``：FloatTensor，形状为 (B, 3)；每图预测配体几何中心，单位 Å。
                pred_center = scatter_mean(pred_pos, index=batch_index, dim=0)
                # ``delta_pos``：FloatTensor，形状为 (B, 3)；从预测中心指向真值中心的平移向量，单位 Å。
                delta_pos = gt_center - pred_center
                # ``delta_dist``：FloatTensor，形状为 (B, 1)；中心偏差欧氏长度，保留末维以便与 (B, 3) 广播，单位 Å。
                delta_dist = torch.norm(delta_pos, p=2, dim=-1)[..., None]
                # ``translation``：FloatTensor，形状为 (B, 3)；偏差大于 radius 的图使用完整中心位移，否则使用零位移，单位 Å。
                translation = torch.where(
                    delta_dist > corr_config.radius,
                    delta_pos,
                    torch.zeros_like(delta_pos)
                )
                # ``pred_pos``：(N, 3)+(B, 3)[N] -> (N, 3)，把每图同一平移广播到所属全部原子。
                pred_pos = pred_pos + translation[batch_index]
            elif self.post_process == 'correct_closest':  # know the closest atom pos within a sphere
                # ``corr_config``：dict/EasyDict，最近原子整图平移修正配置。
                corr_config = self.config.get('correct_closest', None)
                # ``radius``：float，允许最近原子预测偏离已知位置的阈值半径，单位 Å。
                radius = corr_config['radius']
                # find the closest pos
                if 'node_closest' in batch:
                    # ``node_closest``：LongTensor，形状为 (B,)；缓存的每图最近配体原子批次全局索引。
                    node_closest = batch['node_closest']
                else:
                    # ``node_closest``：list[scalar Tensor]，循环中暂存 B 个最近原子全局索引。
                    node_closest = []
                    # ``i_mol``：int，当前图号，同时筛选真值配体坐标与口袋坐标的批归属向量。
                    for i_mol in range(batch['node_type_batch'].max().item()+1):
                        # ``dist_mat``：FloatTensor，形状为 (N_i, P_i)；第 i 图真值配体原子到口袋点的两两距离，单位 Å。
                        dist_mat = torch.cdist(batch['gt_node_pos'][batch['node_type_batch']==i_mol],
                                                  batch['pocket_pos'][batch['pocket_pos_batch']==i_mol])
                        # ``node_closest_this``：int64 标量，离任一口袋点最近的图内配体原子索引。
                        node_closest_this = torch.argmin(dist_mat.min(dim=1)[0], dim=0)
                        # ``node_closest_this``：int64 标量，加上前序图原子数，转为批次全局索引。
                        node_closest_this = node_closest_this + (batch['node_type_batch']<i_mol).sum()
                        node_closest.append(node_closest_this)
                    # ``node_closest``：LongTensor，形状为 (B,)；每图一个最近配体原子的批次全局索引。
                    node_closest = torch.stack(node_closest)
                    # ``batch.node_closest``：LongTensor，形状为 (B,)；缓存每图最近配体原子全局索引。
                    batch['node_closest'] = node_closest
                # correct pos
                # ``gt_closest_pos``：FloatTensor，形状为 (B, 3)；已知最近配体原子的真值坐标，单位 Å。
                gt_closest_pos = batch['gt_node_pos'][node_closest]
                # ``pred_closest_pos``：FloatTensor，形状为 (B, 3)；对应原子的网络预测坐标，单位 Å。
                pred_closest_pos = outputs['pred_pos'][node_closest]
                # ``delta_pos``：FloatTensor，形状为 (B, 3)；从预测最近点指向已知最近点的位移，单位 Å。
                delta_pos = gt_closest_pos - pred_closest_pos
                # ``delta_dist``：FloatTensor，形状为 (B, 1)；逐图最近点偏差长度，单位 Å。
                delta_dist = torch.norm(delta_pos, p=2, dim=-1)[..., None]
                # ``translation``：FloatTensor，形状为 (B, 3)；偏差大于 radius 的图采用完整位移，否则采用零位移，单位 Å。
                translation = torch.where(
                    delta_dist > radius,
                    delta_pos,
                    torch.zeros_like(pred_closest_pos)
                )
                # ``pred_pos``：FloatTensor，形状为 (N, 3)；独立副本，避免整图平移改写 outputs.pred_pos。
                pred_pos = outputs['pred_pos'].clone()
                # (N, 3)+(B, 3)[N] -> (N, 3)，按原子图号广播每图最近点位移。
                pred_pos += translation[batch['node_type_batch']]
            elif self.post_process == 'flex_to_free':  # change the mode from flex to free
                # ``corr_config``：dict/EasyDict，包含 flexible 切换到 free 的规范进度阈值。
                corr_config = self.config.get('flex_to_free', None)
                # ``change_step``：float，高于该 step 保持 flexible，低于或等于时切换到 free。
                change_step = corr_config['change_step']
                if batch['step'] > change_step:
                    # ``pred_pos``：FloatTensor，形状为 (N, 3)；高噪声阶段按 flexible 刚性域/扭转约束投影，单位 Å。
                    pred_pos = correct_pos_batch(batch, outputs)  # use_pos can be 'gt' or not since the rigidity is not changed for step > change_step
                else:
                    # mode change
                    # ``batch.task_setting``：list[str] 长度 B，把每图的自由度模式都从原值覆盖为 free。
                    batch['task_setting'] = ['free' for _ in batch['task_setting']]
                    # fixed indicators
                    # ``batch.fixed_halfdist``：0/1, (H,)，清空全部固定距离标记，使后续 free 步不再保持刚性域距离。
                    batch['fixed_halfdist'] = torch.zeros_like(batch['fixed_halfdist'])
                    # not correct pos
                    # ``pred_pos``：FloatTensor，形状为 (N, 3)；切换后的首个 free 状态直接采用网络预测，单位 Å。
                    pred_pos = outputs['pred_pos']
            elif self.post_process == 'transparency':  # out = in
                # ``pred_pos``：FloatTensor，形状为 (N, 3)；跳过模型坐标预测，把本步模型输入原样传到下一步，单位 Å。
                pred_pos = batch['pos_in'].clone()
            else:
                raise NotImplementedError('not implemented for post_process:', self.post_process)
            
        # ``fixed_pos``：BoolTensor，形状为 (N,)；最终硬约束掩码；优先级高于上述所有投影和修正分支。
        fixed_pos = (batch['fixed_pos'] == 1)
        # ``pred_pos``：(N, 3) 掩码写入，最终把所有 fixed 原子恢复为调用前当前状态坐标。
        pred_pos[fixed_pos] = batch['node_pos'][fixed_pos].clone()

        # ``batch.node_pos``：FloatTensor，形状为 (N, 3)；写回下一步 _fetch_data 读取的当前配体状态，单位 Å。
        batch['node_pos'] = pred_pos.clone()
        # node_type and halfedge_type are is unchanged
        
        # if 'confidence_pos' in outputs:
        #     batch['confidence_pos'] = outputs['confidence_pos']
        #     batch['confidence_node'] = outputs['confidence_node']
        #     batch['confidence_halfedge'] = outputs['confidence_halfedge']
        return batch

# XXX
@register_sample_noise('dock')
class DockSamplNoiser(ConfSampleNoiser):
    """
    复用构象坐标 noiser，并把任务名固定为 ``dock``。

    与 ``ConfSampleNoiser`` 的唯一代码差异是 ``task_name='dock'``；该任务名同时产生两个语义效果:
        - ``prior=from_train`` 时从 mixed 训练配置选择 docking 子先验。
        - ``add_noise`` 不执行 ``task == 'conf'`` 的逐图重居中，因此保留配体相对口袋的平移与旋转。

    其余输入字段、level 形状、结构化运动注释、fixed 约束和输出写回契约完全继承父类。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, task_name='dock')

@register_sample_noise('mixed')
class MixedSampleNoiser:
    """
    按 ``batch.task`` 把样本分派给相应任务噪声器。

    构造参数:
        - config.name: str, 固定为 ``mixed``。
        - config.individual: list[config], 每项是一个子噪声器配置；``task_cfg.name`` 同时作为注册名和分派键。
        - ``*args``/``**kwargs``: 原样传给每个 ``get_sample_noiser``，通常含类别数、mode、device 与训练参考配置。

    派生字段:
        - noiser_dict: dict[str, BaseSampleNoiser], 从任务名到已实例化子噪声器的映射。

    调用输入:
        - batch.task: str|list[str], 训练变换通常传 str；采样批次可能传同值列表。

    调用输出:
        - batch: 对应子噪声器原地新增 ``node_in``、``pos_in``、``halfedge_in`` 后的同一对象。
    """
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        # ``config.name``：str，固定为 ``mixed``，选择当前分派器。
        # ``config.individual[*].name``：str，子 noiser 注册名与 ``batch.task`` 分派键。
        # ``config.individual[*].prior``：Mapping|str，子任务先验配置。
        # ``config.individual[*].level``：Mapping，子任务信息等级配置。
        # ``config.individual[*].num_steps``：int|缺省，子任务 sample 迭代步数。
        # ``config.individual[*].reassign_in``：bool|缺省，子任务训练输入同构重排开关。
        # ``self.config``：EasyDict，保留上述 mixed 顶层和有序子任务叶。
        self.config = config
        # ``self.noiser_dict``：dict[str, noiser]，键来自每个 task_cfg.name；同名项会覆盖先前实例。
        self.noiser_dict = {}
        # ``task_cfg.name``：str，当前子 noiser 的注册名与分派键。
        # ``task_cfg.prior``：Mapping|str，当前子任务先验配置。
        # ``task_cfg.level``：Mapping，当前子任务信息等级配置。
        # ``task_cfg.num_steps``：int|缺省，当前子任务 sample 步数。
        # ``task_cfg.reassign_in``：bool|缺省，当前子任务训练同构重排开关。
        for task_cfg in config.individual:
            # ``self.noiser_dict``：BaseSampleNoiser 子类实例，以 task_cfg.name 为分派键；同名项覆盖旧实例。
            self.noiser_dict[task_cfg.name] = get_sample_noiser(task_cfg, *args, **kwargs)

    def __call__(self, batch, *args, **kwargs):
        """按图级任务名把同任务批次分派给对应子 noiser。

        输入参数:
            - batch: PyG Data|Batch，至少含图级 ``task`` 叶及被子 noiser 读取的分子状态、prompt 和运动注释叶。
            - batch.task: str|list[str]，单样本任务名或 PyG 收集后的逐图任务名列表。
            - ``*args``: 位置参数，原样传给选中的 ``BaseSampleNoiser.__call__``。
            - ``**kwargs``: 关键字参数，原样传给选中的 ``BaseSampleNoiser.__call__``。

        返回值:
            - batch: PyG Data|Batch，选中子 noiser 写入 ``node_in``、``pos_in`` 和 ``halfedge_in`` 后的同一容器。
        """
        # ``task``：str|list[str]，列表时假定批次内同任务并选择第一项；此处没有独立的一致性校验。
        task = batch['task']
        if isinstance(task, list):
            # ``task``：str，采样批次假定所有 task 同值并取首项；此处不验证一致性。
            task = task[0]
        return self.noiser_dict[task](batch, *args, **kwargs)


@register_sample_noise('dynamic')
class DynamicSettingSampleNoiser:
    def __init__(self, config, *args, **kwargs):
        self.config = config
        # process phase
        self.phases = config.phases
        self.total_steps = sum(self.phases.num_steps)
        self.cum_steps = np.cumsum(self.phases.num_steps)

        base_noise_cfg = config.base_noise
        base_noise_cfg.num_steps = self.total_steps
        self.base_noiser = get_sample_noiser(base_noise_cfg, *args, **kwargs)

    def __call__(self, batch, *args, **kwargs):
        step = kwargs['step']
        global_step = self.total_steps * (1 - step)
        index_stage = np.where(step > self.cum_steps)[0]
        settings_this_stage = self.phases.settings[index_stage]
        
        # renew setting step
        phase_step = self.phases.num_steps[index_stage]
        phase_interval = self.phases.step_intervals[index_stage]
        phase_bins = phase_interval / phase_step
        phase_local_step = global_step - max(self.cum_steps[index_stage-1], 0)
        step = phase_interval[0] - phase_bins * phase_local_step
        assert step > phase_interval[1], 'step out of interval'
        kwargs.update({'step': step})
        
        # supress settings
        batch = self._overwrite_settings(batch, settings_this_stage)
        return self.base_noiser(batch, *args, **kwargs)
        
    def _overwrite_settings(self, batch, new_settings):
        old_settings = batch.setting
        if isinstance(old_settings, list):
            assert len(old_settings) == len(set(old_settings)), 'settings are not unique for the batch'
            new_settings = [new_settings for _ in old_settings]
        else:
            raise NotImplementedError('not implement for the setting types')
        batch.update({'settings': new_settings})
        return batch
    
    def steps_loop(self, *args, **kwargs):
        return self.base_noiser.steps_loop(*args, **kwargs)

    def outputs2batch(self, batch, outputs):
        return self.base_noiser.outputs2batch(batch, outputs)


@register_sample_noise('denovo')
class DenovoSampleNoiser(BaseSampleNoiser):
    def __init__(self,
        config, num_node_types, num_edge_types,
        mode='sample', device='cpu', ref_config=None, task_name='denovo',
        **kwargs
    ):
        super().__init__(task_name, config, num_node_types, num_edge_types,
                mode, device, ref_config, **kwargs)
        
        # define prior
        prior_config = config.prior if config.prior != 'from_train' else self.ref_prior_config
        self.prior = MolPrior(prior_config, num_node_types, num_edge_types).to(device)

        # define info level
        self.level = MolInfoLevel(config.level, device=device, mode=mode)
        
        self.post_process = config.get('post_process', None)
        

    def sample_level(self, step, batch):
        level_dict = {}
        level_node, level_pos, level_halfedge = self.level.sample_for_mol(
            step,
            n_node=batch['node_type'].shape[0],
            n_pos=batch['node_type'].shape[0],
            n_edge=batch['halfedge_type'].shape[0],
        )
        level_dict.update({
            f'node': level_node,
            f'pos': level_pos,
            f'halfedge': level_halfedge,
        })
        
        
        if 'scaling_level_node' in batch:
            level_dict['node'] = level_dict['node'] ** batch['scaling_level_node']
        elif 'scaling_noise_node' in batch:
            level_dict['node'] = 1 - (1 - level_dict['node']) * batch['scaling_noise_node']
        if 'scaling_level_pos' in batch:
            level_dict['pos'] = level_dict['pos'] ** batch['scaling_level_pos']
        elif 'scaling_noise_pos' in batch:
            level_dict['pos'] = 1 - (1 - level_dict['pos']) * batch['scaling_noise_pos']
        if 'scaling_level_halfedge' in batch:
            level_dict['halfedge'] = level_dict['halfedge'] ** batch['scaling_level_halfedge']
        elif 'scaling_noise_halfedge' in batch:
            level_dict['halfedge'] = 1 - (1 - level_dict['halfedge']) * batch['scaling_noise_halfedge']
        
        return level_dict

    def add_noise(self, node_type, node_pos, halfedge_type, batch,
                  from_prior=False, level_dict=None):
        
        task = self._get_task(batch)
        # # recenter before add noise
        if (task == 'denovo'):
            batch_node = getattr(batch, 'node_type_batch',
                        torch.zeros(node_type.shape[0], dtype=torch.long, device=node_pos.device))  
            node_pos_center = scatter_mean(node_pos, batch_node, dim=0, dim_size=batch_node.max()+1)[batch_node]
            node_pos = node_pos - node_pos_center
            if self.mode == 'train':
                batch.update({'node_pos': node_pos.clone()})
        
        noised_data = self.prior.add_noise(
            node_type, node_pos, halfedge_type, 
            level_dict=level_dict, from_prior=from_prior)
        pos_in = noised_data[1]

        # # recenter after add noise
        if (task == 'denovo'):
            batch_node = getattr(batch, 'node_type_batch',
                        torch.zeros(node_type.shape[0], dtype=torch.long, device=node_pos.device))
            pos_in_center = scatter_mean(pos_in, batch_node, dim=0, dim_size=batch_node.max()+1)[batch_node]
            pos_in = pos_in - pos_in_center

        in_dict = {'node':noised_data[0], 'pos':pos_in, 'halfedge':noised_data[2]}
        return in_dict
    
    def outputs2batch(self, batch, outputs):
        
        if self.post_process is None:
            batch['node_type'] = outputs['pred_node'].argmax(-1)
            batch['node_pos'] = outputs['pred_pos']
            batch['halfedge_type'] = outputs['pred_halfedge'].argmax(-1)
        elif self.post_process == 'redock':
            fixed_node = batch['fixed_node']
            fixed_node_bool = (fixed_node == 1)
            fixed_halfedge = batch['fixed_halfedge']
            fixed_halfedge_bool = (fixed_halfedge == 1)
            batch['node_type'][~fixed_node_bool] = outputs['pred_node'].argmax(-1)[~fixed_node_bool]
            batch['halfedge_type'][~fixed_halfedge_bool] = outputs['pred_halfedge'].argmax(-1)[~fixed_halfedge_bool]
            batch['node_pos'] = outputs['pred_pos']

            redock_config = self.config.redock
            start_step = redock_config.start_step
            step = batch['step']

            if step <= start_step:  # now in dock mode
                fixed_node = torch.ones_like(fixed_node)
                fixed_halfedge = torch.ones_like(fixed_halfedge)
                batch['fixed_node'] = fixed_node
                batch['fixed_halfedge'] = fixed_halfedge
        elif self.post_process == 'corr_shape':
            step = batch['step']
            cfg_shape = self.config['corr_shape']
            corr_th_step = cfg_shape.get('corr_th_shape', 0.1)
            
            if step > corr_th_step:
                letter = cfg_shape['letter']
                length = cfg_shape.get('length', 12)
                height = cfg_shape.get('height', 2)
                corr_th_dist = cfg_shape.get('corr_th_dist', 2)
                
                delta_all = []
                pred_pos = outputs['pred_pos']
                for i_batch in range(batch['node_type_batch'].max() + 1):
                    this_batch = (batch['node_type_batch'] == i_batch)
                    
                    pred_pos_batch = pred_pos[this_batch].detach().cpu().numpy()
                    n_points = pred_pos_batch.shape[0]
                    shape_points = get_points_from_letter(letter, n_points, length=length, height=height)
                    
                    # calc dist mat and match
                    dist_mat = np.linalg.norm(pred_pos_batch[:, None] - shape_points[None], axis=-1)
                    row_ind, col_ind = linear_sum_assignment(dist_mat)
                    
                    # get delta
                    shape_points = shape_points[col_ind]
                    delta_vec = shape_points - pred_pos_batch
                    delta_dist = np.linalg.norm(delta_vec, axis=-1, keepdims=True)
                    delta_vec = np.where(delta_dist > corr_th_dist, delta_vec * (step - corr_th_step)/(1-corr_th_step), 0)
                    delta_all.append(delta_vec)
                delta_all = np.concatenate(delta_all, axis=0)
                delta_all = torch.tensor(delta_all, dtype=pred_pos.dtype, device=pred_pos.device)
            else:
                delta_all = 0
            
            batch['node_pos'] = outputs['pred_pos'] + delta_all
            batch['node_type'] = outputs['pred_node'].argmax(-1)
            batch['halfedge_type'] = outputs['pred_halfedge'].argmax(-1)
            
        else:
            raise NotImplementedError('not implement for the post_process types')
        
        if self.config.get('scaling_level', False):
            cfd_node = torch.sigmoid(outputs['confidence_node'][:, 0])
            cfd_pos = torch.sigmoid(outputs['confidence_pos'][:, 0])
            cfd_halfedge = torch.sigmoid(outputs['confidence_halfedge'][:, 0])

            batch_node = batch['node_type_batch']
            batch_halfedge = batch['halfedge_type_batch']
            n_batch = batch_node.max() + 1

            scaling_level_node = []
            scaling_level_pos = []
            scaling_level_halfedge = []
            for i_batch in range(n_batch):
                cfd_node_this = cfd_node[batch_node==i_batch]
                cfd_pos_this = cfd_pos[batch_node==i_batch]
                cfd_halfedge_this = cfd_halfedge[batch_halfedge==i_batch]
                
                s = self.config.scaling_level.s
                # scaling_node = 1 - ((cfd_node_this - cfd_node_this.min()) / (cfd_node_this.max() - cfd_node_this.min() + 1e-8)) * 2
                scaling_node = torch.median(cfd_node_this) - cfd_node_this
                scaling_node = scaling_node / scaling_node.max()
                scaling_node = scaling_node.clamp(min=0)
                scaling_node = s ** scaling_node  # 2 -> 0.5: cfd low -> high
                scaling_level_node.append(scaling_node)
                
                # scaling_pos = 1 - ((cfd_pos_this - cfd_pos_this.min()) / (cfd_pos_this.max() - cfd_pos_this.min() + 1e-8)) * 2
                scaling_pos = torch.median(cfd_pos_this) - cfd_pos_this
                scaling_pos = scaling_pos / scaling_pos.max()
                scaling_pos = scaling_pos.clamp(min=0)
                scaling_pos = s ** scaling_pos
                scaling_level_pos.append(scaling_pos)
                
                # scaling_halfedge = 1 - ((cfd_halfedge_this - cfd_halfedge_this.min()) / (cfd_halfedge_this.max() - cfd_halfedge_this.min() + 1e-8)) * 2
                scaling_halfedge = torch.median(cfd_halfedge_this) - cfd_halfedge_this
                scaling_halfedge = scaling_halfedge / scaling_halfedge.max()
                scaling_halfedge = scaling_halfedge.clamp(min=0)
                scaling_halfedge = s ** scaling_halfedge
                scaling_level_halfedge.append(scaling_halfedge)
            
            scaling_level_node = torch.cat(scaling_level_node)
            scaling_level_pos = torch.cat(scaling_level_pos)
            scaling_level_halfedge = torch.cat(scaling_level_halfedge)
            batch.update({
                'scaling_level_node': scaling_level_node,
                'scaling_level_pos': scaling_level_pos,
                'scaling_level_halfedge': scaling_level_halfedge,
            })
        elif self.config.get('shift_level', False):
            end_step = self.config.shift_level.end_step
            step = batch['step']

            scaling = (step - end_step) / (self.init_step - end_step) + 0.01
            scaling = np.clip(scaling, 0.01, 1)
            batch.update({
                'scaling_noise_node': scaling,
                'scaling_noise_pos': scaling,
                'scaling_noise_halfedge': scaling,
            })
        elif self.config.get('shift_typelevel', False):
            end_step = self.config.shift_typelevel.end_step
            step = batch['step']

            scaling = (step - end_step) / (self.init_step - end_step) + 0.01
            scaling = np.clip(scaling, 0.01, 1)
            batch.update({
                'scaling_noise_node': scaling,
                'scaling_noise_halfedge': scaling,
            })
        
        
        return batch
    
    def additional_process(self, batch, in_dict):
        if self.post_process == 'redock':
            fixed_node = batch['fixed_node'].bool()
            fixed_halfedge = batch['fixed_halfedge'].bool()
            in_dict['node'][fixed_node] = batch['node_type'][fixed_node].clone()
            in_dict['halfedge'][fixed_halfedge] = batch['halfedge_type'][fixed_halfedge].clone()
        return in_dict


@register_sample_noise('sbdd')
class SBDDSamplNoiser(DenovoSampleNoiser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, task_name='sbdd')



@register_sample_noise('fbdd')
@register_sample_noise('maskfill')
class MaskfillSampleNoiser(BaseSampleNoiser):
    def __init__(self,
        config, num_node_types, num_edge_types,
        mode='sample', device='cpu', ref_config=None, task_name='maskfill',
        **kwargs
    ):
        super().__init__(task_name, config, num_node_types, num_edge_types,
            mode, device, ref_config, **kwargs)
        
        # define prior
        prior_part1 = config.prior.part1 if config.prior.part1 != 'from_train' else self.ref_prior_config.part1
        self.prior_p1 = MolPrior(prior_part1, num_node_types, num_edge_types).to(device)
        prior_part2 = config.prior.part2 if config.prior.part2 != 'from_train' else self.ref_prior_config.part2
        self.prior_p2 = MolPrior(prior_part2, num_node_types, num_edge_types).to(device)

        # define info level
        self.level_p1 = MolInfoLevel(config.level.part1, device=device, mode=mode)
        self.level_p2 = MolInfoLevel(config.level.part2, device=device, mode=mode)
        

    def sample_level(self, step, batch):
        # # level for part 1, part 2 and part1-part2
        level_dict = {}
        
        # for part1
        setting = self._get_setting(batch)
        part1_pert = setting['part1_pert']
        leveller = self.level_p1
        if part1_pert == 'fixed':
            pass
        elif part1_pert == 'free':
            level_pos_p1 = leveller.sample_for_mol(step, n_pos=batch['node_p1'].shape[0])
            level_dict.update({
                'pos_p1': level_pos_p1,
            })
        elif part1_pert == 'small':
            n_node = n_pos = batch['node_p1'].shape[0]
            n_halfedge = batch['halfedge_p1'].shape[0]
            level_node_p1, level_pos_p1, level_halfedge_p1 = leveller.sample_for_mol(step,
                n_node=n_node, n_pos=n_pos, n_edge=n_halfedge,)
            level_dict.update({
                'node_p1': level_node_p1, 'pos_p1': level_pos_p1, 'halfedge_p1': level_halfedge_p1
            })
            task = self._get_task(batch)
            if task == 'ar':  # noise_level p1 is scaled by ratio of p1. NOTE: seems not used
                batch_node = batch['node_type_batch']
                n_nodes_batch = batch_node.bincount()
                node_p1 = batch['node_p1']
                batch_p1 = batch_node[node_p1]
                if len(batch_p1) != 0:
                    n_p1_batch = batch_p1.bincount()
                else:
                    n_p1_batch = torch.zeros_like(n_nodes_batch)
                ratio_p1_batch = n_p1_batch.float() / n_nodes_batch.float()
                ratio_p1_node = ratio_p1_batch[batch_p1]
                ratio_p1_halfedge = ratio_p1_batch[batch['halfedge_type_batch'][batch['halfedge_p1']]]
                level_dict.update({
                    'node_p1': (1-level_node_p1) * ratio_p1_node + level_node_p1,
                    'pos_p1': (1-level_pos_p1) * ratio_p1_node + level_pos_p1,
                    'halfedge_p1': (1-level_halfedge_p1) * ratio_p1_halfedge + level_halfedge_p1,
                })
                raise NotImplementedError('not implemented for ar noise_level. seems not used')

        elif part1_pert == 'rigid':
            n_trans = n_rot = torch.sum(batch['n_domain']).item()
            L_trans_p1, L_rot_p1 = leveller.sample_for_mol(step, n_trans=n_trans, n_rot=n_rot)
            level_dict.update({
                'trans_p1': L_trans_p1, 'rot_p1': L_rot_p1
            })
        elif part1_pert == 'flexible':
            n_trans = n_rot = torch.sum(batch['n_domain']).item()
            n_tor = batch['tor_bonds_anno'].shape[0]
            L_trans_p1, L_rot_p1, L_tor_p1 = leveller.sample_for_mol(step,
                n_trans=n_trans, n_rot=n_rot, n_tor=n_tor)
            level_dict.update({
                'trans_p1': L_trans_p1, 'rot_p1': L_rot_p1, 'tor_p1': L_tor_p1
            })
        
        # for part2 and edge of p1p2
        leveller = self.level_p2
        n_node = n_pos = batch['node_p2'].shape[0]
        n_halfedge = batch['halfedge_p2'].shape[0]
        n_halfedge_p1p2 = batch['halfedge_p1p2'].shape[0]
        L_node_p2, L_pos_p2, L_halfedge_p2_and_p1p2 = leveller.sample_for_mol(step,
            n_node=n_node, n_pos=n_pos, n_edge=(n_halfedge + n_halfedge_p1p2))
        level_dict.update({
            'node_p2': L_node_p2,
            'pos_p2': L_pos_p2,
            'halfedge_p2': L_halfedge_p2_and_p1p2[:n_halfedge],
        })
        
        # p1p2 fixed
        level_p1p2 = L_halfedge_p2_and_p1p2[n_halfedge:]
        fixed_halfedge = batch['fixed_halfedge']
        halfedge_p1p2 = batch['halfedge_p1p2']
        fixed_p1p2 = (fixed_halfedge[halfedge_p1p2] == 1)
        level_p1p2[fixed_p1p2] = 1
        level_dict.update({
            'halfedge_p1p2': level_p1p2,
        })
        
        return level_dict
    
    def has_pocket(self, batch):
        return batch['pocket_pos'].shape[0] > 0

    def add_noise(self, node_type, node_pos, halfedge_type, batch,
                   from_prior=False, level_dict=None):
        
        task = self._get_task(batch)
        setting = self._get_setting(batch)
        part1_pert = setting['part1_pert']

        # centering before add_noise
        # if task == 'linking':
        if (not self.has_pocket(batch)) and (task != 'linking'):
            batch_node = getattr(batch, 'node_type_batch',
                        torch.zeros(node_type.shape[0], dtype=torch.long, device=node_pos.device))
            node_pos_center = scatter_mean(node_pos, batch_node, dim=0, dim_size=batch_node.max()+1)[batch_node]
            node_pos = node_pos - node_pos_center
            batch.update({'node_pos': node_pos.clone()})  #NOTE: a little leakage during sampling since p2 gaussian center is gt p1 center

        # part 1 noise
        if task in ['maskfill', 'linking', 'sbdd', 'denovo']:  # sbdd, denovo in ar mode
            # from_prior_part1 = False
            from_prior_part1 = self.prior_p1.config.get('from_prior', False) and from_prior
        elif task in ['fbdd', 'growing']:
            # from_prior_part1 = from_prior
            from_prior_part1 = self.prior_p1.config.get('from_prior', True) and from_prior
        else:
            raise ValueError(f'Unknown task: {task} to set from_prior_part1')
        from_prior_part1 = from_prior_part1 and self.prior_p1.config.get('from_prior', True)
        prior = self.prior_p1
        level_dict_p1 = {k[:-3]:v for k, v in level_dict.items() if k.endswith('_p1')}
        
        if part1_pert == 'fixed':
            pass
        elif part1_pert == 'small':
            node_p1 = batch['node_p1']
            halfedge_p1 = batch['halfedge_p1']
            node_type[node_p1], node_pos[node_p1], halfedge_type[halfedge_p1] = prior.add_noise(
                node_type[node_p1], node_pos[node_p1], halfedge_type[halfedge_p1],
                level_dict_p1, from_prior=from_prior_part1,  # not from prior for part 1 because part 1 is not totally masked
            )
        elif part1_pert == 'free':
            node_p1 = batch['node_p1']
            node_pos[node_p1] = prior.add_noise(
                None, node_pos[node_p1], None,
                level_dict=level_dict_p1, from_prior=from_prior_part1, pos_only=True
            )
        else:  # 'rigid', 'flexible'
            node_p1 = batch['node_p1']
            additional_dict = {
                'pos_only': True,
                'node_type': None, 'halfedge_type': None, # placeholder
                'domain_node_index': batch['domain_node_index']
            }
            if part1_pert == 'flexible':
                additional_dict.update({
                    'tor_bonds_anno': batch['tor_bonds_anno'],
                    'twisted_nodes_anno': batch['twisted_nodes_anno'],
                })
            assert all([n in node_p1 for n in batch['domain_node_index'][1]]), 'node in domain not in node_p1'
            node_pos = prior.add_noise(
                node_pos=node_pos, level_dict=level_dict_p1,
                from_prior=from_prior_part1, **additional_dict,
            )
        
        # part 2 noise
        prior = self.prior_p2
        level_dict_p2 = {k[:-3]:v for k, v in level_dict.items() if k.endswith('_p2')}
        node_p2 = batch['node_p2']
        halfedge_p2 = batch['halfedge_p2']
        node_type[node_p2], node_pos[node_p2], halfedge_type[halfedge_p2] = prior.add_noise(
            node_type[node_p2], node_pos[node_p2], halfedge_type[halfedge_p2],
            level_dict_p2, from_prior
        )
        halfedge_p1p2 = batch['halfedge_p1p2']
        halfedge_type[halfedge_p1p2] = prior.halfedge.add_noise(
            halfedge_type[halfedge_p1p2], level_dict['halfedge_p1p2'], from_prior
        )
        
        
        # centering after add_noise
        # if task == 'linking':
        # if not self.has_pocket(batch):
        if (not self.has_pocket(batch)) and (task != 'linking'):
            if part1_pert != 'fixed':
            # only centering in non-fixed case where global coord system doesn't exist
                pos_in_center = scatter_mean(node_pos, batch_node, dim=0, dim_size=batch_node.max()+1)[batch_node]
                node_pos = node_pos - pos_in_center
        
        in_dict = {'node':node_type, 'pos':node_pos, 'halfedge': halfedge_type}
        return in_dict

    def outputs2batch(self, batch, outputs):
        
        # task = self._get_task(batch)
        # if task == 'ar':
        #     return self.outputs2batch_ar(batch, outputs)
        
        setting = self._get_setting(batch)
        part1_pert = setting['part1_pert']

        # node_p1, node_p2 = batch['node_p1'], batch['node_p2']
        fixed_node = (batch['fixed_node'] == 1)
        fixed_pos = (batch['fixed_pos'] == 1)
        fixed_halfedge = (batch['fixed_halfedge'] == 1)

        batch['node_type'][~fixed_node] = outputs['pred_node'][~fixed_node].argmax(-1).clone()
        batch['node_pos'][~fixed_pos] = outputs['pred_pos'][~fixed_pos].clone()
        batch['halfedge_type'][~fixed_halfedge] = outputs['pred_halfedge'][~fixed_halfedge].argmax(-1).clone()
        
        if part1_pert =='rigid':
            pred_pos = correct_pos_batch_no_tor(batch, outputs)
            batch['node_pos'] = pred_pos
        elif part1_pert == 'flexible':
            pred_pos = correct_pos_batch(batch, outputs)
            batch['node_pos'] = pred_pos
        
        return batch

    def outputs2batch_ar2(self, batch, outputs):
        # if len(batch['pocket_pos']) > 0:
        #     raise NotImplementedError('pocket_pos not supported for ar yet. note the initial atom')
        device = batch['node_type'].device

        fixed_node = (batch['fixed_node'] == 1)
        fixed_pos = (batch['fixed_pos'] == 1)
        fixed_halfedge = (batch['fixed_halfedge'] == 1)
        batch['node_type'][~fixed_node] = outputs['pred_node'][~fixed_node].argmax(-1).clone()
        batch['node_pos'][~fixed_pos] = outputs['pred_pos'][~fixed_pos].clone()
        batch['halfedge_type'][~fixed_halfedge] = outputs['pred_halfedge'][~fixed_halfedge].argmax(-1).clone()

        
        # follow_batch = [k[:-6] for k in batch.keys if k.endswith('_batch')]
        follow_batch = [k[:-6] for k in batch.keys() if k.endswith('_batch')]
        mol_data_list = batch.to_data_list()
        for data in mol_data_list:
            if data['node_type'].shape[0] < data['gt_node_type'].shape[0]:
                n_frag = 6
                # new nodes
                data.update({
                    'node_type': torch.cat([data['node_type'], torch.zeros(n_frag, dtype=torch.long, device=device)]),
                    'node_pos': torch.cat([data['node_pos'], torch.zeros(n_frag, 3, dtype=torch.float, device=device)]),
                    'is_peptide': torch.cat([data['is_peptide'],
                                data['is_peptide'][0] * torch.zeros(n_frag, dtype=torch.long, device=device)]),
                })
                # new edges
                halfedge_index = data['halfedge_index']
                halfedge_type = data['halfedge_type']
                n_nodes = data['node_type'].shape[0]
                edge_type_mat = torch.zeros(n_nodes, n_nodes, dtype=torch.long, device=device)
                for i_edge in range(halfedge_type.shape[0]):
                    edge_type_mat[halfedge_index[0, i_edge], halfedge_index[1, i_edge]] = halfedge_type[i_edge]
                # edge_type_mat[:, -1] = -1
                # edge_type_mat[-1, :] = -1
                halfedge_index = torch.triu_indices(n_nodes, n_nodes, offset=1, device=device)
                halfedge_type = edge_type_mat[halfedge_index[0], halfedge_index[1]]
                data.update({
                    'halfedge_index': halfedge_index,
                    'halfedge_type': halfedge_type
                })
                # node_p1, node_p2
                node_p1 = torch.arange(n_nodes-n_frag, dtype=torch.long, device=device)
                # node_p2 = torch.tensor([n_nodes-n_frag], dtype=torch.long, device=device)
                node_p2 = torch.arange(n_nodes-n_frag, n_nodes, dtype=torch.long, device=device)
                # halfedge_p1, halfedge_p2, halfedge_p1p2
                is_left_p1 = (halfedge_index[0] < n_nodes-n_frag)
                is_right_p1 = (halfedge_index[1] < n_nodes-n_frag)
                is_halfedge_p1 = is_left_p1 & is_right_p1
                is_halfedge_p2 = (~is_left_p1) & (~is_right_p1)
                is_halfedge_p1p2 = (~is_halfedge_p1) & (~is_halfedge_p2)
                halfedge_p1 = torch.nonzero(is_halfedge_p1).squeeze(-1)
                halfedge_p2 = torch.nonzero(is_halfedge_p2).squeeze(-1)
                halfedge_p1p2 = torch.nonzero(is_halfedge_p1p2).squeeze(-1)
                # fixed
                fixed_node = torch.zeros(n_nodes, dtype=torch.long, device=device)
                fixed_pos = torch.zeros(n_nodes, dtype=torch.long, device=device)
                fixed_halfedge = torch.zeros(halfedge_type.shape[0], dtype=torch.long, device=device)
                fixed_halfdist = torch.zeros(halfedge_type.shape[0], dtype=torch.long, device=device)
            else: # finished
                n_nodes = data['node_type'].shape[0]
                halfedge_index = data['halfedge_index']
                node_p1 = torch.arange(n_nodes, dtype=torch.long, device=device)
                node_p2 = torch.tensor([], dtype=torch.long, device=device)
                halfedge_p1 = torch.arange(halfedge_index.shape[1], dtype=torch.long, device=device)
                halfedge_p2 = torch.tensor([], dtype=torch.long, device=device)
                halfedge_p1p2 = torch.tensor([], dtype=torch.long, device=device)
                # fixed
                # fixed_node = torch.ones(n_nodes, dtype=torch.long, device=device)
                # fixed_pos = torch.ones(n_nodes, dtype=torch.long, device=device)
                # fixed_halfedge = torch.ones(halfedge_index.shape[1], dtype=torch.long, device=device)
                # fixed_halfdist = torch.ones(halfedge_index.shape[1], dtype=torch.long, device=device)
                fixed_node = torch.zeros(n_nodes, dtype=torch.long, device=device)
                fixed_pos = torch.zeros(n_nodes, dtype=torch.long, device=device)
                fixed_halfedge = torch.zeros(halfedge_index.shape[1], dtype=torch.long, device=device)
                fixed_halfdist = torch.zeros(halfedge_index.shape[1], dtype=torch.long, device=device)
            data.update({
                'node_p1': node_p1,
                'node_p2': node_p2,
                'halfedge_p1': halfedge_p1,
                'halfedge_p2': halfedge_p2,
                'halfedge_p1p2': halfedge_p1p2,
                'fixed_node': fixed_node,
                'fixed_pos': fixed_pos,
                'fixed_halfedge': fixed_halfedge,
                'fixed_halfdist': fixed_halfdist,
            })
        batch = Batch.from_data_list(mol_data_list, follow_batch=follow_batch)

        return batch

    def additional_process(self, batch, in_dict):
        
        # reset those in_dict with fixed==1
        fixed_node = (batch['fixed_node'] == 1)
        in_dict['node'][fixed_node] = batch['node_type'][fixed_node].clone()
        fixed_pos = (batch['fixed_pos'] == 1)
        in_dict['pos'][fixed_pos] = batch['node_pos'][fixed_pos].clone()
        fixed_halfedge = (batch['fixed_halfedge'] == 1)
        in_dict['halfedge'][fixed_halfedge] = batch['halfedge_type'][fixed_halfedge].clone()
        return in_dict

    def outputs2batch_ar(self, batch, outputs, step_ar=None, cfd_traj=None):
        
        ar_config = self.config.ar_config
        ar_strategy = getattr(ar_config, 'strategy', 'default')
        
        if ar_strategy == 'default':
        
            batch_node = batch['node_type_batch']
            node_p1 = batch['node_p1']
            node_p2 = batch['node_p2']
            is_node_p1 = torch.zeros_like(batch_node, dtype=torch.bool)
            is_node_p1[node_p1] = True

            # get the node to be added to p1 from p2 for each data
            pred_node = outputs['pred_node']
            cfd_node = outputs['confidence_node']
            cfd_pos = outputs['confidence_pos']
            n_batch = batch_node.max() + 1
            n_sizes = batch_node.bincount()
            added_node = []
            for i_batch in range(n_batch):
                is_curr_p2 = (batch_node == i_batch) & (~is_node_p1)
                if is_curr_p2.sum() == 0:
                    continue
                pred_node_curr_p2 = pred_node[is_curr_p2]
                cfd_node_curr_p2 = cfd_node[is_curr_p2]
                cfd_pos_curr_p2 = cfd_pos[is_curr_p2]
                
                # # selecte generated nodes of this ar step
                ref_prob_type = ar_config.ref_prob_type
                if ref_prob_type == 'pred_node':
                    ref_prob = F.softmax(pred_node_curr_p2, dim=-1)
                    ref_prob = ref_prob.max(-1)[0]  # prob of the selected node_type
                elif ref_prob_type == 'cfd_node':
                    ref_prob = torch.sigmoid(cfd_node_curr_p2)[:, 0]
                elif ref_prob_type == 'cfd_pos':
                    ref_prob = torch.sigmoid(cfd_pos_curr_p2)[:, 0]
                else:
                    raise NotImplementedError('ref_prob_type not implemented:', ref_prob_type)
                ref_prob = (ref_prob-ref_prob.min()) / (ref_prob.max() - ref_prob.min()+1e-4)+1e-3
                
                size_select = ar_config.size_select
                select_strategy = ar_config.select_strategy
                if size_select >= 1:  # select fixed number
                    size_select = int(size_select)
                elif size_select < 1 and size_select > 0:  # select ratio
                    size_select = int(np.round(size_select * is_curr_p2.sum().item()))
                    size_select = max(size_select, 1)
                if select_strategy == 'random':
                    sel_node_curr = torch.unique(torch.multinomial(ref_prob, num_samples=size_select, replacement=True))
                elif select_strategy == 'top':
                    sel_node_curr = torch.argsort(ref_prob, descending=True)[:size_select]
                
                index_sel_node_curr = torch.nonzero(is_curr_p2)[sel_node_curr]
                added_node.extend(index_sel_node_curr)
            
            # change node_p1 and node_p2
            is_node_p1[torch.cat(added_node)] = True
            node_p1 = torch.nonzero(is_node_p1).squeeze(-1)
            node_p2 = torch.nonzero(~is_node_p1).squeeze(-1)

            # change halfedge_p1, halfedge_p2, halfedge_p1p2
            halfedge_index = batch['halfedge_index']
            left_in_p1 = is_node_p1[halfedge_index[0]]
            right_in_p1 = is_node_p1[halfedge_index[1]]
            is_halfedge_p1 = left_in_p1 & right_in_p1
            is_halfedge_p2 = (~left_in_p1) & (~right_in_p1)
            is_halfedge_p1p2 = (~is_halfedge_p1) & (~is_halfedge_p2)
            halfedge_p1 = torch.nonzero(is_halfedge_p1).squeeze(-1)
            halfedge_p2 = torch.nonzero(is_halfedge_p2).squeeze(-1)
            halfedge_p1p2 = torch.nonzero(is_halfedge_p1p2).squeeze(-1)
            
            part1_pert = self._get_setting(batch)['part1_pert']
            fixed_node = batch['fixed_node']
            fixed_pos = batch['fixed_pos']
            fixed_halfedge = batch['fixed_halfedge']
            fixed_halfdist = batch['fixed_halfdist']
            if part1_pert == 'small':
                pass
            elif part1_pert == 'fixed':
                fixed_node[node_p1] = 1
                fixed_pos[node_p1] = 1
                fixed_halfedge[halfedge_p1] = 1
                fixed_halfdist[halfedge_p1] = 1
                
            
            # # fixed the mol if no atoms in p2 (finished)
            batch_halfedge = batch['halfedge_type_batch']
            is_node_p2 = ~is_node_p1
            batch_mol_p2 = torch.unique(batch_node[is_node_p2])
            batch_mol_noinp2 = torch.tensor([i for i in range(batch_node.max()+1) if i not in batch_mol_p2],
                                                dtype=torch.long, device=batch_node.device)
            is_node_noinp2 = (batch_node[:, None]==batch_mol_noinp2[None]).any(-1)
            is_halfedge_noinp2 = (batch_halfedge[:, None]==batch_mol_noinp2[None]).any(-1)
            fixed_node[is_node_noinp2] = 1
            fixed_pos[is_node_noinp2] = 1
            fixed_halfedge[is_halfedge_noinp2] = 1
            # is_finished = torch.zeros_like(batch_node, dtype=torch.bool)
            # is_finished[is_node_noinp2] = True
            
            # reset 
            temperature = len(batch_node) / len(node_p1)
            batch.update({
                'node_p1': node_p1,
                'node_p2': node_p2,
                'halfedge_p1': halfedge_p1,
                'halfedge_p2': halfedge_p2,
                'halfedge_p1p2': halfedge_p1p2,

                # 'node_type': torch.multinomial(F.softmax(outputs['pred_node']/temperature, -1), 1)[:,0],
                # 'node_type': outputs['pred_node'].argmax(-1),
                # 'node_pos': outputs['pred_pos'],
                # 'halfedge_type': outputs['pred_halfedge'].argmax(-1),
                
                'fixed_node': fixed_node,
                'fixed_pos': fixed_pos,
                'fixed_halfedge': fixed_halfedge,
                'fixed_halfdist': fixed_halfdist,
            })
        elif ar_strategy == 'refine_partial':
            batch_node = batch['node_type_batch']
            # mol_size = batch_node.bincount()[batch_node]

            threshold_node = ar_config.threshold_node
            threshold_pos = ar_config.threshold_pos
            threshold_bond = ar_config.threshold_bond
            max_ar_step = ar_config.max_ar_step
            max_p2_ratio = ar_config.get('max_p2_ratio', 1)
            change_init_step = ar_config.get('change_init_step', None)
            if change_init_step is not None:
                self.init_step = change_init_step
            
            cfd_node = torch.sigmoid(outputs['confidence_node'][:, 0])
            cfd_pos = torch.sigmoid(outputs['confidence_pos'][:, 0])
            cfd_halfedge = torch.sigmoid(outputs['confidence_halfedge'][:, 0])
    
            # edge to bond, get cfd_node_with_bond
            halfedge_index = batch['halfedge_index']
            pred_halfedge = outputs['pred_halfedge'].argmax(-1)
            is_halfbond = (pred_halfedge > 0)
            halfbond_index = halfedge_index[:, is_halfbond]
            cfd_halfbond = cfd_halfedge[is_halfbond]
            bond_index = torch.cat([halfbond_index, halfbond_index.flip(0)], dim=-1)
            cfd_bond = torch.cat([cfd_halfbond, cfd_halfbond], dim=0)
            cfd_node_with_bond = scatter_mean(cfd_bond, bond_index[0], dim=0, dim_size=batch['node_type'].shape[0])
            
            # select nodes
            sel_node_curr_p2 = []
            possibility_p2_list = [
                (threshold_node - cfd_node).clamp(min=0),
                (threshold_pos - cfd_pos).clamp(min=0),
                (threshold_bond - cfd_node_with_bond).clamp(min=0),
            ]
            for possibility_p2 in possibility_p2_list:
                size_select = int(np.round((possibility_p2>0).sum().item() *
                                (1 - step_ar / max_ar_step) *
                                max_p2_ratio))
                if size_select > 0:
                    sel_this = torch.unique(torch.multinomial(
                        possibility_p2+1e-7, num_samples=size_select, replacement=True))
                else:
                    sel_this = []
                sel_node_curr_p2.extend(sel_this)
            sel_node_curr_p2 = torch.unique(torch.tensor(sel_node_curr_p2))
            sel_node_curr_p2 = sel_node_curr_p2 if len(sel_node_curr_p2) > 0 else []
            
            is_node_p2 = torch.zeros_like(batch_node, dtype=torch.bool)
            is_node_p2[sel_node_curr_p2] = True
            is_node_p1 = ~is_node_p2
            # print('is_node_p2', is_node_p2.sum().item())
            
            # change node_p1 and node_p2
            node_p1 = torch.nonzero(is_node_p1).squeeze(-1)
            node_p2 = torch.nonzero(is_node_p2).squeeze(-1)
            
            # change halfedge_p1, halfedge_p2, halfedge_p1p2
            left_in_p1 = is_node_p1[halfedge_index[0]]
            right_in_p1 = is_node_p1[halfedge_index[1]]
            is_halfedge_p1 = left_in_p1 & right_in_p1
            is_halfedge_p2 = (~left_in_p1) & (~right_in_p1)
            is_halfedge_p1p2 = (~is_halfedge_p1) & (~is_halfedge_p2)
            halfedge_p1 = torch.nonzero(is_halfedge_p1).squeeze(-1)
            halfedge_p2 = torch.nonzero(is_halfedge_p2).squeeze(-1)
            halfedge_p1p2 = torch.nonzero(is_halfedge_p1p2).squeeze(-1)

            batch.update({
                'node_p1': node_p1,
                'node_p2': node_p2,
                'halfedge_p1': halfedge_p1,
                'halfedge_p2': halfedge_p2,
                'halfedge_p1p2': halfedge_p1p2,
                
                # 'node_type': outputs['pred_node'].argmax(-1),
                # 'node_pos': outputs['pred_pos'],
                # 'halfedge_type': outputs['pred_halfedge'].argmax(-1),
            })
        elif ar_strategy == 'refine':  # juan (involution) with neighbor 
            batch_node = batch['node_type_batch']
            node_pos = outputs['pred_pos']
            # get config
            threshold_node = ar_config.threshold_node
            threshold_pos = ar_config.threshold_pos
            threshold_bond = ar_config.threshold_bond
            threshold_cfd_traj = ar_config.get('threshold_cfd_traj', 0)  # cfd pos traj. last half mean
            max_ar_step = ar_config.max_ar_step
            max_p2_ratio = ar_config.get('max_p2_ratio', 1)
            change_init_step = ar_config.get('change_init_step', None)
            if change_init_step is not None:
                self.init_step = change_init_step
            r = ar_config.r

            # select center
            if cfd_traj is not None:
                cfd_pos_lasthalf = torch.sigmoid(torch.stack(cfd_traj[-int(self.num_steps*self.init_step*0.5):])).mean(0)
            cfd_node = torch.sigmoid(outputs['confidence_node'][:, 0])
            cfd_pos = torch.sigmoid(outputs['confidence_pos'][:, 0])
            cfd_halfedge = torch.sigmoid(outputs['confidence_halfedge'][:, 0])
    
            # edge to bond, get cfd_node_with_bond
            halfedge_index = batch['halfedge_index']
            pred_halfedge = outputs['pred_halfedge'].argmax(-1)
            is_halfbond = (pred_halfedge > 0)
            halfbond_index = halfedge_index[:, is_halfbond]
            cfd_halfbond = cfd_halfedge[is_halfbond]
            bond_index = torch.cat([halfbond_index, halfbond_index.flip(0)], dim=-1)
            cfd_bond = torch.cat([cfd_halfbond, cfd_halfbond], dim=0)
            cfd_node_with_bond = scatter_mean(cfd_bond, bond_index[0], dim=0, dim_size=batch['node_type'].shape[0])
            
            is_center_p2 = (
                (cfd_pos_lasthalf <= threshold_cfd_traj) |
                (cfd_node <= threshold_node) |
                (cfd_pos <= threshold_pos) |
                (cfd_node_with_bond <= threshold_bond)
            )
            # index_min_cfd_pos = scatter_min(cfd_pos, batch_node, dim=0, dim_size=batch_node.max()+1)[1]
            index_min_cfd_pos = torch.zeros_like(batch_node, dtype=torch.bool)
            is_center_p2[index_min_cfd_pos] = True
            # is_center_p2[batch['node_p1']] = False  # tenure
            if 'is_finished' in batch:
                is_center_p2[batch['is_finished']] = False
            center_pos = node_pos[is_center_p2]
            batch_center = batch_node[is_center_p2]
            # select neighbor
            assign_index = radius(x=node_pos, y=center_pos, r=r,
                                  batch_x=batch_node, batch_y=batch_center)
            sel_node_curr_p2 = torch.unique(assign_index[1])

            # print(step_ar, sel_node_curr_p2.detach().cpu().numpy())
            # print(cfd_pos.detach().cpu().numpy())
            if step_ar >= max_ar_step:
                sel_node_curr_p2 = []
            is_node_p2 = torch.zeros_like(batch_node, dtype=torch.bool)
            is_node_p2[sel_node_curr_p2] = True
            is_node_p1 = ~is_node_p2
            # print('is_node_p2', is_node_p2.sum().item())

            # change node_p1 and node_p2
            node_p1 = torch.nonzero(is_node_p1).squeeze(-1)
            node_p2 = torch.nonzero(is_node_p2).squeeze(-1)
            
            # change halfedge_p1, halfedge_p2, halfedge_p1p2
            halfedge_index = batch['halfedge_index']
            left_in_p1 = is_node_p1[halfedge_index[0]]
            right_in_p1 = is_node_p1[halfedge_index[1]]
            is_halfedge_p1 = left_in_p1 & right_in_p1
            is_halfedge_p2 = (~left_in_p1) & (~right_in_p1)
            is_halfedge_p1p2 = (~is_halfedge_p1) & (~is_halfedge_p2)
            halfedge_p1 = torch.nonzero(is_halfedge_p1).squeeze(-1)
            halfedge_p2 = torch.nonzero(is_halfedge_p2).squeeze(-1)
            halfedge_p1p2 = torch.nonzero(is_halfedge_p1p2).squeeze(-1)
            
            
            # fixed (free mode of p1)
            # reset fixed indicators
            setting = self._get_setting(batch)
            if setting['part1_pert'] == 'small':
                fixed_node = batch['fixed_node']
                fixed_pos = batch['fixed_pos']
                fixed_halfedge = batch['fixed_halfedge']
            elif setting['part1_pert'] == 'free':
                fixed_node = torch.zeros_like(batch['fixed_node'])
                fixed_node[is_node_p1] = 1
                fixed_pos = torch.zeros_like(batch['fixed_pos'])
                fixed_halfedge = torch.zeros_like(batch['fixed_halfedge'])
                fixed_halfedge[is_halfedge_p1] = 1
            
            # fixed the mol if no atoms in p2 (finished)
            batch_halfedge = batch['halfedge_type_batch']
            batch_mol_p2 = torch.unique(batch_node[is_node_p2])
            batch_mol_noinp2 = torch.tensor([i for i in range(batch_node.max()+1) if i not in batch_mol_p2],
                                                dtype=torch.long, device=batch_node.device)
            is_node_noinp2 = (batch_node[:, None]==batch_mol_noinp2[None]).any(-1)
            is_halfedge_noinp2 = (batch_halfedge[:, None]==batch_mol_noinp2[None]).any(-1)
            fixed_node[is_node_noinp2] = 1
            fixed_pos[is_node_noinp2] = 1
            fixed_halfedge[is_halfedge_noinp2] = 1
            is_finished = torch.zeros_like(batch_node, dtype=torch.bool)
            is_finished[is_node_noinp2] = True

            batch.update({
                'node_p1': node_p1,
                'node_p2': node_p2,
                'halfedge_p1': halfedge_p1,
                'halfedge_p2': halfedge_p2,
                'halfedge_p1p2': halfedge_p1p2,
                'is_finished': is_finished,
                
                # 'node_type': outputs['pred_node'].argmax(-1),
                # 'node_pos': outputs['pred_pos'],
                # 'halfedge_type': outputs['pred_halfedge'].argmax(-1),
                
                'fixed_node': fixed_node,
                'fixed_pos': fixed_pos,
                'fixed_halfedge': fixed_halfedge,
            })
        else:
            raise NotImplementedError('ar_strategy not implemented:', ar_strategy)
            
            
        return batch

@register_sample_noise('pepdesign')
class PepdesignSampleNoiser(BaseSampleNoiser):
    def __init__(self,
        config, num_node_types, num_edge_types,
        mode='sample', device='cpu', ref_config=None, task_name='pepdesign',
        **kwargs
    ):
        super().__init__(task_name, config, num_node_types, num_edge_types,
            mode, device, ref_config, **kwargs)
        
        # define prior
        prior_bb = config.prior.bb if config.prior.bb != 'from_train' else self.ref_prior_config.bb
        self.prior_bb = MolPrior(prior_bb, num_node_types, num_edge_types).to(device)
        prior_sc = config.prior.sc if config.prior.sc != 'from_train' else self.ref_prior_config.sc
        self.prior_sc = MolPrior(prior_sc, num_node_types, num_edge_types).to(device)

        # define info level
        self.level_bb = MolInfoLevel(config.level.bb, device=device, mode=mode)
        self.level_sc = MolInfoLevel(config.level.sc, device=device, mode=mode)

    def sample_level(self, step, batch):
        
        level_dict = {}
        setting = self._get_setting(batch)
        mode = setting['mode']
        n_node_sc = batch['node_sc'].shape[0]
        n_halfedge_sc = batch['halfedge_sc'].shape[0]
        n_halfedge_bbsc = batch['halfedge_bbsc'].shape[0]
        
        if mode == 'full':
            level_pos_bb = self.level_bb.sample_for_mol(step, n_pos=batch['node_bb'].shape[0])
            level_node_sc, level_pos_sc, level_halfedge_sc_and_bbsc = self.level_sc.sample_for_mol(
                step, n_node=n_node_sc, n_pos=n_node_sc, n_edge=(n_halfedge_sc + n_halfedge_bbsc))
            level_dict.update({
                'pos_bb': level_pos_bb,
                'node_sc': level_node_sc,
                'pos_sc': level_pos_sc,
                'halfedge_sc': level_halfedge_sc_and_bbsc[:n_halfedge_sc],
                'halfedge_bbsc': level_halfedge_sc_and_bbsc[n_halfedge_sc:],
            })
        elif mode == 'sc':
            level_node_sc, level_pos_sc, level_halfedge_sc_and_bbsc = self.level_sc.sample_for_mol(
                step, n_node=n_node_sc, n_pos=n_node_sc, n_edge=(n_halfedge_sc + n_halfedge_bbsc))
            level_dict.update({
                'node_sc': level_node_sc,
                'pos_sc': level_pos_sc,
                'halfedge_sc': level_halfedge_sc_and_bbsc[:n_halfedge_sc],
                'halfedge_bbsc': level_halfedge_sc_and_bbsc[n_halfedge_sc:],
            })
        elif mode == 'packing':
            level_pos_sc = self.level_sc.sample_for_mol(step, n_pos=n_node_sc)
            level_dict.update({
                'pos_sc': level_pos_sc,
            })

        # halfedge_bbsc fixed
        if 'halfedge_bbsc' in level_dict:
            level_bbsc = level_dict['halfedge_bbsc']
            fixed_halfedge = batch['fixed_halfedge']
            halfedge_bbsc = batch['halfedge_bbsc']
            fixed_bbsc = (fixed_halfedge[halfedge_bbsc] == 1)
            level_bbsc[fixed_bbsc] = 1
            level_dict.update({
                'halfedge_bbsc': level_bbsc,
            })
        
        return level_dict
    
    def add_noise(self, node_type, node_pos, halfedge_type, batch,
                   from_prior=False, level_dict=None):
        
        setting = self._get_setting(batch)
        mode = setting['mode']

        # bb noise
        if mode == 'full':
            level_dict_bb = {k[:-3]:v for k, v in level_dict.items() if k.endswith('_bb')}
            node_bb = batch['node_bb']
            from_prior_bb = from_prior and self.prior_bb.config.get('from_prior', True)
            node_pos[node_bb] = self.prior_bb.add_noise(
                None, node_pos[node_bb], None,
                level_dict=level_dict_bb, from_prior=from_prior_bb, pos_only=True
            )
        
        # sc noise
        node_sc = batch['node_sc']
        halfedge_sc = batch['halfedge_sc']
        halfedge_bbsc = batch['halfedge_bbsc']
        level_dict_sc = {k[:-3]:v for k, v in level_dict.items() if k.endswith('_sc')}
        from_prior_sc = from_prior and self.prior_sc.config.get('from_prior', True)
        if mode in ['full', 'sc']:
            node_type[node_sc], node_pos[node_sc], halfedge_type[halfedge_sc] = self.prior_sc.add_noise(
                node_type[node_sc], node_pos[node_sc], halfedge_type[halfedge_sc],
                level_dict_sc, from_prior_sc
            )
            halfedge_type[halfedge_bbsc] = self.prior_sc.halfedge.add_noise(
                halfedge_type[halfedge_bbsc], level_dict['halfedge_bbsc'], from_prior_sc
            )
        elif mode == 'packing':
            node_pos[node_sc] = self.prior_sc.add_noise(
                None, node_pos[node_sc], None,
                level_dict=level_dict_sc, from_prior=from_prior_sc, pos_only=True
            )
        else:
            raise ValueError(f'Unknown mode: {mode}')

        in_dict = {'node':node_type, 'pos':node_pos, 'halfedge': halfedge_type}
        return in_dict

    def outputs2batch(self, batch, outputs):
        
        setting = self._get_setting(batch)
        mode = setting['mode']

        # node_p1, node_p2 = batch['node_p1'], batch['node_p2']
        fixed_node = (batch['fixed_node'] == 1)
        fixed_pos = (batch['fixed_pos'] == 1)
        fixed_halfedge = (batch['fixed_halfedge'] == 1)

        batch['node_type'][~fixed_node] = outputs['pred_node'][~fixed_node].argmax(-1).clone()
        batch['node_pos'][~fixed_pos] = outputs['pred_pos'][~fixed_pos].clone()
        batch['halfedge_type'][~fixed_halfedge] = outputs['pred_halfedge'][~fixed_halfedge].argmax(-1).clone()
        
        return batch


@register_sample_noise('custom')
class CustomSampleNoiser(BaseSampleNoiser):
    def __init__(self,
        config, num_node_types, num_edge_types,
        mode='sample', device='cpu', ref_config=None, task_name='custom',
        **kwargs
    ):
        super().__init__(task_name, config, num_node_types, num_edge_types,
            mode, device, ref_config, **kwargs)
        
        self.noiser_names = list(config.prior.keys())
        # # define prior
        self.prior_dict = {}
        for this_prior in config.prior.items():
            name = this_prior[0]
            prior = MolPrior(this_prior[1], num_node_types, num_edge_types).to(device)
            self.prior_dict[name] = prior

        # # define info_level
        self.level_dict = {}
        for this_level in config.level.items():
            name = this_level[0]
            leveller = MolInfoLevel(this_level[1], device=device, mode=mode)
            self.level_dict[name] = leveller
            assert name in self.noiser_names, f'Undefined name {name} of level_dict'
        assert len(self.prior_dict) == len(self.level_dict), 'config size mismatch: leveller'

        # # mapper of noisers
        self.mapper_dict = config.mapper
        assert all(name in self.noiser_names for name in self.mapper_dict.keys()), f'Undefined name {name} in mapper'
        assert len(self.prior_dict) == len(self.mapper_dict), 'config size mismatch: mapper'
        
        # # correction. post correcting preset variables
        self.correction = config.get('correction', [])

    def sample_level(self, step, batch):
        level_dict = {}
        for name in self.noiser_names:
            leveller = self.level_dict[name]
            mapper = self.mapper_dict[name]
            level_kwargs = {}
            what_list = []
            if 'node' in mapper:
                part_list = mapper['node']
                level_kwargs['n_node'] = sum(batch[f'node_part_{part}'].shape[0] for part in part_list)
                what_list.append('node')
            if 'pos' in mapper:
                part_list = mapper['pos']
                level_kwargs['n_pos'] = sum(batch[f'node_part_{part}'].shape[0] for part in part_list)
                what_list.append('pos')
            if 'edge' in mapper:
                part_list = mapper['edge']
                level_kwargs['n_edge'] = sum(batch[f'halfedge_part_{parts[0]}_{parts[1]}'].shape[0] for parts in part_list)
                what_list.append('halfedge')

            this_info_level = leveller.sample_for_mol(step, **level_kwargs)
            if len(level_kwargs) == 1:
                this_info_level = [this_info_level]  # stupid
            level_dict.update({ f'{what}_noiser_{name}': this_info_level[i]
                               for i, what in enumerate(what_list)})
        return level_dict

    def has_pocket(self, batch):
        return batch['pocket_pos'].shape[0] > 0

    def add_noise(self, node_type, node_pos, halfedge_type, batch,
                   from_prior=False, level_dict=None):
        
        for name in self.noiser_names:
            prior = self.prior_dict[name]
            mapper = self.mapper_dict[name]
            level_dict_part = {k.replace(f'_noiser_{name}', ''): v for k, v in level_dict.items()
                               if k.endswith(f'_noiser_{name}')}
            
            node_part = torch.concat([batch[f'node_part_{part}'] for part in mapper['node']]) if 'node' in mapper else []
            pos_part = torch.concat([batch[f'node_part_{part}'] for part in mapper['pos']]) if 'pos' in mapper else []
            halfedge_part = torch.concat([batch[f'halfedge_part_{part[0]}_{part[1]}'] for 
                                part in mapper['edge']]) if 'edge' in mapper else []
            noised_mol = prior.add_noise(
                node_type[node_part], node_pos[pos_part], halfedge_type[halfedge_part],
                level_dict_part, (from_prior and prior.config.get('from_prior', True))
            )
            if prior.pos_only:
                node_pos[pos_part] = noised_mol
            else:
                node_type[node_part], node_pos[pos_part], halfedge_type[halfedge_part] = noised_mol
        
        in_dict = {'node':node_type, 'pos':node_pos, 'halfedge': halfedge_type}
        return in_dict

    def outputs2batch(self, batch, outputs):
        
        # task = self._get_task(batch)
        # if task == 'ar':
        #     return self.outputs2batch_ar(batch, outputs)
        
        fixed_node = (batch['fixed_node'] == 1)
        fixed_pos = (batch['fixed_pos'] == 1)
        fixed_halfedge = (batch['fixed_halfedge'] == 1)

        # # protect correction
        if 'node' in self.correction:
            node_part_corr = torch.concat([batch[f'node_part_{part}'] for part in self.correction['node']])
            node_corrected = batch['node_type'][node_part_corr].clone()
        if 'pos' in self.correction:
            pos_part_corr = torch.concat([batch[f'node_part_{part}'] for part in self.correction['pos']])
            pos_corrected = batch['node_pos'][pos_part_corr].clone()
        if 'edge' in self.correction:
            halfedge_part_corr = torch.concat([batch[f'halfedge_part_{part[0]}_{part[1]}'] for 
                                part in self.correction['edge']])
            halfedge_corrected = batch['halfedge_type'][halfedge_part_corr].clone()
        
        batch['node_type'][~fixed_node] = outputs['pred_node'][~fixed_node].argmax(-1).clone()
        batch['node_pos'][~fixed_pos] = outputs['pred_pos'][~fixed_pos].clone()
        batch['halfedge_type'][~fixed_halfedge] = outputs['pred_halfedge'][~fixed_halfedge].argmax(-1).clone()

        # # apply correction
        if 'node' in self.correction:
            batch['node_type'][node_part_corr] = node_corrected
        if 'pos' in self.correction:
            batch['node_pos'][pos_part_corr] = pos_corrected
        if 'edge' in self.correction:
            batch['halfedge_type'][halfedge_part_corr] = halfedge_corrected
        
        return batch

    def _get_correction(self, batch):
        if 'node' in self.correction:
            node_part_corr = torch.concat([batch[f'node_part_{part}'] for part in self.correction['node']])
            node_corrected = batch[node_part_corr]
        if 'pos' in self.correction:
            pos_part_corr = torch.concat([batch[f'node_part_{part}'] for part in self.correction['pos']])
            pos_corrected = batch[pos_part_corr]
        if 'edge' in self.correction:
            halfedge_part_corr = torch.concat([batch[f'halfedge_part_{part[0]}_{part[1]}'] for 
                                part in self.correction['edge']])
            halfedge_corrected = batch[halfedge_part_corr]
