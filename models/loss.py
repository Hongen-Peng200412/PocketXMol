"""定义 PocketXMol 的联合恢复损失与变量级置信度监督。

构象与 docking 训练使用 ``IndividualTasksLoss``：训练入口通过 ``get_loss_func`` 按
``config.loss.name`` 实例化它，再把同一联合批次拆成 ``mixed``、``conf``、
``dock`` 等统计口径。模型始终直接预测干净的原子类别、三维坐标和半边类别；
损失再依据 ``fixed_*`` prompt 区分待恢复变量与作为条件给出的已知变量。

记号：``N`` 为批内配体原子总数，``H`` 为无向完整图半边总数，``D`` 为二面角
实例数，``B`` 为图数，``C_n/C_e`` 分别为原子与半边类别数。坐标单位均为 Å。
"""
from copy import deepcopy
import numpy as np
import logging
from torch import nn
import torch
from torch.nn import functional as F
from torch_geometric.nn.pool import radius

from models.diffusion import index_to_log_onehot
from models.transition import GeneralCategoricalTransition, ContigousTransition
from models.corrector import get_dihedral_batch

# ``LOSS_DICT``：dict[str, type[nn.Module]]，键来自 YAML ``loss.name``，值为尚未实例化的损失类。
LOSS_DICT = {}

def register_loss(name):
    """创建把损失类写入全局注册表的装饰器。

    输入参数:
        - name: str，YAML ``loss.name`` 使用的稳定注册键。

    返回值:
        - decorator: callable，接收一个 ``nn.Module`` 子类，将其写入 ``LOSS_DICT[name]`` 后原样返回该类。
    """

    def decorator(cls):
        # ``cls``：type[nn.Module]，尚未实例化的损失类。
        # ``LOSS_DICT[name]``：type[nn.Module]，把当前类写到注册键 ``name``；同名键沿用 Python 字典的后写入覆盖语义。
        LOSS_DICT[name] = cls
        return cls
    return decorator

def get_loss_func(config, *args, **kwargs):
    """根据损失配置创建一个损失模块。

    输入参数:
        - config: OmegaConf|Mapping，损失配置子树。
        - config.name: str，可选，``LOSS_DICT`` 注册键；缺失时回退历史键 ``asymloss``，reduced 训练显式为 ``individual_tasks``。
        - *args: tuple，原样透传给所选损失类构造器的位置参数。
        - **kwargs: dict，原样透传给所选损失类构造器的关键字参数。

    返回值:
        - loss_module: nn.Module，按 ``LOSS_DICT[config.name]`` 实例化的损失对象。

    异常:
        - ValueError: 配置键未注册到 ``LOSS_DICT``。
    """

    if 'name' not in config:
        logging.warning('No loss type specified, using asymloss by default')
    # ``name``：str；缺省回退键为历史名称 ``asymloss``，但是否注册由当前模块决定。
    name = config.get('name', 'asymloss')
    if name not in LOSS_DICT:
        raise ValueError(f'Unknown loss type: {name}')
    return LOSS_DICT[name](config, *args, **kwargs)



@register_loss('individual_tasks')
class IndividualTasksLoss(nn.Module):
    """按任务与 fixed prompt 分流联合恢复损失。

    配置字段:
        - config.weights.node: float，待恢复原子类别交叉熵进入 ``<scope>/total`` 的权重。
        - config.weights.pos: float，待恢复坐标 MSE 进入 ``<scope>/total`` 的权重。
        - config.weights.edge: float，待恢复半边类别交叉熵进入 ``<scope>/total`` 的权重。
        - config.weights.dist: float，同一刚体域内待恢复半边距离 MSE 的权重。
        - config.weights.dih: float，已注释二面角 ``1-cos(Δθ)`` 的权重。
        - config.weights.fixed_node: float，条件原子类别交叉熵的权重。
        - config.weights.fixed_pos: float，条件坐标 MSE 的权重。
        - config.weights.fixed_edge: float，条件半边类别交叉熵的权重。
        - config.weights.fixed_dist: float，条件半边距离 MSE 的权重。
        - config.tasks: list[str]，需要单独记录的图级任务标签；必须与 ``batch.task`` 字符串一致。
        - config.confidence: Mapping|None，可选置信度损失配置；存在时建立 ``ConfidenceLoss``，仅把 ``cfd_total`` 加到 ``mixed/total``。
        - config.pocket_dist.radius: float，可选，建立配体—口袋监督邻接的半径，单位 Å。
        - config.pocket_dist.weight: float，可选，配体—口袋距离 MSE 进入 ``mixed/total`` 的乘子。
        - config.size_weighted: bool，可选；为真时半边逐元素损失按所属分子的原子数尺度归一化。

    输入字段:
        - batch.node_type: LongTensor，形状为 (N,)，干净原子类别索引。
        - batch.node_pos: FloatTensor，形状为 (N, 3)，口袋中心化后的干净配体坐标，单位 Å。
        - batch.halfedge_type: LongTensor，形状为 (H,)，完整无向半边类别索引，包含无键类别。
        - batch.halfedge_index: LongTensor，形状为 (2, H)，第 h 列是 ``halfedge_type[h]`` 的两个批内原子端点。
        - batch.fixed_node: LongTensor|BoolTensor，形状为 (N,)，0 表示原子类别待恢复，1 表示条件。
        - batch.fixed_pos: LongTensor|BoolTensor，形状为 (N,)，0 表示坐标待恢复，1 表示条件。
        - batch.fixed_halfedge: LongTensor|BoolTensor，形状为 (H,)，0 表示半边类别待恢复，1 表示条件。
        - batch.fixed_halfdist: LongTensor|BoolTensor，形状为 (H,)，0 表示端点距离待恢复，1 表示条件。
        - batch.node_type_batch: LongTensor，形状为 (N,)，每个原子的图归属索引，取值范围为 ``[0, B)``。
        - batch.halfedge_type_batch: LongTensor，形状为 (H,)，每条半边的图归属索引，取值范围为 ``[0, B)``。
        - batch.task: list[str]，长度为 B，第 b 项是图 b 的任务名。
        - batch.domain_node_index: LongTensor，形状为 (2, K)，第一行是连续刚体域号，第二行是域内批次原子索引。
        - batch.tor_bonds_anno: LongTensor，形状为 (T, 3)，每行依次为扭转执行层级与旋转键两个批内端点。
        - batch.dihedral_pairs_anno: LongTensor，形状为 (D, 3)，每行依次为旋转键编号及四原子二面角的两个外侧端点。
        - outputs.pred_node: FloatTensor，形状为 (N, C_n)，干净原子类别 logits。
        - outputs.pred_pos: FloatTensor，形状为 (N, 3)，与 ``node_pos`` 对齐的干净坐标预测，单位 Å。
        - outputs.pred_halfedge: FloatTensor，形状为 (H, C_e)，干净半边类别 logits。
        - outputs.confidence_node: FloatTensor，可选，形状为 (N, 1)，启用置信度辅助损失时的原子 confidence 原始输出。
        - outputs.confidence_pos: FloatTensor，可选，形状为 (N, 1)，启用置信度辅助损失时的坐标 confidence 原始输出。
        - outputs.confidence_halfedge: FloatTensor，可选，形状为 (H, 1)，启用置信度辅助损失时的半边 confidence 原始输出。

    返回字段:
        - loss_dict.<scope>/node: 标量 Tensor，scope 为 ``mixed`` 或配置任务名，表示待恢复原子类别交叉熵。
        - loss_dict.<scope>/fixed_node: 标量 Tensor，条件原子类别交叉熵。
        - loss_dict.<scope>/pos: 标量 Tensor，待恢复坐标逐分量 MSE，单位 Å²。
        - loss_dict.<scope>/fixed_pos: 标量 Tensor，条件坐标逐分量 MSE，单位 Å²。
        - loss_dict.<scope>/edge: 标量 Tensor，待恢复半边类别交叉熵。
        - loss_dict.<scope>/fixed_edge: 标量 Tensor，条件半边类别交叉熵。
        - loss_dict.<scope>/dist: 标量 Tensor，同域待恢复半边距离 MSE，单位 Å²。
        - loss_dict.<scope>/fixed_dist: 标量 Tensor，条件半边距离 MSE，单位 Å²。
        - loss_dict.<scope>/dih: 标量 Tensor，二面角 ``1-cos(Δθ)`` 均值，无量纲。
        - loss_dict.<scope>/total: 标量 Tensor，上述当前 scope 叶按 ``config.weights`` 加权后的和。
        - loss_dict.mixed/cfd_total: 标量 Tensor，可选，三个 confidence 分量的配置加权和。
        - loss_dict.mixed/cfd_node: 标量 Tensor，可选，原子 confidence BCE。
        - loss_dict.mixed/cfd_pos: 标量 Tensor，可选，坐标 confidence MSE。
        - loss_dict.mixed/cfd_edge: 标量 Tensor，可选，半边 confidence BCE。
        - loss_dict.mixed/p_dist: 标量 Tensor，可选，乘过 ``pocket_dist.weight`` 的配体—口袋距离 MSE。

    注意:
        - ``mixed/*`` 覆盖整批并用于训练，``<task>/*`` 是同一损失在任务子集上的日志。
        - fixed 项不是被忽略项，而是显式监督模型保持作为条件给出的变量。
    """

    def __init__(self, config):
        """保存联合任务损失配置，并按可选子树建立辅助损失模块。

        输入参数:
            - config.weights: Mapping，损失叶名到 ``<scope>/total`` 汇总权重的映射。
            - config.weights.node: float，待恢复原子类别交叉熵权重。
            - config.weights.pos: float，待恢复坐标 MSE 权重。
            - config.weights.edge: float，待恢复半边类别交叉熵权重。
            - config.weights.dist: float，同域待恢复半边距离 MSE 权重。
            - config.weights.dih: float，二面角 ``1-cos(Δθ)`` 权重。
            - config.weights.fixed_node: float，条件原子类别交叉熵权重。
            - config.weights.fixed_pos: float，条件坐标 MSE 权重。
            - config.weights.fixed_edge: float，条件半边类别交叉熵权重。
            - config.weights.fixed_dist: float，条件半边距离 MSE 权重。
            - config.tasks: list[str]，需要单独统计损失的 ``batch.task`` 标签。
            - config.confidence: Mapping，可选，存在时传给 ``ConfidenceLoss`` 建立置信度辅助监督。
            - config.pocket_dist: Mapping，可选，存在时启用配体—口袋距离辅助监督。
            - config.pocket_dist.radius: float，建立配体—口袋监督邻接的半径，单位 Å。
            - config.pocket_dist.weight: float，配体—口袋距离 MSE 的乘子。
            - config.size_weighted: bool，可选，是否按所属图的原子数尺度归一化半边逐元素损失。
        """
        super().__init__()
        # ``self.config``：原始 OmegaConf/映射配置；仅保存引用，不在构造器内改写。
        self.config = config
        # ``self.weights``：``loss_leaf -> float``；在 ``forward`` 汇总每个任务的 total 时读取。
        self.weights = config.weights
        # ``self.task_list``：``list[str]``；决定额外建立哪些逐任务布尔掩码与日志键。
        self.task_list = config.tasks
        # ``self.ce_loss``：对 ``(N, C_n)`` 或 ``(H, C_e)`` logits 返回未归约的 ``(N,)``/``(H,)`` 交叉熵。
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        if hasattr(config, 'confidence'):
            # ``self.cfd_loss``：嵌套置信度损失模块；其参数与主去噪网络共同反向传播。
            self.cfd_loss = ConfidenceLoss(config.confidence)
        else:
            # ``self.cfd_loss``：``None`` 表示训练配置没有启用 confidence head 监督。
            self.cfd_loss = None
        # ``self.pocket_dist_config``：可选口袋距离配置；``None`` 时不构造配体—口袋辅助项。
        self.pocket_dist_config = getattr(config, 'pocket_dist', None)
        # ``self.size_weighted``：bool；控制半边逐元素损失是否除以图尺度权重。
        self.size_weighted = getattr(config, 'size_weighted', False)
            
        
    def _selected_mean_loss(self, loss_total, select, size_node=None):
        """对布尔选择后的逐元素损失求均值，并显式处理空集合。

        输入参数:
            - loss_total: Tensor，形状为 (L, ...)；第一维与 ``select`` 对齐，坐标分支为 (N, 3)，类别分支为 (N,) 或 (H,)。
            - select: BoolTensor，形状为 (L,)，选择参与当前任务与 fixed 分组的实体。
            - size_node: FloatTensor|None，形状为 (L,)；当前只在半边分支传入，作为逐半边除数。

        返回值:
            - loss_mean: 标量 Tensor；空选择返回同设备 0，坐标分支同时对所选原子与 xyz 三个分量求均值。
        """

        if select.sum() == 0:
            return torch.tensor(0.0, device=loss_total.device)
        else:
            if size_node is None:
                return torch.mean(loss_total[select])
            else:
                return torch.mean(loss_total[select] / (size_node[select]+1e-3))
        
    def pos_loss(self, node_pos, pred_pos, unfixed_pos, task_index_dict=None):
        """计算按任务拆分的坐标 MSE。

        输入参数:
            - node_pos: FloatTensor，形状为 (N, 3)，干净坐标标签，单位 Å。
            - pred_pos: FloatTensor，形状为 (N, 3)，模型坐标预测，单位 Å。
            - unfixed_pos: BoolTensor，形状为 (N,)，为真表示坐标待恢复。
            - task_index_dict: dict[str, BoolTensor]|None，每个叶形状为 (N,)，为真表示原子属于该任务；为空时建立覆盖全部原子的 ``mixed`` 叶。

        返回字段:
            - loss_pos_tasked.<task>/pos: 标量 Tensor，当前任务待恢复原子的逐分量 MSE，单位 Å²。
            - loss_pos_tasked.<task>/fixed_pos: 标量 Tensor，当前任务条件原子的逐分量 MSE，单位 Å²。
        """

        # ``loss_pos_total``：FloatTensor，形状为 (N, 3)，逐原子逐坐标分量平方误差，单位 Å²。
        loss_pos_total = F.mse_loss(pred_pos, node_pos, reduction='none')

        if task_index_dict is None:
            # ``task_index_dict``：缺省 mixed 掩码覆盖全部 N 个原子；dtype/设备与 ``unfixed_pos`` 一致。
            task_index_dict = {'mixed': torch.ones_like(unfixed_pos)}

        # ``loss_pos_tasked``：任务名到两个标量项的扁平映射，供 total 汇总。
        loss_pos_tasked = {}
        # ``task``：str，当前任务名，用作返回叶键前缀。
        # ``select``：BoolTensor，形状为 (N,)，按原子标出属于当前任务的实体。
        for task, select in task_index_dict.items():
            # ``loss_pos``：标量 Å²；只统计当前任务且 fixed_pos=0 的原子及 xyz 分量。
            loss_pos = self._selected_mean_loss(loss_pos_total, unfixed_pos & select)
            # ``loss_pos_fixed``：标量 Å²；统计当前任务中作为条件给出的坐标。
            loss_pos_fixed = self._selected_mean_loss(loss_pos_total, (~unfixed_pos) & select)
            loss_pos_tasked.update({
                f'{task}/pos': loss_pos,
                f'{task}/fixed_pos': loss_pos_fixed
            })

        return loss_pos_tasked
    
    def node_loss(self, node_type, pred_node, unfixed_node, task_index_dict=None):
        """计算按任务拆分的原子类别交叉熵。

        输入参数:
            - node_type: LongTensor，形状为 (N,)，干净原子类别标签。
            - pred_node: FloatTensor，形状为 (N, C_n)，原子类别 logits。
            - unfixed_node: BoolTensor，形状为 (N,)，为真表示原子类别待恢复。
            - task_index_dict: dict[str, BoolTensor]|None，每个叶形状为 (N,)，为真表示原子属于该任务；为空时建立覆盖全部原子的 ``mixed`` 叶。

        返回字段:
            - loss_node_tasked.<task>/node: 标量 Tensor，当前任务待恢复原子的平均交叉熵。
            - loss_node_tasked.<task>/fixed_node: 标量 Tensor，当前任务条件原子的平均交叉熵。
        """

        # ``loss_node_total``：FloatTensor，形状为 (N,)，与原子顺序一一对齐的未归约交叉熵。
        loss_node_total = self.ce_loss(pred_node, node_type)

        if task_index_dict is None:
            # ``task_index_dict``：缺省 mixed 掩码覆盖全部原子。
            task_index_dict = {'mixed': torch.ones_like(unfixed_node)}

        # ``loss_node_tasked``：扁平的 ``<task>/node`` 与 ``fixed_node`` 标量映射。
        loss_node_tasked = {}
        # ``task``：str，当前任务名，用作返回叶键前缀。
        # ``select``：BoolTensor，形状为 (N,)，按原子标出属于当前任务的实体。
        for task, select in task_index_dict.items():
            # ``loss_node``：当前任务待恢复原子的平均交叉熵标量。
            loss_node = self._selected_mean_loss(loss_node_total, unfixed_node & select)
            # ``loss_node_fixed``：当前任务条件原子的平均交叉熵标量。
            loss_node_fixed = self._selected_mean_loss(loss_node_total, (~unfixed_node) & select)
            loss_node_tasked.update({
                f'{task}/node': loss_node,
                f'{task}/fixed_node': loss_node_fixed
            })
    
        return loss_node_tasked
        
    def halfedge_loss(self, halfedge_type, pred_halfedge, unfixed_halfedge, task_index_dict=None, size_node=None):
        """计算按任务拆分的完整半边类别交叉熵。

        输入参数:
            - halfedge_type: LongTensor，形状为 (H,)，干净完整半边类别标签。
            - pred_halfedge: FloatTensor，形状为 (H, C_e)，半边类别 logits。
            - unfixed_halfedge: BoolTensor，形状为 (H,)，为真表示半边类别待恢复。
            - task_index_dict: dict[str, BoolTensor]|None，每个叶形状为 (H,)，为真表示半边属于该任务；为空时建立 ``mixed`` 叶。
            - size_node: FloatTensor|None，形状为 (H,)，可选的逐半边图尺度除数。

        返回字段:
            - loss_edge_tasked.<task>/edge: 标量 Tensor，当前任务待恢复半边的平均交叉熵，可选经过图尺度归一化。
            - loss_edge_tasked.<task>/fixed_edge: 标量 Tensor，当前任务条件半边的平均交叉熵，可选经过图尺度归一化。
        """

        # ``loss_edge_total``：FloatTensor，形状为 (H,)，每条无向半边的未归约交叉熵。
        loss_edge_total = self.ce_loss(pred_halfedge, halfedge_type)

        if task_index_dict is None:
            # ``task_index_dict``：缺省 mixed 掩码覆盖全部 H 条半边。
            task_index_dict = {'mixed': torch.ones_like(unfixed_halfedge)}

        # ``loss_edge_tasked``：扁平的逐任务待恢复/固定半边标量映射。
        loss_edge_tasked = {}
        # ``task``：str，当前任务名，用作返回叶键前缀。
        # ``select``：BoolTensor，形状为 (H,)，按半边标出属于当前任务的实体。
        for task, select in task_index_dict.items():
            # ``loss_edge``：当前任务待恢复半边的平均交叉熵，可选按图尺度归一化。
            loss_edge = self._selected_mean_loss(loss_edge_total, unfixed_halfedge & select, size_node)
            # ``loss_edge_fixed``：当前任务条件半边的平均交叉熵。
            loss_edge_fixed = self._selected_mean_loss(loss_edge_total, (~unfixed_halfedge) & select, size_node)
            loss_edge_tasked.update({
                f'{task}/edge': loss_edge,
                f'{task}/fixed_edge': loss_edge_fixed
            })
        
        return loss_edge_tasked

    def dist_loss(self, gt_dist, pred_dist, unfixed_dist, batch, task_index_dict=None):
        """监督同一刚体域内的待恢复距离及所有 fixed 距离。

        输入参数:
            - gt_dist: FloatTensor，形状为 (H,)，干净半边端点距离，单位 Å。
            - pred_dist: FloatTensor，形状为 (H,)，预测坐标导出的端点距离，单位 Å。
            - unfixed_dist: BoolTensor，形状为 (H,)，为真表示 ``fixed_halfdist==0``。
            - batch: PyG Batch，提供完整半边端点、配体原子数与刚体域注释。
            - batch.halfedge_index: LongTensor，形状为 (2, H)，第 h 列对应 ``gt_dist[h]`` 与 ``pred_dist[h]``。
            - batch.domain_node_index: LongTensor，形状为 (2, K)，第一行是域号，第二行是域内批次原子索引。
            - batch.node_type: LongTensor，形状为 (N,)，这里只借用长度、dtype 与 device 建立逐原子域号。
            - task_index_dict: dict[str, BoolTensor]|None，每个叶形状为 (H,)，为真表示半边属于该任务。

        返回字段:
            - loss_dist_tasked.<task>/dist: 标量 Tensor，当前任务同一非负刚体域内待恢复半边的距离 MSE，单位 Å²。
            - loss_dist_tasked.<task>/fixed_dist: 标量 Tensor，当前任务全部条件半边的距离 MSE，单位 Å²。

        注意:
            - 待恢复距离还必须满足“两端属于同一非负刚体域”，fixed 距离分支不套用该域筛选。
            - free 构象/docking 没有域注释时，待恢复 distance 选择可以为空并返回标量 0。
        """

        if task_index_dict is None:
            # ``task_index_dict``：缺省 mixed 掩码覆盖全部 H 条半边。
            task_index_dict = {'mixed': torch.ones_like(unfixed_dist)}
            
        # ``halfedge_index``：LongTensor，形状为 (2, H)，列 h 的两个端点对应 ``gt_dist[h]``。
        halfedge_index = batch['halfedge_index']
        # ``domain_node_index``：LongTensor，形状为 (2, K)，第一行域号，第二行批内原子号。
        domain_node_index = batch['domain_node_index']
        # ``domain_index_of_node``：LongTensor，形状为 (N,)；-1 表示该原子不在任何已注释域。
        domain_index_of_node = - torch.ones_like(batch['node_type'])
        # ``domain_index_of_node``：按 K 个域注释条目写回域号；第二行原子索引决定被覆盖的位置。
        domain_index_of_node[domain_node_index[1]] = domain_node_index[0]
        # ``inner_domain``：BoolTensor，形状为 (H,)；仅保留同一非负域内且待恢复的半边。
        inner_domain = ((domain_index_of_node[halfedge_index[0]] >= 0) & 
            (domain_index_of_node[halfedge_index[0]] == domain_index_of_node[halfedge_index[1]]) & 
            unfixed_dist)

        # ``loss_dist_tasked``：逐任务 ``dist/fixed_dist`` 标量 MSE 的扁平映射。
        loss_dist_tasked = {}
        # ``task``：str，当前任务名，用作返回叶键前缀。
        # ``select``：BoolTensor，形状为 (H,)，按半边标出属于当前任务的实体。
        for task, select in task_index_dict.items():
            if (inner_domain & select).sum() == 0:
                # ``loss_dist``：空选择时为同设备标量 0，避免 ``mean(empty)=nan``。
                loss_dist = torch.tensor(0.0, device=gt_dist.device)
            else:
                # ``loss_dist``：当前任务同域待恢复距离的平均平方误差，单位 Å²。
                loss_dist = F.mse_loss(pred_dist[inner_domain & select], gt_dist[inner_domain & select], reduction='mean')
            if ((~unfixed_dist)&select).sum() == 0:
                # ``loss_dist_fixed``：当前任务没有 fixed distance 时的标量 0。
                loss_dist_fixed = torch.tensor(0.0, device=gt_dist.device)
            else:
                # ``loss_dist_fixed``：所有当前任务 fixed 半边距离的平均平方误差，单位 Å²。
                loss_dist_fixed = F.mse_loss(pred_dist[(~unfixed_dist)&select], gt_dist[(~unfixed_dist)&select], reduction='mean')
            loss_dist_tasked.update({
                f'{task}/dist': loss_dist,
                f'{task}/fixed_dist': loss_dist_fixed
            })
        return loss_dist_tasked

    def dihedral_loss(self, node_pos, pred_pos, batch, task_index_dict=None):
        """以周期连续的 ``1-cos(Δθ)`` 监督已注释二面角。

        输入参数:
            - node_pos: FloatTensor，形状为 (N, 3)，干净配体坐标，单位 Å。
            - pred_pos: FloatTensor，形状为 (N, 3)，模型预测坐标，单位 Å。
            - batch: PyG Batch，提供旋转键与二面角外侧端点注释。
            - batch.tor_bonds_anno: LongTensor，形状为 (T, 3)，每行是执行层级与旋转键端点。
            - batch.dihedral_pairs_anno: LongTensor，形状为 (D, 3)，每行是旋转键编号与两个外侧端点。
            - task_index_dict: dict[str, BoolTensor]|None，每个叶形状为 (D,)，沿二面角实例而非旋转键对齐。

        返回字段:
            - loss_dih_tasked.<task>/dih: 标量 Tensor，当前任务二面角 ``1-cos(predθ-gtθ)`` 的均值，无量纲；无实例时为 0。
        """

        # ``pred_sin``：FloatTensor，形状为 (D, 1)，预测二面角的正弦值。
        # ``pred_cos``：FloatTensor，形状为 (D, 1)，预测二面角的余弦值。
        pred_sin, pred_cos = get_dihedral_batch(pred_pos, batch['tor_bonds_anno'], batch['dihedral_pairs_anno'])
        # ``gt_sin``：FloatTensor，形状为 (D, 1)，干净二面角的正弦值，与 ``pred_sin`` 逐实例对齐。
        # ``gt_cos``：FloatTensor，形状为 (D, 1)，干净二面角的余弦值，与 ``pred_cos`` 逐实例对齐。
        gt_sin, gt_cos = get_dihedral_batch(node_pos, batch['tor_bonds_anno'], batch['dihedral_pairs_anno'])

        if task_index_dict is None:
            # ``task_index_dict``：缺省 mixed 掩码覆盖 D 个二面角实例。
            task_index_dict = {'mixed': torch.ones(len(pred_sin), device=pred_sin.device, dtype=torch.bool)}

        # ``cos_delta``：FloatTensor，形状为 (D, 1)，等于 cos(predθ-gtθ)，天然处理 ±π 周期边界。
        cos_delta = pred_cos * gt_cos + pred_sin * gt_sin
        # ``loss_dih_tasked``：逐任务无量纲二面角损失映射。
        loss_dih_tasked = {}
        # ``task``：str，当前任务名，用作返回叶键前缀。
        # ``select``：BoolTensor，形状为 (D,)，按二面角实例标出属于当前任务的实体。
        for task, select in task_index_dict.items():
            if len(cos_delta[select]) == 0:
                # ``loss_dih``：当前任务没有二面角实例时的标量 0。
                loss_dih = torch.tensor(0.0, device=node_pos.device)
            else:
                # ``loss_dih``：当前任务所有二面角的平均 ``1-cos(Δθ)``。
                loss_dih = torch.mean(1 - cos_delta[select])
            loss_dih_tasked.update({
                f'{task}/dih': loss_dih
            })

        return loss_dih_tasked
    
    def forward(self, batch, outputs, tasked=True):
        """计算整批联合恢复损失，并生成 mixed 与逐任务日志项。

        输入参数:
            - batch.node_type: LongTensor，形状为 (N,)，干净原子类别索引。
            - batch.node_pos: FloatTensor，形状为 (N, 3)，干净配体局部坐标，单位 Å。
            - batch.halfedge_type: LongTensor，形状为 (H,)，干净完全图半边类别。
            - batch.halfedge_index: LongTensor，形状为 (2, H)，半边端点，数值索引批内配体原子。
            - batch.fixed_node: LongTensor|BoolTensor，形状为 (N,)，1 表示原子类别为条件。
            - batch.fixed_pos: LongTensor|BoolTensor，形状为 (N,)，1 表示坐标为条件。
            - batch.fixed_halfedge: LongTensor|BoolTensor，形状为 (H,)，1 表示半边类别为条件。
            - batch.fixed_halfdist: LongTensor|BoolTensor，形状为 (H,)，1 表示半边端点距离为条件。
            - batch.node_type_batch: LongTensor，形状为 (N,)，每个原子的图归属编号。
            - batch.halfedge_type_batch: LongTensor，形状为 (H,)，每条半边的图归属编号。
            - batch.task: list[str]，长度为 B，第 b 项是图 b 的任务名。
            - batch.domain_node_index: LongTensor，形状为 (2, K)，刚体域—原子归属索引。
            - batch.tor_bonds_anno: LongTensor，形状为 (T, 3)，扭转层级和两个轴端点。
            - batch.dihedral_pairs_anno: LongTensor，形状为 (Q, 3)，扭转行号和两个二面角外侧端点。
            - outputs.pred_node: FloatTensor，形状为 (N, C_n)，干净原子类别 logits。
            - outputs.pred_pos: FloatTensor，形状为 (N, 3)，干净配体局部坐标预测，单位 Å。
            - outputs.pred_halfedge: FloatTensor，形状为 (H, C_e)，干净半边类别 logits。
            - outputs.confidence_node: FloatTensor，形状为 (N, 1)，可选原子 confidence 原始输出。
            - outputs.confidence_pos: FloatTensor，形状为 (N, 1)，可选坐标 confidence 原始输出。
            - outputs.confidence_halfedge: FloatTensor，形状为 (H, 1)，可选半边 confidence 原始输出。
            - tasked: bool，为真时同时建立配置任务掩码与逐任务日志，为假时只建立 ``mixed/*``。

        返回字段:
            - loss_dict.<scope>/node: 标量 Tensor，scope 为 ``mixed`` 或任务名，表示待恢复原子类别交叉熵。
            - loss_dict.<scope>/fixed_node: 标量 Tensor，条件原子类别交叉熵。
            - loss_dict.<scope>/pos: 标量 Tensor，待恢复坐标逐分量 MSE，单位 Å²。
            - loss_dict.<scope>/fixed_pos: 标量 Tensor，条件坐标逐分量 MSE，单位 Å²。
            - loss_dict.<scope>/edge: 标量 Tensor，待恢复半边类别交叉熵。
            - loss_dict.<scope>/fixed_edge: 标量 Tensor，条件半边类别交叉熵。
            - loss_dict.<scope>/dist: 标量 Tensor，同域待恢复半边距离 MSE，单位 Å²。
            - loss_dict.<scope>/fixed_dist: 标量 Tensor，条件半边距离 MSE，单位 Å²。
            - loss_dict.<scope>/dih: 标量 Tensor，二面角 ``1-cos(Δθ)`` 均值。
            - loss_dict.<scope>/total: 标量 Tensor，当前 scope 各分量按 ``config.weights`` 加权后的和。
            - loss_dict.mixed/cfd_total: 标量 Tensor，可选 confidence 总损失。
            - loss_dict.mixed/cfd_node: 标量 Tensor，可选原子 confidence BCE。
            - loss_dict.mixed/cfd_pos: 标量 Tensor，可选坐标 confidence MSE。
            - loss_dict.mixed/cfd_edge: 标量 Tensor，可选半边 confidence BCE。
            - loss_dict.mixed/p_dist: 标量 Tensor，可选加权配体—口袋距离 MSE。
        """

        # ``node_pos``：FloatTensor，形状为 (N, 3)，干净监督坐标，单位 Å。
        node_pos = batch['node_pos']
        # ``node_type``：LongTensor，形状为 (N,)，干净原子类别索引。
        node_type = batch['node_type']
        # ``halfedge_type``：LongTensor，形状为 (H,)，干净完整半边类别索引。
        halfedge_type = batch['halfedge_type']
        # ``halfedge_index``：LongTensor，形状为 (2, H)，与 ``halfedge_type`` 顺序严格对齐。
        halfedge_index = batch['halfedge_index']
        # ``gt_dist``：FloatTensor，形状为 (H,)，由干净坐标计算的半边端点距离，单位 Å。
        gt_dist = torch.norm(node_pos[halfedge_index[0]] - node_pos[halfedge_index[1]], dim=-1)

        # ``pred_pos``：FloatTensor，形状为 (N, 3)，模型直接给出的干净坐标预测，单位 Å。
        pred_pos = outputs['pred_pos']
        # ``pred_node``：FloatTensor，形状为 (N, C_n)，原子类别 logits。
        pred_node = outputs['pred_node']
        # ``pred_halfedge``：FloatTensor，形状为 (H, C_e)，完整半边类别 logits。
        pred_halfedge = outputs['pred_halfedge']
        # ``pred_dist``：FloatTensor，形状为 (H,)，从预测坐标导出的端点距离，单位 Å。
        pred_dist = torch.norm(pred_pos[halfedge_index[0]] - pred_pos[halfedge_index[1]], dim=-1)
        
        # ``unfixed_node``：BoolTensor，形状为 (N,)；真值表示原子类别是当前任务的待恢复变量。
        unfixed_node = (batch['fixed_node'] == 0)
        # ``unfixed_pos``：BoolTensor，形状为 (N,)；真值表示坐标需由模型恢复。
        unfixed_pos = (batch['fixed_pos'] == 0)
        # ``unfixed_halfedge``：BoolTensor，形状为 (H,)；真值表示半边类别需恢复。
        unfixed_halfedge = (batch['fixed_halfedge'] == 0)
        # ``unfixed_dist``：BoolTensor，形状为 (H,)；真值表示端点距离未作为结构约束固定。
        unfixed_dist = (batch['fixed_halfdist'] == 0)
        
        if self.size_weighted:
            # ``index_batch``：LongTensor，形状为 (N,)，每个原子的图归属，取值范围 ``[0,B)``。
            index_batch = batch['node_type_batch']
            # ``size``：FloatTensor，形状为 (B,)，每个图的配体原子数。
            size = torch.bincount(index_batch).float()
            # ``size_node``：FloatTensor，形状为 (H,)，把所属图的 ``N_b+10`` 广播到每条半边。
            size_node = (size[batch['halfedge_type_batch']] + 10)
            # ``size_node``：归一化后全批均值为 1，仅改变不同分子半边损失的相对权重。
            size_node = size_node / torch.mean(size_node)
        else:
            # ``size_node``：FloatTensor，形状为 (H,) 的全 1；等价于不做图尺度加权。
            size_node = torch.ones_like(halfedge_type).float()
        
        if tasked:
            # ``device``：坐标所在设备，供新建任务掩码避免 CPU/GPU 混用。
            device = node_pos.device
            # ``task_list``：list[str]，来自配置而不是从当前批次动态去重。
            task_list = self.task_list
            # task_list = np.unique(batch['task']).tolist()
            # ``task_batch_dict``：task -> BoolTensor，形状为 (B,)，图级任务归属。
            task_batch_dict = {}
            # ``task_node_dict``：task -> BoolTensor，形状为 (N,)；mixed 先覆盖全部原子。
            task_node_dict = {'mixed': torch.ones_like(unfixed_node)}
            # ``task_edge_dict``：task -> BoolTensor，形状为 (H,)；mixed 先覆盖全部半边。
            task_edge_dict = {'mixed': torch.ones_like(unfixed_halfedge)}
            # ``task_dih_dict``：task -> BoolTensor，形状为 (D,)；mixed 覆盖全部二面角实例。
            task_dih_dict = {'mixed': torch.ones(len(batch['dihedral_pairs_anno']), device=device, dtype=torch.bool)}
            # ``batch_node``：LongTensor，形状为 (N,)，从图级任务掩码扩展到原子的索引桥梁。
            batch_node = batch['node_type_batch']
            # ``batch_halfedge``：LongTensor，形状为 (H,)，从图级任务掩码扩展到半边的索引桥梁。
            batch_halfedge = batch['halfedge_type_batch']
            # ``batch_dih``：LongTensor，形状为 (D,)；用每个二面角的第一个外侧端点确定所属图。
            batch_dih = batch_node.index_select(0, batch['dihedral_pairs_anno'][:,1])
            # ``task``：str，当前任务名；用于构造图级掩码并按 batch 索引广播到原子、半边和二面角实体。
            for task in task_list:
                # ``task_batch_dict[task]``：BoolTensor，形状为 (B,)，按 batch.task 字符串逐图比较。
                # ``t``：str，``batch.task`` 中当前图的任务标签；与 ``task`` 比较后形成一个图级布尔值。
                task_batch_dict[task] = torch.tensor([t==task for t in batch['task']], device=device)
                # ``task_node_dict[task]``：BoolTensor，形状为 (N,)，按原子图归属索引广播任务标签。
                task_node_dict[task] = task_batch_dict[task].index_select(0, batch_node)
                # ``task_edge_dict[task]``：BoolTensor，形状为 (H,)，按半边图归属广播任务标签。
                task_edge_dict[task] = task_batch_dict[task].index_select(0, batch_halfedge)
                # ``task_dih_dict[task]``：BoolTensor，形状为 (D,)，按二面角所属图广播任务标签。
                task_dih_dict[task] = task_batch_dict[task].index_select(0, batch_dih)
        else:
            # ``task_list``：不做逐任务拆分时，只有 mixed 项；下列四个映射均传 ``None`` 触发各损失函数的缺省掩码。
            task_list = []
            # ``task_batch_dict``：None，不建立图级命名任务掩码。
            # ``task_node_dict``：None，原子类别与坐标损失回退到覆盖全部原子的 mixed 掩码。
            # ``task_edge_dict``：None，半边类别与距离损失回退到覆盖全部半边的 mixed 掩码。
            # ``task_dih_dict``：None，二面角损失回退到覆盖全部二面角实例的 mixed 掩码。
            task_batch_dict, task_node_dict, task_edge_dict, task_dih_dict = None, None, None, None
        
        # ``loss_list``：list[dict[str, scalar Tensor]]，长度为 5，保留五类损失映射的固定汇总顺序。
        loss_list = [
            # ``loss_list[0]``：逐任务原子类别损失映射，叶为 ``node`` 与 ``fixed_node``。
            self.node_loss(node_type, pred_node, unfixed_node, task_node_dict),
            # ``loss_list[1]``：逐任务坐标损失映射，叶为 ``pos`` 与 ``fixed_pos``。
            self.pos_loss(node_pos, pred_pos, unfixed_pos, task_node_dict),
            # ``loss_list[2]``：逐任务半边类别损失映射，叶为 ``edge`` 与 ``fixed_edge``。
            self.halfedge_loss(halfedge_type, pred_halfedge, unfixed_halfedge, task_edge_dict, size_node),
            # ``loss_list[3]``：逐任务端点距离损失映射，叶为 ``dist`` 与 ``fixed_dist``。
            self.dist_loss(gt_dist, pred_dist, unfixed_dist, batch, task_edge_dict),
            # ``loss_list[4]``：逐任务二面角损失映射，叶为 ``dih``。
            self.dihedral_loss(node_pos, pred_pos, batch, task_dih_dict),
        ]
        # ``d``：dict[str, scalar Tensor]，``loss_list`` 中当前待展平的单类损失映射。
        # ``k``：str，当前损失叶的 ``<scope>/<loss_type>`` 完整键。
        # ``v``：标量 Tensor，当前完整键对应的未汇总损失值。
        # ``loss_dict``：展平后的逐任务损失映射；此时尚不含 ``<task>/total``。
        loss_dict = {k:v for d in loss_list for k,v in d.items()}

        # ``task``：str，依次汇总混合批次与各命名任务；同一任务只加总以前缀 ``<task>/`` 开头的叶损失。
        for task in ['mixed'] + task_list:
            # ``loss_type``：str，当前 scope 下参与加权的 ``<task>/<loss_type>`` 完整键。
            # ``l``：标量 Tensor，``loss_type`` 对应的未汇总损失值。
            # ``loss_dict[<task>/total]``：标量 Tensor，按叶键配置权重加总的当前任务训练/日志总损失。
            loss_dict[f'{task}/total'] = sum([self.weights[loss_type.replace(task+'/', '')] * l for
                                              loss_type, l in loss_dict.items() if loss_type.startswith(task+'/')])

        if self.cfd_loss is not None:
            # ``loss_cfd``：dict[str, scalar Tensor]，目标由 detached 主预测构造的置信度损失映射。
            # ``loss_cfd.cfd_total``：三个置信度分量按配置权重加总的标量。
            # ``loss_cfd.cfd_node``：原子类别正确性置信度 BCE 标量。
            # ``loss_cfd.cfd_pos``：逐原子坐标置信度连续目标 MSE 标量。
            # ``loss_cfd.cfd_edge``：半边类别正确性置信度 BCE 标量。
            loss_cfd = self.cfd_loss(batch, outputs)
            # 只有 ``cfd_total`` 参与反向项；逐任务 total 不重复加入该辅助监督。
            loss_dict['mixed/total'] += loss_cfd['cfd_total']
            # ``k``：str，当前置信度损失叶名，取 ``cfd_total``、``cfd_node``、``cfd_pos`` 或 ``cfd_edge``。
            # ``v``：标量 Tensor，当前置信度损失叶值。
            loss_dict.update({'mixed/'+k:v for k, v in loss_cfd.items()})

        if self.pocket_dist_config is not None:
            # ``loss_pocket_dist``：配体—口袋邻域距离的加权 MSE 标量，单位经权重缩放。
            loss_pocket_dist = self.pocket_dist_loss(batch, outputs)
            # 口袋距离辅助项同样只加到整批 ``mixed/total``。
            loss_dict['mixed/total'] += loss_pocket_dist
            loss_dict.update({'mixed/p_dist': loss_pocket_dist})
        
        return loss_dict
    
    def pocket_dist_loss(self, batch, outputs):
        """监督预测配体到固定口袋邻域原子的距离。

        输入参数:
            - batch: PyG Batch，提供干净配体/口袋坐标和各自图归属。
            - batch.pocket_pos: FloatTensor，形状为 (P, 3)，固定口袋原子局部坐标，单位 Å。
            - batch.node_pos: FloatTensor，形状为 (N, 3)，干净配体坐标，单位 Å。
            - batch.pocket_pos_batch: LongTensor，形状为 (P,)，口袋原子到图的归属索引。
            - batch.node_type_batch: LongTensor，形状为 (N,)，配体原子到图的归属索引。
            - outputs.pred_pos: FloatTensor，形状为 (N, 3)，预测配体坐标，单位 Å。

        返回值:
            - loss_pocket_dist: 标量 Tensor，固定邻接上的距离 MSE 乘 ``pocket_dist.weight``，未加权值单位为 Å²。

        注意:
            - ``torch_geometric.radius(x=pocket, y=ligand)`` 返回第一行配体索引 y、第二行口袋索引 x。
            - 邻接由干净配体坐标在配置半径内建立，预测分支复用同一原子对，因此预测坐标移动不会改变监督集合。
        """

        # ``gt_pocket_pos``：FloatTensor，形状为 (P, 3)，固定口袋原子局部坐标，单位 Å。
        gt_pocket_pos = batch['pocket_pos']
        # ``gt_node_pos``：FloatTensor，形状为 (N, 3)，干净配体坐标，单位 Å。
        gt_node_pos = batch['node_pos']
        # ``knn_node_pocket``：LongTensor，形状为 (2, E_lp)；第一行配体索引，第二行口袋索引。
        knn_node_pocket = radius(x=gt_pocket_pos, y=gt_node_pos, r=self.pocket_dist_config['radius'],
                                 batch_x=batch['pocket_pos_batch'],
                                 batch_y=batch['node_type_batch'])
        # ``gt_dist``：FloatTensor，形状为 (E_lp,)，监督邻接中配体—口袋原子距离，单位 Å。
        gt_dist = torch.norm(gt_node_pos[knn_node_pocket[0]] - gt_pocket_pos[knn_node_pocket[1]], dim=-1)
        
        # ``pred_node_pos``：FloatTensor，形状为 (N, 3)，与干净配体原子顺序对齐的预测坐标。
        pred_node_pos = outputs['pred_pos']
        # ``pred_dist``：FloatTensor，形状为 (E_lp,)，在固定邻接上的预测配体—口袋距离，单位 Å。
        pred_dist = torch.norm(pred_node_pos[knn_node_pocket[0]] - gt_pocket_pos[knn_node_pocket[1]], dim=-1)
        
        # ``loss_pocket_dist``：邻域距离平均平方误差，初始单位 Å²。
        loss_pocket_dist = F.mse_loss(gt_dist, pred_dist)
        # 配置权重直接乘到该标量，再加入 ``mixed/total``。
        loss_pocket_dist *= self.pocket_dist_config['weight']
        return loss_pocket_dist

@register_loss('all_tasks')
class AllTasksLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
        
    def pos_loss(self, node_pos, pred_pos, unfixed_pos):
        loss_pos_total = F.mse_loss(pred_pos, node_pos, reduction='none')
        if unfixed_pos.sum() == 0:
            loss_pos = torch.tensor(0.0, device=node_pos.device)
        else:
            loss_pos = torch.mean(loss_pos_total[unfixed_pos])
        if (~unfixed_pos).sum() == 0:
            loss_pos_fixed = torch.tensor(0.0, device=node_pos.device)
        else:
            loss_pos_fixed = torch.mean(loss_pos_total[~unfixed_pos])
        return loss_pos, loss_pos_fixed
    
    def node_loss(self, node_type, pred_node, unfixed_node):
        loss_node_total = self.ce_loss(pred_node, node_type)
        if unfixed_node.sum() == 0:
            loss_node = torch.tensor(0.0, device=node_type.device)
        else:
            loss_node = torch.mean(loss_node_total[unfixed_node])
        if (~unfixed_node).sum() == 0:
            loss_node_fixed = torch.tensor(0.0, device=node_type.device)
        else:
            loss_node_fixed = torch.mean(loss_node_total[~unfixed_node])
        return loss_node, loss_node_fixed
        
    def halfedge_loss(self, halfedge_type, pred_halfedge, unfixed_halfedge):
        loss_edge_total = self.ce_loss(pred_halfedge, halfedge_type)
        if unfixed_halfedge.sum() == 0:
            loss_edge = torch.tensor(0.0, device=halfedge_type.device)
        else:
            loss_edge = torch.mean(loss_edge_total[unfixed_halfedge])
        if (~unfixed_halfedge).sum() == 0:
            loss_edge_fixed = torch.tensor(0.0, device=halfedge_type.device)
        else:
            loss_edge_fixed = torch.mean(loss_edge_total[~unfixed_halfedge])
        return loss_edge, loss_edge_fixed

    def dist_loss(self, gt_dist, pred_dist, unfixed_dist, batch):
        # dist loss only edge in the same domain
        halfedge_index = batch['halfedge_index']
        domain_node_index = batch['domain_node_index']
        domain_index_of_node = - torch.ones_like(batch['node_type'])
        domain_index_of_node[domain_node_index[1]] = domain_node_index[0]
        inner_domain = ((domain_index_of_node[halfedge_index[0]] >= 0) & 
            (domain_index_of_node[halfedge_index[0]] == domain_index_of_node[halfedge_index[1]]) & 
            unfixed_dist)
        if inner_domain.sum() == 0:
            loss_dist = torch.tensor(0.0, device=gt_dist.device)
        else:
            loss_dist = F.mse_loss(pred_dist[inner_domain], gt_dist[inner_domain], reduction='mean')
        # fixed_dist loss:
        if (~unfixed_dist).sum() == 0:
            loss_dist_fixed = torch.tensor(0.0, device=gt_dist.device)
        else:
            loss_dist_fixed = F.mse_loss(pred_dist[~unfixed_dist], gt_dist[~unfixed_dist], reduction='mean')
        return loss_dist, loss_dist_fixed

    def dihedral_loss(self, node_pos, pred_pos, batch):
        pred_sin, pred_cos = get_dihedral_batch(pred_pos, batch['tor_bonds_anno'], batch['dihedral_pairs_anno'])
        gt_sin, gt_cos = get_dihedral_batch(node_pos, batch['tor_bonds_anno'], batch['dihedral_pairs_anno'])

        cos_delta = pred_cos * gt_cos + pred_sin * gt_sin
        # sin_delta = sin_pred * cos_gt - cos_pred * sin_gt

        if len(cos_delta) == 0:
            loss_dih = torch.tensor(0.0, device=node_pos.device)
        else:
            loss_dih = torch.mean(1 - cos_delta)
        return loss_dih
    
    def forward(self, batch, outputs):
        
        
        # # preparation
        # gt
        node_pos = batch['node_pos']
        node_type = batch['node_type']
        halfedge_type = batch['halfedge_type']
        halfedge_index = batch['halfedge_index']
        gt_dist = torch.norm(node_pos[halfedge_index[0]] - node_pos[halfedge_index[1]], dim=-1)

        # prediction
        pred_pos = outputs['pred_pos']
        pred_node = outputs['pred_node']
        pred_halfedge = outputs['pred_halfedge']
        pred_dist = torch.norm(pred_pos[halfedge_index[0]] - pred_pos[halfedge_index[1]], dim=-1)
        
        # divide into fixed and unfixed
        unfixed_node = (batch['fixed_node'] == 0)
        unfixed_pos = (batch['fixed_pos'] == 0)
        unfixed_halfedge = (batch['fixed_halfedge'] == 0)
        unfixed_dist = (batch['fixed_halfdist'] == 0)
        
        # # basic loss
        loss_node, loss_node_fixed = self.node_loss(node_type, pred_node, unfixed_node)
        loss_pos, loss_pos_fixed = self.pos_loss(node_pos, pred_pos, unfixed_pos)
        loss_halfedge, loss_halfedge_fixed = self.halfedge_loss(halfedge_type, pred_halfedge, unfixed_halfedge)
        loss_dist, loss_dist_fixed = self.dist_loss(gt_dist, pred_dist, unfixed_dist, batch)
        
        loss_dih = self.dihedral_loss(node_pos, pred_pos, batch)

        # # total
        loss_unfixed = (
            loss_node * self.config['weights']['node'] + \
            loss_pos * self.config['weights']['pos'] + \
            loss_halfedge * self.config['weights']['edge'] + \
            loss_dist * self.config['weights']['dist'] + \
            loss_dih * self.config['weights']['dihedral']
        )
        loss_fixed = (
            loss_node_fixed * self.config['fixed_weights']['node'] + \
            loss_pos_fixed * self.config['fixed_weights']['pos'] + \
            loss_halfedge_fixed * self.config['fixed_weights']['edge'] + \
            loss_dist_fixed * self.config['fixed_weights']['dist']
        )
        loss = loss_unfixed + loss_fixed

        
        loss_dict = {
            'loss': loss,
            'loss_node': loss_node,
            'loss_pos': loss_pos,
            'loss_edge': loss_halfedge,
            'loss_dist': loss_dist,
            'loss_dih': loss_dih,
            'loss_fixed_total': loss_fixed,

            'loss_fixed/node': loss_node_fixed,
            'loss_fixed/pos': loss_pos_fixed,
            'loss_fixed/edge': loss_halfedge_fixed,
            'loss_fixed/dist': loss_dist_fixed,
            
        }
        return loss_dict



@register_loss('refine_torsional2')
class RefineTorsionalLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, batch, outputs):
        pred_pos = outputs['pred_pos']
        pred_node = outputs['pred_node']
        pred_halfedge = outputs['pred_halfedge']
        
        node_pos = batch['node_pos']
        node_type = batch['node_type']
        halfedge_type = batch['halfedge_type']
        
        # divide into fixed and unfixed
        unfixed_node = batch['fixed_node'] == 0
        unfixed_pos = batch['fixed_pos'] == 0
        unfixed_halfedge = batch['fixed_halfedge'] == 0
        
        
        # # for fixed dist loss
        nodes_0, nodes_1 = batch['halfedge_index']
        unfixed_dist = (batch['fixed_halfdist'] == 0)
        dist_pred = torch.norm(pred_pos[nodes_0] - pred_pos[nodes_1], dim=-1)
        dist_true = torch.norm(node_pos[nodes_0] - node_pos[nodes_1], dim=-1)
        loss_dist_total = F.mse_loss(dist_pred, dist_true, reduction='none')
        loss_dist = torch.mean(loss_dist_total[unfixed_dist]).nan_to_num()
        loss_dist_fixed = torch.mean(loss_dist_total[~unfixed_dist]).nan_to_num()
        
        # # for dihedral loss
        if 'dih_sin' in outputs:
            sin_gt, cos_gt = get_dihedral_batch(node_pos, batch['tor_bonds_anno'], batch['dihedral_pairs_anno'])
            sin_pred, cos_pred = outputs['dih_sin'], outputs['dih_cos']
            cos_delta = cos_pred * cos_gt + sin_pred * sin_gt
            # sin_delta = sin_pred * cos_gt - cos_pred * sin_gt
            loss_dih = torch.mean(1 - cos_delta).nan_to_num()
        else:
            loss_dih = torch.tensor(0.0, device=node_pos.device)
        
        
        # # pos
        loss_pos_total = F.mse_loss(pred_pos, node_pos, reduction='none')
        loss_pos = torch.mean(loss_pos_total[unfixed_pos]).nan_to_num()
        loss_pos_fixed = torch.mean(loss_pos_total[~unfixed_pos]).nan_to_num()

        # # node & edge type
        loss_node_total = self.ce_loss(pred_node, node_type)
        loss_node = torch.mean(loss_node_total[unfixed_node]).nan_to_num()
        loss_node_fixed = torch.mean(loss_node_total[~unfixed_node]).nan_to_num()
        
        loss_edge_total = self.ce_loss(pred_halfedge, halfedge_type)
        loss_edge = torch.mean(loss_edge_total[unfixed_halfedge]).nan_to_num()
        loss_edge_fixed = torch.mean(loss_edge_total[~unfixed_halfedge]).nan_to_num()
        
        # # total
        loss_fixed = (loss_pos_fixed * self.config['weights']['pos'] +\
                     loss_node_fixed * self.config['weights']['node'] +\
                     loss_edge_fixed * self.config['weights']['edge'])
        loss_total = loss_node * self.config['weights']['node'] + \
                     loss_edge * self.config['weights']['edge'] + \
                     loss_fixed * self.config['fixed_weight'] + \
                         loss_dist_fixed * self.config['weights']['fixed_dist']  # dist_fixed is default added
                     
        if 'dist' in self.config['use_pos_losses']:
            loss_total += loss_dist * self.config['weights']['dist']  # unfixed dist loss
        if 'pos' in self.config['use_pos_losses']:
            loss_total += loss_pos * self.config['weights']['pos']
        if 'dihedral' in self.config['use_pos_losses']:
            if 'dih_sin' not in outputs:
                raise ValueError('dihedral loss is not calculated!')
            loss_total += loss_dih * self.config['weights']['dihedral']
        
        loss_dict = {
            'loss': loss_total,
            # 'loss_node': loss_node,
            # 'loss_edge': loss_edge,
            'loss_fixed': loss_fixed,
            'loss_dist_fixed': loss_dist_fixed,

            'loss_dist': loss_dist,
            'loss_pos': loss_pos,
            'loss_dih': loss_dih,
        }
        return loss_dict


@register_loss('refine_torsional')
class RefineTorsionalLoss(nn.Module):
    def __init__(self, config):
        raise NotImplementedError('use refine_torsional2')
        super().__init__()
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, batch, outputs):
        pred_pos = outputs['pred_pos']
        pred_node = outputs['pred_node']
        pred_halfedge = outputs['pred_halfedge']
        pred_pos_corr = outputs['pred_pos_corr']
        
        node_pos = batch['node_pos']
        node_type = batch['node_type']
        halfedge_type = batch['halfedge_type']
        
        # divide into fixed and unfixed
        unfixed_node = batch['fixed_node'] == 0
        unfixed_pos = batch['fixed_pos'] == 0
        unfixed_halfedge = batch['fixed_halfedge'] == 0
        
        
        # # for fixed dist loss
        fixed_halfdist = batch['fixed_halfdist']
        halfedge_index = batch['halfedge_index']
        halfedge_index_fixed_dist = halfedge_index[:, fixed_halfdist==1]
        nodes_0, nodes_1 = halfedge_index_fixed_dist
        dist_pred = torch.norm(pred_pos[nodes_0] - pred_pos[nodes_1], dim=-1)
        dist_true = torch.norm(node_pos[nodes_0] - node_pos[nodes_1], dim=-1)
        loss_dist = F.mse_loss(dist_pred, dist_true)
        
        # # for dihedral loss
        index_tor = batch['dihedral_pairs_anno'][:, 0]
        dihedral_ends = batch['dihedral_pairs_anno'][:, 1:]  # (n_dih, 2)
        dihedral_tor_nodes = batch['tor_bonds_anno'][:, 1:][index_tor]  # (n_dih, 2)
        sin_gt, cos_gt = get_dihedral(node_pos[dihedral_ends[:, 0]], node_pos[dihedral_tor_nodes[:, 0]],
                        node_pos[dihedral_tor_nodes[:, 1]], node_pos[dihedral_ends[:, 1]])
        sin_pred, cos_pred = outputs['dih_sin'], outputs['dih_cos']
        cos_delta = cos_pred * cos_gt + sin_pred * sin_gt
        sin_delta = sin_pred * cos_gt - cos_pred * sin_gt
        loss_dih = torch.mean(1 - cos_delta).nan_to_num() + \
                    0.5 * F.mse_loss(sin_delta, torch.zeros_like(sin_delta))
        
        # # pos
        loss_pos_total = F.mse_loss(pred_pos, node_pos, reduction='none')
        loss_pos = torch.mean(loss_pos_total[unfixed_pos]).nan_to_num()
        loss_pos_fixed = torch.mean(loss_pos_total[~unfixed_pos]).nan_to_num()
        
        # # pos corr
        loss_pos_corr_total = F.mse_loss(pred_pos_corr, node_pos, reduction='none')
        loss_pos_corr = torch.mean(loss_pos_corr_total[unfixed_pos]).nan_to_num()

        # # node & edge type
        loss_node_total = self.ce_loss(pred_node, node_type)
        loss_node = torch.mean(loss_node_total[unfixed_node]).nan_to_num()
        loss_node_fixed = torch.mean(loss_node_total[~unfixed_node]).nan_to_num()
        
        loss_edge_total = self.ce_loss(pred_halfedge, halfedge_type)
        loss_edge = torch.mean(loss_edge_total[unfixed_halfedge]).nan_to_num()
        loss_edge_fixed = torch.mean(loss_edge_total[~unfixed_halfedge]).nan_to_num()
        
        # # total
        loss_fixed = (loss_pos_fixed * self.config['weights']['pos'] +\
                     loss_node_fixed * self.config['weights']['node'] +\
                     loss_edge_fixed * self.config['weights']['edge'])
        loss_total = loss_node * self.config['weights']['node'] + \
                     loss_edge * self.config['weights']['edge'] + \
                     loss_fixed * self.config['fixed_weight']
                     
        if 'fixed_dist' in self.config['use_pos_losses']:
            loss_total += loss_dist * self.config['weights']['dist']
        if 'pos' in self.config['use_pos_losses']:
            loss_total += loss_pos * self.config['weights']['pos']
        if 'pos_corr' in self.config['use_pos_losses']:
            loss_total += loss_pos_corr * self.config['weights']['pos_corr']
        if 'dihedral' in self.config['use_pos_losses']:
            loss_total += loss_dih * self.config['weights']['dihedral']
        
        loss_dict = {
            'loss': loss_total,
            # 'loss_node': loss_node,
            # 'loss_edge': loss_edge,
            'loss_fixed': loss_fixed,

            'loss_pos': loss_pos,
            'loss_dist': loss_dist,
            'loss_pos_corr': loss_pos_corr,
            'loss_dih': loss_dih,
        }
        return loss_dict
    

@register_loss('refine')
class RefineLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.node_transition = deepcopy(node_transition)
        # self.edge_transition = deepcopy(edge_transition)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, batch, outputs):
        pred_pos = outputs['pred_pos']
        pred_node = outputs['pred_node']
        pred_halfedge = outputs['pred_halfedge']
        
        node_pos = batch['node_pos']
        node_type = batch['node_type']
        halfedge_type = batch['halfedge_type']
        # log_node_0 = batch['log_node_0']
        # log_halfedge_0 = batch['log_halfedge_0']

        # log_node_t = batch['log_node_t']
        # log_halfedge_t = batch['log_halfedge_t']
        # time_node = batch['time_node']
        # time_halfedge = batch['time_halfedge']
        
        # divide into fixed and unfixed
        unfixed_node = batch['fixed_node'] == 0
        unfixed_pos = batch['fixed_pos'] == 0
        unfixed_halfedge = batch['fixed_halfedge'] == 0
        
        # # pos
        loss_pos_total = F.mse_loss(pred_pos, node_pos, reduction='none')
        loss_pos = torch.mean(loss_pos_total[unfixed_pos]).nan_to_num()
        loss_pos_fixed = torch.mean(loss_pos_total[~unfixed_pos]).nan_to_num()

        # # node & edge type
        loss_node_total = self.ce_loss(pred_node, node_type)
        loss_node = torch.mean(loss_node_total[unfixed_node]).nan_to_num()
        loss_node_fixed = torch.mean(loss_node_total[~unfixed_node]).nan_to_num()
        
        loss_edge_total = self.ce_loss(pred_halfedge, halfedge_type)
        loss_edge = torch.mean(loss_edge_total[unfixed_halfedge]).nan_to_num()
        loss_edge_fixed = torch.mean(loss_edge_total[~unfixed_halfedge]).nan_to_num()
        
        # # total
        loss_fixed = (loss_pos_fixed * self.config['weights']['pos'] +\
                     loss_node_fixed * self.config['weights']['node'] +\
                     loss_edge_fixed * self.config['weights']['edge'])
        loss_total = loss_pos * self.config['weights']['pos'] + \
                     loss_node * self.config['weights']['node'] + \
                     loss_edge * self.config['weights']['edge'] + \
                     loss_fixed * self.config['fixed_weight']
        
        loss_dict = {
            'loss': loss_total,
            'loss_pos': loss_pos,
            'loss_node': loss_node,
            'loss_edge': loss_edge,
            'loss_fixed': loss_fixed,
        }
        return loss_dict
    

@register_loss('confidence')
class ConfidenceLoss(nn.Module):
    """用 detached 主预测构造变量级置信度目标。

    配置字段:
        - config.prob_1A: float；位于 ``(0, 1)`` 时坐标目标为 ``prob_1A ** error_Å``，其他取值切换到直接预测负坐标误差的历史分支。
        - config.weights.node: float，原子 confidence BCE 的汇总权重。
        - config.weights.pos: float，坐标 confidence MSE 的汇总权重。
        - config.weights.edge: float，半边 confidence BCE 的汇总权重。

    输入字段:
        - batch.node_type: LongTensor，形状为 (N,)，干净原子类别标签。
        - batch.node_pos: FloatTensor，形状为 (N, 3)，干净坐标标签，单位 Å。
        - batch.halfedge_type: LongTensor，形状为 (H,)，干净半边类别标签。
        - outputs.pred_node: FloatTensor，形状为 (N, C_n)，主原子类别 logits。
        - outputs.pred_pos: FloatTensor，形状为 (N, 3)，主坐标预测，单位 Å。
        - outputs.pred_halfedge: FloatTensor，形状为 (H, C_e)，主半边类别 logits。
        - outputs.confidence_node: FloatTensor，形状为 (N, 1)，原子类别正确性的 logit。
        - outputs.confidence_pos: FloatTensor，形状为 (N, 1)，坐标 confidence 原始标量。
        - outputs.confidence_halfedge: FloatTensor，形状为 (H, 1)，半边类别正确性的 logit。

    返回字段:
        - loss_dict.cfd_total: 标量 Tensor，三个 confidence 分量按配置权重加总的辅助损失。
        - loss_dict.cfd_node: 标量 Tensor，原子类别正确性 BCE。
        - loss_dict.cfd_pos: 标量 Tensor，sigmoid confidence 与指数目标之间的 MSE，或历史负误差回归 MSE。
        - loss_dict.cfd_edge: 标量 Tensor，半边类别正确性 BCE。

    注意:
        - ``prob_1A`` 分支中误差 0 Å 对应目标 1，误差 1 Å 对应 ``prob_1A``，它不是 1 Å 阈值分类。
        - 主预测在构造标签时 ``detach``，因此该辅助项只训练三个 confidence head，不经标签路径反向修改 ``pred_*``。
    """

    def __init__(self, config):
        """保存置信度配置，并把连续坐标目标的 1 Å 概率转换为对数 buffer。

        输入参数:
            - config.prob_1A: float，位于 ``(0, 1)`` 时定义误差 1 Å 的连续坐标 confidence 目标。
            - config.weights.node: float，原子 confidence BCE 汇总权重。
            - config.weights.pos: float，坐标 confidence MSE 汇总权重。
            - config.weights.edge: float，半边 confidence BCE 汇总权重。
        """
        super().__init__()
        # ``config.prob_1A``：float，误差 1 Å 对应的连续坐标 confidence 目标。
        # ``config.weights.node``：float，原子 confidence BCE 权重。
        # ``config.weights.pos``：float，坐标 confidence MSE 权重。
        # ``config.weights.edge``：float，半边 confidence BCE 权重。
        # ``self.config``：Mapping，保留上述 confidence 目标与分量权重叶。
        self.config = config
        # ``prob_1A``：float；定义 1 Å 误差对应的连续置信度目标。
        prob_1A = config['prob_1A']
        if prob_1A > 0 and prob_1A < 1:
            # ``log_prob_1A``：持久 buffer 标量 <0；使 ``exp(error_Å*log_prob)`` 可向量化。
            self.register_buffer('log_prob_1A', torch.tensor(np.log(prob_1A)))
        else:  # directly predict minus error
            # 正标量 1 是历史分支哨兵；此时坐标 head 拟合 ``-error_Å``。
            self.register_buffer('log_prob_1A', torch.tensor(1.0))

        # ``self.binary_loss``：类别置信度使用未预先 sigmoid 的 logits，内部完成数值稳定的 BCE。
        self.binary_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, batch, outputs):
        """计算原子、坐标与半边的变量级置信度损失。

        输入参数:
            - batch.node_type: LongTensor，形状为 (N,)，干净原子类别标签。
            - batch.node_pos: FloatTensor，形状为 (N, 3)，干净配体局部坐标标签，单位 Å。
            - batch.halfedge_type: LongTensor，形状为 (H,)，干净半边类别标签。
            - outputs.pred_node: FloatTensor，形状为 (N, C_n)，主原子类别 logits。
            - outputs.pred_pos: FloatTensor，形状为 (N, 3)，主坐标预测，单位 Å。
            - outputs.pred_halfedge: FloatTensor，形状为 (H, C_e)，主半边类别 logits。
            - outputs.confidence_node: FloatTensor，形状为 (N, 1)，原子类别正确性 logit。
            - outputs.confidence_pos: FloatTensor，形状为 (N, 1)，坐标 confidence 原始输出。
            - outputs.confidence_halfedge: FloatTensor，形状为 (H, 1)，半边类别正确性 logit。

        返回字段:
            - loss_dict.cfd_total: 标量 Tensor，三个 confidence 分量按配置权重加总的辅助损失。
            - loss_dict.cfd_node: 标量 Tensor，原子类别正确性 BCE。
            - loss_dict.cfd_pos: 标量 Tensor，坐标 confidence 连续目标 MSE 或历史负误差回归 MSE。
            - loss_dict.cfd_edge: 标量 Tensor，半边类别正确性 BCE。

        注意:
            - ``batch`` 标签与 ``outputs`` 主预测必须保持相同的原子/半边顺序。
        """

        # ``pred_node``：FloatTensor，形状为 (N, C_n)，主原子类别 logits。
        pred_node = outputs['pred_node']
        # ``pred_pos``：FloatTensor，形状为 (N, 3)，主坐标预测，单位 Å。
        pred_pos = outputs['pred_pos']
        # ``pred_halfedge``：FloatTensor，形状为 (H, C_e)，主半边类别 logits。
        pred_halfedge = outputs['pred_halfedge']

        # ``node_type``：LongTensor，形状为 (N,)，干净原子类别标签。
        node_type = batch['node_type']
        # ``node_pos``：FloatTensor，形状为 (N, 3)，干净坐标标签，单位 Å。
        node_pos = batch['node_pos']
        # ``halfedge_type``：LongTensor，形状为 (H,)，干净半边类别标签。
        halfedge_type = batch['halfedge_type']
        
        # ``right_node``：FloatTensor，形状为 (N,) 的 0/1 标签；argmax 与标签比较后切断主分支梯度。
        right_node = (node_type == pred_node.argmax(-1).detach()).to(pred_node.dtype)
        # ``right_pos``：FloatTensor，形状为 (N,)；初始为每个原子的欧氏坐标误差，单位 Å，并 detach。
        right_pos = torch.norm(node_pos - pred_pos, dim=-1).detach()
        if self.log_prob_1A < 0:  # output as logits of cfd prob
            # 变换后 ``right_pos`` 位于 (0,1]；误差 1 Å 时恰为 ``prob_1A``。
            right_pos = torch.exp(right_pos * self.log_prob_1A)
        # ``right_halfedge``：FloatTensor，形状为 (H,) 的 0/1 标签，表示主半边 argmax 是否正确。
        right_halfedge = (halfedge_type == pred_halfedge.argmax(-1).detach()).to(pred_node.dtype)
        
        # ``confidence_node``：FloatTensor，形状为 (N, 1)，原子正确性的未 sigmoid logit。
        confidence_node = outputs['confidence_node']
        # ``confidence_pos``：FloatTensor，形状为 (N, 1)，坐标置信度/负误差的原始预测。
        confidence_pos = outputs['confidence_pos']
        # ``confidence_halfedge``：FloatTensor，形状为 (H, 1)，半边正确性的未 sigmoid logit。
        confidence_halfedge = outputs['confidence_halfedge']
        
        # ``loss_node``：标量 BCE；``[..., 0]`` 去掉单通道维并与形状为 (N,) 的 ``right_node`` 对齐。
        loss_node = self.binary_loss(confidence_node[..., 0], right_node)
        # loss_pos = F.mse_loss(confidence_pos[..., 0], right_pos)  # need sigmoid ??
        if self.log_prob_1A < 0: # output as logits of cfd prob
            # ``loss_pos``：sigmoid 后连续置信度与指数目标之间的标量 MSE。
            loss_pos = F.mse_loss(torch.sigmoid(confidence_pos[..., 0]), right_pos)  # need sigmoid ??
        else: # directly predict minus error
            # ``loss_pos``：历史分支中原始单通道输出与负 Å 误差之间的标量 MSE。
            loss_pos = F.mse_loss(confidence_pos[..., 0], -right_pos)
        # ``loss_halfedge``：标量 BCE；沿 H 条无向半边平均。
        loss_halfedge = self.binary_loss(confidence_halfedge[..., 0], right_halfedge)

        # ``loss_total``：三个分量按配置权重相加的标量，供 mixed 总损失反向传播。
        loss_total = loss_node * self.config['weights']['node'] + \
                     loss_pos * self.config['weights']['pos'] + \
                     loss_halfedge * self.config['weights']['edge']
                     
        # ``loss_dict``：置信度总项与三个未加权分量的扁平映射。
        loss_dict = {
            'cfd_total': loss_total,
            'cfd_node': loss_node,
            'cfd_pos': loss_pos,
            'cfd_edge': loss_halfedge,
        }
        return loss_dict
