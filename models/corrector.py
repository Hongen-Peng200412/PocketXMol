"""把网络坐标投影到固定距离、扭转与刚体运动约束上。

构象生成和 docking 的 ``free`` 采样不会使用几何投影；``flexible``/``rigid``
docking 由 ``DockSamplNoiser.outputs2batch`` 调用本模块，使下一步坐标只包含允许的
域内扭转与整域刚体位姿变化。记号：``N`` 为批内原子数，``H`` 为半边数，``K``
为域注释中的原子实例数，``G`` 为刚体域数，``T`` 为旋转键数，``D`` 为二面角
实例数。所有位置/距离均以 Å 表示，角度通过无量纲 sin/cos 表示。
"""

import torch
import torch.nn as nn
import torch.linalg as la
from torch.nn import functional as F
from torch_scatter import scatter_mean, scatter_sum

try:
    from utils.motion import apply_torsional_rotation_multiple_domains
except:
    pass


def grad_len_to_pos(pos, edge_index, config):
    """把超出允许区间的边长偏差转换为逐原子平均位置梯度。

    输入参数:
        - pos: FloatTensor，形状为 (N, 3)，当前原子坐标，单位 Å。
        - edge_index: LongTensor，形状为 (2, E)，有向边端点；第 0 行是接收归约梯度的左端原子索引。
        - config: Mapping，距离区间势能配置。
        - config.min: float，允许距离下界，单位 Å。
        - config.max: float，允许距离上界，单位 Å。
        - config.std: float，二次势能的距离尺度，单位 Å。

    返回值:
        - grad_pos: FloatTensor，形状为 (N, 3)，按左端原子平均关联有向边梯度后的坐标梯度。

    注意:
        - 当前构象/docking 默认链路未直接调用该函数；仓库其他 noiser 的弹簧约束路径会调用。
    """

    # ``pos_left``：FloatTensor，形状为 (E, 3) Å，与每条有向边第 0 行端点对齐。
    pos_left = pos[edge_index[0]]
    # ``pos_right``：FloatTensor，形状为 (E, 3) Å，与第 1 行端点对齐。
    pos_right = pos[edge_index[1]]
    # ``value_min``：float，允许距离区间的下界，单位 Å。
    # ``value_max``：float，允许距离区间的上界，单位 Å。
    value_min, value_max = config['min'], config['max']
    # ``value_std``：距离势能尺度，单位 Å。
    value_std = config['std']
    # ``lens``：FloatTensor，形状为 (E, 1) Å；保留单通道维以广播到 xyz。
    lens = torch.linalg.norm(pos_left - pos_right, dim=-1).unsqueeze(-1)
    # ``grad_min``：FloatTensor，形状为 (E, 3)，低于下界时沿边方向的梯度。
    grad_min = 2 * (lens - value_min) / value_std**2 / lens * (pos_left - pos_right)
    # ``grad_max``：FloatTensor，形状为 (E, 3)，高于上界时沿边方向的梯度。
    grad_max = 2 * (lens - value_max) / value_std**2 / lens * (pos_left - pos_right)
    # ``grad``：FloatTensor，形状为 (E, 3)；区间内边取 0，区间外选择相应一侧梯度。
    grad = torch.where(lens < value_min, grad_min, 
                       torch.where(lens > value_max, grad_max, 0))
    # ``grad_pos``：FloatTensor，形状为 (N, 3)；按左端原子归约其所有出边梯度的平均值。
    grad_pos = scatter_mean(grad, edge_index[0], dim=0, dim_size=pos.shape[0])
    return grad_pos
    
    
def correct_pos_by_fixed_dist_batch(batch, outputs, use_pos='in', config=None):
    """迭代修正预测坐标，使 fixed 半边距离接近参照距离。

    输入参数:
        - batch: PyG Batch，提供参照坐标、半边端点和 fixed prompt。
        - batch.pos_in: FloatTensor，形状为 (N, 3)，``use_pos=in`` 时的带噪参照坐标，单位 Å。
        - batch.gt_node_pos: FloatTensor，形状为 (N, 3)，``use_pos=gt`` 时的干净参照坐标，单位 Å。
        - batch.fixed_halfdist: LongTensor|BoolTensor，形状为 (H,)，``use_pos=in`` 时选择需保持参照长度的半边。
        - batch.fixed_halfdist_flex: LongTensor|BoolTensor，形状为 (H,)，``use_pos=gt`` 时选择需保持干净长度的半边。
        - batch.halfedge_index: LongTensor，形状为 (2, H)，每列是一条无向完整图半边的批内原子端点。
        - batch.fixed_pos: LongTensor|BoolTensor，形状为 (N,)，为真表示条件锚点原子不可更新。
        - outputs.pred_pos: FloatTensor，形状为 (N, 3)，待修正的网络坐标预测，单位 Å。
        - use_pos: Literal[in,gt]，选择带噪或干净参照及其配套距离掩码。
        - config: Mapping|None，可选迭代配置；为空时使用 10 次迭代和 0.5 步长。
        - config.iters: int，可选，固定距离修正迭代次数。
        - config.lr: float，可选，位置梯度更新步长。

    返回值:
        - pred_pos: FloatTensor，形状为 (N, 3)，固定锚点保持不变、其他原子经距离梯度修正后的坐标，单位 Å。

    注意:
        - 函数只原地更新从 ``outputs.pred_pos`` 克隆出的副本，不修改输入 Batch 或 outputs。
        - 任一迭代的距离 MSE 成为 NaN 时，放弃全部修正并返回原始预测副本。
    """

    if use_pos == 'in':
        # ``pos_in``：FloatTensor，形状为 (N, 3) Å，当前带噪输入的独立参照副本。
        pos_in = batch['pos_in'].detach().clone()
        # ``fixed_halfdist``：BoolTensor，形状为 (H,)，输入几何中需保持长度的无向半边。
        fixed_halfdist = batch['fixed_halfdist'].bool()
    elif use_pos == 'gt':
        # ``pos_in``：FloatTensor，形状为 (N, 3) Å，干净坐标的独立参照副本。
        pos_in = batch['gt_node_pos'].detach().clone()
        # ``fixed_halfdist``：BoolTensor，形状为 (H,)，flexible 模式专用固定距离掩码。
        fixed_halfdist = batch['fixed_halfdist_flex'].bool()
    # ``pred_pos``：FloatTensor，形状为 (N, 3) Å；待优化的网络预测副本。
    pred_pos = outputs['pred_pos'].detach().clone()
    
    # get parameter
    if config is None:
        # ``iters``：默认修正迭代数。
        iters = 10
        # ``lr``：无量纲更新步长，乘到位置梯度上。
        lr = 0.5
        # lamb = 0
    else:
        # ``iters``：配置指定的修正迭代数。
        iters = config['iters']
        # ``lr``：配置指定的更新步长。
        lr = config['lr']

    # get bond index
    # ``halfedge_index``：LongTensor，形状为 (2, H)，每条无向完整图半边只出现一次。
    halfedge_index = batch['halfedge_index']
    # ``edge_index``：LongTensor，形状为 (2, 2H)，拼接原方向与翻转方向的有向边。
    edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=-1)
    # ``fixed_dist``：BoolTensor，形状为 (2H,)，与双向边一一对齐的重复掩码。
    fixed_dist = torch.cat([fixed_halfdist, fixed_halfdist], dim=-1)
    # ``fixed_edge_index``：LongTensor，形状为 (2, E_f)，仅含需要保持长度的有向边。
    fixed_edge_index = edge_index[:, fixed_dist]
    
    # fetch relevant pos
    # ``fixed_pos``：BoolTensor，形状为 (N,)；真值原子作为条件锚点，不参与坐标更新。
    fixed_pos = batch['fixed_pos'].bool()
    
    # spring the pos
    with torch.enable_grad():
        # ``dist_in``：FloatTensor，形状为 (E_f,) Å，参照坐标上的固定边长度。
        dist_in = torch.linalg.norm(pos_in[fixed_edge_index[0]] - pos_in[fixed_edge_index[1]], dim=-1)
        # ``i``：int，当前固定距离修正迭代编号；数值不直接进入更新公式。
        for i in range(iters):
            # ``dist_pred``：FloatTensor，形状为 (E_f,) Å，当前预测副本上的固定边长度。
            dist_pred = torch.linalg.norm(pred_pos[fixed_edge_index[0]] - pred_pos[fixed_edge_index[1]], dim=-1)
            # ``grad``：FloatTensor，形状为 (E_f, 3)；相对长度误差沿有向边向量的更新量。
            grad = 2 * ((dist_pred - dist_in) / dist_in).unsqueeze(-1) * (pred_pos[fixed_edge_index[0]] - pred_pos[fixed_edge_index[1]])
            # ``grad_pos``：FloatTensor，形状为 (N, 3)；按有向边左端原子平均所有关联梯度。
            grad_pos = scatter_mean(grad, fixed_edge_index[0], dim=0, dim_size=pos_in.shape[0])
            # grad = torch.autograd.grad(loss, in_pos)[0]
            pred_pos[~fixed_pos] = pred_pos[~fixed_pos] - grad_pos[~fixed_pos] * lr

            # ``loss``：标量 Å²，用于只检测数值是否变成 NaN，不参与 autograd 更新。
            loss = F.mse_loss(torch.linalg.norm(pred_pos[fixed_edge_index[0]] - pred_pos[fixed_edge_index[1]], dim=-1), dist_in)
            if loss.isnan():
                # ``pred_pos``：数值异常时放弃全部迭代，恢复网络原始预测副本。
                pred_pos = outputs['pred_pos'].detach().clone()
                break
    # print(loss)
    return pred_pos


@torch.no_grad()
def correct_pos_batch_no_tor(batch, outputs):
    """仅用每个域的刚体旋转和平移投影网络坐标。

    输入参数:
        - batch: PyG Batch，提供输入内部几何与刚体域成员注释。
        - batch.pos_in: FloatTensor，形状为 (N, 3)，投影前需保留内部几何的输入坐标，单位 Å。
        - batch.domain_node_index: LongTensor，形状为 (2, K)，第一行是域号，第二行是域内批次原子索引。
        - outputs.pred_pos: FloatTensor，形状为 (N, 3)，网络希望达到的目标坐标，单位 Å。

    返回值:
        - pred_pos_corr: FloatTensor，形状为 (N, 3)，域内仅经刚体旋转/平移后的坐标，单位 Å；域外原子保留网络预测。
    """

    # ``pos_in``：FloatTensor，形状为 (N, 3) Å，投影前允许保留内部几何的输入状态。
    pos_in = batch['pos_in']
    # ``pred_pos``：FloatTensor，形状为 (N, 3) Å，网络希望达到的坐标目标。
    pred_pos = outputs['pred_pos']
    
    # ``pred_pos_corr``：FloatTensor，形状为 (N, 3) Å，按域 Kabsch 投影后的坐标。
    pred_pos_corr = correct_pos(
        no_tor=True,
        pos_in=pos_in.clone(), pos_out=pred_pos.clone(),
        sin_in=None, cos_in=None,
        sin_out=None, cos_out=None,
        domain_node_index=batch['domain_node_index'],
        # domain_center_nodes=batch['domain_center_nodes'],
        tor_bonds_anno=None,
        twisted_nodes_anno=None,
        dihedral_pairs_anno=None,
    )
    return pred_pos_corr

@torch.no_grad()
def correct_pos_batch(batch, outputs, use_pos='in'):
    """先恢复域内扭转变化，再拟合每个域的整体刚体位姿。

    输入参数:
        - batch: PyG Batch，提供内部几何、旋转键、随动原子、二面角与刚体域注释。
        - batch.pos_in: FloatTensor，形状为 (N, 3)，``use_pos=in`` 时的内部几何参照，单位 Å。
        - batch.gt_node_pos: FloatTensor，形状为 (N, 3)，``use_pos=gt`` 时的内部几何参照，单位 Å。
        - batch.tor_bonds_anno: LongTensor，形状为 (T, 3)，每行是执行层级与旋转键端点。
        - batch.twisted_nodes_anno: LongTensor，形状为 (M, 2)，每行是旋转键编号与随动原子索引。
        - batch.dihedral_pairs_anno: LongTensor，形状为 (D, 3)，每行是旋转键编号与两个二面角外侧端点。
        - batch.domain_node_index: LongTensor，形状为 (2, K)，第一行是域号，第二行是域内批次原子索引。
        - outputs.pred_pos: FloatTensor，形状为 (N, 3)，网络目标坐标，单位 Å。
        - outputs.dih_sin: FloatTensor，可选，形状为 (D, 1)，模型显式预测的目标二面角正弦。
        - outputs.dih_cos: FloatTensor，可选，形状为 (D, 1)，模型显式预测的目标二面角余弦。
        - use_pos: Literal[in,gt]，选择带噪或干净内部几何参照。

    返回值:
        - pred_pos_corr: FloatTensor，形状为 (N, 3)，经允许扭转与逐域刚体位姿投影后的坐标，单位 Å；域外原子保留网络预测。

    注意:
        - ``outputs`` 不含 ``dih_sin/dih_cos`` 时，从 ``pred_pos`` 直接计算目标二面角。
    """

    if use_pos == 'in':
        # ``pos_in``：FloatTensor，形状为 (N, 3) Å，当前带噪状态的内部几何参照。
        pos_in = batch['pos_in']
    elif use_pos == 'gt':
        # ``pos_in``：FloatTensor，形状为 (N, 3) Å，干净内部几何参照。
        pos_in = batch['gt_node_pos']
    # ``pred_pos``：FloatTensor，形状为 (N, 3) Å，网络的目标坐标。
    pred_pos = outputs['pred_pos']
    
    if 'dih_sin' not in outputs:
        # ``sin_out``：FloatTensor，形状为 (D, 1)，从预测坐标计算的目标二面角正弦值。
        # ``cos_out``：FloatTensor，形状为 (D, 1)，从预测坐标计算的目标二面角余弦值。
        sin_out, cos_out = get_dihedral_batch(pred_pos, batch['tor_bonds_anno'], batch['dihedral_pairs_anno'])
    else:
        # ``sin_out``：FloatTensor，形状为 (D, 1)，模型显式预测的目标二面角正弦值。
        # ``cos_out``：FloatTensor，形状为 (D, 1)，模型显式预测的目标二面角余弦值。
        sin_out, cos_out = outputs['dih_sin'], outputs['dih_cos']
    # ``sin_in``：FloatTensor，形状为 (D, 1)，输入内部几何二面角的正弦值。
    # ``cos_in``：FloatTensor，形状为 (D, 1)，输入内部几何二面角的余弦值。
    sin_in, cos_in = get_dihedral_batch(pos_in, batch['tor_bonds_anno'], batch['dihedral_pairs_anno'])
    # ``pred_pos_corr``：FloatTensor，形状为 (N, 3) Å，扭转与逐域刚体投影后的坐标。
    pred_pos_corr = correct_pos(
        pos_in=pos_in.clone(), pos_out=pred_pos.clone(),
        sin_in=sin_in, cos_in=cos_in,
        sin_out=sin_out,
        cos_out=cos_out,
        domain_node_index=batch['domain_node_index'],
        # domain_center_nodes=batch['domain_center_nodes'],
        tor_bonds_anno=batch['tor_bonds_anno'],
        twisted_nodes_anno=batch['twisted_nodes_anno'],
        dihedral_pairs_anno=batch['dihedral_pairs_anno'],
    )
    return pred_pos_corr


@torch.no_grad()
def correct_pos(pos_in, pos_out, 
                sin_in, cos_in,
                sin_out, cos_out,
                
                domain_node_index,
                tor_bonds_anno,
                twisted_nodes_anno, dihedral_pairs_anno,
                no_tor=False):
    """把任意网络坐标投影到输入定义的扭转/刚体自由度。

    输入参数:
        - pos_in: FloatTensor，形状为 (N, 3)，提供每个域的内部几何基准，单位 Å。
        - pos_out: FloatTensor，形状为 (N, 3)，提供期望二面角与整体位姿的网络目标，单位 Å。
        - sin_in: FloatTensor|None，形状为 (D, 1)，输入二面角正弦；``no_tor=True`` 时可为空。
        - cos_in: FloatTensor|None，形状为 (D, 1)，输入二面角余弦；``no_tor=True`` 时可为空。
        - sin_out: FloatTensor|None，形状为 (D, 1)，目标二面角正弦；``no_tor=True`` 时可为空。
        - cos_out: FloatTensor|None，形状为 (D, 1)，目标二面角余弦；``no_tor=True`` 时可为空。
        - domain_node_index: LongTensor，形状为 (2, K)，第一行是从 0 连续编号的域号，第二行是域内批次原子索引。
        - tor_bonds_anno: LongTensor|None，形状为 (T, 3)，每行 ``[order, u, v]``，后两列是旋转键端点。
        - twisted_nodes_anno: LongTensor|None，形状为 (M, 2)，每行 ``[tor_id, n]``，声明旋转键与随动批次原子。
        - dihedral_pairs_anno: LongTensor|None，形状为 (D, 3)，每行 ``[tor_id, a, d]``，与键端点组成 ``a-u-v-d`` 二面角。
        - no_tor: bool，为真时跳过扭转，只投影逐域刚体位姿。

    返回值:
        - pos_corrected: FloatTensor，形状为 (N, 3)，域内原子由 ``pos_in`` 经允许运动变换得到，域外原子保留 ``pos_out``，单位 Å。
    """
    
    if not no_tor:
        # ``index_tor``：LongTensor，形状为 (D,)，每个二面角实例所属的旋转键编号。
        index_tor = dihedral_pairs_anno[:, 0]
        # ``cos_tor``：FloatTensor，形状为 (D, 1)，cos(θ_out-θ_in)。
        cos_tor = cos_out * cos_in + sin_out * sin_in
        # ``sin_tor``：FloatTensor，形状为 (D, 1)，sin(θ_out-θ_in)。
        sin_tor = sin_out * cos_in - cos_out * sin_in
        # ``angles_tri``：FloatTensor，形状为 (D, 2)，列顺序固定为 ``[sinΔθ, cosΔθ]``。
        angles_tri = torch.cat([sin_tor, cos_tor], dim=1)  # (n_dih, 2)
        # ``angles_tri``：同一旋转键可有多个外侧原子组合；按 tor_id 平均为 ``(T, 2)``。
        angles_tri = scatter_mean(angles_tri, index_tor, dim=0)
        # ``angles_tri``：归一化回单位圆，避免分量平均后幅度小于 1。
        angles_tri = F.normalize(angles_tri, dim=-1)
        
        # ``tor_order``：LongTensor，形状为 (T,)，规定多键扭转的应用顺序。
        tor_order = tor_bonds_anno[:, 0]
        # ``tor_bonds``：LongTensor，形状为 (T, 2)，每条旋转键的两个批内原子索引。
        tor_bonds = tor_bonds_anno[:, 1:]
        # ``index_tor_twisted``：LongTensor，形状为 (M,)，每个随动原子条目所属的旋转键。
        index_tor_twisted = twisted_nodes_anno[:, 0]
        # ``twisted_nodes``：LongTensor，形状为 (M,)，实际随对应旋转键转动的批内原子索引。
        twisted_nodes = twisted_nodes_anno[:, 1]
        # ``pos_tor``：FloatTensor，形状为 (N, 3) Å，在输入几何上施加所有 Δθ 后的坐标。
        pos_tor = apply_torsional_rotation_multiple_domains(pos_in, 
                                tor_order, tor_bonds, angles_tri,
                                twisted_nodes, index_tor_twisted)
    else:
        # ``pos_tor``：不允许扭转时只克隆输入内部几何，避免后续原地别名。
        pos_tor = pos_in.clone()
        
    # ``domain_index``：LongTensor，形状为 (K,)，域注释每列的刚体域号。
    # ``node_index``：LongTensor，形状为 (K,)，与域号逐列对齐的批内原子号。
    domain_index, node_index = domain_node_index
    # pos_center_tor = pos_tor[domain_center_nodes]
    # pos_center_out = pos_out[domain_center_nodes]
    # global_rot, global_trans = kabsch_batch(pos_center_tor, pos_center_out)
    # ``global_rot``：FloatTensor，形状为 (G, 3, 3)，每个域的无反射旋转矩阵。
    # ``global_trans``：FloatTensor，形状为 (G, 1, 3)，每个域适配行向量公式的平移，单位 Å。
    global_rot, global_trans = kabsch_flatten(pos_tor[node_index], pos_out[node_index], domain_index)
    
    # ``pos_corrected_expand``：FloatTensor，形状为 (K, 1, 3) Å；按每个条目的域号选择 R/t。
    # 代码使用行向量约定 ``x @ R^T + t``，因此旋转矩阵在 matmul 前转置。
    pos_corrected_expand = torch.matmul(
        pos_tor[node_index, None, :],
        global_rot.transpose(1, 2)[domain_index]
    ) + global_trans[domain_index]

    # ``pos_corrected``：FloatTensor，形状为 (N, 3) Å；先保留所有网络预测，再覆盖域内原子。
    pos_corrected = pos_out.clone()
    # ``pos_corrected``：将 K 个域注释条目的投影坐标写回对应批内原子行。
    pos_corrected[node_index] = pos_corrected_expand.squeeze(1)
    return pos_corrected
    



def get_dihedral_batch(pos, tor_bonds_anno, dihedral_pairs_anno):
    """按注释批量计算四原子二面角的 sin/cos。

    输入参数:
        - pos: FloatTensor，形状为 (N, 3)，批内配体原子坐标，单位 Å。
        - tor_bonds_anno: LongTensor，形状为 (T, 3)，每行 ``[order, u, v]``，后两列是旋转键批次原子索引。
        - dihedral_pairs_anno: LongTensor，形状为 (D, 3)，每行 ``[tor_id, a, d]``，给出旋转键编号和两个外侧原子索引。

    返回值:
        - sin: FloatTensor，形状为 (D, 1)，按二面角实例对齐的有向角正弦。
        - cos: FloatTensor，形状为 (D, 1)，按二面角实例对齐的有向角余弦。
    """

    # ``index_tor``：LongTensor，形状为 (D,)，每个二面角实例引用的旋转键编号。
    index_tor = dihedral_pairs_anno[:, 0]
    # ``dihedral_ends``：LongTensor，形状为 (D, 2)，四元组最外侧 a/d 原子索引。
    dihedral_ends = dihedral_pairs_anno[:, 1:]  # (n_dih, 2)
    # ``dihedral_tor_nodes``：LongTensor，形状为 (D, 2)，按 tor_id 取出的中心键 u/v 原子索引。
    dihedral_tor_nodes = tor_bonds_anno[:, 1:][index_tor]  # (n_dih, 2)

    # ``sin``：FloatTensor，形状为 (D, 1)，有向二面角 a-u-v-d 的正弦值。
    # ``cos``：FloatTensor，形状为 (D, 1)，有向二面角 a-u-v-d 的余弦值。
    sin, cos = get_dihedral(pos[dihedral_ends[:, 0]], pos[dihedral_tor_nodes[:, 0]],
                    pos[dihedral_tor_nodes[:, 1]], pos[dihedral_ends[:, 1]])
    return sin, cos


def get_dihedral(p0, p1, p2, p3):
    """用 Praxeolitic 公式计算 ``p0-p1-p2-p3`` 有向二面角。

    输入参数:
        - p0: FloatTensor，形状为 (D, 3)，第一个外侧原子坐标，单位 Å。
        - p1: FloatTensor，形状为 (D, 3)，中心键第一个原子坐标，单位 Å。
        - p2: FloatTensor，形状为 (D, 3)，中心键第二个原子坐标，单位 Å。
        - p3: FloatTensor，形状为 (D, 3)，第二个外侧原子坐标，单位 Å。

    返回值:
        - sin: FloatTensor，形状为 (D, 1)，有向二面角正弦，无量纲。
        - cos: FloatTensor，形状为 (D, 1)，有向二面角余弦，无量纲。

    注意:
        - 实现只使用一次叉积，方向由中心轴 ``p1-p2`` 与 ``cross(w, v)`` 的符号约定决定。
    """

    # ``b0``：FloatTensor，形状为 (D, 3) Å，从中心原子 p1 指向外侧原子 p0。
    b0 = p0 - p1
    # b1 = p2 - p1
    # ``b1``：FloatTensor，形状为 (D, 3) Å，沿中心键从 p2 指向 p1。
    b1 = p1 - p2
    # ``b2``：FloatTensor，形状为 (D, 3) Å，从中心原子 p2 指向外侧原子 p3。
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    # 归一化后 ``b1`` 无量纲，确保后续平面投影不受中心键长度影响。
    b1 = F.normalize(b1, dim=-1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    # ``v``：FloatTensor，形状为 (D, 3) Å，b0 在垂直于中心轴平面上的投影。
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    # ``w``：FloatTensor，形状为 (D, 3) Å，b2 在同一垂直平面上的投影。
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    # 归一化后 ``v`` 为平面内单位方向。
    v = F.normalize(v, dim=-1)
    # 归一化后 ``w`` 为平面内单位方向。
    w = F.normalize(w, dim=-1)

    # angle between v and w in a plane is the torsion angle
    # v and w is normalized
    # ``cos``：FloatTensor，形状为 (D, 1)，两个平面投影方向的点积。
    cos = (v * w).sum(-1, keepdim=True)
    # sin = (torch.cross(v, w) * b1).sum()
    # ``sin``：FloatTensor，形状为 (D, 1)，叉积沿中心轴的有向分量。
    sin = (torch.cross(w, v) * b1).sum(-1, keepdim=True)
    return sin, cos


def kabsch_flatten(X, Y, domain_index):
    """在展平的多域原子表上逐域求 Kabsch 刚体变换。

    输入参数:
        - X: FloatTensor，形状为 (K, 3)，待对齐的域内源坐标，单位 Å。
        - Y: FloatTensor，形状为 (K, 3)，与 ``X`` 逐原子对应的目标坐标，单位 Å。
        - domain_index: LongTensor，形状为 (K,)，每行对应的域号，预期从 0 连续编号到 G-1。

    返回值:
        - R: FloatTensor，形状为 (G, 3, 3)，排除镜像反射的逐域最优旋转矩阵。
        - translation: FloatTensor，形状为 (G, 1, 3)，适配行向量公式 ``X @ R^T + translation`` 的逐域平移，单位 Å。
    """
    # ``n_domain``：0 维整型 Tensor/可作尺寸的 G，假设 domain_index 非空且连续。
    n_domain = domain_index.max() + 1
    # ``X_mean``：FloatTensor，形状为 (G, 3) Å，每个源域的质心。
    X_mean = scatter_mean(X, domain_index, dim=0) # (n_domain, 3)
    # ``Y_mean``：FloatTensor，形状为 (G, 3) Å，每个目标域的对应原子质心。
    Y_mean = scatter_mean(Y, domain_index, dim=0) # (n_domain, 3)
    # ``X_centered``：FloatTensor，形状为 (K, 3) Å，按域质心中心化的源坐标。
    X_centered = (X - X_mean[domain_index])  # (N, 3)
    # ``Y_centered``：FloatTensor，形状为 (K, 3) Å，按域质心中心化的目标坐标。
    Y_centered = (Y - Y_mean[domain_index])  # (N, 3)
    
    # Compute the covariance matrix
    # ``covariance_matrix``：FloatTensor，形状为 (G, 3, 3) Å²，逐域累积 X^T Y。
    covariance_matrix = torch.zeros([n_domain, 3, 3], device=X.device, dtype=X.dtype)  # (n_domain, 3, 3)
    # ``i_domain``：int，当前刚体域号，取值范围为 ``[0, G)``。
    for i_domain in range(n_domain):
        # ``this_domain``：BoolTensor，形状为 (K,)，选出当前域的对应原子行。
        this_domain = (domain_index == i_domain)
        # ``covariance_matrix``：当前域协方差为中心化源坐标转置乘中心化目标坐标。
        covariance_matrix[i_domain] = torch.matmul(X_centered[this_domain].transpose(0, 1),
                                                   Y_centered[this_domain])  # (3, 3)
    
    # ``U``：FloatTensor，形状为 (G, 3, 3)，逐域协方差矩阵 SVD 的左奇异向量，计算 dtype 为 float32。
    # ``_``：FloatTensor，形状为 (G, 3)，未使用的逐域奇异值。
    # ``Vt``：FloatTensor，形状为 (G, 3, 3)，逐域协方差矩阵 SVD 的右奇异向量转置，计算 dtype 为 float32；奇异值未使用。
    U, _, Vt = torch.linalg.svd(covariance_matrix.to(torch.float32)) # (n_domain, 3, 3)
    # ``d``：FloatTensor，形状为 (G,) 的 ±1，检测朴素正交解是否包含镜像反射。
    d = torch.sign(torch.det(torch.matmul(U, Vt).to(torch.float32)))  # (n_domain,)

    # Compute the rotation matrix
    # ``diag_mat``：FloatTensor，形状为 (G, 3, 3)，末轴符号修正矩阵。
    diag_mat = torch.eye(3, device=X.device, dtype=d.dtype).unsqueeze(0).repeat(n_domain, 1, 1)
    # ``diag_mat``：仅把每个域的最后一个对角元替换为 det 符号，以排除镜像解。
    diag_mat[:, -1, -1] = d # (n_domain, 3, 3)
    # ``R``：FloatTensor，形状为 (G, 3, 3)，投回 X dtype 的无反射最优旋转。
    R = torch.matmul(torch.matmul(Vt.transpose(1, 2), diag_mat), U.transpose(1, 2)).to(X.dtype) # (n_domain, 3, 3)
    
    
    # Compute the translation vector
    # translation = Y_mean - X_mean
    # translation = Y_mean - torch.matmul(R, X_mean)  # (N, 1, 3)
    # ``translation``：FloatTensor，形状为 (G, 1, 3) Å，使源域质心映射到目标域质心。
    translation = Y_mean.unsqueeze(1) - torch.matmul(X_mean.unsqueeze(1), R.transpose(1, 2))  # (G, 1, 3)
    
    return R, translation


def kabsch_batch(X, Y):
    """
    Align X to Y using rigid transformation
    see https://en.wikipedia.org/wiki/Kabsch_algorithm
    X, Y: (B, N, 3)
    """
    # Normalize the data by centering at the origin
    X_mean = torch.mean(X, dim=1, keepdim=True) # (B, 1, 3)
    Y_mean = torch.mean(Y, dim=1, keepdim=True) # (B, 1, 3)
    X_centered = (X - X_mean)  # (B, N, 3)
    Y_centered = (Y - Y_mean)  # (B, 3, N)
    
    # Compute the covariance matrix
    covariance_matrix = torch.matmul(X_centered.transpose(1, 2), Y_centered)  # (B, 3, 3)
    
    # Perform Singular Value Decomposition, use float64 to avoid numerical issue
    U, _, Vt = torch.linalg.svd(covariance_matrix.to(torch.float32)) # (B, 3, 3)
    d = torch.sign(torch.det(torch.matmul(U, Vt).to(torch.float32)))

    # Compute the rotation matrix
    diag_mat = torch.eye(3, device=X.device, dtype=d.dtype).unsqueeze(0).repeat(X.shape[0], 1, 1)
    diag_mat[:, -1, -1] = d
    R = torch.matmul(torch.matmul(Vt.transpose(1, 2), diag_mat), U.transpose(1, 2)).to(X.dtype) # (B, 3, 3)
    
    
    # Compute the translation vector
    # translation = Y_mean - X_mean
    # translation = Y_mean - torch.matmul(R, X_mean)  # (N, 1, 3)
    translation = Y_mean - torch.matmul(X_mean, R.transpose(1, 2))  # (N, 1, 3)
    
    return R, translation


def procrustes_analysis_batch(X, Y):
    """
    Align X to Y using rigid transformation
    see https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    and https://en.wikipedia.org/wiki/Procrustes_analysis
    X, Y: (B, N, 3)
    """
    # Normalize the data by centering at the origin
    X_mean = torch.mean(X, dim=1, keepdim=True) # (B, 1, 3)
    Y_mean = torch.mean(Y, dim=1, keepdim=True) # (B, 1, 3)
    X_centered_data_T = (X - X_mean)  # (B, N, 3)
    Y_centered_data = (Y - Y_mean).transpose(1, 2)  # (B, 3, N)
    
    # Compute the covariance matrix
    covariance_matrix = torch.matmul(Y_centered_data, X_centered_data_T)  # (B, 3, 3)
    
    # Perform Singular Value Decomposition, use float64 to avoid numerical issue
    U, _, Vt = torch.linalg.svd(covariance_matrix.to(torch.float64)) # (B, 3, 3)
    
    # Compute the rotation matrix
    R = torch.matmul(U, Vt).to(X.dtype) # (B, 3, 3)
    
    # Compute the translation vector
    # translation = Y_mean - X_mean
    # translation = Y_mean - torch.matmul(R, X_mean)  # (N, 1, 3)
    translation = Y_mean - torch.matmul(X_mean, R.transpose(1, 2))  # (N, 1, 3)
    
    return R, translation


def procrustes_analysis_one_sample(X, Y):
    """
    Align X to Y using rigid transformation
    see https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    and https://en.wikipedia.org/wiki/Procrustes_analysis
    X, Y: (N, 3)
    """
    # Normalize the data by centering at the origin
    X_mean = torch.mean(X, dim=0) # (3,)
    Y_mean = torch.mean(Y, dim=0) # (3,)
    X_centered_data = (X - X_mean).T  # (3, N)
    Y_centered_data = (Y - Y_mean).T  # (3, N)
    
    # Compute the covariance matrix
    covariance_matrix = torch.matmul(Y_centered_data, X_centered_data.T)  # (3, 3)
    
    # Perform Singular Value Decomposition
    U, _, Vt = la.svd(covariance_matrix) # (3, 3)
    
    # Compute the rotation matrix
    R = torch.matmul(U, Vt) # (3, 3)
    
    # Compute the translation vector
    # translation = Y_mean - X_mean
    translation = Y_mean - torch.matmul(R, X_mean)  # (3,)
    
    return R, translation


if __name__ == '__main__':
    
    # Original points
    # X = torch.tensor([[1.0, 1.0, 1.0],
    #                 [2.0, 2.0, 2.0],
    #                 [3.0, 3.0, 3.0]])
    X = torch.tensor([[[1., 0, 0],
                      [1, 0, 2],
                      [0, 0, 2],
                      [0, -5, 0]],
                      [[1., 0, 0],
                      [1, 0, 2],
                      [0, 0, 2],
                      [0, -5, 0]]])

    # Transformed points
    Y = torch.tensor([[[0., 1, 0],
                      [0, 1, 2],
                      [0, 0, 2],
                      [5, 0, 0]],
                      [[0., 1, 0],
                      [0, 1, 2],
                      [0, 0, 2],
                      [5, 0, 0]]])
    # Y = torch.tensor([[2.0, 3.0, 0.0],
    #                   [3.0, 3.0, 1.0],
    #                   [4.0, 4.0, 2.0]])
    # Y = torch.tensor([[ 1.8453,  2.7560, -0.1547],
    #         [ 3.0000,  3.3333,  1.0000],
    #         [ 4.1547,  3.9107,  2.1547]])

    # Perform Procrustes analysis
    # R, translation = procrustes_analysis_batch(X, Y)

    # Perform Kabsch algorithm
    R, translation = kabsch_batch(X, Y)


    print("Rotation Matrix:")
    print(R)
    print("\nTranslation Vector:")
    print(translation)

    print('Apply transformation to X:')
    # print(torch.matmul(R, X.T).T + translation)
    Y_app = torch.matmul(X, R.transpose(1, 2)) + translation
    print(Y_app)
    
    print('Delta:')
    print((Y - Y_app).abs().sum())

