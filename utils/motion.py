"""构象与 docking 共用的三维旋转、扭转传播和旋转角采样工具。

本模块只改变坐标，不改变原子、键或刚体域的离散拓扑。调用方约定坐标以 Å 表示，
扭转角与轴角以弧度表示；所有索引都指向当前 PyG 批次中的配体原子或扭转条目。
"""

import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import vonmises, norm
from torch_scatter import scatter_add


def quat_1ijk_to_mat(quat):
    """把省略实部的四元数参数转换成旋转矩阵。

    输入参数:
        - quat: FloatTensor，形状为 (Q, 3)，每行依次为四元数虚部 ``(i, j, k)``；函数先补固定实部 1，再整体归一化。

    返回值:
        - R_mat: FloatTensor，形状为 (Q, 3, 3)，与 Q 个四元数逐项对齐的旋转矩阵。
    """
    # ``quat_1``：FloatTensor，形状为 (Q, 1)，作为每个未归一化四元数的固定实部 1。
    quat_1 = torch.ones(quat.shape[0], 1, device=quat.device)
    # ``quat_1ijk``：FloatTensor，形状为 (Q, 4)，列顺序为 ``(w, x, y, z)``，此时 ``w=1``。
    quat_1ijk = torch.cat([quat_1, quat], dim=-1)
    # ``quat_1ijk``：[Q, 4] -> [Q, 4]，逐行归一化为单位四元数，避免旋转矩阵含缩放分量。
    quat_1ijk = F.normalize(quat_1ijk, dim=-1)
    return quat_to_mat(quat_1ijk)


def quat_to_mat(quat):
    """把单位四元数逐项展开为三维旋转矩阵。

    输入参数:
        - quat: FloatTensor，形状为 (Q, 4)，列顺序为 ``(w, x, y, z)``，每行必须已归一化。

    返回值:
        - R_mat: FloatTensor，形状为 (Q, 3, 3)，与输入四元数逐项对齐。

    异常:
        - AssertionError: 任一四元数的二范数不近似为 1。
    """
    # 该断言沿最后一维检查 Q 个四元数，防止非单位四元数生成带缩放的矩阵。
    assert torch.allclose(quat.norm(dim=-1), torch.ones_like(quat[:, 0])), 'quaternion is not normalized'
    # ``R_mat``：FloatTensor，形状为 (Q, 3, 3)，先在输入设备上建立零矩阵，再按标准四元数公式逐元素填充。
    R_mat = torch.zeros(quat.shape[0], 3, 3, device=quat.device)
    R_mat[:, 0, 0] = 1 - 2*quat[:, 2]**2 - 2*quat[:, 3]**2  # 1 - 2c^2 - 2d^2
    R_mat[:, 0, 1] = 2*quat[:, 1]*quat[:, 2] - 2*quat[:, 0]*quat[:, 3]  # 2bc - 2ad
    R_mat[:, 0, 2] = 2*quat[:, 1]*quat[:, 3] + 2*quat[:, 0]*quat[:, 2]  # 2bd + 2ac
    R_mat[:, 1, 0] = 2*quat[:, 1]*quat[:, 2] + 2*quat[:, 0]*quat[:, 3]  # 2bc + 2ad
    R_mat[:, 1, 1] = 1 - 2*quat[:, 1]**2 - 2*quat[:, 3]**2  # 1 - 2b^2 - 2d^2
    R_mat[:, 1, 2] = 2*quat[:, 2]*quat[:, 3] - 2*quat[:, 0]*quat[:, 1]  # 2cd - 2ab
    R_mat[:, 2, 0] = 2*quat[:, 1]*quat[:, 3] - 2*quat[:, 0]*quat[:, 2] # 2bd - 2ac
    R_mat[:, 2, 1] = 2*quat[:, 2]*quat[:, 3] + 2*quat[:, 0]*quat[:, 1] # 2cd + 2ab
    R_mat[:, 2, 2] = 1 - 2*quat[:, 1]**2 - 2*quat[:, 2]**2  # 1 - 2b^2 - 2c^2
    return R_mat


# def apply_torsional_rotation_multiple_domains(position, edge_index,
#                                               tor_edge, tor_angle, i_bond_for_tor_edge,
#                                               twisted_edge, i_tor_for_twisted_edge):
def apply_torsional_rotation_multiple_domains(positions, tor_order, tor_bonds, tor_angles,
                                                twisted_nodes, index_tor):
    """按拓扑层级依次施加多条可旋转键的扭转。

    输入参数:
        - positions: FloatTensor，形状为 (N, 3)，当前批次 N 个配体原子的坐标，单位 Å。
        - tor_order: LongTensor，形状为 (R,)，第 r 项是第 r 条旋转键的非负执行层级；父层必须先于受其影响的子层。
        - tor_bonds: LongTensor，形状为 (R, 2)，第 r 行为旋转键 ``(left, right)`` 的批内原子索引；旋转轴方向由 ``right`` 指向 ``left``。
        - tor_angles: FloatTensor，形状为 (R,) 或 (R, 2)，第 r 项是旋转键 r 的角度；标量编码为弧度，双通道编码依次为 ``(sin, cos)``。
        - twisted_nodes: LongTensor，形状为 (M,)，第 m 项是会随某条旋转键转动的批内原子索引。
        - index_tor: LongTensor，形状为 (M,)，第 m 项把 ``twisted_nodes[m]`` 映射到全局旋转键编号 ``r``。

    返回值:
        - positions: FloatTensor，形状为 (N, 3)，依次完成全部层级扭转后的坐标，单位 Å；输入张量本身不被原地改写。

    注意:
        - 同一层级内的旋转同时基于该层开始前的坐标计算；层级之间串行更新，因此上游扭转会携带下游旋转键与随动原子一起移动。
    """
    # i_bond_for_twisted_edge = i_bond_for_tor_edge[i_tor_for_twisted_edge]
    # ``twisted_order``：LongTensor，形状为 (M,)，把每个随动原子映射到其旋转键的执行层级。
    twisted_order = tor_order[index_tor]
    # ``largest_order``：零维整数张量，等于最大执行层级加 1，作为 Python ``range`` 的层数上界。
    largest_order = tor_order.max() + 1
    # ``curr_order``：int，逐次取当前扭转层级，范围为 ``[0, largest_order)``。
    for curr_order in range(largest_order):
        # ``ind_curr_tor``：BoolTensor，形状为 (R,)，选择执行层级等于 ``curr_order`` 的全局旋转键。
        ind_curr_tor = (tor_order == curr_order)
        # ``tor_bonds_curr``：LongTensor，形状为 (R_c, 2)，当前层 R_c 条旋转键的批内端点索引。
        tor_bonds_curr = tor_bonds[ind_curr_tor]
        # ``angles_curr``：FloatTensor，形状为 (R_c,) 或 (R_c, 2)，与当前层旋转键逐行对齐。
        angles_curr = tor_angles[ind_curr_tor]
        
        # fetch twisted pairs related to current
        # ``ind_curr_twisted``：BoolTensor，形状为 (M,)，选择由当前层旋转键驱动的随动原子条目。
        ind_curr_twisted = (twisted_order == curr_order)
        # ``twisted_nodes_curr``：LongTensor，形状为 (M_c,)，当前层 M_c 个随动原子的批内原子索引。
        twisted_nodes_curr = twisted_nodes[ind_curr_twisted]
        # ``index_tor_curr``：LongTensor，形状为 (M_c,)，初始仍引用 R 条旋转键中的全局编号。
        index_tor_curr = index_tor[ind_curr_twisted]
        # ``index_tor_curr``：[M_c] -> [M_c]，减去各全局编号之前被过滤掉的键数，把全局编号压缩为当前层局部编号 ``[0, R_c)``。
        index_tor_curr -= (~ind_curr_tor).cumsum(dim=0)[index_tor_curr]
        # 当前层局部旋转键编号必须从 0 连续出现，否则随动原子与 ``tor_bonds_curr`` 无法逐项对齐。
        assert index_tor_curr.unique().shape[0] == index_tor_curr.max()+1, 'index_tor_curr is wrong'
        # ``positions``：[N, 3] -> [N, 3]，以当前层开始前的坐标同时旋转本层全部随动原子。
        positions = apply_torsional_rotation(positions, tor_bonds_curr, angles_curr,
                                            twisted_nodes_curr, index_tor_curr)
    return positions


# def apply_torsional_rotation(positions, edge_index, tor_edge, tor_angle,
#                              twisted_edge, i_tor_for_twisted):
def apply_torsional_rotation(positions, tor_bonds, tor_angles,
                             twisted_nodes, index_tor):
    """围绕一组旋转键同时转动各键指定侧的原子。

    输入参数:
        - positions: FloatTensor，形状为 (N, 3)，当前批次 N 个配体原子的坐标，单位 Å。
        - tor_bonds: LongTensor，形状为 (R, 2)，第 r 行为旋转键 ``(left, right)`` 的批内原子索引。
        - tor_angles: FloatTensor，形状为 (R,) 或 (R, 2)，第 r 项为弧度角或 ``(sin, cos)`` 编码。
        - twisted_nodes: LongTensor，形状为 (M,)，列出本次需要转动的批内原子索引；实现要求每个原子至多出现一次。
        - index_tor: LongTensor，形状为 (M,)，第 m 项指定 ``twisted_nodes[m]`` 使用第几条局部旋转键和角度。

    返回值:
        - positions_new: FloatTensor，形状为 (N, 3)，只覆盖随动原子坐标的独立副本，单位 Å。

    注意:
        - 旋转轴使用 ``positions[left] - positions[right]``，即从 ``right`` 指向 ``left``；旋转中心取每个随动原子在该轴上的正交投影。
    """
    # # get nodes
    # ``node_tor_left``：LongTensor，形状为 (R,)，逐旋转轴左端的批内原子索引。
    # ``node_tor_right``：LongTensor，形状为 (R,)，逐旋转轴右端的批内原子索引。
    node_tor_left, node_tor_right = tor_bonds.T
    # 同一调用内不允许一个原子同时隶属两条旋转键；跨层复合旋转由外层函数分步完成。
    assert (twisted_nodes.unique().shape[0] == twisted_nodes.shape[0]), 'twisted node appears in over one torsion'
    
    # # get the positions of nodes and vectors of edges
    # ``pos_tor_left``：FloatTensor，形状为 (R, 3)，旋转键左端坐标，单位 Å。
    pos_tor_left = positions[node_tor_left]
    # ``pos_tor_right``：FloatTensor，形状为 (R, 3)，旋转键右端坐标，单位 Å。
    pos_tor_right = positions[node_tor_right]
    # ``pos_twisted``：FloatTensor，形状为 (M, 3)，每个随动原子旋转前的坐标，单位 Å。
    pos_twisted = positions[twisted_nodes]
    
    # ``vec_tor_edge``：FloatTensor，形状为 (R, 3)，从旋转键右端指向左端的轴向量，单位 Å。
    vec_tor_edge = (pos_tor_left - pos_tor_right)
    # ``unit_tor_edge_expand``：FloatTensor，形状为 (M, 3)，归一化旋转轴按 ``index_tor`` 广播到每个随动原子。
    unit_tor_edge_expand = F.normalize(vec_tor_edge, dim=-1)[index_tor]
    # ``vec_twisted_edge``：FloatTensor，形状为 (M, 3)，从所属旋转键左端指向随动原子的向量，单位 Å。
    vec_twisted_edge = pos_twisted - pos_tor_left[index_tor]
    
    # # calculate rotation-related parameters
    # ``rot_axes``：FloatTensor，形状为 (M, 3)，每个随动原子的单位旋转轴。
    rot_axes = unit_tor_edge_expand
    # ``rot_angles``：FloatTensor，形状为 (M,) 或 (M, 2)，所属旋转键的角度按随动原子展开。
    rot_angles = tor_angles[index_tor]
    # ``radius_vec``：FloatTensor，形状为 (M, 3)，从轴上投影点指向随动原子的垂直半径向量，单位 Å。
    radius_vec = vec_twisted_edge - unit_tor_edge_expand * \
        (vec_twisted_edge * unit_tor_edge_expand).sum(-1, keepdims=True)
    # ``rot_center``：FloatTensor，形状为 (M, 3)，每个随动原子在旋转轴上的正交投影，单位 Å。
    rot_center = pos_twisted - radius_vec
    
    # # apply rotation
    # ``pos_twisted_rot``：FloatTensor，形状为 (M, 3)，旋转半径向量后加回轴上中心得到的新坐标，单位 Å。
    pos_twisted_rot = apply_axis_angle_rotation(
        radius_vec, rot_axes, rot_angles) + rot_center
    # ``positions_new``：FloatTensor，形状为 (N, 3)，用于避免原地修改调用方的输入坐标。
    positions_new = positions.clone()
    # ``positions_new``：[N, 3] -> [N, 3]，只把 ``twisted_nodes`` 指定行替换为旋转后坐标。
    positions_new[twisted_nodes] = pos_twisted_rot
    return positions_new


def apply_axis_angle_rotation(positions, rot_axes, rot_angles):
        """使用 Rodrigues 公式逐向量施加轴角旋转。

        输入参数:
            - positions: FloatTensor，形状为 (M, 3)，以各自旋转中心为原点的待旋转向量，通常单位为 Å。
            - rot_axes: FloatTensor，形状为 (M, 3)，逐向量对齐的旋转轴；调用方必须保证每行是单位向量。
            - rot_angles: FloatTensor，形状为 (M,)、(M, 1) 或 (M, 2)；前两种以弧度表示，双通道时列顺序固定为 ``(sin, cos)``。

        返回值:
            - rot_vec: FloatTensor，形状为 (M, 3)，绕对应轴旋转后的向量，与 ``positions`` 单位相同。
        """
        if rot_angles.dim() < rot_axes.dim():
            # ``rot_angles``：[M] -> [M, 1]，实际执行补通道维以便与三维向量广播。
            rot_angles = rot_angles[:, None]
        
        if rot_angles.size(-1) == 2:
            # ``sin_angle``：FloatTensor，形状为 (M, 1)，直接读取双通道角编码的正弦列。
            sin_angle = rot_angles[..., 0:1]
            # ``cos_angle``：FloatTensor，形状为 (M, 1)，直接读取双通道角编码的余弦列。
            cos_angle = rot_angles[..., 1:2]
        else:
            # ``sin_angle``：FloatTensor，形状为 (M, 1)，由弧度角逐项计算。
            sin_angle = torch.sin(rot_angles)
            # ``cos_angle``：FloatTensor，形状为 (M, 1)，由弧度角逐项计算。
            cos_angle = torch.cos(rot_angles)
        # ``rot_vec``：FloatTensor，形状为 (M, 3)，依次叠加平行分量、叉积切向分量和轴向投影分量。
        rot_vec = (
            positions * cos_angle + 
            torch.linalg.cross(rot_axes, positions) * sin_angle +
            rot_axes * (rot_axes * positions).sum(-1, keepdims=True) * (1 - cos_angle)
        )
        return rot_vec


def sample_uniform_angle(sigmas):
    """按 ``sigmas`` 的形状采样均匀圆周角。

    输入参数:
        - sigmas: FloatTensor，任意形状；这里只借用形状、dtype 与 device，不使用数值。

    返回值:
        - angles: FloatTensor，形状与 ``sigmas`` 相同，每项独立服从 ``[-π, π)`` 上的均匀分布，单位为弧度。
    """
    # ``angles``：FloatTensor，形状与 ``sigmas`` 相同，把 ``[0, 1)`` 均匀变量线性映射到 ``[-π, π)``。
    angles = torch.rand_like(sigmas) * 2 * torch.pi - torch.pi
    return angles


def robust_sample_angle(sigmas, sigma_th=0.1):
    """按角噪声尺度混合截断高斯近似与 von Mises 圆周分布。

    输入参数:
        - sigmas: FloatTensor，任意形状，每项是对应扭转角的噪声尺度，单位为弧度。
        - sigma_th: float，选择分布的尺度阈值，单位为弧度；``sigma <= sigma_th`` 使用截断高斯，否则使用 von Mises。

    返回值:
        - samples: FloatTensor，形状、dtype 与 device 均与 ``sigmas`` 相同，角度范围经高斯分支裁剪或圆周分布自然限制在 ``[-π, π]``，单位为弧度。
    """
    # ``samples``：FloatTensor，形状与 ``sigmas`` 相同，先按逐项标准差生成零均值高斯角噪声。
    samples = torch.randn_like(sigmas) * sigmas
    # ``samples``：同形状高斯样本裁剪到 ``[-π, π]``，避免小尺度近似产生超出主值区间的角度。
    samples = samples.clamp(-torch.pi, torch.pi)
    # ``from_vonmises``：BoolTensor，形状与 ``sigmas`` 相同，标记尺度大于阈值、需要圆周分布采样的条目。
    from_vonmises = (sigmas > sigma_th)
    if from_vonmises.any():
        # ``kappa``：FloatTensor 或 ndarray，形状为 (M_v,)，von Mises 集中度 ``1/sigma²``，无量纲。
        kappa = (1 / sigmas[from_vonmises]**2)
        if isinstance(kappa, torch.Tensor):
            # ``kappa``：Tensor -> ndarray，转到 CPU 以满足 SciPy 的输入契约。
            kappa = kappa.cpu().numpy()
        # ``samples_vonmises``：ndarray，形状为 (M_v,)，由 SciPy 采样的零均值圆周角，单位为弧度。
        samples_vonmises = vonmises(kappa=kappa).rvs()
        # ``samples``：把大尺度位置替换为 von Mises 样本，并恢复原张量的 dtype 与 device。
        samples[from_vonmises] = torch.tensor(samples_vonmises, dtype=samples.dtype, device=samples.device)
    return samples

class RobustAngleSO3Distribution(torch.nn.Module):
    """离散近似各向同性 SO(3) 扩散的旋转角边缘分布，并支持按尺度采样。"""

    def __init__(self, sigma_th=4.e-3, n_bins=1000, n_L=1001):
        """预计算旋转角网格及谱展开中与 ``sigma`` 无关的系数。

        输入参数:
            - sigma_th: float，小于该阈值时改用高斯近似的旋转尺度，单位为弧度。
            - n_bins: int，把 ``[0, π)`` 划分成的等宽角度区间数量。
            - n_L: int，SO(3) 谱展开保留的非负阶数数量，对应 ``l=0,...,n_L-1``。
        """
        super().__init__()
        # ``self.sigma_th``：float，小尺度高斯近似阈值，单位为弧度。
        self.sigma_th = sigma_th
        # ``self.n_bins``：int，离散旋转角区间数。
        self.n_bins = n_bins
        # ``self.n_L``：int，谱展开阶数。
        self.n_L = n_L
        # ``bin_width``：float，单个旋转角区间宽度 ``π/n_bins``，单位为弧度。
        bin_width = torch.pi / n_bins
        # ``bins``：FloatTensor，形状为 (n_bins,)，各等宽区间的中心角，范围为 ``(0, π)``，单位为弧度。
        bins = torch.linspace(0, torch.pi, n_bins+1)[:-1] + bin_width / 2
        # ``bins_expand``：[n_bins] -> [1, n_bins, 1]，为批尺度轴与谱阶轴补广播维。
        bins_expand = bins[None, :, None]
        # ``ls``：LongTensor，形状为 (n_L,)，谱展开的阶数 ``0,...,n_L-1``。
        ls = torch.arange(n_L)
        # ``ls``：[n_L] -> [1, 1, n_L]，为批尺度轴与角度网格轴补广播维。
        ls = ls[None, None, :]
        # ``c0``：FloatTensor，形状为 (1, n_bins)，SO(3) 角度测度在每个网格中心的因子。
        c0 = ((1 - torch.cos(bins_expand)) / torch.pi).squeeze(-1)
        # ``c2``：FloatTensor，形状为 (1, n_bins, n_L)，每个角度中心与谱阶的字符展开因子。
        c2 = (2*ls+1) * torch.sin((ls+0.5) * bins_expand) / torch.sin(bins_expand/2)
        # ``self.bin_width``：float，保存的角度区间宽度，单位为弧度。
        self.bin_width = bin_width
        # 以下 buffer 随模块迁移 device、进入 state_dict，但不参与梯度更新。
        self.register_buffer('ls', ls)
        self.register_buffer('bins', bins)
        self.register_buffer('c0', c0)
        self.register_buffer('c2', c2)

    @torch.no_grad()
    def sample(self, sigma, is_uniform=False):
        """为一组 SO(3) 扰动尺度采样旋转角的绝对值。

        输入参数:
            - sigma: FloatTensor，形状为 (S,)，每项是一个刚体域的旋转噪声尺度，单位为弧度。
            - is_uniform: bool；为真时忽略 ``sigma`` 数值并按 SO(3) Haar 测度采样角度，为假时使用尺度相关的谱分布。

        返回值:
            - angles: FloatTensor，形状为 (S,)，每项位于约 ``[0, π]``，表示旋转角绝对值，单位为弧度。
        """
        # ``n``：int，本次需要采样的刚体域/尺度条目数 S。
        n = len(sigma)
        # ``sigma_expand``：[S] -> [S, 1, 1]，为角度网格轴和谱阶轴补广播维。
        sigma_expand = sigma[:, None, None]

        # angle distribution
        if not is_uniform:
            # ``c1``：FloatTensor，形状为 (S, 1, n_L)，尺度相关的谱衰减系数。
            c1 = torch.exp(-self.ls*(self.ls+1) * (sigma_expand**2))
            # ``probs``：FloatTensor，形状为 (S, n_bins)，每个尺度在各旋转角区间上的未归一化概率质量。
            probs = self.c0 * (c1 * self.c2).sum(-1)
        else:
            # ``probs``：FloatTensor，形状为 (S, n_bins)，SO(3) Haar 测度的角度概率质量复制到每个条目。
            probs = self.c0.repeat_interleave(n, dim=0)
        # ``probs``：同形状概率质量裁去谱截断导致的微小负值；``multinomial`` 会按行自行归一化。
        probs = probs.clamp(min=0)
        # ``idx_bins``：LongTensor，形状为 (S,)，逐尺度采到的角度区间索引，取值范围为 ``[0, n_bins)``。
        idx_bins = torch.multinomial(probs, num_samples=1).squeeze(-1)
        # ``angles``：FloatTensor，形状为 (S,)，先取所选区间的中心角，单位为弧度。
        angles = self.bins[idx_bins]
        # ``angles``：同形状中心角加区间内均匀抖动，使离散网格恢复为连续角度。
        angles = angles + self.bin_width * (torch.rand_like(angles) - 0.5)
        
        if not is_uniform:
            # ``idx_gaussian``：BoolTensor，形状为 (S,)，选择尺度小于阈值、需用均值 ``2σ`` 和标准差 ``σ`` 的高斯近似条目。
            idx_gaussian = sigma < self.sigma_th
            # ``angles``：仅覆盖小尺度条目；该近似未在此处裁剪，保持原实现行为。
            angles[idx_gaussian] = (sigma[idx_gaussian] * 2 +
                torch.randn_like(angles[idx_gaussian]) * sigma[idx_gaussian])
        return angles
    
