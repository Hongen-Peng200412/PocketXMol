"""实现 ``local_cov`` 的固定受体体素网格、11³ 局部读取和逐层 FiLMPlus 调制.

主要入口是 :class:`LocalCovConditioner`. ``build_grid`` 先把当前已选口袋内的
50 维受体原子特征硬散射到 80³ 体素, 再与固定 56 维密度通道拼成 106 通道网格.
六个去噪块分别调用 ``forward``; 每次依据该层当前配体坐标重新计算 home 体素,
读取 11³ 局部块, 经该层独立卷积编码器和共享 LayerNorm 后调制配体节点特征.

本模块只返回内存张量, 不读取或写入文件. 外围密度裁块的读取和几何字段由
``docking.density.load_density_input`` 负责; 本模块不移动裁块、模型原点或 home 体素.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


# int, 每个密度体素的输入特征通道数, 固定为 56.
DENSITY_CHANNELS = 56
# int, 每个已选受体原子的特征通道数, 占组合网格的后 50 通道.
RECEPTOR_CHANNELS = 50
# int, 固定密度裁块在 Z、Y、X 三轴上的体素数.
GRID_SIZE = 80
# int, 每个配体原子沿 Z、Y、X 三轴读取的局部体素数.
CUBE_SIZE = 11
# int, 局部卷积编码器输出及特征调制器输入的条件宽度.
CONDITION_DIM = 64


def positions_to_home_zyx(
    positions_xyz: torch.Tensor,
    batch_index: torch.Tensor,
    density_origin_xyz: torch.Tensor,
    density_basis_xyz: torch.Tensor,
) -> torch.Tensor:
    """把原子模型局部 XYZ 坐标转换为所属密度裁块的离散 ZYX 体素索引（home）.

    ``N`` 是待查询原子数, ``B`` 是批次中的密度裁块数; 所有原子输入按第一维对齐.

    输入:
        - positions_xyz: (N, 3), 原子的模型局部 XYZ 坐标, 单位 Å.
        - batch_index: 整型, (N,), 每个原子所属裁块在 density_origin_xyz 和 density_basis_xyz 第一维的索引.
        - density_origin_xyz: (B, 3), 各裁块零号体素角点的模型局部 XYZ 坐标, 单位 Å.
        - density_basis_xyz: (B, 3, 3), 三行分别是体素 X、Y、Z 轴在模型局部 XYZ 坐标中的步进向量, 单位 Å.

    返回:
        - int64, (N, 3), 与 positions_xyz 逐原子对齐的 ZYX 体素索引; 连续体素坐标逐轴取 floor 后换轴, 不截断越界值.

    例如单位基向量、零角点下, 原子位置 (2.2, 3.4, 4.1) Å 对应 (4, 3, 2) 的 ZYX 体素索引.
    逆基矩阵和坐标位移固定使用 float32 计算, 不沿用输入的混合精度.
    """
    # int64, (N,), 每个原子在 density_origin_xyz 和 density_basis_xyz 第一维中的裁块索引.
    batch_index = batch_index.long()
    # float32, (B, 3, 3), 各裁块的逆基矩阵; 局部 XYZ 位移右乘它得到连续体素 XYZ 坐标.
    inverse_basis = torch.linalg.inv(density_basis_xyz.float())
    # float32, (N, 3), 各原子相对所属裁块零号体素角点的模型局部 XYZ 位移, 单位 Å.
    local_xyz = positions_xyz.float() - density_origin_xyz.float()[batch_index]
    # (N, 3), 连续体素 XYZ 坐标; [N, 3] -> [N, 1, 3] 与所属裁块的 [N, 3, 3] 逆基矩阵相乘后移除单维.
    voxel_xyz = torch.bmm(local_xyz[:, None, :], inverse_basis[batch_index]).squeeze(1)
    # int64, (N, 3), 按 ZYX 排列的原子体素索引; 裁块外的负值及过大索引原样保留.
    return torch.floor(voxel_xyz).long()[:, [2, 1, 0]]


def scatter_receptor_features(
    density_input: torch.Tensor,
    receptor_feature: torch.Tensor,
    receptor_pos_xyz: torch.Tensor,
    receptor_batch: torch.Tensor,
    density_origin_xyz: torch.Tensor,
    density_basis_xyz: torch.Tensor,
) -> torch.Tensor:
    """把调用方已选的受体原子特征散射到密度裁块, 组成固定体素网格.

    ``B`` 是批次中的密度裁块数, ``P`` 是已选受体原子数; 原子输入按第一维对齐.
    本函数不选择受体原子; 只让归属体素位于 80³ 裁块内的原子参与逐通道求和.

    输入:
        - density_input: (B, 56, 80, 80, 80), 各裁块的密度通道, 空间轴为 ZYX.
        - receptor_feature: (P, 50), 各受体原子的 50 维特征, 第二维为特征通道.
        - receptor_pos_xyz: (P, 3), 受体原子在模型局部坐标系中的 XYZ 位置, 单位 Å.
        - receptor_batch: 整型, (P,), 各受体原子所属裁块在 density_input 第一维的索引.
        - density_origin_xyz: (B, 3), 各裁块零号体素角点的模型局部 XYZ 坐标, 单位 Å.
        - density_basis_xyz: (B, 3, 3), 三行分别为体素 X、Y、Z 轴的模型局部 XYZ 步进向量, 单位 Å.

    返回:
        - (B, 106, 80, 80, 80), 前 56 通道复制输入密度, 后 50 通道逐体素累加受体特征, 空间轴为 ZYX.

    例如同一裁块的两个受体原子落在体素 (2, 1, 0), 且特征首通道为 1 和 2, 网格第 56 通道在该体素的值为 3.
    越界原子被忽略; 输出沿用 density_input 的数据类型, 不修改输入张量.
    """
    # B 和 ZYX 空间尺寸来自 density_input; 通道数与三个空间尺寸必须匹配固定裁块契约.
    batch_size, density_channels, dim_z, dim_y, dim_x = density_input.shape
    if (density_channels, dim_z, dim_y, dim_x) != (DENSITY_CHANNELS, GRID_SIZE, GRID_SIZE, GRID_SIZE):
        raise ValueError(f"density_input形状必须为(B,56,80,80,80), 实际为{tuple(density_input.shape)}. ")

    # (B, 106, 80, 80, 80), 前 56 通道复制密度, 后 50 通道初始为零并接收受体特征之和.
    grid = density_input.new_zeros((batch_size, DENSITY_CHANNELS + RECEPTOR_CHANNELS, dim_z, dim_y, dim_x))
    grid[:, :DENSITY_CHANNELS].copy_(density_input)
    # int64, (P, 3), 已选受体原子在各自裁块内的 ZYX 体素索引, 尚可含越界值.
    home_zyx = positions_to_home_zyx(receptor_pos_xyz, receptor_batch, density_origin_xyz, density_basis_xyz)
    # bool, (P,), True 表示该受体原子的 ZYX 体素索引均在裁块内; False 对应的 receptor_feature 第一维原子不写入 grid.
    valid = (
        (home_zyx[:, 0] >= 0) & (home_zyx[:, 0] < dim_z)
        & (home_zyx[:, 1] >= 0) & (home_zyx[:, 1] < dim_y)
        & (home_zyx[:, 2] >= 0) & (home_zyx[:, 2] < dim_x)
    )
    if valid.any():
        # P_valid 是 valid 中为 True 的受体原子数; 以下索引与 receptor_feature[valid] 的第一维对齐.
        # int64, (P_valid,), 各有效受体原子在 grid 第一维中的裁块索引.
        batch_valid = receptor_batch.long()[valid]
        # int64, 各为 (P_valid,), 有效受体原子在 grid 的 Z、Y、X 空间轴上的索引.
        z_index, y_index, x_index = home_zyx[valid].unbind(dim=1)
        # int64, (1, 50), 受体特征通道的相对编号, 与 P_valid 个有效受体原子广播配对.
        channel_index = torch.arange(RECEPTOR_CHANNELS, device=grid.device, dtype=torch.long)[None, :]
        # int64, (P_valid, 50), 每个有效受体原子及特征通道在 grid 的 B、通道、Z、Y、X 连续存储中的一维索引; 同体素同通道共享索引.
        flat_index = (
            ((((batch_valid[:, None] * (DENSITY_CHANNELS + RECEPTOR_CHANNELS) + DENSITY_CHANNELS + channel_index) * dim_z + z_index[:, None]) * dim_y + y_index[:, None]) * dim_x + x_index[:, None])
        )
        # [P_valid, 50] -> [P_valid * 50], 展平索引与同序特征, 转为 grid 的设备和数据类型后对相同索引求和.
        grid.view(-1).scatter_add_(
            0,
            flat_index.reshape(-1),
            receptor_feature[valid].to(device=grid.device, dtype=grid.dtype).reshape(-1),
        )
    # (B, 106, 80, 80, 80), 无有效受体原子时后 50 通道仍为零, 前 56 通道始终保留密度.
    return grid


def gather_voxel_cube(
    grid: torch.Tensor,
    center_zyx: torch.Tensor,
    batch_index: torch.Tensor,
    cube_size: int = CUBE_SIZE,
) -> torch.Tensor:
    """围绕各原子的实际 home（所属体素的离散索引）读取局部块, 越过原裁块的部分填零.

    ``B`` 是网格数, ``C`` 是网格通道数, ``N`` 是待查询原子数; ``k`` 等于 cube_size.
    ``Z``、``Y``、``X`` 是输入网格的三个空间尺寸; 当前模型使用 Z=Y=X=80 和奇数 k=11.

    输入:
        - grid: (B, C, Z, Y, X), 密度与受体特征网格, 空间轴为 ZYX.
        - center_zyx: 整型, (N, 3), 各原子在所属裁块中的离散 ZYX home 索引, 可越界.
        - batch_index: 整型, (N,), 与 center_zyx 第一维对齐的原子所属裁块索引, 对应 grid 第一维.
        - cube_size: int, 各空间轴读取的体素数, 默认 11.

    返回:
        - (N, C, k, k, k), 与 center_zyx 逐原子对齐的局部网格, 空间轴为 ZYX; 输入网格外的值为零.

    例如 Z 轴 home 为 0 且 k=11 时, 读取范围是 -5 至 5; 其中 -5 至 -1 的切片填零.
    clamp 只用于安全地索引 grid, 不移动 home; 本函数不返回有效体素掩码.
    """
    # k 为奇数时, radius 是中心到任一端点的体素数; 默认 k=11 时为 5.
    radius = cube_size // 2
    # 三个空间尺寸依次对应 grid 的 Z、Y、X 轴, 当前模型均为 80.
    dim_z, dim_y, dim_x = grid.shape[2:]
    # int64, (k,), 在 Z、Y、X 三轴复用的离散体素偏移, 默认从 -5 到 5.
    offsets = torch.arange(cube_size, device=grid.device, dtype=torch.long) - radius
    # int64, (N, 3), 把各原子的 ZYX home 索引放到 grid 所在设备, 保留越界值.
    center_zyx = center_zyx.to(device=grid.device, dtype=torch.long)
    # int64, (N,), 与 center_zyx 第一维逐原子对齐的 grid 裁块编号.
    batch_index = batch_index.to(device=grid.device, dtype=torch.long)

    # int64, 各为 (N, k), 每个原子在 Z、Y、X 三轴上待读取的真实体素索引; 可以越界.
    z_raw = center_zyx[:, 0, None] + offsets
    y_raw = center_zyx[:, 1, None] + offsets
    x_raw = center_zyx[:, 2, None] + offsets
    # bool, (N, k, k, k), 三个单轴有效范围按 ZYX 广播组合; True 表示该原子局部块的体素位于原裁块内.
    valid = (
        ((z_raw >= 0) & (z_raw < dim_z))[:, :, None, None]
        & ((y_raw >= 0) & (y_raw < dim_y))[:, None, :, None]
        & ((x_raw >= 0) & (x_raw < dim_x))[:, None, None, :]
    )
    # [B, C, Z, Y, X] -> [N, k, k, k, C] -> [N, C, k, k, k]; clamp 只避免越界读取, 随后用 valid 去掉替代值.
    cube = grid[
        batch_index[:, None, None, None],
        :,
        z_raw.clamp(0, dim_z - 1)[:, :, None, None],
        y_raw.clamp(0, dim_y - 1)[:, None, :, None],
        x_raw.clamp(0, dim_x - 1)[:, None, None, :],
    ].permute(0, 4, 1, 2, 3).contiguous()
    # [N, k, k, k] -> [N, 1, k, k, k], 沿 C 通道广播有效掩码; 越界体素的所有通道均置零.
    return cube * valid[:, None].to(cube.dtype)


class LocalCubeEncoder(nn.Module):
    """用三层卷积把每个配体原子的局部密度与受体网格编码为条件向量.

    ``M`` 是当前配体原子分块中的原子数, 每个原子对应一个 11³ 局部块.

    前向输入:
        - cube: (M, 106, 11, 11, 11), 前 56 通道为密度、后 50 通道为受体特征; 空间轴为 ZYX, 与分块内配体原子顺序对齐.

    前向输出:
        - condition: (M, 64), 各配体原子的局部密度和受体条件, 与 cube 第一维逐原子对齐.
    """

    def __init__(self) -> None:
        """建立三层卷积、逐原子空间平均池化及 64 维线性投影."""
        super().__init__()
        # (M, 106, 11, 11, 11) -> (M, 64, 6, 6, 6) -> (M, 64, 3, 3, 3); 最后一层保持 3³, 各层随后进行分组归一化与 SiLU 激活.
        self.convolutions = nn.Sequential(
            nn.Conv3d(106, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        # (M, 64, 3, 3, 3) -> (M, 64, 1, 1, 1) -> (M, 64); 各配体原子独立平均空间体素并投影到条件宽度.
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Linear(64, CONDITION_DIM)

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        """将 (M, 106, 11, 11, 11) 局部块编码为逐原子对齐的 (M, 64) 条件向量."""
        # [M, 64, 1, 1, 1] -> [M, 64], 只展平空间单维, 不合并配体原子轴.
        return self.projection(self.pool(self.convolutions(cube)).flatten(1))


class FiLMPlusCombine(nn.Module):
    """从局部条件生成逐通道缩放与偏移, 调制同一配体原子的节点特征.

    ``N`` 是配体原子数; 条件和节点特征在第一维逐原子对齐.

    构造参数:
        - node_dim: int, 配体节点特征宽度; 正式模型为 320.

    前向输入:
        - node_feature: (N, node_dim), 当前去噪块完成节点更新后的配体原子特征.
        - condition: (N, 64), 各配体原子的局部密度与受体条件.

    前向输出:
        - combined: (N, node_dim), 执行 node_feature * (1 + gamma) + beta 后的逐原子节点特征.
    """

    def __init__(self, node_dim: int) -> None:
        """建立从 64 维条件生成缩放与偏移的网络, 将末层参数初始化为零."""
        super().__init__()
        # int, 配体节点特征宽度, 同时决定 gamma 和 beta 各自的通道数.
        self.node_dim = int(node_dim)
        # (N, 64) -> (N, node_dim) -> (N, 2 * node_dim); 前半为 gamma, 后半为 beta.
        self.generator = nn.Sequential(
            nn.Linear(CONDITION_DIM, self.node_dim),
            nn.SiLU(),
            nn.Linear(self.node_dim, 2 * self.node_dim),
        )
        # 末层输出初始为零, 因而调制开始时不改变 node_feature.
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)

    def forward(self, node_feature: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """按原子生成 (N, node_dim) 的缩放与偏移, 返回同形状特征; 初始输出等于 node_feature."""
        # [N, 2 * node_dim] -> 两个 [N, node_dim], 分别调节同一原子各通道的比例和偏移.
        gamma, beta = self.generator(condition).chunk(2, dim=-1)
        return node_feature * (1.0 + gamma) + beta


class LocalCovConditioner(nn.Module):
    """为六个去噪块管理独立的局部体素编码器与特征调制器, 共享输出归一化层.

    ``B`` 是密度裁块数, ``P`` 是受体原子数, ``N`` 是当前配体原子数.
    六个去噪块读取同一固定网格, 但各自使用当层更新后的配体坐标重新定位 11³ 邻域.

    构造参数:
        - node_dim: int, 配体原子节点特征宽度; 正式模型为 320.
        - num_blocks: int, 去噪块数量, 固定为 6.
        - chunk_size: int, 一次读取局部块并运行卷积的配体原子数, 只允许 4096 或 2048.

    build_grid 输入:
        - density_input: (B, 56, 80, 80, 80), 各裁块的 56 个密度通道, 空间轴为 ZYX.
        - receptor_feature: (P, 50), 已选受体原子的 50 维特征.
        - receptor_pos_xyz: (P, 3), 与 receptor_feature 对齐的受体原子模型局部 XYZ 坐标, 单位 Å.
        - receptor_batch: 整型, (P,), 各受体原子对应 density_input 第一维的裁块索引.
        - density_origin_xyz: (B, 3), 各裁块零号体素角点的模型局部 XYZ 坐标, 单位 Å.
        - density_basis_xyz: (B, 3, 3), 各裁块体素 XYZ 轴在模型局部 XYZ 坐标中的步进向量, 单位 Å.

    build_grid 输出:
        - grid: (B, 106, 80, 80, 80), 前 56 通道为密度, 后 50 通道为同体素受体原子的逐通道特征和.

    forward 输入:
        - block_index: int, 当前去噪块编号, 索引六套独立编码器和调制器.
        - node_feature: (N, node_dim), 当前去噪块已更新的配体原子节点特征.
        - ligand_pos_xyz: (N, 3), 与 node_feature 对齐的当层配体原子模型局部 XYZ 坐标, 单位 Å.
        - ligand_batch: 整型, (N,), 各配体原子对应 grid 第一维的裁块索引.
        - grid: (B, 106, 80, 80, 80), build_grid 构造的固定体素网格, 空间轴为 ZYX.
        - density_origin_xyz: (B, 3), 与 grid 同序的裁块零号体素角点 XYZ 坐标, 单位 Å.
        - density_basis_xyz: (B, 3, 3), 与 grid 同序的体素 XYZ 步进向量, 单位 Å.

    forward 输出:
        - node_feature: (N, node_dim), 与输入配体原子顺序一致的局部条件调制后特征.

    例如 N=5000 且 chunk_size=4096 时, 条件分两次编码 4096 和 904 个原子, 再按原顺序拼接.
    仅训练模式使用激活重计算; 本模块不移动固定网格或修改原子坐标.
    """

    def __init__(self, node_dim: int, num_blocks: int, chunk_size: int) -> None:
        """建立六套独立的局部编码器与调制器, 以及对 64 维条件共用的归一化层."""
        super().__init__()
        if num_blocks != 6:
            raise ValueError(f"local_cov固定需要6个去噪块, 实际为{num_blocks}. ")
        if chunk_size not in (4096, 2048):
            raise ValueError(f"local_cov的chunk_size只允许4096或2048, 实际为{chunk_size}. ")
        # int, 单次编码的配体原子数上限; 按输入原子顺序连续分块.
        self.chunk_size = int(chunk_size)
        # 六个去噪块分别使用 encoders[i] 与 film_layers[i]; 输出条件统一经过同一个 64 维 LayerNorm.
        self.encoders = nn.ModuleList(LocalCubeEncoder() for _ in range(num_blocks))
        self.output_norm = nn.LayerNorm(CONDITION_DIM)
        self.film_layers = nn.ModuleList(FiLMPlusCombine(node_dim) for _ in range(num_blocks))

    def build_grid(
        self,
        density_input: torch.Tensor,
        receptor_feature: torch.Tensor,
        receptor_pos_xyz: torch.Tensor,
        receptor_batch: torch.Tensor,
        density_origin_xyz: torch.Tensor,
        density_basis_xyz: torch.Tensor,
    ) -> torch.Tensor:
        """构造六个去噪块共同读取的固定密度与受体特征网格.

        形状符号:
            - B: 当前批次的密度裁块数.
            - P: 当前批次中已选受体原子数.

        输入:
            - density_input: (B, 56, 80, 80, 80), 每个裁块的密度特征; 后三轴依次为 Z、Y、X.
            - receptor_feature: (P, 50), 已选受体原子的特征; 与 receptor_pos_xyz 和 receptor_batch 第一维逐原子对齐.
            - receptor_pos_xyz: (P, 3), 已选受体原子的模型局部坐标; 最后一维依次为 X、Y、Z, 单位 Å.
            - receptor_batch: 整型, (P,), 每个已选受体原子所属裁块的编号; 数值索引 density_input 第一维.
            - density_origin_xyz: (B, 3), 每个裁块零号体素角点的模型局部坐标; 最后一维依次为 X、Y、Z, 单位 Å.
            - density_basis_xyz: (B, 3, 3), 每个裁块的体素基向量; 第二维依次对应体素 X、Y、Z 轴, 最后一维依次为模型局部 X、Y、Z 位移, 单位 Å.

        返回:
            - grid: (B, 106, 80, 80, 80), 前 56 通道为 density_input, 后 50 通道为落入各体素的受体原子特征和; 后三轴依次为 Z、Y、X.

        越界受体原子不写入 grid. 例如两个受体原子落入同一体素时, 它们的 50 维特征逐通道相加.
        """
        return scatter_receptor_features(
            density_input,
            receptor_feature,
            receptor_pos_xyz,
            receptor_batch,
            density_origin_xyz,
            density_basis_xyz,
        )

    def _encode_chunk(
        self,
        block_index: int,
        grid: torch.Tensor,
        positions_xyz: torch.Tensor,
        batch_index: torch.Tensor,
        density_origin_xyz: torch.Tensor,
        density_basis_xyz: torch.Tensor,
    ) -> torch.Tensor:
        """用当前去噪块的配体坐标提取局部体素块, 返回逐原子 64 维条件.

        形状符号:
            - B: 当前批次的密度裁块数.
            - M: 当前连续分块中的配体原子数, 不大于 self.chunk_size.

        输入:
            - block_index: int, 当前去噪块编号; 数值索引 self.encoders, 取值范围为 [0, 6).
            - grid: (B, 106, 80, 80, 80), build_grid 构造的固定网格; 后三轴依次为 Z、Y、X.
            - positions_xyz: (M, 3), 当前分块配体原子的模型局部坐标; 最后一维依次为 X、Y、Z, 单位 Å.
            - batch_index: 整型, (M,), 每个配体原子所属裁块的编号; 数值索引 grid 第一维.
            - density_origin_xyz: (B, 3), 每个裁块零号体素角点的模型局部坐标; 最后一维依次为 X、Y、Z, 单位 Å.
            - density_basis_xyz: (B, 3, 3), 每个裁块的体素基向量; 第二维依次对应体素 X、Y、Z 轴, 最后一维依次为模型局部 X、Y、Z 位移, 单位 Å.

        返回:
            - condition: (M, 64), 当前去噪块编码的局部密度与受体条件; 与 positions_xyz 第一维逐原子对齐.

        每个配体原子围绕未截断的 home 索引读取 11³ 局部块, 超出 grid 空间边界的位置补零.
        """
        # int64, (M, 3), 当前分块配体原子在所属裁块中的 ZYX 体素索引, 可越界.
        home_zyx = positions_to_home_zyx(positions_xyz, batch_index, density_origin_xyz, density_basis_xyz)
        # (M, 106, 11, 11, 11), 各原子以实际 home 为中心的局部密度与受体网格, 越界体素为零.
        cube = gather_voxel_cube(grid, home_zyx, batch_index)
        # (M, 64), 由当前去噪块专属卷积编码器生成的逐原子条件.
        return self.encoders[block_index](cube)

    def forward(
        self,
        block_index: int,
        node_feature: torch.Tensor,
        ligand_pos_xyz: torch.Tensor,
        ligand_batch: torch.Tensor,
        grid: torch.Tensor,
        density_origin_xyz: torch.Tensor,
        density_basis_xyz: torch.Tensor,
    ) -> torch.Tensor:
        """在当前去噪块节点更新后、边和坐标更新前调制配体原子特征.

        形状符号:
            - B: 当前批次的密度裁块数.
            - N: 当前批次的配体原子总数.
            - M: 一次局部卷积处理的连续配体原子数, 不大于 self.chunk_size.

        输入:
            - block_index: int, 当前去噪块编号; 数值索引 self.encoders 和 self.film_layers, 取值范围为 [0, 6).
            - node_feature: (N, node_dim), 当前去噪块完成节点更新后的配体原子特征.
            - ligand_pos_xyz: (N, 3), 与 node_feature 第一维逐原子对齐的模型局部坐标; 最后一维依次为 X、Y、Z, 单位 Å.
            - ligand_batch: 整型, (N,), 每个配体原子所属裁块的编号; 数值索引 grid 第一维.
            - grid: (B, 106, 80, 80, 80), build_grid 构造的固定密度与受体特征网格; 后三轴依次为 Z、Y、X.
            - density_origin_xyz: (B, 3), 每个裁块零号体素角点的模型局部坐标; 最后一维依次为 X、Y、Z, 单位 Å.
            - density_basis_xyz: (B, 3, 3), 每个裁块的体素基向量; 第二维依次对应体素 X、Y、Z 轴, 最后一维依次为模型局部 X、Y、Z 位移, 单位 Å.

        返回:
            - combined_feature: (N, node_dim), 当前去噪块的局部条件调制后特征; 与 node_feature 第一维逐原子对齐.

        配体原子按原顺序切成若干连续分块. 例如 N=5000 且 self.chunk_size=4096 时, 两个分块分别包含 4096 和 904 个原子. 训练模式通过 checkpoint 在反向传播时重算卷积激活, 推理模式直接保留前向结果. 所有分块按原顺序拼接后经过共享 LayerNorm, 再由当前去噪块专属 FiLMPlus 调制 node_feature.
        """
        # list[torch.Tensor], 每个元素是一个连续配体原子分块的 (M, 64) 条件; 列表顺序与 node_feature 第一维一致.
        condition_parts = []
        for start in range(0, len(ligand_pos_xyz), self.chunk_size):
            # int, 当前连续分块在 ligand_pos_xyz 和 ligand_batch 第一维上的右开区间终点; [start:stop] 最多包含 self.chunk_size 个原子.
            stop = min(start + self.chunk_size, len(ligand_pos_xyz))
            # tuple[torch.Tensor, ...], 依次保存 grid、ligand_pos_xyz[start:stop]、ligand_batch[start:stop]、density_origin_xyz 和 density_basis_xyz, 供当前分块的 _encode_chunk 调用.
            arguments = (
                grid,
                ligand_pos_xyz[start:stop],
                ligand_batch[start:stop],
                density_origin_xyz,
                density_basis_xyz,
            )
            if self.training:
                # 训练时只保留重算卷积所需输入; 反向传播时通过当前去噪块编码器重计算本分块的激活.
                encoder = partial(self._encode_chunk, block_index)
                condition = checkpoint(encoder, *arguments, use_reentrant=False)
            else:
                condition = self._encode_chunk(block_index, *arguments)
            # (M, 64), 当前 [start:stop] 配体原子的局部密度与受体条件; 与 node_feature[start:stop] 逐原子对齐.
            condition_parts.append(condition)
        # (N, 64), 将 condition_parts 中各 (M, 64) 张量沿配体原子轴 dim=0 拼接, 再逐原子归一化 64 个条件通道.
        condition = self.output_norm(torch.cat(condition_parts, dim=0))
        # (N, node_dim), 用当前去噪块的独立 FiLMPlus 将条件逐原子作用于已更新的节点特征.
        return self.film_layers[block_index](node_feature, condition)
