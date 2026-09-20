"""实现 ``local_cov`` 的固定受体体素网格、11³ 局部读取和逐层 FiLMPlus 调制.

主要入口是 :class:`LocalCovConditioner`. ``build_grid`` 先把当前 RA 口袋内的
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


DENSITY_CHANNELS = 56
RECEPTOR_CHANNELS = 50
GRID_SIZE = 80
CUBE_SIZE = 11
CONDITION_DIM = 64


def positions_to_home_zyx(
    positions_xyz: torch.Tensor,
    batch_index: torch.Tensor,
    density_origin_xyz: torch.Tensor,
    density_basis_xyz: torch.Tensor,
) -> torch.Tensor:
    """把模型局部 XYZ 坐标映射为所属 80³ 裁块的离散 ZYX home 体素.

    ``density_origin_xyz`` 是裁块角点, ``density_basis_xyz`` 的三行是源 X、Y、Z
    体素轴在模型局部坐标中的步进向量. 几何计算固定使用 float32, home 对连续
    体素 XYZ 坐标取 ``floor``, 不把越界结果截断到裁块边界.
    """
    # (N,), 每个待查询原子所属实例; 用于选择对应裁块角点和基向量.
    batch_index = batch_index.long()
    # (B,3,3), 把模型局部XYZ位移反解到连续体素XYZ坐标.
    inverse_basis = torch.linalg.inv(density_basis_xyz.float())
    # (N,3), 相对各实例80³裁块边界角点的模型局部XYZ位移, 单位Å.
    local_xyz = positions_xyz.float() - density_origin_xyz.float()[batch_index]
    # (N,3), 连续体素XYZ角点坐标; 行向量右乘基矩阵的逆.
    voxel_xyz = torch.bmm(local_xyz[:, None, :], inverse_basis[batch_index]).squeeze(1)
    return torch.floor(voxel_xyz).long()[:, [2, 1, 0]]


def scatter_receptor_features(
    density_input: torch.Tensor,
    receptor_feature: torch.Tensor,
    receptor_pos_xyz: torch.Tensor,
    receptor_batch: torch.Tensor,
    density_origin_xyz: torch.Tensor,
    density_basis_xyz: torch.Tensor,
) -> torch.Tensor:
    """把已选 RA 受体原子的 50 维特征求和散射, 并与 56 维密度拼成固定网格.

    输入 ``density_input`` 形状为 ``(B,56,80,80,80)``; 受体特征和坐标分别为
    ``(P,50)`` 与 ``(P,3)``. home 越界的受体原子被忽略, 同一体素内的原子逐
    通道求和. 返回 ``(B,106,80,80,80)``, 空间轴顺序为 ZYX.
    """
    # B和三个空间轴来自本批固定密度裁块; 空间维必须严格为80³.
    batch_size, density_channels, dim_z, dim_y, dim_x = density_input.shape
    if (density_channels, dim_z, dim_y, dim_x) != (DENSITY_CHANNELS, GRID_SIZE, GRID_SIZE, GRID_SIZE):
        raise ValueError(f"density_input形状必须为(B,56,80,80,80), 实际为{tuple(density_input.shape)}. ")

    # (B,106,80,80,80), 前56通道原样复制, 后50通道接收当前RA口袋原子求和.
    grid = density_input.new_zeros((batch_size, DENSITY_CHANNELS + RECEPTOR_CHANNELS, dim_z, dim_y, dim_x))
    grid[:, :DENSITY_CHANNELS].copy_(density_input)
    # (P,3), 当前所选受体原子在各自80³裁块中的离散ZYX home, 可含越界值.
    home_zyx = positions_to_home_zyx(receptor_pos_xyz, receptor_batch, density_origin_xyz, density_basis_xyz)
    # (P,), 只让实际home落在裁块内的受体原子参与硬散射, 不截断home.
    valid = (
        (home_zyx[:, 0] >= 0) & (home_zyx[:, 0] < dim_z)
        & (home_zyx[:, 1] >= 0) & (home_zyx[:, 1] < dim_y)
        & (home_zyx[:, 2] >= 0) & (home_zyx[:, 2] < dim_x)
    )
    if valid.any():
        batch_valid = receptor_batch.long()[valid]
        z_index, y_index, x_index = home_zyx[valid].unbind(dim=1)
        channel_index = torch.arange(RECEPTOR_CHANNELS, device=grid.device, dtype=torch.long)[None, :]
        # (P_valid,50), 直接索引B、通道、Z、Y、X五维连续存储; 同索引由scatter_add_求和.
        flat_index = (
            ((((batch_valid[:, None] * (DENSITY_CHANNELS + RECEPTOR_CHANNELS)
                 + DENSITY_CHANNELS + channel_index) * dim_z + z_index[:, None])
               * dim_y + y_index[:, None]) * dim_x + x_index[:, None])
        )
        grid.view(-1).scatter_add_(
            0,
            flat_index.reshape(-1),
            receptor_feature[valid].to(device=grid.device, dtype=grid.dtype).reshape(-1),
        )
    return grid


def gather_voxel_cube(
    grid: torch.Tensor,
    center_zyx: torch.Tensor,
    batch_index: torch.Tensor,
    cube_size: int = CUBE_SIZE,
) -> torch.Tensor:
    """以实际 home 为中心读取局部块, 并把超出 80³ 裁块的位置置零.

    返回张量形状为 ``(N,C,cube_size,cube_size,cube_size)``. 函数只对安全读取
    使用 ``clamp``; 越界位置随后乘零, 不移动 home, 也不返回有效体素掩码.
    """
    # k为奇数; 三个单轴偏移均为[-k//2, ..., k//2].
    radius = cube_size // 2
    dim_z, dim_y, dim_x = grid.shape[2:]
    offsets = torch.arange(cube_size, device=grid.device, dtype=torch.long) - radius
    center_zyx = center_zyx.to(device=grid.device, dtype=torch.long)
    batch_index = batch_index.to(device=grid.device, dtype=torch.long)

    z_raw = center_zyx[:, 0, None] + offsets
    y_raw = center_zyx[:, 1, None] + offsets
    x_raw = center_zyx[:, 2, None] + offsets
    # (N,k,k,k), 标出每个实际home邻域中仍位于固定裁块内的位置.
    valid = (
        ((z_raw >= 0) & (z_raw < dim_z))[:, :, None, None]
        & ((y_raw >= 0) & (y_raw < dim_y))[:, None, :, None]
        & ((x_raw >= 0) & (x_raw < dim_x))[:, None, None, :]
    )
    # 高级索引先得到(N,k,k,k,C), 再恢复为卷积读取的(N,C,k,k,k).
    cube = grid[
        batch_index[:, None, None, None],
        :,
        z_raw.clamp(0, dim_z - 1)[:, :, None, None],
        y_raw.clamp(0, dim_y - 1)[:, None, :, None],
        x_raw.clamp(0, dim_x - 1)[:, None, None, :],
    ].permute(0, 4, 1, 2, 3).contiguous()
    return cube * valid[:, None].to(cube.dtype)


class LocalCubeEncoder(nn.Module):
    """把每个配体原子的 106 通道 11³ 局部块编码为 64 维条件向量.

    前向输入:
        - cube: ``(M,106,11,11,11)``, 当前原子分块的局部体素特征; 空间轴顺序为 ZYX.

    前向输出:
        - condition: ``(M,64)``, 与输入配体原子顺序逐项对齐的局部密度条件.
    """

    def __init__(self) -> None:
        """建立三层固定卷积、全局平均池化和 64 维线性投影."""
        super().__init__()
        # 两次stride-2把11³依次变为6³、3³; 最后一层保持3³.
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
        # 每个局部块独立汇聚空间轴, 再投影为FiLMPlus共用的64维条件宽度.
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Linear(64, CONDITION_DIM)

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        """返回与输入原子顺序一致的 ``(M,64)`` 局部密度条件. """
        return self.projection(self.pool(self.convolutions(cube)).flatten(1))


class FiLMPlusCombine(nn.Module):
    """用局部条件生成逐通道缩放和偏移, 并直接调制配体节点特征.

    构造参数:
        - node_dim: int, 配体节点特征宽度; 正式模型为 320.

    前向输入:
        - node_feature: ``(N,node_dim)``, 当前去噪块完成节点更新后的配体特征.
        - condition: ``(N,64)``, 与配体原子顺序逐项对齐的局部密度条件.

    前向输出:
        - combined: ``(N,node_dim)``, 执行 ``node_feature * (1 + gamma) + beta`` 后的节点特征.
    """

    def __init__(self, node_dim: int) -> None:
        """建立条件生成器, 并把末层权重与偏置初始化为零."""
        super().__init__()
        self.node_dim = int(node_dim)
        # (N,64)->(N,640), 前320维为gamma, 后320维为beta.
        self.generator = nn.Sequential(
            nn.Linear(CONDITION_DIM, self.node_dim),
            nn.SiLU(),
            nn.Linear(self.node_dim, 2 * self.node_dim),
        )
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)

    def forward(self, node_feature: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """执行 ``h * (1 + gamma) + beta``; 初始化时严格返回原节点特征. """
        gamma, beta = self.generator(condition).chunk(2, dim=-1)
        return node_feature * (1.0 + gamma) + beta


class LocalCovConditioner(nn.Module):
    """管理六套局部卷积、一套共享 LayerNorm 和六套独立 FiLMPlus.

    构造参数:
        - node_dim: int, 配体节点特征宽度; 正式模型为 320.
        - num_blocks: int, 去噪块数量; ``local_cov`` 固定为 6.
        - chunk_size: int, 单次抽取和卷积的配体原子数; 只允许 4096 或 2048.

    ``build_grid`` 返回 ``(B,106,80,80,80)`` 固定网格. ``forward`` 接收一个去噪块
    的当前 ``(N,3)`` 配体局部 XYZ 坐标, 按原子分块读取 11³ 邻域, 返回
    ``(N,node_dim)`` 调制后节点特征. 激活重计算只在训练模式启用.
    """

    def __init__(self, node_dim: int, num_blocks: int, chunk_size: int) -> None:
        """按固定数量建立独立卷积与 FiLMPlus, 并建立共享输出归一化."""
        super().__init__()
        if num_blocks != 6:
            raise ValueError(f"local_cov固定需要6个去噪块, 实际为{num_blocks}. ")
        if chunk_size not in (4096, 2048):
            raise ValueError(f"local_cov的chunk_size只允许4096或2048, 实际为{chunk_size}. ")
        self.chunk_size = int(chunk_size)
        # 六套卷积不共享; 只有其输出进入同一套可学习LayerNorm.
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
        """构造六个去噪块共同读取的 ``(B,106,80,80,80)`` 固定网格. """
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
        """按当前层坐标重新计算 home、抽取11³局部块并运行该层卷积编码器. """
        home_zyx = positions_to_home_zyx(positions_xyz, batch_index, density_origin_xyz, density_basis_xyz)
        cube = gather_voxel_cube(grid, home_zyx, batch_index)
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
        """在当前去噪块节点更新后、边和坐标更新前执行局部密度调制. """
        # 各分块按原子原顺序处理, 拼接后仍与node_feature第一维严格对齐.
        condition_parts = []
        for start in range(0, len(ligand_pos_xyz), self.chunk_size):
            stop = min(start + self.chunk_size, len(ligand_pos_xyz))
            arguments = (
                grid,
                ligand_pos_xyz[start:stop],
                ligand_batch[start:stop],
                density_origin_xyz,
                density_basis_xyz,
            )
            if self.training:
                encoder = partial(self._encode_chunk, block_index)
                condition = checkpoint(encoder, *arguments, use_reentrant=False)
            else:
                condition = self._encode_chunk(block_index, *arguments)
            condition_parts.append(condition)
        condition = self.output_norm(torch.cat(condition_parts, dim=0))
        return self.film_layers[block_index](node_feature, condition)
