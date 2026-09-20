"""读取当前定位条件的 80³ 源密度裁块, 并构造固定 56 通道.

主要入口 load_density_input 返回 density_input、density_origin、density_basis、density_start_zyx, 供 PyG 沿首维拼批; 本模块不写文件.
源地图与完整原始受体坐标只读缓存, 包括被受体图排除的 UNK 原子; 通道始终在当前裁块上计算.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from docking.density_channels import DensityChannelConfig, build_density_channels


CROP_SIZE = 80


@lru_cache(maxsize=16)
def read_density_source(root, pdb_id):
    """读取一个 PDB 的实验/模拟地图、共享几何和完整受体坐标.

    输入参数:
        - root: str, 派生资产根目录, 其中 density/<pdb_id> 存放地图, parse/<pdb_id> 存放受体.
        - pdb_id: str, 小写结构编号, 如 6d03.

    返回元组按以下顺序排列:
        - exp: 只读 mmap 数组, (D, H, W), 实验密度; D、H、W 分别对应源数组 Z、Y、X 轴.
        - sim: 只读 mmap 数组, (D, H, W), 与 exp 逐体素对齐的模拟密度.
        - spacing: float32, (3,), 源 XYZ 体素尺寸, 单位 Å; 如 [1.0, 1.0, 1.0], 实际值来自元数据.
        - origin: float32, (3,), 源地图边界角点的世界 XYZ 坐标, 单位 Å; 不是首体素中心.
        - coordinates: float32, (N, 3), receptor_tokens.npz 中全部 N 个受体原子的世界 XYZ 坐标, 单位 Å; 不按残基类型或口袋筛选.
    """
    directory = Path(root) / 'density' / pdb_id
    grids = []
    geometries = []
    for name in ('exp', 'sim'):
        grid = np.load(directory / f'{name}.npy', mmap_mode='r', allow_pickle=False)
        with np.load(directory / f'{name}.npz', allow_pickle=False) as metadata:
            geometry = (np.asarray(metadata['voxel_size'], dtype=np.float32), np.asarray(metadata['origin'], dtype=np.float32))
        grids.append(grid[0])  # [1, D, H, W] -> [D, H, W], 去掉源地图的单通道轴, 保留只读映射.
        geometries.append(geometry)
    if grids[0].shape != grids[1].shape or any(not np.array_equal(a, b) for a, b in zip(*geometries)):
        raise ValueError(f'{pdb_id}: exp/sim源形状或几何不同。')
    if min(grids[0].shape) < CROP_SIZE:
        raise ValueError(f'{pdb_id}: 源密度形状{grids[0].shape}不足{CROP_SIZE}³。')
    with np.load(Path(root) / 'parse' / pdb_id / 'receptor_tokens.npz', allow_pickle=False) as receptor:
        coordinates = np.asarray(receptor['coords'], dtype=np.float32)
    return grids[0], grids[1], *geometries[0], coordinates


# ================================================================================================
def load_density_input(root, pdb_id, query_center_xyz, model_center_xyz):
    """按实际定位中心直接切片, 返回一个配体实例的密度输入和局部几何.

    输入参数:
        - root: str 或 Path, 含 density 与 parse 子目录的派生资产根目录.
        - pdb_id: str, 当前实例所属结构编号, 如 6d03.
        - query_center_xyz: (3,), 用于居中裁块的世界 XYZ 坐标, 单位 Å; 中心模式为实际给定中心, 包络模式为真实配体重原子质心.
        - model_center_xyz: (3,), 既定模型原点的世界 XYZ 坐标, 单位 Å; 中心模式为实际给定中心, 包络模式为实际输入受体原子的算术均值.

    返回字典:
        - density_input: float32, (1, 56, 80, 80, 80), 首维用于拼批, 后三轴为 ZYX; 通道按运算、归一化、后处理顺序排列.
        - density_origin: float32, (1, 3), 实际裁块角点减去模型原点的局部 XYZ 坐标, 单位 Å.
        - density_basis: float32, (1, 3, 3), 三行依次为一个源 X、Y、Z 体素步长在模型坐标系中的向量, 单位 Å; 初值为 diag(spacing).
        - density_start_zyx: int64, (1, 3), 实际裁块在源数组中的起点索引, 如 [[10, 12, 8]].

    裁块起点越界时向地图内移动, 不补零、不改变模型原点. 局部体素中心为 density_origin + (i_xyz + 0.5) @ density_basis.
    对输入施加刚体变换时, 调用方须同时变换 density_origin 并旋转 density_basis, 保持 density_input 数值不变.
    """
    exp, sim, spacing, origin, receptor = read_density_source(str(root), pdb_id)
    query = np.asarray(query_center_xyz, dtype=np.float32).reshape(3)
    center = np.asarray(model_center_xyz, dtype=np.float32).reshape(3)
    # int64, (3,), 从世界 XYZ 重排为源数组 ZYX；80个源体素对应半宽40。
    requested = np.rint(((query - origin) / spacing)[::-1] - CROP_SIZE // 2).astype(np.int64)
    start = np.clip(requested, 0, np.asarray(exp.shape) - CROP_SIZE)
    region = tuple(slice(int(s), int(s) + CROP_SIZE) for s in start)
    corner = origin + start[::-1].astype(np.float32) * spacing  # float32, (3,), 实际裁块边界角点的世界 XYZ 坐标, 单位 Å.
    local = (receptor - corner) / spacing
    inside = np.all((local >= 0) & (local < CROP_SIZE), axis=1)  # bool, (N,), 标记完整受体中落入此裁块的原子.
    home = np.floor(local[inside]).astype(np.int64)  # int64, (K, 3), K 个块内受体原子所在体素的 XYZ 索引.
    mask = np.zeros((CROP_SIZE, CROP_SIZE, CROP_SIZE), dtype=bool)
    mask[home[:,2], home[:,1], home[:,0]] = True  # bool, (80,80,80), 仅标记含受体原子的体素; 不做半径膨胀.
    channels = build_density_channels(exp[region], sim[region], DensityChannelConfig(), mask)
    return {
        'density_input': torch.from_numpy(channels[None]),
        'density_origin': torch.from_numpy((corner - center).astype(np.float32)[None]),
        'density_basis': torch.from_numpy(np.diag(spacing)[None]),
        'density_start_zyx': torch.from_numpy(start[None]),
    }
