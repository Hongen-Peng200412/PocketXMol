"""读取当前定位条件的48³源密度裁块，并构造固定56通道。

主要入口 load_density_input 返回可由 PyG 沿首维拼批的输入和几何；不写文件。
源地图与整个原始受体（包括UNK）只读缓存，通道始终在当前裁块上计算。
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from docking.density_channels import DensityChannelConfig, build_density_channels


@lru_cache(maxsize=16)
def read_density_source(root, pdb_id):
    """返回一个PDB的只读exp/sim、XYZ间距/角点和完整原始受体坐标。"""
    directory = Path(root) / 'density' / pdb_id
    grids = []
    geometries = []
    for name in ('exp', 'sim'):
        grid = np.load(directory / f'{name}.npy', mmap_mode='r', allow_pickle=False)
        with np.load(directory / f'{name}.npz', allow_pickle=False) as metadata:
            geometry = (np.asarray(metadata['voxel_size'], dtype=np.float32), np.asarray(metadata['origin'], dtype=np.float32))
        grids.append(grid[0])
        geometries.append(geometry)
    if grids[0].shape != grids[1].shape or any(not np.array_equal(a, b) for a, b in zip(*geometries)):
        raise ValueError(f'{pdb_id}: exp/sim源形状或几何不同。')
    if min(grids[0].shape) < 48:
        raise ValueError(f'{pdb_id}: 源密度形状{grids[0].shape}不足48³。')
    with np.load(Path(root) / 'parse' / pdb_id / 'receptor_tokens.npz', allow_pickle=False) as receptor:
        coordinates = np.asarray(receptor['coords'], dtype=np.float32)
    return grids[0], grids[1], *geometries[0], coordinates


def load_density_input(root, pdb_id, query_center_xyz, model_center_xyz):
    """返回一个实例的密度字段；坐标为XYZ Å，数组为ZYX。

    density_input: float32 (1,56,48,48,48)，固定运算/归一化/后处理顺序。
    density_origin: float32 (1,3)，裁块角点减模型原点。
    density_basis: float32 (1,3,3)，三行是原XYZ单位体素在当前模型坐标系的向量。
    density_start_zyx: int64 (1,3)，实际源裁块起点，如[[10,12,8]]。
    刚体变换时同时变换origin并旋转basis，不修改density_input。
    """
    exp, sim, spacing, origin, receptor = read_density_source(str(root), pdb_id)
    query = np.asarray(query_center_xyz, dtype=np.float32).reshape(3)
    center = np.asarray(model_center_xyz, dtype=np.float32).reshape(3)
    requested = np.rint(((query - origin) / spacing)[::-1] - 24).astype(np.int64)
    start = np.clip(requested, 0, np.asarray(exp.shape) - 48)
    region = tuple(slice(int(s), int(s) + 48) for s in start)
    corner = origin + start[::-1].astype(np.float32) * spacing
    local = (receptor - corner) / spacing
    inside = np.all((local >= 0) & (local < 48), axis=1)
    home = np.floor(local[inside]).astype(np.int64)
    mask = np.zeros((48,48,48), dtype=bool)
    mask[home[:,2], home[:,1], home[:,0]] = True
    channels = build_density_channels(exp[region], sim[region], DensityChannelConfig(), mask)
    return {
        'density_input': torch.from_numpy(channels[None]),
        'density_origin': torch.from_numpy((corner - center).astype(np.float32)[None]),
        'density_basis': torch.from_numpy(np.diag(spacing)[None]),
        'density_start_zyx': torch.from_numpy(start[None]),
    }
