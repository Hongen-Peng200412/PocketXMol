"""用构造张量核对 ``local_cov`` 的坐标、边界、卷积与零初始化调制。"""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
import torch

import docking.density as density_module
from models.density import (
    FiLMPlusCombine,
    LocalCovConditioner,
    LocalCubeEncoder,
    gather_voxel_cube,
    positions_to_home_zyx,
    scatter_receptor_features,
)


def test_positions_use_xyz_basis_and_return_unclamped_zyx():
    """home 使用实际 XYZ 基向量求逆、取 floor 后重排为 ZYX，且保留越界值。"""
    origin = torch.tensor([[10.0, -2.0, 5.0]])
    basis = torch.tensor([[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]])
    voxel_xyz = torch.tensor([[2.2, 3.4, 4.1], [-1.2, 0.2, 90.0]])
    positions = origin + voxel_xyz @ basis[0]
    observed = positions_to_home_zyx(positions, torch.zeros(2, dtype=torch.long), origin, basis)
    torch.testing.assert_close(observed, torch.tensor([[4, 3, 2], [90, 0, -2]]))


def test_scatter_uses_selected_receptor_atoms_and_sums_same_home():
    """56维密度原样保留，两个同体素受体原子的50维特征逐通道求和。"""
    density = torch.zeros((1, 56, 80, 80, 80), dtype=torch.float16)
    density[:, 3, 2, 1, 0] = 7
    feature = torch.stack([torch.arange(50), 2 * torch.arange(50), torch.ones(50)]).to(torch.float16)
    positions = torch.tensor([[1.2, 2.1, 3.9], [1.8, 2.9, 3.1], [100.0, 0.0, 0.0]])
    grid = scatter_receptor_features(
        density,
        feature,
        positions,
        torch.zeros(3, dtype=torch.long),
        torch.zeros((1, 3)),
        torch.eye(3).unsqueeze(0),
    )
    torch.testing.assert_close(grid[:, :56], density)
    torch.testing.assert_close(grid[0, 56:, 3, 2, 1], 3 * torch.arange(50).to(torch.float16))
    assert grid[0, 56:, :, :, :].sum() == 3 * torch.arange(50).sum()


def test_scatter_matches_pocket_plus_sum_reference():
    """相同输入下, 50 维受体网格与 Pocket_Plus 硬求和散射逐元素一致."""
    reference_path = Path(__file__).resolve().parents[2] / 'Pocket_Plus/src/model/stage1_embed_head.py'
    if not reference_path.is_file():
        pytest.skip('需要同级 Pocket_Plus 源码执行逐元素来源对照.')
    spec = spec_from_file_location('pocket_plus_stage1_embed_head', reference_path)
    reference_module = module_from_spec(spec)
    spec.loader.exec_module(reference_module)

    torch.manual_seed(7)
    density = torch.zeros((2, 56, 80, 80, 80))
    feature = torch.randn((5, 50))
    continuous_xyz = torch.tensor([
        [1.2, 2.3, 3.4],
        [1.8, 2.9, 3.1],
        [79.9, 0.1, 4.2],
        [-0.1, 3.0, 3.0],
        [80.0, 1.0, 1.0],
    ])
    batch = torch.tensor([0, 0, 1, 1, 1])
    observed = scatter_receptor_features(
        density,
        feature,
        continuous_xyz,
        batch,
        torch.zeros((2, 3)),
        torch.eye(3).repeat(2, 1, 1),
    )[:, 56:]
    expected = reference_module.scatter_to_voxel_grid(
        point_feat=feature,
        atom_coord_local_voxel=continuous_xyz,
        point_batch=batch,
        box_shape_zyx=torch.full((2, 3), 80),
        batch_size=2,
        reduce='sum',
        add_occupancy_channels=False,
    )
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_local_window_zero_fills_without_moving_home():
    """边界窗口保持实际home，块外元素为零；完全远离裁块时返回全零。"""
    grid = torch.zeros((1, 1, 8, 8, 8))
    grid[0, 0, 0, 0, 0] = 9
    near = gather_voxel_cube(grid, torch.tensor([[0, 0, 0]]), torch.tensor([0]))
    assert near.shape == (1, 1, 11, 11, 11)
    assert near[0, 0, 5, 5, 5] == 9
    assert near[0, 0, :5].count_nonzero() == 0
    far = gather_voxel_cube(grid, torch.tensor([[-20, -20, -20]]), torch.tensor([0]))
    assert far.count_nonzero() == 0


def test_encoder_shape_and_film_initial_identity():
    """三层卷积输出64维条件，零初始化FiLMPlus严格保持节点特征。"""
    encoder = LocalCubeEncoder().eval()
    cube = torch.randn((2, 106, 11, 11, 11))
    with torch.no_grad():
        condition = encoder(cube)
    assert condition.shape == (2, 64)
    film = FiLMPlusCombine(320)
    node = torch.randn((2, 320))
    torch.testing.assert_close(film(node, condition), node, rtol=0, atol=0)


def test_conditioner_has_independent_encoders_shared_norm_and_chunk_equivalence(monkeypatch):
    """六层卷积与FiLM参数互不共享，共用一套归一化；分块不改变顺序或数值。"""
    conditioner = LocalCovConditioner(node_dim=8, num_blocks=6, chunk_size=4096).eval()
    assert len({id(module) for module in conditioner.encoders}) == 6
    assert len({id(module) for module in conditioner.film_layers}) == 6
    assert conditioner.output_norm.normalized_shape == (64,)
    with torch.no_grad():
        conditioner.film_layers[0].generator[-1].weight.normal_(std=0.01)
        conditioner.film_layers[0].generator[-1].bias.normal_(std=0.01)

    def lightweight_encode(block_index, grid, positions, batch, origin, basis):
        del block_index, grid, batch, origin, basis
        return positions[:, :1].repeat(1, 64)

    monkeypatch.setattr(conditioner, '_encode_chunk', lightweight_encode)
    node = torch.randn((4097, 8))
    positions = torch.arange(4097 * 3, dtype=torch.float32).reshape(4097, 3)
    batch = torch.zeros(4097, dtype=torch.long)
    grid = torch.empty(1)
    origin = torch.zeros((1, 3))
    basis = torch.eye(3).unsqueeze(0)
    conditioner.chunk_size = 4096
    observed = conditioner(0, node, positions, batch, grid, origin, basis)
    conditioner.chunk_size = 2048
    expected = conditioner(0, node, positions, batch, grid, origin, basis)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_each_block_recomputes_home_from_current_positions_and_reuses_grid(monkeypatch):
    """六层分别使用传入的当前坐标计算 home, 同时读取同一份固定网格."""
    conditioner = LocalCovConditioner(node_dim=8, num_blocks=6, chunk_size=4096).eval()
    captured = []

    def capture_encode(block_index, grid, positions, batch, origin, basis):
        captured.append((block_index, grid.data_ptr(), positions_to_home_zyx(positions, batch, origin, basis).clone()))
        return torch.zeros((len(positions), 64))

    monkeypatch.setattr(conditioner, '_encode_chunk', capture_encode)
    node = torch.randn((2, 8))
    batch = torch.zeros(2, dtype=torch.long)
    grid = torch.zeros((1, 106, 80, 80, 80))
    origin = torch.zeros((1, 3))
    basis = torch.eye(3).unsqueeze(0)
    for block_index in range(6):
        positions = torch.tensor([[block_index + 0.2, 1.2, 2.2], [3.2, block_index + 0.2, 4.2]])
        conditioner(block_index, node, positions, batch, grid, origin, basis)

    assert [item[0] for item in captured] == list(range(6))
    assert {item[1] for item in captured} == {grid.data_ptr()}
    for block_index, _, home_zyx in captured:
        torch.testing.assert_close(home_zyx, torch.tensor([[2, 1, block_index], [4, block_index, 3]]))


def test_density_source_rejects_small_axis_and_crop_only_moves_start_inward(tmp_path, monkeypatch):
    """源图任一轴不足 80 时失败; 合法大图靠边裁取只内缩起点, 不补源裁块."""
    density_directory = tmp_path / 'density/demo'
    receptor_directory = tmp_path / 'parse/demo'
    density_directory.mkdir(parents=True)
    receptor_directory.mkdir(parents=True)
    for name in ('exp', 'sim'):
        np.save(density_directory / f'{name}.npy', np.zeros((1, 79, 80, 81), dtype=np.float32))
        np.savez(density_directory / f'{name}.npz', origin=np.zeros(3, dtype=np.float32), voxel_size=np.ones(3, dtype=np.float32))
    np.savez(receptor_directory / 'receptor_tokens.npz', coords=np.zeros((1, 3), dtype=np.float32))
    density_module.read_density_source.cache_clear()
    with pytest.raises(ValueError, match='不足80³'):
        density_module.read_density_source(str(tmp_path), 'demo')

    exp = np.zeros((90, 100, 110), dtype=np.float32)
    monkeypatch.setattr(
        density_module,
        'read_density_source',
        lambda root, pdb_id: (exp, exp, np.ones(3, dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros((0, 3), dtype=np.float32)),
    )
    monkeypatch.setattr(
        density_module,
        'build_density_channels',
        lambda exp_crop, sim_crop, config, mask: np.lib.stride_tricks.as_strided(
            np.zeros(1, dtype=np.float32),
            shape=(56, *exp_crop.shape),
            strides=(0, 0, 0, 0),
            writeable=True,
        ),
    )
    low = density_module.load_density_input(tmp_path, 'demo', np.array([-50.0, -50.0, -50.0]), np.array([7.0, 8.0, 9.0]))
    high = density_module.load_density_input(tmp_path, 'demo', np.array([200.0, 200.0, 200.0]), np.array([7.0, 8.0, 9.0]))
    torch.testing.assert_close(low['density_start_zyx'], torch.tensor([[0, 0, 0]]))
    torch.testing.assert_close(high['density_start_zyx'], torch.tensor([[10, 20, 30]]))
    assert low['density_input'].shape == high['density_input'].shape == (1, 56, 80, 80, 80)
    torch.testing.assert_close(low['density_origin'], torch.tensor([[-7.0, -8.0, -9.0]]))
    torch.testing.assert_close(high['density_origin'], torch.tensor([[23.0, 12.0, 1.0]]))


def test_activation_checkpoint_matches_direct_forward_and_gradients():
    """激活重计算与直接卷积得到相同前向值和参数梯度。"""
    checkpointed = LocalCovConditioner(node_dim=8, num_blocks=6, chunk_size=4096)
    with torch.no_grad():
        checkpointed.film_layers[0].generator[-1].weight.normal_(std=0.01)
    direct = deepcopy(checkpointed).eval()
    grid_a = torch.randn((1, 106, 11, 11, 11), requires_grad=True)
    grid_b = grid_a.detach().clone().requires_grad_(True)
    node_a = torch.randn((2, 8), requires_grad=True)
    node_b = node_a.detach().clone().requires_grad_(True)
    positions = torch.tensor([[5.2, 5.3, 5.4], [5.8, 5.1, 5.6]])
    batch = torch.zeros(2, dtype=torch.long)
    origin = torch.zeros((1, 3))
    basis = torch.eye(3).unsqueeze(0)
    output_a = checkpointed(0, node_a, positions, batch, grid_a, origin, basis)
    output_b = direct(0, node_b, positions, batch, grid_b, origin, basis)
    torch.testing.assert_close(output_a, output_b, rtol=0, atol=0)
    output_a.square().sum().backward()
    output_b.square().sum().backward()
    torch.testing.assert_close(grid_a.grad, grid_b.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(node_a.grad, node_b.grad, rtol=1e-5, atol=1e-6)
    for parameter_a, parameter_b in zip(checkpointed.parameters(), direct.parameters()):
        if parameter_a.grad is not None or parameter_b.grad is not None:
            torch.testing.assert_close(parameter_a.grad, parameter_b.grad, rtol=1e-5, atol=1e-6)
