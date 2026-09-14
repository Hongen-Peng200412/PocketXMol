"""沿Dataset、原特征化、原dock噪声和采样输出核对T0中心契约, 不读取held-out实例."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from rdkit import Chem
from torch_geometric.transforms import Compose

from docking.assets import read_receptor
from docking.dataset import OccurrenceDataset
from docking.sampling import sample_occurrence
from scripts.train_pl import DataModule
from utils.misc import make_config
from utils.sample_noise import get_sample_noiser
from utils.transforms import ConfTransform, FeaturizeMol
from test_docking_data import prepared_data


def center_config(experiment, assets):
    config = make_config(str(Path(__file__).resolve().parents[1] / f'configs/docking/{experiment}.yml'))
    if assets is not None:
        config.data.dataset.update(root=assets.root, derived_root=assets.derived_root, manifest_root=assets.manifest_root)
    config.train.update(num_workers=0, persistent_workers=False)
    return config


def capture_noise(noiser, monkeypatch):
    """只在原add_noise返回处检查公式, 早于同构重分配和固定字段恢复; 返回每次实际噪声调用的证据."""
    original = noiser.add_noise
    calls = []

    def checked(node_type, node_pos, halfedge_type, batch, from_prior, level_dict):
        clean = node_pos.clone()
        receptor = batch.pocket_pos.clone()
        graph = batch.node_type_batch if 'node_type_batch' in batch else torch.zeros_like(node_type)
        sizes = torch.bincount(graph)[graph]
        # 复制随机状态求原GaussianExplodePrior的同一份epsilon; 不消耗实际噪声器的随机流.
        with torch.random.fork_rng(devices=[]):
            gaussian = torch.randn_like(clean) * sizes.sqrt()[:, None].clamp(min=1)
        strength = 1 - level_dict['pos']
        expected = gaussian if from_prior else clean + strength[:, None] * gaussian
        shift = torch.zeros_like(clean)
        result = original(node_type, node_pos, halfedge_type, batch, from_prior=from_prior, level_dict=level_dict)
        torch.testing.assert_close(result['pos'], expected, rtol=1e-5, atol=3e-6)
        torch.testing.assert_close(node_pos, clean, rtol=0, atol=0)
        torch.testing.assert_close(batch.pocket_pos, receptor, rtol=0, atol=0)
        calls.append(dict(clean=clean, graph=graph.clone(), level=level_dict['pos'].clone(), from_prior=from_prior, shift=shift, noisy=result['pos'].clone(), center=batch.pocket_center.clone(), receptor=receptor))
        return result

    monkeypatch.setattr(noiser, 'add_noise', checked)
    return calls


@pytest.mark.parametrize('experiment,protocol', [('B-C-T0-RA', 'C0'), ('B-E-T0-RA', 'E')])
def test_supervised_protocol_from_existing_mechanism(prepared_data, experiment, protocol):
    module = DataModule(center_config(experiment, prepared_data))
    module.setup('fit')
    for loader in (module.train_dataloader(), module.val_dataloader()):
        dataset = loader.dataset
        assert dataset.protocol == protocol
        data = dataset[0]
        if protocol == 'C0':
            torch.testing.assert_close(data.node_pos.mean(0), torch.zeros(3), rtol=0, atol=1e-6)
        elif protocol == 'E':
            torch.testing.assert_close(data.pocket_pos.mean(0), torch.zeros(3), rtol=0, atol=1e-6)


@pytest.mark.parametrize('protocol', ['C0', 'C5'])
def test_boundary_origin_and_noise(prepared_data, monkeypatch, protocol):
    config = center_config('B-C-T0-RA', prepared_data)
    parse_dir = Path(config.data.dataset.root) / 'parse/train_demo'
    with np.load(parse_dir / 'ligand_coords.npz') as archive:
        coordinates = {key: archive[key] for key in archive.files}
    # 只平移本测试的三原子构造配体, 使g精确为0; 避免float32的g加法把15/17 Å等号边界推到阈值外.
    world_truth = coordinates['coords_0'] - coordinates['coords_0'].mean(0)
    coordinates['coords_0'] = world_truth
    np.savez_compressed(parse_dir / 'ligand_coords.npz', **coordinates)
    center = world_truth.mean(0)
    np.testing.assert_array_equal(center, np.zeros(3))
    # 相对g: C/O残基质量中心>15而算术均值14.9<15; 16.6的残基位于真中心阈值外, 另保留严格等号边界.
    relative = np.array([[14., 0, 0], [15.8, 0, 0], [-14., 0, 0], [15., 0, 0], [17., 0, 0], [0., 0, 0], [16.6, 0, 0]], dtype=np.float32)
    np.savez_compressed(parse_dir / 'receptor_tokens.npz', coords=relative + center, element=np.array([6, 8, 6, 6, 6, 6, 6]), res_type=np.zeros(7, dtype=np.uint8), res_index=np.array([0, 0, 1, 2, 3, 4, 5]), is_backbone=np.ones(7, dtype=bool), atom_name=np.array(['CA', 'O', 'CA', 'CA', 'CA', 'CA', 'CA'], dtype='S4'))
    read_receptor.cache_clear()
    module = DataModule(config)
    module.setup('fit')
    dataset = module.train_dataloader().dataset
    if protocol == 'C5':
        # C5只代表给定中心条件; 用构造清单中的明确向量验收, 不修改任何正式冻结向量.
        dataset = OccurrenceDataset(config.data.dataset, 'train', module.transforms, 'RA', 'C5', False)
        dataset.records[0]['center_offset_xyz_A'] = [2., 0., 0.]
    # 用不可调用的随机生成器证明T0装配不抽取方向或半径.
    dataset.rng = Mock(spec=np.random.Generator)
    noiser = module.transforms.transforms[-1].noiser_dict['dock']
    monkeypatch.setattr(noiser, 'sample_level', lambda step, data: {'pos': torch.full((data.num_nodes,), .35)})
    calls = capture_noise(noiser, monkeypatch)
    torch.manual_seed(47)
    sample = dataset[0]
    delta = torch.tensor([2., 0, 0]) if protocol == 'C5' else torch.zeros(3)
    selected = [0, 1, 3, 5, 6] if protocol == 'C5' else [2, 5]
    torch.testing.assert_close(sample.pocket_center[0], torch.tensor(center) + delta)
    torch.testing.assert_close(sample.pocket_pos, torch.tensor(relative[selected]) - delta)
    torch.testing.assert_close(sample.node_pos, torch.tensor(world_truth - center) - delta)
    torch.testing.assert_close(sample.node_pos.mean(0), -delta, atol=1e-6, rtol=0)
    torch.testing.assert_close(calls[0]['shift'], torch.zeros_like(sample.node_pos))
    assert dataset.rng.normal.call_count == dataset.rng.uniform.call_count == 0
    # 相同给定中心与纯蛋白口袋下, RA特征化与原FeaturizePocket逐字段一致; 独立官方版本验收在test_docking_official.py.
    original_noise_config = deepcopy(config.noise)
    original_noise = get_sample_noiser(original_noise_config, 12, 6, mode='train')
    monkeypatch.setattr(original_noise.noiser_dict['dock'], 'sample_level', lambda step, data: {'pos': torch.full((data.num_nodes,), .35)})
    original_transforms = Compose(module.transforms.transforms[:-1] + [original_noise])
    reference = OccurrenceDataset(config.data.dataset, 'train', original_transforms, 'protein', protocol, False)
    reference.records[0]['center_offset_xyz_A'] = [2., 0., 0.]
    torch.manual_seed(47)
    expected = reference[0]
    for key in ('pocket_center', 'pocket_pos', 'pocket_atom_feature', 'pocket_knn_edge_index', 'node_pos', 'node_type', 'halfedge_type', 'node_in', 'pos_in', 'halfedge_in', 'fixed_pos'):
        torch.testing.assert_close(sample[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize('protocol', ['C0', 'C5'])
def test_sampling_condition_mechanism_and_world_restore(prepared_data, tmp_path, monkeypatch, protocol):
    config = center_config('B-C-T0-RA', prepared_data)
    featurizer = FeaturizeMol(config.transforms.featurizer)
    task = ConfTransform(config.transforms.task.individual[0], mode='test')
    dataset = OccurrenceDataset(config.data.dataset, 'validation', Compose([featurizer, task]), 'RA', protocol, False)
    fixed = dataset[0]
    root = Path(__file__).resolve().parents[1]
    noise_config = make_config(str(root / 'configs/sample/test/dock_poseboff/base.yml')).noise
    noise_config.num_steps = 3
    noiser = get_sample_noiser(noise_config, 12, 6, mode='sample', device='cpu', ref_config=config.noise)
    calls = capture_noise(noiser, monkeypatch)
    predictions = []

    def forward(batch):
        assert torch.count_nonzero(batch.gt_node_pos) == 0
        torch.testing.assert_close(batch.pocket_center, fixed.pocket_center.expand(2, 3))
        torch.testing.assert_close(batch.pocket_pos, fixed.pocket_pos.repeat(2, 1))
        # 两个候选分别保留非零质心, 验证后续T0不重新居中或最终对齐.
        per_molecule = torch.tensor([[9., 3., -2.], [-8., -4., 1.]])
        positions = per_molecule[batch.node_type_batch].clone()
        positions[:, 0] += torch.arange(batch.num_nodes) % fixed.num_nodes
        predictions.append(positions.clone())
        return dict(pred_node=torch.nn.functional.one_hot(batch.node_type, 12).float(), pred_pos=positions, pred_halfedge=torch.nn.functional.one_hot(batch.halfedge_type, 6).float(), confidence_pos=torch.ones(batch.num_nodes, 1), confidence_node=torch.ones(batch.num_nodes, 1), confidence_halfedge=torch.ones(len(batch.halfedge_type), 1))

    sampling = deepcopy(config.data)
    sampling.update(model_name='center_contract', split='validation', output_root=str(tmp_path / 'sampling'), num_candidates=2, num_steps=3, batch_size=2, device='cpu')
    result = sample_occurrence(dataset, 0, forward, noiser, featurizer, sampling, protocol)
    assert result['success_count'] == 2, result
    assert len(calls) == 3 and calls[0]['from_prior'] and not calls[1]['from_prior']
    assert torch.count_nonzero(calls[0]['clean']) == 0
    assert calls[0]['noisy'][:fixed.num_nodes].mean(0).norm() > .01  # 单个先验候选不强制质心归零.
    torch.testing.assert_close(calls[1]['clean'], predictions[0])
    assert all(torch.count_nonzero(call['shift']) == 0 for call in calls)
    directory = Path(sampling.output_root) / 'validation' / protocol / 'val_demo/0'
    poses = list(Chem.SDMolSupplier(str(directory / 'poses.sdf')))
    for index, molecule in enumerate(poses):
        local = predictions[-1][index * fixed.num_nodes:(index + 1) * fixed.num_nodes]
        np.testing.assert_allclose(molecule.GetConformer().GetPositions(), (local + fixed.pocket_center).numpy(), rtol=0, atol=6e-5)
    # 定位条件不变, 只污染已装配的GT坐标; 整条真实采样链和最终SDF应完全不变.
    original_getitem = OccurrenceDataset.__getitem__

    def changed_truth(self, index):
        data = original_getitem(self, index)
        data.node_pos += 1000.
        data.gt_node_pos -= 2000.
        return data

    monkeypatch.setattr(OccurrenceDataset, '__getitem__', changed_truth)
    sampling.output_root = str(tmp_path / 'changed_truth')
    repeated = sample_occurrence(dataset, 0, forward, noiser, featurizer, sampling, protocol)
    assert repeated['success_count'] == 2
    repeated_dir = Path(sampling.output_root) / 'validation' / protocol / 'val_demo/0'
    assert (repeated_dir / 'poses.sdf').read_bytes() == (directory / 'poses.sdf').read_bytes()
    for first, second in zip(calls[:3], calls[3:]):
        torch.testing.assert_close(first['noisy'], second['noisy'], rtol=0, atol=0)
