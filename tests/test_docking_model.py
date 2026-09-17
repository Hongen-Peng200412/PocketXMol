"""检查RA受体编码的真实张量行为, 使用构造输入而不读取测试集."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from easydict import EasyDict
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose

from docking.assets import read_receptor, select_pocket
from docking.dataset import OccurrenceDataset
from models.maskfill import PMAsymDenoiser
from utils.transforms import ConfTransform, FeaturizeMol
from test_docking_data import prepared_data


def assemble_batch(prepared_data, branch, receptor_kind):
    config = EasyDict(yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/docking/B-C-T0-RA.yml').read_text(encoding='utf-8')))
    data_config = EasyDict(deepcopy(dict(prepared_data)))
    data_config.update(pocket_mode='center', knn=32)
    featurizer = FeaturizeMol(config.transforms.featurizer)
    task = ConfTransform(EasyDict(settings=dict(free=1.0), free_no_geometry=True), mode='test')
    dataset = OccurrenceDataset(data_config, 'validation', Compose([featurizer, task]), branch, 'C5', False)
    sample = dataset[0]
    if receptor_kind != 'mixed':
        keep = ~sample.pocket_is_nucleic if receptor_kind == 'protein' else sample.pocket_is_nucleic
        new_index = torch.full((len(keep),), -1, dtype=torch.long)
        new_index[keep] = torch.arange(int(keep.sum()))
        edge = sample.pocket_knn_edge_index
        edge_keep = keep[edge[0]] & keep[edge[1]]
        sample.pocket_knn_edge_index = new_index[edge[:, edge_keep]]
        for key in ['pocket_pos', 'pocket_atom_feature', 'pocket_nucleic_feature', 'pocket_is_nucleic']:
            sample[key] = sample[key][keep]
    samples = [sample.clone(), sample.clone()]
    samples[1].node_pos += torch.tensor([.2, .1, -.1])
    for item in samples:
        item.node_in = item.node_type.clone()
        item.pos_in = item.node_pos + .1 * torch.randn_like(item.node_pos)
        item.halfedge_in = item.halfedge_type.clone()
    return config, Batch.from_data_list(samples, follow_batch=featurizer.follow_batch + ['pocket_pos'], exclude_keys=featurizer.exclude_keys + task.exclude_keys)


@pytest.mark.parametrize('branch', ['RA'])
@pytest.mark.parametrize('receptor_kind', ['protein', 'mixed', 'nucleic'])
def test_receptor_forward_and_backward(prepared_data, branch, receptor_kind):
    config, batch = assemble_batch(prepared_data, branch, receptor_kind)
    config.model.nucleic_branch = branch
    model = PMAsymDenoiser(config.model, 12, 6, 25)
    output = model(batch)
    assert output['pred_pos'].shape == batch.node_pos.shape
    assert output['confidence_pos'].shape == (len(batch.node_pos), 1)
    assert all(torch.isfinite(value).all() for value in output.values())
    # 同时让原坐标头和置信度头进入反传, 检查新增投影不会因索引赋值失去梯度.
    objective = output['pred_pos'].square().mean() + output['confidence_pos'].square().mean()
    objective.backward()
    if receptor_kind != 'protein':
        assert model.nucleic_embedder.weight.grad is not None
        assert torch.isfinite(model.nucleic_embedder.weight.grad).all()
        assert model.nucleic_embedder.weight.grad.abs().sum() > 0


def test_original_protein_behavior_with_shared_ra_encoder(prepared_data):
    config, batch = assemble_batch(prepared_data, 'RA', 'protein')
    config.model.nucleic_branch = None
    original = PMAsymDenoiser(config.model, 12, 6, 25).eval()
    with torch.no_grad():
        expected = original(batch)
    for branch in ['RA']:
        config.model.nucleic_branch = branch
        model = PMAsymDenoiser(config.model, 12, 6, 25).eval()
        missing, unexpected = model.load_state_dict(original.state_dict(), strict=False)
        assert not unexpected and all(name.startswith('nucleic_embedder.') for name in missing)
        with torch.no_grad():
            observed = model(batch)
        for key in expected:
            torch.testing.assert_close(observed[key], expected[key], rtol=1e-6, atol=1e-6)


def test_local_cov_dataset_and_zero_initialized_model_equivalence(prepared_data, monkeypatch):
    """核对80³完整数据链，并证明零初始化local_cov不改变同权重RA＋T0模型输出。"""
    root = Path(__file__).resolve().parents[1]
    local_config = EasyDict(yaml.safe_load((root / 'configs/docking/local_cov-C-T0-RA.yml').read_text(encoding='utf-8')))
    base_config = EasyDict(yaml.safe_load((root / 'configs/docking/B-C-T0-RA.yml').read_text(encoding='utf-8')))
    data_config = EasyDict(deepcopy(dict(prepared_data)))
    data_config.update(pocket_mode='center', knn=32)
    featurizer = FeaturizeMol(local_config.transforms.featurizer)
    task = ConfTransform(EasyDict(settings=dict(free=1.0), free_no_geometry=True), mode='test')
    dataset = OccurrenceDataset(
        data_config,
        'validation',
        Compose([featurizer, task]),
        'RA',
        'C0',
        False,
        density_config=local_config.model.density,
    )
    sample = dataset[0]
    assert sample.density_input.shape == (1, 56, 80, 80, 80)
    assert sample.pocket_density_feature.shape == (len(sample.pocket_pos), 50)
    receptor = read_receptor(Path(data_config.root) / 'parse/val_demo/receptor_tokens.npz')
    world_ligand = sample.node_pos + sample.pocket_center
    pocket_mask = select_pocket(
        receptor,
        world_ligand.numpy(),
        world_ligand.mean(dim=0).numpy(),
        'center',
    )
    expected_feature = torch.from_numpy(np.concatenate([
        receptor['feat'][pocket_mask],
        receptor['is_backbone'][pocket_mask, None],
    ], axis=1).astype(np.float32))
    torch.testing.assert_close(sample.pocket_density_feature, expected_feature)
    torch.testing.assert_close(sample.pocket_pos + sample.pocket_center, torch.from_numpy(receptor['coords'][pocket_mask]))
    assert len(receptor['coords']) == 80
    sample.node_in = sample.node_type.clone()
    sample.pos_in = sample.node_pos + 0.1 * torch.randn_like(sample.node_pos)
    sample.halfedge_in = sample.halfedge_type.clone()
    batch = Batch.from_data_list(
        [sample],
        follow_batch=featurizer.follow_batch + ['pocket_pos'],
        exclude_keys=featurizer.exclude_keys + task.exclude_keys,
    )

    base = PMAsymDenoiser(base_config.model, 12, 6, 25).eval()
    local = PMAsymDenoiser(local_config.model, 12, 6, 25).eval()
    incompatible = local.load_state_dict(base.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys and all(name.startswith('denoiser.local_cov.') for name in incompatible.missing_keys)
    layer_calls = []
    original_local_cov = local.denoiser.local_cov.forward

    def capture_local_cov(block_index, node_feature, ligand_pos_xyz, ligand_batch, grid, density_origin_xyz, density_basis_xyz):
        layer_calls.append((block_index, grid.data_ptr(), ligand_pos_xyz.detach().clone()))
        return original_local_cov(block_index, node_feature, ligand_pos_xyz, ligand_batch, grid, density_origin_xyz, density_basis_xyz)

    monkeypatch.setattr(local.denoiser.local_cov, 'forward', capture_local_cov)
    with torch.no_grad():
        expected = base(batch)
        observed = local(batch)
    for key in expected:
        torch.testing.assert_close(observed[key], expected[key], rtol=0, atol=0)
    assert [item[0] for item in layer_calls] == list(range(6))
    assert len({item[1] for item in layer_calls}) == 1
    assert any(not torch.equal(layer_calls[index][2], layer_calls[index + 1][2]) for index in range(5))


def test_local_cov_c0_c5_e_keep_model_origin_separate_from_crop_origin(prepared_data):
    """C0、C5、E 使用各自模型原点; 密度裁块角点只通过局部 origin 表达."""
    root = Path(__file__).resolve().parents[1]
    config = EasyDict(yaml.safe_load((root / 'configs/docking/local_cov-C-T0-RA.yml').read_text(encoding='utf-8')))
    featurizer = FeaturizeMol(config.transforms.featurizer)
    task = ConfTransform(EasyDict(settings=dict(free=1.0), free_no_geometry=True), mode='test')
    transform = Compose([featurizer, task])

    center_data = EasyDict(deepcopy(dict(prepared_data)))
    center_data.update(pocket_mode='center', knn=32)
    center_samples = {}
    center_offsets = {}
    for protocol in ('C0', 'C5'):
        dataset = OccurrenceDataset(center_data, 'validation', transform, 'RA', protocol, False, density_config=config.model.density)
        center_samples[protocol] = dataset[0]
        center_offsets[protocol] = dataset.records[0].get('center_offset_xyz_A', [0.0, 0.0, 0.0])
    torch.testing.assert_close(center_samples['C0'].node_pos.mean(dim=0), torch.zeros(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        center_samples['C5'].node_pos.mean(dim=0),
        -torch.tensor(center_offsets['C5'], dtype=torch.float32),
        atol=1e-6,
        rtol=0,
    )

    envelope_data = EasyDict(deepcopy(dict(prepared_data)))
    envelope_data.update(pocket_mode='envelope', knn=32)
    envelope = OccurrenceDataset(envelope_data, 'validation', transform, 'RA', 'E', False, density_config=config.model.density)[0]
    torch.testing.assert_close(envelope.pocket_pos.mean(dim=0), torch.zeros(3), atol=1e-6, rtol=0)

    source_corner = torch.tensor([1.0, 2.0, 3.0])
    for sample in (*center_samples.values(), envelope):
        torch.testing.assert_close(sample.density_origin[0] + sample.pocket_center[0], source_corner)
