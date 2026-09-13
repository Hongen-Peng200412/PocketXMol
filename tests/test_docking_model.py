"""检查新增受体编码与T1的真实张量行为, 使用构造输入而不读取测试集."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from easydict import EasyDict
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose

from docking.dataset import OccurrenceDataset
from models.maskfill import PMAsymDenoiser
from utils.data import PocketMolData
from utils.sample_noise import ConfSampleNoiser
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


@pytest.mark.parametrize('branch', ['RA', 'RB'])
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
    if branch == 'RB' and receptor_kind != 'protein':
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.nucleic_encoder.parameters())


def test_original_protein_behavior_and_rb_copy(prepared_data):
    config, batch = assemble_batch(prepared_data, 'RB', 'protein')
    config.model.nucleic_branch = None
    original = PMAsymDenoiser(config.model, 12, 6, 25).eval()
    with torch.no_grad():
        expected = original(batch)
    for branch in ['RA', 'RB']:
        config.model.nucleic_branch = branch
        model = PMAsymDenoiser(config.model, 12, 6, 25).eval()
        missing, unexpected = model.load_state_dict(original.state_dict(), strict=False)
        assert not unexpected and all(name.startswith(('nucleic_embedder.', 'nucleic_encoder.')) for name in missing)
        if branch == 'RB':
            model.nucleic_encoder.load_state_dict(model.pocket_encoder.state_dict())
            for name, value in model.pocket_encoder.state_dict().items():
                torch.testing.assert_close(model.nucleic_encoder.state_dict()[name], value, rtol=0, atol=0)
        with torch.no_grad():
            observed = model(batch)
        for key in expected:
            torch.testing.assert_close(observed[key], expected[key], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('level,from_prior', [(0.5, False), (1.0, False), (0.0, True)])
def test_t1_uses_clean_center_and_real_noise_level(batched, level, from_prior):
    config = EasyDict(yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/docking/B-C-T1-RA.yml').read_text(encoding='utf-8')))
    noise_config = config.noise.individual[0]
    noiser = ConfSampleNoiser(noise_config, 12, 6, mode='train', task_name='dock')
    position = torch.tensor([[1., 2., 3.], [3., 2., 3.], [5., 4., 3.], [7., 4., 3.]])
    graph = torch.tensor([0, 0, 1, 1]) if batched else torch.zeros(4, dtype=torch.long)
    given_center = torch.tensor([[4., 1., 0.], [-1., 2., 3.]]) if batched else torch.tensor([[4., 1., 0.]])
    data = PocketMolData(task='dock', task_setting='free', num_nodes=4, given_center_local=given_center)
    if batched:
        data.node_type_batch = graph
    node_type, halfedge_type = torch.zeros(4, dtype=torch.long), torch.ones(2, dtype=torch.long)
    noise_level = {'pos': torch.full((4,), level)}
    noiser.center_translation = False
    torch.manual_seed(17)
    baseline = noiser.add_noise(node_type, position, halfedge_type, data, from_prior=from_prior, level_dict=noise_level)['pos']
    noiser.center_translation = True
    torch.manual_seed(17)
    translated = noiser.add_noise(node_type, position, halfedge_type, data, from_prior=from_prior, level_dict=noise_level)['pos']
    expected_shift = torch.zeros_like(position)
    if not from_prior:
        for index in graph.unique():
            mask = graph == index
            expected_shift[mask] = (1 - level) * (given_center[index] - position[mask].mean(0))
    torch.testing.assert_close(translated - baseline, expected_shift, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(position[0], torch.tensor([1., 2., 3.]), rtol=0, atol=0)
