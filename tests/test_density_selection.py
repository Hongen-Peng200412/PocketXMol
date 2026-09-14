"""用非测试构造张量核对 D2 并集、D3 预测选择与独立辅助监督."""

from itertools import product
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from models.density_readout import DensityReadout
from models.density_selection import DensitySelection, density_segmentation_loss
from models.density_backbone import DensityEncoder
from models.maskfill import PMAsymDenoiser
from torch_geometric.data import Batch
from test_docking_data import prepared_data
from test_docking_sampling import sampling_context


@pytest.mark.parametrize('distance_bias', [False, True])
def test_d2_all_atoms_read_deduplicated_molecule_union(distance_bias):
    """远离裁块的原子仍读取同分子的非空并集, 重合原子不重复计入体素."""
    torch.manual_seed(140)
    reader = DensityReadout(320, dict(mode='D2', attention_backend='sdpa', distance_bias=distance_bias))
    feature = torch.randn(1, 48, 48, 48, 48, requires_grad=True)
    hidden = torch.randn(4, 320, requires_grad=True)
    position = torch.tensor([[.2, .3, .4], [.7, .5, .9], [8.2, 5.7, 2.2], [-9., 1., 1.]], requires_grad=True)
    batch = torch.zeros(4, dtype=torch.long)
    actual = reader(hidden, position, batch, feature, torch.zeros(1, 3), torch.eye(3)[None])
    union = set()
    for center in position.detach().floor().int().tolist():
        for delta in product(range(-3, 4), repeat=3):
            xyz = tuple(a+b for a, b in zip(center, delta))
            if all(0 <= axis < 48 for axis in xyz):
                union.add(xyz[2]*48*48+xyz[1]*48+xyz[0])
    ids = torch.tensor(sorted(union))
    selected = feature.flatten(2).transpose(1, 2)[0, ids]
    query = F.linear(hidden, reader.query.weight, reader.query.bias).reshape(4, 4, 64)
    key = F.linear(selected, reader.key.weight, reader.key.bias).reshape(-1, 4, 64)
    value = F.linear(selected, reader.value.weight, reader.value.bias).reshape(-1, 4, 64)
    scores = torch.einsum('ahd,khd->ahk', query, key)/8
    if distance_bias:
        centers = torch.stack([ids % 48, ids//48 % 48, ids//(48*48)], dim=-1).float()+.5
        scores = scores-F.softplus(reader.beta)[None, :, None]*(position[:, None]-centers[None]).square().sum(-1)[:, None]/100
    expected = reader.alpha*reader.output(torch.einsum('ahk,khd->ahd', scores.softmax(-1), value).reshape(4, 256))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert actual[-1].abs().sum() > 0
    weight = torch.randn_like(actual)
    parameters = [hidden, position, feature, *reader.parameters()]
    actual_grad = torch.autograd.grad((actual*weight).sum(), parameters, retain_graph=True, allow_unused=True)
    expected_grad = torch.autograd.grad((expected*weight).sum(), parameters, allow_unused=True)
    for first, second in zip(actual_grad, expected_grad):
        if first is None:
            assert second is None
        else:
            torch.testing.assert_close(first, second, rtol=3e-4, atol=3e-6)


def test_d2_empty_union_is_zero_and_neighborhood_updates():
    torch.manual_seed(141)
    reader = DensityReadout(320, dict(mode='D2', attention_backend='reference', distance_bias=True))
    feature = torch.randn(2, 48, 48, 48, 48)
    hidden = torch.randn(4, 320)
    batch = torch.tensor([0, 0, 1, 1])
    origins = torch.tensor([[2., -3., 5.], [-7., 4., 2.]])
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    bases = torch.stack([torch.eye(3), torch.diag(torch.tensor([.7, 1.1, 2.]))@rotation])
    indices = torch.tensor([[-5., 1., 1.], [-8., 2., 2.], [3., 4., 5.], [6., 2., 3.]])
    position = origins[batch]+(indices[:, :, None]*bases[batch]).sum(-2)
    actual = reader(hidden, position, batch, feature, origins, bases)
    assert torch.count_nonzero(actual[:2]) == 0
    separate = reader(hidden[2:], position[2:], torch.zeros(2, dtype=torch.long), feature[1:], origins[1:], bases[1:])
    torch.testing.assert_close(actual[2:], separate, rtol=1e-6, atol=1e-7)
    moved = position.clone()
    moved[0] = origins[0]+torch.tensor([2., 2., 2.])
    assert reader(hidden, moved, batch, feature, origins, bases)[:2].abs().sum() > 0
    changed_feature = feature.clone()
    changed_feature[0] += 100
    torch.testing.assert_close(reader(hidden, position, batch, changed_feature, origins, bases)[2:], actual[2:], rtol=0, atol=0)


def test_d3_probability_head_matches_concatenated_projection_and_stable_topk():
    torch.manual_seed(142)
    head = DensitySelection()
    feature = torch.randn(1, 48, 48, 48, 48, requires_grad=True)
    language = torch.randn(1, 768, requires_grad=True)
    logits, indices = head(feature, language)
    joined = torch.cat([feature.flatten(2).transpose(1, 2), language.detach()[:, None].expand(-1, 48**3, -1)], dim=-1)
    weights = torch.cat([head.density_projection.weight, head.language_projection.weight], dim=1)
    expected = F.linear(F.relu(F.linear(joined, weights, head.density_projection.bias)), head.output.weight, head.output.bias)
    torch.testing.assert_close(logits.flatten(2).transpose(1, 2), expected, rtol=1e-4, atol=2e-6)
    probability = logits.float().softmax(1)[:, 1].flatten(1)
    assert indices.shape == (1, 4096)
    assert len(torch.unique(indices)) == 4096
    selected = probability.gather(1, indices)
    assert torch.all(selected[:, 1:] <= selected[:, :-1])
    loss = density_segmentation_loss(logits, torch.zeros((1, 48, 48, 48), dtype=torch.long))['weighted']
    loss.backward()
    assert language.grad is None
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())
    assert feature.grad.abs().sum() > 0
    with torch.no_grad():
        head.output.weight.zero_()
        head.output.bias.zero_()
        _, tied_indices = head(feature.detach(), language.detach())
    torch.testing.assert_close(tied_indices, torch.arange(4096)[None])


@pytest.mark.parametrize('all_background', [False, True])
def test_d3_focal_and_foreground_tversky_reference(all_background):
    torch.manual_seed(143)
    logits = torch.randn(2, 2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    target = torch.zeros((2, 3, 4, 5), dtype=torch.long) if all_background else torch.randint(0, 2, (2, 3, 4, 5))
    actual = density_segmentation_loss(logits, target)
    exponent = (logits-logits.amax(1, keepdim=True)).exp()
    probability = exponent/exponent.sum(1, keepdim=True)
    probability_truth = torch.where(target.bool(), probability[:, 1], probability[:, 0])
    expected_focal = (-.5*(1-probability_truth.clamp(1e-6, 1-1e-6)).square()*probability_truth.log()).mean()
    foreground = probability[:, 1]
    true_positive = foreground[target.bool()].sum()
    false_positive = foreground[~target.bool()].sum()
    false_negative = (1-foreground[target.bool()]).sum()
    expected_dice = 1-(true_positive+1)/(true_positive+.5*false_positive+.5*false_negative+1)
    torch.testing.assert_close(actual['focal'], expected_focal, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual['dice'], expected_dice, rtol=1e-12, atol=1e-12)
    expected = .1*(.7*expected_focal+.3*expected_dice)
    actual_gradient = torch.autograd.grad(actual['weighted'], logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('distance_bias', [False, True])
def test_d3_reads_only_predictions_and_keeps_molecules_separate(distance_bias):
    torch.manual_seed(144)
    reader = DensityReadout(320, dict(mode='D3', attention_backend='sdpa', distance_bias=distance_bias))
    feature = torch.randn(2, 48, 48, 48, 48)
    hidden = torch.randn(3, 320)
    positions = torch.randn(3, 3)
    batch = torch.tensor([0, 0, 1])
    origin = torch.tensor([[0., 0., 0.], [-2., 4., 1.]])
    basis = torch.eye(3).repeat(2, 1, 1)
    indices = torch.stack([torch.arange(4096), torch.arange(6000, 10096)])
    actual = reader(hidden, positions, batch, feature, origin, basis, indices)
    separate = [reader(hidden[batch == i], positions[batch == i], torch.zeros(int((batch == i).sum()), dtype=torch.long), feature[i:i+1], origin[i:i+1], basis[i:i+1], indices[i:i+1]) for i in range(2)]
    torch.testing.assert_close(actual, torch.cat(separate), rtol=1e-5, atol=1e-6)
    changed = feature.flatten(2).clone()
    for i in range(2):
        absent = torch.ones(48**3, dtype=torch.bool)
        absent[indices[i]] = False
        changed[i, :, absent] += 100
    torch.testing.assert_close(reader(hidden, positions, batch, changed.reshape_as(feature), origin, basis, indices), actual, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['D2', 'D3'])
def test_d2_d3_use_complete_d4_encoder(mode):
    config = dict(mode=mode, attention_backend='sdpa', checkpoint=False)
    encoder = DensityEncoder(config)
    reference = DensityEncoder(dict(config, mode='D4'))
    assert len(encoder.down) == 4
    assert set(encoder.state_dict()) == set(reference.state_dict())
    assert all(value.shape == reference.state_dict()[name].shape for name, value in encoder.state_dict().items())


def test_d3_model_preserves_target_isolation_and_shared_prediction_cache(prepared_data):
    """真实六块图网络使用预测索引; 任意修改标签不改变前向, eval 可复用该预测索引."""
    torch.manual_seed(146)
    training, dataset, featurizer, _, noiser = sampling_context(prepared_data, 'RA', 'validation')
    training.model.density = dict(mode='D3', attention_backend='sdpa', distance_bias=True, checkpoint=False)
    source = dataset[0]
    source.density_origin = torch.tensor([[-24., -24., -24.]])
    source.density_basis = torch.eye(3)[None]
    source.density_language = torch.randn(1, 768)
    source.density_target = torch.ones((1, 48, 48, 48), dtype=torch.bool)
    source.gt_node_pos.zero_()
    batch = Batch.from_data_list([source], follow_batch=['pocket_pos', *featurizer.follow_batch], exclude_keys=featurizer.exclude_keys)
    batch = noiser(batch, .5)
    model = PMAsymDenoiser(training.model, featurizer.num_node_types, featurizer.num_edge_types, 25).eval()
    feature = torch.randn(1, 48, 48, 48, 48)
    observed_indices = []

    def record_indices(module, arguments, kwargs):
        observed_indices.append(kwargs['indices'])

    hooks = [reader.register_forward_pre_hook(record_indices, with_kwargs=True) for reader in model.denoiser.density_readers]
    with torch.no_grad():
        actual = model(batch, density_feature=feature)
        assert len(observed_indices) == 6
        assert all(item.data_ptr() == observed_indices[0].data_ptr() for item in observed_indices)
        cached_indices = observed_indices[0].clone()
        batch.density_target.zero_()
        changed_label = model(batch, density_feature=feature)
        cached = model(batch, density_feature=feature, density_indices=cached_indices)
    for hook in hooks:
        hook.remove()
    for name in actual:
        torch.testing.assert_close(actual[name], changed_label[name], rtol=0, atol=0)
        if name != 'density_logits':
            torch.testing.assert_close(actual[name], cached[name], rtol=0, atol=0)
    assert 'density_logits' not in cached
    assert {'confidence_node', 'confidence_pos', 'confidence_halfedge'} <= set(actual)
    model.train()
    with pytest.raises(ValueError, match='训练必须重新编码'):
        model(batch, density_feature=feature, density_indices=cached_indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='需要 CUDA 验收正式混合精度读出')
@pytest.mark.parametrize('mode', ['D2', 'D3'])
def test_cuda_d2_d3_output_and_gradients_match_fp32(mode):
    torch.manual_seed(145)
    reader = DensityReadout(320, dict(mode=mode, attention_backend='sdpa', distance_bias=True)).cuda()
    reference = deepcopy(reader)
    feature = torch.randn(1, 48, 48, 48, 48, device='cuda', requires_grad=True)
    hidden = torch.randn(2, 320, device='cuda', requires_grad=True)
    position = torch.tensor([[.2, .4, .8], [50.9999*.7, 2., 2.]], device='cuda')
    batch = torch.zeros(2, dtype=torch.long, device='cuda')
    origin = torch.zeros(1, 3, device='cuda')
    basis = torch.eye(3, device='cuda')[None]*.7
    indices = torch.arange(4096, device='cuda')[None] if mode == 'D3' else None
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('highest')
        expected = reference(hidden, position, batch, feature, origin, basis, indices)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            actual = reader(hidden, position, batch, feature, origin, basis, indices)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        weight = torch.randn_like(actual)
        expected_gradient = torch.autograd.grad((expected*weight).sum(), [hidden, feature, *reference.parameters()], allow_unused=True)
        actual_gradient = torch.autograd.grad((actual*weight).sum(), [hidden, feature, *reader.parameters()], allow_unused=True)
        for first, second in zip(expected_gradient, actual_gradient):
            if first is None:
                assert second is None
            else:
                torch.testing.assert_close(second, first, rtol=1e-5, atol=1e-6)
    finally:
        torch.set_float32_matmul_precision(previous_precision)
