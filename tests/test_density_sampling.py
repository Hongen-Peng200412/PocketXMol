"""用构造的非测试实例核对固定密度编码在真实采样循环中的复用."""

from pathlib import Path

import pytest
import torch

from docking.sampling import sample_occurrence
from models.maskfill import PMAsymDenoiser
from utils.misc import make_config
from utils.sample_noise import get_sample_noiser
from test_docking_data import prepared_data
from test_docking_sampling import sampling_context


@pytest.mark.skipif(not torch.cuda.is_available(), reason='真实密度编码与采样需要CUDA')
@pytest.mark.parametrize('mode', ['D1', 'D4'])
@pytest.mark.parametrize('backend', ['sdpa', 'flash'])
def test_cached_density_matches_reencoding_in_real_sampling(prepared_data, monkeypatch, mode, backend):
    """逐步比较同一输入的缓存/重新编码输出, 并核对跨候选批次只编码一次."""
    torch.manual_seed(83)
    if backend == 'flash' and not getattr(torch.backends.cuda, 'is_flash_attention_available', lambda: False)():
        pytest.skip('当前 PyTorch 构建未报告可用 Flash 内核; 实际 flash 配置须在 A800 验收')
    training, dataset, featurizer, config, _ = sampling_context(prepared_data, 'RA', 'validation')
    training.model.density = dict(mode=mode, attention_backend=backend, distance_bias=True, checkpoint=False)
    dataset.density_config = training.model.density
    config.update(device='cuda', num_candidates=3, batch_size=2, num_steps=2)
    source = dataset[0]
    root = Path(__file__).resolve().parents[1]
    sample_config = make_config(str(root / 'configs/sample/test/dock_poseboff/base.yml'))
    sample_config.noise.num_steps = config.num_steps
    noiser = get_sample_noiser(sample_config.noise, featurizer.num_node_types, featurizer.num_edge_types, mode='sample', device='cuda', ref_config=training.noise)
    model = PMAsymDenoiser(training.model, featurizer.num_node_types, featurizer.num_edge_types, 25).cuda().eval()
    original_forward = model.forward
    encoding_calls = []
    compared_batches = []

    def record_encoding(module, arguments, output):
        """记录实际编码批量, 用于区分一次缓存和逐步重算的参考调用."""
        encoding_calls.append(arguments[0].shape[0])

    def compare_forward(batch, density_feature=None):
        """同一噪声状态计算两种输入方式, 检查全部预测和置信度后返回缓存结果."""
        assert density_feature is not None
        assert 'density_input' not in batch
        assert torch.count_nonzero(batch.gt_node_pos) == 0
        torch.testing.assert_close(batch.density_origin.cpu(), source.density_origin.expand(batch.num_graphs, -1))
        if batch.num_graphs > 1:
            assert density_feature.stride(0) == 0
        actual = original_forward(batch, density_feature=density_feature)
        batch.density_input = source.density_input.to('cuda').expand(batch.num_graphs, -1, -1, -1, -1)
        expected = original_forward(batch)
        del batch.density_input
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=2e-4, atol=2e-5)
        compared_batches.append(batch.num_graphs)
        return actual

    hook = model.density_encoder.register_forward_hook(record_encoding)
    monkeypatch.setattr(model, 'forward', compare_forward)
    previous_matmul_precision = torch.get_float32_matmul_precision()
    try:
        # 卷积和投影固定严格FP32; flash 配置仅在其内核内使用 bf16, 两条路径共用同一配置.
        # 不让cuDNN因编码批量不同而选择不同的TF32近似.
        # 显式固定matmul, 不受其它测试导入train_pl时的medium全局设置影响; 退出时恢复.
        torch.set_float32_matmul_precision('highest')
        with torch.backends.cudnn.flags(allow_tf32=False):
            result = sample_occurrence(dataset, 0, model, noiser, featurizer, config, 'C5')
    finally:
        hook.remove()
        torch.set_float32_matmul_precision(previous_matmul_precision)
    assert result['status'] == 'success', result
    assert compared_batches == [2, 2, 1, 1]
    assert encoding_calls == [1, 2, 2, 1, 1]
    assert result['model_forward_completed_count'] == 4
