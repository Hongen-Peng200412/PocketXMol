"""用真实官方权重验收GPU训练和采样, 不产生正式实验结果.

本测试须在授权GPU内运行, 使用原DataModule、ModelLightning和Lightning自动优化.
全部关闭W&B. official_weights测试使用构造资产, 固定36×2=72, 验证间隔及更新上限缩为1; 实际磁盘checkpoint只写pytest临时目录, 不删除历史资产.
real_data测试须等共同清单冻结, 读取真实train/validation资产及实际配置的batch/累积, 保持global batch72; 临时训练2次更新并生成2个3步候选, 不读取test划分.
"""

import json
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from easydict import EasyDict
from torch_geometric.transforms import Compose

from docking.dataset import OccurrenceDataset
from docking.evaluation import evaluate_occurrence
from docking.sampling import sample_occurrence
from scripts.train_pl import DataModule, DockingCheckpoint, ModelLightning
from utils.misc import make_config
from utils.sample_noise import get_sample_noiser
from utils.transforms import ConfTransform, FeaturizeMol
from test_docking_data import prepared_data


@pytest.mark.skipif(not torch.cuda.is_available(), reason='需要实际授权的CUDA GPU')
@pytest.mark.parametrize('branch', ['RA'])
def test_official_weights_native_bf16_training_and_stopped_restore(prepared_data, tmp_path, branch):
    """核对原权重、RA投影、bf16原loss更新和完整已停止检查点恢复, 保留学到的核酸参数."""
    config = make_config(str(Path(__file__).resolve().parents[1] / f'configs/docking/B-C-T0-{branch}.yml'))
    config.data.dataset.update(root=prepared_data.root, derived_root=prepared_data.derived_root, manifest_root=prepared_data.manifest_root)
    config.train.update(batch_size=36, accumulate_grad_batches=2, num_workers=0, persistent_workers=False, val_check_interval=1)
    args = SimpleNamespace(num_gpus=1, multi_node=False, resume='')
    data_module = DataModule(config)
    model = ModelLightning(config, args, **data_module.get_in_dims())
    # dict[str,Tensor], 官方旧主干每一项都必须逐值一致, 不能静默漏载旧参数.
    official = torch.load(config.train.initial_checkpoint, map_location='cpu', weights_only=False)
    for name, value in official['state_dict'].items():
        if name.startswith('model.'):
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    del official
    # float32, (128,15), 保存核酸投影初值, 后面确认原loss确实训练新增参数.
    initial_nucleic_weight = model.model.nucleic_embedder.weight.detach().clone()
    checkpoint = DockingCheckpoint(str(tmp_path / 'checkpoints'), 'gpu-contract-check')
    trainer_config = dict(accelerator='gpu', devices=1, precision='bf16-mixed', max_steps=1, max_epochs=-1, logger=False, enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0, check_val_every_n_epoch=None, val_check_interval=1, accumulate_grad_batches=2)
    trainer = pl.Trainer(**trainer_config, callbacks=[checkpoint])
    trainer.fit(model, datamodule=data_module, weights_only=False)
    assert trainer.global_step == 1
    assert checkpoint.stop_reason == 'max_steps' and checkpoint.decline_count == 0
    assert torch.isfinite(trainer.callback_metrics['val/loss'])
    assert not torch.equal(model.model.nucleic_embedder.weight.detach().cpu(), initial_nucleic_weight)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    # dict, 真实Lightning磁盘检查点包含AdamW状态和该次验证后的调度状态.
    saved = torch.load(checkpoint.last_model_path, map_location='cpu', weights_only=False)
    assert saved['global_step'] == 1 and saved['optimizer_states'][0]['state']
    assert saved['callbacks'][checkpoint.state_key]['pocketxmol']['last_validation_step'] == 1
    assert saved['callbacks'][checkpoint.state_key]['pocketxmol']['scheduler']['last_epoch'] == 1
    expected_weights = saved['state_dict']
    # 明确resume让构造器跳过官方初始化; Trainer恢复完整检查点后不能再更新已停止模型.
    resume_args = SimpleNamespace(num_gpus=1, multi_node=False, resume=checkpoint.last_model_path)
    restored = ModelLightning(deepcopy(config), resume_args, **data_module.get_in_dims())
    restored_callback = DockingCheckpoint(str(tmp_path / 'checkpoints'), 'gpu-contract-check')
    # 把恢复上限放宽到2, 从而证明未再更新是停止标记生效, 不是碰巧已达相同上限.
    resumed_trainer = pl.Trainer(**dict(trainer_config, max_steps=2), callbacks=[restored_callback])
    resumed_trainer.fit(restored, datamodule=DataModule(config), ckpt_path=checkpoint.last_model_path, weights_only=False)
    assert resumed_trainer.global_step == 1
    assert restored_callback.stop_reason == 'max_steps'
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value.cpu(), expected_weights[name], rtol=0, atol=0)
    print(f'GPU_CHECK branch={branch} global_batch=72 updates=1 val_loss={float(trainer.callback_metrics["val/loss"])} restore_exact=True')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='需要实际授权的CUDA GPU')
@pytest.mark.parametrize('experiment', ['B-C-T0-RA', 'B-E-T0-RA'])
def test_real_data_training_and_sampling_budget(tmp_path, monkeypatch, experiment):
    """用真实非test资产检查中心C0和包络E训练的显存、原损失及采样评价, 不设姿态质量通过阈值."""
    root = Path(__file__).resolve().parents[1]
    config = make_config(str(root / f'configs/docking/{experiment}.yml'))
    pl.seed_everything(config.train.seed, workers=True)
    data_module = DataModule(config)
    args = SimpleNamespace(num_gpus=1, multi_node=False, resume='')
    model = ModelLightning(config, args, **data_module.get_in_dims())

    def reject_oom(batch):
        # 验收若OOM必须显露, 才能按已批准的成对batch/累积调整; 不在门控内静默裁小样本批.
        raise RuntimeError(f'真实样本GPU验收OOM: {experiment}, batch_size={config.train.batch_size}')

    monkeypatch.setattr(model, 'reduce_batch', reject_oom)
    assert config.train.batch_size * config.train.accumulate_grad_batches == 72
    # 只把本次门控缩为2次更新、1个验证批; 正式YAML不变, 原模型/loss/AdamW/数据加载资源均照配置.
    trainer = pl.Trainer(accelerator='gpu', devices=1, precision=config.train.precision, max_steps=2, max_epochs=-1, logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=1, check_val_every_n_epoch=None, val_check_interval=2*config.train.accumulate_grad_batches, accumulate_grad_batches=config.train.accumulate_grad_batches)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.fit(model, datamodule=data_module, weights_only=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert trainer.global_step == 2 and torch.isfinite(trainer.callback_metrics['val/loss'])
    # 此耗时包含首次真实资产读取与1批验证, 不能当成稳定每步训练速度.
    report = dict(experiment=experiment, scope='gate_not_formal', global_batch=72, batch_size=config.train.batch_size, accumulation=config.train.accumulate_grad_batches, updates=2, fit_elapsed_seconds=elapsed, peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(), peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(), val_loss=float(trainer.callback_metrics['val/loss']))
    report.update(training_protocol=data_module.train_dataloader().dataset.protocol, supervised_validation_protocol=data_module.val_dataloader().dataset.protocol)
    # Lightning训练退出可能把模型移回CPU; 采样前明确恢复到本次CUDA设备.
    model.model.to('cuda').eval()
    featurizer = FeaturizeMol(config.transforms.featurizer)
    task = ConfTransform(EasyDict(settings=dict(free=1.0), free_no_geometry=True), mode='test')
    protocol = 'C5' if config.data.dataset.pocket_mode == 'center' else 'E'
    dataset = OccurrenceDataset(config.data.dataset, 'validation', Compose([featurizer, task]), config.model.nucleic_branch, protocol, False)
    sampling = EasyDict(model_name=f'gate_{experiment}', split='validation', output_root=str(tmp_path / 'sampling'), dataset=config.data.dataset, num_candidates=2, num_steps=3, batch_size=2, device='cuda')
    noise_config = make_config(str(root / 'configs/sample/test/dock_poseboff/base.yml')).noise
    noise_config.num_steps = 3
    noiser = get_sample_noiser(noise_config, featurizer.num_node_types, featurizer.num_edge_types, mode='sample', device='cuda', ref_config=config.noise)
    result = sample_occurrence(dataset, 0, model.model, noiser, featurizer, sampling, protocol)
    assert result['success_count'] == 2, result
    assessment = evaluate_occurrence((sampling, protocol, dataset.records[0]))
    assert assessment['rmsd_count'] == 2, assessment
    assert assessment['evaluation_status'] == 'success', assessment
    report.update(validation_record=dataset.records[0], protocol=protocol, sampling=result, assessment=assessment)
    # JSON只记录本次门控来源、资源和结果; SDF与逐候选结果位于同一pytest临时目录, 不接入正式候选池.
    (tmp_path / 'real_data_gate.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'REAL_DATA_GATE {experiment} global_batch=72 fit_seconds={elapsed:.3f} peak_allocated_bytes={report["peak_memory_allocated_bytes"]} sampling=2/2 evaluation=2/2')
