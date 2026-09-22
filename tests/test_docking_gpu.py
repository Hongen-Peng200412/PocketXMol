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

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from easydict import EasyDict
from rdkit import Chem
from torch_geometric.transforms import Compose

from docking.dataset import ConditionedDockingDataset, OccurrenceDataset
from docking.evaluation import (
    build_receptor_molecule,
    evaluate_occurrence,
    rank_candidate_metrics,
    score_saved_candidates,
)
from docking.assets import read_receptor
from docking.sampling import (
    build_sampling_noiser,
    load_sampling_runtime,
    sample_occurrence,
)
from docking.smiles import read_smiles_coords, read_smiles_graph
from scripts.train_pl import DataModule, DockingCheckpoint, ModelLightning
from utils.misc import make_config
from utils.sample_noise import get_sample_noiser
from utils.transforms import ConfTransform, FeaturizeMol
from test_docking_data import prepared_data


@pytest.mark.skipif(not torch.cuda.is_available(), reason='需要实际授权的CUDA GPU')
@pytest.mark.parametrize(
    'stage,config_name,protocol',
    [
        ('official-c', 'sample-official-CA2-test.yml', 'official-c'),
        ('local-c1', 'sample-local_cov-C-CA2-test.yml', 'local-c1'),
        ('local-e', 'sample-local_cov-E-CA2-test.yml', 'local-e'),
    ],
)
def test_conditioned_non_test_sampling_uses_real_models(
    tmp_path,
    stage,
    config_name,
    protocol,
):
    """用validation实例核对无真值入口可完成真实模型GPU采样并只还原一次世界坐标。"""
    root = Path(__file__).resolve().parents[1]
    sampling = make_config(str(root / 'configs' / 'docking' / config_name))
    sampling.update(
        split='validation',
        output_root=str(tmp_path / stage),
        num_candidates=2,
        num_steps=3,
        batch_size=2,
        device='cuda',
    )
    # 非测试门控读取已验收的GT validation资产；CA2正式根只覆盖冻结test对象。
    sampling.dataset.root = '/storage/penghongen/AdaLigand/Ori_Data'
    train_config, model_config, model, featurizer, transforms, sample_config = (
        load_sampling_runtime(sampling)
    )
    standard = OccurrenceDataset(
        sampling.dataset,
        'validation',
        transforms,
        sampling.receptor_branch,
        'E' if stage == 'local-e' else 'C0',
        False,
        density_config=model_config.get('density'),
    )
    source = standard.records[0]
    coordinates = read_smiles_coords(
        sampling.dataset.smiles_coords_root,
        source['pdb_id'],
        int(source['candidate_id']),
        source['prepared_smiles'],
    )
    record = {
        'pdb_id': source['pdb_id'],
        'candidate_id': int(source['candidate_id']),
        'prepared_smiles': source['prepared_smiles'],
        'sampling_seed': 91021,
        'views': [],
        'center_offset_xyz_A': [0.0, 0.0, 0.0],
    }
    if stage == 'local-e':
        record['envelope_coords_xyz_A'] = coordinates.tolist()
        pocket_mode = 'envelope'
    else:
        record['given_center_xyz_A'] = coordinates.mean(axis=0).tolist()
        pocket_mode = 'center'
    dataset = ConditionedDockingDataset(
        EasyDict(
            root=sampling.dataset.root,
            smiles_root=sampling.dataset.smiles_root,
            knn=int(sampling.dataset.knn),
            pocket_mode=pocket_mode,
        ),
        [record],
        transforms,
        sampling.receptor_branch,
        density_config=model_config.get('density'),
    )
    assert 'smiles_coords_root' not in dataset.config
    noiser = build_sampling_noiser(
        sampling,
        train_config,
        sample_config,
        featurizer,
    )
    result = sample_occurrence(
        dataset,
        0,
        model,
        noiser,
        featurizer,
        sampling,
        protocol,
    )
    assert result['success_count'] == 2, result
    output_dir = (
        Path(sampling.output_root)
        / 'validation'
        / protocol
        / source['pdb_id']
        / str(source['candidate_id'])
    )
    poses = [
        pose
        for pose in Chem.SDMolSupplier(
            str(output_dir / result['pose_file']),
            removeHs=True,
        )
        if pose is not None
    ]
    assert len(poses) == 2
    assert all(
        np.isfinite(pose.GetConformer().GetPositions()).all()
        for pose in poses
    )
    template = read_smiles_graph(
        sampling.dataset.smiles_root,
        source['prepared_smiles'],
    )['mol']
    receptor = read_receptor(
        Path(sampling.dataset.root)
        / 'parse'
        / source['pdb_id']
        / 'receptor_tokens.npz'
    )
    receptor_molecule = build_receptor_molecule(receptor)
    candidates = json.loads((output_dir / result['candidate_file']).read_text())
    metrics = score_saved_candidates(
        candidates, poses, receptor_molecule, template, rmsd_reference=None
    )
    assert not any(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in template.GetAtoms()
    )
    assert not any(
        bond.GetStereo() != Chem.BondStereo.STEREONONE
        for bond in template.GetBonds()
    )
    assert all(item['stereo'] for item in metrics)
    assert [item['sample_index'] for item in rank_candidate_metrics(metrics)] == [
        item['sample_index']
        for item in sorted(
            metrics,
            key=lambda item: (-item['self_ranking'], item['sample_index']),
        )
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='需要实际授权的CUDA GPU')
@pytest.mark.parametrize('branch', ['RA'])
def test_official_weights_native_bf16_training_and_stopped_restore(prepared_data, tmp_path, branch):
    """核对原权重、RA投影、bf16原loss更新和完整已停止检查点恢复, 保留学到的核酸参数."""
    config = make_config(str(Path(__file__).resolve().parents[1] / f'configs/docking/B-C-T0-{branch}.yml'))
    config.data.dataset.update(root=prepared_data.root, derived_root=prepared_data.derived_root, manifest_root=prepared_data.manifest_root, smiles_root=prepared_data.smiles_root, smiles_coords_root=prepared_data.smiles_coords_root)
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
@pytest.mark.parametrize('experiment', ['local_cov-C-T0-RA', 'local_cov-E-T0-RA'])
def test_real_data_training_and_sampling_budget(tmp_path, monkeypatch, experiment):
    """用真实非test资产检查local_cov训练、显存、原损失及采样评价，不设姿态质量阈值。"""
    root = Path(__file__).resolve().parents[1]
    config = make_config(str(root / f'configs/docking/{experiment}.yml'))
    # 门控只运行2次优化器更新, 因此把本次回调验证间隔缩为2; 正式YAML仍为每800次更新验证.
    config.train.val_check_interval = 2
    pl.seed_everything(config.train.seed, workers=True)
    data_module = DataModule(config)
    args = SimpleNamespace(num_gpus=1, multi_node=False, resume='')
    model = ModelLightning(config, args, **data_module.get_in_dims())
    # 第一次更新先使零初始化FiLM获得非零参数; 第二次更新应让六套局部卷积都收到梯度并改变参数.
    initial_encoder_weights = [
        encoder.convolutions[0].weight.detach().cpu().clone()
        for encoder in model.model.denoiser.local_cov.encoders
    ]

    def reject_oom(batch):
        # 验收若OOM必须显露, 才能按已批准的成对batch/累积调整; 不在门控内静默裁小样本批.
        raise RuntimeError(f'真实样本GPU验收OOM: {experiment}, batch_size={config.train.batch_size}')

    monkeypatch.setattr(model, 'reduce_batch', reject_oom)
    assert config.train.batch_size * config.train.accumulate_grad_batches == 72
    # 只把本次门控缩为2次更新、1个验证批; 正式YAML不变, 原模型/loss/AdamW/数据加载资源均照配置.
    checkpoint = DockingCheckpoint(str(tmp_path / 'checkpoints'), f'gate-{experiment}')
    trainer_options = dict(accelerator='gpu', devices=1, precision=config.train.precision, max_steps=2, max_epochs=-1, logger=False, enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=1, check_val_every_n_epoch=None, val_check_interval=2*config.train.accumulate_grad_batches, accumulate_grad_batches=config.train.accumulate_grad_batches)
    trainer = pl.Trainer(**trainer_options, callbacks=[checkpoint])
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.fit(model, datamodule=data_module, weights_only=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert trainer.global_step == 2 and torch.isfinite(trainer.callback_metrics['val/loss'])
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    for encoder, initial_weight in zip(model.model.denoiser.local_cov.encoders, initial_encoder_weights):
        assert not torch.equal(encoder.convolutions[0].weight.detach().cpu(), initial_weight)
    saved = torch.load(checkpoint.last_model_path, map_location='cpu', weights_only=False)
    local_parameter_names = [
        name for name, _ in model.model.named_parameters()
        if name.startswith('denoiser.local_cov.')
    ]
    assert local_parameter_names
    assert all(f'model.{name}' in saved['state_dict'] for name in local_parameter_names)
    saved_optimizer = saved['optimizer_states'][0]
    saved_parameter_ids = [parameter_id for group in saved_optimizer['param_groups'] for parameter_id in group['params']]
    model_parameter_names = [name for name, _ in model.model.named_parameters()]
    parameter_id_by_name = dict(zip(model_parameter_names, saved_parameter_ids))
    assert all(parameter_id_by_name[name] in saved_optimizer['state'] for name in local_parameter_names)
    saved_control = saved['callbacks'][checkpoint.state_key]['pocketxmol']
    assert saved_control['scheduler']['last_epoch'] == 1
    assert saved_control['last_validation_step'] == 2
    assert saved_control['stop_reason'] == 'max_steps'
    # 此耗时包含首次真实资产读取与1批验证, 不能当成稳定每步训练速度.
    report = dict(experiment=experiment, scope='gate_not_formal', global_batch=72, batch_size=config.train.batch_size, accumulation=config.train.accumulate_grad_batches, updates=2, fit_elapsed_seconds=elapsed, peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(), peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(), val_loss=float(trainer.callback_metrics['val/loss']))
    report.update(training_protocol=data_module.train_dataloader().dataset.protocol, supervised_validation_protocol=data_module.val_dataloader().dataset.protocol)
    # Lightning训练退出可能把模型移回CPU; 采样前明确恢复到本次CUDA设备.
    model.model.to('cuda').eval()
    featurizer = FeaturizeMol(config.transforms.featurizer)
    task = ConfTransform(EasyDict(settings=dict(free=1.0), free_no_geometry=True), mode='test')
    sampling = EasyDict(model_name=f'gate_{experiment}', split='validation', output_root=str(tmp_path / 'sampling'), dataset=config.data.dataset, num_candidates=2, num_steps=3, batch_size=2, device='cuda')
    noise_config = make_config(str(root / 'configs/sample/test/dock_poseboff/base.yml')).noise
    noise_config.num_steps = 3
    noiser = get_sample_noiser(noise_config, featurizer.num_node_types, featurizer.num_edge_types, mode='sample', device='cuda', ref_config=config.noise)
    protocols = ('C0', 'C5') if config.data.dataset.pocket_mode == 'center' else ('E',)
    protocol_results = {}
    for protocol in protocols:
        dataset = OccurrenceDataset(config.data.dataset, 'validation', Compose([featurizer, task]), config.model.nucleic_branch, protocol, False, density_config=config.model.get('density'))
        result = sample_occurrence(dataset, 0, model.model, noiser, featurizer, sampling, protocol)
        assert result['success_count'] == 2, result
        assessment = evaluate_occurrence((sampling, protocol, dataset.records[0]))
        assert assessment['rmsd_count'] == 2, assessment
        assert assessment['evaluation_status'] == 'success', assessment
        protocol_results[protocol] = dict(validation_record=dataset.records[0], sampling=result, assessment=assessment)
    report['protocols'] = protocol_results

    expected_weights = saved['state_dict']
    del trainer
    del model
    torch.cuda.empty_cache()
    resume_args = SimpleNamespace(num_gpus=1, multi_node=False, resume=checkpoint.last_model_path)
    restored = ModelLightning(deepcopy(config), resume_args, **data_module.get_in_dims())
    restored_callback = DockingCheckpoint(str(tmp_path / 'checkpoints'), f'gate-{experiment}')
    resumed_trainer = pl.Trainer(**dict(trainer_options, max_steps=3), callbacks=[restored_callback])
    resumed_trainer.fit(restored, datamodule=DataModule(config), ckpt_path=checkpoint.last_model_path, weights_only=False)
    assert resumed_trainer.global_step == 2
    assert restored_callback.stop_reason == 'max_steps'
    assert restored_callback.decline_count == saved_control['decline_count']
    assert restored_callback.scheduler_state == saved_control['scheduler']
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value.cpu(), expected_weights[name], rtol=0, atol=0)
    restored_optimizer = resumed_trainer.optimizers[0].state_dict()
    assert restored_optimizer['param_groups'] == saved_optimizer['param_groups']
    for parameter_id, state in saved_optimizer['state'].items():
        for key, expected_value in state.items():
            observed_value = restored_optimizer['state'][parameter_id][key]
            if torch.is_tensor(expected_value):
                torch.testing.assert_close(observed_value.cpu(), expected_value, rtol=0, atol=0)
            else:
                assert observed_value == expected_value
    # JSON只记录本次门控来源、资源和结果; SDF与逐候选结果位于同一pytest临时目录, 不接入正式候选池.
    (tmp_path / 'real_data_gate.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'REAL_DATA_GATE {experiment} global_batch=72 fit_seconds={elapsed:.3f} peak_allocated_bytes={report["peak_memory_allocated_bytes"]} protocols={list(protocol_results)} restore_exact=True')
