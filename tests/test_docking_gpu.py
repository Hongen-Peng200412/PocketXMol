"""用构造资产和真实官方权重验收GPU训练, 不产生正式实验结果.

本测试须在授权GPU内运行, 使用原DataModule、ModelLightning和Lightning自动优化.
批量36、累积2仍为72; 仅把验证间隔及更新上限缩为1, W&B关闭.
实际磁盘checkpoint只写pytest临时目录, 不删除历史资产. 真实train/validation样本的显存和吞吐另在共同清单冻结后验收.
"""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch

from scripts.train_pl import DataModule, DockingCheckpoint, ModelLightning
from utils.misc import make_config
from test_docking_data import prepared_data


@pytest.mark.skipif(not torch.cuda.is_available(), reason='需要实际授权的CUDA GPU')
@pytest.mark.parametrize('branch', ['RA', 'RB'])
def test_official_weights_native_bf16_training_and_stopped_restore(prepared_data, tmp_path, branch):
    """核对原权重、RB复制、bf16原loss更新和完整已停止检查点恢复, 保留学到的核酸参数."""
    config = make_config(str(Path(__file__).resolve().parents[1] / f'configs/docking/B-C-T1-{branch}.yml'))
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
    if branch == 'RB':
        for name, value in model.model.pocket_encoder.state_dict().items():
            torch.testing.assert_close(model.model.nucleic_encoder.state_dict()[name], value, rtol=0, atol=0)
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
