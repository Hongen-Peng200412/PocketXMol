"""用真实 Lightning CPU 循环检查训练控制顺序, 循环检查点只存内存.

这些是门控测试: 把验证间隔缩为2次更新, 用单个参数代替科学模型, 不加载训练资产、运行 W&B 或生成正式训练产物. CLI拒绝测试只在pytest临时目录保存一份无权重的检查点元数据. tests/ 中的短测试命令不能作为实验运行命令.
"""

import copy
import random
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from lightning_fabric.plugins.io import CheckpointIO
from torch.utils.data import DataLoader, IterableDataset

from scripts.train_pl import DockingCheckpoint
from utils.misc import make_config


class MemoryCheckpointIO(CheckpointIO):
    """保存完整 Lightning 字典的独立内存副本, 任何删除请求都使测试失败."""

    def __init__(self):
        self.checkpoints = {}

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        self.checkpoints[str(path)] = copy.deepcopy(checkpoint)

    def load_checkpoint(self, path, map_location=None, weights_only=None):
        return copy.deepcopy(self.checkpoints[str(path)])

    def remove_checkpoint(self, path):
        raise AssertionError(f"不应删除已有检查点: {path}")


class InfiniteSamples(IterableDataset):
    """提供无限常量输入, 让退出条件只受优化器更新数和真实 Plateau 控制."""

    def __iter__(self):
        while True:
            yield torch.ones(1)


class TinyTrainingModel(pl.LightningModule):
    """以一个可训练参数产生真实反向更新, 用确定的验证值区分原始 best 与1%改善阈值."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(train=SimpleNamespace(val_check_interval=2))
        self.updates = []
        self.validation_steps = []
        self.rng_at_train_start = None

    def training_step(self, batch, batch_idx):
        return self.weight.square()

    def on_train_start(self):
        self.rng_at_train_start = (random.getstate(), np.random.get_state(), torch.get_rng_state())

    def on_before_optimizer_step(self, optimizer):
        self.updates.append(self.global_step + 1)

    def validation_step(self, batch, batch_idx):
        self.validation_steps.append(self.global_step)
        # 10 -> 9.95 是新的原始最小值, 但没有达到 Plateau 的1%相对改善要求.
        self.log("val/loss", torch.tensor(10.0 if self.global_step <= 2 else 9.95), batch_size=1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        self.docking_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", threshold_mode="rel", threshold=0.01, patience=5, factor=0.2, cooldown=0, min_lr=0, eps=1e-8)
        return optimizer


def run_training(memory_io, accumulation, max_steps, resume_path=None):
    """运行真实 CPU 自动优化循环; 虚拟检查点目录不创建文件, 模型无科学资产依赖."""
    checkpoint_dir = str(Path.cwd() / "tmp" / "memory-only-docking-training")
    callback = DockingCheckpoint(checkpoint_dir, "memory-control-test")
    model = TinyTrainingModel()
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=max_steps, max_epochs=-1, callbacks=[callback], logger=False, enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0, check_val_every_n_epoch=None, val_check_interval=2, accumulate_grad_batches=accumulation, plugins=[memory_io])
    trainer.fit(model, train_dataloaders=DataLoader(InfiniteSamples(), batch_size=2), val_dataloaders=DataLoader([torch.ones(1)], batch_size=1), ckpt_path=resume_path, weights_only=False)
    return trainer, model, callback


def test_third_decline_stops_and_saves_updated_scheduler():
    """第三次真实下降后的检查点已有新LR/停止状态, 参数没有再更新, 原始最低loss独立选best."""
    memory_io = MemoryCheckpointIO()
    trainer, model, callback = run_training(memory_io, accumulation=2, max_steps=50)
    assert trainer.global_step == 38
    assert model.updates == list(range(1, 39))
    assert model.validation_steps == list(range(2, 39, 2))
    assert callback.best_model_path.endswith("step=4.ckpt")
    assert abs(float(callback.best_model_score) - 9.95) < 1e-5
    assert callback.decline_count == 3
    assert callback.stop_reason == "plateau"
    saved = memory_io.checkpoints[callback.last_model_path]
    state = saved["callbacks"][callback.state_key]["pocketxmol"]
    assert saved["global_step"] == 38
    assert saved["lr_schedulers"] == []
    assert state["scheduler"]["last_epoch"] == 19
    assert state["scheduler"]["num_bad_epochs"] == 0
    assert state["decline_count"] == 3
    assert state["stop_reason"] == "plateau"
    assert abs(saved["optimizer_states"][0]["param_groups"][0]["lr"] - 8e-7) < 1e-12
    assert len(memory_io.checkpoints) == 20  # 19个验证检查点与一个last.


def test_resume_changed_accumulation_keeps_update_intervals_and_state():
    """从第8次更新继续并把累积2改为3, 验证仍在10、12等更新点执行, 原best和随机状态保留."""
    memory_io = MemoryCheckpointIO()
    _, _, original = run_training(memory_io, accumulation=2, max_steps=50)
    resume_path = str(Path(original.dirpath) / "step=8.ckpt")
    saved_rng = memory_io.checkpoints[resume_path]["callbacks"][original.state_key]["pocketxmol"]["rng"]
    trainer, model, callback = run_training(memory_io, accumulation=3, max_steps=50, resume_path=resume_path)
    assert trainer.global_step == 38
    assert model.updates == list(range(9, 39))
    assert model.validation_steps == list(range(10, 39, 2))
    assert callback.decline_count == 3
    assert callback.best_model_path.endswith("step=4.ckpt")
    assert callback.run_id == "memory-control-test"
    assert model.rng_at_train_start[0] == saved_rng["python"]
    np.testing.assert_equal(model.rng_at_train_start[1], saved_rng["numpy"])
    assert torch.equal(model.rng_at_train_start[2], saved_rng["torch"])


def test_update_cap_runs_final_validation_and_stopped_resume_does_not_update():
    """更新上限处完成最后验证并保存停止原因, 再载入该停止检查点不会增加参数更新."""
    memory_io = MemoryCheckpointIO()
    trainer, model, callback = run_training(memory_io, accumulation=2, max_steps=6)
    assert trainer.global_step == 6
    assert model.validation_steps == [2, 4, 6]
    assert callback.stop_reason == "max_steps"
    trainer, model, callback = run_training(memory_io, accumulation=3, max_steps=10, resume_path=callback.last_model_path)
    assert trainer.global_step == 6
    assert model.updates == []
    assert model.validation_steps == []
    assert callback.stop_reason == "max_steps"


@pytest.mark.parametrize('changed_experiment', ['B-C-T1-RA', 'B-E-T0-RA'])
def test_cli_rejects_resume_with_changed_science(tmp_path, monkeypatch, capsys, changed_experiment):
    root = Path(__file__).resolve().parents[1]
    saved_config = make_config(str(root / 'configs/docking/B-C-T0-RA.yml'))
    callback = DockingCheckpoint(str(tmp_path / 'experiment/checkpoints'), 'no-network-test')
    state = {'callbacks': {callback.state_key: callback.state_dict()}, 'hyper_parameters': {'config': saved_config}}
    checkpoint = tmp_path / 'metadata-only.ckpt'
    torch.save(state, checkpoint)
    monkeypatch.setattr(sys, 'argv', ['train_pl.py', str(root / f'configs/docking/{changed_experiment}.yml'), '--logdir', str(tmp_path / 'experiment'), '--resume', str(checkpoint)])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(root / 'scripts/train_pl.py'), run_name='__main__')
    assert stopped.value.code == 2
    assert '续训科学配置与检查点不同' in capsys.readouterr().err
    assert not (tmp_path / 'experiment/wandb').exists()
