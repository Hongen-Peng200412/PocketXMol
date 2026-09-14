"""核对D3辅助监督接入实际Lightning步骤, 原val/loss保持独立."""

from types import SimpleNamespace

import pytorch_lightning as pl
import torch

from models.density_selection import density_segmentation_loss
from scripts.train_pl import ModelLightning


def test_d3_auxiliary_loss_updates_training_without_changing_best_metric(monkeypatch):
    """使用小型构造logits执行原training_step/validation_step, 检查优化目标与选best指标的区别."""
    module = ModelLightning.__new__(ModelLightning)
    pl.LightningModule.__init__(module)
    module.sync_dist = False
    module.is_docking = True
    logits = torch.randn(2, 2, 3, 4, 5, requires_grad=True)
    original_loss = torch.tensor(2., requires_grad=True)
    batch = SimpleNamespace(num_graphs=2, density_target=torch.randint(2, (2, 3, 4, 5)))
    module.model = lambda data: {'density_logits': logits}
    module.loss_func = lambda data, outputs: {'mixed/total': original_loss}
    logged = {}
    monkeypatch.setattr(module, 'log_dict', lambda values, **kwargs: logged.update(values))
    monkeypatch.setattr(module, 'log', lambda key, value, **kwargs: logged.update({key: value}))
    auxiliary = density_segmentation_loss(logits, batch.density_target)['weighted']
    training = module.training_step(batch, 0)
    torch.testing.assert_close(training, original_loss + auxiliary)
    training.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    assert original_loss.grad == 1
    validation = module.validation_step(batch, 0)
    assert validation['val_mixed/total'] is original_loss
    assert logged['val/loss'] is original_loss
    torch.testing.assert_close(logged['val/density_weighted'], auxiliary)
