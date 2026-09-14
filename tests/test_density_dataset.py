"""用非科学构造资产核对正式Dataset、DataModule与密度裁块的共同原点."""

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose

from docking.dataset import OccurrenceDataset
from scripts.train_pl import DataModule
from utils.misc import make_config
from utils.transforms import FeaturizeMol, get_transforms
from test_docking_data import prepared_data


@pytest.mark.parametrize('protocol', ['C0', 'C5', 'E'])
def test_density_dataset_preserves_input_origin(prepared_data, protocol):
    config = make_config(str(Path(__file__).resolve().parents[1] / 'configs/docking/B-C-T0-RA.yml'))
    dataset_config = config.data.dataset
    dataset_config.update(root=prepared_data.root, derived_root=prepared_data.derived_root, manifest_root=prepared_data.manifest_root)
    dataset_config.pocket_mode = 'envelope' if protocol == 'E' else 'center'
    features = FeaturizeMol(config.transforms.featurizer)
    transforms = Compose([features, get_transforms(config.transforms.task, mode='test', num_node_types=features.num_node_types)])
    plain = OccurrenceDataset(dataset_config, 'train', transforms, 'RA', protocol, False)[0]
    density = OccurrenceDataset(dataset_config, 'train', transforms, 'RA', protocol, False, density_config={'mode': 'D1'})[0]
    torch.testing.assert_close(density.node_pos, plain.node_pos, rtol=0, atol=0)
    torch.testing.assert_close(density.pocket_pos, plain.pocket_pos, rtol=0, atol=0)
    torch.testing.assert_close(density.pocket_center, plain.pocket_center, rtol=0, atol=0)
    assert density.density_input.shape == (1, 56, 48, 48, 48)
    # 构造地图正好48³, 实际起点只能为0; C5/E不得因块无法移动而改变模型原点.
    torch.testing.assert_close(density.density_start_zyx, torch.zeros((1, 3), dtype=torch.long))
    torch.testing.assert_close(density.density_origin + density.pocket_center, torch.tensor([[1., 2., 3.]]))
    torch.testing.assert_close(density.density_basis, torch.diag(torch.tensor([.9, 1., 1.1]))[None])
    batch = Batch.from_data_list([density, density.clone()], exclude_keys=features.exclude_keys)
    assert batch.density_input.shape == (2, 56, 48, 48, 48)
    assert batch.density_basis.shape == (2, 3, 3)
    assert 'density_input' not in plain


@pytest.mark.parametrize('workers', [0, 2])
def test_datamodule_uses_model_density_setting(prepared_data, workers):
    config = make_config(str(Path(__file__).resolve().parents[1] / 'configs/docking/B-C-T0-RA.yml'))
    config.data.dataset.update(root=prepared_data.root, derived_root=prepared_data.derived_root, manifest_root=prepared_data.manifest_root)
    config.model.density = {'mode': 'D1'}
    config.train.update(num_workers=workers, persistent_workers=False, batch_size=1, prefetch_factor=1)
    module = DataModule(config)
    module.setup('fit')
    for loader in (module.train_loader, module.val_loader):
        assert loader.prefetch_factor == (1 if workers else None)
        assert loader.dataset.density_config == config.model.density
        assert loader.dataset.protocol == 'C0'
        assert loader.dataset[0].density_input.shape == (1, 56, 48, 48, 48)
