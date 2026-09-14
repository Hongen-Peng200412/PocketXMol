"""用非科学构造资产核对正式Dataset、DataModule与密度裁块的共同原点."""

from pathlib import Path

import numpy as np
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
    dataset_config.update(root=prepared_data.root, derived_root=prepared_data.derived_root, manifest_root=prepared_data.manifest_root, smiles_root=prepared_data.smiles_root, smiles_coords_root=prepared_data.smiles_coords_root)
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
    config.data.dataset.update(root=prepared_data.root, derived_root=prepared_data.derived_root, manifest_root=prepared_data.manifest_root, smiles_root=prepared_data.smiles_root, smiles_coords_root=prepared_data.smiles_coords_root)
    config.model.density = {'mode': 'D1'}
    config.train.update(num_workers=workers, persistent_workers=False, batch_size=1, prefetch_factor=1)
    module = DataModule(config)
    module.setup('fit')
    for loader in (module.train_loader, module.val_loader):
        assert loader.prefetch_factor == (1 if workers else None)
        assert loader.dataset.density_config == config.model.density
        assert loader.dataset.protocol == 'C0'
        assert loader.dataset[0].density_input.shape == (1, 56, 48, 48, 48)


@pytest.mark.parametrize('protocol', ['C0', 'C5', 'E'])
def test_d3_labels_follow_crop_and_stay_out_of_sampling(prepared_data, monkeypatch, protocol):
    """源ZYX标签按实际裁块起点裁取; 采样即使使用validation清单也不能读取标签文件."""
    config = make_config(str(Path(__file__).resolve().parents[1] / 'configs/docking/B-C-T0-RA.yml'))
    config.data.dataset.update({name: prepared_data[name] for name in ('root', 'derived_root', 'manifest_root', 'smiles_root', 'smiles_coords_root', 'language_root')})
    config.data.dataset.pocket_mode = 'envelope' if protocol == 'E' else 'center'
    config.model.density = {'mode': 'D3'}
    config.train.update(num_workers=0, persistent_workers=False, batch_size=1)
    module = DataModule(config)
    module.setup('fit')
    dataset = module.val_loader.dataset
    dataset.protocol = protocol
    record = dataset.records[0]
    area_path = Path(prepared_data.derived_root) / 'ligand_area' / record['pdb_id'] / f"{record['candidate_id']}.npy"
    # 非零源起点和不对称ZYX索引可检测轴交换; 包含负边界、上边界及最后一个合法体素.
    np.save(area_path, np.array([[10, 21, 32], [57, 67, 77], [9, 20, 30], [58, 20, 30]], dtype=np.int32))
    original_load = __import__('docking.dataset', fromlist=['load_density_input']).load_density_input

    def shifted_crop(*args):
        """保留构造密度和几何, 仅注入待检查的非零裁块索引."""
        fields = original_load(*args)
        fields['density_start_zyx'] = torch.tensor([[10, 20, 30]])
        return fields

    monkeypatch.setattr('docking.dataset.load_density_input', shifted_crop)
    supervised = dataset[0]
    assert supervised.density_language.shape == (1, 768)
    assert supervised.density_target.shape == (1, 48, 48, 48)
    assert supervised.density_target.sum() == 2
    assert supervised.density_target[0, 0, 1, 2] == 1
    assert supervised.density_target[0, 47, 47, 47] == 1
    # 删除测试夹具标签后, 同一划分仍能构造推理输入; 实际测试资产始终不参与本验收.
    area_path.unlink()
    inference = OccurrenceDataset(config.data.dataset, 'validation', dataset.transforms, 'RA', protocol, False, density_config=config.model.density)[0]
    assert 'density_target' not in inference
    torch.testing.assert_close(inference.density_language, supervised.density_language, rtol=0, atol=0)
    batch = Batch.from_data_list([supervised, supervised.clone()], exclude_keys=module.get_featurizers()[0].exclude_keys)
    assert batch.density_language.shape == (2, 768)
    assert batch.density_target.shape == (2, 48, 48, 48)
