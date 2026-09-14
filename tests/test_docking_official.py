"""用固定Git版本的真实官方代码验收T0, 不读取正式测试集或修改服务器.

官方参照为65488cf635c856101dbe703ac97e2f10f58e005c. 通过本地Git对象读取原先验、信息等级、噪声器、采样循环和特征化, 不复制公式作为参照.
真实训练资产由PXM_ACCEPTANCE_ASSETS指定本地只读副本根, 含source/、derived/和manifests/train.jsonl; 缺省时仅该项跳过.
"""

import ast
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose

from docking.dataset import OccurrenceDataset
from models.sample import sample_loop3
from scripts.train_pl import DataModule
from utils.misc import make_config, seed_all
from utils.sample_noise import get_sample_noiser
from utils.transforms import ConfTransform, FeaturizeMol
from test_docking_data import prepared_data
from test_docking_model import assemble_batch


OFFICIAL_REVISION = '65488cf635c856101dbe703ac97e2f10f58e005c'
BASELINE_REVISION = '463d59098bca92b7278839992afd0efc84d5db81'
ROOT = Path(__file__).resolve().parents[1]


def reference_modules(revision):
    modules = {}
    for path in ('utils/prior.py', 'utils/info_level.py', 'utils/sample_noise.py', 'models/sample.py', 'utils/transforms.py', 'models/maskfill.py'):
        source = subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=ROOT).decode('utf-8')
        module = ModuleType('reference_' + path.replace('/', '_'))
        exec(compile(source, f'{revision}:{path}', 'exec'), module.__dict__)
        modules[path] = module
    # 原噪声器的依赖也使用同一固定版本, 不回退到当前先验或信息等级.
    modules['utils/sample_noise.py'].MolPrior = modules['utils/prior.py'].MolPrior
    modules['utils/sample_noise.py'].MolInfoLevel = modules['utils/info_level.py'].MolInfoLevel
    return modules


@pytest.fixture(scope='module')
def official():
    return reference_modules(OFFICIAL_REVISION)


@pytest.fixture(scope='module')
def baseline():
    return reference_modules(BASELINE_REVISION)


def clean_sample(assets, split, protocol):
    config = make_config(str(ROOT / 'configs/docking/B-C-T0-RA.yml'))
    config.data.dataset.update(root=assets.root, derived_root=assets.derived_root, manifest_root=assets.manifest_root)
    config.data.dataset.pocket_mode = 'envelope' if protocol == 'E' else 'center'
    featurizer = FeaturizeMol(config.transforms.featurizer)
    task = ConfTransform(config.transforms.task.individual[0], mode='test')
    dataset = OccurrenceDataset(config.data.dataset, split, Compose([featurizer, task]), 'RA', protocol, False)
    return config, featurizer, task, dataset[0]


@pytest.mark.parametrize('protocol', ['C0', 'C5', 'E'])
@pytest.mark.parametrize('mode', ['train', 'sample'])
def test_t0_noise_matches_official_and_valid_baseline(prepared_data, official, baseline, protocol, mode, monkeypatch):
    config, _, _, data = clean_sample(prepared_data, 'validation', protocol)
    noise_config = config.noise.individual[0] if mode == 'train' else make_config(str(ROOT / 'configs/sample/test/dock_poseboff/base.yml')).noise
    modules = [None, official, baseline]
    for level in (0., .35, 1.):
        observed = []
        for modules_at_revision in modules:
            factory = get_sample_noiser if modules_at_revision is None else modules_at_revision['utils/sample_noise.py'].get_sample_noiser
            noiser = factory(deepcopy(noise_config), 12, 6, mode=mode, device='cpu', ref_config=config.noise)
            monkeypatch.setattr(noiser, 'sample_level', lambda step, batch: {'pos': torch.full((batch.num_nodes,), level)})
            sample = data.clone()
            # 非先验步刻意保留非零局部质心; 精确比较原重分配和fixed恢复之后的输入.
            sample.node_pos += torch.tensor([7., -3., 2.])
            seed_all(97)
            result = noiser(sample, step=1. if mode == 'sample' and level == 0 else .37)
            observed.append(result)
        for reference in observed[1:]:
            for key in ('node_pos', 'node_type', 'halfedge_type', 'node_in', 'pos_in', 'halfedge_in', 'fixed_node', 'fixed_pos', 'fixed_halfedge', 'pocket_pos', 'pocket_center'):
                torch.testing.assert_close(observed[0][key], reference[key], rtol=0, atol=0)


@pytest.mark.parametrize('protocol', ['C0', 'C5'])
def test_full_sampling_loop_matches_official(prepared_data, official, baseline, protocol):
    config, featurizer, task, data = clean_sample(prepared_data, 'validation', protocol)
    data.node_pos.zero_()
    data.gt_node_pos.zero_()
    original = Batch.from_data_list([data.clone(), data.clone()], follow_batch=featurizer.follow_batch + ['pocket_pos'], exclude_keys=featurizer.exclude_keys + task.exclude_keys)
    sample_config = make_config(str(ROOT / 'configs/sample/test/dock_poseboff/base.yml')).noise
    sample_config.num_steps = 100

    def forward(batch):
        # 候选之间保留不同的非零质心, 输出明确依赖带噪坐标, 从而放大错误重新居中的影响.
        assert torch.count_nonzero(batch.gt_node_pos) == 0
        positions = batch.pos_in * .7 + torch.tensor([6., -2., 1.]) * (batch.node_type_batch[:, None] + 1)
        return dict(pred_node=torch.nn.functional.one_hot(batch.node_type, 12).float(), pred_pos=positions, pred_halfedge=torch.nn.functional.one_hot(batch.halfedge_type, 6).float(), confidence_pos=positions[:, :1], confidence_node=torch.ones(batch.num_nodes, 1), confidence_halfedge=torch.ones(len(batch.halfedge_type), 1))

    runs = []
    for modules in (None, official, baseline):
        factory = get_sample_noiser if modules is None else modules['utils/sample_noise.py'].get_sample_noiser
        loop = sample_loop3 if modules is None else modules['models/sample.py'].sample_loop3
        noiser = factory(deepcopy(sample_config), 12, 6, mode='sample', device='cpu', ref_config=config.noise)
        seed_all(19)
        runs.append(loop(original.clone(), forward, noiser, device='cpu', off_tqdm=True))
    for result in runs[1:]:
        for key in ('node_pos', 'pos_in', 'node_in', 'halfedge_in', 'pocket_pos', 'pocket_center'):
            torch.testing.assert_close(runs[0][0][key], result[0][key], rtol=0, atol=0)
        for key in runs[0][1]:
            torch.testing.assert_close(runs[0][1][key], result[1][key], rtol=0, atol=0)
        for key in runs[0][2]:
            assert runs[0][2][key].keys() == result[2][key].keys()
            for field in runs[0][2][key]:
                np.testing.assert_array_equal(runs[0][2][key][field], result[2][key][field])
    # 当前与官方解码均只加回一次模型原点.
    local = runs[0][0].node_pos[:data.num_nodes].numpy()
    arguments = dict(node=data.node_type.numpy(), pos=local, halfedge=data.halfedge_type.numpy(), halfedge_index=data.halfedge_index.numpy(), pocket_center=data.pocket_center.numpy())
    expected = official['utils/transforms.py'].FeaturizeMol(config.transforms.featurizer).decode_output(**arguments)
    actual = featurizer.decode_output(**arguments)
    np.testing.assert_array_equal(actual['atom_pos'], expected['atom_pos'])
    np.testing.assert_array_equal(actual['atom_pos'], local + data.pocket_center.numpy())


def test_original_prior_and_level_have_identical_executable_ast():
    # 包括GaussianExplodePrior的尺寸尺度和clamp、advance等级映射; 注释与Docstring不参与运行.
    for name in ('utils/prior.py', 'utils/info_level.py'):
        sources = [(ROOT / name).read_text(encoding='utf-8'), subprocess.check_output(['git', 'show', f'{OFFICIAL_REVISION}:{name}'], cwd=ROOT).decode('utf-8')]
        trees = []
        for source in sources:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                    node.body.pop(0)
            # 学习历史仅把GaussianExplodePrior类移到冷读顺序更靠前的位置; 逐定义比较, 其余顶层语句仍比较顺序.
            definitions = {node.name: ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
            statements = [ast.dump(node, include_attributes=False) for node in tree.body if not isinstance(node, (ast.ClassDef, ast.FunctionDef))]
            trees.append((definitions, statements))
        assert trees[0] == trees[1], name


@pytest.mark.parametrize('receptor_kind', ['protein', 'mixed', 'nucleic'])
def test_ra_forward_and_gradients_unchanged(prepared_data, baseline, receptor_kind):
    from models.maskfill import PMAsymDenoiser

    config, batch = assemble_batch(prepared_data, 'RA', receptor_kind)
    model = PMAsymDenoiser(config.model, 12, 6, 25).eval()
    previous = baseline['models/maskfill.py'].PMAsymDenoiser(config.model, 12, 6, 25).eval()
    previous.load_state_dict(model.state_dict(), strict=True)
    outputs = [network(batch.clone()) for network in (model, previous)]
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)
    for output in outputs:
        (output['pred_pos'].square().mean() + output['confidence_pos'].square().mean()).backward()
    old_parameters = dict(previous.named_parameters())
    for name, parameter in model.named_parameters():
        old = old_parameters[name]
        assert (parameter.grad is None) == (old.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, old.grad, rtol=0, atol=0)


@pytest.mark.skipif(not os.environ.get('PXM_ACCEPTANCE_ASSETS'), reason='需要只读复制的真实非测试训练资产')
@pytest.mark.parametrize('experiment', ['B-C-T0-RA', 'B-E-T0-RA'])
def test_real_train_t0_against_official(experiment, official):
    assets = Path(os.environ['PXM_ACCEPTANCE_ASSETS'])
    config = make_config(str(ROOT / f'configs/docking/{experiment}.yml'))
    config.data.dataset.update(root=str(assets / 'source'), derived_root=str(assets / 'derived'), manifest_root=str(assets / 'manifests'))
    module = DataModule(config)
    module.setup('fit')
    dataset = module.train_dataloader().dataset
    assert [(record['pdb_id'], record['candidate_id']) for record in dataset.records] == [('5ftl', 0)]
    seed_all(73)
    actual = dataset[0]
    noiser = official['utils/sample_noise.py'].get_sample_noiser(config.noise, 12, 6, mode='train')
    reference = OccurrenceDataset(config.data.dataset, 'train', Compose(module.transforms.transforms[:-1] + [noiser]), 'RA', dataset.protocol, True)
    seed_all(73)
    expected = reference[0]
    for key in ('pocket_center', 'pocket_pos', 'pocket_atom_feature', 'pocket_knn_edge_index', 'node_pos', 'node_type', 'halfedge_type', 'node_in', 'pos_in', 'halfedge_in', 'fixed_pos'):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    print(f'OFFICIAL_EQUIVALENCE {experiment} train/5ftl/0 atoms={actual.num_nodes} pocket={len(actual.pocket_pos)} exact=True')
