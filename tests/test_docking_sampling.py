"""以构造的非测试资产检查真实采样循环、逐候选失败和完整核酸碰撞评价."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from easydict import EasyDict
from rdkit import Chem
from torch_geometric.transforms import Compose

from docking.assets import read_receptor, read_template
from docking.dataset import OccurrenceDataset
from docking.evaluation import evaluate_docking, evaluate_occurrence, summarize_occurrences
from docking.sampling import sample_occurrence
from models.maskfill import PMAsymDenoiser
from utils.misc import make_config
from utils.sample_noise import get_sample_noiser
from utils.transforms import ConfTransform, FeaturizeMol
from test_docking_data import prepared_data


def sampling_context(prepared_data, branch, split):
    root = Path(__file__).resolve().parents[1]
    training = make_config(str(root / 'configs/docking/B-C-T0-RA.yml'))
    training.model.nucleic_branch = None if branch == 'protein' else branch
    featurizer = FeaturizeMol(training.transforms.featurizer)
    transforms = Compose([featurizer, ConfTransform(EasyDict(settings=dict(free=1.0), free_no_geometry=True), mode='test')])
    dataset_config = EasyDict(deepcopy(dict(prepared_data)))
    dataset_config.update(pocket_mode='center', knn=32)
    dataset = OccurrenceDataset(dataset_config, split, transforms, branch, 'C5', False)
    config = EasyDict(model_name=f'constructed_{branch}', split=split, output_root=str(Path(prepared_data.manifest_root).parent / f'sampling_{branch}'), dataset=dataset_config, num_candidates=2, num_steps=3, batch_size=2, device='cpu')
    sample_config = make_config(str(root / 'configs/sample/test/dock_poseboff/base.yml'))
    sample_config.noise.num_steps = config.num_steps
    noiser = get_sample_noiser(sample_config.noise, featurizer.num_node_types, featurizer.num_edge_types, mode='sample', device='cpu', ref_config=training.noise)
    return training, dataset, featurizer, config, noiser


@pytest.mark.parametrize('branch', ['protein', 'RA'])
def test_real_loop_and_resume_from_completed_occurrence(prepared_data, branch, monkeypatch):
    training, dataset, featurizer, config, noiser = sampling_context(prepared_data, branch, 'validation')
    model = PMAsymDenoiser(training.model, featurizer.num_node_types, featurizer.num_edge_types, 25).eval()
    result = sample_occurrence(dataset, 0, model, noiser, featurizer, config, 'C5')
    assert result['status'] == 'success', result
    assert result['success_count'] == 2
    assert result['sampling_batch_attempt_count'] == result['sampling_batch_completed_count'] == 1
    assert result['model_forward_attempt_count'] == result['model_forward_completed_count'] == 3
    directory = Path(config.output_root) / 'validation/C5/val_demo/0'
    candidates = json.loads((directory / 'candidates.json').read_text())
    assert [item['sdf_index'] for item in candidates] == [0, 1]
    with np.load(directory / 'confidence.npz') as archive:
        assert archive['confidence_pos_traj'].shape == (2, 6, 3)
        np.testing.assert_array_equal(archive['sample_index'], [0, 1])
    poses = list(Chem.SDMolSupplier(str(directory / 'poses.sdf')))
    assert all(molecule.GetNumAtoms() == 6 and molecule.GetNumBonds() == 6 for molecule in poses)
    before = (directory / 'poses.sdf').read_bytes()

    def unexpected_forward(*args, **kwargs):
        raise AssertionError('Completed occurrence must not run again')

    monkeypatch.setattr(model, 'forward', unexpected_forward)
    assert sample_occurrence(dataset, 0, model, noiser, featurizer, config, 'C5') == result
    assert (directory / 'poses.sdf').read_bytes() == before


@pytest.mark.parametrize('branch', ['RA'])
def test_empty_envelope_sampling_keeps_failure_denominator(prepared_data, branch):
    """构造验证资产模拟正式E输入失败, 不调用forward且完整保存候选失败与评价分母."""
    _, dataset, featurizer, config, _ = sampling_context(prepared_data, branch, 'validation')
    dataset.config.pocket_mode = 'envelope'
    dataset.protocol = 'E'
    receptor_path = Path(prepared_data.root) / 'parse/val_demo/receptor_tokens.npz'
    with np.load(receptor_path) as archive:
        receptor = {key: archive[key] for key in archive.files}
    receptor['coords'] += 1000
    np.savez_compressed(receptor_path, **receptor)
    read_receptor.cache_clear()
    frozen_manifest = (Path(prepared_data.manifest_root) / 'validation.jsonl').read_bytes()
    result = sample_occurrence(dataset, 0, None, None, featurizer, config, 'E')
    assert result['complete'] and result['status'] == 'failed'
    assert result['success_count'] == 0 and result['failed_count'] == config.num_candidates
    assert result['model_forward_attempt_count'] == 0
    directory = Path(config.output_root) / 'validation/E/val_demo/0'
    candidates = json.loads((directory / 'candidates.json').read_text())
    assert len(candidates) == config.num_candidates
    assert all(item['stage'] == 'preprocess' and 'empty_envelope_pocket: val_demo/0' in item['error'] for item in candidates)
    assessment = evaluate_occurrence((config, 'E', dataset.records[0]))
    summary = summarize_occurrences([assessment])
    assert assessment['generated_count'] == assessment['rmsd_count'] == 0
    assert assessment['candidate_error_counts'] == {'preprocess': config.num_candidates}
    assert summary['occurrence_count'] == 1 and summary['candidate_count'] == config.num_candidates
    assert summary['occurrence_equal']['top1_success_rate'] == 0
    assert (Path(prepared_data.manifest_root) / 'validation.jsonl').read_bytes() == frozen_manifest


def test_official_pure_rna_attempt_is_recorded(prepared_data):
    receptor_path = Path(prepared_data.root) / 'parse/val_demo/receptor_tokens.npz'
    with np.load(receptor_path) as archive:
        receptor = {key: archive[key] for key in archive.files}
    keep = (receptor['res_type'] >= 20) & (receptor['res_type'] < 28)
    np.savez_compressed(receptor_path, **{key: value[keep] for key, value in receptor.items()})
    read_receptor.cache_clear()
    training, dataset, featurizer, config, noiser = sampling_context(prepared_data, 'protein', 'validation')
    model = PMAsymDenoiser(training.model, featurizer.num_node_types, featurizer.num_edge_types, 25).eval()
    result = sample_occurrence(dataset, 0, model, noiser, featurizer, config, 'C5')
    assert result['complete'] and result['num_candidates'] == 2
    # 允许原后端成功或失败, 只核对真实调用状态和逐候选记录, 不预设失败发生在哪个阶段.
    directory = Path(config.output_root) / 'validation/C5/val_demo/0'
    candidates = json.loads((directory / 'candidates.json').read_text())
    assert len(candidates) == 2
    for candidate in candidates:
        assert candidate['status'] in ('success', 'failed')
        assert candidate['error'] is not None if candidate['status'] == 'failed' else candidate['error'] is None
    assert result['model_forward_attempt_count'] >= result['model_forward_completed_count']
    assessment = evaluate_occurrence((config, 'C5', dataset.records[0]))
    assert assessment['pocket_protein_count'] == 0
    assert assessment['pocket_nucleic_fraction'] == 1.0
    assert assessment['num_candidates'] == 2
    if result['success_count'] == 0:
        assert not assessment['top1_success']


def test_complete_nucleic_clashes_unaligned_rmsd_and_failure_denominator(prepared_data):
    _, dataset, _, config, _ = sampling_context(prepared_data, 'protein', 'train')
    config.num_candidates = 3
    record = dataset.records[0]
    receptor_path = Path(prepared_data.root) / 'parse/train_demo/receptor_tokens.npz'
    with np.load(receptor_path) as archive:
        receptor = {key: archive[key] for key in archive.files}
    receptor['coords'][:40] = [1000., 0., 0.]
    receptor['coords'][40:80] = [50., 0., 0.]
    receptor['coords'][80] = [0., 0., 0.]  # UNK位于真值附近, 仍必须排除.
    np.savez_compressed(receptor_path, **receptor)
    read_receptor.cache_clear()
    directory = Path(config.output_root) / 'train/C0/train_demo/0'
    directory.mkdir(parents=True)
    _, _, template, _ = read_template(Path(prepared_data.root) / 'ligand_objects/CCD_ETH.npz')
    with np.load(Path(prepared_data.root) / 'parse/train_demo/ligand_coords.npz') as archive:
        truth = archive['coords_0']
    candidates = []
    with Chem.SDWriter(str(directory / 'poses.sdf')) as writer:
        for index, shift in enumerate([0., 50.]):
            molecule = Chem.Mol(template)
            conformer = Chem.Conformer(len(truth))
            for atom_index, position in enumerate(truth + [shift, 0., 0.]):
                conformer.SetAtomPosition(atom_index, position.tolist())
            molecule.AddConformer(conformer)
            molecule.SetIntProp('sample_index', index)
            writer.write(molecule)
            candidates.append(dict(sample_index=index, status='success', stage='complete', error=None, sdf_index=index, cfd_traj=float(1-index)))
    candidates.append(dict(sample_index=2, status='failed', stage='forward', error='constructed failure', sdf_index=None, cfd_traj=None))
    (directory / 'candidates.json').write_text(json.dumps(candidates))
    sampling = dict(pdb_id='train_demo', occurrence_id=0, model_name=config.model_name, split='train', protocol='C0', object_key=record['object_key'], num_candidates=3, status='partial', candidate_file='candidates.json', pose_file='poses.sdf', inference_seconds=2., elapsed_seconds=3., sampling_batch_attempt_count=2, sampling_batch_completed_count=1, model_forward_attempt_count=4, model_forward_completed_count=3, peak_memory_allocated_bytes=None, peak_memory_reserved_bytes=None)
    (directory / 'result.json').write_text(json.dumps(sampling))
    assessment = evaluate_occurrence((config, 'C0', record))
    metrics = json.loads((directory / 'candidate_metrics.json').read_text())
    assert metrics[0]['no_clashes']
    assert not metrics[1]['no_clashes'] and metrics[1]['num_clashes'] > 0
    assert metrics[0]['rmsd_A'] == pytest.approx(0., abs=1e-6)
    assert metrics[1]['rmsd_A'] == pytest.approx(50., abs=1e-6)
    assert assessment['pocket_nucleic_count'] == 0  # 碰撞检查仍包含口袋外的标准核酸.
    assert assessment['spearman'] == pytest.approx(1.) and assessment['pose_auc'] == 1.
    failed = deepcopy(assessment)
    failed.update(pdb_id='failed_demo', generated_count=0, rmsd_count=0, rank_pair_count=0, top1_success=False, top5_success=False, oracle_success=False, top1_rmsd_A=None, top5_rmsd_A=None, oracle_rmsd_A=None, spearman=None, pose_auc=None, spearman_na_reason='too_few_candidates', pose_auc_na_reason='no_valid_candidates')
    summary = summarize_occurrences([assessment, failed])
    assert summary['occurrence_count'] == 2 and summary['candidate_count'] == 6
    assert summary['occurrence_equal']['top1_success_rate'] == .5
    assert summary['occurrence_equal']['spearman_valid_count'] == summary['occurrence_equal']['spearman_na_count'] == 1


def test_storage_errors_do_not_create_completion_markers(prepared_data, monkeypatch):
    _, dataset, featurizer, config, noiser = sampling_context(prepared_data, 'RA', 'validation')

    def unavailable_storage(*args, **kwargs):
        raise OSError('constructed storage outage')

    monkeypatch.setattr('docking.sampling.read_template', unavailable_storage)
    with pytest.raises(OSError, match='storage outage'):
        sample_occurrence(dataset, 0, None, noiser, featurizer, config, 'C5')
    directory = Path(config.output_root) / 'validation/C5/val_demo/0'
    assert not (directory / 'result.json').exists()
    monkeypatch.setattr('docking.evaluation.read_receptor', unavailable_storage)
    with pytest.raises(OSError, match='storage outage'):
        evaluate_occurrence((config, 'C5', dataset.records[0]))
    assert not (directory / 'assessment.json').exists()


def test_evaluation_rejects_another_candidate_pool(prepared_data):
    _, _, _, config, _ = sampling_context(prepared_data, 'RA', 'validation')
    directory = Path(config.output_root) / 'validation'
    directory.mkdir(parents=True)
    (directory / 'run.json').write_text(json.dumps({'science_config': {'model_name': 'a_different_model'}}))
    with pytest.raises(ValueError, match='differs_from_sampling'):
        evaluate_docking(config)
    assert not (directory / 'summary.json').exists()
