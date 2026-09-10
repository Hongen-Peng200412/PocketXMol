"""用构造资产核对累计筛选、原子编号、口袋几何和测试视图; 另用明确训练实例核对源资产接线, 不读取held-out科学样本."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from easydict import EasyDict
from rdkit import Chem
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from docking.assets import BOND_TYPES, CHIRAL_TYPES, read_receptor, read_template, select_pocket
from docking.dataset import OccurrenceDataset
from docking.preparation import prepare, prepare_object, prepare_pdb, read_jsonl, write_jsonl
from utils.transforms import ConfTransform, FeaturizeMol


def write_template(root, object_key, smiles):
    molecule = Chem.MolFromSmiles(smiles)
    atoms = np.zeros(molecule.GetNumAtoms(), dtype=[('element', 'i1'), ('charge', 'i1'), ('chirality', '?', (7,))])
    for index, atom in enumerate(molecule.GetAtoms()):
        atoms[index]['element'] = atom.GetAtomicNum()
        atoms[index]['charge'] = atom.GetFormalCharge()
        atoms[index]['chirality'][CHIRAL_TYPES.index(atom.GetChiralTag())] = True
    bonds = np.zeros(molecule.GetNumBonds(), dtype=[('atom_1', 'i4'), ('atom_2', 'i4'), ('type', '?', (5,))])
    for index, bond in enumerate(molecule.GetBonds()):
        bonds[index]['atom_1'] = bond.GetBeginAtomIdx()
        bonds[index]['atom_2'] = bond.GetEndAtomIdx()
        bonds[index]['type'][BOND_TYPES.index(bond.GetBondType())] = True
    np.savez_compressed(root / 'ligand_objects' / (object_key.replace(':', '_') + '.npz'), atoms=atoms, bonds=bonds, atom_names=np.array([f'C{i}' for i in range(len(atoms))]), smiles=smiles)
    return molecule


def write_pdb(config, pdb_id, object_keys, molecules):
    root = Path(config.root)
    parse_dir, density_dir, language_dir = root / 'parse' / pdb_id, root / 'density' / pdb_id, Path(config.language_root) / pdb_id
    for path in (parse_dir, density_dir, language_dir):
        path.mkdir(parents=True)
    coordinates, labels, occurrences, languages = {}, {}, [], []
    for candidate_id, object_key in enumerate(object_keys):
        molecule = molecules[object_key]
        coords = np.zeros((molecule.GetNumAtoms(), 3), dtype=np.float32)
        coords[:, 0] = np.arange(len(coords)) * 1.2
        coordinates[f'coords_{candidate_id}'] = coords
        coordinates[f'present_{candidate_id}'] = np.ones(len(coords), dtype=bool)
        labels[f'mask_{candidate_id}'] = np.array([[1, 2, 3]], dtype=np.int32)
        occurrences.append(dict(candidate_id=candidate_id, object_key=object_key, pdb_id=pdb_id, type_tag='small_molecule', kind='CCD', polymer_length=1, is_covalent=False, components=[dict(ccd_id=object_key.split(':')[1])]))
        normalized = Chem.MolToSmiles(molecule, isomericSmiles=False)
        languages.append(dict(pdb_id=pdb_id, candidate_id=candidate_id, status='encoded', error=None, prepared_smiles=normalized, model_smiles=normalized, diagnostics=dict(official_normalization_success=True, official_normalization_error=None, unsupported_tokens=[], token_diagnostic_error=None, all_finite=True)))
        np.savez_compressed(language_dir / f'candidate_{candidate_id}.npz', pdb_id=pdb_id, candidate_id=candidate_id, object_key=object_key, model_name='SMI-TED Light 289M', prepared_smiles=normalized, model_smiles=normalized, embedding=np.ones(768, dtype=np.float32))
    write_jsonl(parse_dir / 'occurrences.jsonl', occurrences)
    write_jsonl(language_dir / 'results.jsonl', languages)
    np.savez_compressed(parse_dir / 'ligand_coords.npz', **coordinates)
    np.savez_compressed(density_dir / 'ligand_area.npz', **labels)
    # 40个标准蛋白、40个标准RNA/DNA和1个UNK; 分别保留供不同受体分支检查.
    coords = np.zeros((81, 3), dtype=np.float32)
    coords[:80, 0] = np.linspace(-2, 2, 80)
    coords[:80, 1] = np.sin(np.arange(80))
    coords[80] = [100, 100, 100]
    res_type = np.array(list(range(20)) * 2 + list(range(20, 28)) * 5 + [28], dtype=np.uint8)
    atom_name = np.array(['CA'] * 40 + ['N9', "C1'", 'OP1', "O3'", 'O1P'] * 8 + ['UNK'], dtype='S4')
    elements = np.array([6] * 40 + [7, 6, 8, 8, 8] * 8 + [6], dtype=np.uint8)
    np.savez_compressed(parse_dir / 'receptor_tokens.npz', coords=coords, element=elements, res_type=res_type, res_index=np.arange(81, dtype=np.int32), is_backbone=np.ones(81, dtype=bool), atom_name=atom_name)
    for name in ('exp', 'sim'):
        np.save(density_dir / f'{name}.npy', np.zeros((1, 48, 48, 48), dtype=np.float32))
        np.savez_compressed(density_dir / f'{name}.npz', origin=np.array([1., 2., 3.]), voxel_size=np.array([.9, 1., 1.1]))
    return occurrences


@pytest.fixture
def prepared_data(tmp_path):
    root = tmp_path / 'source'
    (root / 'ligand_objects').mkdir(parents=True)
    config = EasyDict(root=str(root), derived_root=str(tmp_path / 'derived'), manifest_root=str(tmp_path / 'manifests'), language_root=str(root / 'language'), split_root=str(root / 'split'), test_split=str(root / 'test.json'), workers=1, freeze_seed=3407, sampling_seed=10831)
    Path(config.split_root).mkdir()
    molecules = {key: write_template(root, key, smiles) for key, smiles in [('CCD:ETH', 'CCO'), ('CCD:BEN', 'C1=CC=CC=C1'), ('CCD:ACE', 'CC(C)=O')]}
    train = write_pdb(config, 'train_demo', ['CCD:ETH'] * 4, molecules)
    validation = write_pdb(config, 'val_demo', ['CCD:BEN'], molecules)
    calibration = write_pdb(config, 'cal_demo', ['CCD:ACE'], molecules)
    write_pdb(config, 'test_demo', ['CCD:ETH'] * 11 + ['CCD:BEN'] * 6 + ['CCD:ACE'] * 10, molecules)
    # 1缺失重原子、2语言模型失败、3不在质量清单; 只应保留训练0.
    coord_path = root / 'parse/train_demo/ligand_coords.npz'
    with np.load(coord_path) as archive:
        coords = {key: archive[key] for key in archive.files}
    coords['present_1'][0] = False
    coords['coords_1'][0] = np.nan
    np.savez_compressed(coord_path, **coords)
    language_path = root / 'language/train_demo/results.jsonl'
    language = read_jsonl(language_path)
    language[2].update(status='model_failed', error='constructed failure')
    write_jsonl(language_path, language)
    for split, records in [('train', train[:3]), ('validation', validation), ('calibration', calibration)]:
        (Path(config.split_root) / f'{split}.json').write_text(json.dumps(records), encoding='utf-8')
    Path(config.test_split).write_text(json.dumps(dict(pdb_ids=['test_demo'], occurrence_filter=None)), encoding='utf-8')
    prepare(config, 'index', 0, 1)
    for shard in range(2):
        prepare(config, 'objects', shard, 2)
    for shard in range(2):
        prepare(config, 'samples', shard, 2)
    prepare(config, 'freeze', 0, 2)
    return config


def test_cumulative_filters_and_shared_outputs(prepared_data):
    config = prepared_data
    root = Path(config.manifest_root)
    train = read_jsonl(root / 'train.jsonl')
    assert [(item['pdb_id'], item['candidate_id']) for item in train] == [('train_demo', 0)]
    excluded = read_jsonl(root / 'excluded.jsonl')
    assert {item['candidate_id'] for item in excluded} == {1, 2}
    assert any('incomplete_deposited_heavy_atoms' in item['reason'] for item in excluded)
    assert any('model_failed' in item['reason'] for item in excluded)
    assert len(list((Path(config.derived_root) / 'symmetries').glob('*.npz'))) == 3
    with np.load(Path(config.derived_root) / 'symmetries/CCD_BEN.npz') as archive:
        assert set(archive.files) == {'object_key', 'atom_count', 'matches_iso'}
        assert archive['matches_iso'].shape[1] == 6
    np.testing.assert_array_equal(np.load(Path(config.derived_root) / 'ligand_area/train_demo/0.npy'), [[1, 2, 3]])
    assert not (Path(config.derived_root) / 'ligand_area/train_demo/1.npy').exists()
    # 原资产仍保留缺失状态, 排除流程没有修复它.
    with np.load(Path(config.root) / 'parse/train_demo/ligand_coords.npz') as archive:
        assert not archive['present_1'][0]


def test_views_keep_six_to_ten_and_freeze_once(prepared_data):
    root = Path(prepared_data.manifest_root)
    records = read_jsonl(root / 'test.jsonl')
    assert len(records) == 27
    assert sum('CAP10' in item['views'] for item in records) == 26
    assert sum('HF10_TO5' in item['views'] for item in records) == 21
    assert all(len(item['views']) == 3 for item in records if item['object_key'] in {'CCD:BEN', 'CCD:ACE'})
    assert all('CAP10' in item['views'] for item in records if 'HF10_TO5' in item['views'])
    before = (root / 'test.jsonl').read_bytes()
    prepare(prepared_data, 'freeze', 0, 2)
    assert before == (root / 'test.jsonl').read_bytes()
    assert all(0 <= np.linalg.norm(item['center_offset_xyz_A']) <= 5 for item in records)


def test_residue_mass_center_and_strict_cutoffs():
    receptor = dict(res_index=np.array([10, 10, 20, 30]), element=np.array([6, 8, 6, 6]), coords=np.array([[14., 0, 0], [16., 0, 0], [14.999, 0, 0], [15., 0, 0]]))
    np.testing.assert_array_equal(select_pocket(receptor, np.zeros((1, 3)), np.zeros(3), 'center'), [False, False, True, False])
    receptor['coords'] -= np.array([5., 0, 0])
    np.testing.assert_array_equal(select_pocket(receptor, np.zeros((1, 3)), np.zeros(3), 'envelope'), [False, False, True, False])


@pytest.mark.parametrize('branch', ['RA', 'RB', 'protein'])
def test_original_transforms_and_worker_coverage(prepared_data, branch):
    config = EasyDict(deepcopy(dict(prepared_data)))
    config.update(pocket_mode='center', knn=32)
    featurizer = FeaturizeMol(EasyDict(use_mask_node=True, use_mask_edge=True, chem=dict(atomic_numbers=[6, 7, 8, 9, 15, 16, 17, 5, 35, 53, 34], mol_bond_types=[1, 2, 3, 4])))
    task = ConfTransform(EasyDict(settings=dict(free=1.0), free_no_geometry=True), mode='test')
    dataset = OccurrenceDataset(config, 'test', Compose([featurizer, task]), branch, 'C5', False)
    sample = dataset[0]
    torch.testing.assert_close(sample.node_pos.mean(0), -torch.tensor(dataset.records[0]['center_offset_xyz_A'], dtype=torch.float32), atol=1e-6, rtol=1e-6)
    assert sample.tor_bonds_anno.shape == (0, 3)
    assert sample.domain_node_index.shape == (2, 0)
    assert sample.fixed_halfdist_flex.sum() == 0
    assert sample.pocket_protein_count == 40 and sample.pocket_nucleic_count == 40
    if branch != 'protein':
        assert sample.pocket_nucleic_feature.shape == (80, 15)
        assert torch.all(sample.pocket_nucleic_feature[sample.pocket_is_nucleic].sum(-1) == 3)
        assert torch.all(sample.pocket_atom_feature[sample.pocket_is_nucleic] == 0)
        expected_first = torch.tensor([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.float32)
        expected_sugar = torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.float32)
        expected_old_phosphate = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], dtype=torch.float32)
        torch.testing.assert_close(sample.pocket_nucleic_feature[40], expected_first)
        torch.testing.assert_close(sample.pocket_nucleic_feature[41], expected_sugar)
        # 此处原子44的res_type=24, 即DA, 第8列是核苷酸DA通道.
        torch.testing.assert_close(sample.pocket_nucleic_feature[44], expected_old_phosphate)
        if branch == 'RB':
            edge = sample.pocket_knn_edge_index
            assert torch.equal(sample.pocket_is_nucleic[edge[0]], sample.pocket_is_nucleic[edge[1]])
    assert len(read_receptor(Path(config.root) / 'parse/test_demo/receptor_tokens.npz')['coords']) == 80
    loader = DataLoader(dataset, batch_size=4, num_workers=2, follow_batch=featurizer.follow_batch + ['pocket_pos'], exclude_keys=featurizer.exclude_keys + task.exclude_keys)
    visited = [data_id for batch in loader for data_id in batch.data_id]
    assert len(visited) == len(set(visited)) == 27


def test_bad_valence_is_rejected_without_repair(tmp_path):
    root, derived = tmp_path / 'source', tmp_path / 'derived'
    (root / 'ligand_objects').mkdir(parents=True)
    (derived / 'symmetries').mkdir(parents=True)
    write_template(root, 'CCD:BAD', 'CC')
    path = root / 'ligand_objects/CCD_BAD.npz'
    with np.load(path) as archive:
        original = {key: archive[key] for key in archive.files}
    original['atoms'] = np.repeat(original['atoms'][:1], 6)
    original['atom_names'] = np.array([f'C{i}' for i in range(6)])
    original['bonds'] = np.repeat(original['bonds'][:1], 5)
    original['bonds']['atom_1'] = 0
    original['bonds']['atom_2'] = np.arange(1, 6)
    np.savez_compressed(path, **original)
    result = prepare_object(root, derived, 'CCD:BAD')
    assert result['status'] == 'excluded'
    assert not (derived / 'symmetries/CCD_BAD.npz').exists()
    with np.load(path) as archive:
        assert archive['atoms']['charge'][0] == 0
        np.testing.assert_array_equal(archive['bonds']['atom_2'], np.arange(1, 6))


def test_output_write_failure_is_not_a_scientific_exclusion(prepared_data, monkeypatch):
    config = prepared_data
    root, derived = Path(config.root), Path(config.derived_root)
    def fail_write(*args, **kwargs):
        raise OSError('constructed quota failure')
    with monkeypatch.context() as patch:
        patch.setattr(np, 'savez_compressed', fail_write)
        with pytest.raises(OSError, match='quota'):
            prepare_object(root, derived, 'CCD:ETH')
    objects = {record['object_key']: record for index in range(2) for record in read_jsonl(Path(config.manifest_root) / 'preparation' / f'objects_{index}.jsonl')}
    records = read_jsonl(Path(config.manifest_root) / 'train.jsonl')
    monkeypatch.setattr(np, 'save', fail_write)
    with pytest.raises(OSError, match='quota'):
        prepare_pdb(root, derived, Path(config.language_root), records, objects)


@pytest.mark.skipif(not Path('/storage/penghongen/AdaLigand/Ori_Data').is_dir(), reason='需要服务器只读Ori_Data资产')
def test_real_source_preparation(tmp_path):
    """检查已核对的训练候选5ftl/0/CCD:ADP, 防止构造夹具复刻了错误的语言资产字段约定.

    读取真实模板、沉积坐标、语言向量、受体与两类地图; 仅把本实例派生标签写入pytest临时目录.
    """
    root = Path('/storage/penghongen/AdaLigand/Ori_Data')
    language_root = root / 'stage1_preparation_box_pool_2/ligand_language_models/smi_ted_289m'
    record = dict(split='train', pdb_id='5ftl', candidate_id=0, object_key='CCD:ADP')
    for name in ('symmetries', 'ligand_area'):
        (tmp_path / name).mkdir()
    template = prepare_object(root, tmp_path, record['object_key'])
    assert template['status'] == 'ok', template
    kept, excluded = prepare_pdb(root, tmp_path, language_root, [record], {record['object_key']: template})
    assert excluded == [], excluded
    assert kept == [dict(record, n_heavy_atoms=template['atom_count'])]
    assert (tmp_path / 'ligand_area/5ftl/0.npy').is_file()
