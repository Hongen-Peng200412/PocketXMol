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

from docking.assets import BOND_TYPES, CHIRAL_TYPES, read_receptor, select_pocket
from docking.dataset import EmptyEnvelopePocketError, OccurrenceDataset
from docking.smiles import read_smiles_graph, read_smiles_assets, read_coordinate_archive, UnsupportedSmilesError
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


def write_public_smiles(config, molecules):
    """仅构造测试夹具公共包, 用明确 RDKit 分子同时产生原子与排列字段."""
    root = Path(config.smiles_root)
    root.mkdir(parents=True)
    atoms, charges, edges, kinds, offsets, bond_offsets = [], [], [], [], [0], [0]
    smiles_values, shapes, matches, match_offsets = [], [], [], [0]
    for smiles in sorted({Chem.MolToSmiles(m, isomericSmiles=False) for m in molecules.values()}):
        mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
        smiles_values.append(smiles)
        atoms.extend(a.GetAtomicNum() for a in mol.GetAtoms())
        charges.extend(a.GetFormalCharge() for a in mol.GetAtoms())
        edges.extend((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
        kinds.extend(BOND_TYPES.index(b.GetBondType()) for b in mol.GetBonds())
        offsets.append(len(atoms)); bond_offsets.append(len(edges))
        full = np.array(mol.GetSubstructMatches(mol, uniquify=False, useChirality=True, maxMatches=10000), dtype=np.int32)
        compact = full[:, (full != np.arange(mol.GetNumAtoms())).any(0)]
        shapes.append(compact.shape); matches.extend(compact.ravel()); match_offsets.append(len(matches))
    np.savez_compressed(root / 'smiles_graphs_v1.npz', schema_version=np.int64(1), smiles=np.array(smiles_values), atom_offsets=np.array(offsets,dtype=np.int64), element=np.array(atoms,dtype=np.int16), charge=np.array(charges,dtype=np.int8), atom_in_ring=np.zeros((len(atoms),4),bool), bond_offsets=np.array(bond_offsets,dtype=np.int64), bond_index=np.array(edges,dtype=np.int32).T, bond_type=np.array(kinds,dtype=np.uint8), bond_in_ring=np.zeros((len(edges),4),bool))
    np.savez_compressed(root / 'smiles_symmetries_v1.npz',schema_version=np.int64(1),smiles=np.array(smiles_values),atom_count=np.diff(offsets).astype(np.int32),matches_shape=np.array(shapes,dtype=np.int32),matches_offsets=np.array(match_offsets,dtype=np.int64),matches_iso=np.array(matches,dtype=np.int32))


def refresh_smiles_coords(config, pdb_id):
    """只为测试将修改过的构造坐标同步到公共顺序包, 随后清除读取缓存."""
    root = Path(config.root)
    prepared_path = root / 'stage1_preparation_box_pool_2/ligand_language_models/prepared' / pdb_id / 'prepared_smiles.jsonl'
    prepared = {r['candidate_id']:r['smiles'] for r in read_jsonl(prepared_path)}
    with np.load(root / 'parse' / pdb_id / 'ligand_coords.npz') as archive:
        ids=sorted(int(k.split('_')[1]) for k in archive.files if k.startswith('coords_'))
        coords=[archive[f'coords_{cid}'] for cid in ids]
    # 测试新增的远离受体实例沿用第一个实例的化学图, 坐标单独修改.
    strings=[prepared.get(cid,prepared[0]) for cid in ids]
    target=Path(config.smiles_coords_root);target.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target/f'{pdb_id}.npz',schema_version=np.int64(1),candidate_ids=np.array(ids,dtype=np.int64),prepared_smiles=np.array(strings),coord_offsets=np.r_[0,np.cumsum([len(c) for c in coords])].astype(np.int64),coords=np.concatenate(coords).astype(np.float32))
    read_coordinate_archive.cache_clear()


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
    prepared_path = root / 'stage1_preparation_box_pool_2/ligand_language_models/prepared' / pdb_id
    prepared_path.mkdir(parents=True)
    write_jsonl(prepared_path / 'prepared_smiles.jsonl', [dict(candidate_id=r['candidate_id'], smiles=r['prepared_smiles']) for r in languages])
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
    config.smiles_root=str(root/'smiles_assets')
    config.smiles_coords_root=str(root/'smiles_assets/SMILE_coords/v1')
    Path(config.split_root).mkdir()
    molecules = {key: write_template(root, key, smiles) for key, smiles in [('CCD:ETH', 'CCO'), ('CCD:BEN', 'C1=CC=CC=C1'), ('CCD:ACE', 'CC(C)=O')]}
    write_public_smiles(config, molecules)
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
    for pdb_id in ['train_demo','val_demo','cal_demo','test_demo']:
        refresh_smiles_coords(config,pdb_id)
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
    assert not (Path(config.derived_root) / 'symmetries').exists()
    assert read_smiles_graph(config.smiles_root, 'c1ccccc1')['matches_iso'].shape[1] == 6
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
    assert all(len(item['views']) == 3 for item in records if item['prepared_smiles'] in {'c1ccccc1', 'CC(C)=O'})
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


@pytest.mark.parametrize('branch', ['RA', 'protein'])
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
        edge = sample.pocket_knn_edge_index
        assert torch.any(sample.pocket_is_nucleic[edge[0]] != sample.pocket_is_nucleic[edge[1]])
    assert len(read_receptor(Path(config.root) / 'parse/test_demo/receptor_tokens.npz')['coords']) == 80
    loader = DataLoader(dataset, batch_size=4, num_workers=2, follow_batch=featurizer.follow_batch + ['pocket_pos'], exclude_keys=featurizer.exclude_keys + task.exclude_keys)
    visited = [data_id for batch in loader for data_id in batch.data_id]
    assert len(visited) == len(set(visited)) == 27


@pytest.mark.parametrize('branch', ['RA'])
@pytest.mark.parametrize('split,shuffle', [('train', True), ('validation', False)])
def test_empty_envelope_is_skipped_before_training_or_validation_batch(prepared_data, branch, split, shuffle):
    """空E跳过后训练batch仍完整, 验证有限遍历其余实例, 原点与冻结清单保持正确."""
    config = EasyDict(deepcopy(dict(prepared_data)))
    config.update(pocket_mode='envelope', knn=32)
    manifest = Path(config.manifest_root) / f'{split}.jsonl'
    records = read_jsonl(manifest)
    coordinate_path = Path(config.root) / 'parse' / records[0]['pdb_id'] / 'ligand_coords.npz'
    with np.load(coordinate_path) as archive:
        coords = {key: archive[key] for key in archive.files}
    empty_id = max(int(key.split('_')[1]) for key in coords if key.startswith('coords_')) + 1
    coords[f'coords_{empty_id}'] = coords['coords_0'] + 1000
    coords[f'present_{empty_id}'] = np.ones(len(coords['coords_0']), dtype=bool)
    np.savez_compressed(coordinate_path, **coords)
    refresh_smiles_coords(config, records[0]['pdb_id'])
    records.append(dict(records[0], candidate_id=empty_id))
    write_jsonl(manifest, records)
    frozen_manifest = manifest.read_bytes()
    featurizer = FeaturizeMol(EasyDict(use_mask_node=True, use_mask_edge=True, chem=dict(atomic_numbers=[6, 7, 8, 9, 15, 16, 17, 5, 35, 53, 34], mol_bond_types=[1, 2, 3, 4])))
    task = ConfTransform(EasyDict(settings=dict(free=1.0), free_no_geometry=True), mode='test')
    dataset = OccurrenceDataset(config, split, Compose([featurizer, task]), branch, 'E', shuffle)
    with pytest.raises(EmptyEnvelopePocketError, match=f'{records[0]["pdb_id"]}/{empty_id}'):
        dataset[1]
    dataset.rng = np.random.default_rng(0)  # 首次抽中空实例, 必须跳过后继续组批.
    loader = DataLoader(dataset, batch_size=4, num_workers=0, follow_batch=featurizer.follow_batch + ['pocket_pos'], exclude_keys=featurizer.exclude_keys + task.exclude_keys)
    with pytest.warns(RuntimeWarning, match='empty_envelope_pocket'):
        batches = [next(iter(loader))] if shuffle else list(loader)
    assert len(batches) == 1
    batch = batches[0]
    assert batch.num_graphs == (4 if shuffle else 1)
    assert set(batch.candidate_id.tolist()) == {0}
    assert torch.isfinite(batch.node_pos).all() and torch.isfinite(batch.pocket_center).all()
    receptor = read_receptor(Path(config.root) / 'parse' / records[0]['pdb_id'] / 'receptor_tokens.npz')
    expected_origin = torch.from_numpy(receptor['coords'].mean(axis=0))
    torch.testing.assert_close(batch.pocket_center, expected_origin[None].expand(batch.num_graphs, -1))
    assert manifest.read_bytes() == frozen_manifest and len(dataset.records) == 2

    # 相同配体实例在C0下也选得空口袋, 但给定中心仍提供有限原点; E专属处理不改变该行为.
    center_config = deepcopy(config)
    center_config.pocket_mode = 'center'
    center_dataset = OccurrenceDataset(center_config, split, Compose([featurizer, task]), branch, 'C0', False)
    sample = center_dataset[1]
    assert len(sample.pocket_pos) == 0 and torch.isfinite(sample.node_pos).all()
    torch.testing.assert_close(sample.pocket_center[0], torch.from_numpy(coords[f'coords_{empty_id}'].mean(axis=0)))


def test_missing_public_graph_is_rejected_without_repair(prepared_data):
    root=Path(prepared_data.smiles_root)
    before=(root/'smiles_graphs_v1.npz').read_bytes()
    result=prepare_object(root,'C(C)(C)(C)(C)C')
    assert result['status']=='excluded'
    assert (root/'smiles_graphs_v1.npz').read_bytes()==before


def test_output_write_failure_is_not_a_scientific_exclusion(prepared_data, monkeypatch):
    config = prepared_data
    root, derived = Path(config.root), Path(config.derived_root)
    def fail_write(*args, **kwargs):
        raise OSError('constructed quota failure')
    objects = {record['prepared_smiles']: record for index in range(2) for record in read_jsonl(Path(config.manifest_root) / 'preparation' / f'objects_{index}.jsonl')}
    records = read_jsonl(Path(config.manifest_root) / 'train.jsonl')
    monkeypatch.setattr(np, 'save', fail_write)
    with pytest.raises(OSError, match='quota'):
        prepare_pdb(root, derived, Path(config.language_root), Path(config.smiles_root), Path(config.smiles_coords_root), records, objects)


@pytest.mark.skipif(not Path('/storage/penghongen/AdaLigand/Ori_Data').is_dir(), reason='需要服务器只读Ori_Data资产')
def test_real_source_preparation(tmp_path):
    """检查已核对的训练候选5ftl/0/CCD:ADP, 防止构造夹具复刻了错误的语言资产字段约定.

    读取真实模板、沉积坐标、语言向量、受体与两类地图; 仅把本实例派生标签写入pytest临时目录.
    """
    root = Path('/storage/penghongen/AdaLigand/Ori_Data')
    language_root = root / 'stage1_preparation_box_pool_2/ligand_language_models/smi_ted_289m'
    record=next(r for r in read_jsonl(Path('/storage/penghongen/PocketXMol/data/smiles-v1/frozen/train.jsonl')) if r['pdb_id']=='5ftl' and r['candidate_id']==0)
    for name in ('symmetries', 'ligand_area'):
        (tmp_path / name).mkdir()
    template = prepare_object(root/'smiles_assets', record['prepared_smiles'])
    assert template['status'] == 'ok', template
    kept, excluded = prepare_pdb(root, tmp_path, language_root, root/'smiles_assets', root/'smiles_assets/SMILE_coords/v1', [record], {record['prepared_smiles']: template})
    assert excluded == [], excluded
    assert kept == [dict(record, n_heavy_atoms=template['atom_count'])]
    assert (tmp_path / 'ligand_area/5ftl/0.npy').is_file()
