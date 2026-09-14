"""一次任务: 对冻结实例迁移 SMILES 顺序坐标并逐图核对自同构; 不重新筛选或冻结."""
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from docking.smiles import read_smiles_graph
from legacy_assets import read_template

ROOT = Path('/storage/penghongen/AdaLigand/Ori_Data')
COORDS = ROOT / 'smiles_assets/SMILE_coords/v1'
OLD = Path('/storage/penghongen/PocketXMol/data')
NEW = OLD / 'smiles-v1/frozen'
OUT = Path('/storage/penghongen/tmp/pxm_pre_smiles_20260914/cpu-v1')
PREPARED = ROOT / 'stage1_preparation_box_pool_2/ligand_language_models/prepared'
DERIVED = Path('/storage/penghongen/Adaligand_Build/Ori_Data/pocketxmol')
PAIR_CACHE = {}


def full_matches(compressed, n):
    full = np.tile(np.arange(n, dtype=np.int64), (len(compressed), 1))
    if compressed.size:
        full[:, np.unique(compressed)] = compressed
    return full


def pair_mapping(smiles, object_key):
    key = (smiles, object_key)
    if key in PAIR_CACHE:
        return PAIR_CACHE[key]
    graph = read_smiles_graph(ROOT / 'smiles_assets', smiles)
    atoms, _, old_mol, _ = read_template(ROOT / 'ligand_objects' / (object_key.replace(':', '_') + '.npz'))
    new_mol = graph['mol']
    if old_mol.GetNumAtoms() != new_mol.GetNumAtoms() or old_mol.GetNumBonds() != new_mol.GetNumBonds():
        raise ValueError('graph_size_mismatch')
    mappings = old_mol.GetSubstructMatches(new_mol, uniquify=False, useChirality=True, maxMatches=10000)
    if not mappings or not new_mol.HasSubstructMatch(old_mol, useChirality=True):
        raise ValueError('graph_not_isomorphic')
    mapping = np.asarray(min(mappings), dtype=np.int64)
    with np.load(DERIVED / 'symmetries' / (object_key.replace(':', '_') + '.npz'), allow_pickle=False) as archive:
        old_matches = full_matches(archive['matches_iso'], len(atoms))
    new_matches = full_matches(graph['matches_iso'], len(atoms))
    inverse = np.argsort(mapping)
    transported = inverse[old_matches[:, mapping]]
    if set(map(tuple, transported)) != set(map(tuple, new_matches)):
        raise ValueError(f'symmetry_set_mismatch:{len(old_matches)}:{len(new_matches)}')
    PAIR_CACHE[key] = mapping
    return mapping


def migrate_pdb(arguments):
    pdb_id, records = arguments
    prepared = {int(item['candidate_id']): item['smiles'] for item in map(json.loads, (PREPARED / pdb_id / 'prepared_smiles.jsonl').read_text().splitlines()) if item}
    migrated, evidence, failures = [], [], []
    candidate_ids, smiles_values, coords, offsets = [], [], [], [0]
    with np.load(ROOT / 'parse' / pdb_id / 'ligand_coords.npz', allow_pickle=False) as archive:
        for record in sorted(records, key=lambda value: value['candidate_id']):
            cid = int(record['candidate_id'])
            smiles = prepared[cid]
            new_record = {key: value for key, value in record.items() if key != 'object_key'}
            new_record['prepared_smiles'] = smiles
            try:
                mapping = pair_mapping(smiles, record['object_key'])
                old_coords = archive[f'coords_{cid}']
                if not archive[f'present_{cid}'].all() or not np.isfinite(old_coords).all():
                    raise ValueError('incomplete_or_nonfinite_coords')
                new_coords = old_coords[mapping].astype(np.float32)
                center_error = float(np.max(np.abs(new_coords.mean(0) - old_coords.mean(0))))
                candidate_ids.append(cid)
                smiles_values.append(smiles)
                coords.append(new_coords)
                offsets.append(offsets[-1] + len(new_coords))
                evidence.append(dict(pdb_id=pdb_id, candidate_id=cid, split=record['split'], object_key=record['object_key'], prepared_smiles=smiles, new_to_old=mapping.tolist(), center_max_abs_A=center_error))
            except (ValueError, KeyError, RuntimeError) as error:
                new_record['unsupported_smiles_reason'] = str(error)
                failures.append(dict(pdb_id=pdb_id, candidate_id=cid, split=record['split'], reason=str(error)))
            migrated.append(new_record)
    destination = COORDS / f'{pdb_id}.npz'
    with destination.open('xb') as handle:
        np.savez_compressed(handle, schema_version=np.int64(1), candidate_ids=np.asarray(candidate_ids, dtype=np.int64), prepared_smiles=np.asarray(smiles_values, dtype=np.str_), coord_offsets=np.asarray(offsets, dtype=np.int64), coords=np.concatenate(coords) if coords else np.empty((0, 3), dtype=np.float32))
    return migrated, evidence, failures


def main():
    started = time.time()
    OUT.mkdir(exist_ok=True, parents=True)
    COORDS.mkdir(exist_ok=True, parents=True)
    NEW.mkdir(exist_ok=False)
    by_pdb = defaultdict(list)
    originals = {}
    for split in ('train', 'validation', 'calibration', 'test'):
        originals[split] = [json.loads(line) for line in (OLD / f'{split}.jsonl').read_text().splitlines() if line]
        for record in originals[split]:
            by_pdb[record['pdb_id']].append(record)
    migrated = {}
    failures = []
    count = 0
    with (OUT / 'mapping.jsonl').open('x') as evidence_file, concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        for result, evidence, excluded in pool.map(migrate_pdb, sorted(by_pdb.items()), chunksize=1):
            for record in result:
                migrated[(record['split'], record['pdb_id'], record['candidate_id'])] = record
            for record in evidence:
                evidence_file.write(json.dumps(record) + '\n')
            failures.extend(excluded)
            count += len(result)
            if count % 1000 < len(result):
                print(json.dumps(dict(checked=count, failed=len(failures), elapsed_seconds=time.time()-started)), flush=True)
    for split, records in originals.items():
        with (NEW / f'{split}.jsonl').open('x') as stream:
            for record in records:
                stream.write(json.dumps(migrated[(split, record['pdb_id'], record['candidate_id'])]) + '\n')
    report = dict(checked=count, failed=len(failures), failure_fraction=len(failures)/count, elapsed_seconds=time.time()-started, failures=failures, splits={key:len(value) for key,value in originals.items()}, source_manifest=str(OLD), output_manifest=str(NEW), coords_root=str(COORDS), smiles_graph_sha256=hashlib.sha256((ROOT/'smiles_assets/smiles_graphs_v1.npz').read_bytes()).hexdigest())
    (OUT / 'migration_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if report['failure_fraction'] >= .01:
        raise RuntimeError('Unsupported SMILES fraction reaches 1 percent')


if __name__ == '__main__':
    main()
