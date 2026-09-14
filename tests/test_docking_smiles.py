"""检查候选 SMILES 读取链的身份、芳香显式氢和失败分母; 不宣称旧模型键类别等价."""
from pathlib import Path
from copy import deepcopy

import numpy as np
import pytest
from easydict import EasyDict
from rdkit import Chem

from docking.dataset import OccurrenceDataset
from docking.smiles import UnsupportedSmilesError, read_smiles_assets, read_smiles_graph, read_coordinate_archive, read_smiles_coords
from test_docking_data import prepared_data, write_public_smiles


def test_dataset_never_opens_legacy_graph_or_coordinates(prepared_data, monkeypatch):
    config=EasyDict(deepcopy(dict(prepared_data)))
    config.update(pocket_mode='center',knn=32)
    read_smiles_assets.cache_clear();read_smiles_graph.cache_clear();read_coordinate_archive.cache_clear()
    original=np.load
    def checked(path,*args,**kwargs):
        name=str(path).replace('\\','/')
        assert '/ligand_objects/' not in name and not name.endswith('/ligand_coords.npz')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',checked)
    dataset=OccurrenceDataset(config,'train',lambda value:value,'RA','C0',False)
    sample=dataset[0]
    assert sample.prepared_smiles=='CCO' and sample.num_atoms==3
    np.testing.assert_array_equal(sample.element.numpy(),[6,6,8])


def test_aromatic_explicit_hydrogen_survives_molecule_reconstruction(tmp_path):
    config=EasyDict(smiles_root=str(tmp_path/'public'))
    molecule=Chem.MolFromSmiles('[nH]1cccc1')
    write_public_smiles(config,{'pyrrole':molecule})
    smiles=Chem.MolToSmiles(molecule,isomericSmiles=False)
    graph=read_smiles_graph(config.smiles_root,smiles)
    assert graph['mol'].GetNumAtoms()==5
    assert next(atom.GetTotalNumHs() for atom in graph['mol'].GetAtoms() if atom.GetAtomicNum()==7)==1
    assert np.all(graph['bond_type']==4)
    assert read_smiles_graph(config.smiles_root,smiles) is graph


def test_coordinates_reject_another_exact_smiles(prepared_data):
    with pytest.raises(UnsupportedSmilesError,match='identity_mismatch'):
        read_smiles_coords(prepared_data.smiles_coords_root,'train_demo',0,'OCC')


def test_explicit_unsupported_instance_preserves_records_and_test_failure(prepared_data):
    config=EasyDict(deepcopy(dict(prepared_data)))
    config.update(pocket_mode='center',knn=32)
    dataset=OccurrenceDataset(config,'train',lambda value:value,'RA','C0',False)
    dataset.records.append(dict(dataset.records[0],unsupported_smiles_reason='constructed unsupported graph'))
    with pytest.warns(RuntimeWarning,match='constructed unsupported graph'):
        assert len(list(dataset))==1
    assert len(dataset.records)==2
    dataset.split='test'
    with pytest.raises(UnsupportedSmilesError,match='constructed unsupported graph'):
        dataset[1]
