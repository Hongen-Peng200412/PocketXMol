"""核对CA2标准汇总与strongest-1端到端输入、迭代和计费契约。"""

import json
from copy import deepcopy
from pathlib import Path

import torch
from easydict import EasyDict
from rdkit import Chem
from torch_geometric.transforms import Compose

from docking.dataset import ConditionedDockingDataset, OccurrenceDataset
from docking.end_to_end import prepare_initial_records
from docking.evaluation import (
    matches_explicit_stereo,
    rank_candidate_metrics,
    score_saved_candidates,
)
from docking.final_evaluation import summarize_end_to_end
from docking.smiles import read_smiles_coords
from utils.misc import make_config
from utils.transforms import ConfTransform, FeaturizeMol
from test_docking_data import prepared_data


def test_smiles_template_only_checks_explicit_stereo():
    """未声明立体中心不判错，显式模板仍区分相反构型。"""
    unspecified = Chem.MolFromSmiles("CC(O)F")
    expected = Chem.MolFromSmiles("C[C@H](O)F")
    opposite = Chem.MolFromSmiles("C[C@@H](O)F")
    assert matches_explicit_stereo(opposite, unspecified)
    assert matches_explicit_stereo(expected, expected)
    assert not matches_explicit_stereo(opposite, expected)


def test_unmarked_smiles_template_keeps_stereo_term_in_self_ranking():
    """无显式立体标记时，端到端候选均获得冻结公式中的 stereo 项。"""
    template = Chem.MolFromSmiles("CC(O)F")
    poses = [Chem.Mol(template), Chem.Mol(template)]
    candidates = [
        {"sample_index": 0, "status": "success", "sdf_index": 0, "cfd_traj": 0.4},
        {"sample_index": 1, "status": "success", "sdf_index": 1, "cfd_traj": 0.2},
    ]
    receptor = Chem.MolFromSmiles("")
    metrics = score_saved_candidates(
        candidates, poses, receptor, template, rmsd_reference=None
    )
    assert [item["stereo"] for item in metrics] == [True, True]
    assert [item["self_ranking"] for item in metrics] == [1.4, 1.2]
    assert rank_candidate_metrics(metrics)[0]["sample_index"] == 0


def test_predicted_envelope_reuses_standard_envelope_condition(prepared_data):
    """同一坐标进入标准E与预测E时，口袋、原点、密度几何和模型输入逐项一致。"""
    root = Path(__file__).resolve().parents[1]
    training = make_config(str(root / "configs/docking/local_cov-E-T0-RA.yml"))
    featurizer = FeaturizeMol(training.transforms.featurizer)
    task = ConfTransform(
        EasyDict(settings={"free": 1.0}, free_no_geometry=True),
        mode="test",
    )
    transforms = Compose([featurizer, task])
    dataset_config = EasyDict(deepcopy(dict(prepared_data)))
    dataset_config.update(pocket_mode="envelope", knn=32)
    standard = OccurrenceDataset(
        dataset_config,
        "validation",
        transforms,
        "RA",
        "E",
        False,
        density_config=training.model.density,
    )
    source = standard.records[0]
    coordinates = read_smiles_coords(
        dataset_config.smiles_coords_root,
        source["pdb_id"],
        int(source["candidate_id"]),
        source["prepared_smiles"],
    )
    conditioned_config = EasyDict(
        root=dataset_config.root,
        smiles_root=dataset_config.smiles_root,
        pocket_mode="envelope",
        knn=32,
    )
    conditioned = ConditionedDockingDataset(
        conditioned_config,
        [
            {
                "pdb_id": source["pdb_id"],
                "candidate_id": int(source["candidate_id"]),
                "prepared_smiles": source["prepared_smiles"],
                "sampling_seed": 1,
                "views": [],
                "center_offset_xyz_A": [0.0, 0.0, 0.0],
                "envelope_coords_xyz_A": coordinates.tolist(),
            }
        ],
        transforms,
        "RA",
        density_config=training.model.density,
    )
    expected, actual = standard[0], conditioned[0]
    for field in (
        "pocket_pos",
        "pocket_center",
        "pocket_is_nucleic",
        "pocket_atom_feature",
        "pocket_nucleic_feature",
        "pocket_density_feature",
        "density_input",
        "density_origin",
        "density_basis",
        "density_start_zyx",
        "node_pos",
        "gt_node_pos",
    ):
        torch.testing.assert_close(expected[field], actual[field])
    assert expected.pocket_protein_count == actual.pocket_protein_count
    assert expected.pocket_nucleic_count == actual.pocket_nucleic_count
    assert "smiles_coords_root" not in conditioned.config


def test_initial_manifest_keeps_unselected_and_rank_beyond_twenty(tmp_path):
    """身份正确handoff不受selected、raw_rank或attempt_index 20截断。"""
    predictions = []
    trace = []
    handoff = []
    for source_blob_index, raw_rank, selected in ((4, 1, True), (9, 25, False)):
        predictions.append(
            {
                "pdb_id": "demo",
                "source_blob_index": source_blob_index,
                "centered_box_index": source_blob_index,
                "candidate_selected": selected,
                "predicted_smiles_small_molecule": "CCO",
                "target_smiles": "CCO",
                "matched_occurrence_id": source_blob_index + 100,
            }
        )
        trace.append(
            {
                "source_blob_index": source_blob_index,
                "centered_box_index": source_blob_index,
                "raw_rank": raw_rank,
                "attempt_index": raw_rank,
                "matched_occurrence_id": source_blob_index + 100,
                "target_scope": "stage3_foreground",
                "target_smiles": "CCO",
                "action": "spend_stage3_target",
                "success": True,
            }
        )
        handoff.append(
            {
                "sorting": "source_probability_mean",
                "answer_scope": "small_molecule",
                "receptor_condition": "real_receptor",
                "pdb_id": "demo",
                "source_blob_index": source_blob_index,
                "centered_box_index": source_blob_index,
                "raw_rank": raw_rank,
                "attempt_index": raw_rank,
                "predicted_smiles": "CCO",
                "blob_center_world_xyz_A": [1.0, 2.0, 3.0],
                "checkpoint_sha256": "旧Matcher字段不得进入新清单",
            }
        )
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "predictions": {"real_receptor": predictions},
                "traces": {
                    "source_probability_mean": {
                        "real_receptor": {"small_molecule": {"demo": trace}}
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    handoff_path = tmp_path / "handoff.jsonl"
    handoff_path.write_text(
        "".join(json.dumps(record) + "\n" for record in handoff),
        encoding="utf-8",
    )
    config = EasyDict(
        matcher_evaluation=str(evaluation_path),
        matcher_handoff=str(handoff_path),
        receptor_condition="real_receptor",
        seed_base=1000,
        expected_handoff_count=2,
        output_root=str(tmp_path / "output"),
    )
    records = prepare_initial_records(config)
    assert [record["source_blob_index"] for record in records] == [4, 9]
    assert records[1]["raw_rank"] == records[1]["attempt_index"] == 25
    assert not records[1]["candidate_selected"]
    assert all("sha" not in key.lower() for record in records for key in record)
    for stage in ("official-c", "local-c1"):
        stage_records = [
            json.loads(line)
            for line in (tmp_path / "output" / "inputs" / f"{stage}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert all("matched_occurrence_id" not in item for item in stage_records)
        assert all("target_smiles" not in item for item in stage_records)
    assert records[0]["stage_seeds"] == {
        "official-c": 1000,
        "local-c1": 1001,
        "local-c2": 1002,
        "local-e": 1003,
    }


def test_site_twenty_is_summary_boundary_not_candidate_boundary():
    """attempt 21保留在直接结果中，但不会误计入site@20。"""
    failed = {
        "success": {
            "2.0": {"1": False, "5": False, "50": False},
            "3.0": {"1": False, "5": False, "50": False},
        }
    }
    succeeded = deepcopy(failed)
    succeeded["success"]["2.0"]["1"] = True
    succeeded["success"]["3.0"]["1"] = True
    direct = [
        {
            "pdb_id": "demo",
            "attempt_index": 21,
            "methods": {
                name: deepcopy(succeeded if name == "local_cov-C" else failed)
                for name in ("official-C", "local_cov-C", "C-C", "C-C-E")
            },
        }
    ]
    summary = summarize_end_to_end(direct, ["demo"])
    assert summary["methods"]["local_cov-C"]["20"]["1"]["2.0"][
        "success_count"
    ] == 0
    assert len(direct) == 1 and direct[0]["attempt_index"] == 21
