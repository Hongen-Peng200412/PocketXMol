"""把Matcher strongest-1候选接入PocketXMol的分阶段端到端推理。

本模块只负责三项工作：从Matcher正式产物冻结身份正确的docking输入、根据上一轮
self-ranking生成下一轮中心或包络输入、调用原PocketXMol采样循环执行一个明确阶段。
RMSD、77个PDB分母和完整Matcher轨迹合并由 ``docking.final_evaluation`` 负责。
"""

import json
from pathlib import Path

import numpy as np
from easydict import EasyDict
from rdkit import Chem

from docking.assets import read_receptor
from docking.dataset import ConditionedDockingDataset
from docking.evaluation import (
    build_receptor_molecule,
    rank_candidate_metrics,
    score_saved_candidates,
)
from docking.final_artifacts import (
    candidate_output_dir,
    read_jsonl,
    stage_input_path,
    stage_output_root,
    write_json,
    write_jsonl,
)
from docking.sampling import (
    build_sampling_noiser,
    load_sampling_runtime,
    sample_occurrence,
)
from docking.smiles import read_smiles_graph


STAGES = ("official-c", "local-c1", "local-c2", "local-e")
STAGE_SEED_OFFSET = {stage: offset for offset, stage in enumerate(STAGES)}


def _model_key(stage):
    """把推理阶段映射到配置中的冻结模型名称。"""
    if stage == "official-c":
        return "official"
    if stage in ("local-c1", "local-c2"):
        return "local_c"
    if stage == "local-e":
        return "local_e"
    raise ValueError(f"未知端到端阶段: {stage}")


def _stage_record(base_record, stage):
    """只复制推理必需字段, 评价真值保留在 base.jsonl."""
    record = {
        key: base_record[key]
        for key in (
            "receptor_condition",
            "pdb_id",
            "candidate_id",
            "source_blob_index",
            "centered_box_index",
            "prepared_smiles",
            "blob_center_world_xyz_A",
            "stage_seeds",
            "views",
            "center_offset_xyz_A",
        )
    }
    record["sampling_seed"] = int(record["stage_seeds"][stage])
    record["given_center_xyz_A"] = record["blob_center_world_xyz_A"]
    return record


def prepare_initial_records(config):
    """冻结strongest-1中全部身份正确候选，并建立official-c与local-c1输入。

    只读取 ``source_probability_mean``、``small_molecule`` 和当前 ``receptor_condition``。
    Matcher handoff必须能按 ``(pdb_id, source_blob_index)`` 唯一回连prediction与完整trace；
    记录不保留Matcher文件哈希、checkpoint哈希、运行环境或耗时。
    """
    evaluation = json.loads(Path(config.matcher_evaluation).read_text(encoding="utf-8"))
    handoff = [
        record
        for record in read_jsonl(config.matcher_handoff)
        if record["sorting"] == "source_probability_mean"
        and record["answer_scope"] == "small_molecule"
        and record["receptor_condition"] == config.receptor_condition
    ]
    predictions = {
        (record["pdb_id"], int(record["source_blob_index"])): record
        for record in evaluation["predictions"][config.receptor_condition]
    }
    traces = evaluation["traces"]["source_probability_mean"][
        config.receptor_condition
    ]["small_molecule"]
    trace_items = {
        (pdb_id, int(item["source_blob_index"])): item
        for pdb_id, items in traces.items()
        for item in items
    }
    handoff.sort(
        key=lambda record: (
            record["pdb_id"],
            int(record["raw_rank"]),
            int(record["source_blob_index"]),
        )
    )
    seen = set()
    records = []
    for record_index, handoff_record in enumerate(handoff):
        key = (
            handoff_record["pdb_id"],
            int(handoff_record["source_blob_index"]),
        )
        if key in seen:
            raise ValueError(f"重复handoff候选: {key}")
        seen.add(key)
        prediction = predictions[key]
        trace = trace_items[key]
        predicted_smiles = handoff_record["predicted_smiles"]
        if (
            prediction["predicted_smiles_small_molecule"] != predicted_smiles
            or prediction["target_smiles"] != predicted_smiles
            or not trace["success"]
            or trace["matched_occurrence_id"] != prediction["matched_occurrence_id"]
        ):
            raise ValueError(f"handoff与Matcher完整轨迹不一致: {key}")
        stage_seeds = {
            stage: int(config.seed_base + record_index * len(STAGES) + offset)
            for stage, offset in STAGE_SEED_OFFSET.items()
        }
        records.append(
            {
                "receptor_condition": config.receptor_condition,
                "pdb_id": key[0],
                "candidate_id": key[1],
                "source_blob_index": key[1],
                "centered_box_index": int(handoff_record["centered_box_index"]),
                "candidate_selected": bool(prediction["candidate_selected"]),
                "raw_rank": int(trace["raw_rank"]),
                "attempt_index": int(trace["attempt_index"]),
                "matched_occurrence_id": int(prediction["matched_occurrence_id"]),
                "prepared_smiles": predicted_smiles,
                "target_smiles": prediction["target_smiles"],
                "blob_center_world_xyz_A": [
                    float(value) for value in handoff_record["blob_center_world_xyz_A"]
                ],
                "stage_seeds": stage_seeds,
                "views": [],
                "center_offset_xyz_A": [0.0, 0.0, 0.0],
            }
        )
    if len(records) != int(config.expected_handoff_count):
        raise ValueError(
            f"当前条件冻结{len(records)}条handoff，预期{config.expected_handoff_count}条。"
        )
    write_jsonl(Path(config.output_root) / "inputs" / "base.jsonl", records)
    write_jsonl(
        stage_input_path(config, "official-c"),
        [_stage_record(record, "official-c") for record in records],
    )
    write_jsonl(
        stage_input_path(config, "local-c1"),
        [_stage_record(record, "local-c1") for record in records],
    )
    return records


def _load_stage_candidates(config, stage, record):
    """读取一个已完成阶段的候选、SDF与无真值self-ranking结果。"""
    output_dir = candidate_output_dir(config, stage, record)
    result_path = output_dir / "result.json"
    if not result_path.exists():
        raise ValueError(f"{stage}尚无完成记录")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    candidates = json.loads(
        (output_dir / result["candidate_file"]).read_text(encoding="utf-8")
    )
    poses = []
    if result["pose_file"] is not None:
        poses = list(
            Chem.SDMolSupplier(
                str(output_dir / result["pose_file"]),
                sanitize=True,
                removeHs=True,
            )
        )
    receptor = read_receptor(
        Path(config.receptor_root)
        / "parse"
        / record["pdb_id"]
        / "receptor_tokens.npz"
    )
    receptor_molecule = build_receptor_molecule(receptor)
    template = read_smiles_graph(config.smiles_root, record["prepared_smiles"])["mol"]
    metrics = score_saved_candidates(
        candidates,
        poses,
        receptor_molecule,
        template,
        rmsd_reference=None,
    )
    ranked = rank_candidate_metrics(metrics)
    ranking = {
        "pdb_id": record["pdb_id"],
        "source_blob_index": int(record["source_blob_index"]),
        "stage": stage,
        "top1_sample_index": ranked[0]["sample_index"] if ranked else None,
        "candidate_metrics": metrics,
    }
    if ranked:
        molecule = poses[ranked[0]["sdf_index"]]
        conformer = molecule.GetConformer()
        coordinates = np.asarray(conformer.GetPositions(), dtype=np.float32)
        ranking["top1_center_world_xyz_A"] = coordinates.mean(axis=0).tolist()
        ranking["top1_coords_world_xyz_A"] = coordinates.tolist()
    else:
        ranking["top1_center_world_xyz_A"] = None
        ranking["top1_coords_world_xyz_A"] = None
    write_json(output_dir / "ranking.json", ranking)
    return ranking


def prepare_followup_records(config, target_stage):
    """按上一轮无真值self-ranking建立local-c2或local-e冻结输入。"""
    if target_stage == "local-c2":
        source_stage = "local-c1"
    elif target_stage == "local-e":
        source_stage = "local-c2"
    else:
        raise ValueError("后续输入只允许local-c2或local-e。")
    source_records = read_jsonl(stage_input_path(config, source_stage))
    target_records = []
    for source_record in source_records:
        target_record = {
            key: value
            for key, value in source_record.items()
            if key
            not in (
                "given_center_xyz_A",
                "envelope_coords_xyz_A",
                "input_error",
                "sampling_seed",
            )
        }
        target_record["sampling_seed"] = int(
            target_record["stage_seeds"][target_stage]
        )
        try:
            ranking = _load_stage_candidates(
                config,
                source_stage,
                source_record,
            )
            if ranking["top1_sample_index"] is None:
                raise ValueError(f"{source_stage}没有可排名候选")
            if target_stage == "local-c2":
                target_record["given_center_xyz_A"] = ranking[
                    "top1_center_world_xyz_A"
                ]
            else:
                target_record["envelope_coords_xyz_A"] = ranking[
                    "top1_coords_world_xyz_A"
                ]
        except Exception as error:
            target_record["input_error"] = f"{type(error).__name__}: {error}"
        target_records.append(target_record)
    write_jsonl(stage_input_path(config, target_stage), target_records)
    return target_records


def _sampling_config(config, stage):
    """将端到端总配置投影为原采样循环需要的单模型配置。"""
    model = config.models[_model_key(stage)]
    return EasyDict(
        model_name=model.name,
        train_config=model.train_config,
        checkpoint=model.checkpoint,
        receptor_branch=model.receptor_branch,
        dataset=EasyDict(
            root=config.receptor_root,
            smiles_root=config.smiles_root,
            knn=int(config.knn),
            pocket_mode="envelope" if stage == "local-e" else "center",
        ),
        split="test",
        output_root=str(stage_output_root(config, stage)),
        batch_size=int(config.batch_size),
        num_candidates=int(config.num_candidates),
        num_steps=int(config.num_steps),
        device=config.device,
    )


def sample_end_to_end_stage(config, stage):
    """加载一个冻结模型并为当前受体条件顺序完成一个端到端阶段。"""
    if stage not in STAGES:
        raise ValueError(f"未知端到端阶段: {stage}")
    records = read_jsonl(stage_input_path(config, stage))
    sampling_config = _sampling_config(config, stage)
    train_config, model_config, model, featurizer, transforms, sample_config = (
        load_sampling_runtime(sampling_config)
    )
    stage_dir = stage_output_root(config, stage) / "test" / stage
    run_path = stage_dir / "run.json"
    run_record = {
        "receptor_condition": config.receptor_condition,
        "stage": stage,
        "model_name": sampling_config.model_name,
        "train_config": sampling_config.train_config,
        "checkpoint": sampling_config.checkpoint,
        "receptor_root": sampling_config.dataset.root,
        "smiles_root": sampling_config.dataset.smiles_root,
        "receptor_branch": sampling_config.receptor_branch,
        "pocket_mode": sampling_config.dataset.pocket_mode,
        "knn": sampling_config.dataset.knn,
        "num_candidates": sampling_config.num_candidates,
        "num_steps": sampling_config.num_steps,
        "input_count": len(records),
    }
    if run_path.exists():
        if json.loads(run_path.read_text(encoding="utf-8")) != run_record:
            raise ValueError(f"{stage}现有产物属于不同科学配置。")
    else:
        write_json(run_path, run_record)
    dataset = ConditionedDockingDataset(
        sampling_config.dataset,
        records,
        transforms,
        sampling_config.receptor_branch,
        density_config=model_config.get("density"),
    )
    noiser = build_sampling_noiser(
        sampling_config,
        train_config,
        sample_config,
        featurizer,
    )
    results = []
    for index in range(len(dataset)):
        record = records[index]
        scientific_input = {
            key: record[key]
            for key in (
                "pdb_id",
                "candidate_id",
                "prepared_smiles",
                "sampling_seed",
            )
        }
        for key in ("given_center_xyz_A", "envelope_coords_xyz_A", "input_error"):
            if key in record:
                scientific_input[key] = record[key]
        output_dir = candidate_output_dir(config, stage, record)
        input_path = output_dir / "scientific_input.json"
        result_path = output_dir / "result.json"
        if input_path.exists():
            if json.loads(input_path.read_text(encoding="utf-8")) != scientific_input:
                raise ValueError(f"{stage}现有候选的科学输入与当前清单不一致。")
        elif result_path.exists():
            raise ValueError(f"{stage}现有候选缺少可核对的科学输入记录。")
        else:
            write_json(input_path, scientific_input)
        result = sample_occurrence(
            dataset,
            index,
            model,
            noiser,
            featurizer,
            sampling_config,
            stage,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "stage": stage,
                    "pdb_id": result["pdb_id"],
                    "source_blob_index": result["occurrence_id"],
                    "status": result["status"],
                    "success_count": result["success_count"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return results
