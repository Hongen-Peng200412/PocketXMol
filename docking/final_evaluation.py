"""汇总标准2/3 Å结果，并把PocketXMol结果合并回完整Matcher轨迹。

标准测评读取既有 ``candidate_metrics.json``，不重新前向或覆盖旧assessment。端到端测评
对最终候选池计算未对齐RMSD，同时保留77个PDB的背景、免费跳过和身份错误轨迹。
本模块不在评价产物中写文件哈希、运行环境、软件版本或逐阶段耗时。
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem

from docking.assets import read_receptor
from docking.final_artifacts import candidate_output_dir, read_jsonl, write_json, write_jsonl
from docking.evaluation import (
    build_receptor_molecule,
    rank_candidate_metrics,
    score_saved_candidates,
)
from docking.smiles import read_smiles_coords, read_smiles_graph


RANKING_LIMITS = {"top1": 1, "top5": 5, "oracle": None}


def _minimum_rmsd(metrics, limit):
    """返回self-ranking前limit个候选的最小有效RMSD；None表示全部候选。"""
    selected = metrics if limit is None else metrics[:limit]
    values = [metric["rmsd_A"] for metric in selected if metric["rmsd_A"] is not None]
    return min(values) if values else None


def _summarize_threshold_records(records, thresholds):
    """按occurrence等权和PDB等权汇总连续RMSD及多个严格阈值。"""
    by_pdb = defaultdict(list)
    for record in records:
        by_pdb[record["pdb_id"]].append(record)
    summary = {
        "occurrence_count": len(records),
        "pdb_count": len(by_pdb),
        "thresholds_A": [float(value) for value in thresholds],
    }
    for weighting, groups in (
        ("occurrence_equal", [[record] for record in records]),
        ("pdb_equal", list(by_pdb.values())),
    ):
        statistics = {"rmsd": {}, "success_rate": {}}
        for ranking in RANKING_LIMITS:
            values_per_group = [
                [
                    record[f"{ranking}_rmsd_A"]
                    for record in group
                    if record[f"{ranking}_rmsd_A"] is not None
                ]
                for group in groups
            ]
            valid_means = [float(np.mean(values)) for values in values_per_group if values]
            statistics["rmsd"][ranking] = {
                "mean_A": float(np.mean(valid_means)) if valid_means else None,
                "valid_count": len(valid_means),
            }
            for threshold in thresholds:
                key = f"{float(threshold):.1f}"
                successes = [
                    float(
                        np.mean(
                            [
                                record[f"{ranking}_rmsd_A"] is not None
                                and record[f"{ranking}_rmsd_A"] < threshold
                                for record in group
                            ]
                        )
                    )
                    for group in groups
                ]
                statistics["success_rate"].setdefault(key, {})[ranking] = (
                    float(np.mean(successes)) if successes else None
                )
        summary[weighting] = statistics
    return summary


def summarize_docking_thresholds(config):
    """从标准测评连续RMSD生成严格2/3 Å汇总，并校验2 Å与旧summary一致。"""
    with (Path(config.dataset.manifest_root) / f"{config.split}.jsonl").open(
        encoding="utf-8"
    ) as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    thresholds = [float(value) for value in config.final_thresholds_A]
    output = {
        "model_name": config.model_name,
        "split": config.split,
        "thresholds_A": thresholds,
        "protocols": {},
    }
    for protocol in config.protocols:
        evaluated = []
        for record in records:
            occurrence_dir = (
                Path(config.output_root)
                / config.split
                / protocol
                / record["pdb_id"]
                / str(record["candidate_id"])
            )
            metrics_path = occurrence_dir / "candidate_metrics.json"
            metrics = (
                json.loads(metrics_path.read_text(encoding="utf-8"))
                if metrics_path.exists()
                else []
            )
            ranked = rank_candidate_metrics(metrics)
            result = {
                "pdb_id": record["pdb_id"],
                "occurrence_id": int(record["candidate_id"]),
                "views": record["views"],
            }
            for ranking, limit in RANKING_LIMITS.items():
                selected = metrics if limit is None else ranked[:limit]
                result[f"{ranking}_rmsd_A"] = _minimum_rmsd(selected, None)
            evaluated.append(result)
        old_summaries = json.loads(
            (Path(config.output_root) / config.split / protocol / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        protocol_summaries = {}
        for view in ("ALL", "CAP10", "HF10_TO5"):
            selected = [
                result
                for result in evaluated
                if view == "ALL" or view in result["views"]
            ]
            summary = _summarize_threshold_records(selected, thresholds)
            for weighting in ("occurrence_equal", "pdb_equal"):
                for ranking in RANKING_LIMITS:
                    current = summary[weighting]["success_rate"]["2.0"][ranking]
                    previous = old_summaries[view][weighting][f"{ranking}_success_rate"]
                    if current is None or previous is None or not np.isclose(
                        current,
                        previous,
                        rtol=0.0,
                        atol=1e-12,
                    ):
                        raise ValueError(
                            f"{protocol}/{view}/{weighting}/{ranking}的2 Å汇总未复现旧结果。"
                        )
            protocol_summaries[view] = summary
        output["protocols"][protocol] = protocol_summaries
    write_json(
        Path(config.output_root) / config.split / "threshold_summary.json",
        output,
    )
    return output


def _reference_molecule(config, record):
    """按matched occurrence读取最终RMSD参考，并保持公共SMILES原子顺序。"""
    template = read_smiles_graph(config.smiles_root, record["target_smiles"])["mol"]
    coordinates = read_smiles_coords(
        config.smiles_coords_root,
        record["pdb_id"],
        int(record["matched_occurrence_id"]),
        record["target_smiles"],
    )
    reference = Chem.Mol(template)
    conformer = Chem.Conformer(len(coordinates))
    for atom_index, position in enumerate(coordinates):
        conformer.SetAtomPosition(atom_index, position.tolist())
    reference.AddConformer(conformer, assignId=True)
    return template, reference


def _evaluate_stage(config, stage, record, receptor_molecule, template, reference):
    """评价一条handoff候选在一个最终阶段的50姿态，并保存候选级连续RMSD。"""
    output_dir = candidate_output_dir(config, stage, record)
    result_path = output_dir / "result.json"
    if not result_path.exists():
        return {
            "status": "not_sampled",
            "output_dir": str(output_dir),
            "success_count": 0,
            "top1_sample_index": None,
            "pose_min_rmsd_A": {"1": None, "5": None, "50": None},
            "success": {},
        }
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
    metrics = score_saved_candidates(
        candidates,
        poses,
        receptor_molecule,
        template,
        rmsd_reference=reference,
    )
    ranked = rank_candidate_metrics(metrics)
    write_json(output_dir / "evaluation_metrics.json", metrics)
    minimum = {
        str(limit): _minimum_rmsd(ranked, limit)
        for limit in (1, 5, 50)
    }
    return {
        "status": result["status"],
        "output_dir": str(output_dir),
        "success_count": int(result["success_count"]),
        "top1_sample_index": ranked[0]["sample_index"] if ranked else None,
        "pose_min_rmsd_A": minimum,
        "success": {
            f"{threshold:.1f}": {
                pose_limit: value is not None and value < threshold
                for pose_limit, value in minimum.items()
            }
            for threshold in (2.0, 3.0)
        },
        "pocket_protein_count": result["pocket_protein_count"],
        "pocket_nucleic_count": result["pocket_nucleic_count"],
        "model_origin_world_xyz_A": result["model_origin_world_xyz_A"],
    }


def _read_ranking(config, stage, record):
    """读取C1或C2已经用于迭代控制的无真值ranking记录。"""
    path = candidate_output_dir(config, stage, record) / "ranking.json"
    if not path.exists():
        return None
    ranking = json.loads(path.read_text(encoding="utf-8"))
    return {
        "top1_sample_index": ranking["top1_sample_index"],
        "top1_center_world_xyz_A": ranking["top1_center_world_xyz_A"],
        "ranking_file": str(path),
    }


def summarize_end_to_end(direct_results, pdb_ids):
    """计算固定77个PDB的site@K×pose@M×严格RMSD阈值主表。"""
    by_pdb = defaultdict(list)
    for result in direct_results:
        by_pdb[result["pdb_id"]].append(result)
    summary = {"pdb_count": len(pdb_ids), "methods": {}}
    for method in ("official-C", "local_cov-C", "C-C", "C-C-E"):
        method_summary = {}
        for site_limit in (1, 3, 5, 10, 20):
            site_entry = {}
            for pose_limit in (1, 5, 50):
                pose_entry = {}
                for threshold in (2.0, 3.0):
                    successes = 0
                    for pdb_id in pdb_ids:
                        eligible = [
                            result["methods"][method]
                            for result in by_pdb.get(pdb_id, [])
                            if result["attempt_index"] <= site_limit
                        ]
                        if any(
                            candidate["success"].get(f"{threshold:.1f}", {}).get(
                                str(pose_limit),
                                False,
                            )
                            for candidate in eligible
                        ):
                            successes += 1
                    pose_entry[f"{threshold:.1f}"] = {
                        "success_count": successes,
                        "success_rate": successes / len(pdb_ids),
                    }
                site_entry[str(pose_limit)] = pose_entry
            method_summary[str(site_limit)] = site_entry
        summary["methods"][method] = method_summary
    return summary


def evaluate_end_to_end(config):
    """评价四种获准方法并生成候选索引、完整轨迹和77-PDB主表。"""
    base_records = read_jsonl(Path(config.output_root) / "inputs" / "base.jsonl")
    evaluation = json.loads(Path(config.matcher_evaluation).read_text(encoding="utf-8"))
    predictions = {
        (record["pdb_id"], int(record["source_blob_index"])): record
        for record in evaluation["predictions"][config.receptor_condition]
    }
    direct_results = []
    receptor_cache = {}
    for record in base_records:
        pdb_id = record["pdb_id"]
        try:
            if pdb_id not in receptor_cache:
                receptor = read_receptor(
                    Path(config.receptor_root)
                    / "parse"
                    / pdb_id
                    / "receptor_tokens.npz"
                )
                receptor_cache[pdb_id] = build_receptor_molecule(receptor)
            template, reference = _reference_molecule(config, record)
        except Exception as error:
            failed = {
                "status": "evaluation_asset_failed",
                "error": f"{type(error).__name__}: {error}",
                "success_count": 0,
                "top1_sample_index": None,
                "pose_min_rmsd_A": {"1": None, "5": None, "50": None},
                "success": {
                    threshold: {limit: False for limit in ("1", "5", "50")}
                    for threshold in ("2.0", "3.0")
                },
            }
            stage_results = {
                stage: dict(failed)
                for stage in ("official-c", "local-c1", "local-c2", "local-e")
            }
        else:
            stage_results = {}
            for stage in ("official-c", "local-c1", "local-c2", "local-e"):
                try:
                    stage_results[stage] = _evaluate_stage(
                        config,
                        stage,
                        record,
                        receptor_cache[pdb_id],
                        template,
                        reference,
                    )
                except Exception as error:
                    stage_results[stage] = {
                        "status": "evaluation_stage_failed",
                        "output_dir": str(candidate_output_dir(config, stage, record)),
                        "error": f"{type(error).__name__}: {error}",
                        "success_count": 0,
                        "top1_sample_index": None,
                        "pose_min_rmsd_A": {"1": None, "5": None, "50": None},
                        "success": {
                            threshold: {limit: False for limit in ("1", "5", "50")}
                            for threshold in ("2.0", "3.0")
                        },
                    }
        stage_results["local-c2"]["intermediate_top1"] = _read_ranking(
            config,
            "local-c1",
            record,
        )
        stage_results["local-e"]["first_center_top1"] = _read_ranking(
            config,
            "local-c1",
            record,
        )
        stage_results["local-e"]["second_center_top1"] = _read_ranking(
            config,
            "local-c2",
            record,
        )
        direct_results.append(
            {
                "receptor_condition": config.receptor_condition,
                "pdb_id": pdb_id,
                "source_blob_index": int(record["source_blob_index"]),
                "centered_box_index": int(record["centered_box_index"]),
                "candidate_selected": bool(record["candidate_selected"]),
                "raw_rank": int(record["raw_rank"]),
                "attempt_index": int(record["attempt_index"]),
                "matched_occurrence_id": int(record["matched_occurrence_id"]),
                "predicted_smiles": record["prepared_smiles"],
                "target_smiles": record["target_smiles"],
                "methods": {
                    "official-C": stage_results["official-c"],
                    "local_cov-C": stage_results["local-c1"],
                    "C-C": stage_results["local-c2"],
                    "C-C-E": stage_results["local-e"],
                },
            }
        )
    direct_path = Path(config.output_root) / "docking_results.jsonl"
    write_jsonl(direct_path, direct_results)
    result_by_key = {
        (result["pdb_id"], result["source_blob_index"]): result
        for result in direct_results
    }
    source_traces = evaluation["traces"]["source_probability_mean"][
        config.receptor_condition
    ]["small_molecule"]
    derived_traces = {}
    for pdb_id, items in source_traces.items():
        derived_items = []
        for item in items:
            key = (pdb_id, int(item["source_blob_index"]))
            prediction = predictions.get(key)
            if key in result_by_key:
                docking_stage = {
                    "required": True,
                    "reason": None,
                    "methods": result_by_key[key]["methods"],
                }
            elif item["target_scope"] == "background":
                docking_stage = {"required": False, "reason": "background"}
            elif item["attempt_index"] is None:
                docking_stage = {
                    "required": False,
                    "reason": "non_stage3_free_skip",
                }
            else:
                docking_stage = {
                    "required": False,
                    "reason": "small_molecule_identity_incorrect",
                }
            derived_items.append(
                {
                    **item,
                    "candidate_selected": (
                        bool(prediction["candidate_selected"])
                        if prediction is not None
                        else None
                    ),
                    "predicted_smiles": (
                        prediction["predicted_smiles_small_molecule"]
                        if prediction is not None
                        else None
                    ),
                    "docking_stage": docking_stage,
                }
            )
        derived_traces[pdb_id] = derived_items
    pdb_ids = sorted(source_traces)
    if len(pdb_ids) != int(config.expected_pdb_count):
        raise ValueError(
            f"完整轨迹含{len(pdb_ids)}个PDB，预期{config.expected_pdb_count}个。"
        )
    summary = summarize_end_to_end(direct_results, pdb_ids)
    derived = {
        "model_name": "strongest-1",
        "sorting": "source_probability_mean",
        "answer_scope": "small_molecule",
        "receptor_condition": config.receptor_condition,
        "direct_result_file": str(direct_path),
        "summary": summary,
        "traces": derived_traces,
    }
    evaluation_path = Path(config.output_root) / "evaluation.json"
    write_json(evaluation_path, derived)

    wandb_state_path = Path(config.output_root) / "wandb_run.json"
    wandb_state = (
        json.loads(wandb_state_path.read_text(encoding="utf-8"))
        if wandb_state_path.exists()
        else {}
    )
    try:
        import wandb

        if not wandb_state:
            wandb_state = {
                "run_id": wandb.util.generate_id(),
                "entity": config.wandb.entity,
                "project": config.wandb.project,
            }
            write_json(wandb_state_path, wandb_state)
        run = wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            name=config.wandb.name,
            mode=config.wandb.mode,
            id=wandb_state["run_id"],
            resume="allow",
            job_type="evaluation",
            tags=["strongest-1", config.receptor_condition],
            dir=str(config.output_root),
            config={
                "model_name": "strongest-1",
                "sorting": "source_probability_mean",
                "answer_scope": "small_molecule",
                "receptor_condition": config.receptor_condition,
                "pdb_count": len(pdb_ids),
            },
        )
        run.summary.update(summary)
        run.finish()
    except Exception as error:
        print(
            f"W&B端到端汇报失败，本地结果已保存在{evaluation_path}: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise
    return derived
