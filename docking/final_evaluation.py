"""汇总标准 2/3 Å 指标, 并把 PocketXMol 结果合并回完整 Matcher 轨迹.

标准入口 ``summarize_docking_thresholds`` 读取既有候选指标并写 ``threshold_summary.json``.
端到端入口 ``evaluate_end_to_end`` 写 ``docking_results.jsonl``、扩展 ``evaluation.json`` 和
``wandb_run.json``. 前者只含需要 docking 的身份正确候选, 后者保留固定 77 个 PDB 的背景、
免费跳过、身份错误、推理状态和评价状态. 本模块不运行模型或重新生成候选.
"""

import json
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


# dict[str,int|None], 标准汇总读取 self-ranking 前 1、5 或全部可评价候选.
RANKING_LIMITS = {"top1": 1, "top5": 5, "oracle": None}


def _minimum_rmsd(metrics, limit):
    """返回 self-ranking 前 ``limit`` 个候选中的最小有效 RMSD.

    ``metrics`` 必须已经按 self-ranking 排序, 每项的 ``rmsd_A`` 为 float 或 None. ``limit``
    为正整数时只查看前 M 个姿态, 为 None 时查看全部候选. 返回 float, 单位 Å; 所选候选
    都没有有效 RMSD 时返回 None.
    """
    selected = metrics if limit is None else metrics[:limit]
    values = [metric["rmsd_A"] for metric in selected if metric["rmsd_A"] is not None]
    return min(values) if values else None


def _summarize_threshold_records(records, thresholds):
    """按 occurrence 等权和 PDB 等权汇总连续 RMSD 与严格阈值成功率.

    输入 ``records`` 是实例级记录列表. 每项含 ``pdb_id`` 及 top1、top5、oracle 的最小
    RMSD, 单位 Å; 缺失值为 None. ``thresholds`` 是严格小于关系使用的 Å 阈值列表.

    返回 dict:
        - occurrence_count: int, 当前视图中的实例数, 包括没有有效 RMSD 的实例.
        - pdb_count: int, 当前视图中的不同 PDB 数.
        - thresholds_A: list[float], 按输入顺序保存的 RMSD 阈值, 单位 Å.
        - occurrence_equal: dict, 每个实例等权的 RMSD 均值、有效数和阈值成功率.
        - pdb_equal: dict, 先在每个 PDB 内按实例平均, 再令每个 PDB 等权的相同统计.

    ``occurrence_equal`` 与 ``pdb_equal`` 共享以下子字段:
        - rmsd[ranking].mean_A: float|None, ranking 为 top1、top5 或 oracle 时有效组的平均 RMSD, 单位 Å.
        - rmsd[ranking].valid_count: int, 具有至少一个有效 RMSD 的等权组数.
        - success_rate[threshold][ranking]: float|None, 严格小于 threshold 的组内成功率再按组等权平均.
    """
    # dict[pdb_id, list[dict]], 同一 PDB 的实例级 RMSD 记录保持输入顺序.
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
        # dict, 当前权重定义下 top1/top5/oracle 的连续 RMSD 与严格阈值成功率.
        statistics = {"rmsd": {}, "success_rate": {}}
        for ranking in RANKING_LIMITS:
            # list[list[float]], 每个等权组内的有效实例 RMSD; 空列表仍保留该组的失败分母.
            values_per_group = [
                [
                    record[f"{ranking}_rmsd_A"]
                    for record in group
                    if record[f"{ranking}_rmsd_A"] is not None
                ]
                for group in groups
            ]
            # list[float], 每个至少有一个有效实例的等权组均值; 缺失组不进入连续 RMSD 均值.
            valid_means = [float(np.mean(values)) for values in values_per_group if values]
            statistics["rmsd"][ranking] = {
                "mean_A": float(np.mean(valid_means)) if valid_means else None,
                "valid_count": len(valid_means),
            }
            for threshold in thresholds:
                key = f"{float(threshold):.1f}"
                # list[float], 每个等权组内严格小于阈值的实例比例; 缺失 RMSD 按 False 计入组分母.
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
    """从标准测评的连续 RMSD 生成严格 2/3 Å 汇总.

    输入 ``config`` 指定模型名称、冻结 split 清单、C0/C5/E 协议、既有标准评价根和阈值.
    函数读取每个实例的 ``candidate_metrics.json``, 依据原 self-ranking 计算 top1、top5 与
    oracle 最小 RMSD, 再对 ALL、CAP10、HF10_TO5 三个冻结视图分别汇总.

    写入 ``<output_root>/<split>/threshold_summary.json``:
        - model_name: str, 当前标准测评模型名称.
        - split: str, 当前冻结划分, 正式运行是 test.
        - thresholds_A: list[float], 严格小于关系使用的 RMSD 阈值, 单位 Å.
        - protocols: dict[protocol, dict[view, summary]], 各协议和视图的实例/PDB等权统计; ``summary`` 字段由 ``_summarize_threshold_records`` 定义.

    本函数不重新前向、不修改 ``assessment.json`` 或候选级评价文件. 返回值与落盘 JSON 相同.
    """
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
        # list[dict], 当前协议的实例级 top1/top5/oracle 最小 RMSD, 与冻结清单顺序一致.
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
            # dict, 当前 occurrence 在三个排名范围下的最小有效 RMSD.
            result = {
                "pdb_id": record["pdb_id"],
                "occurrence_id": int(record["candidate_id"]),
                "views": record["views"],
            }
            for ranking, limit in RANKING_LIMITS.items():
                selected = metrics if limit is None else ranked[:limit]
                result[f"{ranking}_rmsd_A"] = _minimum_rmsd(selected, None)
            evaluated.append(result)
        protocol_summaries = {}
        for view in ("ALL", "CAP10", "HF10_TO5"):
            selected = [
                result
                for result in evaluated
                if view == "ALL" or view in result["views"]
            ]
            summary = _summarize_threshold_records(selected, thresholds)
            protocol_summaries[view] = summary
        output["protocols"][protocol] = protocol_summaries
    write_json(
        Path(config.output_root) / config.split / "threshold_summary.json",
        output,
    )
    return output


def _reference_molecule(config, record):
    """按 matched occurrence 构造最终 RMSD 的真值参考分子.

    ``record.target_smiles`` 定义公共重原子图与原子顺序; ``pdb_id`` 和
    ``matched_occurrence_id`` 定位同一顺序的沉积世界 XYZ 坐标, 单位 Å. 返回二元组:
        - template: RDKit Mol, 无构象的目标 SMILES 图, 只作为最终候选的显式立体模板.
        - reference: RDKit Mol, 与 template 原子逐项对齐并附加一个沉积世界坐标构象, 只用于最终 RMSD.
    """
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
    """评价一条 handoff 候选在一个最终阶段的 50 个姿态.

    ``record`` 标识一个身份正确 Matcher 候选, ``receptor_molecule`` 是当前 GT/CA2 完整
    标准受体, ``template`` 是无构象目标图, ``reference`` 是沉积坐标 RMSD 参考. 函数读取
    阶段 ``result.json``、候选状态和 SDF, 按原 self-ranking 排序, 并写
    ``evaluation_metrics.json``.

    返回 dict:
        - status: str, 原采样状态; 结果文件缺失时为 ``not_sampled``.
        - output_dir: str, 当前候选阶段产物目录.
        - success_count: int, 成功写入 SDF 的姿态数.
        - top1_sample_index: int|None, self-ranking 第一名的候选编号.
        - pose_min_rmsd_A: dict[str,float|None], 前 1、5、50 个姿态各自的最小未对齐 RMSD, 单位 Å.
        - success: dict[str,dict[str,bool]], 严格小于 2.0/3.0 Å 在 pose@1/5/50 下是否成功.
        - pocket_protein_count/pocket_nucleic_count: int, 已采样分支中实际口袋的标准蛋白/核酸重原子数; not_sampled 分支不含这两个字段.
        - model_origin_world_xyz_A: list[float], (3,), 已采样分支中模型原点的世界 XYZ 坐标, 单位 Å; not_sampled 分支不含此字段.

    ``not_sampled`` 分支的 ``success`` 为空字典; 汇总时缺失阈值按失败处理. 已采样分支的
    ``success`` 完整保存 2.0/3.0 Å 与 pose@1/5/50 的布尔结果.
    """
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
    """读取 C1 或 C2 已用于迭代控制的无真值 Top-1 记录.

    返回 None 表示该阶段没有 ``ranking.json``. 否则返回 top1 候选编号、(3,) 世界 XYZ
    质心和 ranking 文件路径; 不返回候选 RMSD 或 matched occurrence 真值.
    """
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
    """计算固定 77 个 PDB 的 site@K × pose@M × 严格 RMSD 阈值主表.

    ``direct_results`` 每项是一个身份正确 handoff 候选, 含 ``attempt_index`` 和四种方法的
    pose@1/5/50 成功布尔值. ``pdb_ids`` 是完整 Matcher 轨迹中的固定 PDB 列表, 包括完全
    没有正确候选的 PDB. site@K 只纳入 ``attempt_index <= K`` 的计费候选; K=20 是汇总
    边界, 不是推理截断. 未采样、公共评价资产失败或单阶段评价失败均保留在分母并计失败.

    返回 ``{"pdb_count": 77, "methods": ...}``. ``methods[method][K][M][threshold]`` 保存
    ``success_count`` 与 ``success_rate``; method 为 official-C、local_cov-C、C-C 或 C-C-E,
    K 为 1/3/5/10/20, M 为 1/5/50, threshold 为 2.0/3.0 Å.
    """
    # dict[pdb_id, list[dict]], 每个 PDB 的身份正确 handoff 候选, 保持 direct_results 顺序.
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
    """评价四种方法, 生成直接结果、扩展轨迹、主表和 W&B 汇报.

    输入 ``config`` 指定当前 GT/CA2 条件、Matcher evaluation、端到端输入/产物根、受体
    与 SMILES 资产、固定 PDB 数及 W&B 运行信息. 评价真值只在本函数内由 ``base.jsonl``
    读取, 不回写阶段推理清单.

    落盘文件:
        - ``<output_root>/docking_results.jsonl``: JSONL; 每项对应一个身份正确且实际要求 docking 的 handoff 候选, 保存候选身份、计费顺序和 official-C/local_cov-C/C-C/C-C-E 四种方法的状态、Top-1、pose@M RMSD 与成功布尔值.
        - ``<output_root>/evaluation.json``: dict; ``summary`` 是固定 77-PDB 主表, ``traces`` 保留全部 Matcher 候选顺序、背景、非 Stage3 免费跳过、身份错误及新增 ``docking_stage`` 推理状态.
        - ``<output_root>/wandb_run.json``: dict; 保存可恢复的 W&B run_id、entity 和 project.

    返回值与 ``evaluation.json`` 相同. 单阶段未采样或评价失败时只把该方法记为失败; 公共
    受体、SMILES 或真值坐标资产失败时四种方法均记为失败. 两类失败都不删除候选或 PDB
    分母. W&B 必须保持 online; 汇报失败时让异常直接返回调用方.
    """
    base_records = read_jsonl(Path(config.output_root) / "inputs" / "base.jsonl")
    evaluation = json.loads(Path(config.matcher_evaluation).read_text(encoding="utf-8"))
    predictions = {
        (record["pdb_id"], int(record["source_blob_index"])): record
        for record in evaluation["predictions"][config.receptor_condition]
    }
    # list[dict], 仅包含身份正确且要求 docking 的 handoff 候选, 与 base_records 逐项对齐.
    direct_results = []
    # dict[pdb_id, RDKit Mol], 当前受体条件的完整标准受体距离检查分子, 在同一 PDB 内复用.
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
    # dict[(pdb_id, source_blob_index), dict], 把直接 docking 结果回连完整 Matcher 轨迹.
    result_by_key = {
        (result["pdb_id"], result["source_blob_index"]): result
        for result in direct_results
    }
    source_traces = evaluation["traces"]["source_probability_mean"][
        config.receptor_condition
    ]["small_molecule"]
    # dict[pdb_id, list[dict]], 保持原全部候选顺序并为每项增加 docking_stage.
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
    return derived
