"""把 Matcher strongest-1 候选接入 PocketXMol 的分阶段端到端推理.

主要入口是 ``prepare_initial_records``、``prepare_followup_records`` 和
``sample_end_to_end_stage``. 前两者在 ``<output_root>/inputs`` 生成阶段 JSONL 清单,
后者在 ``<output_root>/<model_family>/test/<stage>/<pdb_id>/<candidate_id>`` 保存 50 个
候选的 SDF、置信度、候选状态和无真值 ``ranking.json``. RMSD、固定 77 个 PDB 分母和
完整 Matcher 轨迹合并由 ``docking.final_evaluation`` 负责.
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


# 四个正式阶段的固定顺序同时定义种子偏移; local-c1/local-c2 共享同一中心模型权重.
STAGES = ("official-c", "local-c1", "local-c2", "local-e")
STAGE_SEED_OFFSET = {stage: offset for offset, stage in enumerate(STAGES)}
MODEL_KEY_BY_STAGE = {
    "official-c": "official",
    "local-c1": "local_c",
    "local-c2": "local_c",
    "local-e": "local_e",
}
SOURCE_STAGE_BY_TARGET = {"local-c2": "local-c1", "local-e": "local-c2"}


def _write_frozen_identity(path, value, protected_result=None):
    """首次写入科学身份 JSON, 续跑时只接受逐字段相同的记录.

    输入参数:
        - path: Path, ``run.json`` 或一个候选的 ``scientific_input.json`` 路径.
        - value: dict, 当前代码将用于推理的科学身份字段.
        - protected_result: Path|None, 已有结果文件; 结果存在而身份文件缺失时不得补写身份.

    副作用:
        - ``path`` 不存在且没有旧结果时, 通过 ``write_json`` 原子写入 ``value``.
        - 已有身份与 ``value`` 不同, 或旧结果缺少可核对身份时, 停止续跑以免混合实验.
    """
    existing = None
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    elif protected_result is None or not protected_result.exists():
        write_json(path, value)
        return
    if existing != value:
        raise ValueError(f"已有产物的科学身份与当前输入不同: {path}")


def _stage_record(base_record, stage):
    """从评价记录复制一个阶段的无真值推理记录.

    输入 ``base_record`` 是 ``base.jsonl`` 中一项, 可含 matched occurrence、目标 SMILES、
    排名和计费字段. 返回记录只保留受体条件、Matcher 候选编号、精确预测 SMILES、世界
    XYZ 中心、派生种子和定位字段; 不复制目标身份、沉积坐标或 RMSD 参考. ``stage`` 决定
    ``sampling_seed``, 初始 official-C 与 local_cov-C 均以 Matcher blob 中心作为模型原点.
    """
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
    """冻结 strongest-1 的全部身份正确候选, 并建立两份首轮中心输入.

    输入参数:
        - config.matcher_evaluation: JSON 路径, 保存 strongest-1 的全部 prediction 与完整计费轨迹.
        - config.matcher_handoff: JSONL 路径, 保存可进入 Stage3 的身份正确候选.
        - config.receptor_condition: str, 当前只读取 ``GT`` 或 ``CA2`` 对应的 Matcher 记录.
        - config.seed_base: int, 与候选文件顺序及阶段偏移共同生成固定采样种子.
        - config.expected_handoff_count: int, 当前受体条件必须冻结的 handoff 候选数.

    落盘文件:
        - ``<output_root>/inputs/base.jsonl``: list[dict] 的 JSONL; 每项保留 PDB、候选编号、selected 标记、计费排名、matched occurrence、预测/目标 SMILES、Matcher 世界 XYZ 中心和四阶段种子, 只供最终评价与阶段清单构造.
        - ``<output_root>/inputs/official-c.jsonl``: list[dict] 的 JSONL; 每项只含 official-C 推理身份、精确预测 SMILES、Matcher 世界 XYZ 中心和该阶段种子.
        - ``<output_root>/inputs/local-c1.jsonl``: list[dict] 的 JSONL; 字段边界与 official-C 相同, 由 local_cov-C 使用.

    返回 ``base.jsonl`` 对应的记录列表. 本函数不读取沉积配体坐标, 也不运行模型.
    """
    # dict, Matcher 正式 evaluation.json; predictions 提供身份结果, traces 提供全部候选顺序与计费状态.
    evaluation = json.loads(Path(config.matcher_evaluation).read_text(encoding="utf-8"))
    # list[dict], 当前受体条件下全部需要 docking 的身份正确 small_molecule 候选, 不按 selected 或前 20 截断.
    handoff = [
        record
        for record in read_jsonl(config.matcher_handoff)
        if record["sorting"] == "source_probability_mean"
        and record["answer_scope"] == "small_molecule"
        and record["receptor_condition"] == config.receptor_condition
    ]
    # dict[(pdb_id, source_blob_index), dict], 按 Matcher 候选身份索引正式预测字段.
    predictions = {
        (record["pdb_id"], int(record["source_blob_index"])): record
        for record in evaluation["predictions"][config.receptor_condition]
    }
    traces = evaluation["traces"]["source_probability_mean"][
        config.receptor_condition
    ]["small_molecule"]
    # dict[(pdb_id, source_blob_index), dict], 按同一身份索引完整计费轨迹中的候选项.
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
    # list[tuple[str,int]], 每项是一个 handoff 的 PDB 与 source_blob_index 联合身份.
    handoff_keys = [
        (record["pdb_id"], int(record["source_blob_index"]))
        for record in handoff
    ]
    if len(set(handoff_keys)) != len(handoff_keys):
        raise ValueError("Matcher handoff 含重复的 PDB 与 source_blob_index 身份")
    # list[dict], 即将写入 base.jsonl 的评价侧记录, 顺序同时冻结各阶段派生种子.
    records = []
    for record_index, handoff_record in enumerate(handoff):
        key = (
            handoff_record["pdb_id"],
            int(handoff_record["source_blob_index"]),
        )
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
        # dict[stage,int], 同一 handoff 的四阶段独立种子; 不混入模型名或受体条件.
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
    """从一个完成阶段计算无真值 self-ranking, 并保存 Top-1 预测构象.

    输入 ``record`` 必须与 ``stage_input_path(config, stage)`` 中一项相同. 函数读取该候选
    的 ``result.json``、``candidates.json`` 和 ``poses.sdf``, 使用当前受体与精确预测
    SMILES 计算碰撞和显式立体项. 返回并写入 ``ranking.json``:
        - top1_sample_index: int|None, self-ranking 第一名的候选编号; 无可排名候选时为 None.
        - top1_center_world_xyz_A: list[float]|None, (3,), Top-1 重原子世界 XYZ 质心, 单位 Å.
        - top1_coords_world_xyz_A: list[list[float]]|None, (N, 3), Top-1 重原子世界 XYZ 坐标, 单位 Å, 原子顺序与精确预测 SMILES 一致.
        - candidate_metrics: list[dict], 50 个候选的置信度、碰撞、显式立体项和 self-ranking; 不含 RMSD.
    """
    output_dir = candidate_output_dir(config, stage, record)
    result_path = output_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    candidates = json.loads(
        (output_dir / result["candidate_file"]).read_text(encoding="utf-8")
    )
    # list[RDKit Mol|None], SDF 成功分子顺序与 candidates[*].sdf_index 对齐.
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
    # dict, 无真值 self-ranking 与 Top-1 预测构象; 后续 C/E 只读取此文件.
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
        # float32, (N, 3), Top-1 配体重原子世界 XYZ 坐标, 单位 Å.
        coordinates = np.asarray(conformer.GetPositions(), dtype=np.float32)
        ranking["top1_center_world_xyz_A"] = coordinates.mean(axis=0).tolist()
        ranking["top1_coords_world_xyz_A"] = coordinates.tolist()
    else:
        ranking["top1_center_world_xyz_A"] = None
        ranking["top1_coords_world_xyz_A"] = None
    write_json(output_dir / "ranking.json", ranking)
    return ranking


def prepare_followup_records(config, target_stage):
    """按上一轮无真值 self-ranking 建立 local-c2 或 local-e 冻结清单.

    ``target_stage=local-c2`` 时读取 local-c1 的 Top-1 世界 XYZ 质心作为新中心;
    ``target_stage=local-e`` 时读取 local-c2 的 Top-1 完整重原子世界 XYZ 坐标作为预测包络.
    返回并写出 ``<output_root>/inputs/<target_stage>.jsonl``. 每项仍保留原候选身份、精确
    预测 SMILES 和独立阶段种子; 上一阶段缺失或没有可排名候选时写入 ``input_error``, 让
    后续采样按失败记录保留评价分母.
    """
    source_stage = SOURCE_STAGE_BY_TARGET[target_stage]
    source_records = read_jsonl(stage_input_path(config, source_stage))
    # list[dict], 与 source_records 逐项对齐的下一阶段无真值输入.
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
                target_record["input_error"] = f"{source_stage}没有可排名候选"
            elif target_stage == "local-c2":
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
    """返回一个阶段传给原采样循环的单模型配置.

    输入 ``config`` 是 GT 或 CA2 的端到端总配置. 返回 EasyDict 固定模型名称、训练 YAML、
    checkpoint、RA/protein 分支、受体/SMILES 根、中心或包络模式、50 个候选、100 步和
    FP32 推理设备. 该投影不含 Matcher 评价真值.
    """
    model = config.models[MODEL_KEY_BY_STAGE[stage]]
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
    """加载一个冻结模型, 顺序完成一个端到端推理阶段.

    输入参数:
        - config: EasyDict, 含受体条件、四阶段模型、受体/SMILES 资产根、采样预算和输出根.
        - stage: str, ``official-c``、``local-c1``、``local-c2`` 或 ``local-e``.

    落盘产物:
        - ``<stage_output_root>/test/<stage>/run.json``: dict; 冻结模型、checkpoint、受体/SMILES 根、RA/protein 分支、口袋模式、kNN、候选数、步数和输入记录数.
        - ``<candidate_dir>/scientific_input.json``: dict; 冻结 PDB、候选编号、精确预测 SMILES、种子及实际中心、预测包络或 ``input_error``; 不含 matched occurrence、目标 SMILES 或沉积坐标.
        - ``<candidate_dir>/result.json`` 及其候选文件: 由 ``sample_occurrence`` 保存状态、50 个候选、世界坐标 SDF 和轨迹置信度.

    返回 list[dict], 与阶段输入清单逐项对齐的采样结果. 已完成候选只有在科学身份逐字段
    相同时才复用; 资源字段不进入候选身份.
    """
    records = read_jsonl(stage_input_path(config, stage))
    sampling_config = _sampling_config(config, stage)
    train_config, model_config, model, featurizer, transforms, sample_config = (
        load_sampling_runtime(sampling_config)
    )
    stage_dir = stage_output_root(config, stage) / "test" / stage
    run_path = stage_dir / "run.json"
    # dict, 阶段级科学身份; batch_size 和 device 不决定已有候选能否复用.
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
    _write_frozen_identity(run_path, run_record)
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
        # dict, 候选级科学身份; 中心、预测包络或上一步错误三者按阶段至多出现一类.
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
        _write_frozen_identity(input_path, scientific_input, protected_result=result_path)
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
