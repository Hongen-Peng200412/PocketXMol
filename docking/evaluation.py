"""评价已保存的 docking 候选池, 复用原 self_ranking、碰撞、立体化学和未对齐 RMSD 定义.

先读 summarize_occurrences 的纯统计, 再顺序读 evaluate_occurrence 的单实例评价和 evaluate_docking 的有限清单与 CPU 并行装配.
本模块不生成新候选. ALL、CAP10、HF10_TO5 从同一冻结清单选择实例, 共用 poses.sdf 和原置信度. 失败和未采样实例始终保留在成功率分母中.
输出在 <output_root>/<split>/<protocol>/<pdb_id>/<occurrence_id>/ 下增加 candidate_metrics.json 和 assessment.json, 在每个 protocol 下保存 occurrences.json 与 summary.json, 在 split 下保存汇集全部协议的 summary.json.
"""

import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from docking.assets import read_receptor, select_pocket
from docking.smiles import read_smiles_graph, read_smiles_coords, UnsupportedSmilesError
from utils.buster_tools import check_identity, check_intermolecular_distance


# 源 res_type 0..27 的标准残基名称, 仅用于还原原碰撞检查器需要的真实残基元数据.
STANDARD_RESIDUES = ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "A", "C", "G", "U", "DA", "DC", "DG", "DT")


def build_receptor_molecule(receptor):
    """把完整标准受体数组转为碰撞检查使用的无键RDKit分子。

    输入 ``receptor`` 来自 ``read_receptor``，坐标为世界XYZ、Å，第一维为全部标准蛋白
    与标准核酸重原子。返回分子不猜测受体共价键，也不用于生成模型，只服务原
    ``check_intermolecular_distance`` 距离检查。
    """
    molecule = Chem.RWMol()
    conformer = Chem.Conformer(len(receptor["coords"]))
    for atom_index, element in enumerate(receptor["element"]):
        atom = Chem.Atom(int(element))
        residue_info = Chem.AtomPDBResidueInfo(
            receptor["atom_name"][atom_index].decode("ascii")
        )
        residue_info.SetResidueName(STANDARD_RESIDUES[int(receptor["res_type"][atom_index])])
        residue_info.SetResidueNumber(int(receptor["res_index"][atom_index]))
        residue_info.SetIsHeteroAtom(False)
        atom.SetMonomerInfo(residue_info)
        molecule.AddAtom(atom)
        conformer.SetAtomPosition(atom_index, receptor["coords"][atom_index].tolist())
    molecule.AddConformer(conformer, assignId=True)
    return molecule.GetMol()


def score_saved_candidates(
    candidates,
    poses,
    receptor_molecule,
    stereo_reference,
    rmsd_reference=None,
    asset_error=None,
    pose_error=None,
):
    """按原碰撞、立体和self-ranking规则评价一个已保存候选池。

    输入参数:
        - candidates: list[dict]，``sample_occurrence`` 保存的完整候选记录。
        - poses: list[RDKit Mol|None]，成功候选按 ``sdf_index`` 对齐的世界坐标分子。
        - receptor_molecule: RDKit Mol，当前受体条件的完整标准受体。
        - stereo_reference: RDKit Mol，标准评价可带沉积构象；端到端中间轮使用无构象的
          精确SMILES模板，因此排序控制不读取真值坐标。
        - rmsd_reference: RDKit Mol|None，最终评价使用带沉积构象的参考；``None`` 表示
          中间轮只排名，不计算RMSD。
        - asset_error/pose_error: str|None，上游资产或SDF读取失败，保留到每个候选错误。

    返回值:
        - metrics: list[dict]，逐候选增加碰撞、立体、self_ranking和可选RMSD字段。

    ``self_ranking = cfd_traj + int(no_clashes) + int(stereo)``，不读取RMSD；并列顺序
    由 ``rank_candidate_metrics`` 按 ``sample_index`` 决定。
    """
    metrics = []
    for candidate in candidates:
        metric = {
            **candidate,
            "no_clashes": False,
            "stereo": False,
            "num_clashes": None,
            "rel_clashes": None,
            "self_ranking": None,
            "rmsd_A": None,
            "rmsd_identity_fallback": False,
            "errors": [],
        }
        if candidate["status"] != "success":
            metric["errors"].append(
                {"stage": candidate["stage"], "error": candidate["error"]}
            )
        elif asset_error is not None or pose_error is not None:
            metric["errors"].append(
                {
                    "stage": "evaluation_assets" if asset_error is not None else "read_pose",
                    "error": asset_error if asset_error is not None else pose_error,
                }
            )
        else:
            try:
                molecule = poses[candidate["sdf_index"]]
                if molecule is None:
                    raise ValueError("unreadable_saved_pose")
                if int(molecule.GetProp("sample_index")) != candidate["sample_index"]:
                    raise ValueError("saved_pose_candidate_index_mismatch")
            except Exception as error:
                metric["errors"].append(
                    {"stage": "read_pose", "error": f"{type(error).__name__}: {error}"}
                )
            else:
                try:
                    clash = check_intermolecular_distance(
                        molecule,
                        receptor_molecule,
                        ignore_types={
                            "hydrogens",
                            "organic_cofactors",
                            "inorganic_cofactors",
                            "waters",
                        },
                        clash_cutoff=0.75,
                    )["results"]
                    num_clashes = int(clash["num_pairwise_clashes"])
                    metric.update(
                        no_clashes=bool(clash["no_clashes"]),
                        num_clashes=num_clashes,
                        rel_clashes=num_clashes / molecule.GetNumAtoms(),
                    )
                except Exception as error:
                    metric["errors"].append(
                        {"stage": "clashes", "error": f"{type(error).__name__}: {error}"}
                    )
                try:
                    metric["stereo"] = bool(
                        check_identity(
                            molecule,
                            stereo_reference,
                            inchi_options="w",
                        )["results"]["stereo"]
                    )
                except Exception as error:
                    metric["errors"].append(
                        {"stage": "stereo", "error": f"{type(error).__name__}: {error}"}
                    )
                if rmsd_reference is not None:
                    try:
                        try:
                            rmsd = rdMolAlign.CalcRMS(
                                deepcopy(molecule),
                                deepcopy(rmsd_reference),
                                maxMatches=30000,
                            )
                        except RuntimeError:
                            metric["rmsd_identity_fallback"] = True
                            atom_map = [
                                [
                                    (atom_index, atom_index)
                                    for atom_index in range(molecule.GetNumAtoms())
                                ]
                            ]
                            rmsd = rdMolAlign.CalcRMS(
                                deepcopy(molecule),
                                deepcopy(rmsd_reference),
                                map=atom_map,
                            )
                        if not np.isfinite(rmsd):
                            raise ValueError("non_finite_rmsd")
                        metric["rmsd_A"] = float(rmsd)
                    except Exception as error:
                        metric["errors"].append(
                            {"stage": "rmsd", "error": f"{type(error).__name__}: {error}"}
                        )
        if (
            candidate["status"] == "success"
            and candidate["cfd_traj"] is not None
            and np.isfinite(candidate["cfd_traj"])
        ):
            metric["self_ranking"] = float(
                candidate["cfd_traj"]
                + int(metric["no_clashes"])
                + int(metric["stereo"])
            )
        metrics.append(metric)
    return metrics


def rank_candidate_metrics(metrics):
    """按冻结self-ranking降序和sample_index升序返回可排名候选。"""
    return sorted(
        (metric for metric in metrics if metric["self_ranking"] is not None),
        key=lambda metric: (-metric["self_ranking"], metric["sample_index"]),
    )


def summarize_occurrences(results):
    """汇总一个模型、定位协议和视图的逐实例评价, 不跨occurrence拼接候选.

    输入 results 是list[dict], 每项为 evaluate_occurrence 的 assessment, 可以为空.
    返回 summary 为 dict, 保存如下字段:
        - occurrence_count: int, 全部冻结实例数, 包括失败实例.
        - pdb_count: int, 上述实例对应的不同PDB数.
        - candidate_count: int, 全部实例的候选预算之和.
        - generated_count: int, 实际写入SDF的候选数.
        - generation_failed_count: int, 预算减去实际生成数.
        - rmsd_count: int, 成功计算RMSD的候选数.
        - rank_pair_count: int, 在各自实例内参与分数/RMSD分析的候选对数之和.
        - sampling_batch_attempt_count: int, 实际进入原采样循环的候选批次数之和.
        - sampling_batch_completed_count: int, 原循环完整返回并完成CUDA同步的候选批次数之和.
        - model_forward_attempt_count: int, 实际进入模型调用的批量前向次数之和.
        - model_forward_completed_count: int, 模型Python调用完整返回次数之和.
        - sampling_status_counts: dict[str,int], 每种采样状态的实例数, 如success或failed.
        - evaluation_status_counts: dict[str,int], 每种评价状态的实例数, 如success或partial.
        - candidate_error_counts: dict[str,int], 每个错误阶段涉及的候选数, 如{"noise":50}; 同一候选的不同检查错误分别计数.
        - spearman_na_reasons: dict[str,int], 各实例Spearman不可计算的原因及数量, 如{"constant_scores":2}.
        - pose_auc_na_reasons: dict[str,int], 各实例AUC不可计算的原因及数量, 如{"single_class":2}.
        - occurrence_equal: dict, 每个实例权重相同; 子字段如下.
            - top1_success_rate: float|None, 第一名RMSD<2 Å的实例比例; 无输出实例为失败, 空视图为None.
            - top5_success_rate: float|None, 前五名存在RMSD<2 Å候选的实例比例.
            - oracle_success_rate: float|None, 所有候选中存在RMSD<2 Å候选的实例比例.
            - top1_rmsd_mean_A: float|None, 第一名RMSD的有效实例均值, 单位Å; 缺失不记0.
            - top1_rmsd_valid_count: int, 上述均值的有效实例数.
            - top5_rmsd_mean_A: float|None, 每实例前五名最小RMSD的有效均值, 单位Å.
            - top5_rmsd_valid_count: int, 上述均值的有效实例数.
            - oracle_rmsd_mean_A: float|None, 每实例全部候选最小RMSD的有效均值, 单位Å.
            - oracle_rmsd_valid_count: int, 上述均值的有效实例数.
            - spearman_mean: float|None, 有定义的实例内Spearman(score,-RMSD)的均值.
            - spearman_median: float|None, 有定义的实例内Spearman中位数.
            - spearman_valid_count: int, Spearman可定义的实例数.
            - spearman_na_count: int, Spearman不可定义的实例数.
            - pose_auc_mean: float|None, 实例内RMSD<2 Å为正类的有效ROC-AUC均值.
            - pose_auc_valid_count: int, AUC可定义的实例数.
            - pose_auc_na_count: int, AUC不可定义的实例数.
        - pdb_equal: dict, 先对同一PDB内实例求均值, 再让不同PDB等权; 缺失统计只排除对应未知值, 成功率仍包含失败实例.
            - top1_success_rate: float|None, 各PDB内部第一名成功率的均值, 空视图为None.
            - top5_success_rate: float|None, 各PDB内部前五名成功率的均值.
            - oracle_success_rate: float|None, 各PDB内部全部候选成功率的均值.
            - top1_rmsd_mean_A: float|None, 各PDB有效第一名RMSD均值的均值, 单位Å.
            - top1_rmsd_valid_count: int, 至少有一个有效第一名RMSD的PDB数.
            - top5_rmsd_mean_A: float|None, 各PDB有效前五名最小RMSD均值的均值, 单位Å.
            - top5_rmsd_valid_count: int, 至少有一个有效前五名最小RMSD的PDB数.
            - oracle_rmsd_mean_A: float|None, 各PDB有效全部候选最小RMSD均值的均值, 单位Å.
            - oracle_rmsd_valid_count: int, 至少有一个有效全部候选最小RMSD的PDB数.
            - spearman_mean: float|None, 各PDB内有效实例Spearman均值的均值.
            - spearman_median: float|None, 各PDB内有效实例Spearman均值的中位数.
            - spearman_valid_count: int, 至少有一个有效Spearman实例的PDB数.
            - spearman_na_count: int, 没有有效Spearman实例的PDB数.
            - pose_auc_mean: float|None, 各PDB内有效实例AUC均值的均值.
            - pose_auc_valid_count: int, 至少有一个有效AUC实例的PDB数.
            - pose_auc_na_count: int, 没有有效AUC实例的PDB数.
        - inference_seconds: float, 组批、原采样循环和输出拆分的实例累计秒数.
        - sampling_seconds: float, 含预处理与写盘的采样累计秒数.
        - evaluation_seconds: float, 评价累计秒数.
        - peak_memory_allocated_bytes: int|None, 实例集合中最大的已记录张量显存峰值, 含模型, 单位字节; 无CUDA记录为None.
        - peak_memory_reserved_bytes: int|None, 实例集合中最大的已记录CUDA缓存分配峰值, 单位字节; 无记录为None.

    重叠视图共用候选, 不得把不同视图的耗时或候选数再相加. NA统计不填0, 空集合均值为None.
    """
    by_pdb = defaultdict(list)
    for result in results:
        by_pdb[result["pdb_id"]].append(result)
    summary = {
        "occurrence_count": len(results),
        "pdb_count": len(by_pdb),
        "candidate_count": sum(result["num_candidates"] for result in results),
        "generated_count": sum(result["generated_count"] for result in results),
        "generation_failed_count": sum(result["num_candidates"] - result["generated_count"] for result in results),
        "rmsd_count": sum(result["rmsd_count"] for result in results),
        "rank_pair_count": sum(result["rank_pair_count"] for result in results),
        "sampling_batch_attempt_count": sum(result["sampling_batch_attempt_count"] for result in results),
        "sampling_batch_completed_count": sum(result["sampling_batch_completed_count"] for result in results),
        "model_forward_attempt_count": sum(result["model_forward_attempt_count"] for result in results),
        "model_forward_completed_count": sum(result["model_forward_completed_count"] for result in results),
        "sampling_status_counts": dict(Counter(result["sampling_status"] for result in results)),
        "evaluation_status_counts": dict(Counter(result["evaluation_status"] for result in results)),
        "candidate_error_counts": dict(sum((Counter(result["candidate_error_counts"]) for result in results), Counter())),
        "spearman_na_reasons": dict(Counter(result["spearman_na_reason"] for result in results if result["spearman_na_reason"] is not None)),
        "pose_auc_na_reasons": dict(Counter(result["pose_auc_na_reason"] for result in results if result["pose_auc_na_reason"] is not None)),
    }
    # 每个 group 对应一个权重单位; 实例等权时长度1, PDB等权时含该PDB的全部冻结实例.
    for weighting, groups in (("occurrence_equal", [[result] for result in results]), ("pdb_equal", list(by_pdb.values()))):
        statistics = {}
        for ranking in ("top1", "top5", "oracle"):
            successes = [float(np.mean([result[f"{ranking}_success"] for result in group])) for group in groups]
            statistics[f"{ranking}_success_rate"] = float(np.mean(successes)) if successes else None
            # PDB内缺失坐标不进入条件RMSD均值, 但仍通过上面的布尔失败值进入成功率分母.
            rmsd_values = [[result[f"{ranking}_rmsd_A"] for result in group if result[f"{ranking}_rmsd_A"] is not None] for group in groups]
            rmsd_means = [float(np.mean(values)) for values in rmsd_values if values]
            statistics[f"{ranking}_rmsd_mean_A"] = float(np.mean(rmsd_means)) if rmsd_means else None
            statistics[f"{ranking}_rmsd_valid_count"] = len(rmsd_means)
        for metric in ("spearman", "pose_auc"):
            values_per_group = [[result[metric] for result in group if result[metric] is not None] for group in groups]
            valid_means = [float(np.mean(values)) for values in values_per_group if values]
            statistics[f"{metric}_mean"] = float(np.mean(valid_means)) if valid_means else None
            statistics[f"{metric}_valid_count"] = len(valid_means)
            statistics[f"{metric}_na_count"] = len(groups) - len(valid_means)
            if metric == "spearman":
                statistics["spearman_median"] = float(np.median(valid_means)) if valid_means else None
        summary[weighting] = statistics
    for field in ("inference_seconds", "sampling_seconds", "evaluation_seconds"):
        summary[field] = float(sum(result[field] for result in results))
    for field in ("peak_memory_allocated_bytes", "peak_memory_reserved_bytes"):
        recorded_peaks = [result[field] for result in results if result[field] is not None]
        summary[field] = max(recorded_peaks) if recorded_peaks else None
    return summary


# ================================================================================================
def evaluate_occurrence(arguments):
    """评价一个occurrence的既有候选, 保存原排序组成项、未对齐RMSD和实例内统计.

    arguments为tuple(config, protocol, record), 便于按完整实例分配CPU进程:
        - config: 与采样相同的EasyDict配置, 本函数读取dataset.root、output_root、split、model_name、num_candidates.
        - protocol: str, 当前C0、C5或E.
        - record: dict, 冻结清单的一条实例, 包括pdb_id、candidate_id、prepared_smiles、views和center_offset_xyz_A.

    产物位于 <output_root>/<split>/<protocol>/<pdb_id>/<occurrence_id>/:
        - candidate_metrics.json: list[dict], 逐候选保留candidates.json全部字段, 其定义见sampling.sample_occurrence, 另增加如下字段.
            - no_clashes: bool, 完整标准蛋白+RNA/DNA受体的原距离检查结果; 检查报错保持False.
            - stereo: bool, 原InChI立体检查结果; 检查报错保持False.
            - num_clashes: int|None, 碰撞原子对数; 检查失败为None.
            - rel_clashes: float|None, 碰撞原子对数除以配体重原子数; 检查失败为None.
            - self_ranking: float|None, cfd_traj+int(no_clashes)+int(stereo), 不读取RMSD; 生成失败或无有限置信度时为None.
            - rmsd_A: float|None, 世界坐标下原RDKit CalcRMS结果, 单位Å, 不做刚体对齐; 不可评为None.
            - rmsd_identity_fallback: bool, 原maxMatches=30000调用报RuntimeError后是否采用原同编号原子映射回退.
            - errors: list[dict], 一个候选可有多个检查错误, 每项包含以下字段.
                - stage: str, 出错阶段, 如clashes或rmsd.
                - error: str, 异常类型与信息, 如ValueError: invalid coordinates.

        - assessment.json: dict, 与函数返回值相同; 字段如下.
            - pdb_id: str, 当前结构编号, 如9v7o.
            - occurrence_id: int, 原candidate_id, 如0.
            - model_name: str, 当前模型名称, 如B-C-T0-RA.
            - split: str, validation或test.
            - protocol: str, 当前C0、C5或E.
            - prepared_smiles: str, 精确SMILES身份, 如CCO.
            - views: list[str], 冻结测试视图; validation为空列表.
            - sampling_complete: bool, 是否已有采样result.json; False不作为可复用缓存.
            - sampling_status: str, success、partial、failed或not_sampled, 沿用实际采样记录.
            - evaluation_status: str, 全预算RMSD可评且无候选错误为success; 至少一个RMSD可评但未满足success条件为partial; 全不可评为failed.
            - num_candidates: int, 固定预算, 正式值50, 始终保留失败分母.
            - generated_count: int, 实际保存SDF的候选数.
            - rmsd_count: int, 成功计算RMSD的候选数.
            - rank_pair_count: int, 当前实例有限self_ranking与RMSD组成的候选对数.
            - candidate_error_counts: dict[str,int], 各失败阶段的候选数, 如{"noise":50}; 同候选的不同检查可分别计数.
            - asset_error: str|None, 模板、沉积坐标、完整受体或评价参考组装的科学错误; 无错误为None, 存储OSError直接传播.
            - top1_sample_index: int|None, 原分数第一名的候选编号; 无可排名候选为None, 并列按编号升序.
            - top1_rmsd_A: float|None, 原分数第一名RMSD, 单位Å; 不可评为None.
            - top5_rmsd_A: float|None, 原分数前五名可评候选的最小RMSD, 单位Å.
            - oracle_rmsd_A: float|None, 全部候选的最小可评RMSD, 单位Å.
            - top1_success: bool, 第一名RMSD严格小于2 Å; 不可评为False.
            - top5_success: bool, 前五名存在RMSD严格小于2 Å的候选.
            - oracle_success: bool, 全部候选存在RMSD严格小于2 Å的候选.
            - spearman: float|None, 当前实例Spearman(self_ranking,-RMSD), 不跨实例拼接候选.
            - spearman_na_reason: str|None, too_few_candidates、constant_scores、constant_rmsd或None.
            - pose_auc: float|None, 当前实例ROC-AUC, self_ranking为分数, RMSD<2 Å为正类.
            - pose_auc_na_reason: str|None, no_valid_candidates、single_class或None.
            - pocket_protein_count: int|None, 当前冻结定位条件选择的标准蛋白重原子数, 官方蛋白过滤前取得; 无法取得为None.
            - pocket_nucleic_count: int|None, 同一口袋的标准RNA/DNA重原子数; 无法取得为None.
            - pocket_nucleic_fraction: float|None, 核酸数除以两类原子总数; 总数0或未知为None.
            - inference_seconds: float, 采样记录中的组批、循环与输出拆分累计秒数; 未采样为0.
            - sampling_seconds: float, 采样记录中含预处理与写盘的累计秒数; 未采样为0.
            - evaluation_seconds: float, 当前实例本次评价秒数.
            - sampling_batch_attempt_count: int, 实际进入原采样循环的候选批次数; 未采样为0.
            - sampling_batch_completed_count: int, 原循环完整返回并完成CUDA同步的候选批次数; 未采样为0.
            - model_forward_attempt_count: int, 实际进入模型调用的批量前向次数; 未采样为0.
            - model_forward_completed_count: int, 模型Python调用完整返回次数; 未采样为0.
            - peak_memory_allocated_bytes: int|None, 采样记录的峰值张量显存, 含模型, 单位字节; 无记录为None.
            - peak_memory_reserved_bytes: int|None, 采样记录的峰值CUDA缓存分配量, 单位字节; 无记录为None.

    完成缓存和采样文件先核对实例与实验身份, 再使用. 原子坐标只用于已经批准的定位和评价参考, 不反馈给生成模型; 完整受体不受口袋截取限制. 写盘异常不会生成完成的assessment标记.
    """
    config, protocol, record = arguments
    started = time.perf_counter()
    identity = {"pdb_id": record["pdb_id"], "occurrence_id": int(record["candidate_id"]), "model_name": config.model_name, "split": config.split, "protocol": protocol}
    occurrence_dir = Path(config.output_root) / config.split / protocol / identity["pdb_id"] / str(identity["occurrence_id"])
    occurrence_dir.mkdir(parents=True, exist_ok=True)
    assessment_path = occurrence_dir / "assessment.json"
    sampling_path = occurrence_dir / "result.json"
    sampling = json.loads(sampling_path.read_text(encoding="utf-8")) if sampling_path.exists() else None
    if sampling is not None and (any(sampling[key] != value for key, value in identity.items()) or sampling["prepared_smiles"] != record["prepared_smiles"] or sampling["num_candidates"] != config.num_candidates):
        raise ValueError("sampling_identity_mismatch")
    if assessment_path.exists() and sampling_path.exists():
        previous = json.loads(assessment_path.read_text(encoding="utf-8"))
        if any(previous[key] != value for key, value in identity.items()) or previous["prepared_smiles"] != record["prepared_smiles"] or previous["num_candidates"] != config.num_candidates:
            raise ValueError("assessment_identity_mismatch")
        if previous["sampling_complete"]:
            return previous
    candidates = json.loads((occurrence_dir / sampling["candidate_file"]).read_text(encoding="utf-8")) if sampling is not None else []
    metrics = []
    pocket_protein_count = pocket_nucleic_count = None
    asset_error = None
    try:
        root = Path(config.dataset.root)
        if "unsupported_smiles_reason" in record:
            raise UnsupportedSmilesError(record["unsupported_smiles_reason"])
        template = read_smiles_graph(config.dataset.smiles_root, record["prepared_smiles"])["mol"]
        # (N, 3), 公共 SMILES 原子顺序的沉积世界 XYZ 坐标, 只用于评价参照和定位条件.
        ligand_coords = read_smiles_coords(config.dataset.smiles_coords_root, identity["pdb_id"], identity["occurrence_id"], record["prepared_smiles"])
        reference = Chem.Mol(template)
        reference_conformer = Chem.Conformer(len(ligand_coords))
        for atom_index, position in enumerate(ligand_coords):
            reference_conformer.SetAtomPosition(atom_index, position.tolist())
        reference.AddConformer(reference_conformer, assignId=True)
        receptor = read_receptor(root / "parse" / identity["pdb_id"] / "receptor_tokens.npz")
        # 当前定位条件与 Dataset 相同; C5只读取冻结偏移, E只以沉积配体定义包络, 不重采样.
        offset = np.asarray(record["center_offset_xyz_A"], dtype=np.float32) if protocol == "C5" else np.zeros(3, dtype=np.float32)
        selected = select_pocket(receptor, ligand_coords, ligand_coords.mean(axis=0) + offset, "envelope" if protocol == "E" else "center")
        pocket_protein_count = int(np.sum(selected & (receptor["res_type"] < 20)))
        pocket_nucleic_count = int(np.sum(selected & (receptor["res_type"] >= 20)))
        receptor_molecule = build_receptor_molecule(receptor)
    except Exception as error:
        if isinstance(error, OSError):
            raise
        asset_error = f"{type(error).__name__}: {error}"
    poses = []
    pose_error = None
    if sampling is not None and sampling["pose_file"] is not None:
        try:
            # 与原RMSD评价的主读取分支相同, 使用默认sanitize读取完整重原子拓扑; 无法读取的候选保留错误和分母.
            poses = list(Chem.SDMolSupplier(str(occurrence_dir / sampling["pose_file"]), sanitize=True, removeHs=True))
        except Exception as error:
            if isinstance(error, OSError):
                raise
            pose_error = f"{type(error).__name__}: {error}"
    metrics = score_saved_candidates(
        candidates,
        poses,
        receptor_molecule if asset_error is None else None,
        reference if asset_error is None else None,
        rmsd_reference=reference if asset_error is None else None,
        asset_error=asset_error,
        pose_error=pose_error,
    )
    ranked = rank_candidate_metrics(metrics)
    assessment = {
        **identity,
        "prepared_smiles": record["prepared_smiles"],
        "views": record["views"],
        "sampling_complete": sampling is not None,
        "sampling_status": sampling["status"] if sampling is not None else "not_sampled",
        "num_candidates": config.num_candidates,
        "generated_count": sum(candidate["status"] == "success" for candidate in candidates),
        "candidate_error_counts": dict(Counter(error["stage"] for metric in metrics for error in metric["errors"])),
        "asset_error": asset_error,
        "top1_sample_index": ranked[0]["sample_index"] if ranked else None,
    }
    for ranking, selected_candidates in (("top1", ranked[:1]), ("top5", ranked[:5]), ("oracle", metrics)):
        rmsd_values = [metric["rmsd_A"] for metric in selected_candidates if metric["rmsd_A"] is not None]
        assessment[f"{ranking}_rmsd_A"] = min(rmsd_values) if rmsd_values else None
        assessment[f"{ranking}_success"] = bool(rmsd_values and min(rmsd_values) < 2.0)
    # (K,2), 每个实际可评分候选的一对 (self_ranking,RMSD), K只属于当前occurrence, 不跨实例拼接.
    rank_pairs = [(metric["self_ranking"], metric["rmsd_A"]) for metric in metrics if metric["self_ranking"] is not None and metric["rmsd_A"] is not None]
    assessment.update(rank_pair_count=len(rank_pairs), spearman=None, spearman_na_reason=None, pose_auc=None, pose_auc_na_reason=None)
    if len(rank_pairs) < 2:
        assessment["spearman_na_reason"] = "too_few_candidates"
    else:
        scores, rmsd_values = np.asarray(rank_pairs, dtype=float).T
        if np.unique(scores).size < 2:
            assessment["spearman_na_reason"] = "constant_scores"
        elif np.unique(rmsd_values).size < 2:
            assessment["spearman_na_reason"] = "constant_rmsd"
        else:
            assessment["spearman"] = float(spearmanr(scores, -rmsd_values)[0])
    if not rank_pairs:
        assessment["pose_auc_na_reason"] = "no_valid_candidates"
    else:
        scores, rmsd_values = np.asarray(rank_pairs, dtype=float).T
        # bool, (K,), 只有RMSD严格小于2 Å才是正类; 单一类别没有可定义的ROC-AUC.
        positives = rmsd_values < 2.0
        if np.unique(positives).size < 2:
            assessment["pose_auc_na_reason"] = "single_class"
        else:
            assessment["pose_auc"] = float(roc_auc_score(positives, scores))
    rmsd_count = sum(metric["rmsd_A"] is not None for metric in metrics)
    total_pocket = pocket_protein_count + pocket_nucleic_count if pocket_protein_count is not None else 0
    assessment.update(
        evaluation_status="success" if rmsd_count == config.num_candidates and not assessment["candidate_error_counts"] else "partial" if rmsd_count else "failed",
        rmsd_count=rmsd_count,
        pocket_protein_count=pocket_protein_count,
        pocket_nucleic_count=pocket_nucleic_count,
        pocket_nucleic_fraction=pocket_nucleic_count / total_pocket if total_pocket else None,
        inference_seconds=float(sampling["inference_seconds"]) if sampling is not None else 0.0,
        sampling_batch_attempt_count=sampling["sampling_batch_attempt_count"] if sampling is not None else 0,
        sampling_batch_completed_count=sampling["sampling_batch_completed_count"] if sampling is not None else 0,
        model_forward_attempt_count=sampling["model_forward_attempt_count"] if sampling is not None else 0,
        model_forward_completed_count=sampling["model_forward_completed_count"] if sampling is not None else 0,
        peak_memory_allocated_bytes=sampling["peak_memory_allocated_bytes"] if sampling is not None else None,
        peak_memory_reserved_bytes=sampling["peak_memory_reserved_bytes"] if sampling is not None else None,
        sampling_seconds=float(sampling["elapsed_seconds"]) if sampling is not None else 0.0,
        evaluation_seconds=time.perf_counter() - started,
    )
    (occurrence_dir / "candidate_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary_assessment = occurrence_dir / "assessment.json.tmp"
    temporary_assessment.write_text(json.dumps(assessment, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary_assessment.replace(assessment_path)
    return assessment


def evaluate_docking(config):
    """按冻结清单评价一个模型的全部获准协议, 测试三个视图共用候选池.

    输入config与sampling.sample_docking相同, 另由本入口读取:
        - evaluation_workers: int, 当前正式CPU任务的进程数8, 本函数不申请资源.
        - wandb: Mapping, W&B汇报设置.
            - entity: str, 当前账号/团队pencounkdual-111.
            - project: str, 当前无密度项目PocketXmol_raw.
            - name: str, 该模型与划分的可读评价名称.
            - mode: str, 必填, 当前配置显式为online; 实际连接失败后可明确改offline重跑, 不静默切换.

    先核对 <output_root>/<split>/run.json 中的科学配置, 防止拿错checkpoint或候选预算后汇报旧结果.
    输出及返回值:
        - <protocol>/occurrences.json: list[dict], 该协议全部实例assessment, 子字段见evaluate_occurrence.
        - <protocol>/summary.json: dict, test的键为ALL、CAP10、HF10_TO5; validation只含ALL, 每个值为summarize_occurrences的完整字典.
        - <output_root>/<split>/summary.json: dict, 同函数返回值, 包含全部协议和W&B结果.
            - model_name: str, 当前模型稳定名称.
            - split: str, validation或test.
            - protocols: dict[str,dict], 第一层键为C0/C5/E中的获准协议, 第二层键为该划分的视图名称, 每个值的字段见summarize_occurrences.
            - wandb: dict, 本地科学汇总落盘后才增加的远端记录状态.
                - run_id: str, W&B生成并在重跑时复用的标识; SDK导入或ID生成之前失败时尚不存在.
                - entity: str, 首次保存的账号/团队; 创建ID之前失败时尚不存在.
                - project: str, 首次保存的项目; 创建ID之前失败时尚不存在.
                - mode: str, 实际请求的online或offline.
                - status: str, online_completed、offline_saved或failed.
                - error: str|None, 失败的异常类型和消息; 成功为None.
        - <output_root>/<split>/wandb_run.json: dict, 为同一模型和划分保存稳定运行身份.
            - run_id: str, W&B运行标识, 如h3k82abc.
            - entity: str, 首次请求的账号/团队.
            - project: str, 首次请求的项目.

    每个CPU进程评价一个完整实例的全部候选. 未完成采样或失败实例保留在成功率分母; validation只评价完整集合, 不生成测试频数视图. 本地结果完整保存后只上传汇总, 不上传SDF/置信度数组. W&B失败明确报错, 保留本地结果及ID供重试.
    """
    split_dir = Path(config.output_root) / config.split
    frozen_run = json.loads((split_dir / "run.json").read_text(encoding="utf-8"))
    science_config = {key: value for key, value in config.items() if key not in ("batch_size", "device", "evaluation_workers", "wandb")}
    if frozen_run["science_config"] != science_config:
        raise ValueError("evaluation_configuration_differs_from_sampling")
    with (Path(config.dataset.manifest_root) / f"{config.split}.jsonl").open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    all_summaries = {"model_name": config.model_name, "split": config.split, "protocols": {}}
    with ProcessPoolExecutor(max_workers=config.evaluation_workers) as executor:
        for protocol in config.protocols:
            # 每个CPU进程处理一个完整occurrence, 50个候选留在同一进程内评价, 避免跨候选重读源资产.
            results = list(executor.map(evaluate_occurrence, ((config, protocol, record) for record in records)))
            summaries = {}
            for view in (("ALL", "CAP10", "HF10_TO5") if config.split == "test" else ("ALL",)):
                selected = [result for result in results if view == "ALL" or view in result["views"]]
                summaries[view] = summarize_occurrences(selected)
            protocol_dir = Path(config.output_root) / config.split / protocol
            protocol_dir.mkdir(parents=True, exist_ok=True)
            (protocol_dir / "occurrences.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            (protocol_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            all_summaries["protocols"][protocol] = summaries
            progress = {"model_name": config.model_name, "protocol": protocol, "split": config.split, "occurrence_count": summaries["ALL"]["occurrence_count"], "top1_success_rate": summaries["ALL"]["occurrence_equal"]["top1_success_rate"]}
            print(json.dumps(progress, ensure_ascii=False), flush=True)
    split_dir = Path(config.output_root) / config.split
    summary_path = split_dir / "summary.json"
    summary_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    # 科学结果已经完整落盘后才连接W&B; 网络或依赖错误不会丢失完成的评价, 也不会被当作成功上传.
    wandb_state_path = split_dir / "wandb_run.json"
    wandb_state = json.loads(wandb_state_path.read_text(encoding="utf-8")) if wandb_state_path.exists() else {}
    try:
        import wandb

        if not wandb_state:
            wandb_state = {"run_id": wandb.util.generate_id(), "entity": config.wandb.entity, "project": config.wandb.project}
            wandb_state_path.write_text(json.dumps(wandb_state, ensure_ascii=False, indent=2), encoding="utf-8")
        run = wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            name=config.wandb.name,
            mode=config.wandb.mode,
            id=wandb_state["run_id"],
            resume="allow",
            job_type="evaluation",
            tags=[config.model_name, config.split],
            dir=str(split_dir),
            config={"model_name": config.model_name, "split": config.split, "protocols": config.protocols, "checkpoint": config.checkpoint, "num_candidates": config.num_candidates, "num_steps": config.num_steps},
        )
        run.summary.update(all_summaries)
        run.finish()
    except Exception as error:
        all_summaries["wandb"] = {**wandb_state, "mode": config.wandb.mode, "status": "failed", "error": f"{type(error).__name__}: {error}"}
        summary_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"W&B评价汇报失败, 本地结果已保存在 {summary_path}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
    all_summaries["wandb"] = {**wandb_state, "mode": config.wandb.mode, "status": "online_completed" if config.wandb.mode == "online" else "offline_saved", "error": None}
    summary_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return all_summaries
