"""使用原 free docking 采样循环生成冻结实例的候选坐标和原始置信度.

先读 sample_occurrence 的单实例生成与保存, 再读 sample_docking 的模型和全量清单装配. 官方模型和 RA 模型使用同一候选预算、噪声调度和结果格式.
输出根由 config.output_root 决定. <split>/<protocol>/<pdb_id>/<occurrence_id>/ 下的 poses.sdf 保存成功候选世界坐标, candidates.json 保存全部候选的状态和置信度汇总, confidence.npz 保存成功候选的原始置信度.
result.json 在其他产物写完后才原子写入, 表示该实例的全部候选已经尝试, 成功数可以为零. 未出现该文件的实例在续跑时按冻结种子重新执行.
"""

import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose

from docking.assets import read_template
from docking.dataset import OccurrenceDataset
from models.maskfill import PMAsymDenoiser
from models.sample import get_cfd_traj, sample_loop3, seperate_outputs2
from utils.misc import make_config, seed_all
from utils.reconstruct import reconstruct_pos
from utils.sample_noise import get_sample_noiser
from utils.transforms import FeaturizeMol, get_transforms


# ================================================================================================
def sample_occurrence(dataset, index, model, noiser, featurizer, config, protocol):
    """完成一个 occurrence 的全部候选尝试, 最后保存完成标记.

    输入参数:
        - dataset: OccurrenceDataset, 已完成累计筛选的实例清单, 按当前 protocol 装配原子和口袋.
        - index: int, 索引 dataset.records 中的当前 occurrence.
        - model: 已加载明确 checkpoint 的 PMAsymDenoiser, eval 模式, 官方蛋白或 RA 分支.
        - noiser: 原 DockSamplNoiser, mode=sample, 使用 config.num_steps 个步骤.
        - featurizer: 原 FeaturizeMol, 将模型坐标加回实际 pocket_center 后解码.
        - config: EasyDict, 完整字段见 sample_docking.
        - protocol: str, C0、C5 或 E, 决定定位条件和输出子目录.

    产物位于 <output_root>/<split>/<protocol>/<pdb_id>/<occurrence_id>/:
        - poses.sdf: 多分子 SDF, 仅包含成功候选(跑通就算成功, 不是RMSD<2埃); 每个分子的 sample_index 属性保存原候选编号, 如3, 拓扑和原子顺序来自完整模板, 坐标为世界 XYZ、Å.
        - candidates.json: list[dict], 长度为 num_candidates, 包括每个失败候选; 各项字段如下.
            - pdb_id: str, 当前结构编号, 如9v7o.
            - occurrence_id: int, 原 candidate_id, 如0.
            - model_name: str, 当前模型的稳定名称, 如 B-C-T0-RA.
            - split: str, validation 或 test.
            - protocol: str, 当前 C0、C5 或 E.
            - sample_index: int, 从0开始的原候选构象编号, 正式50个候选时为0至49; 与成功候选在SDF中的位置不同.
            - status: str, success 或 failed; 失败不补生成新候选.
            - stage: str, 当前候选结束阶段; complete 表示完成, preprocess/batch/prepare_loop/noise/forward/prediction_to_batch/trajectory/synchronize/split/reconstruct/confidence 分别定位预处理、组批、循环准备、加噪、模型调用、预测写回、轨迹处理、CUDA同步、输出拆分、固定图重构及置信度聚合.
            - error: str|None, 失败的异常类型和消息; 成功为 None.
            - sdf_index: int|None, 当前候选在poses.sdf中从0开始的位置; 失败为None. 如原候选0失败、1首先成功, 则该分子sample_index=1而sdf_index=0.
            - cfd_traj: float|None, 原 get_cfd_traj 分数; 正式100步先对全部配体原子取均值, 再平均后50步的原始位置置信度, 全程不做 sigmoid; 非有限分数使候选失败.
            - cfd_pos: float|None, 最后一步所有配体原子的原始位置置信度均值; 空数组或非有限均值写 None.
            - cfd_node: float|None, 最后一步原子类别置信度原始输出的原子均值; 空数组或非有限均值写 None.
            - cfd_edge: float|None, 最后一步半边类别置信度原始输出的半边均值; 空数组或非有限均值写 None.
        - confidence.npz: 仅存在成功候选时写出; K 为成功候选数, N 为完整配体重原子数, H=N*(N-1)/2, T=num_steps.
            - sample_index: int64, (K,), 成功候选原编号, 如[0,2,3], 与SDF分子顺序及下列数组首轴对齐.
            - confidence_pos_traj: float32, (K,N,T), 每个原子每步的原始位置置信度, 无 sigmoid.
            - confidence_pos: float32, (K,N,1), 最后一步原始位置置信度, 原子顺序与完整模板一致.
            - confidence_node: float32, (K,N,1), 最后一步原子类别置信度原始输出, 原子顺序与模板一致.
            - confidence_halfedge: float32, (K,H,1), 最后一步半边类别置信度原始输出, 半边按原完全图上三角顺序排列.

        - result.json: dict, 全部候选尝试和产物写入完成后保存, 返回值为同一字典.
            - pdb_id: str, 当前结构编号.
            - occurrence_id: int, 原 candidate_id.
            - model_name: str, 当前模型稳定名称.
            - split: str, validation 或 test.
            - protocol: str, 当前 C0、C5 或 E.
            - object_key: str, 完整模板身份, 如 CCD:GMP.
            - views: list[str], 冻结测试视图名称; 验证为空列表, 测试从 ALL、CAP10、HF10_TO5 选择.
            - sampling_seed: int, 当前 occurrence 的冻结候选种子, 如10831; 模型和协议不另混入种子.
            - center_offset_xyz_A: list[float], 长度3的冻结 C5 偏移, 世界 XYZ、Å; C0/E也保存此值但不施加.
            - complete: bool, True表示预算内所有候选均已尝试, 不表示全部成功.
            - status: str, success为全部成功, partial为部分成功, failed为成功数0.
            - num_candidates: int, 固定候选预算, 正式值50.
            - success_count: int, 实际写入SDF的成功候选数.
            - failed_count: int, 候选预算减去成功数.
            - num_steps: int, 每轮原采样循环步数, 正式值100.
            - batch_size: int, 同一实例同时采样的候选数上限, 当前正式值50; 预算只有50时, 设置100仍只组一批50.
            - pocket_protein_count: int|None, 按当前定位协议选择、在官方蛋白过滤之前的标准蛋白原子数; 尚未取得时为None.
            - pocket_nucleic_count: int|None, 同一口袋中的标准RNA/DNA原子数, 官方过滤前计数; 尚未取得时为None.
            - pocket_nucleic_fraction: float|None, 核酸原子数除以两类原子总数; 总数0或未取得时为None.
            - model_origin_world_xyz_A: list[list[float]]|None, 正常为(1,3)嵌套列表, 世界XYZ、Å; 官方空蛋白保留其空列表, 更早失败为None.

            - inference_seconds: float, 组批、原采样循环与输出拆分的累计秒数, 不含SDF重构和写盘.
            - sampling_batch_attempt_count: int, 实际进入原采样循环的候选批次数; 组批失败不计入.
            - sampling_batch_completed_count: int, 原循环完整返回且CUDA同步结束的批次数.
            - model_forward_attempt_count: int, 实际进入模型调用的批量前向次数; 初次加噪失败为0, 不按候选数乘步数估计.
            - model_forward_completed_count: int, 模型Python调用完整返回次数; 异步CUDA异常可能在之后的同步阶段出现.
            - peak_memory_allocated_bytes: int|None, 本实例开始重置统计后的峰值张量显存, 含驻留模型; CPU为None, 单位字节.
            - peak_memory_reserved_bytes: int|None, 同一区间的峰值CUDA缓存分配量, 单位字节; CPU为None.
            - elapsed_seconds: float, 当前实例从预处理到写盘的总秒数.

            - candidate_file: str, 相对于当前实例目录的 candidates.json.
            - pose_file: str|None, 成功时为相对文件名 poses.sdf, 成功数0时为None.
            - confidence_file: str|None, 成功时为相对文件名 confidence.npz, 成功数0时为None.

    已存在 result.json 时直接返回已完成记录. 存储读取或写入的 OSError 直接传播, 不生成完成标记; 模型与科学输入异常逐候选保存.
    """
    record = dataset.records[index]
    identity = {"pdb_id": record["pdb_id"], "occurrence_id": int(record["candidate_id"]), "model_name": config.model_name, "split": config.split, "protocol": protocol}
    occurrence_dir = Path(config.output_root) / config.split / protocol / identity["pdb_id"] / str(identity["occurrence_id"])
    result_path = occurrence_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    occurrence_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sampling_device = torch.device(config.device)
    if sampling_device.type == "cuda":
        # 统计从本实例开始的峰值, 包含驻留模型和候选批次; 不用不同实例峰值的累计最大值代替.
        torch.cuda.reset_peak_memory_stats(sampling_device)
    # 原 seed_all 同步 NumPy、Python 和 Torch; 不把路径、协议名或模型名混入冻结种子.
    seed_all(int(record["sampling_seed"]))
    candidate_base = {
        **identity, "status": "failed", "error": None, "sdf_index": None,
        "cfd_traj": None, "cfd_pos": None, "cfd_node": None, "cfd_edge": None,
    }
    candidates = []
    confidence = {key: [] for key in ("confidence_pos_traj", "confidence_pos", "confidence_node", "confidence_halfedge")}
    success_indices = []
    inference_seconds = 0.0
    sampling_batch_attempt_count = sampling_batch_completed_count = 0
    model_forward_attempt_count = model_forward_completed_count = 0
    pocket_protein_count = pocket_nucleic_count = None
    pocket_center = None
    stage = "preprocess"

    try:
        data = dataset[index]
        pocket_protein_count = int(data.pocket_protein_count)
        pocket_nucleic_count = int(data.pocket_nucleic_count)
        pocket_center = data.pocket_center.tolist()
        # (N, 3), 定位条件已经固定, 推理状态和原循环需要的轨迹占位不携带沉积配体坐标.
        data.node_pos = torch.zeros_like(data.node_pos)
        data.gt_node_pos = torch.zeros_like(data.gt_node_pos)
        # 原 decode_output 必须读取真实模型原点; 官方纯核酸若在更早步骤失败, 会留下 preprocess 错误.
        _, _, template, _ = read_template(dataset.root / "ligand_objects" / (record["object_key"].replace(":", "_") + ".npz"))
        template = Chem.Mol(template)
        template.AddConformer(Chem.Conformer(template.GetNumAtoms()), assignId=True)
    except Exception as error:
        if isinstance(error, OSError):
            raise
        # 预处理没有生成任何候选, 仍为预算内每个候选保存具体实例身份和失败阶段.
        error_text = f"{type(error).__name__}: {error}"
        candidates = [{**candidate_base, "sample_index": sample_index, "stage": stage, "error": error_text} for sample_index in range(config.num_candidates)]
    else:
        # Batch 的 follow_batch 必须同时跟踪配体、完全图半边和口袋原子的分子归属.
        follow_batch = ["pocket_pos", *featurizer.follow_batch]
        exclude_keys = list(dict.fromkeys(featurizer.exclude_keys + dataset.transforms.transforms[-1].exclude_keys))
        writer = Chem.SDWriter(str(occurrence_dir / "poses.sdf"))
        try:
            for start in range(0, config.num_candidates, config.batch_size):
                stop = min(start + config.batch_size, config.num_candidates)
                stage = "batch"
                progress = {"stage": stage, "model_forward_attempt_count": 0, "model_forward_completed_count": 0}
                batch_started = time.perf_counter()
                try:
                    # 同一个 occurrence 的复制候选沿 PyG 图维排列, 每个候选的二维图与口袋条件相同.
                    batch = Batch.from_data_list([data.clone() for _ in range(stop - start)], follow_batch=follow_batch, exclude_keys=exclude_keys).to(config.device)
                    # 保留原 100 步循环和置信度轨迹聚合; 坐标和类别轨迹仅在内存短暂存在, 不另存完整去噪轨迹.
                    sampling_batch_attempt_count += 1
                    batch, outputs, trajectories = sample_loop3(batch, model, noiser, device=config.device, off_tqdm=True, progress=progress)
                    progress["stage"] = "synchronize"
                    if batch.node_pos.is_cuda:
                        torch.cuda.synchronize(batch.node_pos.device)
                    sampling_batch_completed_count += 1
                    stage = "split"
                    progress["stage"] = stage
                    generated, individual_outputs, _ = seperate_outputs2(batch, outputs, None, off_tqdm=True)
                    if len(generated) != stop - start or len(individual_outputs) != stop - start:
                        raise ValueError("sampling_batch_size_mismatch")
                    del trajectories
                except Exception as error:
                    if isinstance(error, OSError):
                        raise
                    stage = progress["stage"]
                    error_text = f"{type(error).__name__}: {error}"
                    candidates.extend({**candidate_base, "sample_index": sample_index, "stage": stage, "error": error_text} for sample_index in range(start, stop))
                    # 异常批次不参与后续候选生成, 先释放仍可达的GPU张量再清理缓存.
                    batch = outputs = trajectories = generated = individual_outputs = None
                    if sampling_device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                finally:
                    model_forward_attempt_count += progress["model_forward_attempt_count"]
                    model_forward_completed_count += progress["model_forward_completed_count"]
                    inference_seconds += time.perf_counter() - batch_started

                for local_index, (generated_mol, output) in enumerate(zip(generated, individual_outputs)):
                    candidate = {**candidate_base, "sample_index": start + local_index, "stage": "reconstruct"}
                    try:
                        mol_info = featurizer.decode_output(**generated_mol)
                        # (N, 3), 原 decode_output 已加回实际模型世界原点, 不做任何旋转对齐或中心修正.
                        if not np.isfinite(mol_info["atom_pos"]).all():
                            raise ValueError("non_finite_generated_coordinates")
                        molecule = reconstruct_pos(mol_info, in_mol=template)
                        # 此处直接用原固定拓扑重构函数, 不进入原通用重构包装器的改图回退分支.
                        candidate["stage"] = "confidence"
                        # [N,T] -> [T] -> scalar, 原函数先平均原子, 再平均后半程原始位置输出; 正式100步取后50步, 不做sigmoid.
                        cfd_traj = float(get_cfd_traj(output["confidence_pos_traj"], steps=config.num_steps))
                        if not np.isfinite(cfd_traj):
                            raise ValueError("non_finite_trajectory_confidence")
                        candidate["cfd_traj"] = cfd_traj
                        # 空半边或非有限的辅助置信度均值写为None, 原数组仍保留用于追溯; 排序只读取有限cfd_traj.
                        for field, key in (("cfd_pos", "confidence_pos"), ("cfd_node", "confidence_node"), ("cfd_edge", "confidence_halfedge")):
                            value = float(output[key].mean()) if output[key].numel() else None
                            candidate[field] = value if value is not None and np.isfinite(value) else None
                        # 在写SDF之前构造同一候选的四个数组, 避免缺少某个输出字段时留下未登记的SDF分子.
                        candidate_confidence = {key: output[key].float().cpu().numpy() for key in confidence}
                        molecule.SetIntProp("sample_index", candidate["sample_index"])
                        for key, value in identity.items():
                            molecule.SetProp(key, str(value))
                    except Exception as error:
                        candidate["error"] = f"{type(error).__name__}: {error}"
                    else:
                        # 磁盘、权限或配额错误必须中止任务; 写盘不在候选科学异常的捕获范围内, 因而不会获得完成标记.
                        writer.write(molecule)
                        candidate.update(status="success", stage="complete", sdf_index=len(success_indices))
                        success_indices.append(candidate["sample_index"])
                        # 所有置信度数组的第0维与 success_indices、SDF分子顺序同时对齐.
                        for key in confidence:
                            confidence[key].append(candidate_confidence[key])
                    candidates.append(candidate)
                del batch, outputs, generated, individual_outputs
        finally:
            writer.close()
    success_count = len(success_indices)
    if success_count:
        # [K 个 (N,T)/(N,1)/(H,1)] -> [K,N,T]/[K,N,1]/[K,H,1], 新增的首轴只表示成功候选.
        np.savez_compressed(occurrence_dir / "confidence.npz", sample_index=np.asarray(success_indices, dtype=np.int64), **{key: np.stack(value) for key, value in confidence.items()})
    (occurrence_dir / "candidates.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    total_pocket = (pocket_protein_count + pocket_nucleic_count) if pocket_protein_count is not None else 0
    result = {
        **identity,
        "object_key": record["object_key"],
        "views": record["views"],
        "sampling_seed": int(record["sampling_seed"]),
        "center_offset_xyz_A": record["center_offset_xyz_A"],
        "complete": True,
        "status": "success" if success_count == config.num_candidates else "partial" if success_count else "failed",
        "num_candidates": config.num_candidates,
        "success_count": success_count,
        "failed_count": config.num_candidates - success_count,
        "num_steps": config.num_steps,
        "batch_size": config.batch_size,
        "pocket_protein_count": pocket_protein_count,
        "pocket_nucleic_count": pocket_nucleic_count,
        "pocket_nucleic_fraction": pocket_nucleic_count / total_pocket if total_pocket else None,
        "model_origin_world_xyz_A": pocket_center,
        "inference_seconds": inference_seconds,
        "sampling_batch_attempt_count": sampling_batch_attempt_count,
        "sampling_batch_completed_count": sampling_batch_completed_count,
        "model_forward_attempt_count": model_forward_attempt_count,
        "model_forward_completed_count": model_forward_completed_count,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(sampling_device)) if sampling_device.type == "cuda" else None,
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(sampling_device)) if sampling_device.type == "cuda" else None,
        "elapsed_seconds": time.perf_counter() - started,
        "candidate_file": "candidates.json",
        "pose_file": "poses.sdf" if success_count else None,
        "confidence_file": "confidence.npz" if success_count else None,
    }
    # 同一目录内先写临时JSON再替换, 只有成功写完全部产物的实例才获得完成标记; 不删除已有服务器目录.
    temporary_result = occurrence_dir / "result.json.tmp"
    temporary_result.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary_result.replace(result_path)
    return result


def sample_docking(config):
    """装配明确 checkpoint, 顺序完成一个模型获准的全部定位协议.

    输入 config 的必需字段:
        - model_name: str, 稳定实验名称, 如 B-C-T0-RA.
        - train_config: str, 对应模型明确的训练YAML路径, 提供 model、transforms.featurizer 和 noise.
        - checkpoint: str, 明确的官方或训练 best 路径; 不搜索相邻目录猜测权重.
        - receptor_branch: str, protein保持官方25维蛋白入口, RA启用联合受体图与共享编码器.
        - dataset: Mapping, root、derived_root、manifest_root、knn与OccurrenceDataset相同; pocket_mode按当前协议设置.
        - protocols: list[str], 中心模型为[C0,C5], 包络为[E], 官方为[C0,C5,E].
        - split: str, validation或test, 两个划分分别保存完成标记.
        - output_root: str, 当前模型采样根目录.
        - batch_size: int, 同一occurrence每批候选数上限, 当前正式值50; 不跨实例组批.
        - num_candidates: int, 每实例候选预算, 正式值50.
        - num_steps: int, 每轮原采样步数, 正式值100.
        - device: str, 模型和原采样噪声所在设备, 如cuda.
        - evaluation_workers: int, 仅后续评价读取的CPU进程数, 正式值8.
        - wandb: Mapping, 仅后续评价读取, 子字段见evaluate_docking.

    输出 <output_root>/<split>/run.json, 顶层dict:
        - started_at: str, 首次成功加载模型的UTC时间, ISO8601格式.
        - science_config: dict, 完整采样配置除 batch_size、device、evaluation_workers、wandb 外的字段; 字段定义同上述输入, 用于阻止完成候选混入另一个实验.
        - configuration: dict, 首次完整采样配置快照, 包含资源和W&B记录偏好.
        - training_config: dict, 实际训练YAML的完整内容; model、noise、transforms定义网络、噪声和配体特征, 其余训练字段留作来源.

    模型确实加载成功后才冻结run.json. 各实例文件由sample_occurrence保存; 标准输出打印身份、协议、成功数和耗时, 由正式任务日志留存. 续跑只放行资源和记录偏好变化, 不覆盖已完成候选.
    """
    train_config = make_config(config.train_config)
    split_dir = Path(config.output_root) / config.split
    split_dir.mkdir(parents=True, exist_ok=True)
    run_path = split_dir / "run.json"
    # 资源参数和记录偏好不改变科学条件; 改 checkpoint 或实验定义却复用已完成候选会混合科学结果.
    science_config = {key: value for key, value in config.items() if key not in ("batch_size", "device", "evaluation_workers", "wandb")}
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if previous["science_config"] != science_config or previous["training_config"] != train_config:
            raise ValueError("sampling_output_contains_a_different_experiment")
    featurizer = FeaturizeMol(train_config.transforms.featurizer)

    model_config = deepcopy(train_config.model)
    model_config.nucleic_branch = None if config.receptor_branch == "protein" else config.receptor_branch

    model = PMAsymDenoiser(model_config, featurizer.num_node_types, featurizer.num_edge_types, pocket_in_dim=25).to(config.device)
    checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    # 只加载 Lightning state_dict 中 model. 参数; loss、优化器和调度器状态不进入推理模型.
    model.load_state_dict({key[len("model."):]: value for key, value in checkpoint["state_dict"].items() if key.startswith("model.")}, strict=True)
    model.eval()
    del checkpoint
    if not run_path.exists():
        # 模型确实加载成功后才冻结首次运行记录, 避免错误checkpoint路径占据尚未开始的实验目录.
        run_record = {"started_at": datetime.now(timezone.utc).isoformat(), "science_config": science_config, "configuration": config, "training_config": train_config}
        temporary_run = split_dir / "run.json.tmp"
        temporary_run.write_text(json.dumps(run_record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary_run.replace(run_path)
    task_transform = get_transforms(EasyDict(name="dock", settings={"free": 1}, free_no_geometry=True), mode="test")
    transforms = Compose([featurizer, task_transform])
    # 原 docking 测试配置定义 advance 调度, 直接复用该成熟配置, 只接入已批准的步数; C0/C5使用同一T0噪声公式.
    sample_config = make_config(str(Path(__file__).resolve().parents[1] / "configs/sample/test/dock_poseboff/base.yml"))
    for protocol in config.protocols:
        dataset_config = deepcopy(config.dataset)
        dataset_config.pocket_mode = "envelope" if protocol == "E" else "center"

        # 模型配置唯一决定是否读取密度; C0/C5只改变实际给定中心, 不改变T0采样公式.
        dataset = OccurrenceDataset(dataset_config, config.split, transforms, config.receptor_branch, protocol, shuffle=False, density_config=model_config.get('density'))

        noise_config = deepcopy(sample_config.noise)
        noise_config.num_steps = config.num_steps
        noiser = get_sample_noiser(noise_config, featurizer.num_node_types, featurizer.num_edge_types, mode="sample", device=config.device, ref_config=train_config.noise)
        for index in range(len(dataset.records)):
            result = sample_occurrence(dataset, index, model, noiser, featurizer, config, protocol)
            print(json.dumps({key: result[key] for key in ("model_name", "protocol", "pdb_id", "occurrence_id", "status", "success_count", "num_candidates", "elapsed_seconds")}, ensure_ascii=False), flush=True)
