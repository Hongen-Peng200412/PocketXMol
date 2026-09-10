"""继承原实例划分, 累计筛选合法 docking 样本并准备最少的共享派生资产.

主要入口 prepare 按 index、objects、samples、freeze 四阶段执行. CPU 数由配置 workers 决定, Slurm array 只切分互不重叠的模板或 PDB, 不改变科学筛选.
输入 root 是只读 AdaLigand Ori_Data; split_root 保存原训练/验证/校准质量清单, test_split 保存指定 held-out PDB 集合.
正式输出由 manifest_root 和 derived_root 决定; 中间诊断保存在 manifest_root/preparation, 字段见 prepare. 不复制源基础图、坐标或密度图, 不修复例外、不重算语言表征.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem

from docking.assets import read_receptor, read_template


def read_jsonl(path):
    """读 JSONL 为按文件顺序排列的字典列表; 每个字典表示一个模板或 occurrence 诊断记录."""
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, records):
    """将 records 中的字典逐项写为 UTF-8 JSONL, 严禁把 NaN 写成合法诊断值.

    path 的父目录由 prepare 建立. 对应字段由各阶段的记录构造处说明; 此函数不改变记录或筛选顺序.
    """
    with Path(path).open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def read_occurrences(root, split, pdb_id, candidate_ids):
    """读取一个 PDB 的原身份表, 保留当前划分授权的单残基、非共价 small_molecule.

    输入参数:
        - root: Path, 只读 Ori_Data 根路径.
        - split: str, train、validation、calibration 或 test.
        - pdb_id: str, 如 9v7o, 定位 parse/{pdb_id}/occurrences.jsonl.
        - candidate_ids: set[int] 或 None, 原质量清单允许的实例编号; None 仅表示指定测试 PDB 的全部原实例.

    返回值:
        - records: list[dict], 每条含 split、pdb_id、candidate_id、object_key; candidate_id 即 occurrence_id, object_key 如 CCD:GMP.
        - excluded: list[dict], 相同身份字段加 reason; reason 是具体不满足的条件或读取异常, 未知 object_key 为空字符串.
    """
    records, excluded = [], []
    try:
        occurrences = read_jsonl(root / "parse" / pdb_id / "occurrences.jsonl")
    except (OSError, ValueError) as error:
        ids = [-1] if candidate_ids is None else sorted(candidate_ids)
        return [], [dict(split=split, pdb_id=pdb_id, candidate_id=cid, object_key="", reason=f"occurrence_metadata: {error}") for cid in ids]
    found = set()
    for occurrence in occurrences:
        candidate_id = int(occurrence["candidate_id"])
        if candidate_ids is not None and candidate_id not in candidate_ids:
            continue
        found.add(candidate_id)
        record = dict(split=split, pdb_id=pdb_id, candidate_id=candidate_id, object_key=occurrence["object_key"])
        if occurrence["type_tag"] != "small_molecule":
            reason = "not_original_small_molecule"
        elif occurrence["kind"] != "CCD" or occurrence["polymer_length"] != 1 or len(occurrence["components"]) != 1:
            reason = "not_single_residue_ccd"
        elif occurrence["is_covalent"]:
            reason = "covalent_occurrence"
        else:
            records.append(record)
            continue
        excluded.append(dict(record, reason=reason))
    if candidate_ids is not None:
        excluded.extend(dict(split=split, pdb_id=pdb_id, candidate_id=cid, object_key="", reason="occurrence_metadata_missing_id") for cid in sorted(candidate_ids - found))
    return records, excluded


def prepare_object(root, derived_root, object_key):
    """检查完整模板图与原字符串, 只落盘原 reassign_in 实际读取的手性自同构排列. 同时过滤无碳原子的配体, 并排除配位键.

    输入参数:
        - root: Path, 只读 Ori_Data, 模板路径为 ligand_objects/{object_key 中冒号替换为下划线}.npz.
        - derived_root: Path, 本项目派生根路径, symmetries 子目录已建立.
        - object_key: str, 如 CCD:GMP; 不按 SMILES 合并不同 CCD.

    文件与返回值:
        - symmetries/CCD_GMP.npz: NPZ, 一个 object_key 仅由一个 array 分片拥有并写入.
            - object_key: 标量字符串, 与输入完全一致.
            - atom_count: int64 标量, 完整源模板的重原子数 N, 用于核对排列身份.
            - matches_iso: int64, (M, S), 最多10000种 RDKit 手性自匹配并补恒等排列; 只保留会被置换的 S 个原子列, 值索引完整模板的 N 个原子.
        - record: dict, 含 object_key、status(ok/excluded)、reason(成功时为空)、atom_count(成功时为 N, 失败为 null)、canonical_smiles(源字符串的无手性规范形式, 仅用于语言身份核对, 失败为 null).

    遵循 process_torsional_info 原匹配规则, 不计算真实扭转、刚体域、基础图副本或评价 RMSD 匹配表.
    """
    record = dict(object_key=object_key, status="excluded", reason="", atom_count=None, canonical_smiles=None)
    try:
        template_name = object_key.replace(":", "_") + ".npz"
        atoms, _, molecule, smiles = read_template(root / "ligand_objects" / template_name)
        string_molecule = Chem.MolFromSmiles(smiles)
        if string_molecule is None:
            raise ValueError("invalid_template_smiles")
        # (M, N), 每个候选排列中的值索引完整模板原子; 与原训练 useChirality=True、maxMatches=10000 一致.
        matches = np.array(molecule.GetSubstructMatches(molecule, uniquify=False, useChirality=True, maxMatches=10000), dtype=np.int64)
        natural_order = np.arange(len(atoms), dtype=np.int64)
        # (M,), True 标记恒等排列; 极端截断情况下显式补回, 保持原处理语义.
        is_natural = (matches == natural_order).all(axis=-1)
        if not is_natural.any():
            matches = np.concatenate([natural_order[None], matches], axis=0)
        # (N,), 只保留至少一个自同构会改变编号的原子列; 没有对称原子时 S=0.
        variable_atoms = (matches != natural_order).any(axis=0)
        record.update(status="ok", atom_count=len(atoms), canonical_smiles=Chem.MolToSmiles(string_molecule, isomericSmiles=False))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        record["reason"] = f"template: {error}"
        return record
    # 输出权限、配额或磁盘故障必须让任务失败, 不能把正常模板记成科学排除.
    np.savez_compressed(derived_root / "symmetries" / template_name, object_key=object_key, atom_count=len(atoms), matches_iso=matches[:, variable_atoms])
    return record


def prepare_pdb(root, derived_root, language_root, records, object_results):
    """一次打开一个 PDB 的共同资产, 检查产物完整性, 并拆出它的 ligand-area 标签.

    输入参数:
        - root: Path, 只读 Ori_Data 根路径.
        - derived_root: Path, 本项目派生根路径, ligand_area 子目录已建立.
        - language_root: Path, 已有 smi_ted_289m 根路径; 按 pdb_id/results.jsonl 和 candidate_{id}.npz 读取.
        - records: list[dict], 同一 PDB 的候选记录, 字段为 split、pdb_id、candidate_id、object_key.
        - object_results: dict[str, dict], object_key 到 prepare_object 返回记录, 失败模板直接排除所有相关实例.

    输出:
        - kept: list[dict], 原候选(records)字段加 n_heavy_atoms(完整模板重原子数); 后续 freeze 时再添加固定偏移、种子和视图.
        - excluded: list[dict], 原候选字段加 reason, 如 incomplete_deposited_heavy_atoms 或 language_status:model_failed.
        - ligand_area/{pdb_id}/{candidate_id}.npy: int32, (K, 3), 原 mask_{id} 的 K 个源体素 ZYX 索引, 不复制其它实例或密度数组.

    图、电荷、键级和 SMILES 均不修复. 共用地图的错误影响本 PDB 候选, 单实例错误只排除该实例; 源输入不作任何写入.
    """
    kept, excluded = [], []
    pdb_id = records[0]["pdb_id"]
    eligible = []
    for record in records:
        result = object_results[record["object_key"]]
        if result["status"] == "ok":
            eligible.append(record)
        else:
            excluded.append(dict(record, reason=result["reason"]))
    if not eligible:
        return kept, excluded

    try:
        receptor = read_receptor(root / "parse" / pdb_id / "receptor_tokens.npz")
        if not len(receptor["coords"]) or not np.isfinite(receptor["coords"]).all():
            raise ValueError("empty_or_nonfinite_standard_receptor")
        # 每张密度图只以 mmap 顺序检查一次. Z 轴每次32层, 限制临时布尔数组的内存占用.
        geometry = []
        for name in ("exp", "sim"):
            density = np.load(root / "density" / pdb_id / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if density.ndim != 4 or density.shape[0] != 1 or min(density.shape[1:]) < 48:
                raise ValueError(f"{name}_map_shape:{density.shape}")
            for start in range(0, density.shape[1], 32):
                if not np.isfinite(density[0, start:start + 32]).all():
                    raise ValueError(f"{name}_map_nonfinite")
            with np.load(root / "density" / pdb_id / f"{name}.npz", allow_pickle=False) as metadata:
                # 两个 (3,), 源地图边界角点 O 和实际体素尺寸 d, 均按世界 XYZ, 单位 Å.
                origin, spacing = metadata["origin"], metadata["voxel_size"]
            if origin.shape != (3,) or spacing.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(spacing).all() or np.any(spacing <= 0):
                raise ValueError(f"{name}_map_geometry")
            geometry.append((density.shape[1:], origin, spacing))
            del density
        if geometry[0][0] != geometry[1][0] or not np.array_equal(geometry[0][1], geometry[1][1]) or not np.array_equal(geometry[0][2], geometry[1][2]):
            raise ValueError("exp_sim_geometry_mismatch")
        language_results = {int(item["candidate_id"]): item for item in read_jsonl(language_root / pdb_id / "results.jsonl")}
        coord_archive = np.load(root / "parse" / pdb_id / "ligand_coords.npz", allow_pickle=False)
        area_archive = np.load(root / "density" / pdb_id / "ligand_area.npz", allow_pickle=False)
    except (OSError, ValueError, KeyError) as error:
        excluded.extend(dict(record, reason=f"pdb_assets: {error}") for record in eligible)
        return kept, excluded
    (derived_root / "ligand_area" / pdb_id).mkdir(parents=True, exist_ok=True)


    with coord_archive, area_archive:
        for record in eligible:
            candidate_id = record["candidate_id"]
            try:
                template_name = record["object_key"].replace(":", "_") + ".npz"
                atoms, _, _, _ = read_template(root / "ligand_objects" / template_name)
                # (N, 3) 与 (N,), 完整模板顺序的沉积世界 XYZ 和重原子存在标记; 不提取可见子图.
                coords, present = coord_archive[f"coords_{candidate_id}"], coord_archive[f"present_{candidate_id}"]
                if coords.shape != (len(atoms), 3) or present.shape != (len(atoms),) or not present.all() or not np.isfinite(coords).all():
                    raise ValueError("incomplete_deposited_heavy_atoms")
                language = language_results[candidate_id]
                if language["status"] != "encoded" or language["error"] is not None:
                    raise ValueError(f"language_status:{language['status']}")
                diagnostics = language["diagnostics"]
                if not diagnostics["official_normalization_success"] or diagnostics["official_normalization_error"] is not None or diagnostics["unsupported_tokens"] or diagnostics["token_diagnostic_error"] is not None or not diagnostics["all_finite"]:
                    raise ValueError("language_diagnostics_error")
                # 两个已存字符串可能采用 Kekule/芳香的不同写法. 只比较原 LM 所用的无手性规范身份, 不改写它们或重算向量.
                prepared_molecule = Chem.MolFromSmiles(language["prepared_smiles"])
                model_molecule = Chem.MolFromSmiles(language["model_smiles"])
                if prepared_molecule is None or model_molecule is None:
                    raise ValueError("language_invalid_smiles")
                canonical_smiles = object_results[record["object_key"]]["canonical_smiles"]
                if Chem.MolToSmiles(prepared_molecule, isomericSmiles=False) != canonical_smiles or Chem.MolToSmiles(model_molecule, isomericSmiles=False) != canonical_smiles:
                    raise ValueError("language_input_mismatch_or_invalid_smiles")
                with np.load(language_root / pdb_id / f"candidate_{candidate_id}.npz", allow_pickle=False) as embedding:
                    # (768,), 冻结 SMI-TED 单实例向量; 只检查身份、形状和有限性, 不补算或写副本.
                    vector = embedding["embedding"]
                    # NPZ的model_name保存显示名称, 已核对为SMI-TED Light 289M; 它不是目录名smi_ted_289m.
                    if language["pdb_id"] != pdb_id or str(embedding["pdb_id"].item()) != pdb_id or int(embedding["candidate_id"].item()) != candidate_id or str(embedding["object_key"].item()) != record["object_key"] or str(embedding["model_name"].item()) != "SMI-TED Light 289M":
                        raise ValueError("language_identity_mismatch")
                    if str(embedding["prepared_smiles"].item()) != language["prepared_smiles"] or str(embedding["model_smiles"].item()) != language["model_smiles"]:
                        raise ValueError("language_string_mismatch")
                    if vector.shape != (768,) or not np.isfinite(vector).all():
                        raise ValueError("language_embedding_invalid")
                # int32, (K, 3), 原标签的源 ZYX 体素编号; 全部位于 exp/sim 共用源图内.
                area = area_archive[f"mask_{candidate_id}"]
                if area.ndim != 2 or area.shape[1] != 3 or not np.issubdtype(area.dtype, np.integer) or np.any(area < 0) or np.any(area >= np.asarray(geometry[0][0])):
                    raise ValueError("ligand_area_invalid_indices")
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                excluded.append(dict(record, reason=f"occurrence_assets: {error}"))
                continue
            # 科学检查通过后才写派生标签; 写盘故障向外抛出, 不缩小共同清单的分母.
            np.save(derived_root / "ligand_area" / pdb_id / f"{candidate_id}.npy", area.astype(np.int32, copy=False), allow_pickle=False)
            kept.append(dict(record, n_heavy_atoms=len(atoms)))
    return kept, excluded


# ================================================================================================
def prepare(config, stage, shard_id, shard_count):
    """按依赖顺序组织共同数据准备, 一个 CLI 调用只运行一个明确阶段.

    输入参数:
        - config.root: str, 只读 Ori_Data 根路径.
        - config.split_root: str, 原 train/validation/calibration.json 所在目录, 文件顶层为质量通过的 occurrence 记录列表.
        - config.test_split: str, 指定 test_0.json, 顶层 pdb_ids 为测试 PDB 列表且 occurrence_filter 为 null.
        - config.language_root: str, 原 smi_ted_289m 的逐 PDB 表征目录.
        - config.derived_root: str, 只保存 symmetries 和 ligand_area 的本项目派生根路径.
        - config.manifest_root: str, 当前共同冻结清单和准备诊断目录.
        - config.workers: int, 每个 CPU 任务的并行进程数, 当前8; 底层 BLAS 线程由任务脚本限制为1.
        - config.freeze_seed: int, 当前3407, 冻结 C5 偏移与测试身份内排序.
        - config.sampling_seed: int, 当前10831, 按冻结实例顺序递增分配正式候选池种子.
        - stage: str, index、objects、samples、freeze, 按此先后完成所有分片.
        - shard_id/shard_count: int, 当前数组编号与分片总数, 如0/12; index 使用0/1, freeze 使用0和此前数组的分片总数.

    清单和诊断文件:
        - preparation/sources.jsonl: 候选身份记录, 含 split、pdb_id、candidate_id、object_key, 不含图或坐标.
        - preparation/index_excluded.jsonl: 身份条件排除记录, 上述字段加 reason.
        - preparation/objects_{id}.jsonl: 当前分片每个模板的 object_key、status、reason、atom_count、canonical_smiles, 字段见 prepare_object.
        - preparation/samples_{id}.jsonl: 当前分片通过实例的身份与 n_heavy_atoms.
        - preparation/excluded_{id}.jsonl: 当前分片排除实例的身份与 reason.
        - train/validation/calibration/test.jsonl: 最终累计通过的实例, 原身份字段、n_heavy_atoms 与下列冻结字段.
            - center_offset_xyz_A: list[float], 长度3, 半径 Uniform(0,5) 与均匀球面方向构造的固定 C5 世界 XYZ 偏移, 单位 Å.
            - sampling_seed: int, 该实例在每个模型/协议的候选生成种子, 不承诺改变候选批量后保持相同随机轨迹.
            - views: list[str], test 的 ALL/CAP10/HF10_TO5 归属; 非测试为空列表.
        - excluded.jsonl: 累计排除记录, 每个原候选最多一个首个失败原因.
        - summary.json: dict, 含 source_paths、freeze_seed、sampling_seed、splits 和 excluded_reasons; splits 各项含 occurrences、pdbs、objects、views 计数.

    index/freeze 不自行申请其它资源. objects 按完整 object_key 排序切片, samples 按完整 PDB 排序切片, 因而同一派生文件没有跨 array 写入竞争.
    """
    root, derived_root, output_root = Path(config.root), Path(config.derived_root), Path(config.manifest_root)
    work_root = output_root / "preparation"
    work_root.mkdir(parents=True, exist_ok=True)
    (derived_root / "symmetries").mkdir(parents=True, exist_ok=True)
    (derived_root / "ligand_area").mkdir(parents=True, exist_ok=True)
    if stage == "index":
        if (shard_id, shard_count) != (0, 1):
            raise ValueError("index requires shard 0/1")
        # 两个产物根保存同一字段说明, 让只查看服务器数据的读者也能解释文件; 文档不作为筛选输入.
        readme = Path(__file__).with_name("README.md").read_text(encoding="utf-8")
        (output_root / "README.md").write_text(readme, encoding="utf-8")
        (derived_root / "README.md").write_text(readme, encoding="utf-8")
        requests = []
        for split in ("train", "validation", "calibration"):
            # dict[str, set[int]], 只承接原质量清单里的 small_molecule 编号; 不因 PDB 入选而补回其余原实例.
            by_pdb = defaultdict(set)
            with (Path(config.split_root) / f"{split}.json").open(encoding="utf-8") as stream:
                for record in json.load(stream):
                    if record["type_tag"] == "small_molecule":
                        by_pdb[record["pdb_id"]].add(int(record["candidate_id"]))
            requests.extend((split, pdb_id, ids) for pdb_id, ids in sorted(by_pdb.items()))
        with Path(config.test_split).open(encoding="utf-8") as stream:
            test_source = json.load(stream)
        if test_source["occurrence_filter"] is not None:
            raise ValueError("test source must preserve the approved null occurrence_filter")
        requests.extend(("test", pdb_id, None) for pdb_id in sorted(test_source["pdb_ids"]))
        results = Parallel(n_jobs=config.workers)(delayed(read_occurrences)(root, *request) for request in requests)
        sources = sorted([record for kept, _ in results for record in kept], key=lambda record: (record["split"], record["pdb_id"], record["candidate_id"]))
        excluded = [record for _, errors in results for record in errors]
        write_jsonl(work_root / "sources.jsonl", sources)
        write_jsonl(work_root / "index_excluded.jsonl", excluded)
        print(json.dumps(dict(stage=stage, candidates=len(sources), excluded=len(excluded))), flush=True)

    elif stage == "objects":
        sources = read_jsonl(work_root / "sources.jsonl")
        objects = sorted({record["object_key"] for record in sources})[shard_id::shard_count]
        results = Parallel(n_jobs=config.workers)(delayed(prepare_object)(root, derived_root, object_key) for object_key in objects)
        write_jsonl(work_root / f"objects_{shard_id}.jsonl", results)
        print(json.dumps(dict(stage=stage, shard=shard_id, status_counts=dict(Counter(record["status"] for record in results)))), flush=True)

    elif stage == "samples":
        # 所有 object 分片必须先结束; 每条模板状态是对应模板唯一的准备结果, 不在 occurrence 循环重新生成.
        object_results = {record["object_key"]: record for index in range(shard_count) for record in read_jsonl(work_root / f"objects_{index}.jsonl")}
        by_pdb = defaultdict(list)
        for record in read_jsonl(work_root / "sources.jsonl"):
            by_pdb[record["pdb_id"]].append(record)
        pdb_ids = sorted(by_pdb)[shard_id::shard_count]
        results = Parallel(n_jobs=config.workers)(delayed(prepare_pdb)(root, derived_root, Path(config.language_root), by_pdb[pdb_id], {record["object_key"]: object_results[record["object_key"]] for record in by_pdb[pdb_id]}) for pdb_id in pdb_ids)
        kept = [record for passed, _ in results for record in passed]
        excluded = [record for _, errors in results for record in errors]
        write_jsonl(work_root / f"samples_{shard_id}.jsonl", kept)
        write_jsonl(work_root / f"excluded_{shard_id}.jsonl", excluded)
        print(json.dumps(dict(stage=stage, shard=shard_id, kept=len(kept), excluded=len(excluded))), flush=True)

    elif stage == "freeze":
        if shard_id != 0:
            raise ValueError("freeze requires shard_id=0")
        kept = [record for index in range(shard_count) for record in read_jsonl(work_root / f"samples_{index}.jsonl")]
        excluded = read_jsonl(work_root / "index_excluded.jsonl") + [record for index in range(shard_count) for record in read_jsonl(work_root / f"excluded_{index}.jsonl")]
        kept.sort(key=lambda record: (record["split"], record["pdb_id"], record["candidate_id"]))
        rng = np.random.default_rng(config.freeze_seed)
        for index, record in enumerate(kept):
            direction = rng.normal(size=3)
            # (3,), 方向均匀、半径均匀; 与训练动态偏移采用相同几何分布.
            offset = direction / np.linalg.norm(direction) * rng.uniform(0.0, 5.0)
            record.update(center_offset_xyz_A=offset.tolist(), sampling_seed=int(config.sampling_seed + index), views=[])

        # ----- 处理测试集 -----
        by_object = defaultdict(list)
        for record in kept:
            if record["split"] == "test":
                by_object[record["object_key"]].append(record)
        for object_key in sorted(by_object):
            records = by_object[object_key]
            # 同一身份只产生一次排列. 频数不超过10时, 两个截断视图都完整保留, 包括6..10的身份.
            order = rng.permutation(len(records))
            for rank, index in enumerate(order):
                records[index]["views"] = ["ALL"]
                if len(records) <= 10 or rank < 10:
                    records[index]["views"].append("CAP10")
                if len(records) <= 10 or rank < 5:
                    records[index]["views"].append("HF10_TO5")
        # ----- 处理测试集 -----
        summary = dict(source_paths={key: str(config[key]) for key in ("root", "split_root", "test_split", "language_root", "derived_root")}, freeze_seed=config.freeze_seed, sampling_seed=config.sampling_seed, splits={}, excluded_reasons=dict(Counter(record["reason"] for record in excluded)))
        for split in ("train", "validation", "calibration", "test"):
            # 这个新列表仍然只是收集 kept 中的字典引用，并没有复制字典。因此前面已经写入的 views 仍然存在
            records = [record for record in kept if record["split"] == split]
            write_jsonl(output_root / f"{split}.jsonl", records)
            summary["splits"][split] = dict(occurrences=len(records), pdbs=len({record["pdb_id"] for record in records}), objects=len({record["object_key"] for record in records}), views=dict(Counter(view for record in records for view in record["views"])))
        write_jsonl(output_root / "excluded.jsonl", excluded)
        (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(summary["splits"]), flush=True)
    else:
        raise ValueError(f"unknown preparation stage: {stage}")
