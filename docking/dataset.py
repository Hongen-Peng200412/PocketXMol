"""把冻结 occurrence 清单装配为原 PocketMolData, 接入原 free docking 变换.

主要入口是 OccurrenceDataset. 它读取公共SMILES图、沉积配体坐标和完整标准受体, 在内存选择口袋、编码蛋白/核酸、确定模型原点, 最后交给调用方提供的原配体特征、任务和噪声变换.
本模块不落盘, 不重新构建科学资产. 训练对合法实例均匀有放回抽样; 有限验证按 worker 跨步分片, 不丢尾部.
"""

import json
import warnings
from itertools import count
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torch_geometric.nn import knn_graph

from docking.assets import read_receptor, select_pocket
from docking.density import load_density_input
from docking.smiles import read_smiles_graph, read_smiles_coords, UnsupportedSmilesError
from utils.data import PocketMolData
from utils.transforms import FeaturizePocket


# int64, (20,), AdaLigand 标准氨基酸编号映射到原 PDBProtein.AA_NAME_NUMBER 的类别顺序.
PROTEIN_CLASS = np.array([0, 14, 11, 2, 1, 13, 3, 5, 6, 7, 9, 8, 10, 4, 12, 15, 16, 18, 19, 17], dtype=np.int64)
# 组分 one-hot 的顺序是 base、sugar、phosphate; O3'/O5' 与其它糖环/糖连接氧归入 sugar.
SUGAR_ATOMS = {"C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O3'", "O4'", "O5'"}
PHOSPHATE_ATOMS = {"P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P"}


class EmptyEnvelopePocketError(ValueError):
    """RA的E口袋没有标准受体原子, 无法按实际受体坐标均值定义原点."""


def assemble_docking_condition(
    dataset_config,
    root,
    transforms,
    receptor_branch,
    density_config,
    pdb_id,
    sample_id,
    prepared_smiles,
    graph,
    ligand_coords,
    given_center,
):
    """以一组世界坐标装配中心或包络条件，并执行原配体变换。

    该函数是标准 ``OccurrenceDataset`` 与无真值端到端 Dataset 的唯一受体条件入口。
    两者只在 ``ligand_coords`` 和 ``given_center`` 的来源上不同；残基选择、RA 过滤、
    模型原点、密度查询与原 ``FeaturizeMol``/dock 任务变换完全共用。

    输入参数:
        - dataset_config: Mapping，至少含 ``pocket_mode`` 与 ``knn``；前者为 ``center``
          或 ``envelope``。
        - root: Path，当前受体条件的资产根，含 ``parse``，密度模型还须含 ``density``。
        - transforms: callable，原配体特征化与 dock 任务变换的组合。
        - receptor_branch: str，``protein`` 使用官方蛋白入口，``RA`` 使用共享蛋白/核酸图。
        - density_config: Mapping|None，``None`` 为无密度模型，否则必须是 ``local_cov``。
        - pdb_id: str，当前结构编号。
        - sample_id: int，当前输入在其冻结清单中的实例编号，只用于模型数据身份。
        - prepared_smiles: str，精确 SMILES 字符串。
        - graph: dict，由 ``read_smiles_graph`` 返回，原子顺序与 ``ligand_coords`` 一致。
        - ligand_coords: float32，(N,3)，世界 XYZ、Å。包络模式用它选择残基和查询密度；
          中心模式只把它作为原任务变换所需的坐标占位，正式采样随后清零局部坐标。
        - given_center: float32，(3,)，中心模式的实际给定中心；包络模式不以它定义口袋。

    返回值:
        - data: PocketMolData，已经完成受体装配及调用方传入的原配体变换。

    异常:
        - EmptyEnvelopePocketError: RA 包络没有选入任何标准受体原子。
        - KeyError: ``local_cov`` 受体资产缺少 ``feat``。
    """
    ligand_coords = np.asarray(ligand_coords, dtype=np.float32)
    given_center = np.asarray(given_center, dtype=np.float32).reshape(3)
    if ligand_coords.shape != (len(graph["element"]), 3):
        raise ValueError(
            f"{pdb_id}/{sample_id}: 配体坐标形状{ligand_coords.shape}与"
            f"SMILES重原子数{len(graph['element'])}不一致。"
        )

    receptor = read_receptor(root / "parse" / pdb_id / "receptor_tokens.npz")
    # bool, (P_full,), P_full为完整标准受体重原子数；True保留该原子所属的完整残基。
    selected = select_pocket(receptor, ligand_coords, given_center, dataset_config.pocket_mode)
    protein_count = int(np.sum(selected & (receptor["res_type"] < 20)))
    nucleic_count = int(np.sum(selected & (receptor["res_type"] >= 20)))
    if receptor_branch == "protein":
        selected &= receptor["res_type"] < 20
    # 逐原子数组沿同一个掩码切分；P是当前模型实际输入的口袋重原子数。
    pocket = {key: value[selected] for key, value in receptor.items()}
    if dataset_config.pocket_mode == "envelope" and receptor_branch == "RA" and len(pocket["coords"]) == 0:
        raise EmptyEnvelopePocketError(f"empty_envelope_pocket: {pdb_id}/{sample_id}")

    pocket_pos = torch.from_numpy(pocket["coords"])
    pocket_is_nucleic = torch.from_numpy(pocket["res_type"] >= 20)
    # (M,)和(2,M)，公共SMILES图已经映射为原模型键类别与公共原子顺序。
    bond_types, bond_index = graph["bond_type"], graph["bond_index"]
    data = PocketMolData(
        data_id=f"{pdb_id}_{sample_id}",
        pdbid=pdb_id,
        task="dock",
        db="adaligand",
        candidate_id=sample_id,
        prepared_smiles=prepared_smiles,
        element=torch.from_numpy(graph["element"]),
        pos_all_confs=torch.from_numpy(ligand_coords[None]),
        i_conf_list=[0],
        num_confs=1,
        num_atoms=len(graph["element"]),
        num_bonds=len(bond_types),
        bond_index=torch.from_numpy(np.concatenate([bond_index, bond_index[::-1]], axis=1)),
        bond_type=torch.from_numpy(np.concatenate([bond_types, bond_types])),
        matches_iso=graph["matches_iso"],
        pocket_pos=pocket_pos,
        pocket_protein_count=protein_count,
        pocket_nucleic_count=nucleic_count,
    )
    if receptor_branch == "protein":
        # 官方只读蛋白，并保留原FeaturizePocket对空蛋白的实际行为。
        data.pocket_element = torch.from_numpy(pocket["element"].astype(np.int64))
        data.pocket_atom_to_aa_type = torch.from_numpy(PROTEIN_CLASS[pocket["res_type"]])
        data.pocket_is_backbone = torch.from_numpy(pocket["is_backbone"])
        pocket_config = EasyDict(knn=dataset_config.knn)
        if dataset_config.pocket_mode == "center":
            pocket_config.center = given_center.tolist()
        data = FeaturizePocket(pocket_config)(data)
    else:
        # (1,3)，中心模式取实际给定中心；包络模式取实际选入受体原子的算术均值。
        center = (
            given_center[None]
            if dataset_config.pocket_mode == "center"
            else pocket["coords"].mean(axis=0, keepdims=True)
        )
        data.pocket_center = torch.from_numpy(center.astype(np.float32))
        data.pocket_pos = pocket_pos - data.pocket_center
        data.pocket_is_nucleic = pocket_is_nucleic
        # (P,25)与(P,15)，蛋白和核酸特征按同一口袋原子顺序对齐，另一类型位置保持0。
        protein_features = torch.zeros((len(pocket_pos), 25))
        nucleic_features = torch.zeros((len(pocket_pos), 15))
        protein = ~pocket_is_nucleic
        protein_elements = torch.from_numpy(pocket["element"][~pocket_is_nucleic.numpy()].astype(np.int64))
        protein_features[protein, :4] = (
            protein_elements[:, None] == torch.tensor([6, 7, 8, 16])
        ).float()
        protein_features[protein, 4:24] = torch.nn.functional.one_hot(
            torch.from_numpy(PROTEIN_CLASS[pocket["res_type"][protein.numpy()]]),
            num_classes=20,
        ).float()
        protein_features[protein, 24] = torch.from_numpy(
            pocket["is_backbone"][protein.numpy()].astype(np.float32)
        )
        nucleic_elements = torch.from_numpy(
            pocket["element"][pocket_is_nucleic.numpy()].astype(np.int64)
        )
        nucleic_features[pocket_is_nucleic, :4] = (
            nucleic_elements[:, None] == torch.tensor([6, 7, 8, 15])
        ).float()
        nucleic_features[pocket_is_nucleic, 4:12] = torch.nn.functional.one_hot(
            torch.from_numpy(
                pocket["res_type"][pocket_is_nucleic.numpy()].astype(np.int64) - 20
            ),
            num_classes=8,
        ).float()
        # list[str]，长度P_na；将C1*等源原子名规范为C1'后判定核酸组分。
        names = [
            name.decode("ascii").strip().replace("*", "'")
            for name in pocket["atom_name"][pocket_is_nucleic.numpy()]
        ]
        # int64, (P_na,)，0/1/2分别表示碱基、糖和磷酸。
        components = torch.tensor(
            [
                1 if name in SUGAR_ATOMS else 2 if name in PHOSPHATE_ATOMS else 0
                for name in names
            ],
            dtype=torch.long,
        )
        nucleic_features[pocket_is_nucleic, 12:15] = torch.nn.functional.one_hot(
            components,
            num_classes=3,
        ).float()
        data.pocket_atom_feature = protein_features
        data.pocket_nucleic_feature = nucleic_features
        # int64, (2,E_p)，蛋白和核酸共同组成kNN图，端点索引当前pocket_pos。
        if len(pocket_pos) > 1:
            data.pocket_knn_edge_index = knn_graph(
                data.pocket_pos,
                k=min(dataset_config.knn, len(pocket_pos) - 1),
                flow="target_to_source",
            )
        else:
            data.pocket_knn_edge_index = torch.empty((2, 0), dtype=torch.long)

    if density_config is not None:
        if "feat" not in pocket:
            raise KeyError(f"{pdb_id}/{sample_id}: receptor_tokens.npz缺少local_cov所需的feat字段。")
        # (P,50)，只使用当前口袋实际选中的标准RA原子；UNK已经由read_receptor排除。
        data.pocket_density_feature = torch.from_numpy(
            np.concatenate(
                [
                    pocket["feat"].astype(np.float32),
                    pocket["is_backbone"].astype(np.float32)[:, None],
                ],
                axis=1,
            )
        )
        # (3,)，中心按实际给定中心裁图；包络按用于选袋的配体坐标质心裁图。
        query_center = (
            given_center
            if dataset_config.pocket_mode == "center"
            else ligand_coords.mean(axis=0)
        )
        for name, value in load_density_input(
            root,
            pdb_id,
            query_center,
            data.pocket_center.numpy(),
        ).items():
            data[name] = value
    return transforms(data)


# ================================================================================================
class OccurrenceDataset(IterableDataset):
    """读取一个划分的完整实例并产出单个分子图, 训练与采样使用同一装配规则.

    N 为当前完整配体重原子数, P 为当前模型实际输入的口袋受体重原子数, M 为配体无向化学键数.

    构造参数:
        - dataset_config.root: str, AdaLigand 只读 Ori_Data 根路径.
        - dataset_config.smiles_root: str, 公共 smiles_graphs_v1.npz 与 smiles_symmetries_v1.npz 所在目录.
        - dataset_config.smiles_coords_root: str, 按公共原子顺序迁移的逐 PDB 坐标包目录.
        - dataset_config.derived_root: str, 本项目共享 ligand_area 的派生根路径; 自同构排列从公共SMILES包读取.
        - dataset_config.manifest_root: str, train.jsonl、validation.jsonl、calibration.jsonl、test.jsonl 的冻结清单根路径.
        - dataset_config.pocket_mode: str, center 或 envelope, 决定残基选择和模型原点.
        - dataset_config.knn: int, 口袋编码器的原近邻数, 当前为32.
        - split: str, 如 train 或 validation, 选择同名 JSONL 文件.
        - transforms: callable, 接收 PocketMolData; 训练依次执行 FeaturizeMol、原任务变换和原噪声器.
        - receptor_branch: str, protein 走官方蛋白特征入口, RA 在蛋白与核酸联合图中共享编码器.
        - protocol: str, C0、C5 或 E, 决定定位输入条件; 中心训练和监督验证固定C0; C0 用真实配体几何中心, C5 用中心加已有冻结偏移, E 用包络口袋.
        - shuffle: bool, True 为无限均匀有放回训练流; False 为一次完整有限流, 不控制定位或噪声机制.
        - density_config: Mapping|None, 调用方从model.density传入的唯一密度入口；None兼容原无密度模型，非None固定为`local_cov`。

    清单每条记录:
        - pdb_id: str, 如 9v7o, 定位源 parse 和 density 子目录.
        - candidate_id: int, 如0, 即本项目 occurrence_id, 匹配迁移坐标包 candidate_ids 中的实例编号.
        - prepared_smiles: str, 如 CCO, 精确匹配公共图和自同构包中的字符串, 不重新规范化.
        - unsupported_smiles_reason: str, 仅迁移不支持实例具有此字段; 正式测试记录失败而不删除实例.
        - center_offset_xyz_A: list[float], 长度3, 当前实例冻结的 C5 世界 XYZ 偏移, 单位 Å.
        - sampling_seed: int, 当前实例正式候选池的固定种子; 此类不执行候选生成.
        - views: list[str], 当前实例所属测试视图, 如 ["ALL", "CAP10", "HF10_TO5"].

    新增内存字段:
        - pocket_is_nucleic: bool, (P,), 与 pocket_pos 第一维对齐, True 为标准 RNA/DNA 原子.
        - pocket_nucleic_feature: float32, (P, 15), 核酸的4元素+8核苷酸+3组分 one-hot; 蛋白位置为0.
        - pocket_atom_feature: float32, (P, 25), 原蛋白4元素+20氨基酸+1主链特征; 核酸位置为0.
        - pocket_protein_count/pocket_nucleic_count: int, 当前协议选袋在官方蛋白过滤前的两类重原子数, 供报告核酸占比.
        - pos_all_confs: float32, (1, N, 3), 源沉积世界坐标的单个构象; 原 FeaturizeMol 随后减去 pocket_center.
        - pocket_density_feature: float32, (P,50), 仅`local_cov`提供；源49维受体特征后拼主链0/1标记，与pocket_pos逐原子对齐.
        - density_input: float32, (1, 56, 80, 80, 80), 仅密度模型提供, 一个实例实际裁块的固定通道, PyG沿首维拼成批量B.
        - density_origin: float32, (1, 3), 实际裁块边界角点减pocket_center, 模型局部XYZ坐标, 单位Å.
        - density_basis: float32, (1, 3, 3), 三行分别是源XYZ方向单体素在模型坐标中的向量, 初始为实际间距的对角阵, 单位Å.
        - density_start_zyx: int64, (1, 3), 实际源裁块起点, 如[[10,12,8]], 内缩后仍不改变pocket_center.

    原字段 num_atoms、bond_index、bond_type 和 matches_iso 仍按公共SMILES原子顺序解释; 配体类别、fixed prompt、空刚体域和噪声叶由原变换生成.
    """

    def __init__(self, dataset_config, split, transforms, receptor_branch, protocol, shuffle, density_config=None):
        """保存明确配置并读取唯一的冻结实例清单; 不枚举目录补回被排除的实例."""
        super().__init__()
        self.config = dataset_config
        self.split = split
        self.transforms = transforms
        self.receptor_branch = receptor_branch
        self.protocol = protocol
        self.shuffle = shuffle
        self.density_config = density_config
        if density_config is not None and density_config.get("name") != "local_cov":
            raise ValueError("model.density当前只允许name=local_cov。")
        self.root = Path(dataset_config.root)
        self.derived_root = Path(dataset_config.derived_root)
        with (Path(dataset_config.manifest_root) / f"{split}.jsonl").open(encoding="utf-8") as stream:
            # list[dict], 每个元素是一条已通过累计筛选的 occurrence 记录; 字段见类契约.
            self.records = [json.loads(line) for line in stream if line.strip()]
        self.rng = None

    def __getitem__(self, index):
        """装配 records[index] 的口袋与完整配体图, 再执行调用方指定的原变换.

        返回 PocketMolData, 核心字段见类说明. 中心C0原点为完整配体几何中心g, 评测C5原点为g+delta; delta是清单中已冻结的XYZ向量, 单位Å. 实际给定中心用于选袋和原点, 局部真值质心相应为0或-delta. 本类不抽取随机偏移, 不添加带噪坐标的整体平移.
        已记录的SMILES不支持实例抛出UnsupportedSmilesError; RA空E在求均值前抛出EmptyEnvelopePocketError, 信息包含pdb_id/occurrence_id; __iter__仅在train/validation跳过, 正式采样直接索引并记录输入失败. 其它缺失或损坏资产直接抛出源异常, 不在训练热路径修复.
        """
        record = self.records[index]
        pdb_id, candidate_id = record["pdb_id"], int(record["candidate_id"])
        if "unsupported_smiles_reason" in record:
            raise UnsupportedSmilesError(
                f"{pdb_id}/{candidate_id}: {record['unsupported_smiles_reason']}"
            )
        graph = read_smiles_graph(self.config.smiles_root, record["prepared_smiles"])
        ligand_coords = read_smiles_coords(
            self.config.smiles_coords_root,
            pdb_id,
            candidate_id,
            record["prepared_smiles"],
        )

        # (3,)，occurrence沉积重原子的几何中心，只用于已经批准的定位条件。
        ligand_center = ligand_coords.mean(axis=0)
        # float32, (3,)，C5读取冻结偏移；C0和E不增加中心偏移。
        offset = (
            np.asarray(record["center_offset_xyz_A"], dtype=np.float32)
            if self.protocol == "C5"
            else np.zeros(3, dtype=np.float32)
        )
        # float32, (3,)，实际给定中心同时决定中心口袋和中心模型原点。
        given_center = (ligand_center + offset).astype(np.float32)
        return assemble_docking_condition(
            dataset_config=self.config,
            root=self.root,
            transforms=self.transforms,
            receptor_branch=self.receptor_branch,
            density_config=self.density_config,
            pdb_id=pdb_id,
            sample_id=candidate_id,
            prepared_smiles=record["prepared_smiles"],
            graph=graph,
            ligand_coords=ligand_coords,
            given_center=given_center,
        )

    def __iter__(self):
        """训练无限均匀抽实例, 有限流按worker跨步读取; 仅train/validation跳过并警告RA空E或明确不支持的SMILES实例.

        跳过发生在组批前, 训练仍由有效实例组成完整batch; 验证只对有效E计算原val/loss. 警告记录划分和pdb_id/occurrence_id, 冻结清单不变. 其它异常直接传播; 正式采样按records索引, 不经过这里的跳过逻辑.
        """
        worker = get_worker_info()
        worker_id, worker_count = (0, 1) if worker is None else (worker.id, worker.num_workers)
        # 独立 Generator 只用于训练实例抽样, 不抽取中心偏移, 不改变原噪声器的随机序列.
        if self.rng is None:
            self.rng = np.random.default_rng(torch.initial_seed())
        if self.shuffle:
            # 无限整数流, 每次仍从原冻结清单独立均匀抽样; 仅明确不支持的SMILES实例或空E被拒绝后重新抽取.
            indices = (int(self.rng.integers(len(self.records))) for _ in count())
        else:
            indices = range(worker_id, len(self.records), worker_count)
        for index in indices:
            try:
                yield self[index]
            except (EmptyEnvelopePocketError, UnsupportedSmilesError) as error:
                if self.split not in ("train", "validation"):
                    raise
                warnings.warn(f"{self.split} 跳过 {error}", RuntimeWarning)


class ConditionedDockingDataset(Dataset):
    """从冻结的给定中心或预测构象装配端到端推理输入，不读取沉积配体坐标。

    ``records`` 中每项固定含 ``pdb_id``、``candidate_id``、``prepared_smiles``、
    ``sampling_seed``、``views`` 和 ``center_offset_xyz_A``，以便直接复用
    ``sampling.sample_occurrence``。中心记录另含 ``given_center_xyz_A``；包络记录另含
    ``envelope_coords_xyz_A``。后者是第二次中心推理Top-1的完整重原子世界XYZ坐标，
    原子顺序必须与 ``prepared_smiles`` 的公共图一致。

    本类没有 ``smiles_coords_root``，也不接受 occurrence 真值坐标。中心模式为原任务变换
    构造位于给定中心的零跨度坐标占位；``sample_occurrence`` 在进入原采样器前仍会把局部
    ``node_pos`` 与 ``gt_node_pos`` 清零。包络模式把预测构象同时交给公共选袋和密度查询。
    """

    def __init__(
        self,
        dataset_config,
        records,
        transforms,
        receptor_branch,
        density_config=None,
    ):
        """保存一组已冻结端到端记录与当前模型条件，不枚举或补充其它候选。"""
        self.config = dataset_config
        self.records = records
        self.transforms = transforms
        self.receptor_branch = receptor_branch
        self.density_config = density_config
        if density_config is not None and density_config.get("name") != "local_cov":
            raise ValueError("model.density当前只允许name=local_cov。")
        self.root = Path(dataset_config.root)

    def __len__(self):
        """返回冻结清单的记录数；失败的上一步记录仍保留在长度中。"""
        return len(self.records)

    def __getitem__(self, index):
        """装配一条端到端记录；上一步失败时以明确错误进入采样失败记录。"""
        record = self.records[index]
        pdb_id = record["pdb_id"]
        sample_id = int(record["candidate_id"])
        if record.get("input_error") is not None:
            raise ValueError(record["input_error"])
        graph = read_smiles_graph(self.config.smiles_root, record["prepared_smiles"])
        if self.config.pocket_mode == "center":
            # float32, (3,)，Matcher中心或第一次C的Top-1预测质心，直接作为模型原点。
            given_center = np.asarray(record["given_center_xyz_A"], dtype=np.float32)
            # float32, (N,3)，仅满足原任务变换的形状契约，不表示任何真值或预测构象。
            ligand_coords = np.repeat(given_center.reshape(1, 3), len(graph["element"]), axis=0)
        elif self.config.pocket_mode == "envelope":
            # float32, (N,3)，第二次C的Top-1预测重原子世界坐标，用于严格<10 Å选袋。
            ligand_coords = np.asarray(record["envelope_coords_xyz_A"], dtype=np.float32)
            given_center = ligand_coords.mean(axis=0)
        else:
            raise ValueError(f"未知pocket_mode: {self.config.pocket_mode}")
        return assemble_docking_condition(
            dataset_config=self.config,
            root=self.root,
            transforms=self.transforms,
            receptor_branch=self.receptor_branch,
            density_config=self.density_config,
            pdb_id=pdb_id,
            sample_id=sample_id,
            prepared_smiles=record["prepared_smiles"],
            graph=graph,
            ligand_coords=ligand_coords,
            given_center=given_center,
        )
