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
from torch.utils.data import IterableDataset, get_worker_info
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
        - density_config: Mapping|None, 调用方从model.density传入的密度配置; None兼容原无密度入口, 不另设独立数据科学开关.
        - density_supervision: bool, 仅训练与val/loss入口设为True, 允许D3读取分割标签; 正式采样保留False, 无论清单属于哪个划分都不读取标签.
        - dataset_config.language_root: str, 仅D3读取的冻结SMI-TED逐实例NPZ根目录, 文件为<pdb_id>/candidate_<candidate_id>.npz.

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
        - density_input: float32, (1, 56, 48, 48, 48), 仅密度模型提供, 一个实例实际裁块的固定通道, PyG沿首维拼成批量B.
        - density_origin: float32, (1, 3), 实际裁块边界角点减pocket_center, 模型局部XYZ坐标, 单位Å.
        - density_basis: float32, (1, 3, 3), 三行分别是源XYZ方向单体素在模型坐标中的向量, 初始为实际间距的对角阵, 单位Å.
        - density_start_zyx: int64, (1, 3), 实际源裁块起点, 如[[10,12,8]], 内缩后仍不改变pocket_center.
        - density_language: float32, (1, 768), 仅D3提供的冻结SMI-TED向量, 与精确prepared_smiles匹配.
        - density_target: int64, (1, 48, 48, 48), 仅D3训练与val/loss提供; 标签1为ligand_area中的配体体素, 0为背景, 与实际裁块的ZYX索引对齐.

    原字段 num_atoms、bond_index、bond_type 和 matches_iso 仍按公共SMILES原子顺序解释; 配体类别、fixed prompt、空刚体域和噪声叶由原变换生成.
    """

    def __init__(self, dataset_config, split, transforms, receptor_branch, protocol, shuffle, density_config=None, density_supervision=False):
        """保存明确配置并读取唯一的冻结实例清单; 不枚举目录补回被排除的实例."""
        super().__init__()
        self.config = dataset_config
        self.split = split
        self.transforms = transforms
        self.receptor_branch = receptor_branch
        self.protocol = protocol
        self.shuffle = shuffle
        self.density_config = density_config
        self.density_supervision = density_supervision
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
            raise UnsupportedSmilesError(f"{pdb_id}/{candidate_id}: {record['unsupported_smiles_reason']}")
        graph = read_smiles_graph(self.config.smiles_root, record["prepared_smiles"])
        ligand_coords = read_smiles_coords(self.config.smiles_coords_root, pdb_id, candidate_id, record["prepared_smiles"])

        # (3,), occurrence 的沉积几何中心, 仅用于已批准的定位条件构造.
        ligand_center = ligand_coords.mean(axis=0)
        if self.protocol == "C5":
            # float32, (3,), 评测清单中已冻结的世界XYZ偏移, 单位Å; 训练与val/loss入口固定C0或E.
            offset = np.asarray(record["center_offset_xyz_A"], dtype=np.float32)
        else:
            offset = np.zeros(3, dtype=np.float32)
        # float32, (3,), 实际给定中心的世界XYZ坐标, 同时决定中心口袋选择和模型原点.
        given_center = (ligand_center + offset).astype(np.float32)

        receptor = read_receptor(self.root / "parse" / pdb_id / "receptor_tokens.npz")
        # bool, (P_full,), P_full为完整标准受体重原子数; 依据残基重原子质量中心选择整个残基, True保留该原子.
        selected = select_pocket(receptor, ligand_coords, given_center, self.config.pocket_mode)
        protein_count = int(np.sum(selected & (receptor["res_type"] < 20)))
        nucleic_count = int(np.sum(selected & (receptor["res_type"] >= 20)))
        if self.receptor_branch == "protein":
            selected &= receptor["res_type"] < 20
        # 逐原子数组沿相同掩码切分; P 是当前模型实际输入的口袋重原子数.
        pocket = {key: value[selected] for key, value in receptor.items()}
        # 空E没有可定义的受体均值; 不用配体中心替代, 也不改变标准残基和10 Å选择规则.
        if self.config.pocket_mode == "envelope" and self.receptor_branch == "RA" and len(pocket["coords"]) == 0:
            raise EmptyEnvelopePocketError(f"empty_envelope_pocket: {pdb_id}/{candidate_id}")
        pocket_pos = torch.from_numpy(pocket["coords"])
        pocket_is_nucleic = torch.from_numpy(pocket["res_type"] >= 20)
        # (M,) 和 (2, M), 公共 SMILES 图已映射为原模型键类别, 端点索引相同顺序的原子.
        bond_types, bond_index = graph["bond_type"], graph["bond_index"]
        matches_iso = graph["matches_iso"]
        data = PocketMolData(
            data_id=f"{pdb_id}_{candidate_id}", pdbid=pdb_id, task="dock", db="adaligand",
            candidate_id=candidate_id, prepared_smiles=record["prepared_smiles"],
            element=torch.from_numpy(graph["element"]),
            pos_all_confs=torch.from_numpy(ligand_coords[None]), i_conf_list=[0], num_confs=1,
            num_atoms=len(graph["element"]), num_bonds=len(bond_types),
            bond_index=torch.from_numpy(np.concatenate([bond_index, bond_index[::-1]], axis=1)),
            bond_type=torch.from_numpy(np.concatenate([bond_types, bond_types])),
            matches_iso=matches_iso, pocket_pos=pocket_pos,
            pocket_protein_count=protein_count, pocket_nucleic_count=nucleic_count,
        )
        if self.receptor_branch == "protein":
            # 官方只读蛋白, 并保留原 FeaturizePocket 对空蛋白返回空 center 的真实行为.
            data.pocket_element = torch.from_numpy(pocket["element"].astype(np.int64))
            data.pocket_atom_to_aa_type = torch.from_numpy(PROTEIN_CLASS[pocket["res_type"]])
            data.pocket_is_backbone = torch.from_numpy(pocket["is_backbone"])
            pocket_config = EasyDict(knn=self.config.knn)
            if self.config.pocket_mode == "center":
                pocket_config.center = given_center.tolist()
            data = FeaturizePocket(pocket_config)(data)
        else:
            # (1, 3), 原 PocketXMol 定位规则的世界原点; 包络包含实际选入的蛋白和核酸原子.
            center = given_center[None] if self.config.pocket_mode == "center" else pocket["coords"].mean(axis=0, keepdims=True)
            data.pocket_center = torch.from_numpy(center.astype(np.float32))
            data.pocket_pos = pocket_pos - data.pocket_center
            data.pocket_is_nucleic = pocket_is_nucleic
            # (P, 25) 与 (P, 15), 两类输入投影在同一口袋原子顺序对齐; 另一类型位置保留0.
            protein_features = torch.zeros((len(pocket_pos), 25))
            nucleic_features = torch.zeros((len(pocket_pos), 15))
            protein = ~pocket_is_nucleic
            protein_elements = torch.from_numpy(pocket["element"][~pocket_is_nucleic.numpy()].astype(np.int64))
            protein_features[protein, :4] = (protein_elements[:, None] == torch.tensor([6, 7, 8, 16])).float()
            protein_features[protein, 4:24] = torch.nn.functional.one_hot(torch.from_numpy(PROTEIN_CLASS[pocket["res_type"][protein.numpy()]]), num_classes=20).float()
            protein_features[protein, 24] = torch.from_numpy(pocket["is_backbone"][protein.numpy()].astype(np.float32))
            nucleic_elements = torch.from_numpy(pocket["element"][pocket_is_nucleic.numpy()].astype(np.int64))
            nucleic_features[pocket_is_nucleic, :4] = (nucleic_elements[:, None] == torch.tensor([6, 7, 8, 15])).float()
            nucleic_features[pocket_is_nucleic, 4:12] = torch.nn.functional.one_hot(torch.from_numpy(pocket["res_type"][pocket_is_nucleic.numpy()].astype(np.int64) - 20), num_classes=8).float()
            # list[str], 长度P_na, 当前口袋标准核酸原子名; P_na为核酸原子数, 如C1*规范为C1'.
            names = [name.decode("ascii").strip().replace("*", "'") for name in pocket["atom_name"][pocket_is_nucleic.numpy()]]
            # int64, (P_na,), 组分编号0/1/2分别是碱基、糖、磷酸, 顺序与names一致.
            components = torch.tensor([1 if name in SUGAR_ATOMS else 2 if name in PHOSPHATE_ATOMS else 0 for name in names], dtype=torch.long)
            nucleic_features[pocket_is_nucleic, 12:15] = torch.nn.functional.one_hot(components, num_classes=3).float()
            data.pocket_atom_feature = protein_features
            data.pocket_nucleic_feature = nucleic_features
            # int64, (2, E_p), 蛋白与核酸共同组成kNN图; E_p为有向边数, 端点索引pocket_pos的原子维.
            if len(pocket_pos) > 1:
                data.pocket_knn_edge_index = knn_graph(data.pocket_pos, k=min(self.config.knn, len(pocket_pos) - 1), flow="target_to_source")
            else:
                data.pocket_knn_edge_index = torch.empty((2, 0), dtype=torch.long)
        if self.density_config is not None:
            # (3,), 中心模式只按实际给定中心裁图; E按配体中心请求, 但原点仍为已选受体均值.
            query_center = given_center if self.config.pocket_mode == "center" else ligand_center
            # 密度在原特征化和噪声之前构造, 局部几何使用与配体和受体一致的唯一原点.
            for name, value in load_density_input(self.root, pdb_id, query_center, data.pocket_center.numpy()).items():
                data[name] = value
            if self.density_config['mode'] == 'D3':
                # 冻结语言表示只由当前精确SMILES确定; 不读取旧ligand_object或在训练热路径重新运行语言模型.
                language_path = Path(self.config.language_root) / pdb_id / f'candidate_{candidate_id}.npz'
                with np.load(language_path, allow_pickle=False) as language:
                    if str(language['prepared_smiles'].item()) != record['prepared_smiles']:
                        raise ValueError(f'language_smiles_mismatch: {pdb_id}/{candidate_id}')
                    data.density_language = torch.from_numpy(language['embedding'].astype(np.float32)[None])
                if self.density_supervision:
                    # int32, (K,3), 源地图的K个配体区域ZYX体素索引; 同一内缩起点用于密度和标签, 不改变模型原点.
                    area = np.load(self.derived_root / 'ligand_area' / pdb_id / f'{candidate_id}.npy', allow_pickle=False)
                    # int64, (K,3) -> (K_inside,3), 减起点后只保留三轴均在[0,48)的索引, 指向target的ZYX三维.
                    local_indices = area.astype(np.int64) - data.density_start_zyx.numpy()[0]
                    local_indices = local_indices[np.all((local_indices >= 0) & (local_indices < 48), axis=1)]
                    target = np.zeros((48, 48, 48), dtype=np.int64)
                    target[tuple(local_indices.T)] = 1
                    # 标签仅供辅助损失, 不参与密度token选择、口袋定位或采样; PyG沿首维拼接为(B,48,48,48).
                    data.density_target = torch.from_numpy(target[None])
        return self.transforms(data)

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
