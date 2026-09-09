# 输入、批字段与输出产物概览

本文说明原 PocketXMol 从 LMDB 到 SDF/CSV 的字段。当前六个无密度实验另由 `docking/` 接入 AdaLigand 完整 occurrence，下面先列出该入口与原接口的关系，后续章节保留原任务说明。

## 当前 occurrence 入口

`docking/dataset.py::OccurrenceDataset` 读取冻结清单和只读完整模板，构造相同的 `PocketMolData`。共同数据来源、筛选与派生文件见[数据说明](../docking/README.md)。原配体类别、完全图半边和 loss 保持不变；本项目 free 分支使用空扭转／刚体索引，保留训练用 `matches_iso`。

| 新增或改变含义的字段 | 形状 | 本项目实际含义 |
|---|---|---|
| `pocket_atom_feature` | `(P,25)` | 蛋白原特征；RA/RB中的核酸原子对应全零 |
| `pocket_nucleic_feature` | `(P,15)` | 核酸元素4维、核苷酸8维、碱基／糖基／磷酸基3维；蛋白原子对应全零 |
| `pocket_is_nucleic` | `(P,)`，bool | 标准RNA/DNA原子标记，与口袋坐标逐原子对齐 |
| `given_center_local` | 通常`(1,3)` | 本次给定中心的模型坐标，中心模式为零；供T1读取 |
| `pocket_center` | 通常`(1,3)` | 中心模式用给定世界中心，包络用实际选入口袋原子的世界均值 |

这里 `P` 是当前模型实际输入的口袋原子数。RA使用联合口袋图，RB仅在蛋白内部和核酸内部构图。官方空蛋白仍沿原入口尝试，不补假口袋或原点，结果按实际执行阶段记录。

采样在每个实例目录写 `poses.sdf`、`candidates.json`、成功候选的 `confidence.npz`，最后写 `result.json`。评价增加 `candidate_metrics.json` 和 `assessment.json`。完整字段见[采样模块](../docking/sampling.py)和[评价模块](../docking/evaluation.py)的函数说明；三个测试视图共用候选，完整验证只汇总ALL。

## 1. 记号与容器

下文使用：

- `N`：一个配体/小分子的原子数；
- `H=N(N-1)/2`：完整无向原子对数；
- `P`：口袋原子数；
- `B`：批中图数量；
- `K_a`、`K_b`：原子和原子对类别数。

单样本是 `Mol3DData` 或 `PocketMolData`，继承 PyG `Data`；批处理后成为 PyG `Batch`。字段名后缀 `_batch` 表示每个元素属于批中哪张图。

## 2. 原始数据库输入

### 2.1 两项任务共享的分子字段

| 字段 | 典型形状/类型 | 具体含义 | 首个主要消费者 |
|---|---|---|---|
| `element` | `[N]`, integer | 原子序数 | `FeaturizeMol` |
| `pos_all_confs` | `[C, N, 3]`, float | 同一分子可用的 `C` 个三维构象，坐标单位 Å | `FeaturizeMol` |
| `i_conf_list` | 长度 `C` | 构象在原数据中的标识 | `FeaturizeMol` |
| `num_atoms` | scalar | 原子数 | `FeaturizeMol` |
| `bond_index` | `[2, 2E]`, integer | 双向存储的共价键端点 | `FeaturizeMol` |
| `bond_type` | `[2E]`, integer | 与 `bond_index` 对齐的键类别 | `FeaturizeMol` |
| `num_bonds` | scalar | 无向共价键数 `E` | `FeaturizeMol` |
| `data_id` | string | benchmark/数据库样本标识 | 采样与评测脚本 |

构象和柔性设置还需要旋转键、图路径和对称匹配等预处理字段，例如 `tor_bond_mat`、`tor_twisted_pairs`、`fixed_dist_torsion`、`path_mat`、`nbh_dict`、`matches_iso`。当前默认 `free` 推理不会用几何校正，但 `ConfTransform` 在测试模式仍会构造扭转注释，所以这些字段是否已经并入主 LMDB 记录取决于数据处理版本。

### 2.2 docking 独有的口袋字段

| 字段 | 典型形状/类型 | 具体含义 | 首个主要消费者 |
|---|---|---|---|
| `pocket_element` | `[P]`, integer | 口袋原子序数；当前 featurizer 接受 C、N、O、S | `FeaturizePocket` |
| `pocket_pos` | `[P, 3]`, float | 与参考配体同一坐标系中的口袋原子坐标，单位 Å | `FeaturizePocket` |
| `pocket_atom_to_aa_type` | `[P]`, integer | 每个口袋原子所属的 20 类氨基酸 | `FeaturizePocket` |
| `pocket_is_backbone` | `[P]`, bool/integer | 口袋原子是否属于蛋白主链 | `FeaturizePocket` |
| `pdbid` | string | 复合物/结构标识 | 元数据与输出 |

构象生成没有口袋字段。`FeaturizePocket` 会主动补齐空口袋，使同一模型接口仍然成立。

## 3. 数据库索引如何找到样本

`ForeverTaskDataset` 同时管理“目录 LMDB”和“内容 LMDB”：

1. `assembly_path` 给出第 `idx` 条样本对应的 `data_id`。
2. `(task, db, data_id)` 经 `get_data_key` 变成分号分隔的 LMDB key。
3. `SingleDatabase` 从一个或多个 LMDB 中取记录并依次 `update` 合并。
4. Dataset 写入 `task`、`db` 和 `key`，再运行 transform。

当前测试配置：

| 任务 | assembly | 数据库与内容 |
|---|---|---|
| 构象生成 | `data/test/assemblies/conf_geom.csv` | `geom` 的 `mols`、`torsion`、`decom` LMDB |
| docking | `data/test/assemblies/lmdb/dock_poseboff.lmdb` | `poseboff` 的 `pocmol10` LMDB |

这些是配置和评测脚本约定的下载后路径；当前源码 checkout 未包含对应 benchmark 数据文件。

`TestTaskDataset.__len__` 只取当前唯一 `(task, db)` 的大小；它不是无限数据流。训练则使用 `ForeverTaskDataset.__iter__` 按任务和数据库权重持续抽样。

## 4. Featurizer 产物

### 4.1 口袋

`FeaturizePocket` 产生：

| 字段 | 形状 | 含义 |
|---|---|---|
| `pocket_atom_feature` | `[P, 25]` | 4 类元素 one-hot + 20 类氨基酸 one-hot + 1 个主链指示 |
| `pocket_knn_edge_index` | `[2, E_p]` | 口袋内部 32-NN 有向图 |
| `pocket_center` | `[1, 3]` | 原始口袋坐标的均值；用于移到局部坐标系并在输出时移回 |
| `pocket_pos` | `[P, 3]` | `原坐标 - pocket_center` |

无口袋时对应形状分别为 `[0,25]`、`[2,0]`、`[0,3]` 和 `[0,3]`。

### 4.2 分子

`FeaturizeMol` 从 `pos_all_confs` 随机选一帧作为监督目标，然后产生：

| 字段 | 形状 | 含义 |
|---|---|---|
| `node_type` | `[N]` | 原子序数映射到模型词表后的类别索引 |
| `node_pos` | `[N, 3]` | 局部坐标；docking 减去 `pocket_center`，构象生成减去分子几何中心 |
| `halfedge_index` | `[2, H]` | 所有 `i<j` 原子对，按 `torch.triu_indices` 固定顺序排列 |
| `halfedge_type` | `[H]` | `0` 表示无键，`1..4` 表示单/双/三/芳香键，最后可留给 MASK |
| `is_peptide` | `[N]` | 本轮为全零 |
| `i_conf` | scalar/metadata | 本次选择的原构象标识 |

注意 `halfedge_type` 描述完整分子图上的每个原子对，不等于稀疏共价键列表。模型的边复杂度因此随 `N²` 增长。

## 5. 任务变换产物

`conf` 和 `dock` 都由 `ConfTransform` 处理。

### 5.1 默认 free 设置

| 字段 | 值/形状 | 对两个任务的意义 |
|---|---|---|
| `task_setting` | `'free'` | 每个原子坐标可独立变化 |
| `fixed_node` | `[N]`，全 `1` | 原子类别已知 |
| `fixed_pos` | `[N]`，全 `0` | 三维坐标待生成 |
| `fixed_halfedge` | `[H]`，全 `1` | 二维化学键已知 |
| `fixed_halfdist` | `[H]`，全 `0` | 不强制保持任意原子对距离 |
| `gt_node_type` | `[N]` | 采样前保存的类别参考 |
| `gt_node_pos` | `[N,3]` | 采样前保存的坐标参考，仅评估/轨迹需要 |
| `gt_halfedge_type` | `[H]` | 采样前保存的键参考 |

在训练的 `free` 分支中，刚体和扭转注释是空张量。测试模式还会建立一个覆盖整分子的 domain 和扭转注释，以兼容可能的采样后处理。

### 5.2 flexible/torsional/rigid 设置

这些设置不是本轮默认 benchmark 路径，但字段决定距离与二面角损失的含义：

- `domain_node_index [2, N_domain_nodes]`：第一行是 domain 编号，第二行是全局节点编号；
- `tor_bonds_anno [T, 3]`：扭转顺序与旋转键两端；
- `twisted_nodes_anno [Q, 2]`：每个扭转影响哪些节点；
- `dihedral_pairs_anno [D, 3]`：扭转编号及二面角两端节点；
- `fixed_halfdist`：哪些内部距离应在生成中保持。

## 6. PyG 批处理后的对齐

`DataLoader` 根据 featurizer 的 `follow_batch` 自动添加：

| 字段 | 形状 | 含义 |
|---|---|---|
| `node_type_batch` | `[ΣN]` | 每个分子原子属于哪张图 |
| `halfedge_type_batch` | `[ΣH]` | 每个半边属于哪张图 |
| `pocket_pos_batch` | `[ΣP]` | 每个口袋原子属于哪张图 |

`Mol3DData.__inc__` 决定拼批时哪些索引需要加偏移。例如 `halfedge_index` 按本图 `N` 偏移，`tor_bonds_anno` 的三列分别按扭转数和节点数偏移。这里的偏移协议是后续 GNN、损失和拆批能够对齐的基础。

`exclude_keys` 会在拼批时移除只用于单样本预处理、但不再被模型使用的大型字段，避免无意义的内存开销。

## 7. 加噪后的模型输入

`BaseSampleNoiser.__call__` 在每个训练样本或采样步骤创建：

| 字段 | 形状 | 含义 |
|---|---|---|
| `node_in` | `[ΣN]` | 带噪原子类别；本轮两项任务等于 `node_type` |
| `pos_in` | `[ΣN,3]` | 当前信息水平下的带噪坐标 |
| `halfedge_in` | `[ΣH]` | 带噪键类别；本轮等于 `halfedge_type` |

随后 noiser 再检查 fixed mask，任何 fixed 项一律覆盖回干净值。这一步是硬约束；模型输入中的 prompt 则是学习条件。

默认 `free` 噪声还会构造 `mol_size`：每个原子都携带本分子的 `N`。docking 训练配置的 `sigma_func: sqrt` 用它按 `sqrt(N)` 放大逐原子高斯噪声；构象配置没有该缩放函数。

## 8. 模型输出

`PMAsymDenoiser` 输出统一字典：

| 字段 | 形状 | 训练目标/采样用途 |
|---|---|---|
| `pred_node` | `[ΣN,K_a]` | 对 `node_type` 的类别 logits；采样取 `argmax` |
| `pred_pos` | `[ΣN,3]` | 对干净 `node_pos` 的直接坐标预测 |
| `pred_halfedge` | `[ΣH,K_b]` | 对 `halfedge_type` 的类别 logits |
| `confidence_node` | `[ΣN,1]` | 原子类别预测正确性的 logit |
| `confidence_pos` | `[ΣN,1]` | 与坐标误差相关的置信度输出 |
| `confidence_halfedge` | `[ΣH,1]` | 键类别预测正确性的 logit |

本轮两个任务虽然固定原子和键，模型仍计算这些预测和置信度；fixed mask 决定其损失与采样写回方式。

## 9. 采样状态、拆批与解码

`sample_loop3` 维护四类轨迹：

- `in`：每步重新加噪后的输入；
- `raw`：模型未经离散化的预测；
- `out`：`outputs2batch` 投影后的状态；
- `all`：参考、输入和输出交错组成的完整轨迹。

最终还把每一步的 `confidence_pos` 堆成 `confidence_pos_traj [ΣN,T]`。

`seperate_outputs2` 按 `_batch` 字段把大批次拆回单分子，并保留：

```text
node, pos, halfedge, halfedge_index, pocket_center
```

`FeaturizeMol.decode_output` 再执行：

1. 将局部坐标加回 `pocket_center`；构象生成的空 center 等价于不平移。
2. 类别索引还原为原子序数。
3. 丢弃仍为 MASK 的原子及其相关边。
4. 从完整半边中筛出真实共价键，并恢复为双向 `bond_index`。

`reconstruct_from_generated_with_edges` 会先检查任务。对构象和 docking，它调用 `reconstruct_pos` 重新读取/复制原始 RDKit 分子，只覆盖预测坐标；解码出的元素和键不参与拓扑重建。其他生成任务才从预测元素、坐标和键构造新分子并做价态/芳香性修正。

## 10. 落盘产物

`scripts/sample_drug3d.py` 为一次实验创建独立目录，主要包含：

| 路径/文件 | 内容 |
|---|---|
| `SDF/*.sdf` | 每个生成构象或 docking pose；文件名由全局保存序号生成 |
| `SDF/*.pt` | 配置 `sample.save_output` 指定的逐原子/逐边模型输出 |
| `SDF/*-{in,out,raw,all}.sdf` | 按 `save_traj_prob` 随机保存的轨迹 |
| `gen_info.csv` | `data_id`、`filename`、`i_repeat`、状态、SMILES 与置信度汇总 |
| `samples_all.pt` | 当前实现只保存与成功/失败计数等长的占位列表，用作完成标记 |
| 采样 YAML 与源码副本 | 复现实验配置和当时代码快照 |

`gen_info.csv` 中几个置信度字段不要混用：

- `cfd_pos/node/edge`：最后一次去噪输出在原子/边维度上的均值；
- `cfd_traj`：先对原子求均值，再对轨迹后半段求均值；默认 docking self-ranking 使用这一列。

## 11. 读代码时最值得跟踪的字段

第一次阅读建议只盯住下面这条链：

```text
element / pos_all_confs / bond_*
    → node_type / node_pos / halfedge_*
    → fixed_* + task_setting
    → node_in / pos_in / halfedge_in
    → pred_node / pred_pos / pred_halfedge
    → node / pos / halfedge
    → SDF + gen_info.csv
```

对于 docking，再并行跟踪：

```text
pocket_* raw fields
    → pocket_atom_feature / centered pocket_pos / pocket_knn_edge_index
    → h_pocket
    → dynamic molecule–pocket messages
```
