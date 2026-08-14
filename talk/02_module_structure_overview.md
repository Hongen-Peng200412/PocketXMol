# 模块与调用结构概览

本文从正式入口出发说明两项任务实际调用哪些模块、每个模块接收什么、交出什么，以及推荐的源码阅读顺序。

## 1. 推理主链

```text
采样 YAML + checkpoint 中的训练 YAML
                  │
                  ▼
        scripts/sample_drug3d.py
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
DataModule 构造词表     TestTaskDataset 取原始记录
和 pocket/mol featurizer    │
        └─────────┬─────────┘
                  ▼
FeaturizePocket → FeaturizeMol → ConfTransform
                  │
                  ▼
             PyG DataLoader
                  │
                  ▼
         models/sample.py::sample_loop3
                  │  每步
       ┌──────────┴──────────┐
       ▼                     ▼
Conf/DockSampleNoiser   PMAsymDenoiser
 加噪与 M-Projector      口袋编码 + 几何去噪
       └──────────┬──────────┘
                  ▼
seperate_outputs2 → decode_output → RDKit 重建
                  │
                  ▼
             SDF + gen_info.csv
```

构象生成和 docking 使用相同入口与模型。唯一结构性差别是 docking 的 `pocket_*` 张量非空，构象生成的口袋张量为空。

仓库还提供单复合物示例入口 `scripts/sample_use.py`。它不读取 `TestTaskDataset`，而是从蛋白 PDB 与小分子 SDF/SMILES 直接建立一条 `PocketMolData`，随后复用同一套 featurizer、`ConfTransform`、noiser、`sample_loop3`、候选拆分与重建链；`configs/sample/examples/dock_smallmol*.yml` 走的就是这条入口。

## 2. 配置如何选择实现

### 2.1 采样配置

| 决策 | 构象生成 | docking |
|---|---|---|
| `task.name` | `conf` | `dock` |
| `task.transform.name` | `conf` | `dock` |
| transform 实现 | `ConfTransform` | 同一个 `ConfTransform` |
| `noise.name` | `conf` | `dock` |
| noiser 实现 | `ConfSampleNoiser` | `DockSamplNoiser`，仅把 `task_name` 改为 `dock` |
| setting | `free: 1` | `free: 1, flexible: 0` |
| 口袋 | 空 | 非空、刚性条件 |

采样 YAML 的 `prior: from_train` 表示噪声分布参数从 checkpoint 对应的训练配置里取，而不是使用字符串本身构造 prior。

### 2.2 训练配置

训练使用两个注册分发器：

- `MixedTransform` 根据每条样本的 `data['task']` 选择任务变换；
- `MixedSampleNoiser` 根据批/样本的 `task` 选择任务 noiser。

`ForeverTaskDataset` 负责在取数据前决定 task 和 database。模型本身不读取 `batch['task']`；它通过 prompt、输入噪声和口袋条件理解任务。

## 3. 数据层

### 3.1 `scripts/train_pl.py::DataModule`

这个类既服务训练，也被采样脚本借来恢复 checkpoint 的词表和 featurizer。

- `get_featurizers`：始终把口袋 featurizer 放在分子 featurizer 前，因为分子坐标需要减去已经算出的 `pocket_center`。
- `get_in_dims`：把原子/键类别数和口袋特征维度交给模型构造函数。
- `setup`：仅训练/验证时构造完整 `Compose`、`ForeverTaskDataset` 和 DataLoader。

采样脚本不会调用 `DataModule.setup`，而是自己组装测试 transform 和 `TestTaskDataset`。

### 3.2 `utils/dataset.py`

- `ForeverTaskDataset.setup`：打开 assembly 和各内容 LMDB。
- `ForeverTaskDataset.__iter__`：训练时按配置权重无限或整轮地产生 `(task, db, index)`。
- `ForeverTaskDataset.__getitem__`：解析 `data_id`、组合 LMDB key、合并记录、写入任务元数据并执行 transform。
- `TestTaskDataset`：给单任务测试包装有限长度和整数索引。
- `SingleDatabase`：把以分号连接的多个 LMDB 记录合并成一个 `Data`。

### 3.3 `utils/data.py`

`Mol3DData.__inc__`/`PocketMolData.__inc__` 是批处理索引契约。阅读任何 `*_anno` 字段前，应先确认它在拼批时按什么维度偏移，否则很容易把“样本内节点编号”和“批内全局编号”混淆。

### 3.4 `process/utils_process.py`

`scripts/sample_use.py` 的文件输入并不会直接产生模型张量。`extract_pocket` 先在受体世界坐标系中按参考配体裁剪残基；`get_pocmol_data` 再把小分子 conformer、双向键和口袋原子合并成 `PocketMolData`；`get_input_from_file` 最后补入扭转、自同构和 BRICS/MMPA 分解叶。中心化发生在后续 `FeaturizePocket/FeaturizeMol`，不发生在这一解析层。

### 3.5 `process/process_torsional_info.py`

`get_torsional_info_mol` 是柔性构象与 docking 注释的生产端：它从固定二维图生成 `bond_rotatable`、`tor_twisted_pairs`、`fixed_dist_torsion`、`path_mat`、`nbh_dict`、`matches_graph` 和 `matches_iso`。`get_mol_from_data` 则按 `db/data_id` 从 SDF 或 UNMI 事务恢复与这些索引严格对齐的原始 RDKit 分子。

## 4. 变换层

### 4.1 `FeaturizePocket`

职责是把原始口袋变成：

- 25 维原子特征；
- 固定 32-NN 图；
- 以口袋均值为原点的坐标。

无口袋也返回相同字段的空张量，使下游无需为构象任务另写模型接口。

### 4.2 `FeaturizeMol`

职责是：

- 原子序数编码；
- 从可用构象中抽一个监督帧；
- 与口袋共用局部坐标系，或在无口袋时把分子居中；
- 从稀疏共价键构造完整半边图；
- 在输出阶段执行反向解码。

### 4.3 `ConfTransform`

`conf` 和 `dock` 共用该类。它先抽 `task_setting`，再产生 fixed mask、刚体/扭转注释和测试参考副本。

本轮默认 `free` 设置的核心结果是：类别和键固定，坐标与距离自由。`flexible`、`torsional`、`rigid` 只应作为理解训练混合和几何校正的旁支阅读。

## 5. 噪声与信息水平

### 5.1 `utils/info_level.py`

`MolInfoLevel` 统一训练随机水平和采样调度：

- 训练时 `step=None`，`uniform` 从 `[0,1)` 随机抽样；
- 采样时传入归一化 step，`advance` 将其映射为信息保留水平；
- `step=1` 被硬设为 `level=0`，保证第一次可从先验开始。

### 5.2 `utils/prior.py`

`MolPrior` 是原子、坐标和边 prior 的组合器；构象/docking 以 `pos_only=True` 只创建坐标 prior。

`AllPosPrior` 根据 `level_dict` 的 key 分流：

- `pos`：逐原子高斯；
- `trans`：整域平移；
- `rot`：整域 SO(3) 旋转；
- `tor`：可旋转键扭转。

这些 key 互斥或按柔性模式组合，不是同时无条件施加。

`utils/motion.py` 是这些结构化 prior 与几何校正器共享的低层运动模块：四元数转换负责 SO(3) 旋转，轴角函数负责刚体域旋转，扭转函数依据“旋转键—随动原子”索引只更新指定原子行。其角度均按弧度解释，坐标保持 Å 尺度。

### 5.3 `utils/sample_noise.py`

`BaseSampleNoiser.__call__` 定义通用模板：取当前状态、采样 level、加噪、执行可选预处理、恢复 fixed 变量，再写入 `*_in`。

`ConfSampleNoiser` 增加：

- 按 `task_setting` 选择逐原子或刚体/扭转 level；
- 构象任务在加噪前后按分子重新居中，消除无意义的全局平移；
- 可根据分子对称匹配重排带噪坐标；
- `outputs2batch` 在自由模式直接写回 `pred_pos`，其他模式可调用几何校正。

`DockSamplNoiser` 没有重写上述逻辑，只让 `from_train` 找到训练配置里的 `dock` prior，并避免执行 `task == 'conf'` 的居中分支。

## 6. 模型层

### 6.1 `models/maskfill.py::PMAsymDenoiser`

这是本轮真正使用的顶层模型。其职责边界：

1. 嵌入带噪原子/边类别并拼入 prompt。
2. 编码固定口袋。
3. 把半边复制成双向完整图。
4. 调用上下文几何 GNN 更新原子、边和坐标。
5. 解码类别与置信度。

文件虽然导入 `models.graph.NodeEdgeNet`、GVP 和 IPA 版本，但当前 `name: default` 且未开启 `gvp`，实际只走 `models.graph_context.ContextNodeEdgeNet`。

### 6.2 `models/graph_context.py`

当前可达组件为：

| 组件 | 输入 → 输出 | 作用 |
|---|---|---|
| `ContextNodeBlock` | 节点、分子边、可选上下文边 → 节点 | 聚合分子内与口袋消息，并做残差/LayerNorm |
| `BondFFN` | 边、端点节点、prompt → 消息 | 以 prompt 门控边—节点交互 |
| `EdgeBlock` | 边、两端节点 → 边 | 聚合邻接原子对并更新边隐藏状态 |
| `PosUpdate` | 节点、边、相对向量、距离 → `[N,3]` 位移 | 用标量权重组合相对方向，保证几何等变 |
| `ContextNodeEdgeNet` | 分子图 + 可选口袋 → 更新后的节点、坐标、边 | 逐块重建距离/上下文图并串联上述更新 |

### 6.3 `models/common.py`

当前直接依赖的低层模块主要是：

- `MLP`：线性层、LayerNorm 和激活的通用堆叠；
- `GaussianSmearing`：把一个标量距离展开成径向基特征。

这些模块是通用数学积木，注释应解释张量和数值变换，不绑定到某一个任务。

## 7. 训练与损失

### 7.1 `ModelLightning.training_step`

主训练步只做三件事：

```text
outputs = model(batch)
loss_dict = loss_func(batch, outputs)
return loss_dict['mixed/total']
```

其余逻辑主要是 OOM 时缩小批次、日志和优化器调度。

### 7.2 `IndividualTasksLoss`

损失先基于 fixed mask 分开可生成与固定变量，再根据 `batch['task']` 建立节点、半边和二面角级选择掩码。

对本轮默认 `free` 的 `conf`/`dock`：

- `pos` 是主要 unfixed 坐标 MSE；
- `node`、`edge` 落入较小权重的 fixed 恢复项；
- `fixed_halfdist` 虽为 `0`，但 free 训练样本没有刚体域，`inner_domain` 选择为空，因此 `dist` 为零；
- free 训练样本没有二面角注释，因此 `dih` 也为零；
- 所有任务共享 `ConfidenceLoss`。

损失字典同时含 `mixed/*` 和 `conf/*`、`dock/*` 等记录。真正反向传播的是 `mixed/total`；分任务项用于观察各任务行为。

### 7.3 `ConfidenceLoss`

监督目标由模型主预测即时构造并 `detach`：

- 原子/边：argmax 是否等于真值；
- 坐标：`exp(log(0.2) * error_Å)`，即误差越大目标越小。

坐标置信度输出经 sigmoid 后用 MSE 拟合该目标。它没有直接监督整 pose 的 RMSD；整分子排序分数是后续对逐原子输出聚合得到的代理量。

## 8. 采样、重建与评测

### 8.1 `models/sample.py`

- `sample_loop3`：执行 100 步扰动—去噪—投影，并收集轨迹置信度。
- `seperate_outputs2`：按图拆分最终状态、模型输出和轨迹。
- `get_cfd_traj`：先做原子平均，再对轨迹后半段平均。

### 8.2 `utils/reconstruct.py`

`reconstruct_from_generated_with_edges` 对 `conf`/`dock` 调用 `reconstruct_pos`：从原始数据取得固定拓扑的 RDKit 分子并只覆盖坐标。

`create_sdf_string` 是重建失败与轨迹中间态的保底序列化器：它直接把解码后的 `atom_pos/element/bond_index/bond_type` 写成 V2000 mol block，不执行 sanitize，也不添加 `$$$$` 记录分隔符。

### 8.3 评测入口

| 目的 | 入口 | 主要产物 |
|---|---|---|
| 构象 COV/MAT | `evaluate/evaluate_conf.py` | `metric.txt`, `df_metric.csv` |
| docking pose RMSD | `evaluate/evaluate_dock.py` | `rmsd.csv`, `rank1_rmsd.csv` |
| 碰撞/立体检查与排序 | `scripts/rank_pose.py` + `utils/docking_aux_scores.py` | `aux_scores.csv`, `ranking.csv` |
| tuned confidence | `scripts/believe.py` | `tuned_cfd.csv` |
| PoseBusters | `evaluate/evaluate_by_buster.py` | `buster*.csv` |

指标的数学含义和这些文件怎样组合，见[指标、候选与排序](03_metrics_and_ranking.md)。

`utils/buster_tools.py` 是排序阶段轻量碰撞与 InChI 身份/立体比较的具体实现；它不等同于 `evaluate_by_buster.py` 调用的完整 PoseBusters `redock` 检查集合。

## 9. 推荐源码阅读顺序

### 第一遍：端到端，不进数学细节

1. `configs/sample/test/conf_geom/base.yml`
2. `configs/sample/test/dock_poseboff/base.yml`
3. `scripts/sample_drug3d.py`；单复合物示例另读 `scripts/sample_use.py`
4. `utils/dataset.py::TestTaskDataset/ForeverTaskDataset.__getitem__`
5. `process/utils_process.py::extract_pocket/get_pocmol_data/get_input_from_file`
6. `process/process_torsional_info.py::get_torsional_info_mol/get_mol_from_data`
7. `utils/transforms.py::FeaturizePocket/FeaturizeMol/ConfTransform`
8. `utils/sample_noise.py::BaseSampleNoiser/ConfSampleNoiser`
9. `models/maskfill.py::PMAsymDenoiser`
10. `models/sample.py::sample_loop3/seperate_outputs2/get_cfd_traj`

### 第二遍：模型和训练

1. `models/graph_context.py::ContextNodeEdgeNet`
2. `ContextNodeBlock → EdgeBlock → PosUpdate`
3. `models/common.py::GaussianSmearing/MLP`
4. `utils/prior.py` 与 `utils/motion.py`
5. `models/loss.py::IndividualTasksLoss/ConfidenceLoss`
6. `scripts/train_pl.py::DataModule/ModelLightning.training_step`

### 第三遍：评测与排序

1. `evaluate/evaluate_conf.py`
2. `evaluate/evaluate_dock.py`
3. `scripts/rank_pose.py`
4. `utils/docking_aux_scores.py`
5. `utils/buster_tools.py`
6. `scripts/believe.py`
7. `evaluate/evaluate_by_buster.py`

每遍都沿字段和目标调用路径阅读，不把共享文件中的其他实现纳入本轮学习范围。
