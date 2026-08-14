# 任务档案：分子构象生成

## 1. 一句话定义

给定一个小分子的二维化学图，即原子类别与化学键，生成一个或多个与该图一致的三维原子坐标构象；不提供蛋白口袋条件。

这里的“生成”不是改变分子结构式。原子、键和原子编号保持不变，只寻找该分子可能采用的三维几何形状。

## 2. 形式化定义

令二维分子图为 $G=(A,B)$：

- $A=(a_1,\ldots,a_N)$ 是原子类别；
- $B=(b_{ij})$ 是原子对的键类别；
- $X\in\mathbb R^{N\times3}$ 是待生成坐标。

目标是学习条件分布：

$$
p_\theta(X\mid A,B),
$$

并从中采样构象集合：

$$
G_\theta(A,B)=\{\widehat X^{(1)},\ldots,\widehat X^{(K)}\}.
$$

整体平移和适当旋转不改变分子内部构象，因此构象指标应先消除这两类坐标系自由度。镜像反射则不能一概视为等价：对固定手性的分子，反射可能把一个立体异构体变成其对映体，RDKit `GetBestRMS` 的常规刚体对齐也不把这种反射当作适当旋转。不同可旋转键、环构象和非键相互作用会使同一二维图对应多个合理三维低能构象，所以输出本质上是多模态集合。

## 3. 输入、输出、推理、评分

### 输入

- 必需：原子类别 `A`、二维键图 `B`、原子数和固定原子顺序。
- 训练时还需：一个或多个真实三维构象 `X`。
- 不需要：蛋白口袋、结合位点、配体在全局坐标系中的位置。

### 输出

- 一个候选对应同一 `A,B` 的坐标矩阵 `X_hat [N,3]`。
- 一次 benchmark 为每个二维图输出多个候选构象。
- 仓库把每个候选写为一个 SDF，同时在 `gen_info.csv` 保留 `data_id` 与 repeat 编号。

### 推理目标

候选集合既要接近参考低能构象，又要覆盖同一分子的不同构象模态。只返回一个“平均结构”通常无法满足这个目标。

### 评分

论文沿 Uni-Mol 协议使用 Coverage 和 Matching：

- COV：参考构象中，距离某个生成构象不超过阈值的比例，越高越好；
- MAT：每个参考构象到最近生成构象的 RMSD 平均值，越低越好。

GEOM-Drug 代码阈值为 `1.25 Å`。RMSD 使用 RDKit `GetBestRMS`，允许刚体对齐并处理对称等价原子。

## 4. 论文 benchmark

- 测试集来自 Uni-Mol 使用的构象测试集，含 200 个具有多个参考构象的分子。
- 每个方法为每个分子生成参考构象数的两倍。
- 一个分子含 Si，超出 PocketXMol 当前元素词表，论文评测中忽略。
- 指标先按分子计算 COV/MAT，再汇总 mean 和 median。

当前仓库的相应文件：

- 采样 assembly：`data/test/assemblies/conf_geom.csv`；
- 参考集合：`data/test/conf/rdkit_cluster_data/drugs/test_data_200.pkl`；
- `data_id → smiles` 映射：`data/test/dfs/conf_geom.csv`；
- 采样配置：`configs/sample/test/conf_geom/base.yml`；
- 评测：`evaluate/evaluate_conf.py`。

## 5. PocketXMol 如何表示这个任务

分子记作 $M=(A,X,B)$。该任务的 prompt 是：

| 变量 | 是否固定 | 代码字段 |
|---|---:|---|
| 原子类别 `A` | 是 | `fixed_node = 1` |
| 原子坐标 `X` | 否 | `fixed_pos = 0` |
| 键类别 `B` | 是 | `fixed_halfedge = 1` |
| 原子对距离 | 否，默认 free | `fixed_halfdist = 0` |
| 肽组成 | 否 | `is_peptide = 0` |

因此模型解决的是“只恢复坐标”的条件去噪，而不是重新生成原子或键。

论文说明构象预测和 docking 使用相同 prompt 与噪声，区别仅是构象预测没有口袋约束。代码通过空口袋张量而不是另一套模型实现这一点。

## 6. 数据怎样变成模型输入

### 6.1 取样

`ForeverTaskDataset` 用 `task='conf'` 和 `db='geom'` 解析 assembly，读取：

```text
mols/{data_id};torsion/{data_id};decom/{data_id}
```

训练配置中的 `conf` 还混合 `geom`、`qm9`、`unmi` 和 `cremp` 数据库；任务抽样权重为 `0.12`。这些是联合 checkpoint 的训练来源，不等于 Uni-Mol benchmark 的测试来源。

### 6.2 Featurize

`FeaturizePocket` 先补出空口袋。`FeaturizeMol`：

1. 把原子序数变成类别索引；
2. 从 `pos_all_confs` 随机选一个构象；
3. 减去该构象的几何中心；
4. 建立所有 `i<j` 原子对及其键/无键类别。

居中使模型不需要学习任意全局平移。

### 6.3 Task transform

`ConfTransform` 的当前采样配置只选 `free`。它保存 `gt_*` 参考，设置原子和键固定、坐标自由，并准备可能被几何校正使用的扭转注释。

训练配置不是绝对纯 free：`free: 0.999`、`torsional: 0.001`。绝大多数构象训练样本采用逐原子坐标噪声，极少样本采用扭转噪声。

## 7. 噪声与 100 步生成

### 7.1 训练噪声

`ConfSampleNoiser` 从训练配置得到：

```text
逐原子 Gaussian: sigma_max = 1
torsional:        sigma_max = 0.3 × π
训练 level:       Uniform[0,1)
```

默认 free 分支只使用逐原子 Gaussian。`node_type` 和 `halfedge_type` 原样进入模型。

加噪前后，noiser 会再次按 `node_type_batch` 把每个分子居中。训练时若启用 `reassign_in`，还会从 `matches_iso` 中选择与目标更接近的对称原子排列，减少等价编号造成的监督冲突。

### 7.2 采样

`conf_geom/base.yml` 使用：

- `num_steps: 100`；
- `prior: from_train`；
- `advance` 信息水平调度；
- `num_repeats: 2`；
- `batch_size: 4000`。

第一步从训练 prior 产生近乎完全带噪坐标，之后每一步对当前预测重新加入更弱噪声并预测干净坐标。自由模式的 `outputs2batch` 直接把 `pred_pos` 写为下一状态。

两次 repeat 与 benchmark 的“两倍参考构象数”协议配合；`evaluate_conf.py` 仍会对每个分子的实际候选数做严格检查。

## 8. 模型怎样使用无口袋条件

`PMAsymDenoiser` 仍执行口袋编码器，但输入为空，因此产生空的 `h_pocket`。`ContextNodeEdgeNet` 检测到上下文张量非 `None`，不过动态 kNN 在空口袋场景是否可由当前 PyG 版本安全处理，需要运行环境验证；设计意图是以空上下文退化为纯分子完整图去噪。

分子内部每个块：

1. 用当前坐标重算所有原子对距离；
2. 更新原子和原子对隐藏特征；
3. 从相对方向和标量权重产生坐标位移。

最终 `pred_pos` 是直接恢复的干净坐标。类别与键预测仍存在，但采样器保持原二维图不变。

## 9. 损失中与本任务有关的部分

默认 free 样本的 fixed mask 导致：

- `conf/pos`：所有原子的坐标 MSE，是主要任务监督；
- `conf/fixed_node`：固定原子类别的 CE；
- `conf/fixed_edge`：固定完整半边类别的 CE；
- `conf/dist`：free 样本没有刚体域，`inner_domain` 为空，因此该项为零；torsional 小分支才产生域内距离监督；
- `conf/dih`：free 样本没有二面角注释，返回零；torsional 小分支才有意义；
- confidence：逐原子坐标误差、原子类别正确性和键类别正确性的辅助监督。

这些分任务值用于日志；反向传播仍由包含所有任务的 `mixed/total` 驱动。

## 10. 输出与评测链

```text
sample_drug3d.py
  → SDF/{candidate}.sdf + gen_info.csv
  → evaluate_conf.load_gen 按 data_id 映射到非手性 smiles
  → 与 test_data_200.pkl 中同 smiles 的参考构象集合配对
  → 对每个参考构象计算最近生成 RMSD
  → 每个分子得到 COV/MAT
  → 汇总 mean/median
```

评测加载生成 SDF 时若文件不存在会跳过；但随后候选数检查会阻止静默使用不完整候选集合。

## 11. 这项任务最容易混淆的概念

- 构象生成不是 de novo 3D molecule generation：前者固定二维图，后者连原子和键都要生成。
- 构象生成不是 docking：它不关心分子在蛋白口袋中的绝对 pose。
- 构象 RMSD 必须允许刚体对齐；否则任意居中和旋转都会被误算为误差。
- COV 衡量集合覆盖，MAT 衡量参考到最近候选的平均距离；两者不能互相替代。
- `num_repeats: 2` 不是每个分子只生成两个构象的普遍定义，而是与 assembly 中参考条目和“两倍参考数”协议共同决定实际候选数。

## 12. 建议阅读代码的检查问题

1. `FeaturizeMol` 在哪里消除了全局平移？
2. `fixed_node/fixed_pos/fixed_halfedge` 如何把任务变成“只生成坐标”？
3. 为什么 `ConfSampleNoiser` 在加噪后还要再次居中？
4. `matches_iso` 怎样缓解对称原子编号问题？
5. 空口袋字段怎样保持模型接口一致？
6. 为什么模型仍预测固定的原子和键，损失又怎样降低其权重？
7. `GetBestRMS` 与 docking 使用的 `CalcRMS` 有何不同？
