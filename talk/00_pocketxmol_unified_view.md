# PocketXMol 统一建模总览

本文回答一个总问题：PocketXMol 如何把不同的三维分子生成任务写成同一个学习问题，以及“多任务联合训练”在这套方法里究竟和哪些设计共同起作用。

本文只在需要举例时使用分子构象生成和小分子 docking。两项任务各自的数据、推理和评分协议见对应任务档案。

## 1. 统一问题

论文把一个待生成分子记为

$$
M=(A, X, B),
$$

其中：

- $A=(a_1,\ldots,a_N)$ 是 `N` 个原子的类别；当前训练配置支持 C、N、O、F、P、S、Cl、B、Br、I、Se，另有 MASK 类别。
- $X\in\mathbb{R}^{N\times 3}$ 是原子笛卡尔坐标；代码和评测均按 Å 理解距离。
- $B$ 是所有原子对的键类别，包括无键、单键、双键、三键和芳香键；代码只存 $i<j$ 的完整半边集合，形状为 `[2, N(N-1)/2]`，消息传递前再复制成双向边。

蛋白口袋记为 $K$。它也是原子集合，但只作为固定条件：每个口袋原子具有元素、氨基酸类型、是否主链和三维坐标等特征。PocketXMol 并不把氨基酸残基作为模型的基本生成单位。

任务由 prompt $P$ 和任务噪声 $\xi$ 共同定义。论文的加噪和去噪写作：

$$
\widetilde M=\Phi(M;P,\xi,\beta),
\qquad
(\widehat M,\widehat S)=F_\theta(\widetilde M;P,K),
$$

其中 $\beta\in[0,1]$ 控制加噪程度，$\widehat S$ 是原子类别、坐标和键类别的置信度。一个很关键的代码事实是：`PMAsymDenoiser.forward` 没有接收 $\xi$、$\beta$ 或时间步；它只看到带噪分子、prompt 和可选口袋，因此必须从输入本身判断噪声并恢复干净结构。

## 2. Task prompt：什么已知，什么要生成

prompt 不是自然语言，而是一组二值指示量。`1` 表示该变量已知并应保持，`0` 表示需要生成。

| 论文记号 | 代码字段 | 粒度 | 含义 |
|---|---|---|---|
| $P^{atom}$ | `fixed_node` | `[N]` | 原子类别是否固定 |
| $P^{coor}$ | `fixed_pos` | `[N]` | 原子坐标是否固定 |
| $P^{bond}$ | `fixed_halfedge` | `[H]` | 原子对的键类别是否固定，`H=N(N-1)/2` |
| $P^{dist}$ | `fixed_halfdist` | `[H]` | 原子对距离是否固定，用于刚体或扭转约束 |
| $P^{pep}$ | `is_peptide` | `[N]` | 是否要求生成标准氨基酸组成；本轮两个任务均为 `0` |

前三类是主 prompt；距离和肽指示是辅助 prompt。模型把前四者直接拼进节点或边特征：

```text
节点输入 = 原子类别嵌入 + [fixed_node, fixed_pos] + 可选 is_peptide
边输入   = 键类别嵌入   + [fixed_halfedge, fixed_halfdist]
```

prompt 同时约束三个环节：

1. `ConfTransform` 等任务变换构造 fixed mask。
2. `BaseSampleNoiser.__call__` 加噪后强制把 fixed 部分恢复为原值。
3. 模型把 fixed mask 当作条件特征，学习如何处理不同任务，而不是只依赖采样器的硬覆盖。

因此，多任务并非仅靠一个 `task` 标签区分。`task` 主要用于选择 transform/noiser 和统计分任务损失；真正送入去噪网络的任务描述是原子级 prompt。

## 3. Task noise：怎样破坏已知结构

论文组合四类基本噪声：

- 原子坐标或整体平移上的高斯噪声；
- 原子和键类别上的 categorical 噪声；
- 分子刚体旋转上的 SO(3) 各向同性高斯噪声；
- 可旋转键二面角上的圆周噪声。

不同任务只启用所需成分。当前两项任务的原子和键类别固定，所以 `ConfSampleNoiser` 以 `pos_only=True` 构造 `MolPrior`，不会扰动类别。

训练时，`MolInfoLevel` 为样本随机产生信息水平；采样时，100 个反向步骤通过 `advance` 调度把信息水平从近乎全噪声推进到近乎干净。代码中的 `level` 越接近 `1` 表示保留的信息越多，与直觉中的“噪声强度”方向相反。

## 4. 通用去噪网络

当前 `train_pxm_reduced.yml` 实际选择 `PMAsymDenoiser + ContextNodeEdgeNet`。

### 4.1 口袋编码器

口袋原子特征先由线性层映射到 128 维，再在固定的 32-NN 口袋图上经过 4 个 `node_only` 块。口袋坐标和图在这些块中不更新，只得到上下文节点表示。

构象生成没有口袋。`FeaturizePocket` 会建立形状为 `[0, D]`、`[0, 3]` 和 `[2, 0]` 的空张量；同一模型仍可执行，只是没有分子—口袋消息。

### 4.2 分子编码与去噪

分子采用完整图。每个 `ContextNodeEdgeNet` 块依次：

1. 根据当前坐标重算分子内距离和径向基特征。
2. 若有口袋，则根据当前分子坐标动态重建分子—口袋 32-NN 图。
3. 更新原子隐藏特征。
4. 更新原子对隐藏特征。
5. 用标量权重乘相对方向向量，累加得到坐标位移；口袋条件还提供第二个坐标位移。

训练配置使用 6 个这样的分子块。坐标更新由相对向量构造，因此网络映射在整体平移、旋转或反射下保持 E(3) 等变；原子和边的隐藏表示只依赖距离等不变量。这里的“反射等变”是网络变换性质，不表示构象评测会把固定手性分子的镜像当成同一构象。

### 4.3 解码与置信度

模型分别输出：

- `pred_node [N, K_a]`：原子类别 logits；
- `pred_pos [N, 3]`：恢复后的坐标；
- `pred_halfedge [H, K_b]`：键类别 logits；
- `confidence_node [N, 1]`、`confidence_pos [N, 1]`、`confidence_halfedge [H, 1]`。

半边解码前，两个方向的隐藏特征相加，使最终键预测重新对应无向原子对。

## 5. 联合训练究竟联合了什么

`ForeverTaskDataset` 的训练迭代器先按 `task_db_weights[*].weight` 抽任务，再按该任务的 `db_ratio` 抽数据库。当前精简训练配置给 `conf` 和 `dock` 的任务权重分别为 `0.12` 和 `0.25`；这不是损失权重，而是数据采样概率。

样本依次经过：

```text
FeaturizePocket → FeaturizeMol → MixedTransform[task] → MixedSampleNoiser[task]
```

同一个 checkpoint 因而共同学习：

- 多类分子和蛋白口袋的数据分布；
- 由 prompt 表达的已知/未知变量组合；
- 不同几何噪声和类别噪声下的恢复；
- 有口袋与无口袋两种条件模式。

`IndividualTasksLoss` 会把批中元素按 `batch['task']` 映射回 `conf`、`dock` 等任务，既记录分任务指标，也把所有样本汇总为 `mixed/total` 反向传播。当前总损失的主要权重为：

$$
1.5L_{atom}+2.5L_{pos}+1.5L_{bond}
+0.0005L_{dist}+0.0005L_{dih}+L_{cfd},
$$

另对 prompt 标为 fixed 的变量设置较小的恢复损失。对本轮两项任务而言，坐标均为 unfixed，而原子和键类别均为 fixed。free 样本虽把距离标为 unfixed，但没有刚体域，代码的 `inner_domain` 筛选为空，因此距离和二面角项均为零；它们只在结构化几何 setting 中生效。

## 6. 生成不是标准反向扩散链

`sample_loop3` 对 100 个递减步骤重复：

```text
当前分子 M(t-1)
    → 按当前信息水平重新加噪
    → 通用去噪器直接预测干净分子
    → M-Projector/outputs2batch 写回当前状态
    → 下一步
```

论文强调每次“扰动—去噪”可独立工作，而不是固定的 Markov 反向转移。代码中所谓 `M-Projector` 主要由各 noiser 的 `outputs2batch` 实现：自由构象和默认 docking 直接采用 `pred_pos`，fixed 坐标则覆盖回输入值；柔性或刚体设置还可执行几何校正。

本轮两项任务不是 autoregressive 生成。`is_ar` 虽由 transform 名称传给 `sample_loop3`，只有名称以 `ar` 开头时才进入额外轮次。

## 7. PocketXMol 为什么可能表现好

不能把全部效果笼统归因于“多任务训练”。更准确的拆分如下。

### 7.1 原子级统一表示

小分子和蛋白/肽都被降到原子、坐标和原子对，避免为不同化学实体维护互不兼容的高级 token。论文据此主张不同分子类型之间可以迁移原子相互作用知识。

### 7.2 原子级 task prompt

任务是 fixed mask 的组合，而不是独立输出头。相同网络能表达“只生成坐标”“生成类别与坐标”“固定局部片段”等连续变化，也能在采样中加入局部先验。

### 7.3 多类型、多任务联合数据

任务采样器和共享 checkpoint 确实把多数据库、多任务样本放进同一优化过程。这提供了共享监督，但仅凭当前论文和仓库不能把某项指标提升定量归因到联合训练；需要对应消融实验才能建立因果结论。

### 7.4 多种几何与类别噪声

同一去噪器面对逐原子高斯、整体平移/旋转、扭转和 categorical 扰动。其直接收益是任务覆盖面和采样时先验注入的灵活性。对于默认 docking，论文反而报告逐原子高斯噪声优于柔性噪声，说明“噪声更贴近物理自由度”并不自动意味着效果更好。

### 7.5 E(3) 等变网络

距离产生标量特征，坐标位移由标量权重乘相对向量得到。这使预测不依赖任意全局坐标系，并允许分子内部与分子—口袋相互作用共同推动坐标更新。

### 7.6 置信度、候选预算与后处理

模型不仅生成结构，还学习变量级正确度/坐标误差的代理量。实际 docking 先生成多个 pose，再通过轨迹置信度、碰撞和立体化学检查排序；论文结果也显示增加候选数可提高选中成功 pose 的机会。因此应把“生成器质量”和“多候选选择质量”分开评价。

仓库还提供 tuned confidence checkpoint，对已有 pose 做一次独立前向重评估。当前 confidence YAML 虽设 `init_step: 0.01`，但同时设 `level.min = level.max = 1`；按 `MolInfoLevel` 的实际读取逻辑，输入 pose 不受 prior 扰动。它是独立的排序增强，不应与基础生成模型的 self-confidence 混为一谈。

### 7.7 Autoregressive refinement

仓库对部分变长或分片任务提供 autoregressive refinement，但分子构象生成和默认小分子 docking 都不经过该分支。它属于 PocketXMol 的全局能力，不能拿来解释本轮两个 benchmark 的结果。

## 8. 证据边界与限制

- 论文报告 PocketXMol 在 13 项任务中的广泛表现，但这不等于每项任务都从所有设计获得相同收益。
- 当前口袋是刚性的、预先给定的；模型不联合预测或诱导适配蛋白口袋。
- 分子完整图导致原子对数量为 $O(N^2)$，限制大体系扩展。
- 元素词表有限；构象 benchmark 中含 Si 的一个分子因此被忽略。
- 代码中的置信度是训练目标的代理分数，不是经过概率校准的“该 pose 正确概率”。排序有效性必须由相关性和 top-1 指标单独验证。
- 本文描述当前仓库和论文方法，不声称重新跑出了论文数值。

## 9. 本轮最短阅读路线

1. 先读[输入、批字段与输出产物](01_input_output_overview.md)，建立张量词汇表。
2. 再读[模块与调用结构](02_module_structure_overview.md)，掌握真实入口。
3. 分别读[构象生成档案](tasks/molecular_conformation_generation.md)和[小分子 docking 档案](tasks/small_molecule_docking.md)。
4. 最后读[指标、候选与排序](03_metrics_and_ranking.md)，避免把生成、排序和结构有效性混成一个指标。
