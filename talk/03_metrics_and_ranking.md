# 指标、候选生成与排序

构象生成和 docking 都会一次生成多个三维结构，但“多个”的作用不同：构象任务要覆盖真实构象集合，docking 则要从候选 pose 中选出一个提交结果。本文集中澄清这些评分协议。

## 1. RMSD 不是一个统一操作

对两个具有对应原子的坐标集合 $X,Y\in\mathbb R^{N\times3}$，基础 RMSD 为：

$$
\operatorname{RMSD}(X,Y)
=\sqrt{\frac1N\sum_{i=1}^{N}\lVert x_i-y_i\rVert_2^2}.
$$

实践中还必须回答三个问题：

1. 是否允许刚体对齐？
2. 对称等价原子如何匹配？
3. 比较的是一个参考结构还是一个参考集合？

### 构象生成

`evaluate_conf.py::get_best_rmsd` 使用 RDKit `GetBestRMS`。它会寻找对称等价原子映射，并允许刚体对齐，所以只比较分子内部构象，不惩罚任意全局朝向和位置。

### docking

`evaluate_dock.py::get_rmsd` 使用 RDKit `CalcRMS`，寻找对称匹配但不移动预测配体。因为 benchmark 的蛋白口袋已经对齐，配体在口袋中的绝对平移和旋转正是 docking 误差的一部分，不能再把配体单独对齐到真值。

这一区别是阅读评测代码时最重要的边界之一。

## 2. 构象生成：Coverage 与 Matching

设某个分子的参考构象集合为 $R=\{r_i\}_{i=1}^{n_r}$，生成集合为 $G=\{g_j\}_{j=1}^{n_g}$，对每个参考构象定义最近生成距离：

$$
d_i=\min_j \operatorname{RMSD}(r_i,g_j).
$$

### Coverage，COV

给定阈值 $\delta$：

$$
\operatorname{COV}(R,G;\delta)
=\frac1{n_r}\sum_i \mathbf 1[d_i\le\delta].
$$

它问：参考集合中有多大比例至少被一个生成构象覆盖。高 COV 需要多样性和准确性同时存在。

当前 GEOM-Drug 代码使用 `threshold=1.25` Å。

### Matching，MAT

$$
\operatorname{MAT}(R,G)
=\frac1{n_r}\sum_i d_i.
$$

它问：每个参考构象到最近生成构象的平均距离。MAT 越低越好。

### 聚合方式

`get_rmsd_min` 为每个分子返回一对 COV/MAT；`print_results` 再报告跨分子的 mean 和 median。因此论文表中的总体值不是把所有参考构象混成一个大池后计算。

### 候选预算

论文采用 Uni-Mol 测试集的 200 个分子，并要求每个方法为每个分子生成“参考构象数的两倍”。代码也硬检查：

```text
len(gen_mols) == 2 * len(ref_mols)
```

当前 `conf_geom/base.yml` 的 `num_repeats: 2` 与这一协议配合。论文说明含 Si 的一个分子超出模型元素词表，被忽略。

## 3. docking：先定义候选，再定义选择器

PoseBusters v1 benchmark 含 428 个蛋白—小分子复合物。任务允许模型内部产生多个 pose，但最终每个口袋只提交一个。

若第 $q$ 个口袋有候选 $G_q=\{g_{qj}\}$，排序器给分 $s(g_{qj})$，则 top-1 为：

$$
g_q^*=\arg\max_{g\in G_q}s(g).
$$

主准确率是：

$$
\frac1Q\sum_q \mathbf 1[
\operatorname{RMSD}(g_q^*,r_q)<2\text{ Å}
].
$$

论文报告 self-ranking、tuned ranking 和 oracle ranking：

- self-ranking：用基础模型自身的轨迹置信度及轻量有效性项排序；
- tuned ranking：用独立 tuned confidence checkpoint 重评估 pose，再加相同有效性项；
- oracle：事后选真实 RMSD 最小的候选，只表示当前候选池的上限，不能作为可部署结果。

## 4. Self-confidence 从哪里来

### 4.1 单步坐标置信度

`ConfidenceLoss` 的坐标目标是

$$
y_i=\exp(\log 0.2\cdot e_i)=0.2^{e_i},
$$

其中 $e_i=\lVert x_i-\widehat x_i\rVert_2$ 以 Å 计。网络对 `sigmoid(confidence_pos_i)` 做 MSE 拟合。

所以 `confidence_pos` 的原始输出是 logit，且监督在逐原子误差上，不是整 pose 的 RMSD 标签。

### 4.2 轨迹聚合

`get_cfd_traj` 执行：

1. 对所有原子求平均，得到每一步的分数；
2. 取 100 步轨迹的后半段；
3. 再对这些步求平均。

结果写入 `gen_info.csv::cfd_traj`。这里代码没有再次 sigmoid，因此排序使用的是平均 logit 尺度。

### 4.3 当前 self-ranking 公式

`scripts/rank_pose.py` 使用：

```text
self_ranking = cfd_traj + int(no_clashes) + int(stereo)
```

其中：

- `no_clashes` 来自配体—蛋白原子间距离检查；
- `stereo` 检查生成配体与参考配体的立体化学身份是否一致。

这不是纯模型置信度。两个布尔项各加 `1`，其数值尺度会直接影响排序，因此结果应称为“self-confidence 加轻量结构过滤的 ranking score”。

## 5. Tuned confidence 是第二个模型

`scripts/believe.py` 重新读取已生成 pose，通过 `OverwritePos` 把 pose 坐标放回输入，再使用 `tuned_ranker.ckpt`：

- 仍按 `dock` 任务处理；
- 只运行 1 个步骤；
- 初始噪声尺度为 `0.01`；
- 对输出 `confidence_pos` 做原子平均，保存为 `tuned_cfd.csv`。

若该文件存在，排序公式为：

```text
tuned_ranking = tuned_cfd + int(no_clashes) + int(stereo)
```

因此 tuned ranking 不是在基础模型输出上拟合一条简单校准曲线，而是用另一个 checkpoint 对每个候选执行一次独立前向评分。现有 `tuned_cfd.yml` 将 `level.min` 与 `level.max` 都设为 1；依照代码，prior 不会先扰动候选坐标。

## 6. Top-k 与 oracle 的正确读法

`evaluate_dock.py` 提供两种相关工具：

- `get_topk_metrics`：先按 confidence 保留前 `k` 个，再在其中按真实 RMSD 排序，用于分析“候选给到后续 oracle/人工挑选”的上限；
- `get_rank_metrics`：报告排序前 `1,2,...,k` 个候选中最小 RMSD。

这些分析会使用真实 RMSD 计算“前 k 个里的最好结果”，不等于实际系统自动返回了那个结果。部署指标仍应以排序器选出的 rank-1 为准。

论文报告 oracle 成功率明显高于 self/tuned rank-1，说明候选生成已经包含更多正确 pose，排序仍是主要瓶颈之一。

## 7. PoseBusters 与 RMSD 衡量不同问题

RMSD 只问预测 pose 是否接近实验 pose；PoseBusters 还检查拓扑、键长/角度、内部能量、碰撞、配体—受体距离等物理和化学合理性。

论文同时报告：

- `RMSD < 2 Å` 的比例；
- 同时满足 `RMSD < 2 Å` 与 PB-valid 的比例。

仓库中的步骤是分开的：

1. `evaluate_dock.py` 计算每个候选与 rank-1 的 RMSD。
2. `evaluate_dock.py` 写出 `rank1_rmsd.csv`；外部步骤需把要评测的已选候选整理成 `evaluate_by_buster.py` 硬编码读取的 `rank1_rmsd_bel.csv`。
3. `evaluate_by_buster.py` 对该表中的已选 pose 运行 `PoseBusters(config='redock')`。
4. 后续分析再合并两类结果。

`scripts/rank_pose.py` 的 `no_clashes` 只是一个轻量子检查，不等价于完整 PB-valid。

当前仓库未检索到 `rank1_rmsd_bel.csv` 的生产代码，所以第 2 步是明确的人工/外部交接点，而不是已经由排序脚本自动完成的重命名。

## 8. 论文协议与仓库默认值的差异

| 项目 | 论文 | 当前仓库默认配置 |
|---|---|---|
| docking benchmark | PoseBusters v1，428 个复合物 | `dock_poseboff.lmdb`，命名与 PoseBusters prepared/offline 数据对应 |
| 每口袋候选数 | 100 poses | `dock_poseboff/base.yml` 为 `num_repeats: 50` |
| 去噪步数 | 100 | 100 |
| 默认噪声 | 逐原子 Gaussian | `free: 1`，逐原子 Gaussian |
| 最终提交数 | 每口袋 1 | 排序脚本每 `data_id` 选 1 |
| 能量最小化 | 不使用 | 基础 RMSD 路径不做最小化 |

当前仓库没有在 `base.yml` 中直接解释如何由 50 repeats 组成论文的 100 poses。可能存在两批采样或论文运行配置与公开基础配置不同，但代码证据不足以确定。复现论文数字前必须先确认这一点，不能仅把 `evaluate_dock.py --use_repeats 100` 当成已经生成了 100 个候选。

## 9. 常见误读

- “COV 高”不表示每个生成构象都准确；它只要求每个参考构象附近至少有一个候选。
- “MAT 低”也不保证生成集合多样；多个重复候选仍可能让部分参考构象完全未覆盖。
- docking 的 ligand-only 对齐 RMSD 会错误消除 pose 的全局错位，不能替代当前 `CalcRMS` 协议。
- oracle 不是排序性能，而是候选池上限。
- `cfd_traj`、最后一步 `cfd_pos`、tuned confidence 和 `self_ranking` 是四个不同量。
- `no_clashes=True` 不等于 PB-valid。
- 候选数越多通常越有利于 oracle，也可能有利于 rank-1；比较模型时必须固定采样预算。

## 10. 结果解读模板

阅读或汇报一次实验时，至少同时写清：

```text
任务与数据集：
每个样本的候选数：
去噪步数与噪声设置：
排序分数：self / tuned / oracle / 其他
RMSD 是否允许对齐和对称匹配：
阈值：
是否另做 PoseBusters 或能量最小化：
最终聚合：per-molecule mean/median 或全局比例
```

缺少其中任何一项，两个看似同名的 RMSD/COV/成功率都可能不可直接比较。
