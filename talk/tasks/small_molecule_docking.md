# 任务档案：小分子 docking

## 1. 一句话定义

给定刚性蛋白结合口袋和一个二维结构已知、内部可变形的小分子，预测该小分子在口袋坐标系中的三维结合 pose。

本档案只讨论小分子 docking，不展开线性肽或环肽 docking。

## 2. 形式化定义

令：

- 配体二维图为 $G_L=(A,B)$；
- 配体坐标为 $X_L\in\mathbb R^{N\times3}$；
- 固定口袋为 $K=(F_K,X_K)$，其中 `F_K` 是口袋原子特征，`X_K` 是与实验复合物一致的坐标。

目标是学习：

$$
p_\theta(X_L\mid A,B,K),
$$

并生成多个候选：

$$
\{\widehat X_L^{(1)},\ldots,\widehat X_L^{(K)}\}.
$$

这里的 $X_L$ 同时包含：

- 配体相对口袋的整体平移；
- 配体相对口袋的整体旋转；
- 配体内部可旋转键等构象自由度。

蛋白口袋在本 benchmark 中是刚性的，模型不会更新 `pocket_pos`。

## 3. 输入、输出、推理、评分

### 输入

- 配体原子类别和二维键图；
- 刚性蛋白口袋的原子类别、残基类别、主链指示和三维坐标；
- 配体与口袋必须处于同一目标坐标系的定义中，但推理初始配体坐标会被噪声破坏。

### 输出

- 每个候选是固定配体图对应的 `X_hat [N,3]`；
- 输出坐标位于原蛋白结构坐标系，可直接写入 SDF 与蛋白共同检查；
- 模型同时输出逐原子坐标置信度，供候选排序。

### 推理协议

先为每个口袋生成多个 pose，再由 self-confidence 或 tuned confidence 排序，每个口袋只选 rank-1 参加正式比较。

### 评分

- pose RMSD：在已对齐蛋白口袋下比较预测与实验配体，不再单独移动配体；
- 成功率：rank-1 pose 的 RMSD `< 2 Å` 的复合物比例；
- PB-valid：PoseBusters 的结构有效性检查；论文还报告同时满足 RMSD `< 2 Å` 和 PB-valid 的比例。

## 4. 论文 benchmark

- 数据集为 PoseBusters v1，含 428 个蛋白—小分子复合物。
- 配体柔性，蛋白口袋刚性。
- baseline 按各自默认设置提供一个最终 pose。
- PocketXMol 为每个口袋生成 100 个 pose，再做 self-ranking 或 tuned ranking。
- 评测不做能量最小化。
- oracle ranking 取候选池中真实 RMSD 最小的 pose，只用于显示候选池上限。

当前仓库对应：

- 采样配置：`configs/sample/test/dock_poseboff/base.yml`；
- 测试 assembly：`data/test/assemblies/lmdb/dock_poseboff.lmdb`；
- 参考配体：`data/poseboff/files/mols/{data_id}_mol.sdf`；
- 蛋白：`data/poseboff/files/proteins/{data_id}_pro.pdb`；
- RMSD：`evaluate/evaluate_dock.py`；
- 排序：`scripts/rank_pose.py`；
- PoseBusters：`evaluate/evaluate_by_buster.py`。

## 5. PocketXMol 如何表示这个任务

prompt 与构象生成相同：

| 变量 | 是否固定 | 代码字段 |
|---|---:|---|
| 配体原子类别 | 是 | `fixed_node = 1` |
| 配体坐标 | 否 | `fixed_pos = 0` |
| 配体键类别 | 是 | `fixed_halfedge = 1` |
| 配体内部距离 | 默认 free 下否 | `fixed_halfdist = 0` |
| 肽组成 | 否 | `is_peptide = 0` |

区别在条件 `K`：docking 提供非空口袋，构象生成提供空口袋。模型无需显式的“dock head”。

## 6. 数据怎样变成共同坐标系

### 6.1 口袋

`FeaturizePocket`：

1. 把每个口袋原子编码为 25 维特征；
2. 建立口袋内部 32-NN 图；
3. 计算 `pocket_center = mean(pocket_pos)`；
4. 将口袋坐标减去该中心。

### 6.2 配体

随后 `FeaturizeMol` 选择配体参考构象，并减去同一个 `pocket_center`。因此口袋与配体的相对位置保持不变，只是整体移到数值更稳定的局部坐标系。

输出时 `decode_output` 把 `pocket_center` 加回预测配体坐标，使 SDF 回到原蛋白坐标系。

### 6.3 完整配体图

配体的稀疏共价键被展开为所有 `i<j` 原子对。键类型固定，但无键原子对也具有隐藏边状态；这允许网络显式建模长程配体内部相互作用和当前几何距离。

## 7. 任务 transform 与训练数据

`ConfTransform` 同时注册为 `conf` 和 `dock`。当前采样配置固定选择 `free`，因此每个配体原子可独立移动。

训练配置为：

```text
dock task weight = 0.25
setting: free 0.999, flexible 0.001
database ratios: csd, pbdock, moad, apep, pepbdb
```

这里需要区分任务范围与 checkpoint 来源：本档案只评估小分子 docking，但统一模型的 `dock` 训练混合中确实列有 `apep` 和 `pepbdb`。这说明 checkpoint 的知识来源包含其他分子类型；本轮不进一步展开其肽任务协议。

## 8. 两种 docking 噪声

论文定义两种可选噪声。

### 8.1 Free Gaussian

对每个配体原子坐标独立加高斯噪声。当前默认 benchmark 配置使用这一模式：

```text
task_setting = free
prior.pos = gaussian_simple
sigma_func = sqrt
sigma_max = 1.0
```

`sigma_func: sqrt` 使每个原子的噪声标准差随本分子原子数的平方根增长，并至少截到 `1`。第一步从纯噪声位置开始，后续逐步减弱扰动。

### 8.2 Flexible noise

将三类更结构化的运动组合：

- 整体平移；
- SO(3) 整体旋转；
- 可旋转键扭转。

这种模式通过 `domain_node_index` 和扭转注释保持更多配体内部几何，并在 `outputs2batch` 中用 `correct_pos_batch` 将网络预测投影回相应自由度。

论文明确报告 Gaussian noise 的 docking 效果更好，因而默认使用 free Gaussian。当前 `base_flex.yml` 可作为对照路径，但不属于默认小分子 benchmark。

## 9. 口袋条件怎样进入几何网络

### 9.1 固定口袋表示

口袋先在固定 32-NN 图上经过 4 个节点更新块。因为 `node_only=True`，该编码器不更新口袋坐标或边状态。

### 9.2 动态分子—口袋图

分子去噪的 6 个块中，每一块都根据“当前配体坐标”和固定口袋坐标重新建立 32-NN 上下文边。随着 pose 变化，参与消息传递的邻近口袋原子也会变化。

### 9.3 两条口袋作用路径

口袋条件同时影响：

- `ContextNodeBlock`：口袋节点和距离边特征向配体原子传递上下文消息；
- `ctx_pos_blocks`：配体—口袋相对向量产生额外坐标位移。

因此 pocket 不是最后拼到一个打分头上，而是在每个几何更新块中持续改变配体表示和坐标。

## 10. 100 步采样与候选数

当前 `dock_poseboff/base.yml` 使用：

- `num_steps: 100`；
- `advance` 信息水平调度；
- `batch_size: 400`；
- `num_repeats: 50`；
- 保存逐原子/逐边 confidence 输出。

每步流程：

```text
当前 pose
  → 逐原子重新加噪
  → 口袋条件去噪
  → 直接采用 pred_pos
  → 下一步
```

论文协议写明每口袋 100 poses，而当前基础 YAML 只写 50 repeats。仓库未在该配置旁说明另一半候选如何产生；严格复现论文结果前需要核对实验命令、是否合并两次运行或原论文配置版本。本档案不把 50 自动解释为 100。

## 11. 损失中与本任务有关的部分

默认 free docking 样本的损失结构与构象任务近似，但 `pred_pos` 受到非空口袋条件影响：

- `dock/pos`：所有配体原子的坐标 MSE；
- `dock/fixed_node`、`dock/fixed_edge`：较小权重的固定二维图恢复；
- `dock/dist`：free 样本没有刚体域，`inner_domain` 为空，因此该项为零；极少 flexible 样本才有域内距离监督；
- `dock/dih`：free 样本为空，极少 flexible 样本才启用；
- confidence：逐原子坐标误差和类别正确性的辅助预测。

损失没有直接使用 Vina 能量、PoseBusters、整 pose RMSD 或结合活性标签。模型学习的是从带噪实验结构恢复坐标；benchmark 排名质量来自该恢复学习产生的 confidence 代理。

## 12. 从输出 pose 到 rank-1

### 12.1 基础输出

每个候选写入：

- `SDF/{filename}.sdf`；
- `gen_info.csv` 中的 `cfd_traj`、`cfd_pos`、`cfd_node`、`cfd_edge`；
- 可选 `SDF/{filename}.pt` 中的逐原子/逐边置信度。

### 12.2 辅助检查

`utils/docking_aux_scores.py` 检查：

- `no_clashes`：配体与蛋白是否存在过近原子对；
- `stereo`：生成配体的立体化学是否与参考图一致。

### 12.3 排序

基础 self-ranking：

```text
cfd_traj + int(no_clashes) + int(stereo)
```

tuned ranking 则用另一个 checkpoint 对固定候选做一次独立前向重评估，把 `tuned_cfd` 替换 `cfd_traj`，再加相同两个布尔项。当前 YAML 把信息等级固定为 1，所以这一步不会先用 prior 扰动 pose。

`evaluate_dock.py` 按每个 `data_id` 取得排序最高的 pose，并同时保留 oracle 最小 RMSD 作为候选池上限。

## 13. RMSD 与 PoseBusters 评测链

```text
gen_info.csv + SDF
  → evaluate_dock.py：逐候选 CalcRMS
  → ranking.csv：合并 self/tuned score
  → 每个 data_id 选 rank-1
  → 外部整理为 evaluate_by_buster.py 期望的 rank1_rmsd_bel.csv
  → 统计 RMSD < 2 Å
  → evaluate_by_buster.py：对选中 pose 做 redock 配置的 PB 检查
  → 合并为 RMSD 成功率和 PB-valid 成功率
```

`CalcRMS` 不移动预测配体，保留相对口袋的 pose 误差；对称匹配失败时，代码退回原子索引一一对应。

这里存在一个公开代码交接缺口：`evaluate_dock.py` 自动写出的是 `rank1_rmsd.csv`，`evaluate_by_buster.py` 却硬编码读取 `rank1_rmsd_bel.csv`；当前仓库没有生成后一文件名的可执行入口。因此不能把上图理解为完全自动连通的流水线，运行完整 PB-valid 前必须由外部步骤明确筛选列/候选并准备该 CSV。

## 14. 这项任务最容易混淆的概念

- docking 不是 SBDD：docking 固定配体二维图，只生成 pose；SBDD 还要生成原子和键。
- “柔性配体”不等于默认一定使用 flexible noise；论文默认反而是逐原子 Gaussian。
- 口袋刚性不表示口袋不参与网络；它作为每层动态邻域的固定几何条件。
- self-ranking 不是单纯最后一步 confidence，而是轨迹 confidence 加碰撞/立体检查。
- tuned ranking 使用独立 checkpoint，不能作为基础模型无后处理性能。
- oracle 反映候选池，不反映可用排序器。
- `RMSD < 2 Å` 与 PB-valid 是不同条件；轻量 `no_clashes` 也不等于完整 PB-valid。

## 15. 建议阅读代码的检查问题

1. `pocket_center` 如何保证输入、模型和输出坐标系一致？
2. 为什么配体使用完整图，而口袋使用 kNN 图？
3. 每个 GNN block 为什么要重建分子—口袋 kNN？
4. `free` 和 `flexible` 在 level、prior 与 `outputs2batch` 上分别走哪些分支？
5. fixed 原子/键为什么仍有预测头和恢复损失？
6. `cfd_traj` 是如何从 `[N,100]` 变成一个 pose 分数的？
7. self-ranking 的两个布尔加分会怎样改变分数尺度？
8. 当前 50 repeats 与论文 100 poses 的复现缺口需要什么实验记录才能闭合？
