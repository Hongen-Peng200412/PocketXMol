# `local_cov` 中心模型实验日志

## 当前有效状态

2026-09-20 18:06状态快照：训练、best的C0/C5完整测试、CPU评价与在线记录均已完成。最终best为`step=24800.ckpt`、`val/loss=1.61287522315979`；C0、C5各446个实例、各22300个候选全部生成并评价成功。ALL逐实例等权的原self-ranking Top-1成功率为C0 73.77%、C5 64.13%。评价W&B `aqp4pl9u`为`online_completed`，378693保留`after_lock`并已返回`try_lock`等待。

| 项目 | 当前值 |
|---|---|
| 训练条件 | C0，RA＋T0 |
| 正式测试 | C0、C5；各实例各协议50候选、100步，推理batch_size=50 |
| 资源 | 378693／gnode10，单张A800 |
| 当前有效release | `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_ef9dd39/PocketXMol`，来源提交`ef9dd39` |
| 当前有效训练产物 | `/storage/penghongen/PocketXMol/training/local_cov-C-T0-RA-b72` |
| best与`val/loss` | 当前best为`step=24800.ckpt`，1.61287522315979 |
| W&B | 当前run [db4b74gn](https://wandb.ai/pencounkdual-111/PocketXmol_density/runs/db4b74gn)；首次提前停止run `w47t7b4j`保留 |
| 测试与评价产物 | `/storage/penghongen/PocketXMol/sampling/local_cov-C-T0-RA/test` |
| 评价W&B | [aqp4pl9u](https://wandb.ai/pencounkdual-111/PocketXmol_density/runs/aqp4pl9u)，`online_completed` |
| CPU评价 | 全部完成；ALL Top-1为C0 73.77%、C5 64.13% |

## 完整测试结果

以下成功率和平均RMSD均按实例等权；括号内为RMSD严格小于2 Å的成功实例数。ALL、CAP10、HF10_TO5分别包含446、272、227个实例，并复用同一候选池。

| 协议 | 视图 | 实例数 | Top-1成功率 | Top-5成功率 | Oracle成功率 | Top-1平均RMSD（Å） |
|---|---|---:|---:|---:|---:|---:|
| C0 | ALL | 446 | 73.77%（329） | 83.63%（373） | 93.05%（415） | 2.270 |
| C0 | CAP10 | 272 | 66.91%（182） | 78.31%（213） | 90.44%（246） | 2.641 |
| C0 | HF10_TO5 | 227 | 66.08%（150） | 78.85%（179） | 90.75%（206） | 2.735 |
| C5 | ALL | 446 | 64.13%（286） | 74.89%（334） | 84.30%（376） | 2.825 |
| C5 | CAP10 | 272 | 55.51%（151） | 68.75%（187） | 80.51%（219） | 3.298 |
| C5 | HF10_TO5 | 227 | 56.39%（128） | 70.48%（160） | 80.62%（183） | 3.289 |

PDB等权结果先在每个PDB内部平均，再对PDB平均；ALL、CAP10、HF10_TO5分别包含77、67、65个PDB。

| 协议 | 视图 | Top-1成功率 | Top-5成功率 | Oracle成功率 |
|---|---|---:|---:|---:|
| C0 | ALL | 76.64% | 85.59% | 94.81% |
| C0 | CAP10 | 76.32% | 84.76% | 94.43% |
| C0 | HF10_TO5 | 75.96% | 84.74% | 94.27% |
| C5 | ALL | 66.11% | 76.45% | 87.94% |
| C5 | CAP10 | 66.51% | 76.64% | 86.86% |
| C5 | HF10_TO5 | 66.83% | 77.16% | 85.95% |

置信度头仍与主体模型共同训练，并按原self-ranking规则使用。Spearman相关系数按实例计算候选置信度与RMSD的秩相关；姿态AUC只统计同时包含成功和失败候选的实例。

| 协议 | 视图 | Spearman均值 | Spearman中位数 | 姿态AUC均值 | AUC有效实例 |
|---|---|---:|---:|---:|---:|
| C0 | ALL | 0.3361 | 0.3740 | 0.7552 | 384／446 |
| C0 | CAP10 | 0.3295 | 0.3618 | 0.7488 | 234／272 |
| C0 | HF10_TO5 | 0.3305 | 0.3716 | 0.7447 | 197／227 |
| C5 | ALL | 0.3163 | 0.3327 | 0.7645 | 354／446 |
| C5 | CAP10 | 0.2856 | 0.3029 | 0.7557 | 212／272 |
| C5 | HF10_TO5 | 0.2869 | 0.2942 | 0.7595 | 178／227 |

同一T0模型在冻结C5定位条件下的ALL Top-1成功率比C0低9.64个百分点。该差值描述给定中心、固定口袋和局部原点同时改变后的整体表现；两种条件使用同一T0采样公式，不能将差异单独归因于某一个输入变化。

## 正式运行命令

```bash
bash 训练与运行/sh/train_docking.sh local_cov-C-T0-RA --logdir /storage/penghongen/PocketXMol/training/local_cov-C-T0-RA-b72
/storage/penghongen/PocketXMol/runtime/venv/bin/python scripts/sample_docking.py /storage/penghongen/tmp/pxm_local_cov_20260917_b72/runs/local_cov-C-T0-RA-test/sample-local_cov-C-T0-RA-test.yml
CUDA_VISIBLE_DEVICES='' /storage/penghongen/PocketXMol/runtime/venv/bin/python scripts/evaluate_docking.py /storage/penghongen/tmp/pxm_local_cov_20260917_b72/runs/local_cov-C-T0-RA-test/sample-local_cov-C-T0-RA-test.yml
```

以上命令由378693的动态命令在上述不可变release中顺序执行；训练标准输出和错误分别写入`/storage/penghongen/tmp/pxm_local_cov_20260917_b72/runs/local_cov-C-T0-RA/train.out`和`train.err`。测试配置、采样及评价输出统一保存到同级`local_cov-C-T0-RA-test`目录，采样成功后才执行CPU评价。

## 测试、门控与只读核查

共享验收见[本轮实验日志](本轮实验日志.md)。首次A800门控以36×2完成两次优化器更新。改为72×1后再次门控通过：两次更新耗时415.31秒，峰值显存为64,687,791,104字节已分配、67,448,602,624字节保留，`val/loss=2.9227933884`；C0、C5各生成并评价2个三步候选，完整检查点恢复逐值通过。

正式测试启动前确认训练输出含`Training finished!`和零退出标记，完整检查点的停止原因为`plateau`、下降计数为3；外置配置解析确认C0/C5、test划分、batch50、每实例每协议50候选、100步、8个CPU评价进程和在线W&B。动态命令通过`bash -n`，正式候选目录在启动前不存在。06:01消费378693的`try_lock`后，计算节点确认采样进程读取`step=24800.ckpt`并占用GPU；`after_lock`保持。

## 之前的尝试

- 首次正式运行使用release`PocketXMol_922fc651d3db`、36×2和在线W&B [w47t7b4j](https://wandb.ai/pencounkdual-111/PocketXmol_density/runs/w47t7b4j)。模型训练本身无OOM或非有限损失；进度条的1263表示训练批次数，约对应631次优化器更新，尚未到第一次800步验证。2026-09-17 14:02因误读进度而按真实`kill_lock`提前停止；原训练目录`/storage/penghongen/PocketXMol/training/local_cov-C-T0-RA`和标准输出、错误输出全部保留，不作为有效模型或续训来源。
