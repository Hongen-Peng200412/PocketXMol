# `local_cov` 包络模型实验日志

## 当前有效状态

2026-09-21 05:59状态快照：训练、best的E完整测试、CPU评价与在线记录均已完成。最终best为`step=24800.ckpt`、`val/loss=1.4366419315338135`；E的446个实例、22300个候选全部生成并评价成功。ALL逐实例等权的原self-ranking Top-1成功率为78.03%。评价W&B `okdl1eao`为`online_completed`，379402保留`after_lock`并已返回`try_lock`等待。

| 项目 | 当前值 |
|---|---|
| 训练条件 | E，RA＋T0 |
| 正式测试 | E；每实例50候选、100步，推理batch_size=50 |
| 资源 | 379402／gnode09，单张A800 |
| 当前有效release | `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_ef9dd39/PocketXMol`，来源提交`ef9dd39` |
| 当前有效训练产物 | `/storage/penghongen/PocketXMol/training/local_cov-E-T0-RA-b72` |
| best与`val/loss` | 当前best为`step=24800.ckpt`，1.4366419315338135 |
| W&B | 当前run [beds48bb](https://wandb.ai/pencounkdual-111/PocketXmol_density/runs/beds48bb)；首次提前停止run `wvrbzttx`保留 |
| 测试与评价产物 | `/storage/penghongen/PocketXMol/sampling/local_cov-E-T0-RA/test` |
| 测试配置与运行记录 | `/storage/penghongen/tmp/pxm_local_cov_20260917_b72/runs/local_cov-E-T0-RA-test` |
| 评价W&B | [okdl1eao](https://wandb.ai/pencounkdual-111/PocketXmol_density/runs/okdl1eao)，`online_completed` |
| CPU评价 | 全部完成；ALL Top-1为78.03% |

## 完整测试结果

以下成功率和平均RMSD均按实例等权；括号内为RMSD严格小于2 Å的成功实例数。ALL、CAP10、HF10_TO5分别包含446、272、227个实例，并复用同一候选池。

| 协议 | 视图 | 实例数 | Top-1成功率 | Top-5成功率 | Oracle成功率 | Top-1平均RMSD（Å） |
|---|---|---:|---:|---:|---:|---:|
| E | ALL | 446 | 78.03%（348） | 89.24%（398） | 97.09%（433） | 2.024 |
| E | CAP10 | 272 | 73.16%（199） | 85.29%（232） | 95.96%（261） | 2.275 |
| E | HF10_TO5 | 227 | 72.69%（165） | 86.78%（197） | 96.48%（219） | 2.279 |

PDB等权结果先在每个PDB内部平均，再对PDB平均；ALL、CAP10、HF10_TO5分别包含77、67、65个PDB。

| 协议 | 视图 | Top-1成功率 | Top-5成功率 | Oracle成功率 |
|---|---|---:|---:|---:|
| E | ALL | 80.06% | 90.12% | 97.07% |
| E | CAP10 | 79.31% | 88.69% | 96.72% |
| E | HF10_TO5 | 78.93% | 89.30% | 96.76% |

置信度头仍与主体模型共同训练，并按原self-ranking规则使用。Spearman相关系数按实例计算候选置信度与RMSD的秩相关；姿态AUC只统计同时包含成功和失败候选的实例。

| 协议 | 视图 | Spearman均值 | Spearman中位数 | 姿态AUC均值 | AUC有效实例 |
|---|---|---:|---:|---:|---:|
| E | ALL | 0.3425 | 0.3634 | 0.7711 | 397／446 |
| E | CAP10 | 0.3395 | 0.3605 | 0.7642 | 238／272 |
| E | HF10_TO5 | 0.3486 | 0.3663 | 0.7650 | 198／227 |

## 正式运行命令

```bash
bash 训练与运行/sh/train_docking.sh local_cov-E-T0-RA --logdir /storage/penghongen/PocketXMol/training/local_cov-E-T0-RA-b72
/storage/penghongen/PocketXMol/runtime/venv/bin/python scripts/sample_docking.py /storage/penghongen/tmp/pxm_local_cov_20260917_b72/runs/local_cov-E-T0-RA-test/sample-local_cov-E-T0-RA-test.yml
CUDA_VISIBLE_DEVICES='' /storage/penghongen/PocketXMol/runtime/venv/bin/python scripts/evaluate_docking.py /storage/penghongen/tmp/pxm_local_cov_20260917_b72/runs/local_cov-E-T0-RA-test/sample-local_cov-E-T0-RA-test.yml
```

以上命令由379402的动态命令在上述不可变release中顺序执行；训练标准输出和错误分别写入`/storage/penghongen/tmp/pxm_local_cov_20260917_b72/runs/local_cov-E-T0-RA/train.out`和`train.err`。测试配置、采样及评价输出统一保存到同级`local_cov-E-T0-RA-test`目录，采样成功后才执行CPU评价。

## 测试、门控与只读核查

共享验收见[本轮实验日志](本轮实验日志.md)。首次A800门控以36×2完成两次优化器更新。改为72×1后再次门控通过：两次更新耗时329.99秒，峰值显存为64,452,745,728字节已分配、67,159,195,648字节保留，`val/loss=1.9657713175`；E生成并评价2个三步候选，完整检查点恢复逐值通过。

正式训练完成时核对`last.ckpt`：`global_step=26400`、下降计数3、`stop_reason=plateau`；第26400步`val/loss=1.6110507249832153`，best仍为第24800步。训练标准输出含`Training result`、`Training finished!`和`FORMAL_LOCAL_COV_TRAIN_PROCESS_EXIT_ZERO`，错误输出未发现OOM、Traceback或非有限值。锁框架第32次执行成功，父控制目录重新生成`try_lock_379402`并保留`after_lock`。

正式测试启动前确认外置配置读取`step=24800.ckpt`，并固定E协议、test划分、batch50、每实例50候选、100步、8个CPU评价进程和在线W&B。动态命令通过`bash -n`，正式候选目录在启动前不存在。2026-09-20 23:19:54消费379402的`try_lock`后进入采样，`after_lock`保持。

E的446个实例、22300个候选全部生成成功，采样进程零退出且错误输出为空。8进程CPU评价随后完成全部ALL、CAP10和HF10_TO5视图，最终控制记录为`FORMAL_LOCAL_COV_E_TEST_AND_EVALUATION_EXIT_ZERO`。评价错误输出只有既有RDKit allene立体化学提示和依赖库字段告警，没有Traceback或失败标记；W&B `okdl1eao`为`online_completed`且error为null。

## 之前的尝试

- 首次正式运行使用release`PocketXMol_922fc651d3db`、36×2和在线W&B [wvrbzttx](https://wandb.ai/pencounkdual-111/PocketXmol_density/runs/wvrbzttx)。模型训练本身无OOM或非有限损失，空E口袋按既定规则跳过；进度条的1398表示训练批次数，约对应699次优化器更新，尚未到第一次800步验证。2026-09-17 14:00因误读进度而按真实`kill_lock`提前停止；原训练目录`/storage/penghongen/PocketXMol/training/local_cov-E-T0-RA`和标准输出、错误输出全部保留，不作为有效模型或续训来源。
