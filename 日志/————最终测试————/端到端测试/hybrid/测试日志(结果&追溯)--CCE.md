# 端到端测评：C-C-E

<!-- final-results:begin -->
## 最终结果与产物

- 状态：已完成。GT 280 条和 CA2 268 条直接结果均按 `inputs/base.jsonl` 原顺序写出。本方法的 GT 280 条与 CA2 268 条直接结果全部成功。
- 统计口径：77 个 PDB；site K 为 1、3、5、10、20；pose M 为 1、5、50；RMSD 阈值严格小于 2 Å 或 3 Å。表格单元格为“成功 PDB 数／77（成功率）”。
- GT 结果根：`/storage/penghongen/PocketXMol/final_evaluation/end_to_end/GT`。
- CA2 结果根：`/storage/penghongen/PocketXMol/final_evaluation/end_to_end/CA2`。
- 最终文件：两项结果根下均保存 `docking_results.jsonl`、`evaluation.json`、`wandb_run.json`；直接结果顺序已分别与各自 `inputs/base.jsonl` 精确核对。
- 最终评价 release：`/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_f4d72bbea387/PocketXMol`。
- W&B：GT [z9ei2jzt](https://wandb.ai/pencounkdual-111/PocketXmol_final/runs/z9ei2jzt)；CA2 [so7ostew](https://wandb.ai/pencounkdual-111/PocketXmol_final/runs/so7ostew)。

### GT

| site@K | pose@1 <2 Å | pose@1 <3 Å | pose@5 <2 Å | pose@5 <3 Å | pose@50 <2 Å | pose@50 <3 Å |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 48/77 (62.3%) | 53/77 (68.8%) | 51/77 (66.2%) | 55/77 (71.4%) | 54/77 (70.1%) | 56/77 (72.7%) |
| 3 | 53/77 (68.8%) | 56/77 (72.7%) | 56/77 (72.7%) | 57/77 (74.0%) | 57/77 (74.0%) | 59/77 (76.6%) |
| 5 | 53/77 (68.8%) | 56/77 (72.7%) | 56/77 (72.7%) | 57/77 (74.0%) | 58/77 (75.3%) | 59/77 (76.6%) |
| 10 | 53/77 (68.8%) | 57/77 (74.0%) | 58/77 (75.3%) | 59/77 (76.6%) | 59/77 (76.6%) | 60/77 (77.9%) |
| 20 | 53/77 (68.8%) | 57/77 (74.0%) | 58/77 (75.3%) | 59/77 (76.6%) | 59/77 (76.6%) | 60/77 (77.9%) |

### CA2

| site@K | pose@1 <2 Å | pose@1 <3 Å | pose@5 <2 Å | pose@5 <3 Å | pose@50 <2 Å | pose@50 <3 Å |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 34/77 (44.2%) | 42/77 (54.5%) | 38/77 (49.4%) | 45/77 (58.4%) | 42/77 (54.5%) | 47/77 (61.0%) |
| 3 | 44/77 (57.1%) | 51/77 (66.2%) | 48/77 (62.3%) | 54/77 (70.1%) | 51/77 (66.2%) | 56/77 (72.7%) |
| 5 | 45/77 (58.4%) | 51/77 (66.2%) | 48/77 (62.3%) | 55/77 (71.4%) | 52/77 (67.5%) | 57/77 (74.0%) |
| 10 | 45/77 (58.4%) | 51/77 (66.2%) | 48/77 (62.3%) | 55/77 (71.4%) | 52/77 (67.5%) | 57/77 (74.0%) |
| 20 | 45/77 (58.4%) | 51/77 (66.2%) | 48/77 (62.3%) | 55/77 (71.4%) | 52/77 (67.5%) | 57/77 (74.0%) |

### 最终 CPU 评价正式命令

```bash
python scripts/evaluate_end_to_end.py configs/docking/end-to-end-GT.yml
python scripts/evaluate_end_to_end.py configs/docking/end-to-end-CA2.yml
```

### 并行评价门控

- 最终 release 的 `tests/test_final_evaluation.py` 为 5 项通过。
- validation 非测试样本 `5irx/0` 的四阶段状态均为 `success`；`workers=1` 与 `workers=4` 的逐候选结果、RMSD、Top-1、成功布尔值、汇总和输出顺序精确一致。证据位于 `/storage/penghongen/PocketXMol/gates/end_to_end_parallel_validation/equivalence_v3`。
<!-- final-results:end -->

## 执行过程与历史状态

- 状态：验收完成；GT与CA2的第二次C、无真值排名及预测E均分别完成280/280与268/268条handoff，最终`site@K × pose@M × RMSD`指标正在计算。
- 过程：复用两次local_cov-C，以第二次C的Top-1预测重原子坐标构造现行E口袋，再执行local_cov-E；E的50候选是唯一最终候选池。
- 正式release：`/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_5032ab445d5e/PocketXMol`。
- GT产物根：`/storage/penghongen/PocketXMol/final_evaluation/end_to_end/GT`；结果与W&B待运行完成后填写。
- 最终评价阶段：两个输入域的C-C-E最终候选池均完整，`evaluate_end_to_end.py`正在计算77个PDB分母下的结果表；最终产物尚未写出。

## 正式运行命令

```bash
bash 训练与运行/sh/final_end_to_end.sh end-to-end-GT
bash 训练与运行/sh/final_end_to_end.sh end-to-end-CA2
```

## 测试、门控与只读核查命令

- 同一正式release的CPU门控为30项通过、1项跳过；真实validation GPU门控为3项通过。

## 失败、中断与恢复记录

旧串行最终评价在并行实现通过等价门控后由真实 `kill_lock` 停止；已有 GPU 推理、排序和科学输入产物全部保留。32 进程评价从全部冻结推理产物重新计算，本方法未出现直接结果失败。
