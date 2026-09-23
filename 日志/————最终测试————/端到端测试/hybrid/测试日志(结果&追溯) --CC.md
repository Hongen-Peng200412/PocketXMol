# 端到端测评：C-C

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
| 1 | 46/77 (59.7%) | 55/77 (71.4%) | 50/77 (64.9%) | 55/77 (71.4%) | 55/77 (71.4%) | 56/77 (72.7%) |
| 3 | 50/77 (64.9%) | 57/77 (74.0%) | 54/77 (70.1%) | 57/77 (74.0%) | 58/77 (75.3%) | 58/77 (75.3%) |
| 5 | 51/77 (66.2%) | 57/77 (74.0%) | 54/77 (70.1%) | 57/77 (74.0%) | 58/77 (75.3%) | 58/77 (75.3%) |
| 10 | 51/77 (66.2%) | 57/77 (74.0%) | 55/77 (71.4%) | 59/77 (76.6%) | 59/77 (76.6%) | 59/77 (76.6%) |
| 20 | 51/77 (66.2%) | 57/77 (74.0%) | 55/77 (71.4%) | 59/77 (76.6%) | 59/77 (76.6%) | 59/77 (76.6%) |

### CA2

| site@K | pose@1 <2 Å | pose@1 <3 Å | pose@5 <2 Å | pose@5 <3 Å | pose@50 <2 Å | pose@50 <3 Å |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 31/77 (40.3%) | 43/77 (55.8%) | 38/77 (49.4%) | 46/77 (59.7%) | 43/77 (55.8%) | 47/77 (61.0%) |
| 3 | 42/77 (54.5%) | 52/77 (67.5%) | 48/77 (62.3%) | 54/77 (70.1%) | 51/77 (66.2%) | 56/77 (72.7%) |
| 5 | 44/77 (57.1%) | 52/77 (67.5%) | 48/77 (62.3%) | 55/77 (71.4%) | 53/77 (68.8%) | 57/77 (74.0%) |
| 10 | 44/77 (57.1%) | 52/77 (67.5%) | 48/77 (62.3%) | 55/77 (71.4%) | 53/77 (68.8%) | 58/77 (75.3%) |
| 20 | 44/77 (57.1%) | 52/77 (67.5%) | 48/77 (62.3%) | 56/77 (72.7%) | 53/77 (68.8%) | 58/77 (75.3%) |

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

- 状态：GT与CA2的第二次`local_cov-C`分别完成280/280与268/268条handoff，两个C-C最终候选池均完整；最终`site@K × pose@M × RMSD`指标正在计算。
- 准备复查：GT与CA2的`prepare_end_to_end.py ... local-c2`进程持续占用一个CPU核，未退出且未改变资源锁；`local-c2`候选尚未开始生成。
- 准备进度口径：每处理一个handoff就写一份无真值`ranking.json`；当前GT为66/280、CA2为32/268。全部排名完成后才写第二次C冻结输入并进入GPU采样。
- 阶段切换：GT完成全部第一次C排名并进入第二次C GPU采样；CA2仍在单核排名阶段。两套输入域继续使用同一冻结release和原self-ranking。
- 第二次阶段切换：GT的C-C最终候选池完成280/280；CA2进入第二次C GPU采样。最终`site@K × pose@M × RMSD`表等待各输入域共享CPU评价完成后写入。
- 最终评价阶段：GT与CA2的第二次C候选和无真值排名均已完成，两个输入域已进入`evaluate_end_to_end.py`最终CPU评价；最终结果表尚未写出。
- 过程：复用第一次local_cov-C，按冻结self-ranking选择Top-1质心，再执行第二次local_cov-C；第二次50候选是唯一最终候选池。
- 正式release：`/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_5032ab445d5e/PocketXMol`。
- GT产物根：`/storage/penghongen/PocketXMol/final_evaluation/end_to_end/GT`；结果与W&B待运行完成后填写。

## 正式运行命令

```bash
bash 训练与运行/sh/final_end_to_end.sh end-to-end-GT
bash 训练与运行/sh/final_end_to_end.sh end-to-end-CA2
```

## 测试、门控与只读核查命令

- 同一正式release的CPU门控为30项通过、1项跳过；真实validation GPU门控为3项通过。

## 失败、中断与恢复记录

旧串行最终评价在并行实现通过等价门控后由真实 `kill_lock` 停止；已有 GPU 推理、排序和科学输入产物全部保留。32 进程评价从全部冻结推理产物重新计算，本方法未出现直接结果失败。
