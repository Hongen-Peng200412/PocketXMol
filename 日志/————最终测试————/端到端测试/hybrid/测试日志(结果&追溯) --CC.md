# 端到端测评：C-C

## 当前状态与产物

- 状态：GT的第二次`local_cov-C`已完成280/280，C-C最终候选池完整；CA2完成全部第一次C排名，第二次`local_cov-C`已生成38/268条handoff。
- 准备复查：GT与CA2的`prepare_end_to_end.py ... local-c2`进程持续占用一个CPU核，未退出且未改变资源锁；`local-c2`候选尚未开始生成。
- 准备进度口径：每处理一个handoff就写一份无真值`ranking.json`；当前GT为66/280、CA2为32/268。全部排名完成后才写第二次C冻结输入并进入GPU采样。
- 阶段切换：GT完成全部第一次C排名并进入第二次C GPU采样；CA2仍在单核排名阶段。两套输入域继续使用同一冻结release和原self-ranking。
- 第二次阶段切换：GT的C-C最终候选池完成280/280；CA2进入第二次C GPU采样。最终`site@K × pose@M × RMSD`表等待各输入域共享CPU评价完成后写入。
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

当前无正式运行尝试。
