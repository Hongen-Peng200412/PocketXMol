# 端到端测评：C-C

## 当前状态与产物

- 状态：验收完成；GT正式流程已启动，当前等待第一次local_cov-C候选；CA2尚未启动。
- 过程：复用第一次local_cov-C，按冻结self-ranking选择Top-1质心，再执行第二次local_cov-C；第二次50候选是唯一最终候选池。
- 正式release：`/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_5032ab445d5e/PocketXMol`。
- GT产物根：`/storage/penghongen/PocketXMol/final_evaluation/end_to_end/GT`；结果与W&B待运行完成后填写。

## 正式运行命令

```bash
bash 训练与运行/sh/final_end_to_end.sh end-to-end-GT
```

## 测试、门控与只读核查命令

- 同一正式release的CPU门控为30项通过、1项跳过；真实validation GPU门控为3项通过。

## 失败、中断与恢复记录

当前无正式运行尝试。
