# 端到端测评：C-C-E

## 当前状态与产物

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

当前无正式运行尝试。
