# 端到端测评：C-C

## 当前状态与产物

- 状态：验收完成；GT与CA2的第一次`local_cov-C`分别完成280/280和268/268，两套输入域均正在依据第一次C的Top-1预测准备第二次C输入。
- 准备复查：GT与CA2的`prepare_end_to_end.py ... local-c2`进程持续占用一个CPU核，未退出且未改变资源锁；`local-c2`候选尚未开始生成。
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
