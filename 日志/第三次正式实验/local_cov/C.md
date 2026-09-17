# `local_cov` 中心模型实验日志

## 当前有效状态

2026-09-17 12:43状态快照：378693／gnode10已推进至优化器更新约80步。36×2、bf16-mixed和在线W&B正常，学习率为`1e-4`，未见OOM或非有限损失；尚未到第一次800步`val/loss`。

| 项目 | 当前值 |
|---|---|
| 训练条件 | C0，RA＋T0 |
| 正式测试 | C0、C5；各实例各协议50候选、100步，推理batch_size=50 |
| 资源 | 378693／gnode10，单张A800 |
| 正式release | `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_922fc651d3db/PocketXMol`，来源提交`3c0dd29` |
| 训练产物 | `/storage/penghongen/PocketXMol/training/local_cov-C-T0-RA` |
| best与`val/loss` | 尚未产生 |
| W&B | `pencounkdual-111/PocketXmol_density`，run id `w47t7b4j`，[在线运行](https://wandb.ai/pencounkdual-111/PocketXmol_density/runs/w47t7b4j) |
| CPU评价 | 尚未执行 |

## 正式运行命令

```bash
bash 训练与运行/sh/train_docking.sh local_cov-C-T0-RA
```

该命令由378693的动态命令在上述不可变release中执行；标准输出和错误分别写入`/storage/penghongen/tmp/pxm_local_cov_20260917/runs/local_cov-C-T0-RA/train.out`和`train.err`。

## 测试、门控与只读核查

共享验收见[本轮实验日志](本轮实验日志.md)。A800门控以36×2完成两次优化器更新，峰值显存为32,274,164,736字节已分配、34,902,900,736字节保留，`val/loss=3.0541803837`；C0、C5各生成并评价2个三步候选，完整检查点恢复逐值通过。

## 之前的尝试

尚无。
