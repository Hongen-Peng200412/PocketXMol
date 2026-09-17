# `local_cov` 包络模型实验日志

## 当前有效状态

2026-09-17 12:36：379402／gnode09已从官方参数启动正式训练。36×2、bf16-mixed和在线W&B已生效，当前正常运行；尚未到第一次800步`val/loss`。

| 项目 | 当前值 |
|---|---|
| 训练条件 | E，RA＋T0 |
| 正式测试 | E；每实例50候选、100步，推理batch_size=50 |
| 资源 | 379402／gnode09，单张A800 |
| 正式release | `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_922fc651d3db/PocketXMol`，来源提交`3c0dd29` |
| 训练产物 | `/storage/penghongen/PocketXMol/training/local_cov-E-T0-RA` |
| best与`val/loss` | 尚未产生 |
| W&B | `pencounkdual-111/PocketXmol_density`，run id `wvrbzttx`，[在线运行](https://wandb.ai/pencounkdual-111/PocketXmol_density/runs/wvrbzttx) |
| CPU评价 | 尚未执行 |

## 正式运行命令

```bash
bash 训练与运行/sh/train_docking.sh local_cov-E-T0-RA
```

该命令由379402的动态命令在上述不可变release中执行；标准输出和错误分别写入`/storage/penghongen/tmp/pxm_local_cov_20260917/runs/local_cov-E-T0-RA/train.out`和`train.err`。

## 测试、门控与只读核查

共享验收见[本轮实验日志](本轮实验日志.md)。A800门控以36×2完成两次优化器更新，峰值显存为32,042,027,520字节已分配、34,714,157,056字节保留，`val/loss=1.8953429461`；E生成并评价2个三步候选，完整检查点恢复逐值通过。

## 之前的尝试

尚无。
