# `local_cov` 包络模型实验日志

## 当前有效状态

2026-09-17 18:05状态快照：正式训练已推进至约1513次优化器更新，尚未到第1600步验证；当前best仍为step800、`val/loss=1.796922206878662`。训练日志持续更新，无OOM、Traceback或非有限值；空E口袋继续按既定规则跳过并记录，379402仍处于运行态。

| 项目 | 当前值 |
|---|---|
| 训练条件 | E，RA＋T0 |
| 正式测试 | E；每实例50候选、100步，推理batch_size=50 |
| 资源 | 379402／gnode09，单张A800 |
| 当前有效release | `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_ef9dd39/PocketXMol`，来源提交`ef9dd39` |
| 当前有效训练产物 | `/storage/penghongen/PocketXMol/training/local_cov-E-T0-RA-b72` |
| best与`val/loss` | 当前best为`step=800.ckpt`，1.796922206878662 |
| W&B | 当前run [beds48bb](https://wandb.ai/pencounkdual-111/PocketXmol_density/runs/beds48bb)；首次提前停止run `wvrbzttx`保留 |
| CPU评价 | 尚未执行 |

## 正式运行命令

```bash
bash 训练与运行/sh/train_docking.sh local_cov-E-T0-RA --logdir /storage/penghongen/PocketXMol/training/local_cov-E-T0-RA-b72
```

该命令由379402的动态命令在上述不可变release中执行；标准输出和错误分别写入`/storage/penghongen/tmp/pxm_local_cov_20260917_b72/runs/local_cov-E-T0-RA/train.out`和`train.err`。

## 测试、门控与只读核查

共享验收见[本轮实验日志](本轮实验日志.md)。首次A800门控以36×2完成两次优化器更新。改为72×1后再次门控通过：两次更新耗时329.99秒，峰值显存为64,452,745,728字节已分配、67,159,195,648字节保留，`val/loss=1.9657713175`；E生成并评价2个三步候选，完整检查点恢复逐值通过。

## 之前的尝试

- 首次正式运行使用release`PocketXMol_922fc651d3db`、36×2和在线W&B [wvrbzttx](https://wandb.ai/pencounkdual-111/PocketXmol_density/runs/wvrbzttx)。模型训练本身无OOM或非有限损失，空E口袋按既定规则跳过；进度条的1398表示训练批次数，约对应699次优化器更新，尚未到第一次800步验证。2026-09-17 14:00因误读进度而按真实`kill_lock`提前停止；原训练目录`/storage/penghongen/PocketXMol/training/local_cov-E-T0-RA`和标准输出、错误输出全部保留，不作为有效模型或续训来源。
