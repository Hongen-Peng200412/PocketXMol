# `local_cov` 中心模型实验日志

## 当前有效状态

2026-09-17 14:02状态快照：首次正式运行在进度条显示1263个训练批次时，因把该值误读为优化器更新数而按378693真实`kill_lock`提前停止。36×2下实际只约631次优化器更新，尚未到第一次800步验证，因此没有检查点属于正常现象。`after_lock`和全部产物保留，资源已恢复`try_lock`。原验证系统不修改；72×1 GPU验收通过后从官方参数在独立产物目录重启。

| 项目 | 当前值 |
|---|---|
| 训练条件 | C0，RA＋T0 |
| 正式测试 | C0、C5；各实例各协议50候选、100步，推理batch_size=50 |
| 资源 | 378693／gnode10，单张A800 |
| 当前有效release | 待72×1配置验收后冻结；沿用原验证系统 |
| 当前有效训练产物 | 待重启后生成 |
| best与`val/loss` | 尚未产生 |
| W&B | 重启run待生成；首次失败run `w47t7b4j`保留 |
| CPU评价 | 尚未执行 |

## 正式运行命令

```bash
bash 训练与运行/sh/train_docking.sh local_cov-C-T0-RA
```

该命令由378693的动态命令在上述不可变release中执行；标准输出和错误分别写入`/storage/penghongen/tmp/pxm_local_cov_20260917/runs/local_cov-C-T0-RA/train.out`和`train.err`。

## 测试、门控与只读核查

共享验收见[本轮实验日志](本轮实验日志.md)。A800门控以36×2完成两次优化器更新，峰值显存为32,274,164,736字节已分配、34,902,900,736字节保留，`val/loss=3.0541803837`；C0、C5各生成并评价2个三步候选，完整检查点恢复逐值通过。

## 之前的尝试

- 首次正式运行使用release`PocketXMol_922fc651d3db`、36×2和在线W&B [w47t7b4j](https://wandb.ai/pencounkdual-111/PocketXmol_density/runs/w47t7b4j)。模型训练本身无OOM或非有限损失；进度条的1263表示训练批次数，约对应631次优化器更新，尚未到第一次800步验证。2026-09-17 14:02因误读进度而按真实`kill_lock`提前停止；原训练目录`/storage/penghongen/PocketXMol/training/local_cov-C-T0-RA`和标准输出、错误输出全部保留，不作为有效模型或续训来源。
