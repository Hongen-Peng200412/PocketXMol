# `local_cov` 中心模型实验日志

## 当前有效状态

2026-09-20 06:03状态快照：正式训练已在第26400次优化器更新后正常结束，最终模型选择best为`step=24800.ckpt`、`val/loss=1.61287522315979`。best的C0/C5完整测试已于06:01在378693／gnode10启动，计算节点确认正式采样进程和GPU显存占用；采样成功后将由同一动态命令执行CPU评价。`after_lock`和全部训练产物保留。

| 项目 | 当前值 |
|---|---|
| 训练条件 | C0，RA＋T0 |
| 正式测试 | C0、C5；各实例各协议50候选、100步，推理batch_size=50 |
| 资源 | 378693／gnode10，单张A800 |
| 当前有效release | `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_ef9dd39/PocketXMol`，来源提交`ef9dd39` |
| 当前有效训练产物 | `/storage/penghongen/PocketXMol/training/local_cov-C-T0-RA-b72` |
| best与`val/loss` | 当前best为`step=24800.ckpt`，1.61287522315979 |
| W&B | 当前run [db4b74gn](https://wandb.ai/pencounkdual-111/PocketXmol_density/runs/db4b74gn)；首次提前停止run `w47t7b4j`保留 |
| CPU评价 | 尚未执行 |

## 正式运行命令

```bash
bash 训练与运行/sh/train_docking.sh local_cov-C-T0-RA --logdir /storage/penghongen/PocketXMol/training/local_cov-C-T0-RA-b72
/storage/penghongen/PocketXMol/runtime/venv/bin/python scripts/sample_docking.py /storage/penghongen/tmp/pxm_local_cov_20260917_b72/runs/local_cov-C-T0-RA-test/sample-local_cov-C-T0-RA-test.yml
CUDA_VISIBLE_DEVICES='' /storage/penghongen/PocketXMol/runtime/venv/bin/python scripts/evaluate_docking.py /storage/penghongen/tmp/pxm_local_cov_20260917_b72/runs/local_cov-C-T0-RA-test/sample-local_cov-C-T0-RA-test.yml
```

以上命令由378693的动态命令在上述不可变release中顺序执行；训练标准输出和错误分别写入`/storage/penghongen/tmp/pxm_local_cov_20260917_b72/runs/local_cov-C-T0-RA/train.out`和`train.err`。测试配置、采样及评价输出统一保存到同级`local_cov-C-T0-RA-test`目录，采样成功后才执行CPU评价。

## 测试、门控与只读核查

共享验收见[本轮实验日志](本轮实验日志.md)。首次A800门控以36×2完成两次优化器更新。改为72×1后再次门控通过：两次更新耗时415.31秒，峰值显存为64,687,791,104字节已分配、67,448,602,624字节保留，`val/loss=2.9227933884`；C0、C5各生成并评价2个三步候选，完整检查点恢复逐值通过。

正式测试启动前确认训练输出含`Training finished!`和零退出标记，完整检查点的停止原因为`plateau`、下降计数为3；外置配置解析确认C0/C5、test划分、batch50、每实例每协议50候选、100步、8个CPU评价进程和在线W&B。动态命令通过`bash -n`，正式候选目录在启动前不存在。06:01消费378693的`try_lock`后，计算节点确认采样进程读取`step=24800.ckpt`并占用GPU；`after_lock`保持。

## 之前的尝试

- 首次正式运行使用release`PocketXMol_922fc651d3db`、36×2和在线W&B [w47t7b4j](https://wandb.ai/pencounkdual-111/PocketXmol_density/runs/w47t7b4j)。模型训练本身无OOM或非有限损失；进度条的1263表示训练批次数，约对应631次优化器更新，尚未到第一次800步验证。2026-09-17 14:02因误读进度而按真实`kill_lock`提前停止；原训练目录`/storage/penghongen/PocketXMol/training/local_cov-C-T0-RA`和标准输出、错误输出全部保留，不作为有效模型或续训来源。
