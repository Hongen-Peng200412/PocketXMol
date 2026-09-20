# `local_cov` 包络模型实验日志

## 当前有效状态

2026-09-21 02:27状态快照：正式训练最终best为`step=24800.ckpt`、`val/loss=1.4366419315338135`。379402上的完整测试集E推理已完成270／446个实例、13500个候选，全部成功；采样错误输出为空，`after_lock`保留。CPU评价将在全部E候选生成成功后由同一动态命令执行。

| 项目 | 当前值 |
|---|---|
| 训练条件 | E，RA＋T0 |
| 正式测试 | E；每实例50候选、100步，推理batch_size=50 |
| 资源 | 379402／gnode09，单张A800 |
| 当前有效release | `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_ef9dd39/PocketXMol`，来源提交`ef9dd39` |
| 当前有效训练产物 | `/storage/penghongen/PocketXMol/training/local_cov-E-T0-RA-b72` |
| best与`val/loss` | 当前best为`step=24800.ckpt`，1.4366419315338135 |
| W&B | 当前run [beds48bb](https://wandb.ai/pencounkdual-111/PocketXmol_density/runs/beds48bb)；首次提前停止run `wvrbzttx`保留 |
| 测试与评价产物 | `/storage/penghongen/PocketXMol/sampling/local_cov-E-T0-RA/test` |
| 测试配置与运行记录 | `/storage/penghongen/tmp/pxm_local_cov_20260917_b72/runs/local_cov-E-T0-RA-test` |
| CPU评价 | 尚未执行；E推理运行中，成功后自动接续 |

## 正式运行命令

```bash
bash 训练与运行/sh/train_docking.sh local_cov-E-T0-RA --logdir /storage/penghongen/PocketXMol/training/local_cov-E-T0-RA-b72
/storage/penghongen/PocketXMol/runtime/venv/bin/python scripts/sample_docking.py /storage/penghongen/tmp/pxm_local_cov_20260917_b72/runs/local_cov-E-T0-RA-test/sample-local_cov-E-T0-RA-test.yml
CUDA_VISIBLE_DEVICES='' /storage/penghongen/PocketXMol/runtime/venv/bin/python scripts/evaluate_docking.py /storage/penghongen/tmp/pxm_local_cov_20260917_b72/runs/local_cov-E-T0-RA-test/sample-local_cov-E-T0-RA-test.yml
```

以上命令由379402的动态命令在上述不可变release中顺序执行；训练标准输出和错误分别写入`/storage/penghongen/tmp/pxm_local_cov_20260917_b72/runs/local_cov-E-T0-RA/train.out`和`train.err`。测试配置、采样及评价输出统一保存到同级`local_cov-E-T0-RA-test`目录，采样成功后才执行CPU评价。

## 测试、门控与只读核查

共享验收见[本轮实验日志](本轮实验日志.md)。首次A800门控以36×2完成两次优化器更新。改为72×1后再次门控通过：两次更新耗时329.99秒，峰值显存为64,452,745,728字节已分配、67,159,195,648字节保留，`val/loss=1.9657713175`；E生成并评价2个三步候选，完整检查点恢复逐值通过。

正式训练完成时核对`last.ckpt`：`global_step=26400`、下降计数3、`stop_reason=plateau`；第26400步`val/loss=1.6110507249832153`，best仍为第24800步。训练标准输出含`Training result`、`Training finished!`和`FORMAL_LOCAL_COV_TRAIN_PROCESS_EXIT_ZERO`，错误输出未发现OOM、Traceback或非有限值。锁框架第32次执行成功，父控制目录重新生成`try_lock_379402`并保留`after_lock`。

正式测试启动前确认外置配置读取`step=24800.ckpt`，并固定E协议、test划分、batch50、每实例50候选、100步、8个CPU评价进程和在线W&B。动态命令通过`bash -n`，正式候选目录在启动前不存在。2026-09-20 23:19:54消费379402的`try_lock`后进入采样，`after_lock`保持。

## 之前的尝试

- 首次正式运行使用release`PocketXMol_922fc651d3db`、36×2和在线W&B [wvrbzttx](https://wandb.ai/pencounkdual-111/PocketXmol_density/runs/wvrbzttx)。模型训练本身无OOM或非有限损失，空E口袋按既定规则跳过；进度条的1398表示训练批次数，约对应699次优化器更新，尚未到第一次800步验证。2026-09-17 14:00因误读进度而按真实`kill_lock`提前停止；原训练目录`/storage/penghongen/PocketXMol/training/local_cov-E-T0-RA`和标准输出、错误输出全部保留，不作为有效模型或续训来源。
