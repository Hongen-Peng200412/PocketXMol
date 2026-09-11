# B-C-T1-RA正式训练

本实验属于已批准的六个无密度模型，承接 [共同准备与实现](00-实现与共同数据准备.md) 和 [中心T0纠偏](02-T0中心契约修复与重训.md)。它使用RA核酸编码、中心T1机制；不从正确或错误T0的训练检查点续训。

训练已于22400次优化器更新后正常停止，最低原C5 val/loss对应20800步检查点。停止状态和best已从checkpoint核实，后续直接执行 [完整测试与评价](05-B-C-T1-RA推理与评价.md)。

## 固定训练条件

配置为 `configs/docking/B-C-T1-RA.yml`。训练每次抽一份半径0–5 Å均匀、方向球面均匀的delta，用完整g+delta选袋并定原点；原高斯后加入s*delta，监督目标和受体不跟随该新增平移。原val/loss读取验证清单冻结C5偏移及同一T1公式。g为完整配体重原子中心，s=1-level_dict['pos']。正式姿态评价仍保留C0与C5，推理首步为原纯高斯，后续保留中心相关重新加噪。

从规定官方pxm参数初始化，重新建立AdamW及调度状态；单张A800使用batch_size=72、累积1、bf16，名义全局批量72。初始lr=1e-4、warmup=0，每800个优化器更新计算原val/loss；Plateau相对阈值1%、patience=5、factor=0.2，第三次实际下降停止，上限40000步。最低原验证损失选择best，last和全部定期检查点保留。

## 正式运行命令与产物

在既有371591的独立冻结release中运行：

```bash
bash 训练与运行/sh/train_docking.sh B-C-T1-RA
```

产物目录为 `/storage/penghongen/PocketXMol/training/B-C-T1-RA/`，W&B账号／项目为 `pencounkdual-111/PocketXmol_raw`，新run独立创建。[正确T0的测试推理与CPU评价](04-B-C-T0-RA推理与评价.md)完成并记录后进入本实验；实际launch和run id见下文。本模型训练后也直接测试，不安排完整验证集采样；训练期间原val/loss和best选择保留。

本次执行沿用已审查和实际运行的ec06dbd源码，使用既有不可变release `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_fa0d957b2d3f/PocketXMol` 中的B-C-T1-RA.yml与train_docking.sh。该配置明确center_translation=true、RA、72×1、原官方初始化和独立输出目录；本次不传--resume。运行来源由该release确定，不吸收其他代理的未提交工作区文件。

## 实际启动

2026-09-12，确认前一模型测试报告完成、371591处于try_lock等待、after_lock保留且目标训练目录不存在后，接入上述正式命令。master请求时间为00:42:12；gnode09时钟约慢3分钟。控制器第14次执行的PocketXMol launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/train_B-C-T1-RA_job371591_20260912T003912`，实际训练主进程为16005，cgroup确认属于371591。

启动元数据为 `/storage/penghongen/PocketXMol/control/371591/train_B-C-T1-RA_start.json`，正式动态命令副本为同目录train_B-C-T1-RA_run_cmd.sh。该次out／err读取起点分别为57342372／73878；旧T0训练、已取消validation和已完成test日志不混入本次状态判断。after_lock保留，运行时try_lock和kill_lock均不存在。

W&B训练run为 [nzkna4ow](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/nzkna4ow)，名称B-C-T1-RA。输出明确从规定的官方pocketxmol.ckpt加载模型参数、使用bf16，约20.2M参数全部可训练；目标目录独立创建。启动检查时已完成62个训练批次（72×1对应62次优化器更新），约1.01步／秒，lr=1e-4，原loss及置信度损失正常记录，无OOM或Traceback。首次正式验证在800次更新时进行，最终结果见下文。

运行沿用已通过自查、两轮独立审查及CPU／GPU验收的实现，不增加科学开关或重建共同资产。测试命令和验收结果见日志00及02，本文件上面的命令是正式训练命令。371591的after_lock保留，不申请额外GPU，不释放或删除旧T0产物。

## 正常停止与best核对

2026-09-12，控制器第14次执行成功，训练主进程16005结束，try_lock恢复，after_lock保留。本次out／err范围未发现Traceback、CUDA OOM、reduce_batch或NaN记录。独立Slurm检查步骤371591.1读取last.ckpt及best，确认如下：

| 检查项 | 结果 |
|---|---|
| global_step／最后验证步 | 均为22400 |
| stop_reason／下降次数 | plateau／3 |
| last中优化器及调度器lr | 均为8.000000000000002e-7 |
| 最低原val/loss | 2.831010341644287 |
| best | checkpoints/step=20800.ckpt，global_step=20800 |
| 保存数量 | 28个定期检查点及last，共29个文件，全部保留 |
| W&B run id／模型参数键 | nzkna4ow／1130个model.键，与RA一致 |

best以原验证损失的绝对最小值选择。20800步比17600步的2.8402891159057617更低，但改善幅度未达调度器要求的1%，因此调度器的显著改善基准仍为17600步；其后六次验证未达要求，于22400步触发第三次下降并停止。这不影响20800步作为正式测试检查点。T1验证输入为C5，不能把其验证损失与T0的C0验证损失直接解释为同条件模型比较。

以下是只读检查点核查命令，属于验收，不是正式训练或测试推理：

```bash
srun --jobid=371591 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /storage/penghongen/PocketXMol/runtime/venv/bin/python -B /storage/penghongen/tmp/pocketxmol_checkpoint_20260912/inspect_t1_ra.py
```

核查报告保存为 `/storage/penghongen/PocketXMol/training/B-C-T1-RA/training_summary_20260912.json`。报告验证了停止步、下降次数、W&B身份、全部定期检查点存在、best确为记录中的最低损失及best文件实际步数；不修改任何checkpoint。测试配置和后续运行证据见日志05。
