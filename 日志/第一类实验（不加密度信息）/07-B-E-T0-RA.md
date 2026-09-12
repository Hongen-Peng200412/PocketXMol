# B-E-T0-RA训练与配套测试

本文件记录六个无密度模型中的第3个实验，依据 [科学契约](../../想法/方案草稿/9-8-科学契约.md)、[工程细节](../../想法/方案草稿/9-8-工程与实现细节.md) 和 [边界清单](../../想法/方案草稿/9-8-边界与核查清单.md)。按用户授权，在371591完成 [B-C-T1-RA测试报告](05-B-C-T1-RA推理与评价.md) 后，使用同一A800运行本模型训练、E测试、CPU评价与记录，再进入第4个B-C-T0-RB。

当前首次正式训练因空E口袋导致NaN而停止，尚无有效检查点或正式E测试结果。已保留异常运行的源码、配置、W&B及日志，371591的after_lock继续保留；正在只读统计训练与验证中的具体空口袋实例，处理口径已提交用户决定，见下文。

## 固定训练条件

配置为 `configs/docking/B-E-T0-RA.yml`，源码和配置均复用已审查的ec06dbd运行副本 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_fa0d957b2d3f/PocketXMol`。本次没有修改生产函数、训练配置或共同资产。

包络口袋选择残基重原子质量中心到任一完整配体重原子距离严格小于10 Å的完整受体残基；模型原点为实际输入受体重原子世界坐标的算术均值，配体与受体共同减去该原点。RA在联合受体图中共享编码器，保留核酸特征投影；center_translation=false，始终使用原dock高斯噪声链。训练与原val/loss均使用E条件，不抽新增中心偏移。

从规定官方pocketxmol.ckpt只加载模型参数，不传--resume；独立建立AdamW及调度状态，保留原loss、置信度头和训练目标。单卡batch_size=72、累积1、bf16、15个数据worker，全局名义批量72。初始lr=1e-4、warmup=0，每800次优化器更新计算原val/loss；Plateau采用相对阈值1%、patience=5、factor=0.2，第三次实际下降立即停止，最多40000步。best取最低原验证损失，保留last及全部定期检查点。

## 正式命令与产物

正式训练在上述冻结release中执行：

```bash
bash 训练与运行/sh/train_docking.sh B-E-T0-RA
```

训练根为 `/storage/penghongen/PocketXMol/training/B-E-T0-RA/`，W&B账号与项目为 `pencounkdual-111/PocketXmol_raw`、名称B-E-T0-RA，实际run为[so6e0mvy](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/so6e0mvy)。使用371591、gnode09的1张A800及16核CPU，沿其实际Pocket_Plus控制目录重用资源，保留after_lock，不另申请GPU或CPU作业。

## 运行前核查与后续步骤

包络T0-RA代码和72×1配置已经通过两遍主代理自查、两轮全面独立审查及真实非测试样本GPU验收，见 [共同准备](00-实现与共同数据准备.md) 和 [T0纠偏](02-T0中心契约修复与重训.md)。本次沿用通过验收的配置，不重复试训或学习率搜索。启动前再次核对现有资源已回到try_lock、旧评价进程退出、after_lock保留、新训练根不存在及release配置与本地一致。

训练正常结束后，在Slurm内核对停止原因和实际best检查点，再建立明确best的E测试配置。完整测试446个实例，每实例50候选、100步、推理batch50，复用冻结种子和三个既定视图；推理完成后在同一371591作业内用8个CPU评价进程完成结果汇总。不进行训练后完整验证集采样或评价。

## 实际接入

B-C-T1-RA完整报告和本次运行登记已由2a1bed8提交。接入前，核对371591仍为gnode09的1张A800／16核，前一评价控制器第16次执行成功、旧进程33502消失、try_lock及after_lock存在、kill_lock不存在，新训练根不存在。本地与冻结release中B-E-T0-RA.yml原始字节一致，SHA256均为cce20716d5a2408233f2142d85ce637c5e2e2b0245d657b8365faa1ec28826c8；没有修改科学配置。

于2026-09-12 master时间16:43:45写入正式动态命令后，移除已获准的try_lock_371591，控制器第17次执行开始。实际训练进程9532，cgroup确认为job_371591。PocketXMol launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/train_B-E-T0-RA_job371591_20260912T164054`；gnode09时钟约慢3分钟，保留各自记录，不跨主机时间估算时长。

启动元数据为 `/storage/penghongen/PocketXMol/control/371591/train_B-E-T0-RA_start.json`，动态命令副本为同目录train_B-E-T0-RA_run_cmd.sh；本次out／err起点83210385／132728。科学源码与配置来自上述fa0d957b2d3f的PocketXMol release，旧Pocket_Plus控制器仅负责资源控制。after_lock保留，未使用kill_lock或scancel，旧训练及测试产物保留。

实际输出已确认从规定官方pocketxmol.ckpt加载，CUDA与bf16启用，W&B在线创建so6e0mvy；本地W&B记录为训练根下wandb/run-20260912_164135-so6e0mvy。没有--resume，不继承前一模型的参数、优化器或调度状态。加载阶段没有Traceback、CUDA OOM或reduce_batch；这不代表训练数值有效，后续NaN情况见下文。

## 空E口袋导致的异常停止

后续启动检查发现train/pos、train/total及置信度损失持续NaN。回读本次日志，第16步显示的损失仍有限，首次NaN出现在17步附近；此后的更新持续异常。stderr同时反复报告docking/dataset.py:149对空数组求均值，明确指出E原点构造问题，不能将其解释为正常训练。

按用户已授予的371591 kill_lock权限，核对/proc/9532/cmdline确为本次B-E-T0-RA及cgroup为job_371591后，创建实际kill_lock。节点时间2026-09-12 16:43:58记录停止请求，最后观察到179步；控制器第17次执行退出137，进程9532消失，try_lock恢复，after_lock保留，kill_lock由控制器处理。没有scancel或删除产物。停止证据为 `/storage/penghongen/PocketXMol/control/371591/train_B-E-T0-RA_nan_stop_20260912.json`。尚未到第一次800步验证，因此checkpoints目录为空；W&B so6e0mvy属于此次异常训练，不能作为有效模型。

独立只读逻辑排查确认：准备阶段只验证整个PDB存在标准受体，没有保证每个配体实例的E选袋非空。E按标准残基质量中心到任一配体重原子严格小于10 Å筛选，RA没有额外删除核酸；没有残基满足条件时，对空坐标求均值产生NaN原点，原FeaturizeMol继续减去该原点，使监督坐标成为NaN。中心模型使用有限的给定中心，不能用中心模型的稳定训练证明E始终有受体。

本次只读CPU诊断脚本为本地 `tmp/pxm-20260912/audit_empty_envelope.py`，服务器副本位于 `/storage/penghongen/tmp/pocketxmol_empty_envelope_20260912/audit_empty_envelope.py`。使用同一371591的8核CPU扫描冻结train／validation，调用未修改的生产select_pocket；不读取test，不重建或改写任何冻结清单及源资产。以下是诊断验收命令，不是正式训练或评价命令：

```bash
srun --jobid=371591 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /storage/penghongen/PocketXMol/runtime/venv/bin/python -B /storage/penghongen/tmp/pocketxmol_empty_envelope_20260912/audit_empty_envelope.py
```

诊断输出为同目录audit.log与empty_envelope_audit.json；后者保存完整空口袋身份、标准受体数量、最小COM距离及C0／冻结C5选袋数量。首批真实实例包含6j3z/11、6j3z/22，其最近标准残基COM分别距配体12.726、16.348 Å，确实不满足E条件；原始受体最近原子距离分别2.610、2.714 Å，近处原子被标准残基过滤排除。完整计数待扫描完成后记录。

已向用户提出处理口径：只在E的训练及val/loss流跳过空口袋并记录具体实例，冻结文件保持不变；完整测试若遇空口袋则记录输入构造失败并保留分母；随后从官方权重在独立目录重新训练E，保留已有中心模型结果。当前尚待答复，没有修改Dataset、阈值、残基范围、原点、loss或训练参数。

## 计划与实现差异

已发现实现缺口：部分冻结训练实例的E受体为空，代码直接求均值并污染训练；两步GPU验收没有覆盖这些实例。空E训练／监督验证的处理尚未在契约定义，等待用户决定后再作最小修改。E有效训练、best选择和测试评价均未完成，不把异常运行计作有效实验。
