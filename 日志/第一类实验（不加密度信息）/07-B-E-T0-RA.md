# B-E-T0-RA训练与配套测试

本文件记录六个无密度模型中的第3个实验，依据 [科学契约](../../想法/方案草稿/9-8-科学契约.md)、[工程细节](../../想法/方案草稿/9-8-工程与实现细节.md) 和 [边界清单](../../想法/方案草稿/9-8-边界与核查清单.md)。按用户授权，在371591完成 [B-C-T1-RA测试报告](05-B-C-T1-RA推理与评价.md) 后，使用同一A800运行本模型训练、E测试、CPU评价与记录，再进入第4个B-C-T0-RB。

首次正式训练因空E口袋导致NaN而停止，异常产物全部保留。用户已接受空E处理建议，最小修复及必要审查验收通过；2026-09-12已在371591从官方权重重新训练，独立目录B-E-T0-RA-nonemptyE，W&B为lukfzmf3，after_lock保留。尚无本模型有效best或正式E测试结果。

## 固定训练条件

配置为 `configs/docking/B-E-T0-RA.yml`，科学参数和共同资产保持不变。首次异常训练使用ec06dbd运行副本 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_fa0d957b2d3f/PocketXMol`；重新训练使用该基线加下文经过审查验收的空E最小修复。

包络口袋选择残基重原子质量中心到任一完整配体重原子距离严格小于10 Å的完整受体残基；模型原点为实际输入受体重原子世界坐标的算术均值，配体与受体共同减去该原点。RA在联合受体图中共享编码器，保留核酸特征投影；center_translation=false，始终使用原dock高斯噪声链。训练与原val/loss均使用E条件，不抽新增中心偏移。

从规定官方pocketxmol.ckpt只加载模型参数，不传--resume；独立建立AdamW及调度状态，保留原loss、置信度头和训练目标。单卡batch_size=72、累积1、bf16、15个数据worker，全局名义批量72。初始lr=1e-4、warmup=0，每800次优化器更新计算原val/loss；Plateau采用相对阈值1%、patience=5、factor=0.2，第三次实际下降立即停止，最多40000步。best取最低原验证损失，保留last及全部定期检查点。

## 首次运行的命令与产物（已停止）

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

诊断输出为同目录audit.log与empty_envelope_audit.json；后者保存完整空口袋身份、标准受体数量、最小COM距离及C0／冻结C5选袋数量。本地完整副本为 `tmp/pxm-20260912/empty_envelope_audit.json`。Slurm步骤371591.2于节点时间16:47:13至17:05:47扫描65290个训练实例与781个验证实例，共8092个PDB；空E为训练46个、验证0个，E有效训练实例数为65244，验证仍为781。没有读取测试集。

全部46个空E训练实例如下，编号是原始candidate_id，也就是冻结清单中的配体occurrence编号；RA与RB共用同一标准受体范围和E选袋规则。

| PDB | occurrence编号 | 实例数 |
|---|---|---:|
| 6j3z | 11、22、34、171、197 | 5 |
| 6j40 | 31、151、189、226、238、256、273、384、403、421、472 | 11 |
| 7vh5 | 2、3、4、5、6、7、8、9、11、21、22、23、24、25、26、37、38、39、40、41、42、43 | 22 |
| 8dn2 | 7 | 1 |
| 8dn5 | 25 | 1 |
| 8j5k | 60、73、86、240、253、266 | 6 |

6j3z/11、6j3z/22的最近标准残基COM分别距配体12.726、16.348 Å，确实不满足E条件；原始受体最近原子距离分别2.610、2.714 Å，近处原子被标准残基过滤排除。以上是数据流跳过规则的完整身份审计，不是新建或替换冻结筛选清单。

用户已明确接受处理口径：只在E的训练及val/loss流跳过空口袋并记录具体实例，冻结文件保持不变；完整测试若遇空口袋则记录输入构造失败并保留分母；随后从官方权重在独立目录重新训练E，保留已有中心模型结果。三份契约及docking/README已按这项决定回填，原10 Å阈值、标准残基范围、有效E原点、loss和训练参数不变。

最小实现只修改docking/dataset.py：明确的EmptyEnvelopePocketError在RA/RB空E求均值前抛出；迭代器仅在train／validation捕获该错误并记录警告，组批前跳过。警告保存划分及实例身份，不统计每次拒绝抽样次数；其他异常仍传播。正式采样继续直接索引records，由已有preprocess错误路径保存预算内候选失败，不修改采样器或评价器。新增非测试构造验收覆盖RA/RB、训练完整batch、有限验证、冻结文件不变、同一配体实例选得空中心口袋时的有限给定原点，以及正式采样失败与评价分母。

提交独立核查前，主代理完成两遍自查。第一遍按dataset.py、test_docking_data.py、test_docking_sampling.py顺序核对异常类、__getitem__与__iter__职责及实际调用关系，确认没有新科学开关、嵌套包装或对其他异常的吞掉；训练索引随机流与有限worker分片保持原定义。第二遍按注释skill及示例核对空坐标、原点、实例索引、train／validation跳过边界、正式直接索引及失败分母的类／方法／代码说明，补齐当前契约文档。Python AST解析通过，正式模型、loss与采样噪声代码没有修改。

CPU验收以原fa0d957b2d3f运行副本为基线，在隔离临时副本 `/storage/penghongen/tmp/pocketxmol_empty_envelope_fix_20260912/PocketXMol` 中仅覆盖本次修改的Dataset及两份测试文件；不吸收共享工作区其他改动。首次通过长命令参数传送文件时Windows在建立SSH进程前报“文件名或扩展名太长”，没有执行远端写入；随后使用统一helper已有的 `-InputFile` 将LF脚本通过stdin发送，未修改helper。验收命令与日志如下，均不属于正式实验：

```bash
srun --jobid=371591 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 bash /storage/penghongen/tmp/pocketxmol_empty_envelope_fix_20260912/PocketXMol/ops/run_docking_checks.sh tests/test_docking_data.py tests/test_docking_sampling.py
```

本次验收输出为 `/storage/penghongen/tmp/pocketxmol_empty_envelope_fix_20260912/cpu_checks.log`，结果为22 passed、9 warnings，用时33.58秒。完整空E扫描同时使用该作业另外8核CPU，两个CPU步骤合计不超过已分配的16核。

完整扫描结束后，在371591.4使用同一隔离副本完成真实训练输入GPU验收。一次任务脚本为本地 `tmp/pxm-20260912/check_empty_envelope_gpu.py`，远端为 `/storage/penghongen/tmp/pocketxmol_empty_envelope_fix_20260912/check_empty_envelope_gpu.py`。只抽取冻结train中的空E实例6j3z/11与非空5ftl/0，放入两份临时验收清单，分别模拟无限训练与有限val/loss；不使用真实验证或测试样本，也不改写冻结文件。以下为验收命令，不属于正式训练：

```bash
srun --jobid=371591 --overlap --nodes=1 --ntasks=1 --cpus-per-task=16 bash -c 'ulimit -Sn 65536; cd /storage/penghongen/tmp/pocketxmol_empty_envelope_fix_20260912/PocketXMol; exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/storage/penghongen/tmp /storage/penghongen/PocketXMol/runtime/venv/bin/python -B /storage/penghongen/tmp/pocketxmol_empty_envelope_fix_20260912/check_empty_envelope_gpu.py'
```

RA、RB分别从官方权重执行8次真实优化器更新，均为72×1、bf16、15个worker；每批72个有效实例，全部为5ftl/0，空E实例在组批前被跳过。每次更新前的监督坐标、带噪坐标与原点均有限，每次反向后的梯度及最终参数均有限；末次原路径val/loss分别为1.163779、1.168035。fit耗时分别15.99、15.16秒，峰值已分配显存分别25660605952、25687365632字节。输出为同临时根gpu_checks.log与gpu_gate/results.json，均含EMPTY_E_GPU_GATE_PASS；未创建正式检查点或W&B运行。日志中的无logger提示来自验收明确关闭W&B，不属于数值错误。

三类独立审查各完成两轮全面核查，均通过，布局审查已批准异常类和既有方法中的修改。第一轮将测试注释中的“空受体”改为“选得空口袋”，并消除本记录“已完成两遍自查”与“仍需自查”的矛盾；第二轮修正CPU验收副本的覆盖方向说明，随后仅对此句窄核通过。审查引起的修改仅涉及说明文字，没有扩大科学行为或生产代码范围。

## 独立重训的正式命令与产物

最小修复和必要验收通过后，371591继续使用1张A800与16核CPU，新的正式命令为：

```bash
bash 训练与运行/sh/train_docking.sh B-E-T0-RA --logdir /storage/penghongen/PocketXMol/training/B-E-T0-RA-nonemptyE
```

独立训练根为 `/storage/penghongen/PocketXMol/training/B-E-T0-RA-nonemptyE/`，接入前已核对不存在。仍读取原B-E-T0-RA.yml，从规定官方权重加载模型参数，不传--resume；优化器、调度状态和W&B run id均重新建立。旧B-E-T0-RA目录及so6e0mvy不覆盖、不续训。

修复提交为c9cc4a1。正式源码release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_43742cdf8166/PocketXMol`，由原ec06dbd运行副本加入本次Dataset、两份测试及对应文档构成，不吸收共享工作区外来文件。发布前确认Dataset与实际GPU验收副本字节一致，测试AST一致，训练配置仍与原配置字节相同。临时发布源位于 `/storage/penghongen/tmp/pocketxmol_empty_envelope_release_20260912/PocketXMol`。

2026-09-12 master时间17:24:37保存独立启动记录与动态命令后，移除已获准try_lock，371591控制器第18次执行开始；after_lock保留，本次接入没有使用kill_lock。实际主进程39120，另有15个数据worker，均在job_371591；没有--resume，命令行末次--logdir明确指向新根。实际launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/train_B-E-T0-RA_nonemptyE_job371591_20260912T172127`，节点时钟约慢3分钟，不混用两个主机时间估算耗时。

启动记录为 `/storage/penghongen/PocketXMol/control/371591/train_B-E-T0-RA_nonemptyE_start.json`，新动态命令副本为同目录train_B-E-T0-RA_nonemptyE_run_cmd.sh；修改前命令另存train_B-E-T0-RA_before_nonemptyE_run_cmd.sh。累计out／err起点为83348997／139462。W&B新运行是[lukfzmf3](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/lukfzmf3)，名称仍为B-E-T0-RA，独立id与目录明确区别于异常so6e0mvy；本地记录为新训练根下wandb/run-20260912_172138-lukfzmf3。

正式启动检查已观察到161次优化器更新，约1.09步／秒，lr=1e-4；原损失和置信度损失均有限，没有Traceback、OOM、reduce_batch或空均值警告。日志已明确记录6j40/273、6j3z/22等空E跳过，规则确实进入正式数据流。这里只确认重训正常开始，完整训练、best选择和E测试仍须继续。

## 计划与实现差异

已发现并经用户确认处理的实现缺口：部分冻结训练实例的E受体为空，原代码直接求均值并污染训练；此前两步GPU验收没有覆盖这些实例。当前已按批准口径实施最小修复，两遍自查、两轮三类独立审查、CPU回归及真实训练输入GPU验收已通过。E有效训练、best选择和测试评价均未完成，不把异常运行计作有效实验。
