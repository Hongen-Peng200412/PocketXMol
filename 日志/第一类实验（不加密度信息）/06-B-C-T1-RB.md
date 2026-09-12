# B-C-T1-RB训练与配套测试

本文件记录六个无密度模型中的第5个实验。用户2026-09-12新增授权：378693负责第5个B-C-T1-RB、第6个B-E-T0-RB及各自完整测试与评价；371591继续当前B-C-T1-RA及第3、4个实验。每张卡独立按训练、测试、评价与记录的顺序串行推进，评价使用该A800作业已分配的CPU。

依据为 [科学契约](../../想法/方案草稿/9-8-科学契约.md)、[工程细节](../../想法/方案草稿/9-8-工程与实现细节.md) 和 [边界清单](../../想法/方案草稿/9-8-边界与核查清单.md)。训练已在22400次优化器更新后因第三次实际学习率下降正常停止；原C5 val/loss最低检查点为17600步，损失2.835408。完整C0／C5各446实例、22300候选生成成功，已在同卡启动CPU评价。

## 固定训练条件

配置为 `configs/docking/B-C-T1-RB.yml`，来源为已审查的ec06dbd运行副本 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_fa0d957b2d3f/PocketXMol`。RB分别编码蛋白与核酸，初始化后把蛋白编码器参数复制给核酸编码器；保留原主体模型、loss、置信度头及训练目标。

中心T1训练每次抽一份delta=r*u，r均匀取0–5 Å、u为均匀球面方向；完整g+delta选袋与定原点，原高斯后增加s*delta，s=1-level_dict['pos']，监督目标与受体不跟随新增平移。原val/loss使用冻结C5偏移。正式测试保留C0与C5，两种输入都执行T1采样。

从规定官方pocketxmol.ckpt开始，不传--resume；独立建立AdamW与调度状态。batch_size=72、累积1、bf16、15个数据worker，名义全局批量72。lr=1e-4、warmup=0，每800次优化器更新验证；Plateau相对阈值1%、patience=5、factor=0.2，第三次实际下降立即停止，上限40000步。best按最低原验证损失选择，保留last和全部定期检查点。

## 资源核对与正式命令

378693的Slurm记录确认：RUNNING、gnode10、1张A800、16核CPU、--pre_hold及--after_hold。实际控制目录 `/home/penghongen/Feedback/Pocket_Plus/allocations/378693/`，pre_lock_378693在父目录。接入前节点只存在该作业等待pre_lock的控制器进程43832，无正在运行的计算任务，after_lock存在，try_lock与kill_lock不存在；无需终止旧计算。

正式训练命令在上述冻结release中执行：

```bash
bash 训练与运行/sh/train_docking.sh B-C-T1-RB
```

训练产物目录为 `/storage/penghongen/PocketXMol/training/B-C-T1-RB/`，接入前已确认不存在；W&B为 `pencounkdual-111/PocketXmol_raw`、名称B-C-T1-RB，实际run为[wmgkgurr](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/wmgkgurr)。新的动态命令与启动元数据保存于 `/storage/penghongen/PocketXMol/control/378693/`，实际launch见下文。

## 实际接入

首次启动前核对混用了本地LF归一化字节与服务器原CRLF字节，断言在创建控制记录及操作锁之前停止。重新核对原始字节，两端训练YAML完全相同，SHA256均为ef6aa7acef6ee88286abd65d4e9cc3afc66f899bfa273a22db4a12ef4692b325；没有修改训练配置或源码。

2026-09-12 master时间11:20:47，保存原动态命令及新正式命令后，移除获准的pre_lock_378693开始第一次执行，after_lock保留，没有使用kill_lock或scancel。实际训练主进程50944，cgroup确认属于378693。PocketXMol launch为 `/home/penghongen/Feedback/PocketXMol/launches/378693/train_B-C-T1-RB_job378693_20260912T111926`；节点时钟与master有差异，保留各自记录。

控制记录为 `/storage/penghongen/PocketXMol/control/378693/train_B-C-T1-RB_start.json`，命令副本train_B-C-T1-RB_run_cmd.sh，原命令副本original_run_cmd_before_B-C-T1-RB.sh。本次out／err起点136／0。实际源码来自上述fa0d957b2d3f的PocketXMol release，控制器外层Pocket_Plus release仅负责既有资源控制。运行时pre、try、kill锁均不存在，after_lock保留。

实际启动输出确认从规定官方pocketxmol.ckpt加载，CUDA启用、bf16混合精度、约20.7M参数全部可训练；W&B正常在线创建wmgkgurr，本地记录位于训练根wandb/run-20260912_112017-wmgkgurr。未传--resume，不承接其它训练参数或优化器状态。

启动验收时已进入80次优化器更新，约1.04步／秒，lr=1e-4，原loss和置信度损失正常记录；未发现Traceback、CUDA OOM、reduce_batch或NaN。此处仅确认正式训练正常进入更新，不代表训练完成或已选定best。

## 验收来源与后续步骤

该RB／T1代码及72×1资源配置已经完成两遍主代理自查、两轮全面独立审查和真实非测试样本GPU验收，见 [共同准备日志](00-实现与共同数据准备.md)；T0纠偏之后T1机制保持，见 [中心纠偏日志](02-T0中心契约修复与重训.md)。本次仅接入新增授权资源，未修改生产函数或科学配置，不重复模型试训。

训练正常结束后在Slurm内核实停止原因和实际best，再建立明确检查点的完整C0／C5 test配置，batch50、每实例50候选100步。推理完成后在同一378693执行既有evaluate_docking.sh，使用分配内CPU并保留after_lock，不另排纯CPU任务。完成本模型结果记录后再运行第6个B-E-T0-RB。不安排完整验证集采样或新增实验。

## 训练完成与best核对

正式训练输出记录updates=22400、stop_reason=plateau及Training finished。2026-09-12节点时间17:16:22，在378693.0的8核CPU内读取last与best，确认下降次数3、最后验证步22400、28个定期检查点和last均保留；W&B id仍为wmgkgurr。调度器和优化器末次学习率均为8e-7，第三次下降后没有再更新参数。best为 `checkpoints/step=17600.ckpt`，原C5 val/loss=2.8354082107543945，是全部定期损失的最小值；best包含1236个model参数键，所有参数有限。

只读核对脚本为 `/storage/penghongen/tmp/pocketxmol_checkpoint_20260912/inspect_t1_rb.py`，训练摘要保存为 `/storage/penghongen/PocketXMol/training/B-C-T1-RB/training_summary_20260912.json`，其中checkpoint_losses保存全部28个检查点及损失。以下为产物核对命令，不是正式训练或推理命令：

```bash
srun --jobid=378693 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /storage/penghongen/PocketXMol/runtime/venv/bin/python -B /storage/penghongen/tmp/pocketxmol_checkpoint_20260912/inspect_t1_rb.py
```

## 完整测试配置与正式命令

`configs/docking/sample-B-C-T1-RB-test.yml`明确读取本次保存的RB训练配置及17600步best。输入为446个测试实例的C0、冻结C5；两种条件均启用T1后续中心相关重新加噪，首步纯高斯。每实例每协议50候选、100步，batch50；输出为 `/storage/penghongen/PocketXMol/sampling/B-C-T1-RB/test/`，三个视图共用候选池。

正式采样命令如下；完成全部C0/C5推理后才执行下一条CPU评价命令，二者分别登记launch和进程。

```bash
bash 训练与运行/sh/sample_docking.sh B-C-T1-RB-test
```

```bash
bash 训练与运行/sh/evaluate_docking.sh B-C-T1-RB-test
```

评价使用378693已分配CPU中的8个工作进程，W&B为pencounkdual-111/PocketXmol_raw，名称B-C-T1-RB_test；实际run id在评价汇总连接W&B后记录。候选生成已经完成，完整姿态评价结果尚未产生。

主代理按配置读取顺序及注释规范完成两遍自查，使用本地项目环境的YAML解析核对：相对已运行的T1-RA测试配置，仅替换模型名、RB分支、实际best、对应训练配置和独立输出／W&B名称；其余字段完全一致。两轮独立配置审查均通过，没有修改采样器、评价器或科学预算。

## 完整测试实际接入

训练结束后的W&B后台进程曾因GraphQL接口超时而重试，主进程50944等待其收尾；节点时间17:22:40，后台流自行关闭，随后控制器第1次执行成功、try_lock恢复。确认50944及其W&B子进程51729均退出后才接入采样，没有为此使用kill_lock或更改W&B模式。

测试配置与训练完成记录提交为3001cf9；正式测试release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_c59a88a03678/PocketXMol`，由原ec06dbd源码仅加入本次测试配置和记录构成，临时发布源为 `/storage/penghongen/tmp/pocketxmol_t1_rb_test_20260912/PocketXMol`。未混入外来工作区文件，亦未改变中心模型代码。

2026-09-12 master时间17:26:58保存启动记录与动态命令后，移除获准try_lock，378693控制器第2次执行开始，after_lock保留。实际launch为 `/home/penghongen/Feedback/PocketXMol/launches/378693/sample_B-C-T1-RB_test_job378693_20260912T172535`；节点时钟约慢1分20秒，时间记录分别保留。启动元数据为 `/storage/penghongen/PocketXMol/control/378693/sample_B-C-T1-RB_test_start.json`，同目录保存新命令sample_B-C-T1-RB_test_run_cmd.sh与接入前命令sample_B-C-T1-RB_test_before_run_cmd.sh；累计out／err起点为25698979／8263。测试YAML的LF字节SHA256为2ac841d703635e00be10ad928140e96afd1f838ffdf0b779aae87ab3a0329969。

实际采样主进程26403，cgroup属于378693。推理已严格加载17600步best并写出test/run.json，记录中的RB、T1、C0/C5、50候选、100步、batch50及CPU8均与配置相同。启动检查时已有2个C0实例完成，例如11jb/1生成50个候选、失败0、执行100次模型前向，用时33.49秒，峰值已分配显存2538523648字节；未见Traceback或OOM。这里只确认正式推理正常开始，完整测试与CPU评价的完成情况将在后续记录。

## 推理耗时与资源观测

2026-09-12 master时间23:03检查时，C0已完成446实例、22300候选，C5完成134实例、6700候选，全部成功且没有OOM或重试。比较已完成C5的同一批134实例，其C0／C5累计单实例耗时分别为4050.03／6854.39秒，逐实例耗时比中位数为1.468；近期9hyu/22、31、106的C5耗时分别96.17、136.42、83.86秒，对应C0为35.39、49.09、26.67秒，口袋原子数相近或更少。候选预算、100次模型前向和配置均未变化。

gnode10时间23:05的只读核查确认：378693分配的是GPU IDX:0，UUID为GPU-0e253751-cce3-c71d-c2df-a222fec3759b；本作业仅有采样进程26403使用它，同一GPU另有3个作业外计算进程。只核对同卡进程数量和是否属于378693，没有追踪这些进程的任务内容，也没有对它们执行操作。观测时GPU利用率100%、P0、SM频率1410 MHz、温度69°C，降频原因掩码为0；本进程3秒内使用3.02秒CPU，I/O计数不变，没有磁盘等待。并发使用可能影响耗时，现有观测没有隔离其因果贡献。已向用户报告资源情况并询问是否继续当前推理或由用户安排资源调整；现有推理继续按已授权的预算运行，不更改科学配置、停止进程或操作其他任务。

## 全部候选生成完成与核对

2026-09-13核对时，控制器第2次执行已成功、主进程26403已退出，try_lock恢复且after_lock保留。C0与C5均完成446个实例、22300个候选，没有生成失败；本次采样日志区间没有Traceback、CUDA OOM或reduce_batch。

同一378693的8核CPU产物核对通过：实际配置与test/run.json一致，全部实例身份、object_key、视图、冻结种子和C5向量与test.jsonl相同；每实例50候选、100步、一批50、100次已完成批量forward。候选身份、编号、有限轨迹置信度、SDF编号和姿态／置信度文件存在性均通过核对。没有修改冻结清单或补生成候选。

| 协议 | 实例／候选 | 实例耗时合计（秒） | 推理耗时合计（秒） | 已完成批量forward | 峰值分配显存（字节） |
|---|---|---:|---:|---:|---:|
| C0 | 446／22300 | 13279.412 | 13208.046 | 44600 | 3234062848 |
| C5 | 446／22300 | 17366.621 | 17298.558 | 44600 | 3227094528 |

逐实例耗时合计约8小时30分46秒，不含模型启动和协议切换等开销。C5期间的同卡并发观测见上一节；不能将两协议耗时差全部归因于T1公式或口袋变化。

完成证据为 `/storage/penghongen/PocketXMol/control/378693/sample_B-C-T1-RB_test_complete_20260913.json`。只读核对脚本为 `/storage/penghongen/tmp/pocketxmol_sampling_20260913/inspect_t1_rb_sampling.py`，本地副本为 `tmp/pxm-20260913/inspect_t1_rb_sampling.py`。以下是产物核对命令，不是正式采样或评价命令：

```bash
srun --jobid=378693 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /storage/penghongen/PocketXMol/runtime/venv/bin/python -B /storage/penghongen/tmp/pocketxmol_sampling_20260913/inspect_t1_rb_sampling.py
```

## 同一作业内的CPU评价启动

完成候选核对后，2026-09-13 master时间02:56:31接入正式命令 `bash 训练与运行/sh/evaluate_docking.sh B-C-T1-RB-test`。配置和源码继续使用同一c59a88a03678 release，不重新生成候选或改动科学设置；378693已分配16核CPU，正式评价使用8个工作进程，入口关闭CUDA，不另申请CPU作业。

控制器第3次执行，实际评价主进程10865，cgroup属于378693，CUDA_VISIBLE_DEVICES为空。launch为 `/home/penghongen/Feedback/PocketXMol/launches/378693/evaluate_B-C-T1-RB_test_job378693_20260913T025524`。启动记录为 `/storage/penghongen/PocketXMol/control/378693/evaluate_B-C-T1-RB_test_start.json`，同目录保存新命令evaluate_B-C-T1-RB_test_run_cmd.sh及此前采样命令evaluate_B-C-T1-RB_test_before_run_cmd.sh；本次out／err起点25867448／8359。after_lock保留，完整评价和汇总仍在执行。

## 计划与实现差异

未发现本次T1-RB训练与批准契约的实质差异；训练与完整候选生成已完成，CPU评价及最终报告仍未完成。
