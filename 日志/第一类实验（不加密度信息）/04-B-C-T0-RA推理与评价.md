# 正确中心T0-RA的测试推理与评价

本记录承接 [中心T0修复与重训](02-T0中心契约修复与重训.md)。正确训练的best为21600步，原C0 val/loss=1.7970343828201294；训练在31200步第三次学习率下降后正常完成。按用户2026-09-11最新纠正，每个模型训练后立即完成测试集推理、CPU评价及结果记录，再启动下一模型。训练中原val/loss、调度及best选择保留；训练后完整验证集采样和评价取消。

## 当前测试配置与正式命令

正式配置为 `configs/docking/sample-B-C-T0-RA-test.yml`，读取 `/storage/penghongen/PocketXMol/training/B-C-T0-RA-C0/train_config/B-C-T0-RA.yml` 及同根 `checkpoints/step=21600.ckpt`。使用RA、center_translation=false和C0／C5：C5保留清单中的冻结给定中心，执行T0原采样；整条轨迹口袋和原点固定，首步纯高斯，最终只加回原点一次。原self-ranking、置信度、碰撞及立体化学项不变，GT不参与排名。

测试ALL包含446个实例、77个PDB；CAP10／HF10_TO5分别为272／227个实例，三视图共用候选池。每实例每协议50候选、100步，两协议共44600条轨迹。batch_size由25改为50，每实例50候选一次组批；失败不补生成，完整有效清单不按表现删样本。

在已授权371591、单张A800的独立release中执行：

```bash
bash 训练与运行/sh/sample_docking.sh B-C-T0-RA-test
```

C0和C5的全部测试实例均尝试完毕后，从本次采样release的PocketXMol根目录提交8核CPU评价；源码和配置沿用生成这些候选时的ec06dbd版本：

```bash
bash 训练与运行/submit_task.sh --sh evaluate_docking.sh --resource cpu --cpus 8 -- B-C-T0-RA-test
```

以上是当前正式命令。测试输出为 `/storage/penghongen/PocketXMol/sampling/B-C-T0-RA-C0/test/`，按C0／C5和实例分目录；模型名末尾的C0目录标记正确训练来源，不限制测试给定中心条件。保留poses.sdf、candidates.json、confidence.npz、result.json；CPU评价增加candidate_metrics.json、assessment.json、occurrences.json及summary.json。字段定义见docking/sampling.py与docking/evaluation.py。

W&B评价使用 `pencounkdual-111/PocketXmol_raw`、名称B-C-T0-RA-C0_test；在本地评价结果落盘后创建汇总run。完成CPU评价、失败记录和结果汇报后才启动已登记但尚未提交的B-C-T1-RA。

## 已取消的完整验证集采样

此前d788edb依据当时文档建立validation／test两份配置，完成两遍配置自查、原make_config解析和独立逻辑代理窄核后，于371591启动完整validation。这一训练后完整验证集工作偏离用户普遍意图，已按用户最新要求停止，不再继续采样或提交评价。原训练期间的val/loss不受影响。

历史采样release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_b3b44f062a12/PocketXMol`，launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/sample_B-C-T0-RA_validation_job371591_20260911T020258`。原启动记录及动态命令在 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T0-RA_validation_start.json` 和sample_B-C-T0-RA_validation_run_cmd.sh。master与gnode09时钟约有3分钟差异，保留各自原值，耗时不跨主机时间相减。

停止前通过/proc/19347/cmdline核对实际配置为sample-B-C-T0-RA-validation.yml，并通过cgroup确认属于job_371591。首次使用ps进行精确命令比较因默认显示宽度截断而未匹配，没有写锁或改动任务；改用/proc完整命令核实后，于gnode09时间2026-09-11 14:16:43创建实际kill_lock。控制器明确终止进程组19347，第12次执行退出137，随后恢复try_lock，after_lock保留、kill_lock已由控制器处理、旧进程消失。未scancel、释放资源、删除或覆盖检查点和候选。

停止记录为 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T0-RA_validation_stop_20260911.json`。已完成标记共有C0 781个实例／39050个成功候选、C5 436个实例／21800个成功候选；被中断实例的未完成文件也原样保留。已有validation产物仅为历史记录，不继续完成或评价，不作为当前正式测试成绩。

## 测试接入与检查

用户随后允许推理batch_size=50作为普遍选项，更大的有效批量能提速时可用100。现有sample_occurrence逐实例执行，每实例预算为50，所以配置100实际也只组一批50，单改数字不会提速。当前T0及官方正式测试配置均改为50，保持各自原科学配置。尚未增加跨实例组批实现。

ec06dbd保存上述两份测试配置的批量修改。本地解析并与HEAD前一版比较，除batch_size从25变50外全部值一致，50候选和100步不变；该配置检查只通过Python stdin读取YAML，没有采样或评价。先前已审查的Python执行逻辑没有变化，不重复整轮代码审查或GPU门控。

非删除同步成功、确认旧进程停止且after_lock和try_lock均在后，已在同一371591接入正式测试命令。启动元数据为 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T0-RA_test_start.json`，动态命令为同目录sample_B-C-T0-RA_test_run_cmd.sh；master记录请求时间14:23:03，本次out／err起点57173771／73688。配置明确21600步best、RA、T0、C0／C5、batch50及50×100预算。

控制器第13次执行的实际release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_fa0d957b2d3f/PocketXMol`，launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/sample_B-C-T0-RA_test_job371591_20260911T142022`。test/run.json在严格加载模型成功后写入，记录同一best及batch50。首三个实例11jb/0、1、2均50／50成功、100步，每实例实际完成100次批量forward；总耗时分别34.93、33.49、33.53秒，峰值张量显存约2.50 GB。与batch25时相比，组批次数由2变1；不同实例的耗时不能直接用来估算加速比例。after_lock保留，正式测试期间try_lock不存在、kill_lock不存在。

必要代码验收及历史短GPU测试见日志00、02，不能作为正式测试成绩。完整测试采样已正常完成，CPU评价随后执行，下一模型未启动。

## 完整测试采样结果

2026-09-11，控制器第13次执行正常结束，采样进程42558退出，try_lock恢复、after_lock保留、kill_lock不存在。C0和C5分别覆盖完整446个测试实例，每个实例50个候选、100步，共44600个候选全部生成成功。此处“生成成功”仅指采样产物成功写出，姿态RMSD和self-ranking效果由后续CPU评价给出。

| 协议 | 实例数 | 成功候选数 | 生成失败数 | 逐实例总耗时合计（秒） | 逐实例平均耗时（秒） | 峰值张量显存（GiB） |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 446 | 22300 | 0 | 13341.13 | 29.91 | 2.99 |
| C5 | 446 | 22300 | 0 | 13354.37 | 29.94 | 2.99 |

两个协议各完成44600次批量forward，每个实例均为一批50个候选。逐实例总耗时合计约7小时25分钟，包括实例读取和落盘；不把两台主机的墙钟时间相减作为耗时。未增加候选或更改科学配置。

完成核对记录为 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T0-RA_test_complete_20260911.json`。按冻结test.jsonl的(pdb_id, candidate_id)与结果中的(pdb_id, occurrence_id)对应，已核对实例集合完全相同、全部完成标记、50×100预算、batch50、冻结种子、C5向量、测试视图及三种候选文件存在；run.json确认使用21600步best、RA和T0。该只读产物核对及记录不重新采样，也不替代正式CPU评价。

## CPU评价执行

已按上面的正式评价命令提交Job 378916，申请单节点8核CPU、不申请GPU。作业已在cnode02运行；实际release复用 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_fa0d957b2d3f/PocketXMol`，与采样相同。launch为 `/home/penghongen/Feedback/PocketXMol/launches/378916/evaluate_docking_job378916_20260911T215541_a1`，调度日志和实际命令保存在 `/home/penghongen/Feedback/PocketXMol/allocations/378916/`。

本任务仅评价已完成的test候选，不生成新候选。逐实例CPU评价完成后汇总C0／C5下的ALL、CAP10、HF10_TO5，并将汇总上传到配置指定的W&B记录；当前尚未确认评价完成。371591的after_lock继续保留，B-C-T1-RA尚未启动。

## 工作区修改边界

其他代理的工作区修改原样保留，不纳入本任务提交。本任务按自己的goal依次完成训练、测试与结果记录；此前额外安排的另一任务状态跟踪已按用户要求撤回。
