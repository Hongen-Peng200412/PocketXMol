# 正确中心T0-RA的测试推理与评价

本记录承接 [中心T0修复与重训](02-T0中心契约修复与重训.md)。正确训练的best为21600步，原C0 val/loss=1.7970343828201294；训练在31200步第三次学习率下降后正常完成。按用户2026-09-11最新纠正，每个模型训练后立即完成测试集推理、CPU评价及结果记录，再启动下一模型。训练中原val/loss、调度及best选择保留；训练后完整验证集采样和评价取消。

## 当前测试配置与正式命令

正式配置为 `configs/docking/sample-B-C-T0-RA-test.yml`，读取 `/storage/penghongen/PocketXMol/training/B-C-T0-RA-C0/train_config/B-C-T0-RA.yml` 及同根 `checkpoints/step=21600.ckpt`。使用RA、center_translation=false和C0／C5：C5保留清单中的冻结给定中心，执行T0原采样；整条轨迹口袋和原点固定，首步纯高斯，最终只加回原点一次。原self-ranking、置信度、碰撞及立体化学项不变，GT不参与排名。

测试ALL包含446个实例、77个PDB；CAP10／HF10_TO5分别为272／227个实例，三视图共用候选池。每实例每协议50候选、100步，两协议共44600条轨迹。batch_size由25改为50，每实例50候选一次组批；失败不补生成，完整有效清单不按表现删样本。

在已授权371591、单张A800的独立release中执行：

```bash
bash 训练与运行/sh/sample_docking.sh B-C-T0-RA-test
```

C0和C5的全部测试实例均尝试完毕后，再从服务器项目根提交8核CPU评价：

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

必要代码验收及历史短GPU测试见日志00、02，不能作为正式测试成绩。本模型完整测试和CPU评价尚在执行范围内，下一模型未启动。

## 并行任务边界

用户2026-09-11说明，任务 `01a08f70-63cc-72e3-9d7a-4a04421931f2` 可能独立完成F-5（B-C-T1-RB）、F-6（B-E-T0-RB）及官方对照。F-5／F-6分别计划使用单张H100、32核和after_hold；官方使用374480_0对应的实际JobId=377521、gnode08、单张A100、8核。上述是用户提供的分工与资源信息，是否已开始或完成须另核对实际进程和产物。

本任务继续使用371591完成当前T0测试，再按既定顺序推进后续模型；到达可能重叠的模型或官方对照前，先核对另一任务状态，避免重复运行。不操作其作业；其新增或修改的工作区文件原样保留，不纳入本任务提交。当前发现的F-5.yml、F-6.yml、sample-official-377521-test.yml及相应三份日志均属于另一代理。若出现实际冲突，按用户后续说明处理。
