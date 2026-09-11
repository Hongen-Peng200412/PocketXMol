# 正确中心T0-RA的测试推理与评价

本记录承接 [中心T0修复与重训](02-T0中心契约修复与重训.md)。正确训练的best为21600步，原C0 val/loss=1.7970343828201294；训练在31200步第三次学习率下降后正常完成。按用户2026-09-11最新纠正，每个模型训练后立即完成测试集推理、CPU评价及结果记录，再启动下一模型。训练中原val/loss、调度及best选择保留；训练后完整验证集采样和评价取消。

本模型的测试推理和CPU评价现已全部完成。ALL视图按实例等权的Top-1成功率为C0 60.09%、C5 43.05%；两协议各446个实例、22300个候选，全部取得RMSD和self-ranking，无生成或候选评价失败。完整结果见下表及 [W&B评价记录](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/v4jilbdr)。

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

W&B评价使用 `pencounkdual-111/PocketXmol_raw`、名称B-C-T0-RA-C0_test；在本地评价结果落盘后创建汇总run v4jilbdr并成功上传。本模型测试记录收口后，进入已登记的B-C-T1-RA。

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

必要代码验收及历史短GPU测试见日志00、02，不能作为正式测试成绩。下述完整测试推理及CPU评价使用正式预算，已正常完成。

## 完整测试采样结果

2026-09-11，控制器第13次执行正常结束，采样进程42558退出，try_lock恢复、after_lock保留、kill_lock不存在。C0和C5分别覆盖完整446个测试实例，每个实例50个候选、100步，共44600个候选全部生成成功。此处“生成成功”仅指采样产物成功写出，姿态RMSD和self-ranking效果由后续CPU评价给出。

| 协议 | 实例数 | 成功候选数 | 生成失败数 | 逐实例总耗时合计（秒） | 逐实例平均耗时（秒） | 峰值张量显存（GiB） |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 446 | 22300 | 0 | 13341.13 | 29.91 | 2.99 |
| C5 | 446 | 22300 | 0 | 13354.37 | 29.94 | 2.99 |

两个协议各完成44600次批量forward，每个实例均为一批50个候选。逐实例总耗时合计约7小时25分钟，包括实例读取和落盘；不把两台主机的墙钟时间相减作为耗时。未增加候选或更改科学配置。

完成核对记录为 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T0-RA_test_complete_20260911.json`。按冻结test.jsonl的(pdb_id, candidate_id)与结果中的(pdb_id, occurrence_id)对应，已核对实例集合完全相同、全部完成标记、50×100预算、batch50、冻结种子、C5向量、测试视图及三种候选文件存在；run.json确认使用21600步best、RA和T0。该只读产物核对及记录不重新采样，也不替代正式CPU评价。

## CPU评价执行

Job 378916已在cnode02以单节点8核CPU正常完成，退出码0。Slurm记录开始于2026-09-11 21:58:14、结束于23:52:48，墙钟耗时1小时54分34秒、累计CPU时间14小时59分16秒，最大常驻内存约7.11 GiB。实际release复用 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_fa0d957b2d3f/PocketXMol`，与采样相同。launch为 `/home/penghongen/Feedback/PocketXMol/launches/378916/evaluate_docking_job378916_20260911T215541_a1`，调度日志在 `/home/penghongen/Feedback/PocketXMol/allocations/378916/`；正式动态命令副本保留在launch。CPU作业按提交时未启用after_hold的设置自动结束，371591的after_lock继续保留。

C0／C5各446份assessment.json和各22300个候选的指标记录均已核对：生成数、有效RMSD数、有效评分配对数完全一致；asset_error及候选错误为空，未触发RMSD按原子编号匹配的回退。所有失败仍按既定规则进入分母，本次实际失败数为0。两协议的occurrences.json、summary.json以及test/summary.json完整落盘；W&B状态为online_completed。

## 测试结果

RMSD单位为Å，沿原CalcRMS计算，不移动或刚体对齐预测姿态；成功阈值严格小于2 Å。Top-1使用原self-ranking最高的候选；Top-5在评分最高的5个候选内取最低RMSD；oracle在全部50个候选内取最低RMSD，用来表示候选池的可达结果，不用于选择实际输出。三个测试视图复用同一候选池。

以下成功率按实例等权，括号中为成功实例数；平均RMSD也按实例等权。ALL、CAP10、HF10_TO5的PDB数分别为77、67、65。

| 协议 | 视图 | 实例数 | Top-1成功率 | Top-5成功率 | Oracle成功率 | Top-1平均RMSD |
|---|---|---:|---:|---:|---:|---:|
| C0 | ALL | 446 | 60.09%（268） | 72.42%（323） | 82.29%（367） | 2.994 |
| C0 | CAP10 | 272 | 48.16%（131） | 64.34%（175） | 77.57%（211） | 3.620 |
| C0 | HF10_TO5 | 227 | 47.58%（108） | 64.32%（146） | 78.41%（178） | 3.538 |
| C5 | ALL | 446 | 43.05%（192） | 53.81%（240） | 61.66%（275） | 4.108 |
| C5 | CAP10 | 272 | 33.09%（90） | 42.65%（116） | 52.57%（143） | 4.746 |
| C5 | HF10_TO5 | 227 | 32.60%（74） | 42.73%（97） | 52.42%（119） | 4.788 |

PDB等权先在每个PDB内部平均其测试实例的成功指标，再对该视图中的PDB等权平均，避免含多个配体实例的PDB获得更大权重。

| 协议 | 视图 | Top-1成功率 | Top-5成功率 | Oracle成功率 |
|---|---|---:|---:|---:|
| C0 | ALL | 57.35% | 72.06% | 83.85% |
| C0 | CAP10 | 55.32% | 69.57% | 82.16% |
| C0 | HF10_TO5 | 53.97% | 68.79% | 81.72% |
| C5 | ALL | 42.50% | 51.23% | 58.47% |
| C5 | CAP10 | 38.99% | 46.97% | 56.75% |
| C5 | HF10_TO5 | 37.23% | 46.03% | 55.52% |

同一T0模型在冻结C5定位条件下的ALL Top-1成功率比C0低17.04个百分点。逐实例配对后，有91个实例由C0成功变为C5失败，15个由失败变为成功。C5会同时改变选袋和局部原点。当前结果描述T0在两种定位输入下的表现，尚不能比较T1或RA／RB。

### 评分与姿态误差的对应

每个实例先用其50个候选计算self-ranking与负RMSD的Spearman相关系数，再对实例等权平均或取中位数。Pose AUC同样逐实例计算，以RMSD<2 Å为正类；全部候选只有一个类别时AUC无定义，这些实例保留在成功率分母中，但不进入AUC均值。所有实例的Spearman均有效。

| 协议 | 视图 | Spearman均值 | Spearman中位数 | Pose AUC均值 | AUC有效实例／全部实例 |
|---|---|---:|---:|---:|---:|
| C0 | ALL | 0.2155 | 0.2176 | 0.6535 | 316／446 |
| C0 | CAP10 | 0.2104 | 0.1976 | 0.6603 | 195／272 |
| C0 | HF10_TO5 | 0.2312 | 0.2356 | 0.6709 | 169／227 |
| C5 | ALL | 0.2085 | 0.2078 | 0.6826 | 249／446 |
| C5 | CAP10 | 0.1986 | 0.1852 | 0.6855 | 135／272 |
| C5 | HF10_TO5 | 0.1962 | 0.1832 | 0.6862 | 113／227 |

ALL中，C0有99个实例、C5有83个实例的候选池包含成功姿态，但Top-1没有选中。此处仅记录既定self-ranking的表现，不改变评分规则或增加评分器实验。

### 对接结果与口袋核酸占比

核酸占比为当前协议选出的标准RNA／DNA重原子数，除以标准蛋白与RNA／DNA重原子总数；因此同一实例在C0、C5下的比例可以不同。两个协议均有439个纯蛋白口袋和7个含核酸口袋；这7个实例全部属于三个测试视图。

| 协议 | 口袋分组 | 实例数 | Top-1成功数 | Top-5成功数 | Oracle成功数 |
|---|---|---:|---:|---:|---:|
| C0 | 核酸占比为0 | 439 | 264 | 317 | 360 |
| C0 | 核酸占比大于0 | 7 | 4 | 6 | 7 |
| C5 | 核酸占比为0 | 439 | 190 | 238 | 272 |
| C5 | 核酸占比大于0 | 7 | 2 | 2 | 3 |

下表完整列出7个含核酸实例。每个RMSD单元格依次为Top-1／Top-5／oracle，单位Å；实例编号均为0。

| PDB | C0核酸占比 | C0 RMSD | C5核酸占比 | C5 RMSD |
|---|---:|---|---:|---|
| 9q16 | 19.43% | 0.946／0.946／0.888 | 10.05% | 3.045／3.045／2.462 |
| 9r3d | 12.23% | 2.804／2.804／1.551 | 4.69% | 5.936／5.005／4.466 |
| 9shy | 20.92% | 2.909／1.334／1.334 | 24.70% | 3.702／3.640／3.472 |
| 9v7o | 100.00% | 5.972／1.211／1.211 | 100.00% | 3.318／3.318／1.284 |
| 9z2n | 19.31% | 0.953／0.920／0.881 | 14.74% | 0.970／0.911／0.800 |
| 9z2u | 12.01% | 1.088／0.781／0.781 | 32.28% | 6.957／2.418／2.072 |
| 9z3d | 11.52% | 0.982／0.811／0.780 | 18.34% | 1.283／1.114／1.114 |

纯RNA的9v7o/0在两个协议下均完成50个候选和全部评价，并计入正式分母；C0的Top-5及oracle成功，C5仅oracle成功。含核酸样本只有7个，以上逐实例结果用于描述和后续比较，不能单凭这组样本确定核酸比例与成功率的稳定关系。

### 结果保存与评价限制

完整逐候选指标位于test/<协议>/<pdb_id>/<occurrence_id>/candidate_metrics.json，逐实例结果位于assessment.json；两个协议的occurrences.json与summary.json，以及test/summary.json保存全部汇总字段。W&B v4jilbdr已完成在线上传。用于本地整理报告的副本在 `tmp/pxm-20260912/B-C-T0-RA-test-results.json`，来源是上述服务器汇总及逐实例文件，正式产物仍以服务器目录为准。

评价日志有624条RDKit“暂不支持allene-style立体化学并忽略”的提示，没有Traceback；提示次数不代表独立异常实例数。当前立体化学项沿用原RDKit实现，对该类立体化学的区分能力受其支持范围限制。本次未因此更改分母、图结构或评价规则。

## 工作区修改边界

其他代理的工作区修改原样保留，不纳入本任务提交。本任务按自己的goal依次完成训练、测试与结果记录；此前额外安排的另一任务状态跟踪已按用户要求撤回。
