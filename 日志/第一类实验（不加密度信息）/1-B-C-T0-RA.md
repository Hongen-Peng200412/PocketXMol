# 1-B-C-T0-RA

**当前：修正配体芳香键编码的复验已获准，等待SMILES新入口验收；尚未开始训练。** 本次从官方pxm权重重新初始化模型、优化器和调度状态，采用公共精确SMILES图，不恢复旧模板编码的checkpoint。资源为378693／gnode10／A800／16CPU；训练bf16-mixed，推理官方FP32张量路径。

| 当前复验字段 | 已确定内容 |
|---|---|
| 配置 | configs/docking/B-C-T0-RA-SMILES.yml |
| 训练产物根 | /storage/penghongen/PocketXMol/training/B-C-T0-RA-SMILES |
| 测试协议 | C0/C5；每实例每协议50候选、100步 |
| best／W&B／测试指标 | 尚未产生；旧值见后半部分，不充当本次结果 |

## 正式运行命令

验收通过后执行的短入口为 `bash 训练与运行/sh/train_docking.sh B-C-T0-RA-SMILES`，当前未执行。训练结束读取实际best后分别记录采样和评价命令，不生成完整验证集候选。

## 验收与资源控制记录

2026-09-14 19:52，主代理在gnode10确认378693正在执行Matcher的smiles_identity.experiment，进程组13179，随后按授权创建实际kill_lock。控制器已终止该进程组并回到try_lock；after_lock保留，GPU0空闲。接管证据根为/storage/penghongen/tmp/pxm_formal_smiles_20260914/takeover-378693。接管不删除Matcher文件或checkpoint，原运行命令另有副本。新链验收命令后续记录于[SMILES日志](../预实验（一 --二之间）/1-SMILES构图与GPU验收.md)。

## 之前的尝试：旧配体编码结果

以下保留原实验的结果、源码和失败历史；其中“正确”“有效”描述当时的T0／口袋修复，不代表已对齐官方芳香键输入。

**正确重训、C0／C5完整测试和评价均已完成。** ALL 的 Top-1 成功率为60.09%／43.05%，两协议各446实例、22300候选全部生成并评价成功。有效训练在31200次更新后停止。

更新核查：2026-09-13 11:09（服务器 master，UTC+8）。依据 [科学契约](../../想法/方案草稿/9-8-科学契约.md)、[工程细节](../../想法/方案草稿/9-8-工程与实现细节.md) 和 [边界清单](../../想法/方案草稿/9-8-边界与核查清单.md)，本文件统一保存本模型的训练、测试、评价与尝试历史。共同准备见 [实现与共同数据准备](../实现与共同数据准备.md)，全实验进度见 [总日志](总日志&分析/总日志.md)。

## 当前有效运行与产物

| 项目 | 当前事实 |
|---|---|
| 训练产物根 | `/storage/penghongen/PocketXMol/training/B-C-T0-RA-C0/` |
| 正式测试检查点 | `checkpoints/step=21600.ckpt`；C0 val/loss=1.7970343828201294 |
| 训练 W&B | [ceqrh2ve](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/ceqrh2ve) |
| 测试与评价产物 | `/storage/penghongen/PocketXMol/sampling/B-C-T0-RA-C0/test/` |
| 评价 W&B | [v4jilbdr](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/v4jilbdr) |

训练保留原路径 `val/loss` 和最低损失检查点选择；训练结束后直接完整测试，不做训练后完整验证集采样。每实例每协议50候选、100步，原 self-ranking（原置信度及碰撞、立体化学项组成的候选排序）不变。完整结果优先放在下文，执行核查和失败尝试放在后部。

ALL为446个实例的完整测试集合。按完整模板身份object_key在ALL中的频数分组，只有频数大于10的身份才沿冻结顺序截取：CAP10保留前10个，HF10_TO5保留前5个；频数不超过10的身份全部保留。两个视图分别272和227个实例，与ALL复用候选，不重复生成或相加计数。

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

以上是本次已经执行的正式命令；独立CPU作业发生在09-12改用A800自带CPU的安排之前。测试输出为 `/storage/penghongen/PocketXMol/sampling/B-C-T0-RA-C0/test/`，按C0／C5和实例分目录；模型名末尾的C0目录标记正确训练来源，不限制测试给定中心条件。保留poses.sdf、candidates.json、confidence.npz、result.json；CPU评价增加candidate_metrics.json、assessment.json、occurrences.json及summary.json。字段定义见docking/sampling.py与docking/evaluation.py。

W&B评价使用 `pencounkdual-111/PocketXmol_raw`、名称B-C-T0-RA-C0_test；在本地评价结果落盘后创建汇总run v4jilbdr并成功上传。本模型测试记录收口后，进入已登记的B-C-T1-RA。

## 正常训练完成与最佳检查点

2026-09-11检查到本次正式训练正常结束，控制器第11次执行成功并恢复try_lock；371591及after_lock保留，kill_lock不存在。训练输出明确 `updates=31200, stop_reason=plateau`，fit进度显示耗时8小时8分8秒。训练期间读取本次启动字节之后的out／err，未发现OOM、reduce_batch裁批、Traceback或非有限loss记录。

01:50在既有371591内启动8核Slurm步骤，只在CPU读取本次last和best，不通过普通SSH会话加载模型。last保存global_step=31200、last_validation_step=31200、decline_count=3、stop_reason=plateau，优化器与调度器学习率均为8e-7，W&B run id仍为ceqrh2ve。这证明第三次实际下降已经记录，之后没有新增优化器更新；W&B最后一条train/lr=4e-6属于下降前的训练日志，不替代检查点状态。

best为 `/storage/penghongen/PocketXMol/training/B-C-T0-RA-C0/checkpoints/step=21600.ckpt`，原始C0验证损失为1.7970343828201294；已核对best内部global_step=21600，model参数键1130个。39个定期检查点和last均保留，最后一次验证损失为1.9864825010299683。该best仅依据本模型的原val/loss选择；当时正式姿态比较尚待共同C0/C5评价，现已完成，结果见本文前部，不能与旧错误C5验证损失直接比较。

完成状态与39次验证损失记录于 `/storage/penghongen/PocketXMol/training/B-C-T0-RA-C0/training_summary_20260911.json`。一次性检查脚本保存在 `/storage/penghongen/tmp/pocketxmol_checkpoint_20260911/inspect_corrected_t0.py`；以下是检查命令，不是正式训练或采样命令：

```bash
srun --jobid=371591 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /storage/penghongen/PocketXMol/runtime/venv/bin/python -B /storage/penghongen/tmp/pocketxmol_checkpoint_20260911/inspect_corrected_t0.py
```

训练结束期间用户继续整理evaluation.py的Docstring和sampling.py的空行。核对去除Docstring后的完整AST不变、B-C-T0-RA.yml解析值不变后，按其明确授权提交为3ccd5c1；该YAML仅行尾状态变化，没有实际提交差异。没有改变已完成训练的release。

## 新T0正式运行命令

通过两遍主代理自查、两轮独立审查和必要验收后，在同一371591的冻结release根运行：

```bash
bash 训练与运行/sh/train_docking.sh B-C-T0-RA --logdir /storage/penghongen/PocketXMol/training/B-C-T0-RA-C0
```

新产物根为 `/storage/penghongen/PocketXMol/training/B-C-T0-RA-C0/`；使用原入口已有的--logdir参数，不新增重训专用执行器。没有传--resume，从规定官方pxm模型参数初始化并重新建立优化器和调度状态，已创建独立W&B run。

本次修复版本为693c7ec。gnode09节点记录16:49:29接入既有371591控制器，16:49:45建立独立launch：`/home/penghongen/Feedback/PocketXMol/launches/371591/train_B-C-T0-RA-C0_job371591_20260910T164945`。实际冻结release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_9fa23b858837/PocketXMol`，W&B在线记录为 [pencounkdual-111/PocketXmol_raw/ceqrh2ve](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/ceqrh2ve)。这与旧错误运行9wkyn4qn及旧目录完全独立。

启动后日志明确加载规定官方pocketxmol.ckpt，没有恢复旧检查点；训练输出目录中src/docking/dataset.py及src/scripts/train_pl.py均包含本次C0纠偏，保存的train_config/B-C-T0-RA.yml为center_translation=false、batch_size=72、accumulate_grad_batches=1，且无resume字段。已观察到第24次更新、lr=1e-4，原loss正常计算，W&B已同步；首次800步验证尚未到达。此处仅记录正式启动事件，后续每次验证不重复写handoff。

动态命令保存在 `/storage/penghongen/PocketXMol/control/371591/train_B-C-T0-RA-C0_run_cmd.sh`，启动元数据为同目录train_B-C-T0-RA-C0_start.json；本次累计out／err起始字节为20880102／62752，后续检查只读取此后的新运行输出。after_lock保留、训练期间try_lock不存在，正常结束后由原控制器恢复try_lock。用户在运行版本冻结后自行提交的eaf1165仅为.gitignore增加/tmp/，已原样保留，不改写正在运行的release。

用户随后补充docking/sampling.py及models/sample.py的阅读说明，并移动pad_and_stack的定义位置。按其授权核对后保存为b1c9270：忽略Docstring、按同名函数出现次序逐一比较函数AST，并核对模块非函数语句，执行逻辑和签名均不变。明确SDF中的成功是完成生成流程，不是RMSD<2 Å；补充sample_index与sdf_index的失败后编号示例，限定100步单轮不触发多段分数分支。未重跑训练或改写运行release。

## 有效运行的执行与核查记录

本节按实际事件保留执行证据；其中启动阶段的进度与后续计划属于当时记录，当前完成状态以文档开头为准。

### 测试接入与检查

用户随后允许推理batch_size=50作为普遍选项，更大的有效批量能提速时可用100。现有sample_occurrence逐实例执行，每实例预算为50，所以配置100实际也只组一批50，单改数字不会提速。当前T0及官方正式测试配置均改为50，保持各自原科学配置。尚未增加跨实例组批实现。

ec06dbd保存上述两份测试配置的批量修改。本地解析并与HEAD前一版比较，除batch_size从25变50外全部值一致，50候选和100步不变；该配置检查只通过Python stdin读取YAML，没有采样或评价。先前已审查的Python执行逻辑没有变化，不重复整轮代码审查或GPU门控。

非删除同步成功、确认旧进程停止且after_lock和try_lock均在后，已在同一371591接入正式测试命令。启动元数据为 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T0-RA_test_start.json`，动态命令为同目录sample_B-C-T0-RA_test_run_cmd.sh；master记录请求时间14:23:03，本次out／err起点57173771／73688。配置明确21600步best、RA、T0、C0／C5、batch50及50×100预算。

控制器第13次执行的实际release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_fa0d957b2d3f/PocketXMol`，launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/sample_B-C-T0-RA_test_job371591_20260911T142022`。test/run.json在严格加载模型成功后写入，记录同一best及batch50。首三个实例11jb/0、1、2均50／50成功、100步，每实例实际完成100次批量forward；总耗时分别34.93、33.49、33.53秒，峰值张量显存约2.50 GB。与batch25时相比，组批次数由2变1；不同实例的耗时不能直接用来估算加速比例。after_lock保留，正式测试期间try_lock不存在、kill_lock不存在。

必要代码验收及历史短GPU测试见[共同准备](../实现与共同数据准备.md)及[第1模型的纠偏记录](1-B-C-T0-RA.md)，不能作为正式测试成绩。下述完整测试推理及CPU评价使用正式预算，已正常完成。

### 完整测试采样结果

2026-09-11，控制器第13次执行正常结束，采样进程42558退出，try_lock恢复、after_lock保留、kill_lock不存在。C0和C5分别覆盖完整446个测试实例，每个实例50个候选、100步，共44600个候选全部生成成功。此处“生成成功”仅指采样产物成功写出，姿态RMSD和self-ranking效果由后续CPU评价给出。

| 协议 | 实例数 | 成功候选数 | 生成失败数 | 逐实例总耗时合计（秒） | 逐实例平均耗时（秒） | 峰值张量显存（GiB） |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 446 | 22300 | 0 | 13341.13 | 29.91 | 2.99 |
| C5 | 446 | 22300 | 0 | 13354.37 | 29.94 | 2.99 |

两个协议各完成44600次批量forward，每个实例均为一批50个候选。逐实例总耗时合计约7小时25分钟，包括实例读取和落盘；不把两台主机的墙钟时间相减作为耗时。未增加候选或更改科学配置。

完成核对记录为 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T0-RA_test_complete_20260911.json`。按冻结test.jsonl的(pdb_id, candidate_id)与结果中的(pdb_id, occurrence_id)对应，已核对实例集合完全相同、全部完成标记、50×100预算、batch50、冻结种子、C5向量、测试视图及三种候选文件存在；run.json确认使用21600步best、RA和T0。该只读产物核对及记录不重新采样，也不替代正式CPU评价。

### CPU评价执行

Job 378916已在cnode02以单节点8核CPU正常完成，退出码0。Slurm记录开始于2026-09-11 21:58:14、结束于23:52:48，墙钟耗时1小时54分34秒、累计CPU时间14小时59分16秒，最大常驻内存约7.11 GiB。实际release复用 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_fa0d957b2d3f/PocketXMol`，与采样相同。launch为 `/home/penghongen/Feedback/PocketXMol/launches/378916/evaluate_docking_job378916_20260911T215541_a1`，调度日志在 `/home/penghongen/Feedback/PocketXMol/allocations/378916/`；正式动态命令副本保留在launch。CPU作业按提交时未启用after_hold的设置自动结束，371591的after_lock继续保留。

C0／C5各446份assessment.json和各22300个候选的指标记录均已核对：生成数、有效RMSD数、有效评分配对数完全一致；asset_error及候选错误为空，未触发RMSD按原子编号匹配的回退。所有失败仍按既定规则进入分母，本次实际失败数为0。两协议的occurrences.json、summary.json以及test/summary.json完整落盘；W&B状态为online_completed。

### 工作区修改边界

其他代理的工作区修改原样保留，不纳入本任务提交。本任务按自己的goal依次完成训练、测试与结果记录；此前额外安排的另一任务状态跟踪已按用户要求撤回。

## 实验过程与失败尝试

以下为追溯用记录。已结束阶段中的“尚未”“随后”等表述仅说明当时状态；本文件开头的当前状态优先。历史命令不应再次执行，旧产物不作为有效模型或测试结果。

### 旧错误T0训练

本记录的训练错误地把中心T0与C5定位绑定，训练每次偏移中心、val/loss使用冻结C5；用户明确的正确T0应使用C0训练和监督验证。已通过371591实际kill_lock协议停止，旧产物全部保留，不能作为正确T0基线，也不能用于新T0初始化或续训。[中心契约修复与重训记录](1-B-C-T0-RA.md)取代本记录的当前执行状态。

本实验是六个无密度模型中的第一个：中心口袋，T0保持原位置噪声，RA使用独立核酸投影和共同受体编码器。依据 [科学契约](../../想法/方案草稿/9-8-科学契约.md) 和 [工程细节](../../想法/方案草稿/9-8-工程与实现细节.md)；共同数据、环境和前置验收见 [准备记录](../实现与共同数据准备.md)。本记录只保存正式实验，不把前述短检查的权重或候选用作实验产物。

#### 来源与配置

- 实际旧配置：`/storage/penghongen/PocketXMol/training/B-C-T0-RA/train_config/B-C-T0-RA.yml`，以该副本和旧release源码为准，不能用修正后的同名工作区YAML解释旧运行。训练种子2023，错误地使用每次重采样C5训练、冻结C5和完整781个验证实例计算原val/loss。
- 冻结清单：`/storage/penghongen/PocketXMol/data/`，有效freeze作业376632；训练65290、验证781，校准361与测试446均不进入训练或val/loss模型选择。
- 初始权重：`/storage/penghongen/PocketXMol_official_test/extracted/data/trained_models/pxm/checkpoints/pocketxmol.ckpt`。只继承官方模型参数，新建优化器；全部主体和新增受体参数共同训练。
- 环境：`/storage/penghongen/PocketXMol/runtime/venv/bin/python`，继承pxm_phase1的Torch2.6.0、Lightning2.6.0和RDKit2023.9.3，项目层补充W&B0.21.1。
- 资源：已获准的371591，gnode09，一张A800、16核；按用户最新要求优先单卡72、累积1、有效全局72、bf16，15个DataLoader worker。训练子进程文件句柄软上限65536，TMPDIR为`/storage/penghongen/tmp`。
- AdamW，lr=1e-4、warmup=0，原loss和置信度目标。每800个优化器更新计算原val/loss，按1%相对改善、patience=5、factor=0.2的原Plateau下降；第三次实际下降停止，最多40000更新。best取原验证损失最低者，保留last及全部定期检查点。
- W&B：`pencounkdual-111/PocketXmol_raw`，名称B-C-T0-RA，online；实际run id为 `9wkyn4qn`，[在线记录](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/9wkyn4qn)。密钥不进入日志。

#### 正式运行命令

在371591每次冻结的PocketXMol release根执行：

```bash
bash 训练与运行/sh/train_docking.sh B-C-T0-RA
```

输出根为 `/storage/penghongen/PocketXMol/training/B-C-T0-RA/`：src保存Python源码副本，train_config保存实际配置，checkpoints保存检查点，wandb保存运行记录。正式启动前已确认该实验目录不存在。

沿用原allocation控制器和锁目录 `/home/penghongen/Feedback/Pocket_Plus/allocations/371591/`；out与err仍在该目录。动态命令调用PocketXMol既有release／launch工具。after_lock保留，同一卡按实验顺序使用。

#### 实际启动

72×1真实GPU检查通过后，已在原371591的第9次执行启动本实验。计算节点生成的启动标识为 `train_B-C-T0-RA_job371591_20260910T112700`；实际release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_dbfde52856a9/PocketXMol`，launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/train_B-C-T0-RA_job371591_20260910T112700`。本地实现提交为 `136c621`；release还包含工作区原有的docking/preparation.py说明文字改动，未把这些改动擅自纳入提交。服务器源码目录的旧.git元数据不代表本次实现版本，以冻结release和本地记录追溯。

动态命令副本和启动留证分别保存为 `/storage/penghongen/PocketXMol/control/371591/train_B-C-T0-RA_run_cmd.sh`、同目录train_B-C-T0-RA_start.json。留证记录本次在累计out、err中的起始字节位置360594和54949，读取本次错误时不混入先前GPU检查的失败输出。脚本语法检查通过后才写入该作业run_cmd并释放try_lock；没有取消allocation，after_lock保留。

启动已确认加载官方参数、bf16混合精度、单GPU、全部20.2 M参数可训练。W&B在线创建成功，记录目录为 `/storage/penghongen/PocketXMol/training/B-C-T0-RA/wandb/run-20260910_112743-9wkyn4qn`，源码与实际配置副本已保存。首次稳定检查已到第70个优化器更新，lr仍为1e-4，未见本次OOM或批量裁减；尚未到第800步的首次完整验证。约1.08更新／秒只是启动阶段观察，不作全程耗时承诺。

#### 当前结果与后续

2026-09-10收到明确纠偏后，在gnode09核对Slurm371591、实际run_cmd、release、launch和主进程8895的命令及cgroup归属，确认正在运行本实验。通过 `/home/penghongen/Feedback/Pocket_Plus/allocations/371591/kill_lock_371591` 请求停止，控制器确认终止进程组8895，退出137，随后恢复try_lock；after_lock和RUNNING状态的A800 allocation保留，没有使用scancel。

停止时累计日志最后进度为17901次更新，最后完成的定期验证为17600步，原始val/loss最低2.13159。这些数字只描述错误C5训练条件的历史，不能作为正确T0成绩。旧目录、源码、配置、所有检查点、W&B 9wkyn4qn和累计out／err均保留；停止证据为 `/storage/penghongen/PocketXMol/control/371591/invalid_T0_20260910_stop.json`。没有为该错误模型生成正式候选或测试成绩。

### T0纠偏与验收过程

本记录落实用户2026-09-10明确纠偏，并同步更新 [科学契约](../../想法/方案草稿/9-8-科学契约.md)、[工程细节](../../想法/方案草稿/9-8-工程与实现细节.md)和[边界清单](../../想法/方案草稿/9-8-边界与核查清单.md)。范围是解除“中心模式自动偏移”，验证完整坐标路径并从官方参数重训受影响的中心T0。T1已有公式、包络E、六模型和官方评价矩阵不变，不重建完整图、累计筛选、对称排列或冻结C5向量，不扩展密度实现。

#### 错误运行与保留证据

此前唯一正式训练是B-C-T0-RA；它错误地用动态C5训练和冻结C5计算val/loss，已停止并标为无效。旧运行详见 [原训练记录](1-B-C-T0-RA.md)。主进程8895经Slurm371591及cgroup核实后，用实际控制器的kill_lock终止，最后日志17901步、最近定期检查点17600步，退出137。after_lock保留、try_lock已恢复，A800 allocation继续RUNNING。没有scancel、资产删除或旧结果覆盖。

旧产物根 `/storage/penghongen/PocketXMol/training/B-C-T0-RA/` 和W&B `9wkyn4qn` 只保留作错误条件证据。停止记录为 `/storage/penghongen/PocketXMol/control/371591/invalid_T0_20260910_stop.json`。中心T0-RB及其余四个模型尚未正式启动；后续中心T0-RB直接使用修正实现，包络T0不受这一中心偏移错误影响。

#### 最小实现与版本

- `docking/dataset.py::OccurrenceDataset.__getitem__`：无限流仅在protocol明确为C5时抽取偏移，C0不再被pocket_mode=center触发。C5训练仍共用一份完整delta选袋和定原点，有限C5仍读冻结向量。
- `scripts/train_pl.py::DataModule.setup`：只读取已有dock.center_translation，将中心T0训练／监督验证设为C0、中心T1设为C5、包络设为E；没有新增可冲突的科学开关。
- `utils/sample_noise.py` 的T1公式和 `docking/sampling.py` 的正式C0/C5能力保持原样。T1新增项在原同构重分配之前，受体与监督目标不随其移动；最终只加回原点一次。
- 三份当前规格、六份配置的字段说明、共同数据说明和运行说明同步更新。B-C-T0-RA的新W&B显示名为B-C-T0-RA-C0，其余训练超参数、名义72×1、官方初始化和停止规则不变。

本轮继续同一个端到端任务的实现分支codex/pxm-receptor-baselines，开始修复时HEAD为f79e940，共同学习基点仍为0412824adbdd4e4f229b572d9a6bfde75d3fdbc9。没有重写已运行release依赖的历史。开始时的assets.py、dataset.py、preparation.py未提交改动已备份到 `tmp/pxm-t0-correction-20260910/`；用户随后明确允许核对仅为注释／Docstring后顺手提交，按此授权处理。临时目录不进入正式源码。

用户注释单独保存为6cd023f；两处生产修复、测试、配置说明和当前契约保存为ec01242。使用原sync_code.ps1非删除同步，保留服务器Git和旧release。首轮CPU／GPU检查共用冻结源码 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_5a4259c2b381/PocketXMol`。a1132dd保存第一轮审查后的测试补强与说明；生产AST和六份配置解析值仍与已通过GPU检查的ec01242相同。

#### 必要验收命令与结果

以下均为测试，不创建正式W&B记录，不能作为训练命令。`tests/test_docking_centers.py` 覆盖六模型监督条件、边界残基选袋、局部目标与噪声、同分子共享和不同分子独立平移、四种T0/T1与C0/C5组合、首步纯先验、后续实际信息等级、真值隔离和世界坐标只还原一次。公式在原add_noise返回、同构重分配之前核对；相同C0与蛋白口袋下，T0与原特征化及缺省关闭新增平移的原dock链逐字段比较。

本地测试环境使用 `tmp/pxm-20260910/venv/Scripts/python.exe`，CUDA_VISIBLE_DEVICES为空、PYTHONUTF8=1、OMP／MKL／OPENBLAS线程各1、PYTEST_DISABLE_PLUGIN_AUTOLOAD=1。首次命令如下，12项通过、真实服务器项跳过，耗时20.26秒；后续补入两个动态训练实例合批的独立平移检查后须复测。

```powershell
tmp/pxm-20260910/venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --basetemp=tmp/pxm-t0-correction-20260910/pytest-local-1 --tb=short tests/test_docking_centers.py
```

真实源检查直接读取已冻结train的5ftl/0，不重新准备任何科学资产。GPU资源检查使用现有real_data测试的B-C-T0-RA项，从官方参数做2次更新、原C0监督验证和C5采样评价；它是必要短检查，不产生正式模型成绩。实际服务器命令、release、launch及通过证据如下。

组合复测使用相同环境及pytest参数，basetemp改为pytest-local-2，测试文件为test_docking_centers.py、test_docking_data.py、test_docking_model.py、test_docking_sampling.py。40项通过、2项因缺少本地服务器资产跳过，新增合批测试1项失败：该测试在正常训练合批之后再次调用噪声器，却沿用加载器排除列表移除了task_setting。实际训练先在单分子变换中加噪再合批，不存在此字段问题。测试保留task_setting后，以pytest-local-3完整复测中心文件，13项通过、1项真实服务器检查跳过，耗时36.16秒。未改动的既有数据／模型／采样测试在前述组合运行中为28项通过、1项跳过；不将本地跳过计作真实资产验收通过。

服务器CPU测试命令，在同步后的项目根提交：

```bash
bash 训练与运行/submit_task.sh --sh ops/run_docking_checks.sh --resource cpu --cpus 8 --job-name pxm_t0_centers -- -s --tb=short tests/test_docking_centers.py
```

作业376811在cnode01使用8核，于16:37:12建立launch，64秒后COMPLETED，退出0；pytest共14项通过、无跳过，耗时51.79秒，包含真实train/5ftl/0。其余准备阶段输出属于pytest构造样本，正式共同数据未重建。launch为 `/home/penghongen/Feedback/PocketXMol/launches/376811/run_docking_checks_job376811_20260910T163712_a1`，out／err保留于 `/home/penghongen/Feedback/PocketXMol/allocations/376811/`。

边界夹具补强后使用同一CPU命令，只将job-name改为pxm_t0_centers_final。作业376817仍为cnode01、8核，COMPLETED、退出0，总33秒；14项通过、无跳过，pytest耗时22.98秒，真实5ftl/0的T0／T1口袋分别为488／465个重原子（仅诊断）。release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_795ded5b3f0d/PocketXMol`，launch为 `/home/penghongen/Feedback/PocketXMol/launches/376817/run_docking_checks_job376817_20260910T164553_a1`，out／err位于同号allocations目录。

本次GPU短验收在371591现有A800内运行以下测试命令：

```bash
bash ops/run_docking_gpu_checks.sh -k 'real_data and B-C-T0-RA'
```

16:34:41确认try_lock和after_lock存在、kill_lock不存在，先保存旧动态命令，再写入上述专用检查并解除try_lock；after_lock始终保留。命令留证 `/storage/penghongen/PocketXMol/control/371591/T0_C0_gpu_gate_20260910T163441.sh`；launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/docking_T0_C0_gpu_checks_job371591_20260910T163502`。

GPU测试1项通过、4项未选中，耗时61.57秒，控制器第10次执行成功后恢复try_lock。实际训练72×1、bf16，两次优化器更新及一批原val/loss均为C0；同一模型以C5正常生成并评价2个候选。短fit耗时36.36秒，峰值分配48224273408字节（约44.91 GiB）、预留50092572672字节，未触发验收中的OOM失败钩子。验证损失3.28679只属于这次两步短检查，不是正式模型成绩。完整JSON为 `/storage/penghongen/tmp/pocketxmol_gpu_checks_docking_T0_C0_gpu_checks_job371591_20260910T163502/pytest/test_real_data_training_and_sa0/real_data_gate.json`。

#### 主代理自查与独立审查

提交独立审查前已完成两遍自查。第一遍按文件顺序检查OccurrenceDataset及DataModule.setup的职责、位置、调用、嵌套和Docstring：生产改动沿既有入口内联，只改变条件选择，没有新增生产函数、包装、默认值或校验框架；新增测试独立放在tests，不进入生产调用链。测试构造漏掉task_setting的问题已按原调用顺序修正。

第二遍按代码注释skill及其示例核对类、函数、配置和关键行：明确C0/C5、XYZ与Å、完整delta、局部目标0／-delta、实际s与先验分支、同构重分配之前的比较边界。三份规格和运行说明已去除当前“T0/T1都偏移”的表述，旧实验日志保留错误事实。用户暂存的assets.py、dataset.py、preparation.py经去除Docstring后的AST比较确认不改变执行逻辑；仅清除preparation.py新增空白行尾空格。原sample_noise.py及sampling.py执行代码保持不变，六模型配置除T0-RA的新W&B显示名外参数值不变。

本次纠偏已完成两轮独立审查，三类分别负责Git与函数布局、注释与文档、科学逻辑与验收覆盖；范围只覆盖本次中心契约修复及其实际调用链。此前完整实现已完成的审查没有重新扩大范围。本次三类审查和必要CPU／GPU验收均已通过。

第一轮三类审查均未发现生产逻辑阻塞。布局审查再次批准OccurrenceDataset.__getitem__及DataModule.setup的具名框架入口例外，核实用户注释AST不变及双线延续。文字审查提出补齐g／delta／s／Z定义、区分此前实现与本次纠偏的完成状态、统一一处用户注释标点，均已整理且保留原意。

逻辑审查指出原边界夹具虽能发现中心改变，却不能区分质量均值与算术均值，也不能区分完整delta与s*delta选袋。已将C/O残基坐标改为14／15.8 Å，使质量中心>15而算术均值14.9<15；新增16.6 Å残基，完整delta=2选入、s*delta=1.3不选入；保留C0的15 Å及C5的17 Å严格等号边界。真实样本的口袋原子数变化改为诊断信息，因为合法中心改变不保证数量变化。修正仅涉及测试与说明，生产执行代码及配置值保持ec01242所验收的状态。

改后主代理核对两个受改测试函数的职责、调用与关键行含义，再执行相同本地测试入口，basetemp=tmp/pxm-t0-correction-20260910/pytest-boundary-4，选择 `tests/test_docking_centers.py -k training_boundary`；2项通过、12项未选，耗时17.57秒。第二轮三类独立审查均通过，无生产逻辑或Git阻塞项，之后只针对实际发现的问题窄核。

第二轮逻辑审查指出float32的g=1.2会把构造例中的15／17 Å等号距离略微推大。已仅平移pytest临时三原子配体，使其g精确为0并显式断言；其余NPZ数组保留，真实资产不变。使用pytest-boundary-5重复上述两项边界命令，2项通过、12项未选，耗时7.58秒。逻辑审查的内存窄核确认C0／C5等号距离均精确为15.0，能够区分<与<=，该问题关闭。服务器的真实样本检查函数未再改动，生产执行代码和已通过GPU检查的配置值也未变，不重复GPU验收。

#### 计划与实现差异

有害差异已修复：中心T0训练和监督验证改为C0，T1仍保留配套的C5定位与平移机制。共同科学资产没有变化。两遍主代理自查、两轮独立审查、必要CPU／GPU验收和正确T0独立重训均已完成。按用户2026-09-11最新要求，训练后完整验证集采样已取消；当前先完成本模型的测试集C0／C5推理、CPU评价和记录，再逐一完成其余模型。原训练val/loss与best选择保留，最终完成测试报告和双线Git收口。

### 已取消的完整验证集采样

此前d788edb依据当时文档建立validation／test两份配置，完成两遍配置自查、原make_config解析和独立逻辑代理窄核后，于371591启动完整validation。这一训练后完整验证集工作偏离用户普遍意图，已按用户最新要求停止，不再继续采样或提交评价。原训练期间的val/loss不受影响。

历史采样release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_b3b44f062a12/PocketXMol`，launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/sample_B-C-T0-RA_validation_job371591_20260911T020258`。原启动记录及动态命令在 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T0-RA_validation_start.json` 和sample_B-C-T0-RA_validation_run_cmd.sh。master与gnode09时钟约有3分钟差异，保留各自原值，耗时不跨主机时间相减。

停止前通过/proc/19347/cmdline核对实际配置为sample-B-C-T0-RA-validation.yml，并通过cgroup确认属于job_371591。首次使用ps进行精确命令比较因默认显示宽度截断而未匹配，没有写锁或改动任务；改用/proc完整命令核实后，于gnode09时间2026-09-11 14:16:43创建实际kill_lock。控制器明确终止进程组19347，第12次执行退出137，随后恢复try_lock，after_lock保留、kill_lock已由控制器处理、旧进程消失。未scancel、释放资源、删除或覆盖检查点和候选。

停止记录为 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T0-RA_validation_stop_20260911.json`。已完成标记共有C0 781个实例／39050个成功候选、C5 436个实例／21800个成功候选；被中断实例的未完成文件也原样保留。已有validation产物仅为历史记录，不继续完成或评价，不作为当前正式测试成绩。
