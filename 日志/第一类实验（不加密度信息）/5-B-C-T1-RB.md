# 5-B-C-T1-RB

**A800上的训练、C0／C5完整测试和评价均已完成。** ALL 的 Top-1 成功率为58.97%／39.46%；两协议各446实例、22300候选全部生成并评价成功。训练在22400次更新后停止。

更新核查：2026-09-13 11:09（服务器 master，UTC+8）。依据 [科学契约](../../想法/方案草稿/9-8-科学契约.md)、[工程细节](../../想法/方案草稿/9-8-工程与实现细节.md) 和 [边界清单](../../想法/方案草稿/9-8-边界与核查清单.md)，本文件统一保存本模型的训练、测试、评价与尝试历史。共同准备见 [实现与共同数据准备](../实现与共同数据准备.md)，全实验进度见 [总日志](../总日志.md)。

## 当前有效运行与产物

| 项目 | 当前事实 |
|---|---|
| 训练产物根 | `/storage/penghongen/PocketXMol/training/B-C-T1-RB/` |
| 正式测试检查点 | `checkpoints/step=17600.ckpt`；C5 val/loss=2.8354082107543945 |
| 训练 W&B | [wmgkgurr](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/wmgkgurr) |
| 测试与评价产物 | `/storage/penghongen/PocketXMol/sampling/B-C-T1-RB/test/` |
| 评价 W&B | [6wvylxgi](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/6wvylxgi) |

训练保留原路径 `val/loss` 和最低损失检查点选择；训练结束后直接完整测试，不做训练后完整验证集采样。每实例每协议50候选、100步，原 self-ranking（原置信度及碰撞、立体化学项组成的候选排序）不变。完整结果优先放在下文，执行核查和失败尝试放在后部。

ALL为446个实例的完整测试集合。按完整模板身份object_key在ALL中的频数分组，只有频数大于10的身份才沿冻结顺序截取：CAP10保留前10个，HF10_TO5保留前5个；频数不超过10的身份全部保留。两个视图分别272和227个实例，与ALL复用候选，不重复生成或相加计数。

## 测试结果

RMSD单位为Å，沿原CalcRMS计算，不移动或刚体对齐预测姿态。Top-1取原self-ranking评分最高的候选，Top-5取评分最高的5个候选中的最低RMSD，oracle取全部50个候选中的最低RMSD。oracle只说明候选池中存在的最好姿态，不用于选择实际输出。三个测试视图共用候选池，实例集合重叠，不相加计数。

下表成功率与平均RMSD按实例等权，括号内为成功数。ALL、CAP10、HF10_TO5分别包含77、67、65个PDB。

| 协议 | 视图 | 实例数 | Top-1成功率 | Top-5成功率 | Oracle成功率 | Top-1平均RMSD |
|---|---|---:|---:|---:|---:|---:|
| C0 | ALL | 446 | 58.97%（263） | 75.34%（336） | 87.67%（391） | 2.943 |
| C0 | CAP10 | 272 | 48.90%（133） | 67.65%（184） | 83.46%（227） | 3.443 |
| C0 | HF10_TO5 | 227 | 50.22%（114） | 68.72%（156） | 82.82%（188） | 3.299 |
| C5 | ALL | 446 | 39.46%（176） | 50.22%（224） | 60.54%（270） | 4.008 |
| C5 | CAP10 | 272 | 29.78%（81） | 40.07%（109） | 53.68%（146） | 4.661 |
| C5 | HF10_TO5 | 227 | 30.40%（69） | 40.97%（93） | 55.51%（126） | 4.597 |

PDB等权先在每个PDB内部平均其测试实例成功指标，再对该视图中的PDB平均，避免多实例PDB获得更大权重。

| 协议 | 视图 | Top-1成功率 | Top-5成功率 | Oracle成功率 |
|---|---|---:|---:|---:|
| C0 | ALL | 56.87% | 75.08% | 87.80% |
| C0 | CAP10 | 55.41% | 73.21% | 85.71% |
| C0 | HF10_TO5 | 55.58% | 71.91% | 84.71% |
| C5 | ALL | 40.63% | 51.05% | 58.89% |
| C5 | CAP10 | 39.40% | 49.99% | 58.52% |
| C5 | HF10_TO5 | 38.44% | 49.68% | 58.99% |

同一T1-RB在C5条件下的ALL Top-1比C0低19.51个百分点。逐实例配对为99个从C0成功变为C5失败，12个反向变化；两协议仍执行相同T1采样机制，仅给定中心、固定口袋和原点按各自定位条件构造。

### 评分与姿态误差的对应

每个实例用其50个候选计算self-ranking与负RMSD的Spearman相关系数，再对实例求均值或中位数。Pose AUC是以RMSD<2 Å为正类的逐实例ROC曲线下面积；仅有一种类别时AUC无定义，该实例仍保留在成功率分母中。所有Spearman均有效，以下AUC缺失均由单一类别造成。

| 协议 | 视图 | Spearman均值 | Spearman中位数 | Pose AUC均值 | AUC有效实例／全部实例 |
|---|---|---:|---:|---:|---:|
| C0 | ALL | 0.2933 | 0.3163 | 0.7054 | 343／446 |
| C0 | CAP10 | 0.2467 | 0.2523 | 0.6821 | 206／272 |
| C0 | HF10_TO5 | 0.2555 | 0.2604 | 0.6839 | 172／227 |
| C5 | ALL | 0.2212 | 0.2460 | 0.6855 | 246／446 |
| C5 | CAP10 | 0.1875 | 0.1930 | 0.6471 | 141／272 |
| C5 | HF10_TO5 | 0.1988 | 0.2102 | 0.6450 | 122／227 |

C0有128个实例、C5有94个实例的候选池中存在成功姿态，但原self-ranking的Top-1未选中。本次忠实记录这种差异，继续保留原评分规则和候选池。

### 对接结果与口袋核酸占比

核酸占比为当前协议选出的标准RNA／DNA重原子数除以标准蛋白与RNA／DNA重原子总数。两协议各439个纯蛋白口袋、7个含核酸口袋；7个含核酸实例全部属于三个测试视图。

| 协议 | 口袋分组 | 实例数 | Top-1成功数 | Top-5成功数 | Oracle成功数 |
|---|---|---:|---:|---:|---:|
| C0 | 核酸占比为0 | 439 | 261 | 331 | 384 |
| C0 | 核酸占比大于0 | 7 | 2 | 5 | 7 |
| C5 | 核酸占比为0 | 439 | 175 | 223 | 268 |
| C5 | 核酸占比大于0 | 7 | 1 | 1 | 2 |

全部含核酸实例列于下表，occurrence编号均为0；RMSD依次为Top-1／Top-5／oracle，单位Å。

| PDB | C0核酸占比 | C0 RMSD | C5核酸占比 | C5 RMSD |
|---|---:|---|---:|---|
| 9q16 | 19.43% | 0.964／0.883／0.883 | 10.05% | 2.586／2.586／2.520 |
| 9r3d | 12.23% | 7.452／1.773／1.415 | 4.69% | 7.174／6.492／4.138 |
| 9shy | 20.92% | 2.835／2.835／1.444 | 24.70% | 3.571／3.517／3.492 |
| 9v7o | 100.00% | 6.041／2.997／0.969 | 100.00% | 3.717／3.083／2.709 |
| 9z2n | 19.31% | 1.723／1.704／0.664 | 14.74% | 0.948／0.699／0.661 |
| 9z2u | 12.01% | 2.078／1.671／0.876 | 32.28% | 6.879／4.855／3.262 |
| 9z3d | 11.52% | 2.134／1.051／0.982 | 18.34% | 2.813／2.813／1.979 |

纯RNA实例9v7o/0在两个协议下都完成50个候选及全部评价，计入正式分母；C0只有oracle成功，C5三项均失败。含核酸实例仅7个，以上描述不能单独确定核酸占比与成功率的稳定关系。

### 与T1-RA的同协议比较

比较对象为[T1-RA的20800步best完整测试](2-B-C-T1-RA.md)。两模型均按各自原C5 val/loss选best，使用相同测试清单、冻结种子、C5向量、50×100预算、原T1采样和评价代码；主要实验差异为RA联合受体编码与RB分别编码。下表按ALL实例等权，差值为RB减RA，单位为百分点。

| 协议 | 指标 | RA成功数／446 | RB成功数／446 | 成功率差值 |
|---|---|---:|---:|---:|
| C0 | Top-1 | 264 | 263 | -0.22 |
| C0 | Top-5 | 327 | 336 | +2.02 |
| C0 | Oracle | 388 | 391 | +0.67 |
| C5 | Top-1 | 188 | 176 | -2.69 |
| C5 | Top-5 | 230 | 224 | -1.35 |
| C5 | Oracle | 270 | 270 | 0.00 |

C0的Top-1逐实例配对为25个从RA成功变为RB失败、24个反向变化；C5分别为18和6个。PDB等权的ALL Top-1差值为C0 -2.83、C5 -2.93个百分点。RB在C0的Top-5与oracle较高，在C5的Top-1与Top-5较低；这些结果尚未给出所有指标一致的分支优劣。继续既定实验，最终分支选择留给用户，不依据测试结果更换检查点或增加实验。

## 结果保存与限制

正式产物根为 `/storage/penghongen/PocketXMol/sampling/B-C-T1-RB/test/`。协议下的 `<pdb_id>/<occurrence_id>/candidate_metrics.json`、assessment.json分别保存逐候选和逐实例评价；协议occurrences.json、summary.json及test/summary.json保存汇总。W&B 6wvylxgi已上传规定汇总。

本地副本 `tmp/pxm-20260913/B-C-T1-RB-test-results.json`来自服务器汇总及完整逐实例文件，正式产物仍以服务器目录为准。报告成功数、实例等权和PDB等权成功率、RMSD均值、Spearman／AUC汇总、核酸分组与逐实例配对均从该副本核对；本地核对脚本为 `tmp/pxm-20260913/report_t1_rb.py`，输出REPORT_AGGREGATE_AUDIT_PASS。这些是结果整理，没有重新采样或评价。

评价stderr有591条RDKit“暂不支持allene-style立体化学并忽略”的提示，没有Traceback。提示数不代表独立异常实例数；原立体化学评价的支持范围受该RDKit版本限制。本次没有因此修改图、分母或评价规则，也没有RMSD按原子编号匹配的回退。

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

## 完整测试配置与正式命令

`configs/docking/sample-B-C-T1-RB-test.yml`明确读取本次保存的RB训练配置及17600步best。输入为446个测试实例的C0、冻结C5；两种条件均启用T1后续中心相关重新加噪，首步纯高斯。每实例每协议50候选、100步，batch50；输出为 `/storage/penghongen/PocketXMol/sampling/B-C-T1-RB/test/`，三个视图共用候选池。

正式采样命令如下；完成全部C0/C5推理后才执行下一条CPU评价命令，二者分别登记launch和进程。

```bash
bash 训练与运行/sh/sample_docking.sh B-C-T1-RB-test
```

```bash
bash 训练与运行/sh/evaluate_docking.sh B-C-T1-RB-test
```

评价使用378693已分配CPU中的8个工作进程，W&B为pencounkdual-111/PocketXmol_raw，名称B-C-T1-RB_test，实际run id为6wvylxgi。候选生成、完整姿态评价及在线汇总均已完成。

主代理按配置读取顺序及注释规范完成两遍自查，使用本地项目环境的YAML解析核对：相对已运行的T1-RA测试配置，仅替换模型名、RB分支、实际best、对应训练配置和独立输出／W&B名称；其余字段完全一致。两轮独立配置审查均通过，没有修改采样器、评价器或科学预算。

## 训练完成与best核对

正式训练输出记录updates=22400、stop_reason=plateau及Training finished。2026-09-12节点时间17:16:22，在378693.0的8核CPU内读取last与best，确认下降次数3、最后验证步22400、28个定期检查点和last均保留；W&B id仍为wmgkgurr。调度器和优化器末次学习率均为8e-7，第三次下降后没有再更新参数。best为 `checkpoints/step=17600.ckpt`，原C5 val/loss=2.8354082107543945，是全部定期损失的最小值；best包含1236个model参数键，所有参数有限。

只读核对脚本为 `/storage/penghongen/tmp/pocketxmol_checkpoint_20260912/inspect_t1_rb.py`，训练摘要保存为 `/storage/penghongen/PocketXMol/training/B-C-T1-RB/training_summary_20260912.json`，其中checkpoint_losses保存全部28个检查点及损失。以下为产物核对命令，不是正式训练或推理命令：

```bash
srun --jobid=378693 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /storage/penghongen/PocketXMol/runtime/venv/bin/python -B /storage/penghongen/tmp/pocketxmol_checkpoint_20260912/inspect_t1_rb.py
```

## 有效运行的执行与核查记录

本节按实际事件保留执行证据；其中启动阶段的进度与后续计划属于当时记录，当前完成状态以文档开头为准。

### 实际接入

首次启动前核对混用了本地LF归一化字节与服务器原CRLF字节，断言在创建控制记录及操作锁之前停止。重新核对原始字节，两端训练YAML完全相同，SHA256均为ef6aa7acef6ee88286abd65d4e9cc3afc66f899bfa273a22db4a12ef4692b325；没有修改训练配置或源码。

2026-09-12 master时间11:20:47，保存原动态命令及新正式命令后，移除获准的pre_lock_378693开始第一次执行，after_lock保留，没有使用kill_lock或scancel。实际训练主进程50944，cgroup确认属于378693。PocketXMol launch为 `/home/penghongen/Feedback/PocketXMol/launches/378693/train_B-C-T1-RB_job378693_20260912T111926`；节点时钟与master有差异，保留各自记录。

控制记录为 `/storage/penghongen/PocketXMol/control/378693/train_B-C-T1-RB_start.json`，命令副本train_B-C-T1-RB_run_cmd.sh，原命令副本original_run_cmd_before_B-C-T1-RB.sh。本次out／err起点136／0。实际源码来自上述fa0d957b2d3f的PocketXMol release，控制器外层Pocket_Plus release仅负责既有资源控制。运行时pre、try、kill锁均不存在，after_lock保留。

实际启动输出确认从规定官方pocketxmol.ckpt加载，CUDA启用、bf16混合精度、约20.7M参数全部可训练；W&B正常在线创建wmgkgurr，本地记录位于训练根wandb/run-20260912_112017-wmgkgurr。未传--resume，不承接其它训练参数或优化器状态。

启动验收时已进入80次优化器更新，约1.04步／秒，lr=1e-4，原loss和置信度损失正常记录；未发现Traceback、CUDA OOM、reduce_batch或NaN。此处仅确认正式训练正常进入更新，不代表训练完成或已选定best。

### 验收来源与后续步骤

该RB／T1代码及72×1资源配置已经完成两遍主代理自查、两轮全面独立审查和真实非测试样本GPU验收，见 [共同准备日志](../实现与共同数据准备.md)；T0纠偏之后T1机制保持，见 [中心纠偏日志](1-B-C-T0-RA.md)。本次仅接入新增授权资源，未修改生产函数或科学配置，不重复模型试训。

训练正常结束后在Slurm内核实停止原因和实际best，再建立明确检查点的完整C0／C5 test配置，batch50、每实例50候选100步。推理完成后在同一378693执行既有evaluate_docking.sh，使用分配内CPU并保留after_lock，不另排纯CPU任务。完成本模型结果记录后再运行第6个B-E-T0-RB。不安排完整验证集采样或新增实验。

### 完整测试实际接入

训练结束后的W&B后台进程曾因GraphQL接口超时而重试，主进程50944等待其收尾；节点时间17:22:40，后台流自行关闭，随后控制器第1次执行成功、try_lock恢复。确认50944及其W&B子进程51729均退出后才接入采样，没有为此使用kill_lock或更改W&B模式。

测试配置与训练完成记录提交为3001cf9；正式测试release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_c59a88a03678/PocketXMol`，由原ec06dbd源码仅加入本次测试配置和记录构成，临时发布源为 `/storage/penghongen/tmp/pocketxmol_t1_rb_test_20260912/PocketXMol`。未混入外来工作区文件，亦未改变中心模型代码。

2026-09-12 master时间17:26:58保存启动记录与动态命令后，移除获准try_lock，378693控制器第2次执行开始，after_lock保留。实际launch为 `/home/penghongen/Feedback/PocketXMol/launches/378693/sample_B-C-T1-RB_test_job378693_20260912T172535`；节点时钟约慢1分20秒，时间记录分别保留。启动元数据为 `/storage/penghongen/PocketXMol/control/378693/sample_B-C-T1-RB_test_start.json`，同目录保存新命令sample_B-C-T1-RB_test_run_cmd.sh与接入前命令sample_B-C-T1-RB_test_before_run_cmd.sh；累计out／err起点为25698979／8263。测试YAML的LF字节SHA256为2ac841d703635e00be10ad928140e96afd1f838ffdf0b779aae87ab3a0329969。

实际采样主进程26403，cgroup属于378693。推理已严格加载17600步best并写出test/run.json，记录中的RB、T1、C0/C5、50候选、100步、batch50及CPU8均与配置相同。启动检查时已有2个C0实例完成，例如11jb/1生成50个候选、失败0、执行100次模型前向，用时33.49秒，峰值已分配显存2538523648字节；未见Traceback或OOM。这里只确认正式推理正常开始，完整测试与CPU评价的完成情况将在后续记录。

### 推理耗时与资源观测

2026-09-12 master时间23:03检查时，C0已完成446实例、22300候选，C5完成134实例、6700候选，全部成功且没有OOM或重试。比较已完成C5的同一批134实例，其C0／C5累计单实例耗时分别为4050.03／6854.39秒，逐实例耗时比中位数为1.468；近期9hyu/22、31、106的C5耗时分别96.17、136.42、83.86秒，对应C0为35.39、49.09、26.67秒，口袋原子数相近或更少。候选预算、100次模型前向和配置均未变化。

gnode10时间23:05的只读核查确认：378693分配的是GPU IDX:0，UUID为GPU-0e253751-cce3-c71d-c2df-a222fec3759b；本作业仅有采样进程26403使用它，同一GPU另有3个作业外计算进程。只核对同卡进程数量和是否属于378693，没有追踪这些进程的任务内容，也没有对它们执行操作。观测时GPU利用率100%、P0、SM频率1410 MHz、温度69°C，降频原因掩码为0；本进程3秒内使用3.02秒CPU，I/O计数不变，没有磁盘等待。并发使用可能影响耗时，现有观测没有隔离其因果贡献。已向用户报告资源情况并询问是否继续当前推理或由用户安排资源调整；现有推理继续按已授权的预算运行，不更改科学配置、停止进程或操作其他任务。

### 全部候选生成完成与核对

2026-09-13核对时，控制器第2次执行已成功、主进程26403已退出，try_lock恢复且after_lock保留。C0与C5均完成446个实例、22300个候选，没有生成失败；本次采样日志区间没有Traceback、CUDA OOM或reduce_batch。

同一378693的8核CPU产物核对通过：实际配置与test/run.json一致，全部实例身份、object_key、视图、冻结种子和C5向量与test.jsonl相同；每实例50候选、100步、一批50、100次已完成批量forward。候选身份、编号、有限轨迹置信度、SDF编号和姿态／置信度文件存在性均通过核对。没有修改冻结清单或补生成候选。

| 协议 | 实例／候选 | 实例耗时合计（秒） | 推理耗时合计（秒） | 已完成批量forward | 峰值分配显存（字节） |
|---|---|---:|---:|---:|---:|
| C0 | 446／22300 | 13279.412 | 13208.046 | 44600 | 3234062848 |
| C5 | 446／22300 | 17366.621 | 17298.558 | 44600 | 3227094528 |

逐实例耗时合计约8小时30分46秒，不含模型启动和协议切换等开销。C5期间的同卡并发观测见后部“推理耗时与资源观测”；不能将两协议耗时差全部归因于T1公式或口袋变化。

完成证据为 `/storage/penghongen/PocketXMol/control/378693/sample_B-C-T1-RB_test_complete_20260913.json`。只读核对脚本为 `/storage/penghongen/tmp/pocketxmol_sampling_20260913/inspect_t1_rb_sampling.py`，本地副本为 `tmp/pxm-20260913/inspect_t1_rb_sampling.py`。以下是产物核对命令，不是正式采样或评价命令：

```bash
srun --jobid=378693 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /storage/penghongen/PocketXMol/runtime/venv/bin/python -B /storage/penghongen/tmp/pocketxmol_sampling_20260913/inspect_t1_rb_sampling.py
```

### 同一作业内的CPU评价启动

完成候选核对后，2026-09-13 master时间02:56:31接入正式命令 `bash 训练与运行/sh/evaluate_docking.sh B-C-T1-RB-test`。配置和源码继续使用同一c59a88a03678 release，不重新生成候选或改动科学设置；378693已分配16核CPU，正式评价使用8个工作进程，入口关闭CUDA，不另申请CPU作业。

控制器第3次执行，实际评价主进程10865，cgroup属于378693，CUDA_VISIBLE_DEVICES为空。launch为 `/home/penghongen/Feedback/PocketXMol/launches/378693/evaluate_B-C-T1-RB_test_job378693_20260913T025524`。启动记录为 `/storage/penghongen/PocketXMol/control/378693/evaluate_B-C-T1-RB_test_start.json`，同目录保存新命令evaluate_B-C-T1-RB_test_run_cmd.sh及此前采样命令evaluate_B-C-T1-RB_test_before_run_cmd.sh；本次out／err起点25867448／8359。启动时after_lock保留；完整评价和汇总的完成证据见下文。

### CPU评价完成与完整性核对

控制器第3次执行成功，主进程10865退出，try_lock恢复，after_lock保留，kill_lock不存在。W&B状态为online_completed、error为空，实际run id为6wvylxgi。两协议各22300个候选的RMSD与self-ranking均有限，资产错误、候选错误及按原子编号匹配的RMSD回退均为0。

在378693已分配CPU中核对每个assessment.json与协议occurrences.json完全一致，身份和视图与冻结清单相同，每个candidate_metrics.json包含原编号0至49。Top-1排序及Top-1／Top-5／oracle汇总与原候选指标一致，成功阈值始终严格小于2 Å。C0／C5逐实例评价耗时合计为22685.149／22701.547秒，这是8进程并行时各实例耗时的累加，不等于墙钟时长。本次未单独记录精确评价墙钟时间。

完成证据为 `/storage/penghongen/PocketXMol/control/378693/evaluate_B-C-T1-RB_test_complete_20260913.json`。核对脚本为 `/storage/penghongen/tmp/pocketxmol_evaluation_20260913/inspect_t1_rb_evaluation.py`，本地副本为 `tmp/pxm-20260913/inspect_t1_rb_evaluation.py`。以下命令仅核对已保存结果，不再次评价RMSD：

```bash
srun --jobid=378693 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /storage/penghongen/PocketXMol/runtime/venv/bin/python -B /storage/penghongen/tmp/pocketxmol_evaluation_20260913/inspect_t1_rb_evaluation.py
```

### 计划与实现差异

本模型训练、完整C0／C5候选生成、同卡CPU评价和报告均已完成，没有发现科学契约的实质差异。378693继续保留，按既定次序进入第6个B-E-T0-RB；本模型全部产物和W&B记录保留。

## 实验过程与失败尝试

以下为追溯用记录。已结束阶段中的“尚未”“随后”等表述仅说明当时状态；本文件开头的当前状态优先。历史命令不应再次执行，旧产物不作为有效模型或测试结果。

### 并行尝试F-5（尚在排队，不是失败）

2026-09-13 11:09只读squeue确认378587仍为PENDING／Priority。它是同一科学模型的独立运行尝试，不作为第8个实验，也不与已完成A800结果混合。本次仅归并文档，未取消、重提或修改该任务；后续执行权仍由负责它的原agent和用户决定。

本记录依据 [科学契约](../../想法/方案草稿/9-8-科学契约.md)、[工程细节](../../想法/方案草稿/9-8-工程与实现细节.md)及用户2026-09-11新增资源授权，执行既定六模型中的第5个模型B-C-T1-RB。任务由“核查PocketXMol执行前准备（3: 独立做另外半边）”负责；用户接受与另一对话重复计算，但各次产物必须独立。本文不改变共同数据、T1公式、RB构造或实验预算。

#### 正式配置与命令

配置为 `configs/docking/F-5.yml`，科学与资源参数复制自B-C-T1-RB.yml，仅W&B显示名称改为F-5_B-C-T1-RB。使用官方初始参数、独立优化器和调度状态；训练动态C5、监督验证冻结C5，T1训练平移及后续采样中心修正均保留。名义global batch为72，单卡batch72、累积1、bf16；每800次优化器更新原val/loss，沿用已冻结停止规则。DataLoader暂保留已验收的15个worker，32核是本作业申请量。

在服务器 `/home/penghongen/My_Project/PocketXMol` 提交：

```bash
bash 训练与运行/submit_task.sh --sh train_docking.sh --resource h100 --gpus 1 --cpus 32 --after_hold --job-name F-5 -- F-5
```

该作业中的正式训练命令为：

```bash
bash 训练与运行/sh/train_docking.sh F-5
```

训练产物为 `/storage/penghongen/PocketXMol/training/F-5/`，首次配置位于train_config/F-5.yml；checkpoints保留定期检查点及last，W&B位于pencounkdual-111/PocketXmol_raw、名称F-5_B-C-T1-RB，由入口生成独立run id。不得使用另一对话的B-C-T1-RB目录续训或混入候选。

训练完成后读取本任务实际最低原val/loss的best，再建立明确的测试配置并登记命令；预定测试输出根 `/storage/penghongen/PocketXMol/sampling/F-5/`。仅测试C0/C5，每实例每协议50候选、100步、batch50；ALL/CAP10/HF10_TO5复用候选，分别为446/272/227个实例。随后完成8进程CPU评价和结果记录。不运行完整验证集采样或评价。

#### 验收与执行状态

两遍主代理自查限定于新增配置、命令及输出隔离：第一遍用原make_config解析，确认除W&B名称外与既定模型配置相同；第二遍核对配置注释、训练入口的默认输出根、恢复限制及实验命名。没有新增或修改Python函数。Windows首次配置解析因默认GBK读取UTF-8注释失败；使用Python的-X utf8后通过，未修改项目解析器。

上述配置检查通过本地Python stdin执行，不加载模型、不读测试样本，不是正式训练或测试命令。原实现及T0修复的必要验收见[共同准备](../实现与共同数据准备.md)及[第1模型的纠偏记录](1-B-C-T0-RA.md)。独立代理对本次三份配置与三份日志完成两轮限定范围审查：配置等价、命令、科学协议和输出隔离通过；官方失败产物说明已修正，最新unstaged规则经窄核关闭，无剩余问题。

已于master时间2026-09-11 16:26:07提交，实际JobId为 **378587**，Slurm名称F-5，单张H100、32核、after_hold，无pre_hold。提交后首次squeue为PENDING，原因为Priority；实际release/launch及W&B id待启动后核实。提交请求、完整命令、返回码和作业编号保存在 `/storage/penghongen/PocketXMol/control/parallel-20260911/F-5_submit.json`。当前科学源码基准为f856ad8，本次F-5.yml及日志为未提交新增文件；安全同步成功，未删除远端文件。

H100仅申请这一张，使用after_hold；结束后保留after_lock。排队及稳定运行期间按用户要求用多次Start-Sleep -Seconds 300组成60或90分钟静默等待，醒来检查并报告有意义变化。用户随后明确：本对话全部新增及修改文件始终保持unstaged，不执行git add或git commit；不处理其他人的修改或暂存内容。运行来源以基准提交f856ad8、本次未提交文件及实际release/launch共同留证。

#### 计划与实现差异

用户明确新增并行资源并允许重复计算；本任务使用独立配置名称、输出根及W&B记录，科学定义与既定第5模型相同。训练、测试、CPU评价及最终结果尚未完成。
