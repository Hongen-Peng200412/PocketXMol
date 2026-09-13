# 6-B-E-T0-RB

**训练、完整E测试和CPU评价均已完成。** ALL的Top-1／Top-5／oracle成功率为64.35%／81.84%／90.36%；446个实例、22300个候选全部生成并评价成功，逐候选指标和汇总核对通过，W&B已上传。训练在30400次更新后因第三次学习率下降停止，实际best为21600步、原E val/loss=1.6228482723236084；它是全部38个定期检查点的最低原验证损失，1236个模型参数键均有限。全部检查点、last及训练W&B记录保留。

更新核查：2026-09-13 18:50（服务器 master，UTC+8）。依据 [科学契约](../../想法/方案草稿/9-8-科学契约.md)、[工程细节](../../想法/方案草稿/9-8-工程与实现细节.md) 和 [边界清单](../../想法/方案草稿/9-8-边界与核查清单.md)，本文件统一保存本模型的训练、测试、评价与尝试历史。共同准备见 [实现与共同数据准备](../实现与共同数据准备.md)，全实验进度见 [总日志](../总日志.md)。

## 当前有效运行与产物

| 项目 | 当前事实 |
|---|---|
| 训练产物根 | `/storage/penghongen/PocketXMol/training/B-E-T0-RB/` |
| 正式测试检查点 | `checkpoints/step=21600.ckpt`，相对于上述训练根；原E val/loss=1.6228482723236084 |
| 训练 W&B | [423nfmpm](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/423nfmpm) |
| 测试与评价产物 | `/storage/penghongen/PocketXMol/sampling/B-E-T0-RB/test/`；全部446实例已有候选、assessment、candidate_metrics及最终summary，核对完成 |
| 评价 W&B | [zykj1eit](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/zykj1eit)，上传完成 |

训练保留原路径 `val/loss` 和最低损失检查点选择；训练结束后直接完整测试，不做训练后完整验证集采样。每实例每协议50候选、100步，原 self-ranking（原置信度及碰撞、立体化学项组成的候选排序）不变。完整结果优先放在下文，执行核查和失败尝试放在后部。

ALL为446个实例的完整测试集合。按完整模板身份object_key在ALL中的频数分组，只有频数大于10的身份才沿冻结顺序截取：CAP10保留前10个，HF10_TO5保留前5个；频数不超过10的身份全部保留。两个视图分别272和227个实例，与ALL复用候选，不重复生成或相加计数。

## E测试结果

RMSD单位为Å，使用原CalcRMS，不移动或刚体对齐预测姿态，成功阈值严格小于2 Å。Top-1取原self-ranking最高的候选，Top-5取评分最高5个候选中的最低RMSD，oracle取全部50个候选中的最低RMSD；oracle反映候选池上限，不用于选择实际输出。三个视图复用同一候选池，实例集合重叠，不相加计数。

下表按实例等权，括号内为成功实例数。ALL、CAP10、HF10_TO5分别包含77、67、65个PDB。

| 视图 | 实例数 | Top-1成功率 | Top-5成功率 | Oracle成功率 | Top-1平均RMSD |
|---|---:|---:|---:|---:|---:|
| ALL | 446 | 64.35%（287） | 81.84%（365） | 90.36%（403） | 2.830 |
| CAP10 | 272 | 58.09%（158） | 75.37%（205） | 86.76%（236） | 3.244 |
| HF10_TO5 | 227 | 57.71%（131） | 75.33%（171） | 87.22%（198） | 3.235 |

PDB等权先在每个PDB内平均实例成功指标，再对该视图的PDB平均。

| 视图 | Top-1成功率 | Top-5成功率 | Oracle成功率 |
|---|---:|---:|---:|
| ALL | 63.66% | 79.53% | 89.92% |
| CAP10 | 63.59% | 76.99% | 88.49% |
| HF10_TO5 | 62.70% | 76.45% | 88.27% |

### 评分与姿态误差的对应

Spearman为每个实例的50个候选中self-ranking与负RMSD的相关系数，再按实例求均值或中位数。Pose AUC为逐实例的ROC曲线下面积，以RMSD<2 Å为正类；只有一种类别时AUC无定义，但该实例仍留在成功率分母中。所有Spearman均有效，AUC缺失均因单一类别。

| 视图 | Spearman均值 | Spearman中位数 | Pose AUC均值 | AUC有效实例／全部实例 |
|---|---:|---:|---:|---:|
| ALL | 0.2506 | 0.2485 | 0.6835 | 367／446 |
| CAP10 | 0.2632 | 0.2626 | 0.7000 | 218／272 |
| HF10_TO5 | 0.2716 | 0.2859 | 0.7097 | 185／227 |

ALL中116个实例的候选池包含成功姿态，但原self-ranking的Top-1没有选中。本次记录原评分的实际表现，未增加评分器或改变候选池。

### 对接结果与口袋核酸占比

核酸占比为E口袋中标准RNA／DNA重原子数除以标准蛋白与RNA／DNA重原子总数。测试含439个纯蛋白口袋、7个含核酸口袋，后者均在三个测试视图中。

| 口袋分组 | 实例数 | Top-1成功数 | Top-5成功数 | Oracle成功数 |
|---|---:|---:|---:|---:|
| 核酸占比为0 | 439 | 282 | 360 | 396 |
| 核酸占比大于0 | 7 | 5 | 5 | 7 |

全部含核酸实例的occurrence编号均为0，逐实例结果如下，RMSD单位Å。

| PDB | 核酸占比 | Top-1 RMSD | Top-5 RMSD | Oracle RMSD |
|---|---:|---:|---:|---:|
| 9q16 | 26.41% | 1.096 | 0.916 | 0.850 |
| 9r3d | 5.21% | 5.128 | 3.834 | 1.868 |
| 9shy | 24.58% | 1.593 | 0.933 | 0.933 |
| 9v7o | 100.00% | 3.309 | 3.082 | 1.594 |
| 9z2n | 26.12% | 0.905 | 0.870 | 0.793 |
| 9z2u | 14.35% | 0.971 | 0.971 | 0.744 |
| 9z3d | 13.25% | 1.169 | 0.797 | 0.737 |

纯RNA实例9v7o/0完成全部50候选和评价并计入正式分母；仅oracle达到严格小于2 Å。7个含核酸实例均有成功候选，但样本少，不能据此确定核酸占比与成功率的稳定关系。核酸分支的最终选择留待用户查看六模型报告后决定。

## 结果保存与核对

正式根 `/storage/penghongen/PocketXMol/sampling/B-E-T0-RB/test/` 保存原候选、姿态、逐候选candidate_metrics.json、逐实例assessment.json，以及E/occurrences.json、E/summary.json和该根下的summary.json。评价W&B zykj1eit已上传规定汇总。

本地整理副本 `tmp/pxm-20260913/B-E-T0-RB-test-results.json` 来自服务器最终汇总及全部逐实例文件，正式产物仍以服务器为准。成功数、实例及PDB等权成功率、RMSD均值、Spearman／AUC和核酸分组均已从副本核对；`tmp/pxm-20260913/report_e_rb.py` 输出REPORT_AGGREGATE_AUDIT_PASS。核查没有重新采样、计算RMSD或调整排序。 本阶段报告经主代理两遍自查和独立代理同范围两轮完整核查：18条数字表格、来源、严格阈值和分母均通过；首轮指出的汇总文件路径重复test层已修正，第二轮无新问题。F-6历史保持原样，未扩大生产代码审查或接管其他任务。

评价stderr有316条既有RDKit allene立体化学不支持提示，无Traceback，不代表316个独立异常实例。该类立体化学的区分受现有RDKit支持范围限制；本次不改图、不改变分母或原评价规则。全部22300个候选的RMSD和self-ranking均有限，候选评价错误为0，RMSD均未使用按原子编号匹配的回退。

## 最近一次进度与资源状态

采样第5次和CPU评价第6次执行均成功，主进程33421及29594均已退出；after_lock及恢复的try_lock存在，kill_lock不存在。446份assessment均报告评价成功，最终summary与W&B完成状态已核对。逐实例采样／评价耗时分别累计12638.16／22419.74秒，这些数不是整个作业墙钟时间。未见Traceback、OOM、非有限数值或降低batch的记录，完整结果见前文。

完整检查点摘要为 `/storage/penghongen/PocketXMol/training/B-E-T0-RB/training_summary_20260913.json`。本资源承担的第5、6个模型均已完成训练、完整测试及评价，资源和全部产物继续保留。本任务只剩第4模型的完整C5及CPU评价、六模型总报告和既定Git收口，不安排新实验或完整验证集采样。

## 正式测试配置与命令

`configs/docking/sample-B-E-T0-RB-test.yml`读取本次训练保存的配置和21600步best，receptor_branch=RB、center_translation=false、protocols=[E]、split=test。与已完成E-RA测试相比，仅替换核酸分支、模型名称及各自训练／产物身份，固定清单、50候选、100步、batch50和8进程评价保持。采样原点仍为实际受体重原子均值，空E测试输入失败保留候选及实例分母。

本次没有新增或修改生产Python函数。主代理第一遍自查核对配置读取关系、模型与数据分支、检查点和输出身份，第二遍核对YAML注释及日志含义；随后独立代理在同一范围完成两轮只读核查，均通过。YAML解析比较确认只改变模型身份、RB分支及各自产物位置；核查未扩展已通过的空E生产代码范围。检查点摘要本地副本为 `tmp/pxm-20260913/B-E-T0-RB-training-summary.json`。

以下正式采样命令已于2026-09-13接入378693执行；其release、launch和启动记录见下文。

```bash
bash 训练与运行/sh/sample_docking.sh B-E-T0-RB-test
```

以下正式CPU评价命令已于2026-09-13 17:45接入同一A800作业运行：

```bash
bash 训练与运行/sh/evaluate_docking.sh B-E-T0-RB-test
```

## 固定科学与训练条件

配置为 `configs/docking/B-E-T0-RB.yml`。包络E按标准受体残基重原子质量中心到任一真实配体重原子的距离严格小于10 Å选袋，模型原点为实际输入受体重原子的坐标算术均值。训练、val/loss和正式测试均为E，T0始终关闭新增整体平移。RB分别构建和编码蛋白／核酸受体图，从官方模型初始化后复制蛋白编码器参数给核酸编码器；原主体、loss、置信度头与原dock高斯链保持。

采用用户已批准的空E规则：仅训练和val/loss迭代在组批前跳过空口袋并警告记录身份，正式测试保留输入构造失败及全部分母。训练清单仍65290条，其中46个空E身份已完整审计，E有效训练实例65244；验证781条无空E。完整清单、对称排列和冻结C5向量均不重建。空E修复、46个身份和RA／RB真实训练输入验收见[E-T0-RA的空口袋审计](3-B-E-T0-RA.md)。

训练从规定的官方 `pocketxmol.ckpt` 参数重新开始，不传--resume，不承接T1-RB或E-RA参数；优化器、调度器和W&B运行身份独立建立。配置为batch72、梯度累积1、bf16、15个数据worker，固定有效全局批量72。AdamW初始lr=1e-4、weight_decay=0.001、betas=0.99／0.999、eps=1e-8、warmup=0。每800次优化器更新沿原路径计算val/loss；Plateau相对改善阈值1%、patience5、factor0.2，第三次实际下降立即停止，上限40000步。best取所有定期检查点的最低原E val/loss，保留last和全部定期检查点。

## 来源与运行前核对

使用已修复的正式release `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_43742cdf8166/PocketXMol`，源码基线ec06dbd加空E修复c9cc4a1，与有效E-RA重训所用源码相同。该版本已完成两遍主代理自查、两轮三类独立审查、22项CPU回归以及RA／RB各8步的真实训练输入GPU验收。本次没有新增或修改生产函数，不重复扩大审查或验收范围。

配置解析核对：相对已完成E-RA有效训练的配置，只有model.nucleic_branch从RA变为RB、train.wandb.name变为B-E-T0-RB，其余字段完全一致。目标训练根已检查不存在；正式接入时再次确认，避免覆盖已有产物。只复用明确release，不复制共享工作区外来文件。

## 正式训练命令与产物

以下是本模型正式训练命令，检查脚本和产物核对命令另行记录：

```bash
bash 训练与运行/sh/train_docking.sh B-E-T0-RB
```

独立训练根为 `/storage/penghongen/PocketXMol/training/B-E-T0-RB/`，训练保存配置位于其train_config子目录，定期检查点及last位于checkpoints子目录。W&B使用pencounkdual-111/PocketXmol_raw，名称B-E-T0-RB，实际run id为423nfmpm。

资源控制目录仍为 `/home/penghongen/Feedback/Pocket_Plus/allocations/378693/`，在前一评价正常结束、try_lock恢复后保存新旧动态命令及启动记录，再移除try_lock开始训练。after_lock保留，不使用scancel，不删除旧训练、测试、W&B或日志。

本模型训练结束后直接用实际best完成446个实例的E测试，每实例50个候选、100步，推理batch优先50。之后用同一A800作业的8个CPU评价进程完成ALL／CAP10／HF10_TO5及核酸比例分析。明确best的采样配置和短正式命令在训练结束后登记；不安排完整验证集采样或新增实验。

## 正式接入

2026-09-13 master时间05:06:47保存新旧命令与启动记录，移除try_lock后由378693控制器第4次执行开始训练，after_lock保留。接入前再次确认E-RB目标根不存在、前一T1-RB评价正常结束且W&B上传完成，动态命令确实是本任务的T1-RB评价入口。没有使用kill_lock或修改已完成实验的产物。

本次仍使用43742cdf8166 release，正式记录提交为d86a452；训练配置SHA256为d4e03bf8a21553c2af786f3563a689e68b812ba5f58573368dd83aa8c556ef4b。实际launch为 `/home/penghongen/Feedback/PocketXMol/launches/378693/train_B-E-T0-RB_job378693_20260913T050539`。启动记录 `/storage/penghongen/PocketXMol/control/378693/train_B-E-T0-RB_start.json`，同目录保存train_B-E-T0-RB_run_cmd.sh和修改前的train_B-E-T0-RB_before_run_cmd.sh。累计out／err起点为25869442／59362，后续仅以这一区间记录本次训练。

实际主进程57306及15个数据worker均在job_378693，没有--resume；日志确认从规定官方pocketxmol.ckpt加载模型，bf16及全部模型参数参与训练。新W&B运行是[423nfmpm](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/423nfmpm)，本地记录位于训练根wandb/run-20260913_050609-423nfmpm。最初更新的原损失与置信度损失均有限，且6j40/273、6j3z/22等空E实例已有明确跳过警告。

正式启动检查推进至109次优化器更新，约1.04步／秒，lr=1e-4；日志没有非有限损失、Traceback、OOM或reduce_batch，已记录6个不同空E训练身份。这里只确认正式训练正常开始，完整训练和best选择仍须继续。

## 计划与实现差异

本实验沿用已批准的E-T0-RB配置与空E规则，没有新增科学条件。正式训练及best核对、E完整推理、CPU评价和报告均已完成，候选及评价产物核对通过。

## 正式E测试接入

2026-09-13 master时间13:34:08保存新旧动态命令及启动记录，移除恢复的try_lock，由378693控制器第5次执行接续测试。接入前确认正式训练主进程已退出、前一动态命令仍为本模型训练入口、独立采样根不存在；after_lock保留，未使用kill_lock，没有覆盖训练产物。

实际release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_d3ffb62996bd/PocketXMol`，以已审查的43742cdf8166源码副本增加本次测试配置，生产代码不变。配置提交b666daa，LF字节SHA256为483efb6ee6c6340271b4a839340a0767ecc411a40c8d8bbc5269d0c3db2abbb0；release创建时再次解析核对保存的训练配置、best及全部测试预算。

实际launch为 `/home/penghongen/Feedback/PocketXMol/launches/378693/sample_B-E-T0-RB_test_job378693_20260913T133258`，节点时钟比master慢约1分钟。启动记录 `/storage/penghongen/PocketXMol/control/378693/sample_B-E-T0-RB_test_start.json`，同目录保存sample_B-E-T0-RB_test_run_cmd.sh和修改前的sample_B-E-T0-RB_test_before_run_cmd.sh。当前采样out／err从60987569／208301字节读取，避免混入此前训练日志。运行主进程33421的Slurm cgroup已核对。

## 训练完成与检查点只读核对

本次训练日志从 `/home/penghongen/Feedback/Pocket_Plus/allocations/378693/out` 和 `err` 的25869442／59362字节开始读取。控制器第4次执行成功，训练主进程57306退出，W&B完成上传。训练进度栏耗时7小时58分24秒，平均约1.06次更新／秒，无Traceback、OOM或非有限值记录；日志记录46个不同空E训练身份，验证跳过数为0。

在378693.3的8核CPU内加载last和实际best，确认stop_reason=plateau、decline_count=3、global_step=last_validation_step=30400，优化器和调度器末次学习率均为8.000000000000002e-7，第三次下降后没有继续更新。全部38个定期检查点存在，best21600的1236个model参数键均有限，训练W&B身份为423nfmpm。

调度器的相对改善阈值是1%，其用于比较的最低值为1.6360849142074585；最终best按原始val/loss最低值独立选择，故使用1.6228482723236084对应的21600步。这两种比较用途不同，不用调度器的比较值替代best选择。

以下是已执行的产物核对命令，不是正式训练、推理或评价命令；脚本只读取已有检查点并新写training_summary_20260913.json，未改写检查点。部署记录在本地 `tmp/pxm-20260913/check_e_rb_checkpoint.sh`。

```bash
srun --jobid=378693 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /storage/penghongen/PocketXMol/runtime/venv/bin/python -B /storage/penghongen/tmp/pocketxmol_checkpoint_20260913/inspect_e_rb.py
```

## E完整推理核对与正式CPU评价接入

采样控制器第5次执行正常结束，446个实例及22300个候选均成功。逐实例采样／模型推理耗时累计12638.163894／12564.934873秒，共44600次批量forward，峰值张量显存3250520064字节；累计耗时不是整个作业墙钟时间。没有Traceback、OOM或降低batch记录。

只读核查逐一对照冻结test.jsonl的实例身份、object_key、视图、种子和C5向量，检查每实例50个唯一候选编号、候选身份、有限置信度及姿态／置信度文件。实际run.json与d3ffb62996bd release中的配置完全一致，采样PID33421退出，after_lock与恢复的try_lock存在。完成证据为 `/storage/penghongen/PocketXMol/control/378693/sample_B-E-T0-RB_test_complete_20260913.json`。

以下仅为已生成产物的核对命令，不是正式采样或评价命令。脚本本地副本 `tmp/pxm-20260913/inspect_e_rb_sampling.py`，部署与执行入口 `check_e_rb_sampling.sh`；从已通过的E-RA检查脚本只替换模型身份、release、作业和原进程编号，没有修改生产代码或科学计算。

```bash
srun --jobid=378693 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /storage/penghongen/PocketXMol/runtime/venv/bin/python -B /storage/penghongen/tmp/pocketxmol_sampling_20260913/inspect_e_rb_sampling.py
```

2026-09-13 master时间17:45:24保存新旧动态命令及启动记录，核实前一命令确为本模型采样、完整产物核对通过、summary.json尚不存在，随后移除恢复的try_lock，控制器第6次接续正式评价。after_lock保留，没有使用kill_lock、删除候选或重跑采样。

评价沿用d3ffb62996bd release及b666daa配置，实际launch为 `/home/penghongen/Feedback/PocketXMol/launches/378693/evaluate_B-E-T0-RB_test_job378693_20260913T174435`。启动记录 `/storage/penghongen/PocketXMol/control/378693/evaluate_B-E-T0-RB_test_start.json`，同目录保存evaluate_B-E-T0-RB_test_run_cmd.sh和修改前的evaluate_B-E-T0-RB_test_before_run_cmd.sh。评价out／err读取起点61072223／208397字节。主进程29594的cgroup及CUDA不可见、OMP／MKL／OpenBLAS线程各1均已确认；17:48确认8个评价子进程29710至29717均约100%单核CPU，无Traceback，首批逐实例文件尚未产生。

## CPU评价完成与产物核对

18:50 master检查确认评价PID29594退出、控制器第6次执行成功、try_lock恢复及W&B zykj1eit的online_completed状态。随后在原378693作业的8核CPU中逐一核对446份assessment和22300份候选指标：实例身份、原候选编号、按self-ranking排序后的Top-1、Top-5和oracle、严格RMSD<2 Å阈值、三视图冻结分母均一致。候选错误、RMSD按编号回退和Traceback均为0。

完成证据为 `/storage/penghongen/PocketXMol/control/378693/evaluate_B-E-T0-RB_test_complete_20260913.json`。以下仅核对已有评价文件，不是正式评价命令，不重新计算RMSD；本地副本为 `tmp/pxm-20260913/inspect_e_rb_evaluation.py`，部署与执行入口为同目录check_e_rb_evaluation.sh。

```bash
srun --jobid=378693 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /storage/penghongen/PocketXMol/runtime/venv/bin/python -B /storage/penghongen/tmp/pocketxmol_evaluation_20260913/inspect_e_rb_evaluation.py
```

## 实验过程与失败尝试

以下为追溯用记录。已结束阶段中的“尚未”“随后”等表述仅说明当时状态；本文件开头的当前状态优先。历史命令不应再次执行，旧产物不作为有效模型或测试结果。

### 并行尝试F-6（尚在排队，不是失败）

2026-09-13 11:09只读squeue确认378588仍为PENDING／Priority。它是本模型的另一独立尝试，不能与当前A800重训混写。其旧提交记录不证明将来运行时已包含空E修复；实际源码、运行是否继续由负责该任务的原agent和用户处理。本次不变更该任务。

本记录依据 [科学契约](../../想法/方案草稿/9-8-科学契约.md)、[工程细节](../../想法/方案草稿/9-8-工程与实现细节.md)及用户2026-09-11新增资源授权，执行既定六模型中的第6个模型B-E-T0-RB。任务由“核查PocketXMol执行前准备（3: 独立做另外半边）”负责；用户接受重复计算，各次输出独立。

#### 正式配置与命令

配置 `configs/docking/F-6.yml` 复制B-E-T0-RB.yml，仅W&B显示名称为F-6_B-E-T0-RB。包络选择完整受体残基，残基重原子质量中心到任一真实配体重原子的距离严格小于10 Å；模型原点为实际输入受体重原子的坐标算术均值，始终关闭T1新增平移。使用RB核酸分支、官方初始参数和独立优化器及调度状态；batch72、累积1、bf16、15个DataLoader worker及每800更新原val/loss等参数不变。

在服务器 `/home/penghongen/My_Project/PocketXMol` 提交：

```bash
bash 训练与运行/submit_task.sh --sh train_docking.sh --resource h100 --gpus 1 --cpus 32 --after_hold --job-name F-6 -- F-6
```

该作业中的正式训练命令为：

```bash
bash 训练与运行/sh/train_docking.sh F-6
```

独立训练根 `/storage/penghongen/PocketXMol/training/F-6/`，首次配置train_config/F-6.yml；保留所有定期检查点和last。W&B使用pencounkdual-111/PocketXmol_raw、名称F-6_B-E-T0-RB，入口创建独立run id。另一对话的B-E-T0-RB产物不作为本任务恢复或测试来源。

训练完成后固定本任务最低原val/loss的实际best，再登记测试配置和正式命令；预定测试输出根 `/storage/penghongen/PocketXMol/sampling/F-6/`。只测试E，每实例50候选、100步、batch50，ALL/CAP10/HF10_TO5共用候选，随后8进程CPU评价并记录结果。保留训练中的val/loss，不运行训练后完整验证集采样或评价。

#### 验收与执行状态

主代理第一遍用原make_config确认新增配置只改变W&B名称；第二遍核对包络说明、正式命令、独立输出根和检查点来源。配置检查使用本地Python -X utf8及stdin，不运行科学样本，不是正式实验命令。没有新增或修改Python函数。独立代理对三配置、三日志完成两轮限定审查；官方失败产物说明及最新unstaged规则已修正并关闭窄核，无剩余问题。

已于master时间2026-09-11 16:26:08提交，实际JobId为 **378588**，Slurm名称F-6，单张H100、32核、after_hold，无pre_hold。首次squeue为PENDING/Priority，实际release/launch和W&B id在启动后补记。提交记录 `/storage/penghongen/PocketXMol/control/parallel-20260911/F-6_submit.json` 保存请求、完整命令及返回的作业编号。安全同步已完成，未删除远端文件；配置来源为f856ad8对应原模型加本次未提交的F-6.yml。

本任务只申请一张H100、32核、after_hold；保留after_lock和全部产物。排队或稳定运行时静默等待60/90分钟，每段Start-Sleep -Seconds 300；醒来再检查并通知。用户随后明确本对话所有文件始终保持unstaged，不执行git add或git commit；其他人的修改和暂存内容不处理。记录基准提交f856ad8及实际release/launch，不能把本次未提交配置标成已提交版本。

#### 计划与实现差异

并行执行及独立产物命名来自用户明确授权，科学定义与既定第6模型一致；训练、测试、CPU评价及报告尚未完成。
