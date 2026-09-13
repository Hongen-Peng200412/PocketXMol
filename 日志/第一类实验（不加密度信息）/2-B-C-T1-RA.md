# 2-B-C-T1-RA

**训练、C0／C5完整测试和评价均已完成。** ALL 的 Top-1 成功率为59.19%／42.15%；两协议各446实例、22300候选全部生成并评价成功。训练在22400次更新后停止。

更新核查：2026-09-13 11:09（服务器 master，UTC+8）。依据 [科学契约](../../想法/方案草稿/9-8-科学契约.md)、[工程细节](../../想法/方案草稿/9-8-工程与实现细节.md) 和 [边界清单](../../想法/方案草稿/9-8-边界与核查清单.md)，本文件统一保存本模型的训练、测试、评价与尝试历史。共同准备见 [实现与共同数据准备](../实现与共同数据准备.md)，全实验进度见 [总日志](../总日志.md)。

## 当前有效运行与产物

| 项目 | 当前事实 |
|---|---|
| 训练产物根 | `/storage/penghongen/PocketXMol/training/B-C-T1-RA/` |
| 正式测试检查点 | `checkpoints/step=20800.ckpt`；C5 val/loss=2.831010341644287 |
| 训练 W&B | [nzkna4ow](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/nzkna4ow) |
| 测试与评价产物 | `/storage/penghongen/PocketXMol/sampling/B-C-T1-RA/test/` |
| 评价 W&B | [n9ddvnid](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/n9ddvnid) |

训练保留原路径 `val/loss` 和最低损失检查点选择；训练结束后直接完整测试，不做训练后完整验证集采样。每实例每协议50候选、100步，原 self-ranking（原置信度及碰撞、立体化学项组成的候选排序）不变。完整结果优先放在下文，执行核查和失败尝试放在后部。

ALL为446个实例的完整测试集合。按完整模板身份object_key在ALL中的频数分组，只有频数大于10的身份才沿冻结顺序截取：CAP10保留前10个，HF10_TO5保留前5个；频数不超过10的身份全部保留。两个视图分别272和227个实例，与ALL复用候选，不重复生成或相加计数。

## 测试结果

RMSD单位为Å，沿原CalcRMS计算，不移动或刚体对齐预测姿态；成功阈值严格小于2 Å。Top-1使用原self-ranking最高的候选；Top-5在评分最高的5个候选内取最低RMSD；oracle在全部50个候选内取最低RMSD，表示候选池能够达到的结果，不用于选择实际输出。三个视图共用候选池，实例集合重叠，不能相加计作额外测试。

以下成功率和平均RMSD均按实例等权，括号内为成功实例数。ALL、CAP10、HF10_TO5分别包含77、67、65个PDB。

| 协议 | 视图 | 实例数 | Top-1成功率 | Top-5成功率 | Oracle成功率 | Top-1平均RMSD |
|---|---|---:|---:|---:|---:|---:|
| C0 | ALL | 446 | 59.19%（264） | 73.32%（327） | 87.00%（388） | 2.980 |
| C0 | CAP10 | 272 | 50.37%（137） | 65.44%（178） | 82.35%（224） | 3.480 |
| C0 | HF10_TO5 | 227 | 51.10%（116） | 66.08%（150） | 81.94%（186） | 3.376 |
| C5 | ALL | 446 | 42.15%（188） | 51.57%（230） | 60.54%（270） | 3.915 |
| C5 | CAP10 | 272 | 32.35%（88） | 41.54%（113） | 51.84%（141） | 4.501 |
| C5 | HF10_TO5 | 227 | 33.04%（75） | 42.73%（97） | 52.86%（120） | 4.459 |

PDB等权先在每个PDB内部平均其测试实例的成功指标，再对该视图中的PDB等权平均，避免含多个配体实例的PDB获得更大权重。

| 协议 | 视图 | Top-1成功率 | Top-5成功率 | Oracle成功率 |
|---|---|---:|---:|---:|
| C0 | ALL | 59.71% | 72.42% | 87.26% |
| C0 | CAP10 | 58.39% | 70.75% | 85.40% |
| C0 | HF10_TO5 | 56.45% | 70.22% | 84.56% |
| C5 | ALL | 43.56% | 53.00% | 58.47% |
| C5 | CAP10 | 42.41% | 52.35% | 57.86% |
| C5 | HF10_TO5 | 41.85% | 52.49% | 57.39% |

同一T1模型在C5条件下的ALL Top-1成功率比C0低17.04个百分点。逐实例配对后，92个实例由C0成功变为C5失败，16个由失败变为成功。C0与C5决定给定中心、选袋和局部原点，两种条件均执行T1后续中心相关加噪。

### 评分与姿态误差的对应

每个实例先用50个候选计算self-ranking与负RMSD的Spearman相关系数，再对实例等权平均或取中位数。Pose AUC是逐实例计算的ROC曲线下面积，以RMSD<2 Å为正类；只有一个类别时AUC无定义，该实例保留在成功率分母中，仅不进入AUC均值。所有实例的Spearman均有效，本次所有AUC缺失均来自单一类别。

| 协议 | 视图 | Spearman均值 | Spearman中位数 | Pose AUC均值 | AUC有效实例／全部实例 |
|---|---|---:|---:|---:|---:|
| C0 | ALL | 0.2885 | 0.2992 | 0.6954 | 338／446 |
| C0 | CAP10 | 0.2417 | 0.2425 | 0.6673 | 202／272 |
| C0 | HF10_TO5 | 0.2507 | 0.2510 | 0.6705 | 170／227 |
| C5 | ALL | 0.2169 | 0.2381 | 0.6990 | 243／446 |
| C5 | CAP10 | 0.1868 | 0.1950 | 0.6755 | 135／272 |
| C5 | HF10_TO5 | 0.2008 | 0.2060 | 0.6745 | 115／227 |

ALL中，C0有124个实例、C5有82个实例的候选池包含成功姿态，但Top-1没有选中。这是既定self-ranking的实测表现，不据此调整评分规则或新增评分器实验。

### 对接结果与口袋核酸占比

核酸占比为当前协议选出的标准RNA／DNA重原子数，除以标准蛋白与RNA／DNA重原子总数；同一实例在C0、C5下的比例可以不同。两协议各有439个纯蛋白口袋、7个含核酸口袋，7个含核酸实例全部属于三个测试视图。

| 协议 | 口袋分组 | 实例数 | Top-1成功数 | Top-5成功数 | Oracle成功数 |
|---|---|---:|---:|---:|---:|
| C0 | 核酸占比为0 | 439 | 260 | 322 | 381 |
| C0 | 核酸占比大于0 | 7 | 4 | 5 | 7 |
| C5 | 核酸占比为0 | 439 | 187 | 228 | 268 |
| C5 | 核酸占比大于0 | 7 | 1 | 2 | 2 |

以下完整列出7个含核酸实例；实例编号均为0，RMSD依次为Top-1／Top-5／oracle，单位Å。

| PDB | C0核酸占比 | C0 RMSD | C5核酸占比 | C5 RMSD |
|---|---:|---|---:|---|
| 9q16 | 19.43% | 1.158／0.925／0.925 | 10.05% | 2.833／2.736／2.715 |
| 9r3d | 12.23% | 7.300／1.989／1.324 | 4.69% | 6.757／5.872／4.195 |
| 9shy | 20.92% | 2.810／2.810／1.484 | 24.70% | 3.521／3.521／3.187 |
| 9v7o | 100.00% | 2.942／2.913／1.010 | 100.00% | 3.514／3.013／2.490 |
| 9z2n | 19.31% | 1.754／1.504／0.719 | 14.74% | 0.890／0.792／0.784 |
| 9z2u | 12.01% | 1.004／1.004／0.871 | 32.28% | 7.658／4.874／3.523 |
| 9z3d | 11.52% | 0.752／0.752／0.752 | 18.34% | 2.460／1.807／1.656 |

纯RNA的9v7o/0在两个协议下均完成50个候选及全部评价，并计入正式分母；C0只有oracle成功，C5的三项均未达到2 Å阈值。含核酸样本仅7个，以上结果支持逐实例描述，不能单凭这组样本确定核酸比例与成功率的稳定关系。

### 与正确T0-RA的同协议比较

比较对象为 [正确T0-RA的21600步best测试](1-B-C-T0-RA.md)，两模型使用相同测试清单、候选种子、C5偏移、候选预算、评价代码和RA分支。下表按ALL实例等权，差值为T1减T0，单位为百分点。

| 协议 | 指标 | T0成功数／446 | T1成功数／446 | 成功率差值 |
|---|---|---:|---:|---:|
| C0 | Top-1 | 268 | 264 | -0.90 |
| C0 | Top-5 | 323 | 327 | +0.90 |
| C0 | Oracle | 367 | 388 | +4.71 |
| C5 | Top-1 | 192 | 188 | -0.90 |
| C5 | Top-5 | 240 | 230 | -2.24 |
| C5 | Oracle | 275 | 270 | -1.12 |

C0的Top-1逐实例配对为41个由T0成功变为T1失败、37个反向变化；C5分别为38和34个。PDB等权的ALL Top-1则是T1较高，C0／C5分别提高2.36／1.06个百分点。两种加权方式给出的变化不同，当前结果没有给出T1在全部指标上一致改善的证据。T0训练验证使用C0、T1使用C5，其best验证损失不能直接横向比较；正式比较使用上面的共同测试协议，不依据测试结果重新选checkpoint或调整科学配置。

## 结果保存、核查与限制

正式根为 `/storage/penghongen/PocketXMol/sampling/B-C-T1-RA/test/`。各协议下的 `<pdb_id>/<occurrence_id>/candidate_metrics.json` 保存逐候选RMSD、评分和错误，assessment.json保存逐实例评价；协议occurrences.json、summary.json及test/summary.json保存完整汇总。W&B n9ddvnid已经上传全部规定汇总。

本地整理副本 `tmp/pxm-20260912/B-C-T1-RA-test-results.json` 来自上述服务器汇总及全部逐实例文件，正式产物以服务器目录为准。报告的成功数、PDB等权均值、Spearman／AUC汇总及逐实例配对均由这些文件重新计算，与正式summary一致；该核对是只读的结果验收，不是再次运行采样或评价。

评价阶段有584条RDKit“暂不支持allene-style立体化学并忽略”的提示，无Traceback。提示次数不代表独立异常实例数；原立体化学评价对该类结构的区分能力受RDKit支持范围限制。本次无RMSD回退，不因这些提示改图、改变分母或替换原评价规则。

## 固定训练条件

配置为 `configs/docking/B-C-T1-RA.yml`。训练每次抽一份半径0–5 Å均匀、方向球面均匀的delta，用完整g+delta选袋并定原点；原高斯后加入s*delta，监督目标和受体不跟随该新增平移。原val/loss读取验证清单冻结C5偏移及同一T1公式。g为完整配体重原子中心，s=1-level_dict['pos']。正式姿态评价仍保留C0与C5，推理首步为原纯高斯，后续保留中心相关重新加噪。

从规定官方pxm参数初始化，重新建立AdamW及调度状态；单张A800使用batch_size=72、累积1、bf16，名义全局批量72。初始lr=1e-4、warmup=0，每800个优化器更新计算原val/loss；Plateau相对阈值1%、patience=5、factor=0.2，第三次实际下降停止，上限40000步。最低原验证损失选择best，last和全部定期检查点保留。

## 正式运行命令与产物

在既有371591的独立冻结release中运行：

```bash
bash 训练与运行/sh/train_docking.sh B-C-T1-RA
```

产物目录为 `/storage/penghongen/PocketXMol/training/B-C-T1-RA/`，W&B账号／项目为 `pencounkdual-111/PocketXmol_raw`，新run独立创建。[正确T0的测试推理与CPU评价](1-B-C-T0-RA.md)完成并记录后进入本实验；实际launch和run id见下文。本模型训练后也直接测试，不安排完整验证集采样；训练期间原val/loss和best选择保留。

本次执行沿用已审查和实际运行的ec06dbd源码，使用既有不可变release `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_fa0d957b2d3f/PocketXMol` 中的B-C-T1-RA.yml与train_docking.sh。该配置明确center_translation=true、RA、72×1、原官方初始化和独立输出目录；本次不传--resume。运行来源由该release确定，不吸收其他代理的未提交工作区文件。

## 固定输入与正式命令

配置为 `configs/docking/sample-B-C-T1-RA-test.yml`，使用训练时保存的 `/storage/penghongen/PocketXMol/training/B-C-T1-RA/train_config/B-C-T1-RA.yml` 和同训练根下 `checkpoints/step=20800.ckpt`。RA受体分支、中心T1机制保持：首步原纯高斯；后续原高斯后增加s乘以局部给定中心与预测重原子质心之差，s=1-level_dict['pos']。C0与C5只决定给定中心和固定口袋，均执行T1采样；定位后不再读取真值中心或真值偏移。

测试ALL清单446个实例，每协议每实例50个候选、100步、batch_size=50，沿用冻结候选种子和C5偏移。CAP10的272个实例和HF10_TO5的227个实例复用同一候选池。原置信度头、轨迹汇总、self-ranking与CPU评价口径均沿用已验收实现。

在371591所用独立release中执行正式测试推理：

```bash
bash 训练与运行/sh/sample_docking.sh B-C-T1-RA-test
```

按用户2026-09-12新增安排，推理全部完成后，在同一371591作业中从同一release执行正式评价，使用自带CPU和现有8个评价进程：

```bash
bash 训练与运行/sh/evaluate_docking.sh B-C-T1-RA-test
```

推理和评价产物为 `/storage/penghongen/PocketXMol/sampling/B-C-T1-RA/test/`；各协议的逐实例候选、姿态、评价文件与汇总均保留。W&B评价使用 `pencounkdual-111/PocketXmol_raw`，名称B-C-T1-RA_test，实际run id为n9ddvnid。371591的A800和16核CPU继续保留；评价脚本关闭CUDA，仅使用分配内CPU，不另提交纯CPU作业，after_lock保留。

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

核查报告保存为 `/storage/penghongen/PocketXMol/training/B-C-T1-RA/training_summary_20260912.json`。报告验证了停止步、下降次数、W&B身份、全部定期检查点存在、best确为记录中的最低损失及best文件实际步数；不修改任何checkpoint。测试配置和后续运行证据见本文件的测试与评价记录。

## 执行与核查记录

本节按实际事件保留执行证据；其中启动阶段的进度与后续计划属于当时记录，当前完成状态以文档开头为准。

### 实际启动

2026-09-12，确认前一模型测试报告完成、371591处于try_lock等待、after_lock保留且目标训练目录不存在后，接入上述正式命令。master请求时间为00:42:12；gnode09时钟约慢3分钟。控制器第14次执行的PocketXMol launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/train_B-C-T1-RA_job371591_20260912T003912`，实际训练主进程为16005，cgroup确认属于371591。

启动元数据为 `/storage/penghongen/PocketXMol/control/371591/train_B-C-T1-RA_start.json`，正式动态命令副本为同目录train_B-C-T1-RA_run_cmd.sh。该次out／err读取起点分别为57342372／73878；旧T0训练、已取消validation和已完成test日志不混入本次状态判断。after_lock保留，运行时try_lock和kill_lock均不存在。

W&B训练run为 [nzkna4ow](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/nzkna4ow)，名称B-C-T1-RA。输出明确从规定的官方pocketxmol.ckpt加载模型参数、使用bf16，约20.2M参数全部可训练；目标目录独立创建。启动检查时已完成62个训练批次（72×1对应62次优化器更新），约1.01步／秒，lr=1e-4，原loss及置信度损失正常记录，无OOM或Traceback。首次正式验证在800次更新时进行，最终结果见下文。

运行沿用已通过自查、两轮独立审查及CPU／GPU验收的实现，不增加科学开关或重建共同资产。测试命令和验收结果见[共同准备](../实现与共同数据准备.md)及[第1模型的纠偏记录](1-B-C-T0-RA.md)，本文件上面的命令是正式训练命令。371591的after_lock保留，不申请额外GPU，不释放或删除旧T0产物。

### 运行前核查

20800步best由Slurm内的训练检查点核查确认，具体命令及结果见训练日志。本次只增加明确best的测试配置，模型、特征化、噪声链、评价代码和共同资产均沿用既有版本。主代理第一遍按实际消费字段核对配置职责、保存配置和best路径，第二遍核对注释中的原点、实际噪声强度及T1步骤。未新增或修改生产函数。

配置验收：本地yaml.safe_load解析通过；与已完成的T0测试配置比较，仅模型名、保存训练配置、checkpoint、T1开关、输出根及W&B名称发生预期变化，其余字段完全一致；C0/C5 test、RA、batch50、50候选、100步及8核评价断言通过。独立代理t1_test_config_review完成一次仅针对本配置及本文的窄范围只读核查，确认实际字段消费与科学注释一致，无阻断问题。此前代码链的两轮全面独立审查已结束，本次不重复扩大范围。

本次release以已审查的ec06dbd运行副本为基础，只加入此测试配置；不从共享工作区复制其他代理的未提交文件。配置及训练完成记录提交为69a9507。服务器临时装配目录为 `/storage/penghongen/tmp/pocketxmol_t1_test_20260912/PocketXMol`；与原fa0d957b2d3f运行副本逐文件比较，唯一差异是新增sample-B-C-T1-RA-test.yml。部署时69a9507版本的本地和服务器配置SHA256均为ed5888a9b23addd68a6c169ef5d94dcece59de366223820d7bcb462e4c03b42f；之后本地仅更新评价资源注释，运行副本不变。这是部署核对，不引入新的科学数据登记机制。

### 正式启动

2026-09-12，确认371591运行于gnode09、前一训练正常完成、after_lock与try_lock存在、kill_lock不存在、测试输出根尚不存在后，接入上述正式推理命令。master请求时间06:58:52，gnode09时钟约慢3分钟；控制器第15次执行，实际采样主进程57606，cgroup确认属于371591。

本次实际运行release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_55258ac14af2/PocketXMol`，PocketXMol launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/sample_B-C-T1-RA_test_job371591_20260912T065551`。旧资源控制器外层的Pocket_Plus release只承载锁控制，本次科学代码与配置由上述PocketXMol release固定。

启动元数据为 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T1-RA_test_start.json`，动态命令副本为同目录sample_B-C-T1-RA_test_run_cmd.sh。本次out／err读取起点83039931／82141；不把旧训练或前一测试日志混作本次证据。运行时after_lock保留，try_lock和kill_lock不存在。实际候选和完成情况在核对产物后记录。

启动核对：模型严格加载成功后写入test/run.json，明确20800步best、T1、C0/C5、batch50及50×100预算。首两个C0实例11jb/0和11jb/1均完成50／50候选、100次批量forward，耗时34.98／33.48秒，峰值张量显存约2.50 GB；未发现Traceback或CUDA OOM。这里只确认正式候选生成正常，姿态质量由完整推理后的CPU评价确定。

### 完整测试候选生成完成

2026-09-12，控制器第15次执行成功，采样主进程57606退出，try_lock恢复，after_lock保留，kill_lock不存在。两协议各446个实例、22300个候选全部生成成功；本次日志无Traceback或CUDA OOM。完成核对记录为 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T1-RA_test_complete_20260912.json`。

已逐实例核对与冻结test清单的身份集合、object_key、候选种子、C5偏移和视图完全相同；每实例严格50候选、100步、一批50、100次已完成forward。所有候选身份、编号、状态及姿态／置信度文件存在性核对通过。未重新生成清单或按测试表现筛选实例。

| 协议 | 实例／候选 | 实例耗时合计（秒） | 推理耗时合计（秒） | 已完成批量forward | 峰值分配显存（字节） |
|---|---|---|---|---|---|
| C0 | 446／22300 | 13400.881 | 13282.236 | 44600 | 3209630720 |
| C5 | 446／22300 | 13128.801 | 13037.641 | 44600 | 3205350400 |

实例耗时合计约7小时22分10秒，不含模型启动及协议之间的少量开销。候选生成成功只说明产物成功写出；姿态质量由随后在371591已分配CPU内完成的评价确定，使用8个评价进程。

### 同一作业内的CPU评价

完成上述核对后，于2026-09-12 master时间14:30:34接入正式命令 `bash 训练与运行/sh/evaluate_docking.sh B-C-T1-RA-test`。继续使用371591、gnode09已经分配的16核CPU，配置保留8个评价进程，没有另申请CPU作业。实际评价主进程33502，cgroup确认属于371591，CUDA_VISIBLE_DEVICES为空。

代码和采样配置继续来自同一55258ac14af2 release，控制器第16次执行；本次独立launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/evaluate_B-C-T1-RA_test_job371591_20260912T142745`。启动记录 `/storage/penghongen/PocketXMol/control/371591/evaluate_B-C-T1-RA_test_start.json`，同目录保存evaluate_B-C-T1-RA_test_run_cmd.sh。本次out／err起点83208404／82237，after_lock保留。

控制器第16次执行成功结束，评价进程33502退出，try_lock恢复，after_lock保留，kill_lock不存在。完成核对记录为 `/storage/penghongen/PocketXMol/control/371591/evaluate_B-C-T1-RA_test_complete_20260912.json`，核对时间为2026-09-12 16:36（北京时间）。W&B汇总run n9ddvnid状态为online_completed、error为空，与训练run独立。

已逐实例核对冻结清单身份、独立assessment.json与协议occurrences.json完全一致；每实例50条candidate_metrics，编号0至49，全部RMSD和self-ranking有限、错误列表为空。两协议各22300条候选指标，无资产错误、候选错误或按原子编号匹配的RMSD回退。Top-1／Top-5／oracle成功标记与严格小于2 Å的原始RMSD一致。

C0／C5逐实例评价耗时合计分别为22771.545／22744.151秒。这是8进程并行执行时各实例耗时的累加，不能当作墙钟时长；本次没有单独记录评价阶段的精确墙钟时长，不用整个371591作业时长替代。所有评价均使用已生成的候选，没有补生成或增加评分系统。

### 计划与实现差异

本模型训练、规定测试和CPU评价已完成，未发现科学契约的有害差异。按用户09-12资源安排，评价在原371591作业内完成，属于执行资源调整；候选池、评价指标和分母不变。本记录范围已经完成，六模型总任务的其余训练与测试按已指定两张A800继续。

## 实验过程与失败尝试

未记录需要废弃的正式训练或测试尝试。运行前验收和正式启动证据保留在上一节，不另建实验文件。
