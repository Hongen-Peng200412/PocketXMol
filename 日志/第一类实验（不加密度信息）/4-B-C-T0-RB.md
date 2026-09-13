# 4-B-C-T0-RB

**正式训练、C0／C5完整测试、CPU评价和报告核对均已完成。** 两协议各446个实例、22300个候选全部生成并评价成功；ALL Top-1成功率为C0 59.64%、C5 40.36%。训练在31200次更新后因第三次学习率下降停止，实际best为21600步、原C0 val/loss=1.8065478801727295，是全部39个定期检查点的最低原验证损失；1236个模型参数键均有限。全部检查点、last及W&B记录保留。

更新核查：2026-09-14 00:51（服务器 master，UTC+8）。依据 [科学契约](../../想法/方案草稿/9-8-科学契约.md)、[工程细节](../../想法/方案草稿/9-8-工程与实现细节.md) 和 [边界清单](../../想法/方案草稿/9-8-边界与核查清单.md)，本文件统一保存本模型的训练、测试、评价与尝试历史。共同准备见 [实现与共同数据准备](../实现与共同数据准备.md)，全实验进度见 [总日志](../总日志.md)。

## 当前有效运行与产物

| 项目 | 当前事实 |
|---|---|
| 训练产物根 | `/storage/penghongen/PocketXMol/training/B-C-T0-RB/` |
| 正式测试检查点 | `checkpoints/step=21600.ckpt`，相对于上述训练根；原C0 val/loss=1.8065478801727295 |
| 训练 W&B | [hthglbuy](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/hthglbuy) |
| 测试与评价产物 | `/storage/penghongen/PocketXMol/sampling/B-C-T0-RB/test/`；C0与C5各446实例的候选和评价均已完成 |
| 评价 W&B | [d1osmk6j](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/d1osmk6j)，上传完成 |

训练保留原路径 `val/loss` 和最低损失检查点选择；训练结束后直接完整测试，不做训练后完整验证集采样。每实例每协议50候选、100步，原 self-ranking（原置信度及碰撞、立体化学项组成的候选排序）不变。完整结果优先放在下文，执行核查和失败尝试放在后部。

ALL为446个实例的完整测试集合。按完整模板身份object_key在ALL中的频数分组，只有频数大于10的身份才沿冻结顺序截取：CAP10保留前10个，HF10_TO5保留前5个；频数不超过10的身份全部保留。两个视图分别272和227个实例，与ALL复用候选，不重复生成或相加计数。

## 测试结果

RMSD单位为Å，沿原CalcRMS计算，不移动或刚体对齐预测姿态。Top-1取原self-ranking评分最高的候选，Top-5取评分最高的5个候选中的最低RMSD，oracle取全部50个候选中的最低RMSD；成功阈值均为严格小于2 Å。oracle只表示候选池中的最好姿态，不用于选择实际输出。

下表按实例等权，括号为成功数。ALL、CAP10、HF10_TO5分别包含77、67、65个PDB，三个视图复用候选且实例集合重叠，不相加计数。

| 协议 | 视图 | 实例数 | Top-1成功率 | Top-5成功率 | Oracle成功率 | Top-1平均RMSD |
|---|---|---:|---:|---:|---:|---:|
| C0 | ALL | 446 | 59.64%（266） | 71.97%（321） | 81.84%（365） | 3.142 |
| C0 | CAP10 | 272 | 47.79%（130） | 63.97%（174） | 76.84%（209） | 3.763 |
| C0 | HF10_TO5 | 227 | 46.26%（105） | 63.88%（145） | 77.09%（175） | 3.704 |
| C5 | ALL | 446 | 40.36%（180） | 53.14%（237） | 62.56%（279） | 4.272 |
| C5 | CAP10 | 272 | 30.51%（83） | 42.65%（116） | 52.21%（142） | 4.926 |
| C5 | HF10_TO5 | 227 | 29.96%（68） | 43.17%（98） | 52.42%（119） | 4.838 |

PDB等权先在每个PDB内部平均测试实例成功指标，再对该视图中的PDB平均。

| 协议 | 视图 | Top-1成功率 | Top-5成功率 | Oracle成功率 |
|---|---|---:|---:|---:|
| C0 | ALL | 57.41% | 72.25% | 83.49% |
| C0 | CAP10 | 55.14% | 70.88% | 81.75% |
| C0 | HF10_TO5 | 52.73% | 69.98% | 81.18% |
| C5 | ALL | 40.92% | 50.42% | 64.28% |
| C5 | CAP10 | 37.73% | 46.78% | 60.63% |
| C5 | HF10_TO5 | 35.54% | 45.39% | 59.68% |

同一T0-RB的C5 ALL Top-1比C0低19.28个百分点；逐实例配对为100个从C0成功变为C5失败，14个反向变化。两协议均保留原T0采样机制，差别是按各自给定中心固定口袋和原点。

### 评分与姿态误差的对应

每个实例用50个候选计算self-ranking与负RMSD的Spearman相关系数，再对实例求均值或中位数。Pose AUC是以RMSD<2 Å为正类的逐实例ROC曲线下面积；仅有一种类别时无定义，该实例仍计入成功率分母。所有Spearman均有效，下表AUC缺失均由单一类别造成。

| 协议 | 视图 | Spearman均值 | Spearman中位数 | Pose AUC均值 | AUC有效实例／全部实例 |
|---|---|---:|---:|---:|---:|
| C0 | ALL | 0.2190 | 0.2080 | 0.6654 | 315／446 |
| C0 | CAP10 | 0.2151 | 0.2093 | 0.6825 | 193／272 |
| C0 | HF10_TO5 | 0.2353 | 0.2400 | 0.6927 | 167／227 |
| C5 | ALL | 0.1996 | 0.2059 | 0.6741 | 259／446 |
| C5 | CAP10 | 0.1873 | 0.1765 | 0.6839 | 140／272 |
| C5 | HF10_TO5 | 0.1842 | 0.1800 | 0.6878 | 119／227 |

C0、C5各有99个实例的候选池包含成功姿态，但原self-ranking的Top-1未选中；本次保留原排序规则和候选池。

### 对接结果与口袋核酸占比

核酸占比为当前协议选出的标准RNA／DNA重原子数除以标准蛋白与RNA／DNA重原子总数。两个协议各439个纯蛋白口袋、7个含核酸口袋；7个含核酸实例均属于三个测试视图。

| 协议 | 口袋分组 | 实例数 | Top-1成功数 | Top-5成功数 | Oracle成功数 |
|---|---|---:|---:|---:|---:|
| C0 | 核酸占比为0 | 439 | 262 | 315 | 358 |
| C0 | 核酸占比大于0 | 7 | 4 | 6 | 7 |
| C5 | 核酸占比为0 | 439 | 178 | 235 | 274 |
| C5 | 核酸占比大于0 | 7 | 2 | 2 | 5 |

全部含核酸实例列于下表，occurrence编号均为0；RMSD依次为Top-1／Top-5／oracle，单位Å。

| PDB | C0核酸占比 | C0 RMSD | C5核酸占比 | C5 RMSD |
|---|---:|---|---:|---|
| 9q16 | 19.43% | 2.591／1.810／0.937 | 10.05% | 2.711／2.470／2.229 |
| 9r3d | 12.23% | 2.741／2.741／1.802 | 4.69% | 7.854／4.877／4.449 |
| 9shy | 20.92% | 2.941／1.489／1.188 | 24.70% | 3.663／3.313／1.683 |
| 9v7o | 100.00% | 1.611／1.611／1.396 | 100.00% | 3.528／2.874／1.374 |
| 9z2n | 19.31% | 1.066／1.045／0.748 | 14.74% | 0.857／0.827／0.687 |
| 9z2u | 12.01% | 1.111／0.793／0.647 | 32.28% | 8.624／2.100／1.212 |
| 9z3d | 11.52% | 0.822／0.731／0.731 | 18.34% | 1.157／1.035／0.860 |

纯RNA实例9v7o/0在两个协议均完成50个候选及全部评价，保留在正式分母；C0三项均成功，C5只有oracle成功。含核酸实例仅7个，不能据此单独确定核酸占比与成功率的稳定关系。

### 与正确T0-RA的同协议比较

比较对象为[正确T0-RA的21600步best完整测试](1-B-C-T0-RA.md)，产物来自B-C-T0-RA-C0/test，排除旧错误T0结果。两模型均按各自原C0 val/loss选best，使用相同冻结测试清单、种子、C5向量、50×100预算、T0采样与评价；主要实验差异为RA联合编码受体、RB分别编码蛋白与核酸。下表按ALL实例等权，差值为RB减RA，单位百分点。

| 协议 | 指标 | RA成功数／446 | RB成功数／446 | 成功率差值 |
|---|---|---:|---:|---:|
| C0 | Top-1 | 268 | 266 | -0.45 |
| C0 | Top-5 | 323 | 321 | -0.45 |
| C0 | Oracle | 367 | 365 | -0.45 |
| C5 | Top-1 | 192 | 180 | -2.69 |
| C5 | Top-5 | 240 | 237 | -0.67 |
| C5 | Oracle | 275 | 279 | +0.90 |

C0的Top-1逐实例配对为14个从RA成功变为RB失败、12个反向变化；C5分别为21和9个。PDB等权ALL Top-1差值为C0 +0.07、C5 -1.58个百分点。C5的RB oracle较高而Top-1较低，当前单次训练结果没有在各指标上给出一致分支优劣；核酸分支仍由用户选择，不据测试结果更换best或增加实验。

## 结果保存与限制

正式测试与评价根为 `/storage/penghongen/PocketXMol/sampling/B-C-T0-RB/test/`。该根下的summary.json保存全部协议汇总；每个协议的occurrences.json和summary.json保存逐实例与三视图统计，协议下的`<pdb_id>/<occurrence_id>/candidate_metrics.json`、assessment.json保存逐候选及逐实例评价。评价W&B d1osmk6j已上传规定汇总。

本地副本 `tmp/pxm-20260914/B-C-T0-RB-test-results.json`来自上述服务器summary及完整逐实例文件；正式产物以服务器目录为准。`tmp/pxm-20260914/report_c_t0_rb.py`核对成功数、实例与PDB等权成功率、RMSD均值、Spearman／AUC、核酸分组及逐实例配对，输出REPORT_AGGREGATE_AUDIT_PASS；这是结果核对，没有重新采样或计算RMSD。

评价stderr有604条RDKit“暂不支持allene-style立体化学并忽略”的提示，不代表604个独立失败实例；原立体化学评价受当前RDKit支持范围限制。全部候选指标有限，候选错误、RMSD按原子编号匹配回退和Traceback均为0；没有因此改变图、分母或评价规则。

主代理先核对正式来源、数字、保留历史和命令边界，再核对指标定义、单位、比较口径、中文表达及链接；独立代理随后完成两轮同范围核查，35条数字表格、逐实例配对、AUC缺失原因及上述完成事实均通过。本次没有修改生产Python函数，不重新扩大已关闭的代码审查。

## 最后一次进度核查

采样控制器第22次执行成功，主进程39974已退出，实际run.json与已审查的RB、T0、C0／C5及best21600配置一致。C0、C5各446份result.json均为complete，50候选、100步、batch50均符合预算，共44600个候选成功、失败0。两协议逐实例采样耗时分别累计13137.32／13015.68秒，此数不是整个作业墙钟时间；没有NaN、Traceback、OOM或降低batch记录。随后CPU评价第23次执行成功，主进程44073退出；after_lock及恢复的try_lock存在，kill_lock不存在，训练产物未覆盖。首次接入核查时，11jb/0、11jb/1各完成50个候选和100次批量forward，耗时34.03／33.38秒，峰值张量显存约2.54 GB。

完整训练摘要为 `/storage/penghongen/PocketXMol/training/B-C-T0-RB/training_summary_20260913.json`。完整候选、逐候选评价、三个视图及核酸占比报告均已核对；本模型已完成，不安排完整验证集采样。

## 正式测试配置与命令

`configs/docking/sample-B-C-T0-RB-test.yml`读取本次保存的训练配置及21600步best，receptor_branch=RB、center_translation=false、protocols=[C0,C5]、split=test。与正确T0-RA的测试配置相比，仅替换RB分支、模型名称及各自训练／产物身份；冻结C5偏移、种子、50候选、100步、batch50和8进程评价保持。

中心推理始终按实际给定中心选袋及定原点；T0+C5保留冻结偏移后的输入中心，首步及后续均使用原T0高斯链，不强制候选质心归零。定位后采样器不读取真值中心或偏移，输出只加回模型原点一次，不做最终质心对齐。

本次没有新增或修改生产Python函数。主代理第一遍核对配置读取关系、模型与数据分支、检查点和输出身份，第二遍核对YAML注释及日志含义；独立代理随后在同一范围完成两轮只读核查，均通过。YAML解析比较确认相对正确T0-RA仅有六个模型身份、路径与分支字段变化，科学条件及预算保持；未扩大既有生产代码审查。

以下正式采样命令已于2026-09-13接入371591执行；release、launch和启动记录见下文。

```bash
bash 训练与运行/sh/sample_docking.sh B-C-T0-RB-test
```

两协议完整候选核对通过后，以下正式CPU评价命令已于2026-09-13 23:16接入同一A800作业执行：

```bash
bash 训练与运行/sh/evaluate_docking.sh B-C-T0-RB-test
```

## 中心T0与RB契约

配置为 `configs/docking/B-C-T0-RB.yml`。训练及原val/loss均使用C0：给定中心和模型原点C为完整配体重原子几何中心g，按标准受体残基重原子质量中心到g的距离严格小于15 Å选取完整残基。配体和受体共同减去C，局部监督目标x*的几何中心为0；只执行原高斯 `x_in=x*+s*σ(N)*ε`，s为1-level_dict['pos']，沿用原GaussianExplodePrior的尺度和逐原子噪声。center_translation=false，训练不抽新增中心偏移、不增加整分子平移。

RB分别构建和编码标准蛋白与核酸受体图，从官方权重初始化后复制蛋白编码器参数给核酸编码器。保留原主体模型、loss、置信度头及其目标、同构重分配和原固定字段恢复。

正式测试沿用446个实例的C0和冻结C5条件。T0+C5保持实际给定的偏移中心选袋及定原点，后续仍仅用T0原高斯；首步使用原纯高斯先验，不强制候选实际质心归零。定位后采样器不读取真值中心或偏移，输出只加回实际模型原点一次，不进行最终质心对齐。

## 固定训练配置与来源

源码使用 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_43742cdf8166/PocketXMol`，基线ec06dbd已含中心T0纠偏，追加c9cc4a1仅处理空E数据流，不改变中心模型。中心T0完整路径验收、原官方权重及72×1训练验收已经通过；RA／RB模型接线和既有生产代码已完成两遍自查及两轮独立审查。本次没有新增或修改生产函数，不扩大科学范围或重复验收。

配置解析核对：相对同release的B-C-T0-RA.yml，只将model.nucleic_branch改为RB、train.wandb.name改为B-C-T0-RB，其余字段一致；pocket_mode=center、center_translation=false，实际DataModule据此选C0训练与监督验证。冻结训练清单65290、验证781、校准361、测试446保持，E专用空口袋跳过规则不改变中心数据流。目标训练根已确认不存在，正式接入时再次核对。

训练从规定官方pocketxmol.ckpt参数开始，不传--resume，独立创建AdamW、调度器及W&B身份。batch72、累积1、bf16、15个数据worker，有效全局批量72。AdamW初始lr=1e-4、weight_decay=0.001、betas=0.99／0.999、eps=1e-8、warmup=0；每800次优化器更新计算原val/loss，Plateau相对阈值1%、patience5、factor0.2，第三次实际下降立即停止，上限40000步。best按最低原C0 val/loss选择，last和全部定期检查点保留。

## 正式训练命令与产物

以下是正式训练命令，产物核对或验收命令另行记录：

```bash
bash 训练与运行/sh/train_docking.sh B-C-T0-RB
```

独立训练根为 `/storage/penghongen/PocketXMol/training/B-C-T0-RB/`，保存的训练配置及检查点分别位于train_config和checkpoints子目录。W&B使用pencounkdual-111/PocketXmol_raw、名称B-C-T0-RB，实际run id为hthglbuy。

371591控制目录仍为 `/home/penghongen/Feedback/Pocket_Plus/allocations/371591/`。前一E评价正常结束且try_lock恢复后，保存新旧动态命令与启动记录，再移除try_lock运行。after_lock和此前有效及异常产物全部保留，不使用scancel。

训练完成后直接用实际best测试C0／C5，每实例每协议50候选、100步、batch优先50；之后用同一作业8个CPU评价进程完成三个视图和核酸比例分析。明确best的采样配置及短正式命令在训练结束后登记，不执行训练后完整验证集采样，不额外增加实验。

## 正式接入

2026-09-13 master时间06:14:42保存新旧动态命令及启动记录，移除try_lock后由371591控制器第21次执行启动。接入前再次确认目标训练根不存在、E-RA评价进程退出且W&B汇总成功，现有动态命令确为本任务E-RA评价入口。after_lock保留，没有使用kill_lock。

实际源码release为43742cdf8166，登记提交3101cff；配置SHA256为19809af372ac4c127efd8dab8c3850992cbb1164211a258eaebf4a4b0d6f8279。launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/train_B-C-T0-RB_job371591_20260913T061132`。启动记录 `/storage/penghongen/PocketXMol/control/371591/train_B-C-T0-RB_start.json`，同目录保存train_B-C-T0-RB_run_cmd.sh和修改前的train_B-C-T0-RB_before_run_cmd.sh。累计out／err起点113849959／311189。

实际主进程7586和15个数据worker均属于371591，命令行没有--resume，训练输出指向独立B-C-T0-RB根。日志确认从规定官方pocketxmol.ckpt加载模型，使用bf16，全部模型参数参与训练。W&B新运行是[hthglbuy](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/hthglbuy)，本地记录为训练根wandb/run-20260913_061159-hthglbuy。启动检查28步时原损失及置信度损失均有限，没有NaN、Traceback、OOM或reduce_batch；完整训练和原C0验证继续执行。

## 计划与实现差异

本实验沿用已纠正的中心T0契约和既定RB配置，没有新增科学开关、代码或资产重建。完整训练及best核对已完成，C0／C5完整推理和候选核对已完成，CPU评价及完整报告均已核对完成。

## 正式C0／C5测试接入

2026-09-13 master时间14:58:53保存新旧动态命令及启动记录，移除恢复的try_lock，由371591控制器第22次执行接续测试。接入前确认训练主进程7586退出、前一动态命令仍为本模型训练入口、独立采样根不存在；after_lock保留，未使用kill_lock，没有覆盖训练产物。

实际release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_07ea8b8a2406/PocketXMol`，以已审查43742cdf8166源码副本增加本次测试配置，生产代码不变。配置提交933a49c，LF字节SHA256为5968646c8d56be437a0131df0110ab22bce611bd4d242d6a6b66b55478499650；创建release时再次核对保存的训练配置、实际best及全部测试预算。

实际launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/sample_B-C-T0-RB_test_job371591_20260913T145551`，节点时钟比master慢约3分钟。启动记录 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T0-RB_test_start.json`，同目录保存sample_B-C-T0-RB_test_run_cmd.sh和修改前的sample_B-C-T0-RB_test_before_run_cmd.sh。当前采样out／err从149921972／321399字节读取，避免混入此前训练日志。采样主进程39974的Slurm cgroup已核对。

## 完整候选核对与正式CPU评价接入

采样控制器第22次执行成功，PID39974退出；两协议各446个实例、22300个候选，合计44600个候选全部生成成功。C0／C5逐实例采样耗时累计13137.317372／13015.679950秒，模型推理累计13051.676773／12948.097865秒，各44600次批量forward；峰值张量显存分别3234062848／3227094528字节。累计耗时不是整个作业墙钟时间。

在原371591作业的8核CPU中，只读核对实际run.json与07ea8b8a2406配置、冻结test.jsonl的身份、object_key、视图、种子和C5向量，逐一检查每实例50个唯一候选编号、有限置信度及姿态／置信度文件。T0新增平移保持关闭，C0、C5两个协议均完整通过。完成证据 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T0-RB_test_complete_20260913.json`，无Traceback、OOM或降低batch记录。

以下仅为已保存候选的核对命令，不是正式推理或评价命令。本地副本为 `tmp/pxm-20260913/inspect_c_t0_rb_sampling.py`，执行入口为同目录check_c_t0_rb_sampling.sh。复用已通过的T1-RB产物核查结构，仅改本模型身份、best、T0预期值、release、作业与原进程编号，未更改科学计算。

```bash
srun --jobid=371591 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /storage/penghongen/PocketXMol/runtime/venv/bin/python -B /storage/penghongen/tmp/pocketxmol_sampling_20260913/inspect_c_t0_rb_sampling.py
```

2026-09-13 master时间23:16:22，确认两协议完整核对通过、前一动态命令确为本模型采样、summary.json尚不存在，保存新旧命令及启动记录后移除恢复的try_lock，控制器第23次接续CPU评价。after_lock保留，没有使用kill_lock、删除候选或重跑推理。

评价沿用07ea8b8a2406 release和933a49c测试配置，实际launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/evaluate_B-C-T0-RB_test_job371591_20260913T231331`，节点时钟比master慢约3分钟。启动记录 `/storage/penghongen/PocketXMol/control/371591/evaluate_B-C-T0-RB_test_start.json`，同目录保留evaluate_B-C-T0-RB_test_run_cmd.sh及修改前的evaluate_B-C-T0-RB_test_before_run_cmd.sh。评价out／err读取起点150090468／321495字节。

23:17 master检查无Traceback，尚无首批assessment；主进程44073及8个子进程44162至44169均已出现。主进程Slurm cgroup、CUDA_VISIBLE_DEVICES为空、OMP／MKL／OpenBLAS各1线程和实际launch均已核对。评价W&B在最终汇总阶段创建，当前尚无评价run id。

## CPU评价完成与逐候选核对

2026-09-14 00:51 master检查确认评价主进程44073退出、控制器第23次执行成功，after_lock及恢复的try_lock存在、kill_lock不存在。C0／C5各446份assessment及22300份候选指标完成，W&B d1osmk6j为online_completed且error为空。

在371591的8核CPU中核对全部44600个候选的编号、有限RMSD与self-ranking、排序及严格RMSD<2 Å的Top-1／Top-5／oracle，逐实例与协议汇总一致，三个视图分母正确。完成证据为 `/storage/penghongen/PocketXMol/control/371591/evaluate_B-C-T0-RB_test_complete_20260914.json`。本地脚本与执行记录分别为tmp/pxm-20260914/inspect_c_t0_rb_evaluation.py和check_c_t0_rb_evaluation.sh。

以下是已执行的产物核对命令，不是正式推理或评价命令；它只读现有科学产物，并新写核对证据，不重新计算RMSD。

```bash
srun --jobid=371591 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /storage/penghongen/PocketXMol/runtime/venv/bin/python -B /storage/penghongen/tmp/pocketxmol_evaluation_20260914/inspect_c_t0_rb_evaluation.py
```

## 训练完成与检查点只读核对

当前训练日志从 `/home/penghongen/Feedback/Pocket_Plus/allocations/371591/out` 和 `err` 的113849959／311189字节起读取。全部39次原C0验证正常保存，31200步验证后正常停止，W&B hthglbuy上传完成。训练进度栏耗时8小时13分35秒，平均约1.05次更新／秒；未见Traceback、OOM、非有限损失或reduce_batch记录。

在371591.8的8核CPU内加载last和实际best，确认stop_reason=plateau、decline_count=3、global_step=last_validation_step=31200，优化器和调度器末次学习率均为8.000000000000002e-7，第三次下降后没有继续更新。best21600对应的1.8065478801727295是全部39个定期检查点的最低原验证损失；39个检查点均存在，best的1236个model参数键全部有限。

以下是已执行的产物核对命令，不是正式训练、推理或评价命令。脚本只读取已有检查点并新写training_summary_20260913.json，不改写检查点；本地部署记录为 `tmp/pxm-20260913/check_c_t0_rb_checkpoint.sh`，摘要副本为 `tmp/pxm-20260913/B-C-T0-RB-training-summary.json`。

```bash
srun --jobid=371591 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /storage/penghongen/PocketXMol/runtime/venv/bin/python -B /storage/penghongen/tmp/pocketxmol_checkpoint_20260913/inspect_c_t0_rb.py
```

## 实验过程与失败尝试

本模型尚无失败或中断的正式尝试；它直接使用已纠正的C0训练契约，不继承错误T0-RA参数。
