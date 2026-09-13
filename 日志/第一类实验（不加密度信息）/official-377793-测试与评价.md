# 377793：官方测试结果与执行记录

官方冻结模型的C0/C5/E测试和CPU评价已于2026-09-12完成，九组视图汇总已保存并上传 [W&B official_377793_test](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/x8diqywv)。每协议446个实例，445个实例成功生成并评价全部50个候选，9qkz/0保留50个预处理失败记录。374480_1（实际377793，gnode05）已回到try_lock，after_lock及全部产物保留。本对话另有F-5/F-6仍在H100队列，官方完成不代表这两项训练完成。

本记录依据 [科学契约](../../想法/方案草稿/9-8-科学契约.md)，接替 [377521官方测试](official-377521-测试与评价.md)。用户因怀疑gnode08异常，亲自执行scancel 374480_0，随后最终明确改为接管 **374480_1**，并要求不操作374480_2。本次只更换运行资源和独立产物目录，保留官方权重、测试协议、候选预算和评价定义。

## 九组正式测试结果

C0用完整配体重原子几何中心定位；C5在该中心加上清单中冻结的0–5 Å偏移。两者按受体残基重原子质量中心到给定中心的距离严格小于15 Å选袋，官方模型均执行T0采样。E按残基质量中心到任一真实配体重原子的距离严格小于10 Å选袋，使用真实配体形状提供定位信息。三个协议描述同一官方权重面对不同定位输入时的表现。

ALL为完整测试集合。以完整object_key统计ALL内身份频数，只有频数大于10的身份才按同一冻结排序截取：CAP10保留前10个，HF10_TO5保留前5个；原频数不超过10的身份均全部保留。三个重叠视图分别有446/272/227个实例、77/67/65个PDB，共用各协议已生成的候选。

self-ranking沿用原置信度轨迹分数加无碰撞指示值和立体化学通过指示值；置信度分数先对全部配体原子取均值，再平均100步轨迹的后50步原始位置置信度，不做sigmoid。Top-1看第一名，Top-5看前五名中最小RMSD，50候选最佳看全部候选的最小RMSD；成功均定义为未刚体对齐的重原子RMSD严格小于2 Å。成功率包含9qkz/0这一失败实例；RMSD均值只使用有坐标的实例，三个视图分别为445/271/226个有效实例，PDB等权的RMSD均值分别为76/66/64个有效PDB。RMSD沿用RDKit CalcRMS和maxMatches=30000，本次没有触发按原子编号匹配的备用计算。

### 实例等权结果

以下数值由最终summary.json直接格式化，百分比保留两位小数，RMSD和相关统计保留三位小数。

| 协议 | 视图 | 实例 / PDB 数 | Top-1成功率 | Top-5成功率 | 50候选最佳成功率 | Top-1均值RMSD（Å） |
|---|---|---:|---:|---:|---:|---:|
| C0 | ALL | 446 / 77 | 44.84% | 52.02% | 63.00% | 4.512 |
| C0 | CAP10 | 272 / 67 | 31.99% | 39.34% | 53.31% | 5.676 |
| C0 | HF10_TO5 | 227 / 65 | 32.16% | 39.65% | 54.19% | 5.823 |
| C5 | ALL | 446 / 77 | 31.17% | 41.48% | 52.91% | 5.352 |
| C5 | CAP10 | 272 / 67 | 23.16% | 29.41% | 41.91% | 6.284 |
| C5 | HF10_TO5 | 227 / 65 | 24.67% | 31.28% | 42.73% | 6.196 |
| E | ALL | 446 / 77 | 63.90% | 74.44% | 87.22% | 3.162 |
| E | CAP10 | 272 / 67 | 55.88% | 67.65% | 83.09% | 4.014 |
| E | HF10_TO5 | 227 / 65 | 56.83% | 67.84% | 81.50% | 4.165 |

### PDB等权结果

先在每个PDB内平均实例结果，再对PDB等权平均；成功率仍包含失败实例。

| 协议 | 视图 | Top-1成功率 | Top-5成功率 | 50候选最佳成功率 | Top-1均值RMSD（Å） |
|---|---|---:|---:|---:|---:|
| C0 | ALL | 39.09% | 50.43% | 61.12% | 6.878 |
| C0 | CAP10 | 35.96% | 46.81% | 58.27% | 7.594 |
| C0 | HF10_TO5 | 34.85% | 45.14% | 57.39% | 7.808 |
| C5 | ALL | 35.09% | 41.58% | 51.68% | 7.100 |
| C5 | CAP10 | 31.09% | 37.01% | 48.14% | 7.977 |
| C5 | HF10_TO5 | 30.66% | 36.51% | 47.12% | 8.069 |
| E | ALL | 67.46% | 77.86% | 87.85% | 5.635 |
| E | CAP10 | 65.72% | 75.29% | 86.27% | 6.333 |
| E | HF10_TO5 | 64.28% | 74.00% | 84.97% | 6.507 |

### 排序分数与姿态误差

Spearman为同实例候选的self-ranking分数与负RMSD的相关系数；AUC以RMSD严格小于2 Å为正类。先在实例内部计算，再按相应方式平均。括号为该均值的有效实例数或有效PDB数；无定义的值不填0。

| 协议 | 视图 | 实例等权Spearman | PDB等权Spearman | 实例等权AUC | PDB等权AUC |
|---|---|---:|---:|---:|---:|
| C0 | ALL | 0.225 (445) | 0.233 (76) | 0.713 (256) | 0.693 (55) |
| C0 | CAP10 | 0.148 (271) | 0.222 (66) | 0.677 (137) | 0.698 (45) |
| C0 | HF10_TO5 | 0.141 (226) | 0.210 (64) | 0.682 (117) | 0.701 (42) |
| C5 | ALL | 0.190 (445) | 0.218 (76) | 0.714 (228) | 0.710 (54) |
| C5 | CAP10 | 0.143 (271) | 0.203 (66) | 0.692 (112) | 0.719 (44) |
| C5 | HF10_TO5 | 0.149 (226) | 0.195 (64) | 0.703 (95) | 0.727 (41) |
| E | ALL | 0.372 (445) | 0.353 (76) | 0.769 (349) | 0.779 (69) |
| E | CAP10 | 0.323 (271) | 0.353 (66) | 0.746 (203) | 0.778 (58) |
| E | HF10_TO5 | 0.332 (226) | 0.351 (64) | 0.758 (167) | 0.769 (56) |

### 实际耗时与显存

以下只使用ALL，三个重叠视图不能重复累加。采样累计时间包含预处理和写盘；评价时间是逐实例耗时之和，8个进程并行，因此不是作业墙钟时间。显存区分张量分配峰值和CUDA缓存分配峰值，均含驻留模型。

| 协议 | 采样累计小时 | 评价逐实例累计小时 | 张量显存峰值（GiB） | CUDA缓存峰值（GiB） |
|---|---:|---:|---:|---:|
| C0 | 3.943 | 6.668 | 2.987 | 11.418 |
| C5 | 3.881 | 6.680 | 2.984 | 13.176 |
| E | 3.771 | 6.690 | 2.996 | 14.986 |

## 产物、完整性与失败口径

全部科学结果位于服务器 `/storage/penghongen/PocketXMol/sampling/official-377793/test/`。根目录summary.json汇集C0/C5/E及每协议的ALL/CAP10/HF10_TO5，wandb_run.json保存运行身份x8diqywv；各协议的summary.json和occurrences.json分别保存视图汇总和446个逐实例评价。每个 `<协议>/<pdb_id>/<occurrence_id>/` 保存candidates.json、result.json、candidate_metrics.json和assessment.json；成功实例另有poses.sdf和confidence.npz，失败实例的姿态及置信度文件字段为null。逐字段定义分别见 [采样函数的产物说明](../../docking/sampling.py) 和 [评价函数的产物说明](../../docking/evaluation.py)。

逐协议核对了冻结test.jsonl、446份assessment.json、occurrences.json和三个视图的实例集合，均完全一致。每协议22300份候选指标中22250份有RMSD和可用于排序分析的分数，44500次模型批量前向均完成；除了9qkz/0的50份preprocess错误，没有其他候选评价错误。直接从逐实例布尔结果重新计算实例等权及PDB等权的Top-1/Top-5/50候选最佳成功率，与九组summary.json在1e-12内一致。

9qkz/0（CCD:ANP）在三个协议均因原特征化的 `AssertionError: unknown element in pocket` 失败；没有前向尝试，未修补元素定义或补生成候选。该实例属于三个视图，始终进入成功率分母。各协议所有446个实例都有可解释的口袋核酸占比，其中7个大于0；这些逐实例字段保留在assessment.json中，供后续六模型的共同分析使用。

在每协议22250个已生成且有RMSD的候选中，无碰撞检查通过数C0/C5/E为20572/19718/20980，原立体化学检查通过数为8619/7879/8586。stderr保存了RDKit关于allene-style立体化学不支持的提示；这里的立体化学统计遵循原检查器的定义。原子编号备用RMSD计算的触发数三个协议均为0。ALL的AUC无定义实例中，C0/C5/E分别有189/217/96个仅含单一类别的实例，另各有9qkz/0这个无有效候选实例；这些值保持缺失，没有填0。

最终汇总文件时间为2026-09-12 07:47:34（服务器文件时间，UTC+8）；W&B记录状态为online_completed，error为null。控制器out确认第二次执行成功并重新建立try_lock_377793，after_lock保留。测试与CPU评价使用下面两条已经执行的正式命令，没有新增完整验证集采样。

## 资源、配置与正式命令

已只读确认374480_1实际JobId为 **377793**，gnode05，单张A100、8核、96G内存。原cryoatom2预测已成功结束；控制目录为 `/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/test_0_chain06/运行日志与统计/slurm/allocations/377793/`，after_lock_377793在此目录，try_lock_377793在父目录。操作前再次核对进程和锁，然后仅消费该try_lock开始新命令，after_lock及旧产物保留。

配置为 `configs/docking/sample-official-377793-test.yml`，与377521配置相比只改变output_root及W&B名称。官方model_name、protein输入、T0、C0/C5/E、test、batch50、每实例50候选和100步均不变。输出根 `/storage/penghongen/PocketXMol/sampling/official-377793/test/`，W&B评价名称official_377793_test。ALL/CAP10/HF10_TO5有446/272/227个实例，各协议三视图共用候选；官方共九组汇总。旧377521的run.json和空poses.sdf不覆盖或移动。

权重固定为 `/storage/penghongen/PocketXMol_official_test/extracted/data/trained_models/pxm/checkpoints/pocketxmol.ckpt`，配套训练配置为同pxm根的train_config/train.yml；本次没有训练官方模型或读取其他模型的检查点。推理清单为 `/storage/penghongen/PocketXMol/data/test.jsonl`，完整源资产来自 `/storage/penghongen/AdaLigand/Ori_Data`，共同准备产物来自 `/storage/penghongen/Adaligand_Build/Ori_Data/pocketxmol`，均未在本次重新生成。

在本作业创建的PocketXMol冻结release根，顺序执行：

```bash
bash 训练与运行/sh/sample_docking.sh official-377793-test
bash 训练与运行/sh/evaluate_docking.sh official-377793-test
```

推理正常退出后，第二条使用本allocation已有8核和同一配置，入口关闭CUDA可见性；完成后继续after_hold。不追加完整验证集采样，不另申请GPU，不运行独立ranker。

## 核查与执行记录

主代理第一遍用原make_config比较新旧配置，只有output_root和wandb.name的值变化；第二遍核对实际array索引1、JobId377793、gnode05、8核、正式命令及所有路径中的编号。配置解析是本地Python -X utf8的stdin检查，不是正式推理。此前三份配置和日志已完成两轮独立审查；本次主代理仅对迁移编号、路径和资源作窄核，没有改动Python逻辑。

已于gnode05时间2026-09-11 17:37:24按try_lock协议请求启动。操作前/proc中只有原控制器PID68145，无活跃Python预测；原run_cmd备份到 `/storage/penghongen/PocketXMol/control/377793/run_cmd_377793_before_official_20260911.sh`，新动态命令为同目录official_377793_test_run_cmd.sh。新命令沿用已检查的两阶段调用，只替换作业编号和来源标记，bash -n语法检查通过。

启动记录 `/storage/penghongen/PocketXMol/control/377793/official_377793_test_start.json` 保存实际资源、array索引1、旧任务备份及out/err起点17645/161。after_lock保留，未使用kill_lock或scancel。源码基准为b612032及本对话未提交的配置/日志；本次实际冻结副本为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_dceb9298cebe/PocketXMol`，启动证据目录为 `/home/penghongen/Feedback/PocketXMol/launches/377793/official_377793_test_job377793_20260911T173936`。本对话所有改动永远unstaged，不执行git add/commit，不处理其他人的修改或暂存内容。

run.json记录模型严格加载完成于2026-09-11 17:40:16（UTC+8）。17:48的只读检查确认C0的11jb实例0至6均已完成，每实例50个候选全部成功，候选预算和100步未变。实例0耗时62.38秒、实例1耗时48.73秒，实例2至6耗时35.34至35.60秒。实例6记录一次采样批次、100次模型前向全部完成，峰值分配显存2500749312字节；这只代表启动时的这些实例，不外推整个测试集速度。C5和E尚未开始，推理结束后仍由已登记的第二条正式命令自动进行CPU评价。

18:51例行检查时，C0已完成124/446个实例，共6200个候选全部成功；C5/E尚未开始。377793仍RUNNING，after_lock存在、try_lock与kill_lock均不存在。F-5/F-6仍为PENDING/Priority；检查后继续按用户要求静默等待60分钟，本次例行进度不另写handoff。

21:23检查时，C0已完成410/446个实例，其中409个成功、1个失败。失败实例为9qkz/0（CCD:ANP）：`candidates.json`记录阶段preprocess，错误为 `AssertionError: unknown element in pocket`；原 `utils/transforms.py::FeaturizePocket` 的元素检查在模型前向前终止该实例。其50个候选均已登记失败，模型前向尝试次数为0，poses.sdf和confidence.npz未生成，result.json相应路径为null。证据目录为 `/storage/penghongen/PocketXMol/sampling/official-377793/test/C0/9qkz/0/`。该实例仍保留在ALL/CAP10/HF10_TO5评价分母中，没有修改官方元素定义或重新生成候选，其余实例正常继续。

22:55检查确认C0已处理446个实例，其中445个成功，共22250个成功候选和9qkz/0的50个失败候选；C5已处理142个实例、7100个候选全部成功，E尚未开始。官方任务保持RUNNING和after_lock，未重新提交推理或评价；F-5/F-6仍PENDING/Priority。

2026-09-12 01:59检查确认C0与C5均已处理446个实例，各有445个成功实例、22250个成功候选，以及9qkz/0的50个失败候选。E已自动开始，47个实例、2350个候选全部成功。CPU评价尚未开始；官方资源与after_lock均正常，F-5/F-6仍PENDING/Priority。

2026-09-12 06:03检查确认三个协议的推理均正常结束，并已自动进入同作业的CPU评价。每协议446个实例，445个成功实例、22250个成功候选，9qkz/0的50个候选因同一官方元素检查失败；三个协议合计66750个成功候选和150个失败记录。通过原控制器68145的进程树，确认评价主进程325990及8个工作进程，均执行同一份sample-official-377793-test.yml；检查时C0已有236份assessment.json。after_lock保留，尚未回到try_lock。

推理结束后的只读核查以data/test.jsonl的446个唯一(pdb_id, candidate_id)为依据，逐一读取三个协议的result.json和candidates.json：各协议均无缺失或额外实例；model_name=official、split=test、协议名称、complete、batch_size=50、num_candidates=50和num_steps=100均一致；每实例候选编号恰为0至49，身份和成功/失败计数均与result.json对应。该核查是经SSH运行的Python标准库stdin检查，不是正式采样或评价命令，没有改写任何候选。CPU评价完成后再核对九组视图汇总与W&B记录。

2026-09-11接管前的只读检查中，374480_2实际JobId是374480、节点gnode07，当时仍有CryoAtom2对11jb的计算；本对话没有修改该作业的锁、命令或进程。它不属于当前授权目标，后续没有再操作。

## 计划与实现差异

中性差异：按用户明确指示迁移官方资源，科学条件不变，旧未完成尝试保留；CPU评价直接使用已分配的8核。官方测试和评价已完成，九组结果及失败分母已核查，未发现需修改科学契约的差异。未完成范围是F-5/F-6的排队、训练与配套测试，记录在各自日志中，未因本次迁移重新提交。
