# SMILES 构图与 GPU 验收

本预实验实现用户于2026-09-14批准的 Stage3 SMILES 构图迁移，承接[科学契约](../../想法/方案草稿/9-8-科学契约.md)。它改变图和坐标的读取方式，不重新筛选冻结成员、不生成新的 C5 向量、不训练正式模型。

## 当前状态与产物

**当前结论：公共SMILES图输入与T0机制和固定官方实现实质对齐，主代理与独立科学审查均已核对；不能称为全部GPU数值严格相同。** 新链已集成，历史编码偏差、各次严格失败和数值测量全部保留。正式运行仍等待本次密度入口门控及双线收口。

2026-09-14的补充GPU验收使用三个训练实例：5uar/0（10重原子）、3jbv/0（20）、6baj/14（50），各C0/C5/E，共9个输入；未读取test样本。参照为固定官方65488cf的解析、原噪声及预测回写，同一RA扩展和官方初始参数。

| 检查 | 实际结果与解释边界 |
|---|---|
| 完整图、口袋、原点、局部坐标 | 9输入与固定官方解析后的输入精确一致 |
| FP32单步前向／原loss／梯度 | 9输入通过预设门限；最大输出绝对差6.86646e-5，梯度相对L2最大2.43286e-4（限1e-3）；前向/loss容差atol=rtol=1e-4 |
| AdamW一次更新 | 均有限，最大参数差1.99966e-4；仅完成差异测量，没有逐参数等价断言，也未测同版重复更新差，不能记为优化器数值等价通过 |
| bf16 | 三个C0输入均有限；跨输入链梯度差8.53%／14.72%／11.23%，同参考重复为8.27%／23.93%／9.35%；保留原混合精度及归约，不强制逐元素相同 |
| 同状态100步采样 | 每输入2候选，9输入全部完成100步；900状态的官方噪声和预测回写精确一致，GT位置清零、口袋/原点固定、世界原点只加一次 |
| 相同模型重复前向 | 8输入的100步满足上述严格门限；6baj/14 E第84次（零起始83）超门限，严格失败保留，不转成SMILES不支持实例 |
| 自由轨迹比较 | 3jbv/0 C5跨官方/当前的终点坐标差0.0237839 Å，同版重复差0.0237842 Å；logit差分别0.639889与0.639858，不声称自由轨迹相同 |

失败定位：相同batch、参数和随机状态下，原pocket_encoder先出现3.57628e-6特征差；第4层仅9.53674e-7 Å坐标差触发一个等距kNN成员切换，受体编号485换成490，两者float32距离同为8.91702365875 Å。后续层放大该离散变化。固定官方的邻居构造、节点／边／位置算子与当前源码一致；无密度时不执行新分支。这说明严格超差可以由原GPU重复性与等距近邻边界解释，没有据此修改kNN、精度或科学公式。

汇总及原始报告已保存至本地`tmp/formal-smiles-density/smiles-gpu-summary.json`与`smiles-gpu-evidence.json`；服务器证据根`/storage/penghongen/tmp/pxm_formal_smiles_20260914/gates-smiles`，含每次launch/run_cmd、严格失败、逐步报告、`module-v1/failed_state.pt`和`module_traces.pt`。独立科学审查认为上述证据支持图输入/T0机制实质对齐，要求保留“通过／只测量／严格未通过”的区分；不额外重跑整套GPU检查。

### 官方解析与精度的补充核查

固定官方版本为`65488cf635c856101dbe703ac97e2f10f58e005c`。官方训练预处理通过默认化学检查的RDKit分子调用`utils/parser.py:parse_3d_mol`；官方SMILES推理入口经过`MolFromSmiles`与去氢，再调用同一解析函数。函数读取`GetBondType()`，把RDKit芳香键12映射为模型类别4。旧`docking/dataset.py`却从源`bonds`数组取类别，不使用`read_template`中经过`SanitizeMol`的分子的键。这是历史接入与官方解析的偏差，不能以保持旧输入为由要求新管线复现它。

本次在服务器用只读CPU命令执行该Git版本提取的原函数，逐精确SMILES比较公共包与官方解析的元素、排序后的双向键端点及键类别：1979个SMILES全部一致，失败0个，其中1360个包含芳香键，覆盖现有66878实例。检查仅涉及离散图，不使用测试坐标、不进行GPU运算、不改服务器文件，不能替代局部坐标、噪声、梯度及完整采样验收。证据保存于本地`tmp/density-preexperiments/official-input-audit-20260914.json`。

本地作者权重包`model_weights.tar.gz`中的`data/trained_models/pxm/train_config/train.yml`第548行明确为`precision: bf16-mixed`，与固定官方源码公开训练配置一致。官方`sample_use.py`及`sample_drug3d.py`直接构造模型、加载权重并`eval()`，未包裹混合精度autocast；但二者导入`train_pl`会设置`torch.set_float32_matmul_precision('medium')`。因此推理是FP32张量路径，不能称为所有乘加均严格FP32。用户已接受正式训练沿用官方bf16混合精度、推理沿用官方FP32张量路径；严格数学等价诊断可另外使用highest FP32，不能把诊断设置冒充官方默认。

历史六模型及官方冻结权重对照都通过旧Dataset取配体键类别。旧结果保留，但其解释限定为旧编码条件；新旧成绩差异可能同时包含编码修正。用户已批准修正后的无密度中心／包络RA＋T0重训及官方冻结权重C0/C5/E测试，专用资源378693已按实际kill_lock接管；旧源码、产物和资源after_lock均保留。之前“接受偏离官方的芳香编码”的提问不成立；后续应以官方解析为新链的离散输入参照。

核查命令属于只读验收：`git show 65488cf635c856101dbe703ac97e2f10f58e005c:utils/parser.py`、同版本`process/utils_process.py`与训练／推理入口读取；作者压缩包由本地Python `tarfile`只读打开。实际全量图命令由`Invoke-ProjectSsh.ps1 -Command`传入`/storage/penghongen/PocketXMol/runtime/venv/bin/python -B -c`，在内存执行固定官方函数与比较脚本。首次传入整个parser超过Windows命令长度，未执行远端检查；缩小为实际函数后完成。无正式运行命令。

当前GPU门控证据根为`/storage/penghongen/tmp/pxm_formal_smiles_20260914/gates-smiles`，首轮日志`gate-v1.out`、`gate-v1.err`，冻结源为`source_5f7bd8e`。该命令属于验收，不是正式训练或测试集采样。

### 原先以旧接入为参照的检查记录

| 检查 | 实际覆盖与结论 |
|---|---|
| RDKit化学图与运输后的完整自同构集合 | 四划分66,878实例全部通过；这只证明化学图和对称集合，不证明模型实际键类别一致 |
| 原模型实际element和halfedge键类别 | 28,650/66,878=42.8392%实例不一致，涉及1,389/2,039种SMILES与旧模板身份组合；全部旧原始键表使用单/双等类别，芳香类别4出现0次，公共包把其中芳香键编码为4 |
| 分划分原始模型键类别不一致 | train 27,983；validation 284；calibration 118；test 265 |
| 独立SMILES Kekulize可行性 | 即使对公共SMILES独立生成单双键交替式，仍有16,477/66,878=24.6374%实例与旧具体键类型图不同，涉及653种身份组合；不能用简单Kekulize宣称恢复旧数值契约 |
| GPU前向、原loss、梯度、一次AdamW更新 | 已完成5个非测试实例×C0/C5/E×FP32/bf16=30组比较，第6个实例的离散键类型断言失败后停止；原生bf16预设严格数值门限未通过 |
| 固定输入bf16定位 | 6d03/18同一输入旧链重复梯度差12.21%与11.19%，新旧差12.90%；诊断中仅将bf16消息归约临时升到FP32后，旧重复为0，新旧梯度0.603%，坐标最大差2.98e-7 Å；正式源码未作该修改 |
| 本地功能回归 | 既有数据、采样、中心与官方对照功能34通过、3跳过；新增SMILES读取与失败分母测试4通过；该结果不替代上述真实科学验收 |

开始2026-09-14 15:38:59（北京时间），主要工作截止18:38:59。两项科学差异已提前向主代理报告。所有历史源码、失败尝试、已写资产及资源控制文件保留。

资源为371591、gnode09、A800 80 GiB、16 CPU；控制目录为`/home/penghongen/Feedback/Pocket_Plus/allocations/371591`。CPU迁移使用其中8核，没有新增CPU申请。所有验收均经该作业实际锁协议，`after_lock_371591`始终保留。

| 产物或来源 | 精确位置与含义 |
|---|---|
| 公共图及排列 | `/storage/penghongen/AdaLigand/Ori_Data/smiles_assets/{smiles_graphs_v1.npz,smiles_symmetries_v1.npz}`，Matcher已发布版本；图包SHA256为`e8aad7336ccce54b01e606dfc79a82297f5bf7b86897a393bcfccf0c8d820625`（仅验收使用） |
| 候选迁移坐标 | `/storage/penghongen/AdaLigand/Ori_Data/smiles_assets/SMILE_coords/v1/{pdb_id}.npz`；坐标迁移完成，但整条新模型输入尚未通过科学验收 |
| 候选迁移清单 | `/storage/penghongen/PocketXMol/data/smiles-v1/frozen/{train,validation,calibration,test}.jsonl`；旧冻结成员、顺序、views、C5及sampling_seed保留，身份改为精确prepared_smiles |
| 原型失败资产 | `SMILE_coords`根部逐PDB包和`data/smiles-v1`根部四清单；首次Mol构造缺少芳香显式氢语义导致错误，保留为失败尝试，禁止正式使用 |
| CPU证据 | `/storage/penghongen/tmp/pxm_pre_smiles_20260914/cpu-v1/{migration_report.json,mapping.jsonl}`；mapping含new_to_old、旧模板身份、划分及质心误差 |
| 原始模型图差异 | 同任务根`diagnose-v1/raw_model_graph_report.json`，列出全部28,650个实例及逐种键类型差异 |
| 独立Kekulize诊断 | 同任务根`kekule_feasibility.json`，列出全部不匹配实例；没有据此修改正式编码 |
| GPU证据 | 同任务根`gpu-v1/{limits.json,samples.json,gpu_report.json}`；报告全部输出误差、旧重复误差、各loss、梯度和AdamW参数误差 |
| bf16定位证据 | 同任务根`bf16-repeat-v1/report.json`；固定同一原点与噪声，比较原生和仅诊断使用的FP32消息归约 |
| 源码与Git | `codex/pre-smiles`从`a78ae8f`开始；远端`code`、`code-cpu-v1`、`code-gpu-v1`分别保存具体尝试源码，未覆盖共享项目源码 |

## 判据与覆盖口径

不支持比例必须严格小于已检查实例的1%；若四划分66,878实例全部检查，则最多668个。训练及监督验证跳过明确不支持实例，测试保留446分母及失败记录。CPU与GPU覆盖数分别汇报。

GPU在同一口袋、原点、物理原子与半边映射以及随机噪声下比较前向、原loss、梯度和一次AdamW更新；仅相同随机种子不是等价证明。样本只来自train、validation或calibration。比特差异可容忍，系统性科学差异不得归入少数例外。

## 正式运行命令

本预实验不提交正式训练或测试集采样命令。

## 迁移与验收命令

以下均为一次迁移或验收命令，不是正式实验命令。工作目录分别为该尝试隔离源码根，Python固定为`/storage/penghongen/PocketXMol/runtime/venv/bin/python`。

```bash
python -u tmp/pre-smiles-20260914/migrate.py
python -u tmp/pre-smiles-20260914/gpu_equivalence.py
python -u tmp/pre-smiles-20260914/diagnose.py
python -u tmp/pre-smiles-20260914/bf16_repeat.py
python -u tmp/pre-smiles-20260914/kekule_feasibility.py
```

实际动态命令依次保存在任务根`launch_cpu_v1.sh`、`launch_gpu_v1.sh`、`launch_diagnose_v1.sh`、`launch_bf16_repeat_v1.sh`、`launch_kekule_feasibility.sh`，每次执行还通过原`create_launch.sh`保存`/home/penghongen/Feedback/PocketXMol/launches/371591/`下独立launch与真实run_cmd；stdout/stderr为任务根对应`cpu-v1`、`gpu-v1`、`diagnose-v1`、`bf16-repeat-v1`、`kekule`的`.out/.err`。

本地功能回归使用项目已验收Windows环境，`PYTHONUTF8=1`，`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`：

```powershell
& C:/Users/15919/Desktop/PocketXMol/tmp/pxm-20260910/venv/Scripts/python.exe -m pytest tests/test_docking_data.py tests/test_docking_sampling.py tests/test_docking_centers.py tests/test_docking_official.py -q --disable-warnings --maxfail=2
& C:/Users/15919/Desktop/PocketXMol/tmp/pxm-20260910/venv/Scripts/python.exe -m pytest tests/test_docking_smiles.py -q
```

本地汇总证据为`tmp/pre-smiles-20260914/audit_summary.json`，包含全量报告摘要、固定输入GPU误差、真实launch目录及停止后的锁状态；完整逐实例证据保存在上述服务器路径。最后核查资源为`after_lock=true`、`kill_lock=false`、父目录`try_lock=true`，已返回保留资源的等待状态。

## 执行过程与之前的尝试

1. 读取AGENTS与用户点名skills及注释示例；主代理在全部refs/工作树唯一最新且干净的共同基点创建隔离实现树。
2. 首次Slurm启动因Git归档中的shell脚本CRLF退出，尚未执行科学计算；保留旧源码，将调用的launch脚本复制成LF后重新启动。
3. 首次图读取器只按公共NPZ元素、电荷和芳香键构建Mol，缺少`[nH]`显式氢语义，导致9,997个错误排除。该尝试66,878实例耗时65.91秒，未通过1%门限；图读取器改为首次按精确SMILES解析并缓存SDF/RMSD所需Mol，模型数组继续直接读取公共包。
4. 第二次迁移独立落盘到`SMILE_coords/v1`，8个CPU进程14.94秒完成全量化学同构和自同构集合核对；未覆盖首次失败文件，也未重新筛选或生成C5。
5. GPU真实运算暴露原始halfedge键类别与bf16重复误差问题。补充全量原始模型图比较与固定输入诊断，确认不能把化学同构通过写成GPU输入等价通过。
6. 独立Kekulize诊断证明简单转换仍不足以匹配原始单双键选择。停止扩大运行，等待用户确定模型键编码及bf16严格验收口径；不启动正式密度训练。

## 尚未完成的验收

当前未完成完整100步新旧轨迹、最终self-ranking/RMSD对照以及通过后的独立审查；这些步骤须在上述科学口径解决后进行。候选代码功能可运行，不代表预实验已经通过。主代理负责汇报用户选择、公共入口集成及最终双线Git收口。
