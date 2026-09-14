# SMILES 构图与 GPU 验收

本预实验实现用户于2026-09-14批准的 Stage3 SMILES 构图迁移，承接[科学契约](../../想法/方案草稿/9-8-科学契约.md)。它改变图和坐标的读取方式，不重新筛选冻结成员、不生成新的 C5 向量、不训练正式模型。

## 当前状态与产物

**状态：预实验未通过，等待用户确定科学口径。** 新链仅存在于隔离实现分支和隔离验收源码中，没有提交正式训练或测试集采样。发现问题后停止扩大GPU验收，没有耗满三小时、没有改变正式归约精度、没有自动排除42.84%的实例。

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
