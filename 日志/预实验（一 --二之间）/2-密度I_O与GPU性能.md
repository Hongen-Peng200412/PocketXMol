# 密度读取与GPU性能预实验

## 本次前置集成的补充入口验收

2026-09-14，SMILES新数据链与D1/D4/D2/D3中心／包络八个配置均通过真实非测试GPU入口验收：8通过、0失败、0跳过，705.84秒。每配置从官方初始权重开始，仅执行2次优化器更新、1验证批、2候选3步及CPU评价；全部候选和RMSD计数均为2/2，原val/loss与新增参数保持有限，D3另检查辅助损失有限。验证／短采样实例为validation的5irx/0，没有使用测试集。其后D1/D4/D2/D3×sdpa/flash的固定缓存／重新编码检查8通过、0跳过，29.82秒。

这两项是必要入口验收，不是正式训练或模型效果评价；没有新正式W&B、best或正式测试指标，不延长已经结束的三小时I/O调优循环。下表fit耗时包含加载和冷启动，不能用于比较稳定吞吐；GiB按2^30字节换算。

| 配置 | 批量×累积 | 完整更新 | fit秒数（含加载） | 峰值已分配GiB | 采样／评价候选 |
|---|---:|---:|---:|---:|---|
| D1-C-T0-RA | 72×1 | 2 | 175.688 | 54.545 | 2/2、2/2 |
| D1-E-T0-RA | 72×1 | 2 | 86.147 | 54.012 | 2/2、2/2 |
| D4-C-T0-RA | 36×2 | 2 | 82.997 | 59.073 | 2/2、2/2 |
| D4-E-T0-RA | 36×2 | 2 | 60.552 | 58.891 | 2/2、2/2 |
| D2-C-T0-RA | 36×2 | 2 | 65.647 | 59.069 | 2/2、2/2 |
| D2-E-T0-RA | 36×2 | 2 | 60.012 | 58.886 | 2/2、2/2 |
| D3-C-T0-RA | 36×2 | 2 | 83.104 | 59.115 | 2/2、2/2 |
| D3-E-T0-RA | 36×2 | 2 | 60.335 | 58.807 | 2/2、2/2 |

源提交为5af144f，实际release为`/storage/penghongen/tmp/pxm_formal_smiles_20260914/density-gate/releases/PocketXMol_706febc31afc/PocketXMol`。launch为`density_gate_5af144f_r2`；同一证据根下`gate-5af144f-r2.out/.err`、`training-sampling.xml`、`cached-sampling.xml`保存原始结果。本地完整JSON及各原文件SHA256保存于`tmp/formal-smiles-density/density-gpu-evidence.json`。逐配置记录位于`/storage/penghongen/tmp/pocketxmol_gpu_checks_density_gate_5af144f_r2/pytest/*/real_data_gate.json`。

门控命令（已执行，非正式运行）：`bash ops/run_docking_gpu_checks.sh -k 'real_data and (D1 or D4 or D2 or D3)' --junitxml <证据根>/training-sampling.xml`；随后`python -B -m pytest tests/test_density_sampling.py -q -s -p no:cacheprovider --basetemp <证据根>/pytest-cache-sampling --junitxml <证据根>/cached-sampling.xml`。Python是PocketXMol专用venv，W&B关闭。CUDA PID43112已核实属于379402并使用指定GPU-adbf8fc8-5a4a-87e3-853b-c9cadcbdf74b。

首次20:28启动在GPU运算前因归档shell的CRLF失败，模型未运行；保留原始归档、失败日志和run_cmd，仅将本次新源副本36份shell转为LF，再于gnode09时间20:30:46重试。逐文件换行证据为`shell-line-ending-normalization.json`，科学代码及配置未改变。20:44全部完成，20:48以后只读复核控制器PID2383等待、after_lock及父try_lock保留、kill_lock不存在，指定GPU 0%／5MiB。旧服务器产物未删除。

本次集成收口后停止推进，正式密度及无密度复验等待用户再次明确允许。后续资源、顺序及监测计划见[预实验总日志](总日志.md#待再次批准的后续执行计划)。

## 之前的尝试：三小时I/O预实验及其结果

以下记录使用当时旧配体图，全部保留。D1持续供数未达目标，D4在已测三个窗口内达标；本次新SMILES入口短验收不改写这些性能事实，也不声称已经证明D2/D3的持续性能。

2026-09-14 18:34状态：预实验已结束，共24次改进，未启动正式八模型训练。D1选72×1，D4选36×2，均32 worker、prefetch1、cuDNN benchmark开启并保留距离项。D1的60更新窗口平均1.96571秒、等待29.21%，三段稳定窗口未全部达到目标；D4的48更新窗口平均4.57112秒，三段GPU中位数均100%、等待约0.02%。D1-E与D4-E均通过4次完整更新；实际A800数值测试12 passed、采样缓存测试4 passed无skip。最后的MADV_RANDOM候选仅有受前缀/缓存混杂的小幅差异，未集成正式读取逻辑，完整56通道模块与ALL保持原样。指定GPU无计算进程，控制器PID2383等待，after_lock和try_lock保留、kill_lock不存在。

开始时间为2026-09-14 15:40:56（北京时间），主要工作截止18:40:56；全部GPU测试在18:34前结束，每次首次真实forward后观察不超过5分钟。GPU利用率90%及读取等待不超过10%是优化目标，未达到的D1供数目标如实保留，后续按最快稳定资源组合继续，不延长本次搜索。全部真实训练链使用原有配体图；SMILES新图的科学分歧由另一预实验处理，不把本次结果当作其验收。

## 当前有效配置与三段窗口

四个配置文件为`configs/docking/D1-C-T0-RA.yml`、`D1-E-T0-RA.yml`、`D4-C-T0-RA.yml`、`D4-E-T0-RA.yml`。D1批量72、累积1；D4批量36、累积2；均保持global batch72、32个单线程加载进程、prefetch_factor1、cudnn_benchmark=true、bf16-mixed、checkpoint=true和distance_bias=true。32进程共享作业原16CPU，未申请额外CPU。D4的72×1在反向阶段OOM、0完整更新，不选用。

以下更新编号从1开始。按加载进程数×每进程预取批数×micro batch/72估算预取容量：D1为32次完整更新，D4为16次。剔除该容量后，把已有完整更新均分成三个连续窗口；实际时长从窗口首更新开始到末更新结束，GPU中位数仅取该区间内的实际1秒轮询样本，等待比例为各更新读取等待之和除以完整更新时间之和。

|结构/试验|更新编号|实际时长（秒）|GPU采样数|GPU中位数|等待比例|
|---|---|---:|---:|---:|---:|
|D1，第20次|33–42|27.311|24|41.5%|50.537%|
|D1，第20次|43–51|25.592|22|35.0%|48.356%|
|D1，第20次|52–60|12.893|11|92.0%|0.0375%|
|D4，第23次|17–27|50.533|44|100%|0.02086%|
|D4，第23次|28–38|50.918|44|100%|0.02214%|
|D4，第23次|39–48|45.673|40|100%|0.01878%|

三段均有实际采样，但时长仅约13–51秒，不能推断长时间正式训练始终保持这些性能。D1前两段未达目标；D4三段均达到本窗口的目标。完整冷/暖耗时、CPU、RSS、显存、读取计数、实例前缀及原始JSON的SHA256在`tmp/pre-density-20260914/final_benchmark_summary.json`，原始逐更新和逐秒metrics仍保留服务器任务根`benchmarks/*/result.json`。本记录的GB均按10^9字节换算，原始JSON保留字节数。

本记录依据当前三契约及本次用户批准的密度选择，覆盖真实加载、forward、backward与AdamW更新的性能预实验；不启动正式D1/D4/D2/D3八模型训练。三契约入口为[科学契约](../../想法/方案草稿/9-8-科学契约.md)、[工程细节](../../想法/方案草稿/9-8-工程与实现细节.md)、[核查清单](../../想法/方案草稿/9-8-边界与核查清单.md)。

## 固定科学内容

56通道在48³源体素裁块上按Pocket_Plus既有顺序计算；受体占据掩码来自整个原始受体，包括UNK；RA图继续排除UNK。D1只运行三次下采样与原四层8头×192维三维RoPE注意力，输出6³×256；D4运行完整四次下采样/上采样U-Net，输出48³×48。单次密度编码，无recycle与无关输出头。

去噪器每个节点更新后、边及位置更新前注入独立4头×64维读出，输出320维，alpha初始0.1、g恒为1。D4按当前原子home体素±3取与裁块的交集，空集合零残差。距离偏置为qk/8−softplus(beta)×||(x−p)/10 Å||²，初始系数1；以完整优化器更新时间相比普通Flash增时不超过10%为统一选择依据。D2/D3留待后续实施，不能把D1性能代表完整U-Net。

## 资源与证据

资源为Slurm实际JobId379402（数组显示379402_0）、gnode09、A800 80GB、16CPU，GPU UUID为GPU-adbf8fc8-5a4a-87e3-853b-c9cadcbdf74b。控制目录为`/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration/运行日志与统计/slurm/allocations/379402/`；try_lock在其父目录。首次只读核查时控制器PID2383处于等待，after_lock与try_lock存在、kill_lock不存在、指定GPU空闲。

代码使用隔离实现工作树`C:/Users/15919/Desktop/PocketXMol-worktrees/pre-density`，起点a78ae8f。服务器代码与预实验产物使用本次独立临时根；每次实际运行冻结自己的release，不覆盖共享代码或已有实验产物。

## 已完成的本地验收

`tests/test_density.py`与共同数据入口`tests/test_density_dataset.py`：11 passed。检查48³裁块、实际XYZ spacing与角点、原始UNK受体掩码、D1停止于第三次下采样、D4越界交集及空集合、共同刚体变换、CUDA bf16近体素边界，以及密度距离偏置与三维RoPE注意力的输出/梯度一致性。首次测试发现多batch增广常数项广播错误和混合精度残差写回类型不一致，均已修正后重跑通过。

## 测试命令

以下是测试命令，不是正式训练：

```powershell
& 'C:/Users/15919/Desktop/PocketXMol/tmp/pxm-20260910/venv/Scripts/python.exe' -X utf8 -B -m pytest -q tests/test_density.py tests/test_density_dataset.py --disable-warnings
```

指定Slurm作业中的数值测试命令为`bash tmp/pre-density-20260914/preflight.sh`，实际Python为`/storage/penghongen/PocketXMol/runtime/venv/bin/python`。临时任务根为`/storage/penghongen/tmp/pocketxmol_density_20260914`，其中`source_equivalence.log`和`density_tests_gpu.log`保存数值证据。精度诊断未单独tee到文件，其真实证据是上述Slurm控制目录`out`中attempt6的`ATTENTION_PRECISION`记录；对应release为`source_43a3e41e18a2`。每次动态命令保存在任务根`control/<尝试名>_run_cmd.sh`，冻结目录在`releases/`，启动证据在`launches/379402/`。现有PXM环境为torch2.6.0+cu124，缺失的einops0.8.1仅安装于本任务`deps/`，未修改共享环境。

## 尝试历史

门控0a（控制器attempt3）在读取上传脚本时因CRLF失败，随后统一上传脚本为LF。门控0b（attempt4，release `source_b90e55fa0d1e`）发现PXM环境缺少einops，已隔离补齐依赖。门控0c（attempt5，release `source_468d969a1370`）通过成熟RoPE和单次完整U-Net核对，但bf16 QK增广对FP32显式距离参考最大绝对误差0.06934，未通过既定输出门限，未放宽门限。

门控0d（attempt6，release `source_43a3e41e18a2`）比较注意力精度：bf16增广输出相对RMS误差0.916%、输入梯度0.85%至1.38%；fp16增广分别0.115%、0.149%至0.224%；FP32显式距离分数的SDPA数学内核分别0.168%、0.153%至0.278%。后者避免增广平方项的低精度舍入，选为下一轮真实性能候选。高效SDPA内核拒绝FP32偏置与bf16 query的组合，未采用。门控0e正在实际bf16 autocast上下文重验该候选；完成前不标通过。

门控0e（attempt7，release `source_d042fa1a2343`）通过当时全部检查，未放宽输出或梯度门限：普通Flash输入梯度相对RMS误差约0.27%；显式FP32距离SDPA数学内核各输入梯度误差为4.75e-6至1.02e-5，位置及系数梯度误差为1.74e-7至2.45e-7。完整成熟U-Net对照当时使用32³随机输入，仅证明该尺寸；真实48³源码对照列入下一必要门控，不能由32³结果替代。随后指定A800上的密度pytest为7 passed，用时4.73秒，覆盖bf16体素边界。

第1次（attempt8）在导入阶段因隔离打包遗漏`process/`失败，未进入forward；补齐代码依赖后重新冻结。第2次D1与第3次D4（attempt9/10，release `source_45ed28f7caba`）各完成12次原dock噪声、原loss和AdamW更新，官方权重匹配1128个tensor，所有模型参数及总体梯度保持有限。原始证据分别为临时根`benchmarks/02_D1_b24_w8/result.json`和`benchmarks/03_D4_b24_w8/result.json`。

这两轮均固定144个跨训练清单抽取的实例顺序循环，读取会重复命中地图和裁块，因此明确降格为计算/缓存基线。D1首更新30.393秒，其中读取等待21.740秒；后11次完整更新平均2.772秒、中位2.428秒，前向+反向+AdamW平均2.339秒，读取等待11.85%。进程树RSS峰值21.18GB、已分配显存22.54GB；第一轮仅粗略按首forward后10秒统计GPU利用率约53%，后续以明确的每次更新时间边界统计。D4同样完整运行四次下采样/上采样及局部343邻域，后续完整更新约5.63–5.98秒，显存峰值44.03GB，不把D1结果外推给完整U-Net。

两轮的作业内测试命令均为`python -u tmp/pre-density-20260914/benchmark.py --config configs/docking/D1-C-T0-RA.yml --output <本轮独立目录> --batch 24 --workers 8 --seconds 150 --updates 12 --deadline 1789382216`，D4仅替换配置文件名。执行的是当时默认`samples=144`的冻结版本；后续默认已改为完整清单均匀抽样，不能用新脚本的默认值解释旧证据。

第4次（attempt11，release `source_42b601188e43`）改为完整清单/15 worker，在pin-memory线程收到`received 0 items of ancdata`后退出，未完成完整更新。临时入口此前漏了正式`train_docking.sh`已有的`ulimit -Sn 65536`，已对齐本进程文件描述符限制；未修改共享环境。之后第6次同规模流水线稳定完成，支持此项修正有效。

门控0f（attempt12，release `source_6756ad0ea44a`）通过真正48³成熟完整U-Net输出对照，并通过三维RoPE的强制Flash bf16输出/输入及参数梯度对照，非零梯度最大相对RMS误差0.006685。批量读出首次纯相对门限被`key.bias`理论零梯度约1e-9绝对舍入误差触发；随后保留非零梯度相对判据，理论零梯度另外报告绝对RMS。第5次（attempt13，release `source_b1bcff7ac2e9`）的bf16理论零梯度绝对差1.3522e-5；未进入真实更新。第7次（attempt15，release `source_b265a3225796`）进一步发现bf16标量alpha乘法按分子归约和整批归约的梯度差：旧结果0.107421875、差值0.0224609375，未通过非零梯度检查。

针对alpha的候选修正是把密度输出投影提升到alpha参数的FP32类型后再乘，以避免bf16标量梯度的分组归约误差。新比较参考仅在最终密度投影处加FP32乘法，原冻结bf16逐分子版本仍保留；不能把修正参考后的通过描述为与旧bf16逐元素等价。此处不修改原dock噪声、监督或原训练损失归约。第8次先验收该候选；如果失败，明确使用冻结旧读出继续I/O基准，避免门控占满三小时。

第6次（attempt14，release `source_b1bcff7ac2e9`）使用完整训练清单、固定种子2023的均匀有放回抽样、原逐分子读出，24×3/15 worker完成24次更新。官方参数1128个tensor正确加载，新密度编码84个参数tensor、读出60个参数tensor均有有限梯度，首次范数分别1.28908和0.33204，最终模型参数均有限。冷首更新18.591秒；后23次平均4.647秒、中位3.707秒，计算2.391秒，供数等待46.22%；已分配显存峰值27.26GB。每72实例的密度加载及构造墙钟总和平均44.87秒，图处理9.48秒，CPU只用约4.73核，显示大量I/O等待。原始证据在`benchmarks/06_D1_b24_w15_full_reference/result.json`，完整测试参数为`--batch 24 --workers 15 --seconds 180 --updates 24 --readout-reference 1 --deadline 1789382216`。

32 worker前只读核查：379402 cgroup内存上限1,081,638,649,856字节，gnode09当时可用约869GiB。后续基准记录CPU亲和核数、作业内存上限、RSS以及进程树`read_bytes/read_chars/write_bytes`，避免把加载进程数当成新增CPU配额。

第8次FP32 alpha乘法候选仍未通过，alpha梯度差0.0209833、参考0.1103005，保留该失败。最后一轮将新密度Q/K/V与输出投影也放入FP32段，普通Flash仅内核入口转换bf16；原主干的medium矩阵设置保持原样。发现medium允许TF32后，坐标换算改用三维逐元素FP32乘加，以保护floor边界。参考明确分为旧逐分子bf16、highest完整FP32数学结果和新实际medium结果，不宣称与旧bf16逐元素一致。

第9次数值门控检查D1/D4、距离开关和是否bf16共8组合，包含两分子3/2个原子、体素边界与块外交集，以及全部输入、alpha、beta和Q/K/V/输出投影梯度。全部通过；bf16时非零梯度相对RMS误差最大0.006884，输出最大绝对误差2.772e-5。理论零key.bias梯度独立保留绝对误差，未混入相对误差汇总。原始记录在任务根`readout_equivalence_fp32.log`，本地完整副本为`tmp/pre-density-20260914/readout_equivalence_final.json`；随后A800执行`tests/test_density.py`为11 passed/6.20秒。

第9次测试动态命令为`bash tmp/pre-density-20260914/trial09.sh`，先运行上述门控，再以新批量读出执行D4、24×3、32 worker、完整清单均匀抽样种子2024、16更新/150秒窗口。整个动态命令已加按固定deadline=1789382216计算剩余时间的外层timeout，benchmark在数据准备前设置总预算alarm，首次forward后再限制为300秒与总剩余时间较小值。截止不因重启或门控重新计算。

第9次（attempt17，release `source_6f388ffc356c`）完成16更新且所有参数有限。冷首更新112.072秒，后15更新平均4.696533秒、中位4.691531秒、前向/反向/AdamW合计4.596246秒；供数等待0.017825%、GPU平均97.0968%、CPU4.94核、RSS79.81GB、已分配显存44.36GB。此轮采用完整清单的种子2024随机前缀，不能与旧缓存144实例基线混为同一数据窗口。原始证据`benchmarks/09_D4_b24_w32_full_batched_fp32/result.json`。

第10–14次动态测试命令为`bash tmp/pre-density-20260914/trial10_14.sh`，D1分别24×3、36×2、72×1，D4分别36×2、72×1；全部32 worker、种子2024，共用每次72实例前缀。每个子试验独立加载官方权重、生成模型、保存result.json和OOM等失败，不跨批量复用模型状态。D1最多24更新/120秒、D4最多16更新/150秒；不是正式训练。

第10/11/12次D1均完成24次更新且参数有限，共用消耗样本前缀SHA256 `928be3fe87452f2de03dde95f7b4da42f30311f9e19761f2467a70adcee978c5`。24×3、36×2、72×1的冷首更新分别34.699、44.254、73.106秒；后23更新平均1.614201、1.476974、1.366064秒，GPU平均87.03%、87.63%、90.78%，已分配显存峰值26.63、38.19、66.21GB。完整JSON分别在`benchmarks/10_D1_b24_w32_full/`、`11_D1_b36_w32_full/`、`12_D1_b72_w32_full/`。32 worker配prefetch1可预取32个micro batch；72×1的24更新短于该容量，因此此轮主要用于计算与显存选择，不能单独证明持续I/O达标。后续改用60更新和不同随机前缀，并单独统计后半段。

指定A800上的最终密度pytest为12 passed/7.09秒，包含原训练medium设置和highest设置下的bf16体素边界；证据`density_tests_gpu_medium.log`。北京时间17:25后SMILES任务已确认停止全部CPU/GPU负载，后续窗口不受该任务并发干扰。

第13次D4、36×2完成16更新，冷首更新142.798秒，后15更新平均4.590298秒、中位4.564674秒，计算4.488567秒、供数等待0.012670%、GPU97.5484%、CPU4.92核、RSS84.42GB、已分配显存66.25GB；与第9次共用相同1152实例前缀。第14次72×1首次forward成功，反向申请3.80GiB时设备只剩1.70GiB，PyTorch已分配70.62GiB、另保留未分配6.32GiB，抛出CUDA OutOfMemoryError，0次完整更新；失败保存在`benchmarks/14_D4_b72_w32_full/result.json`，没有据forward成功宣称该组合稳定。

临时基准使用`torch.backends.cudnn.benchmark=True`及`prefetch_factor=1`；原正式入口此前未显式对齐。主代理在a940482接入`train.cudnn_benchmark`和`train.prefetch_factor`，缺省保持旧行为；最终密度配置明确true/1。冷首更新包含cuDNN算法搜索，不能按热态速度估计首次启动。

第15–18次动态命令为`bash tmp/pre-density-20260914/trial_gate_and_15_18.sh`。先从主代理0a9c849完整git归档创建独立release，执行`pytest tests/test_density_sampling.py -q`的D1/D4×sdpa/flash四项真实采样检查，固定编码与逐步重编码逐项比较；根日志为`root_sampling_gate_0a9c849.log`。随后D1用72×1、D4用36×2，分别关闭/开启距离项，按相同种子2024前缀和完整AdamW更新时间配对；D1最多24更新，D4最多16更新，每项首次forward后最多150秒。推理修复仅改变FP32输入进入Flash内核的类型适配，不改变已冻结训练autocast性能代码。

采样门控第一次在归档的`create_release.sh`第2行因`pipefail\r`退出，尚未进入pytest，不能作为Flash数值失败或通过。只对0a9c849归档内`.sh`文件转换LF，Python文件内容保持原提交；`sampling_gate_0a9c849/source/root_gate_archive_manifest.json`记录原归档SHA256和转换文件清单，原失败保留控制目录err。重试测试命令为`bash tmp/pre-density-20260914/trial_gate_and_19_20.sh`，先重试采样，再执行供数窗口。

第15/16次D1均完成24更新、参数有限，比较后23个配对完整更新：普通Flash平均1.345325秒、带距离项1.358320秒，增加0.9659%；对应前向/反向/AdamW合计1.251478与1.262329秒。第17/18次D4均完成16更新、参数有限，后15个配对完整更新：普通局部注意力4.676952秒、带距离项4.666072秒，差-0.2326%；计算合计4.573437与4.561790秒。D4两项都是显式343邻域局部实现，不把它称为Flash。两种结构均满足完整更新增时≤10%的批准规则，最终统一保留距离项；负差仅表示本窗口耗时接近，不宣称距离项必然加速。

第19/20次固定D1 72×1、距离开启，分别15 worker/种子2025和32 worker/种子2026，完整训练清单均匀有放回抽样，最多60更新且首次forward后≤240秒。不同前缀降低直接复用同一实例的影响，但不清除共享操作系统文件缓存，因此结合`read_bytes`报告实际存储读取量，不将所有差异归因于进程数。随后仅按已选组合检查D1/D4包络入口，并对D4 36×2/32 worker执行最多48更新、首次forward后≤285秒的持续窗口，总改进次数不超过23、统一截止不变。

重试后的root采样门控实际4 passed、9 warnings、19.43秒，无skip。D1/D4×sdpa/flash四项均核对真实采样循环的固定编码与逐次重编码输出，包含三个候选拆成2/1批次、每批两步采样以及跨候选批次只编码一次。root门控release为`source_2b2c8cee0d7e`，启动证据`launches/379402/density_sampling_gate_0a9c849/`，原始日志`root_sampling_gate_0a9c849.log`。这些是已构造的非测试实例验收，不使用正式test数据或启动正式八模型实验。

第19/20次均完成60更新且最终参数有限。15/32 worker的冷首更新为77.474/86.540秒，后59次完整更新平均2.889477/1.965708秒，等待占53.3579%/29.2087%；后半窗口平均3.328139/2.251414秒、等待59.2766%/38.3831%。暖窗口GPU均值43.11%/64.92%，实际read_bytes增加142.288/112.059GB，进程树RSS峰值70.96/141.33GB，已分配显存69.71/65.99GB。选择32 worker作为目前最快稳定供数组合，但两个窗口均未达到持续读取等待≤10%的目标，不以短时预取表象宣称达标。原始证据为`benchmarks/19_D1_b72_w15_seed2025_long/result.json`和`benchmarks/20_D1_b72_w32_seed2026_long/result.json`。

第21–23次测试命令为`bash tmp/pre-density-20260914/trial21_23.sh`：D1-E 72×1/32 worker与D4-E 36×2/32 worker各最多4次完整更新，再以新的种子2029前缀检查D4-C 48次完整更新。最后预留的第24次获准测试只读memmap的进程局部MADV_RANDOM建议；先保留在临时性能入口，并核对首个实例全部56通道及几何逐元素相同，只有明确性能收益才考虑正式读取入口。此建议不修改地图、56通道数学或系统缓存设置。

第21/22次包络入口均完成4次真实加载、forward、backward与AdamW更新，全部模型参数有限，所有活跃密度参数有有限梯度。D1-E冷首更新84.234秒、后3次平均1.536967秒，已分配显存峰值59.23GB；D4-E后3次完整更新分别4.488856、4.594933和4.489049秒。原始证据分别为`benchmarks/21_D1_E_b72_w32_seed2027/result.json`与`benchmarks/22_D4_E_b36_w32_seed2028/result.json`；这些短窗口只验证包络实际训练入口，持续I/O由独立长窗口判断。

第23次D4-C完成48次完整更新，超过32 worker、每进程预取1个micro batch、36×2组合的16次全局更新预取容量。准备阶段2.408秒，冷首更新141.418秒（其中读取等待32.088秒，包含cuDNN首次算法搜索）；后47次平均4.571123秒，前向/反向/AdamW合计4.469354秒，读取等待0.018913%，后半窗口4.572701秒/等待0.020053%。暖窗口GPU均值96.0688%、CPU约5.33核，read_bytes增加125.206GB；冷首更新计数窗口增加30.190GB。密度读取与56通道构造的工作进程累计墙钟为冷首更新33.634秒、暖态每72实例29.689秒；该字段包括I/O等待，不是CPU纯计算时间。进程树RSS峰值87.18GB，已分配/保留显存峰值65.82/82.37GB，最终参数与全部活跃密度梯度有限。原始结果在`benchmarks/23_D4_C_b36_w32_seed2029/result.json`，release为`source_496ac6b4662e`，消耗实例前缀SHA256为`d6fe3eb722c0e5f0d42b917f648033fb154e2cf0cb5dda6ed57c48af44d799e6`。

第24次测试命令为`bash tmp/pre-density-20260914/trial24.sh`，D1-C 72×1/32 worker、种子2030新均匀前缀、最多60更新、首次forward后≤240秒，仍受固定总截止约束。其MADV_RANDOM仅为当前进程的只读memmap给出随机访问建议，降低稀疏裁块读取不需要的连续预读；不删除操作系统缓存，也不改变56通道数值。与第20次存在不同随机前缀及系统缓存状态，最终分别报告冷/暖read_bytes与密度累计处理耗时，不把全部差异直接归因于该建议。

第24次完成60更新，最终参数及全部活跃密度梯度有限，首个真实实例56通道和全部几何字段逐元素相同。准备2.086秒，冷首更新81.363秒、读取等待61.276秒，冷计数窗口read_bytes增加65.446GB；暖态平均1.842350秒、计算1.265162秒、等待26.1476%，read_bytes增加103.192GB。密度读取与构造的工作进程累计墙钟为冷首更新37.914秒、暖态40.048秒/72实例；相应第20次为48.166/44.984秒。暖窗口GPU均值68.16%、中位91%，CPU约9.35核，RSS峰值137.28GB，已分配/保留显存峰值67.11/69.47GB。新前缀的完整更新均值比第20次快6.28%、暖读取量约少7.9%，但前缀和文件缓存混杂且幅度有限，不足以确认该建议有独立明确收益，故只保留试验记录，不修改正式`docking/density.py`或完整通道模块。

第24次在预取之后的三个窗口为更新33–42、43–51、52–60，实际时长28.325/13.798/15.665秒，GPU样本24/12/13个，中位数25.5%/92.5%/77%，等待52.2853%/6.4831%/21.4586%。其中两段未达标，也不能据整段GPU中位91%称为供数稳定。原始结果`benchmarks/24_D1_b72_w32_seed2030_madv_random/result.json`，release为`source_f42ce7b9d94f`，消耗实例前缀SHA256为`9c6f28c22f74de7a4b8e588f508912f32376c51f9e1ff0016dbfbcfa24842f80`。

最终只读核查命令为`bash tmp/pre-density-20260914/resource_status.sh`，在gnode09运行；结果保存在`tmp/pre-density-20260914/final_resource_status.json`。北京时间约18:34核实控制器PID2383处于sleeping且没有子进程，指定GPU利用率0%、占用5MiB、没有计算进程；after_lock_379402和父目录try_lock_379402存在，kill_lock_379402不存在。节点时钟比本地UTC工具约慢4分钟，整个执行仍使用最初约定的固定deadline=1789382216与外层timeout，没有延长预算。本次自然结束，不需要写kill_lock或终止资源作业，未scancel、未清缓存、未改公共Conda或共享数据资产。

实现与测试交付至此结束，不再增加第25次尝试。正式运行仍需在项目隔离环境中提供einops0.8.1；本次仅通过任务根`deps/`提供该依赖，主项目已另列`requirements-density.txt`。D1持续供数目标未达这一限制保留，D2/D3和正式八模型由后续获准阶段处理。
