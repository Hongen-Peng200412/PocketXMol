# 密度读取与GPU性能预实验

2026-09-14 16:58状态：D1第6次已从完整训练清单均匀有放回抽样，完成24次真实AdamW更新。24×3、15 worker的持续热态平均4.647秒/更新，供数等待46.22%、GPU平均34.83%、CPU约4.73核、RSS峰值41.29GB；前几次快速更新受预取影响，不能据此宣称I/O达标。第8次正在尝试32个单线程加载进程覆盖I/O等待，仍使用379402原有16CPU，不申请额外CPU。批量读出和FP32残差标量乘法是待验收候选，尚未作为正式稳定实现；距离保留与批量组合仍待完整更新比较。

当前状态：正在实现和验收D1、D4密度训练链。开始时间为2026-09-14 15:40:56（北京时间），主要工作截止18:40:56；最多24次改进，每次首次真实forward后观察不超过5分钟。GPU利用率90%及读取等待不超过10%是优化目标，预算结束且科学正确、运行稳定时报告最快稳定配置，不无限延长。

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
