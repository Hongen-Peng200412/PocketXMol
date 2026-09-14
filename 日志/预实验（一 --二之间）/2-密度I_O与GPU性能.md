# 密度读取与GPU性能预实验

2026-09-14 16:16状态：已在指定A800通过成熟Pocket_Plus三维RoPE输出、输入及参数梯度核对、完整单次U-Net输出核对，以及实际bf16 autocast下普通Flash和显式FP32距离分数的输出/梯度核对。GPU密度测试7 passed。开始第1次D1全局72、24×3真实更新基准；尚未取得完整优化器更新耗时，不能据此声称性能目标达标。

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

指定Slurm作业中的数值测试命令为`bash tmp/pre-density-20260914/preflight.sh`，实际Python为`/storage/penghongen/PocketXMol/runtime/venv/bin/python`。临时任务根为`/storage/penghongen/tmp/pocketxmol_density_20260914`，其中`source_equivalence.log`、`attention_precision.log`和`density_tests_gpu.log`保存数值证据；每次动态命令保存在`control/<尝试名>_run_cmd.sh`，对应冻结目录保存在`releases/`，启动证据在`launches/379402/`。现有PXM环境为torch2.6.0+cu124，缺失的einops0.8.1仅安装于本任务`deps/`，未修改共享环境。

## 尝试历史

门控0a（控制器attempt3）在读取上传脚本时因CRLF失败，随后统一上传脚本为LF。门控0b（attempt4，release `source_b90e55fa0d1e`）发现PXM环境缺少einops，已隔离补齐依赖。门控0c（attempt5，release `source_468d969a1370`）通过成熟RoPE和单次完整U-Net核对，但bf16 QK增广对FP32显式距离参考最大绝对误差0.06934，未通过既定输出门限，未放宽门限。

门控0d（attempt6，release `source_43a3e41e18a2`）比较注意力精度：bf16增广输出相对RMS误差0.916%、输入梯度0.85%至1.38%；fp16增广分别0.115%、0.149%至0.224%；FP32显式距离分数的SDPA数学内核分别0.168%、0.153%至0.278%。后者避免增广平方项的低精度舍入，选为下一轮真实性能候选。高效SDPA内核拒绝FP32偏置与bf16 query的组合，未采用。门控0e正在实际bf16 autocast上下文重验该候选；完成前不标通过。

门控0e（attempt7，release `source_d042fa1a2343`）全部通过，未放宽输出或梯度门限：普通Flash输入梯度相对RMS误差约0.27%；显式FP32距离SDPA数学内核各输入梯度误差为4.75e-6至1.02e-5，位置及系数梯度误差为1.74e-7至2.45e-7。随后指定A800上的密度pytest为7 passed，用时4.73秒，覆盖bf16体素边界。
