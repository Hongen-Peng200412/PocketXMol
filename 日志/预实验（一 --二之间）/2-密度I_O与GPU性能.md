# 密度读取与GPU性能预实验

当前状态：正在实现和验收D1、D4密度训练链。开始时间为2026-09-14 15:40:56（北京时间），主要工作截止18:40:56；最多24次改进，每次首次真实forward后观察不超过5分钟。GPU利用率90%及读取等待不超过10%是优化目标，预算结束且科学正确、运行稳定时报告最快稳定配置，不无限延长。

本记录依据当前三契约及本次用户批准的密度选择，覆盖真实加载、forward、backward与AdamW更新的性能预实验；不启动正式D1/D4/D2/D3八模型训练。三契约入口为[科学契约](../../想法/方案草稿/9-8-科学契约.md)、[工程细节](../../想法/方案草稿/9-8-工程与实现细节.md)、[核查清单](../../想法/方案草稿/9-8-边界与核查清单.md)。

## 固定科学内容

56通道在48³源体素裁块上按Pocket_Plus既有顺序计算；受体占据掩码来自整个原始受体，包括UNK；RA图继续排除UNK。D1只运行三次下采样与原四层8头×192维三维RoPE注意力，输出6³×256；D4运行完整四次下采样/上采样U-Net，输出48³×48。单次密度编码，无recycle与无关输出头。

去噪器每个节点更新后、边及位置更新前注入独立4头×64维读出，输出320维，alpha初始0.1、g恒为1。D4按当前原子home体素±3取与裁块的交集，空集合零残差。距离偏置为qk/8−softplus(beta)×||(x−p)/10 Å||²，初始系数1；以完整优化器更新时间相比普通Flash增时不超过10%为统一选择依据。D2/D3留待后续实施，不能把D1性能代表完整U-Net。

## 资源与证据

资源为Slurm实际JobId379402（数组显示379402_0）、gnode09、A800 80GB、16CPU，GPU UUID为GPU-adbf8fc8-5a4a-87e3-853b-c9cadcbdf74b。控制目录为`/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration/运行日志与统计/slurm/allocations/379402/`；try_lock在其父目录。首次只读核查时控制器PID2383处于等待，after_lock与try_lock存在、kill_lock不存在、指定GPU空闲。

代码使用隔离实现工作树`C:/Users/15919/Desktop/PocketXMol-worktrees/pre-density`，起点a78ae8f。服务器代码与预实验产物使用本次独立临时根；每次实际运行冻结自己的release，不覆盖共享代码或已有实验产物。

## 已完成的本地验收

`tests/test_density.py`：6 passed。检查48³裁块、实际XYZ spacing与角点、原始UNK受体掩码、D1停止于第三次下采样、D4越界交集及空集合、共同刚体变换，以及密度距离偏置与三维RoPE注意力的输出/梯度一致性。首次测试发现多batch增广常数项广播错误，已修正后重跑通过。

## 测试命令

以下是测试命令，不是正式训练：

```powershell
& 'C:/Users/15919/Desktop/PocketXMol/tmp/pxm-20260910/venv/Scripts/python.exe' -B -m pytest -q tests/test_density.py --disable-warnings
```

## 尝试历史

尚未执行真实GPU预实验。后续在此记录每次命令、配置、退出状态、冷暖读取、完整更新时间、等待比例、CPU/RSS/GPU与显存观测，以及最快稳定配置。
