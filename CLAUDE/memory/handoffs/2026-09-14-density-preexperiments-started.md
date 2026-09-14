# Handoff: 两项密度前预实验已获准并接管资源

Date: 2026-09-14

## 当前状态

非密度收口已完成，Learn/CUMULATIVE为a78ae8f。用户随后批准SMILES构图迁移与单A800密度性能两项预实验，各三小时；通过后按D1→D4→D2→D3完成八个RA＋T0模型。当前任务已建立预实验goal并开始实现，尚未开始正式密度训练。此记录保存两项作业提交节点，持续状态以[预实验总日志](../../../日志/预实验（一 --二之间）/总日志.md)及逐实验日志为准。

## 已完成内容

主代理确认全部Git引用及工作树的提交者时间，a78ae8f为唯一最新且工作区干净；从该点创建codex/density-preexperiments。SMILES代理工作树为 `C:/Users/15919/Desktop/PocketXMol-worktrees/pre-smiles`、codex/pre-smiles；密度代理为同级pre-density、codex/pre-density。主代理负责公共Dataset、训练入口集成和共享文档；代理分别提交自己的实现，再由主代理集成。当前3契约、AGENTS、阶段索引和映射已回填新授权。

SMILES代理已用371591的8个CPU进程完成初轮66878实例迁移，报告0失败、14.94秒；主代理只读读取报告核实。但GPU验收发现bf16梯度超界及个别原始离散键类别不同，正在补核；不能把sanitize后的图同构当成GPU等价，SMILES链尚未验收通过。

密度模块与模型接线已形成，主代理补齐DataModule、Dataset和sampling的model.density唯一配置入口、官方初始化允许的新参数前缀；本地密度／数据集成10项通过。发现并修正bf16 autocast可能改变home体素选择，几何段固定FP32，已补CUDA贴边检查，真实GPU门控正在执行。

## 已定边界与资源

- SMILES预实验从07:38:59 UTC开始，截止10:38:59 UTC；密度从07:40:56 UTC开始，截止10:40:56 UTC，即北京时间18:38:59／18:40:56。主要工作到时停止扩大验收，按实际证据汇总，不另起三小时。
- SMILES不支持率按实际检查occurrence严格小于1%；全66878完成时最多668例。CPU与GPU覆盖分开报告，未测不声称通过，测试保持446分母。系统数值漂移不按例外豁免。
- 当前坐标产物为 `/storage/penghongen/AdaLigand/Ori_Data/smiles_assets/SMILE_coords/v1`，新清单为 `/storage/penghongen/PocketXMol/data/smiles-v1/frozen`。用户明确允许在SMILE_coords新增；旧公共图、旧坐标及原型尝试保留不覆盖。公共图版本来自已发布smiles_graphs_v1.npz和smiles_symmetries_v1.npz。
- 371591在gnode09，控制 `/home/penghongen/Feedback/Pocket_Plus/allocations/371591`，用于SMILES GPU与其CPU验收；运行证据 `/storage/penghongen/tmp/pxm_pre_smiles_20260914`。
- 379402_0实际JobId和锁编号379402，控制 `/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration/运行日志与统计/slurm/allocations/379402`，用于密度GPU性能；运行证据 `/storage/penghongen/tmp/pocketxmol_density_20260914`。两卡各16CPU，保留after_lock，不用scancel，不处理未指定任务。
- 密度性能最多24次改进，每次首次forward后最多5分钟。有效global72，比较24×3、36×2、72×1；利用率目标未达但科学正确稳定时，预算到报告最快配置继续。距离偏置完整更新增时不超过10%才保留，否则统一普通FlashAttention，计入同一预算。
- 密度固定56通道；通道mask使用含UNK完整原始受体，RA图仍排UNK。Pocket_Plus U-Net无recycle，D1为6³编码，D4/D2完整U-Net与7³邻域／并集；D3预测top4096、SMITED条件头隐藏64、eta=.1，best仍原dock val/loss。

## 下一步

继续两个代理的有界预实验，定位SMILES实际GPU差异；若需要改变科学含义即向用户报告选择，不能静默放宽门限或改正式精度。密度性能可独立使用原已验收图链继续，不能依赖未通过的SMILES链。主代理集成后执行两遍自查，独立布局／Git、注释、逻辑各一轮全面审查，再仅复核既有问题；函数布局例外由用户指定的审查代理批准。

两项预实验验收与双线Git收口后建立正式八模型goal。两卡分别按D1→D4→D2→D3推进中心、包络模型，每模型训练后直接best测试和CPU评价；不完整验证集采样。新增或撤卡只调整获准模型，最近完整checkpoint恢复可重做未保存进度。逐实验与总日志同步，正式短命令和测试／门控命令分开。没有CLAUDE/memory/index.json或projects状态文件，因此只新增本handoff，不另造记忆系统。

## 重新打开的文件

- [预实验共享日志](../../../日志/预实验（一 --二之间）/总日志.md)
- [科学契约](../../../想法/方案草稿/9-8-科学契约.md)
- [工程细节](../../../想法/方案草稿/9-8-工程与实现细节.md)
- [边界与核查清单](../../../想法/方案草稿/9-8-边界与核查清单.md)
- [计划执行映射](../../../日志/计划执行映射.md)
