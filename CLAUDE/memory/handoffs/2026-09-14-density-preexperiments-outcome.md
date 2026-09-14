# Handoff: 密度预实验结束，SMILES模型输入仍待决定

Date: 2026-09-14

## 当前状态

两项预实验的主要工作已结束，未启动正式密度训练。SMILES迁移的化学图／坐标检查完成66878实例，但原始模型键类别28650例（42.8392%）不等价，独立Kekulize后仍有16477例不等价，不能按1%例外处理。用户尚未批准从旧单双键模板统一改为公共SMILES芳香键输入；候选代码与坐标全部保留，正式入口未切换。

密度D1/D4实现、ALL56数值门控、真实GPU训练与采样验收、主代理两遍自查及一轮独立全面审查后的窄复核已完成。性能在三小时内止于第24次；D4达到本次三段持续供数目标，D1未达到。详细状态与来源见[预实验总日志](../../../日志/预实验（一 --二之间）/总日志.md)，本handoff不替代实验日志。

## 已完成内容

- D1采用72×1，D4采用36×2，均32加载进程、预取1批、cuDNN benchmark=true、bf16-mixed、激活重算。D4的72×1反向OOM，不选用。
- D1常规完整60更新热态均值1.965708秒、等待29.2087%；预取后三段GPU中位41.5/35/92%，前两段未达标。D4完整48更新热态4.571123秒、等待0.018913%；三段GPU中位均100%，等待约0.02%。这些13–51秒的稳定窗口不保证正式长训练表现。
- 距离项完整更新开销D1为+0.9659%、D4为-0.2326%，按已批准10%规则保留。最后MADV_RANDOM试验仍未达标且受不同前缀／页缓存混杂，未纳入正式代码；通道模块保持原ALL实现。
- A800上的D1/D4×sdpa/flash真实采样为4通过、0跳过，修复了FP32缓存编码不兼容Flash的问题。只在新密度Flash内核入口转bf16并恢复输出类型，原T0机制不变。包络两个入口各完成4次真实AdamW更新。
- 本机同范围实现检查38通过、5跳过；3项Flash由A800补实测，2项旧真实T0检查缺少本机复制资产，未声称运行。XML为tmp/density-preexperiments/implementation-end.xml。

## Git、产物与资源

共同基点a78ae8f36155a4bdae54aa79b6d30f79ffebc59e；实现分支codex/density-preexperiments，学习分支Learn/density-preexperiments。精确端点及等价验证报告为tmp/density-preexperiments/dual-endpoints.json，通过后Learn/CUMULATIVE进入本轮学习端点；下一轮从累计学习端点开始。SMILES候选另留codex/pre-smiles的db0f77f70e4c9e2bfaf5e0b39946404031e3e736，未整分支集成。临时预实验代码退出当前Git活跃树，历史、磁盘及服务器证据保留；不删除其他修改或推送远端。

密度证据根/storage/penghongen/tmp/pocketxmol_density_20260914，SMILES证据根/storage/penghongen/tmp/pxm_pre_smiles_20260914；候选坐标在/storage/penghongen/AdaLigand/Ori_Data/smiles_assets/SMILE_coords/v1，候选清单在/storage/penghongen/PocketXMol/data/smiles-v1/frozen。

379402已只读确认after_lock和try_lock存在、kill_lock不存在，控制器PID2383 sleeping、无子进程，指定GPU无计算进程；不再新增GPU尝试。371591最终也由执行代理只读核实：原控制目录/home/penghongen/Feedback/Pocket_Plus/allocations/371591内after_lock、父目录try_lock存在，kill_lock不存在；PID20225 sleeping且无子进程，指定GPU同为0%／5MiB且无计算进程。两项资源全部保留。未经新许可不删服务器产物、不取消Slurm资源。当前einops0.8.1只位于预实验隔离deps，正式环境尚未安装。

## 待决定与下一步

需要用户明确是否接受公共SMILES芳香键编码，从而将验收改为新输入链的GPU正确性；此变化会使与旧无密度结果的差异同时包含模型键编码变化。未经决定，不切换候选数据，不启动正式密度模型，不自行补无密度对照。

解决输入边界后，按已批准D1→D4→D2→D3执行中心、包络两模型，训练结束直接best测试和CPU评价，不完整验证集采样。D2/D3仍须实现及验收，不能当作本次已完成。正式启动先准备项目隔离依赖，再记录极短正式命令，勿把tmp性能／门控命令当正式入口。持续复查I/O，D1尚有明确瓶颈；新增或撤回GPU只调整获准独立模型排程，恢复最近完整checkpoint允许重做未保存进度。

## 重新打开的文件

- [预实验共享日志](../../../日志/预实验（一 --二之间）/总日志.md)
- [SMILES预实验](../../../日志/预实验（一 --二之间）/1-SMILES构图与GPU验收.md)
- [密度性能预实验](../../../日志/预实验（一 --二之间）/2-密度I_O与GPU性能.md)
- [科学契约](../../../想法/方案草稿/9-8-科学契约.md)
- [密度模块接口](../../../models/README-density.md)
