# Handoff：SMILES与密度前置集成收口，正式实验等待再次授权

Date: 2026-09-14

## 当前状态

用户最新要求先完成全部前置集成，包括SMILES迁移、GPU验收、审查、日志和双线Git，然后汇报具体计划并停止本阶段goal。正式密度八模型与修正编码的无密度复验全部等待用户再次明确允许，不能沿用旧端到端授权自动启动。三张A800保留after_lock及父try_lock，没有正式训练、完整测试、正式评价或新正式W&B记录。

当前读[预实验总日志](../../../日志/预实验（一 --二之间）/总日志.md)前半即可了解验收、局限、资源与后续计划；后半是历史尝试，不能将旧“待实施／正在”视为当前状态。全项目[阶段索引](../../../日志/总日志.md)只放简短入口。

## 已完成

精确SMILES新链已进入Dataset、采样及评价，使用公共图、精确字符串和按同一顺序保存的沉积坐标。66878实例化学图及迁移映射通过，1979个精确SMILES离散输入与固定官方65488cf全部一致。旧ligand_object接入有28650实例未按官方芳香键类别编码；历史六模型及官方对照因此需要明确旧编码限定，旧产物没有删除或覆盖。

非测试GPU检查覆盖三个训练实例×C0/C5/E：图、原点和局部坐标一致，FP32前向／梯度达到既定门限，900个同状态原噪声及回写步骤精确一致，GT位置清零、口袋原点固定、世界坐标只还原一次。结论为输入及T0机制实质对齐，不能写成全部GPU数值相同。AdamW一次更新只记录有限性及最大参数差1.99966e-4，未证明逐参数等价。6baj/14 E第84步严格重复失败，原GPU归约引起约1e-6 Å差值，触发原kNN等距成员切换；证据与失败状态保留，无精度或kNN改写。

D2与D3已经集成。D2按每分子当前坐标去重汇合7³体素邻域；D3采用冻结768维语言表征、完整48³特征及固定4096预测选择，标签只用于训练和监督验证，原val/loss选best不变。D1/D4旧实现对照及D3原Pocket_Plus辅助损失数值检查完成。主代理两遍自查完成，布局/Git、注释、科学三类独立审查完成一轮全面检查及问题窄复核；具名布局例外已经由布局代理批准。

八个密度配置在379402实际A800完成真实train/validation、bf16前向/反向、两次AdamW更新及2候选3步采样和CPU评价，8通过、0跳过，705.84秒；后续四类结构×sdpa/flash缓存检查8通过、0跳过，29.82秒。源5af144f，release在`/storage/penghongen/tmp/pxm_formal_smiles_20260914/density-gate/releases/PocketXMol_706febc31afc/PocketXMol`，结果XML与原日志同根，逐配置JSON在`/storage/penghongen/tmp/pocketxmol_gpu_checks_density_gate_5af144f_r2/pytest/`。首次CRLF启动失败未执行GPU，仅本次新发布副本shell转LF，原归档及变换证据保留。

本机实现与学习候选同范围检查均85通过、8跳过（198.96秒、288.59秒）；跳过为5项本机Flash内核不可用和3项服务器非测试资产未复制，不混入通过计数。完整证据、本机学习端点结果、两条线确切哈希及全树核验保存于`tmp/formal-smiles-density/dual-endpoints.json`和对应XML。共同基点92d08d6，实现线`codex/smiles-density-formal`长期保留，学习线`Learn/smiles-density-integration`按文档→数据→模型→组合→全部测试排列；只有端点等价且检查完成后快进Learn/CUMULATIVE。下一次实现从最终Learn/CUMULATIVE开始，不从旧worker分支恢复。

## 决策与限制

2026-09-14前置集成收口：实现与学习候选的同范围本机检查均为85通过、8跳过（分别198.96秒、288.59秒）；必要真实D3-C入口在实现端已通过，学习端另外实际运行1通过、0跳过，112.07秒。学习端launch为`learning_d30a132_D3C_gate`，输出`density-gate/learning-training-sampling.xml`，本地完整记录`tmp/formal-smiles-density/learning-gpu-evidence.json`。这次只重跑同一真实小范围，没有正式运行。 goal已确认paused并保持暂停。

后续固定RA＋T0，训练bf16-mixed、推理官方FP32张量路径及medium矩阵设置，保留原噪声、同构重分配、loss、置信度和self-ranking。中心训练/val用C0，包络E，冻结C5只在评测层提供实际中心。ALL56通道保留原模块，不recycle、不重采样、不改变裁块内缩前确定的模型原点。原OOM裁批仍按现有已批准契约保留。

三小时I/O预实验24次已结束：D1 72×1的持续数据等待29.2087%、GPU均值64.92%，未达到三个窗口目标；D4 36×2等待0.018913%、GPU均值96.07%，三个窗口中位数均100%。D4 72×1反向OOM。不能把短入口验收外推为D2/D3长期性能，也不能自动延长本次优化预算。

## 资源与下一步

371591／gnode09指定GPU-0e53cfe2-fd47-7b09-8bed-d71aa08d25a1，未来中心D1→D4→D2→D3；379402_0（实际379402）同节点指定GPU-adbf8fc8-5a4a-87e3-853b-c9cadcbdf74b，未来包络同顺序。两卡20:48以后只读确认控制器等待、after_lock和父try_lock保留、kill_lock不存在、GPU各0%／5MiB。控制目录分别为`/home/penghongen/Feedback/Pocket_Plus/allocations/371591`及`/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration/运行日志与统计/slurm/allocations/379402`。

378693／gnode10指定GPU-0e253751-cce3-c71d-c2df-a222fec3759b，未来无密度B-C-T0-RA-SMILES→B-E-T0-RA-SMILES→official-SMILES测试。Matcher进程组13179已按明确授权经实际kill_lock停止，控制器43832和after_lock保留，旧源码、命令和产物未删除。20:55核实控制器无子进程、父try_lock存在、kill_lock不存在、GPU 0%／2MiB，证据`tmp/formal-smiles-density/resource-378693-final.json`。不要再次终止或重新提交Matcher。

得到用户再次允许后，三卡并行独立模型；每模型官方参数重新初始化训练→best的完整测试→同资源CPU评价后再下一项。D1 72×1，D4/D2/D3 36×2，无密度72×1。训练当前契约是每800更新val/loss、相对阈值1%/patience5/factor0.2、第三次实际降学习率停止、上限40000更新；不要恢复旧1000步或240000上限。测试每实例每协议50候选100步，batch50；训练结束不做完整验证集采样。正式初期、稳定窗口和换卡后记录I/O/GPU，优先复查D1瓶颈；资源增减只能按用户届时明确授权，撤卡接受从最近完整last恢复并重做未保存更新。

正式命令未执行，不能把门控当成正式运行。已准备的短入口及产物名在预实验总日志与对应实验日志；获准启动前记录实际release、launch、W&B和命令。密度每D_i一份日志含中心／包络两模型，无密度复验继续原第1／3／7日志，旧尝试在后半。取得新进度同步逐实验和所属总日志，只有阶段完成或明确分叉再写handoff。

## 重新打开的文件

- [预实验总日志及后续计划](../../../日志/预实验（一 --二之间）/总日志.md)
- [SMILES证据与严格数值边界](../../../日志/预实验（一 --二之间）/1-SMILES构图与GPU验收.md)
- [密度GPU入口与历史I/O证据](../../../日志/预实验（一 --二之间）/2-密度I_O与GPU性能.md)
- [正式密度阶段入口](../../../日志/第二类实验（密度分支的训练）/总日志.md)
- [当前工程契约](../../../想法/方案草稿/9-8-工程与实现细节.md)
