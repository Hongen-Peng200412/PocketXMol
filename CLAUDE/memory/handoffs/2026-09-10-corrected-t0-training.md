# Handoff: 中心T0纠偏完成并从官方参数独立重训

Date: 2026-09-10

## Current State

本记录取代2026-09-10-first-formal-training.md的当前状态和中心训练条件。旧记录只保存当时历史，其中“中心训练每次随机C5”对T0是错误的。端到端goal仍active：完成六个无密度模型、官方对照、完整验证／测试和报告，之后完成双线Git；密度正式实验仍止于用户选择。

正确的B-C-T0-RA已在既有371591、gnode09、单张A800 80 GiB和16核上从官方参数重新开始。正式命令为 `bash 训练与运行/sh/train_docking.sh B-C-T0-RA --logdir /storage/penghongen/PocketXMol/training/B-C-T0-RA-C0`。没有--resume，优化器和调度器重新建立。实际运行源码对应693c7ec，release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_9fa23b858837/PocketXMol`，launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/train_B-C-T0-RA-C0_job371591_20260910T164945`。

W&B在线run为 `pencounkdual-111/PocketXmol_raw/ceqrh2ve`，显示名B-C-T0-RA-C0；训练产物根 `/storage/penghongen/PocketXMol/training/B-C-T0-RA-C0/`。启动记录已观察第24个优化器更新、lr=1e-4；保存配置为72×1、center_translation=false、无resume，运行源码快照明确T0使用C0。首次800步验证尚未到达。本记录只对应启动事件，不按每次检查或验证改写。

## Completed

- 已核实旧B-C-T0-RA的进程8895属于371591，使用该作业实际kill_lock终止，最终日志17901步、最后定期检查点17600步、退出137。after_lock及allocation保留，未scancel。旧目录 `/storage/penghongen/PocketXMol/training/B-C-T0-RA/`、源码、配置、检查点、日志和W&B 9wkyn4qn全部保留，均为错误C5训练／验证条件证据，不可当正确T0基线或新运行初始化。未为旧模型生成正式候选。
- 生产只改两处：OccurrenceDataset的动态偏移条件变成shuffle且protocol=C5；DataModule.setup从已有dock.center_translation选择中心T0的C0、中心T1的C5、包络E。原T1噪声公式、正式sampling输入能力、原模型／loss／置信度保持。三份9-8契约、六配置说明、README、实验日志和映射已同步。
- 本次纠偏完成两遍主代理自查、两轮三类独立审查，之后只窄核具体问题。布局代理批准既有OccurrenceDataset.__getitem__及DataModule.setup入口例外。用户注释去除Docstring后的AST不变，依用户新授权保存于6cd023f；纠偏提交ec01242、审查补强a1132dd、精确边界及留证693c7ec。用户随后自行提交eaf1165，仅忽略/tmp/，已保留。
- 服务器CPU376811及加强边界后的376817都以8核完成14项测试，包含真实非test训练5ftl/0；该实例T0／T1口袋原子数488／465仅作诊断。最后精确等号夹具仅把pytest构造三原子配体平移到g=0，本地两项窄测通过，独立窄核证实<与<=可区分。真实资产未变。
- 371591中的新T0 GPU短验收通过：72×1、bf16、两次更新、原C0验证、一组C5的2候选采样及评价；峰值分配约44.91 GiB。验收运行不创建正式W&B或科学成绩，测试命令与正式命令分别记录在实验日志02。生产AST和配置解析值在后续测试／说明整理后保持不变，没有重复不必要的GPU验收。
- 共同数据不重新准备。冻结train／validation／calibration／test数量仍65290／781／361／446；test的ALL／CAP10／HF10_TO5为446／272／227。原坏图13及坏SMILES192已按实例排除，不修复。源语言模型身份为SMI-TED Light 289M。其它准备证据仍见日志00。

## Decisions

T0训练C=g，g是完整配体重原子世界XYZ坐标均值；以残基重原子质量中心到g严格<15 Å选袋，配体和受体共同减C，仅原高斯。T1每次训练只抽一份delta=r*u，r在0–5 Å均匀、u独立球面均匀，以完整g+delta选袋并定原点，局部GT均值-delta；原高斯后、同构重分配和固定字段恢复之前加同分子共享s*delta，s=1-level_dict['pos']。受体和目标不跟随新增平移。监督验证T0用C0、T1用冻结C5、包络用E。

推理实际给定中心c和口袋整条轨迹固定，C=c。首步T0/T1均原纯高斯，不强制候选质心归零。后续T0只原高斯，T1还加-s*mean(Z)，Z为当前干净局部预测；不再读GT中心／偏移，不另抽delta或截断到5 Å，最终只加回C一次。T0+C5与T1+C0都保留。包络原判据和原点不变，恒T0。六模型及既有C0/C5/E评价矩阵不增不减。未来密度T0训练请求g、T1请求g+delta，裁块内缩不改原点，本次未扩展密度。

其它训练与评估口径继续以三份9-8文档为准：无密度优先72×1、bf16；AdamW lr=1e-4、warmup0，每800个优化器更新走完整原val/loss；Plateau相对阈值1%、patience5、factor0.2，第3次实际下降停止，上限40000。保留原始val/loss最低best、last和所有定期检查点。原loss、置信度头和self-ranking不变，不适配tuned ranker。正式每实例50候选、100步；完整validation只ALL，test三视图共用候选池，所有失败保留分母。

仅可用已获准371591 GPU，不申请额外GPU或修改其它作业；CPU每任务8核，纯CPU并发≤96核。服务器删除仍需明确许可，原AdaLigand资产只读。只允许当前作业内已授权的run_cmd／try_lock／kill_lock操作，after_lock必须保留。

## Open Questions

当前没有阻止训练的待用户决定项。密度正式实验选择和新增GPU权限仍未开放。用户允许其对我们文件所做的注释／Docstring修改在核实不改逻辑后顺手提交；不要丢弃或覆盖其它用户修改。

## Next Actions

1. 训练已正常推进，按用户要求静默等待60或90分钟；使用多次Start-Sleep -Seconds 300，单次工具等待不超过60秒以便接收输入。不要heartbeat，不要因正常训练而结束goal。
2. 醒来检查371591、新运行ceqrh2ve、优化器更新、C0验证、OOM／非有限值／裁批、W&B和检查点。新启动元数据 `/storage/penghongen/PocketXMol/control/371591/train_B-C-T0-RA-C0_start.json` 保存out／err起始字节20880102／62752，只看此后输出，避免把旧错误训练或短验收日志当成新训练。正常结束后原控制器恢复try_lock，保留after_lock；结束时按实际检查点状态核对第三次下降或更新上限、best及完整停止状态。
3. 继续其余五个模型和正式候选验证／测试。中心T0-RB尚未运行，直接采用修正代码；包络T0不受本次错误影响，T1保留原科学机制。仅一张GPU，训练／采样顺序安排；CPU评价可在已授权上限内并行。每次正式启动前留独立实验记录和极短正式命令，采样配置填写实际best，不能用旧错误T0或门控权重。
4. 训练完成、正式候选任务提交或报告形成时再记录有意义的handoff。完整六模型及官方报告完成前不把goal标complete。当前仍是同一实现分支codex/pxm-receptor-baselines，共同基点0412824；实现端点仍随后续正式配置和报告变化，不提前推进Learn/CUMULATIVE。最终重建学习历史并做端点等价核验。

## Files To Reopen

- [中心修复与重训日志](../../../日志/第一类实验（不加密度信息）/02-T0中心契约修复与重训.md)：全部纠偏、审查、测试、正式命令和实际产物证据。
- [旧错误训练日志](../../../日志/第一类实验（不加密度信息）/01-B-C-T0-RA.md)：无效条件和停止证据；不得从其目录续训。
- [共同准备日志](../../../日志/第一类实验（不加密度信息）/00-实现与共同数据准备.md)、[总日志](../../../日志/总日志.md)、[计划映射](../../../日志/计划执行映射.md)及三份当前9-8规格。
- `/home/penghongen/Feedback/Pocket_Plus/allocations/371591/`：实际控制器的run_cmd、out、err、after_lock；try_lock位于其父目录。Pocket_Plus名称来自历史allocation，不授权操作其它作业。
- 项目代码同步根 `/home/penghongen/My_Project/PocketXMol`；运行用冻结release。项目环境 `/storage/penghongen/PocketXMol/runtime/venv/bin/python`，共享派生根 `/storage/penghongen/Adaligand_Build/Ori_Data/pocketxmol`，原源根 `/storage/penghongen/AdaLigand/Ori_Data` 只读。
