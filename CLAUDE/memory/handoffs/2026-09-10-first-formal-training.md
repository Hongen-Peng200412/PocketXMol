# Handoff: 首个无密度模型正式训练已启动

Date: 2026-09-10

## Current State

端到端goal仍处于active：完成六个无密度模型、官方对照、完整候选验证／测试和最终报告，随后完成双线Git收口。当前只启动了第一个正式模型B-C-T0-RA，其余五个及正式采样尚未开始。密度正式实验须等用户看过无密度结果后选择；不能自行增加对照或学习率试错。

B-C-T0-RA在已授权371591、gnode09的一张A800和16核上运行。正式命令为 `bash 训练与运行/sh/train_docking.sh B-C-T0-RA`。实际release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_dbfde52856a9/PocketXMol`，launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/train_B-C-T0-RA_job371591_20260910T112700`。W&B run为 `pencounkdual-111/PocketXmol_raw/9wkyn4qn`，online；已看到第70个优化器更新、lr=1e-4，尚未到首次800步验证。这是启动事件记录，不把后续每次轮询写入handoff。

## Completed

- 主实现已完成两遍自查、两轮三类独立审查及实际问题的窄复核；函数布局例外由用户授权的独立审查代理批准。不得为同一实现重开第三轮全面审查或常驻代理。
- 累计实例筛选有效完成。正式freeze作业376632成功，`/storage/penghongen/PocketXMol/data/`中的train／validation／calibration／test分别为65290／781／361／446个实例；测试ALL／CAP10／HF10_TO5为446／272／227。保留／排除并集和原148655候选完整对应，66878保留、81777排除。
- 13个坏模板、192个坏SMILES按实例排除，不修复也不重新计算语言向量。实际语言NPZ的model_name为 `SMI-TED Light 289M`，早期误写导致空冻结，已修正；失败30份记录保存在data/preparation/failed_model_name_20260910T092948/。真实源数据CPU检查376591及重跑数组376592、最终freeze376632均成功。8月Builder只作历史。
- CPU必要检查、官方参数及bf16 GPU检查已通过。用户最新要求六份无密度配置优先72×1，提交136c621已落实；真实GPU检查2项通过，实际batch72、不累积、各2次优化器更新及2候选评价，峰值分配显存约36.7／44.3 GiB。该短检查的权重、候选和指标不是正式产物。
- 正式训练使用短TMPDIR `/storage/penghongen/tmp`，子进程文件句柄软上限65536，硬上限保持131072；原控制器软上限1024不变。这已解决真实DataLoader临时socket路径和共享张量句柄问题。

## Decisions

用户最新批注覆盖早期讨论。权威文件为三份9-8规格，当前无密度72×1、bf16、全局72。显存不足时可按硬件调整batch与累积但全局72不变；密度仍建议24×3。AdamW lr=1e-4，warmup0；每800个优化器更新沿原val/loss路径完整验证，Plateau相对阈值1%、patience5、factor0.2，第3次实际下降停止，最多40000更新。best按原始val/loss最低值，保留last及全部定期检查点。

六模型为中心T0、中心T1、包络T0分别配RA／RB。中心训练每次随机C5，验证／测试C5固定；种子2023／3407／10831。保留原loss、置信度头和self-ranking，不适配tuned ranker。官方纯RNA test 9v7o/0仍在全部测试视图中，正常尝试原蛋白路径并忠实记录失败阶段，不伪造蛋白。完整validation只汇总ALL；测试三视图共用候选池。每实例正式50候选、100步，所有失败保留分母。官方3种口袋协议和训练模型相应协议合计13个模型／协议组合，均需validation和test；训练期间每800步不生成候选。

服务器只使用已获准371591 GPU；纯CPU任务每个8核、最多12个并行，总96核。原AdaLigand资产只读，派生根为 `/storage/penghongen/Adaligand_Build/Ori_Data/pocketxmol`，项目产物根为 `/storage/penghongen/PocketXMol`，项目venv位于其runtime/venv。服务器删除和其它任务修改仍须用户明确许可；本作业内获准的run_cmd／try_lock／kill_lock接管操作可继续，必须保留after_lock，不得scancel或操作其它进程。

## Open Questions

当前没有阻止训练的待用户决定项。后续如需新GPU，必须取得用户新增授权。用户尚未选择密度正式实验，不能提前运行。

Git仍在实现分支codex/pxm-receptor-baselines，共同基点0412824adbdd4e4f229b572d9a6bfde75d3fdbc9，训练代码对应136c621。Learn/CUMULATIVE尚未推进。工作区docking/preparation.py有此前发现的未提交Docstring／空行修改，非本轮代理修改，已保留且未提交；同步release包含这些说明文字。tmp/是本轮临时产物，不纳入版本。服务器旧.git并非实际实现版本，以release追溯。终局须按文档、理解顺序代码、最后全部测试建立学习历史，端点等价后才能快进Learn/CUMULATIVE，并保证它是唯一最新提交；不得混入用户未提交改动。

## Next Actions

1. 保持当前训练运行。稳定时按用户要求安静等待60或90分钟，使用重复Start-Sleep -Seconds 300；单次工具等待不超过60秒，保持能处理新输入。不要创建heartbeat，不要因为稳定运行而结束goal。
2. 醒来检查371591、当前训练进度、OOM／非有限值／批量裁减、W&B和检查点。只读取本次启动之后的累计日志；当前输出根为 `/storage/penghongen/PocketXMol/training/B-C-T0-RA/`。训练完成时核对完整检查点中的停止原因、步数、调度状态、best和val/loss，不依据文件存在猜完成。
3. 顺序完成其余五个模型和正式采样。每次开始前增加逐实验日志、保存极短正式命令与真实release／launch；训练后的采样配置必须填写实际best路径。CPU评价可在获准96核总限内与下一模型GPU训练并行。
4. 汇总完整验证和测试报告，保留逐实例结果与核酸占比分层分析，交给用户选择核酸分支及密度后续。完成实现／学习端点等价核验后再推进Learn/CUMULATIVE。全部目标完成前不能标goal complete。

## Files To Reopen

- [首个训练记录](../../../日志/第一类实验（不加密度信息）/1-B-C-T0-RA.md)：正式参数、命令、运行地址和状态。
- [共同准备记录](../../../日志/实现与共同数据准备.md)：数据来源、所有准备作业、失败留证和测试命令。
- [总日志](../../../日志/第一类实验（不加密度信息）/总日志&分析/总日志.md)、[计划映射](../../../日志/计划执行映射.md)及其链接的三份最新规格。
- `/home/penghongen/Feedback/Pocket_Plus/allocations/371591/`：当前run_cmd、out、err、after_lock；try_lock在其父目录。该控制位置来自旧allocation，不能因Pocket_Plus名称而误操作其它作业。
- `/storage/penghongen/PocketXMol/control/371591/train_B-C-T0-RA_start.json`：本次out／err起始字节360594／54949；同目录train_B-C-T0-RA_run_cmd.sh保存动态命令。当前after_lock保留，训练期间try_lock不存在，正常结束后控制器重建try_lock。
