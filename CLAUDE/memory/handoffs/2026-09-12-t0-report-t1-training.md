# Handoff: 正确T0完整报告已完成，T1-RA正式训练中

Date: 2026-09-12

## Current State

goal仍active。正确B-C-T0-RA已完成纠偏、官方参数独立重训、C0／C5完整test、CPU评价、W&B上传及完整报告。当前在371591、gnode09、单张A800上正式训练B-C-T1-RA，从官方参数开始，72×1、bf16；主进程16005，W&B nzkna4ow，启动检查已到62次更新，lr=1e-4、约1.01步／秒，无OOM或Traceback。

用户要求每模型训练→直接test→CPU评价和记录，然后下一模型。训练后完整validation采样及评价已取消，旧validation候选仅保留历史；训练中的原val/loss、best和调度不变。用户明确要求专注本任务goal，不跟踪其它任务或核对其重叠；他人的工作区文件保留、不提交。当前所有正式运行沿用明确的不可变release，不吸收他人工作区文件。

## Current Training Command And Sources

正式命令：`bash 训练与运行/sh/train_docking.sh B-C-T1-RA`。

- 源码ec06dbd，实际release `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_fa0d957b2d3f/PocketXMol`，配置configs/docking/B-C-T1-RA.yml。
- 实际PocketXMol launch `/home/penghongen/Feedback/PocketXMol/launches/371591/train_B-C-T1-RA_job371591_20260912T003912`；旧Pocket_Plus控制器第14次执行。
- 训练目录 `/storage/penghongen/PocketXMol/training/B-C-T1-RA/`，run [nzkna4ow](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/nzkna4ow)，名称B-C-T1-RA。
- 启动元数据 `/storage/penghongen/PocketXMol/control/371591/train_B-C-T1-RA_start.json`，命令副本train_B-C-T1-RA_run_cmd.sh；out／err读取起点57342372／73878。
- 实际控制器目录 `/home/penghongen/Feedback/Pocket_Plus/allocations/371591/`，try_lock在父目录。after_lock保留，正常训练时try_lock和kill_lock不存在。任务GPU上限仍仅371591；CPU任务每个8核、纯CPU并发上限96核。

T1科学机制已冻结：训练每次一份delta=r*u，r均匀0–5 Å、方向球面均匀；完整g+delta选袋和定原点，原高斯后加s*delta。监督验证读冻结C5并用同一公式；推理首步纯先验，后续保留-s*mean(Z)。不新增开关、不改官方loss／置信度。每800优化器步验证，AdamW1e-4、Plateau相对1%、patience5、factor0.2，第三次下降立即停或最多40000步。当前还没有完成状态或已核实best。

## T0 Completed Results

正确训练W&B ceqrh2ve，目录training/B-C-T0-RA-C0；31200步第三次下降正常停止，best=21600、原C0 val/loss=1.7970343828201294。旧错误T0训练及旧validation产物保留。

测试输出 `/storage/penghongen/PocketXMol/sampling/B-C-T0-RA-C0/test/`，batch50，每实例每协议50候选100步。C0／C5各446个实例、22300个候选全部生成及评价成功，RMSD和评分配对均完整，无候选错误或RMSD按编号匹配回退。完成核对记录control/371591/sample_B-C-T0-RA_test_complete_20260911.json。

CPU Job 378916，cnode02、8核，COMPLETED、exit0、耗时1:54:34、最大常驻内存约7.11 GiB；按原提交的无after_hold设置自动释放。评价使用相同ec06dbd release，launch目录launches/378916/evaluate_docking_job378916_20260911T215541_a1。W&B [v4jilbdr](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/v4jilbdr)已online_completed。

按实例等权，ALL Top-1／Top-5／oracle成功率：C0为60.09%／72.42%／82.29%，C5为43.05%／53.81%／61.66%；成功定义RMSD<2 Å。完整三视图、PDB等权、Spearman／AUC、7个含核酸实例及纯RNA结果都已写入日志04。RDKit日志624条allene-style立体化学忽略提示已注明限制，不因警告改分母。报告已提交63cc0f0并打开请求排入本任务面板。

## Next Actions

1. T1训练稳定时静默等60／90分钟，每次通过12／18个Start-Sleep -Seconds 300完成；每次工具等待≤60秒，不heartbeat、不结束goal。保存local exec session id；functions.wait若因host重置失效，轮询原write_stdin session，不重启服务器任务。
2. 只按当前T1 out／err起点检查进度、错误和checkpoint。正常完成后在Slurm资源内加载last和best，核实实际停止原因、下降次数及原val/loss最低的best；不能凭日志最后一步或W&B最后lr猜测停止状态。
3. 根据实际best建立本模型的C0／C5 test配置，batch50、50候选100步，然后在同一371591推理。完整test结束后提交8核CPU评价，形成报告才训练下一个模型。不得恢复已取消的完整validation采样；不按测试结果改其余冻结模型配置。
4. 继续本任务六模型及官方对照、总报告和双线Git，密度正式实验等用户选择。共同资产不重建。当前实现分支codex/pxm-receptor-baselines，Learn/CUMULATIVE基点0412824；最终端点等价后再推进学习分支。已有两遍自查、两轮独立审查及必要CPU／GPU验收均已闭合，单纯运行和报告不再扩大代码审查范围。

## Files To Reopen

- [当前T1-RA训练](../../../日志/第一类实验（不加密度信息）/03-B-C-T1-RA.md)。
- [完整T0测试报告](../../../日志/第一类实验（不加密度信息）/04-B-C-T0-RA推理与评价.md)、[T0纠偏与训练](../../../日志/第一类实验（不加密度信息）/02-T0中心契约修复与重训.md)。
- 三份9-8契约、日志/总日志.md、日志/计划执行映射.md。

本地临时结果副本tmp/pxm-20260912/B-C-T0-RA-test-results.json包含服务器summary及全部occurrences，已解析并与逐候选记录核对。统一SSH helper通过Console.Out输出，PowerShell赋值不能直接捕获；需要下载JSON时在当前进程用StringWriter临时替换Console.Out并在finally恢复，再解析保存，避免把大JSON刷进聊天。不要修改共享helper或输出凭据。
