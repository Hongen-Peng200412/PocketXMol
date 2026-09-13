# Handoff: 正确T0测试采样完成，CPU评价运行中

Date: 2026-09-11

## Current State

本记录承接2026-09-11-direct-test-only.md。正确T0的C0／C5测试采样已正常完成；8核CPU评价Job 378916正在cnode02运行，尚未确认完成。goal仍active：依次完成每个模型的训练→直接测试→CPU评价及记录，再进入下一模型，最终完成六个无密度模型和官方对照报告及双线Git；密度正式实验等待用户选择。训练后完整验证集采样和评价已取消，保留原训练val/loss及best选择。

## Completed

正确T0从官方参数独立重训，31200步正常停止，best为21600步、C0 val/loss=1.7970343828201294；训练目录 `/storage/penghongen/PocketXMol/training/B-C-T0-RA-C0/`，W&B ceqrh2ve。旧错误T0及已取消validation产物原样保留。

正式测试命令 `bash 训练与运行/sh/sample_docking.sh B-C-T0-RA-test` 使用ec06dbd、RA、T0、batch50、每实例50候选100步。C0／C5各446个实例、22300个候选，全部生成成功；各44600次批量forward。逐实例耗时合计分别13341.127184022218和13354.369349695742秒，峰值张量显存分别3209630720和3205350400字节。已核对全部实例集合、完成标记、预算、冻结种子／偏移／视图及产物文件存在。完成记录 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T0-RA_test_complete_20260911.json`。

371591控制器第13次执行成功，gnode09采样进程42558已退出，try_lock恢复、after_lock保留、kill_lock不存在。输出根 `/storage/penghongen/PocketXMol/sampling/B-C-T0-RA-C0/test/`。

## Current Command And Sources

CPU评价正式命令：`bash 训练与运行/submit_task.sh --sh evaluate_docking.sh --resource cpu --cpus 8 -- B-C-T0-RA-test`。

提交时工作目录为采样使用的同一release：`/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_fa0d957b2d3f/PocketXMol`。Job 378916实际复用该release，launch为 `/home/penghongen/Feedback/PocketXMol/launches/378916/evaluate_docking_job378916_20260911T215541_a1`，out／err／动态命令在 `/home/penghongen/Feedback/PocketXMol/allocations/378916/`。使用冻结配置sample-B-C-T0-RA-test.yml，不同步其他代理的工作区文件。W&B评价run在本地科学汇总落盘后才创建并上传，尚无已确认run id。

## Next Actions

1. 核对378916实际状态及test/C0、C5下各446份assessment.json。完成后读取test/summary.json、两协议occurrences.json及candidate_metrics.json中的错误统计，核对W&B状态和完整分母，完成T0测试报告。重点包括三视图、self-ranking／oracle、核酸占比、逐实例相关性和Pose AUC；不能把生成成功当作姿态成功。
2. 本模型评价和记录收口后，按日志03正式命令启动B-C-T1-RA；仍只用371591，保留after_lock。后续继续按本任务goal推进。用户明确撤回了额外跟踪另一任务及核对重叠的安排，不读取或跟踪其他任务状态；其他代理的工作区修改原样保留、不提交。
3. 稳定运行或排队时静默等60／90分钟，用12／18次Start-Sleep -Seconds 300；每次工具等待不超过60秒，不heartbeat、不结束goal。记录本地sleep session id，以便工具host重置后轮询原会话，不能因观察超时重启服务器任务。
4. 当前实现分支codex/pxm-receptor-baselines，共同Learn/CUMULATIVE基点0412824。完整实现及T0纠偏的两遍自查、两轮独立审查和必要CPU／GPU验收均已完成；本次资源字段与记录更新不扩展审查范围。最终全部工作完成后再收口学习分支并验证端点等价，保护他人的未提交文件。

## Files To Reopen

- [T0测试及CPU评价记录](../../../日志/第一类实验（不加密度信息）/1-B-C-T0-RA.md)。
- [T0修复与训练](../../../日志/第一类实验（不加密度信息）/1-B-C-T0-RA.md)、[下一模型T1-RA](../../../日志/第一类实验（不加密度信息）/2-B-C-T1-RA.md)。
- 三份9-8契约、日志/总日志.md、日志/计划执行映射.md。
