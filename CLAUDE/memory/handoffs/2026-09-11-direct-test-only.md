# Handoff: 取消完整验证集采样并转入直接测试

Date: 2026-09-11

## Current State

本记录取代2026-09-11-t0-complete-and-inference.md中的执行顺序和validation任务安排。用户明确纠正：保留训练期间原val/loss、调度和best选择，训练结束后直接进行完整测试，不再安排完整验证集采样或评价。每个模型测试推理、CPU评价及记录收口后，再训练下一个模型。goal仍active，最终完成六个无密度模型及官方测试报告、双线Git；密度正式实验等待用户选择。

当前同一371591、gnode09、单张A800正在运行正确T0的完整test：`bash 训练与运行/sh/sample_docking.sh B-C-T0-RA-test`。源码ec06dbd，配置sample-B-C-T0-RA-test.yml，RA、T0、C0／C5、batch50，每实例每协议50候选、100步。模型为正确训练的21600步best。首三个11jb实例各完成50候选、100次批量forward，约33–35秒，没有生成失败。

## Completed

- 正确T0训练已正常完成：`/storage/penghongen/PocketXMol/training/B-C-T0-RA-C0/`，W&B ceqrh2ve；31200步第三次下降后停止，last保存decline_count=3、stop_reason=plateau、lr=8e-7。最低原C0 val/loss为1.7970343828201294，对应 `checkpoints/step=21600.ckpt`。39个定期检查点和last保留，不使用旧错误T0目录或W&B 9wkyn4qn。
- 按用户要求核实完整命令和cgroup后，gnode09时间14:16:43创建371591实际kill_lock，终止validation采样进程组19347。控制器第12次执行退出137并恢复try_lock，after_lock保留，旧进程消失，kill_lock由控制器处理。未scancel、释放资源或删除产物。停止记录 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T0-RA_validation_stop_20260911.json`。
- validation已有C0 781个实例／39050候选、C5 436个实例／21800候选完成，全部保留；中断实例未完成文件也不删除。不补齐、不评价这些validation产物。此前没有提交任何完整validation CPU评价任务。
- T0及官方正式测试配置batch25改50，其他解析值完全不变，提交ec06dbd并安全同步。用户允许50作为普遍选项，更大有效批量能提速时可用100；当前入口每实例仅50候选，设置100实际仍是同一批50，因此未虚报提速或增加候选预算。
- 三份9-8契约、运行说明、总日志、映射及日志02／03／04已同步取消训练后完整验证集工作。历史validation配置仅标为停用留证，保留其原值。docking/sampling.py仅更新两处批量Docstring，执行逻辑不变。

## Decisions

T0／T1科学契约及六模型、官方C0／C5／E测试矩阵保持。T0训练／监督验证为C0，T1训练动态C5／监督验证冻结C5，包络E；训练800步原val/loss及Plateau相对1%、patience5、第三次下降或40000步停止规则不变。其它模型训练配置已冻结，不根据先看到的测试结果改变。

共同数据不重建：train／validation／calibration／test为65290／781／361／446；test ALL／CAP10／HF10_TO5为446／272／227，三视图共用候选。坏图／语言问题按实例排除，纯核酸在有效清单中保留。此次只变运行范围及推理批量，不改冻结C5、种子、候选预算、排序或评价核。

仅371591 GPU获准；保留after_lock，不申请其它GPU、不改其它任务或删除服务器资产。CPU任务每个8核、纯CPU并发≤96核。服务器模型加载和计算经Slurm，SSH仅控制及轻量只读；临时文件放/storage/penghongen/tmp。

## Next Actions

1. 当前test运行稳定后按用户要求静默等60或90分钟，用多次Start-Sleep -Seconds 300；每次工具等待≤60秒，不heartbeat、不结束goal。曾发生本机挂起／代码工具代次变化：若functions.wait报告stale，改为在新functions.exec中轮询原write_stdin session，不重启服务器任务；本地时间超过等待截止即恢复检查。
2. 读取test的新增日志和逐实例result.json，分别统计C0／C5各446个实例。实际控制器仍在 `/home/penghongen/Feedback/Pocket_Plus/allocations/371591/`，try_lock在父目录。启动记录 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T0-RA_test_start.json` 保存out／err起点57173771／73688；不要再按validation或旧训练起点读状态。
3. 本次实际release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_fa0d957b2d3f/PocketXMol`，launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/sample_B-C-T0-RA_test_job371591_20260911T142022`。master请求时间14:23:03，gnode09时钟约慢3分钟；耗时使用程序计时。输出根 `/storage/penghongen/PocketXMol/sampling/B-C-T0-RA-C0/test/`，run.json冻结best和batch50。
4. 测试两个协议全部采样结束后，提交 `bash 训练与运行/submit_task.sh --sh evaluate_docking.sh --resource cpu --cpus 8 -- B-C-T0-RA-test`。记录实际CPU作业、release／launch、W&B评价run、逐实例及三视图汇总、错误分母和核酸占比／排序分析。完成T0测试报告后才启动B-C-T1-RA；日志03只预登记，未提交。
5. 其余五模型逐个完成训练→直接测试→CPU评价→记录；官方只做完整test的C0／C5／E，不运行sample-official-validation.yml。全部测试报告完成后收口双线Git。当前实现分支codex/pxm-receptor-baselines，共同学习基点0412824；不提前推进Learn/CUMULATIVE，不改写已有实现历史。
6. 完整实现及T0纠偏均已两遍自查、两轮三类独立审查并关闭窄核问题；本次实际best配置也已独立窄核。单个资源字段及文档纠正不重新扩大到第三轮全面代码审查。用户注释改动经AST核实后可顺手提交，保护其它未授权改动。

## Files To Reopen

- [T0测试与取消验证记录](../../../日志/第一类实验（不加密度信息）/1-B-C-T0-RA.md)：当前正式命令、取消证据、预算、版本与运行来源。
- [T0修复和训练结果](../../../日志/第一类实验（不加密度信息）/1-B-C-T0-RA.md)、[T1预登记](../../../日志/第一类实验（不加密度信息）/2-B-C-T1-RA.md)、[总日志](../../../日志/总日志.md)、[计划映射](../../../日志/计划执行映射.md)。
- 三份9-8契约、训练与运行/README.md、configs/docking/sample-B-C-T0-RA-test.yml及sample-official-test.yml。
