# Handoff: 官方测试改用374480_1实际377793

Date: 2026-09-11

## Current State

本记录更新2026-09-11-parallel-f5-f6-official.md中的官方资源，适用于当前任务01a08f70-63cc-72e3-9d7a-4a04421931f2。F-5为378587、F-6为378588，最近17:48检查仍PENDING/Priority，不重新提交。官方当前唯一可接管目标是 **374480_1，实际JobId377793，gnode05，单张A100、8核**，已于节点时间17:37:24消费try_lock请求执行；官方模型已加载并开始正常推理，C0前7个实例均成功完成50个候选。

用户最初指定的374480_0实际377521，已由用户亲自scancel；用户曾误写改用374480_2，随后立即明确纠正为374480_1并要求不要动_2。374480_2实际JobId374480、gnode07，仍在CryoAtom2预测11jb；本对话只读查看过，未改它任何锁、命令或进程，后续不得操作它。继承历史中的371591也不属于本对话控制范围。

## Completed

- 377521的sacct记录为CANCELLED by 1351，结束17:27:37，batch退出15；用户解释怀疑gnode08异常。旧任务实际PocketXMol release为PocketXMol_db3c806a5acc，launch为official_377521_test_job377521_20260911T162940。官方模型已严格加载，但C0/C5/E无任何result.json，仅run.json及C0/11jb/0空poses.sdf；CPU评价未开始。全部保留。
- 新增configs/docking/sample-official-377793-test.yml，只改变旧独立配置的output_root及wandb.name；新的root是 `/storage/penghongen/PocketXMol/sampling/official-377793/test/`，W&B评价名称official_377793_test。官方权重、protein输入、T0、C0/C5/E、test、batch50、每实例50候选100步与三视图均不变。
- 主代理完成新旧make_config比较和编号/资源/路径窄核，bash -n验证新动态命令；没有新增Python逻辑，不重做第三轮全面审查。安全同步成功。所有本对话新增/修改继续unstaged，未git add/commit；当前基准提交b612032是另一对话的已有提交，本对话未修改它。
- 已核对377793原预测成功完成、try_lock和after_lock均在；节点/proc仅见控制器PID68145，没有活跃Python预测。备份原run_cmd后写入新命令，消费try_lock；after_lock保留，未用kill_lock/scancel。新命令在PocketXMol冻结release中顺序运行sample与evaluate，后者使用同allocation现有8核。
- 已核实实际冻结副本 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_dceb9298cebe/PocketXMol` 和启动证据 `/home/penghongen/Feedback/PocketXMol/launches/377793/official_377793_test_job377793_20260911T173936`。模型严格加载完成于17:40:16（UTC+8），17:48已完成C0的11jb实例0至6，每实例50/50成功；实例2至6约35秒一个实例，C5/E尚未开始。

## Decisions

本对话仅完成F-5、F-6和官方测试的goal；用户接受重复计算，但独立目录和W&B记录避免混写。其他工作区修改简单保留，不处理。所有本对话文件永远unstaged，禁止git add/commit，不触碰他人暂存；不建立新worktree，不推进Learn/CUMULATIVE。

H100任务保持单卡32核after_hold、72×1/bf16/原训练规则。每模型正常结束后固定自己的实际best，再执行预定test及CPU评价；无完整验证集采样，无密度任务。资源迁移只是执行地点变化，不增加实验。

排队及稳定训练/推理按用户最新要求静默60或90分钟，使用多次Start-Sleep -Seconds 300；不设heartbeat，不因等待结束goal。H100首次被发现RUNNING时通知用户，必要时通知原对话。

## Next Actions

1. 检查377793新启动，沿用实际原控制目录 `/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/test_0_chain06/运行日志与统计/slurm/allocations/377793/`。try_lock在父目录、after_lock在本目录。启动元数据 `/storage/penghongen/PocketXMol/control/377793/official_377793_test_start.json` 的out/err起点 **17645/161**；不要再按377521的17624读取。
2. 新命令备份为同control根run_cmd_377793_before_official_20260911.sh，新命令留证official_377793_test_run_cmd.sh。正式命令为 `bash 训练与运行/sh/sample_docking.sh official-377793-test`，接着 `bash 训练与运行/sh/evaluate_docking.sh official-377793-test`；推理正常退出后自动评价。原控制器每次仍先冻结AdaLigand来源，可能需几分钟，再由动态命令创建PocketXMol release/launch。
3. 推理已确认正常，按用户要求静默等待60或90分钟后查看进度。保留失败样本及分母，不额外生成候选。C0/C5/E各446个实例，ALL/CAP10/HF10_TO5为446/272/227，共九组官方汇总；sample正常退出后evaluate已自动串联，不要重复提交。
4. H100如获批，记录378587/F-5和378588/F-6各自release/launch/W&B id；训练输出training/F-5、training/F-6，测试预定sampling/F-5、sampling/F-6，后续配置待实际best后创建。
5. 三任务完整收口后更新独立日志与重要handoff，仍保持所有改动unstaged。之前的60分钟本地等待和旧节点只读进程查询均已完成；当前没有需要继续收取的旧SSH查询。

## Files To Reopen

- [377793当前官方记录](../../../日志/第一类实验（不加密度信息）/official-377793-测试与评价.md)、[377521取消记录](../../../日志/第一类实验（不加密度信息）/official-377521-测试与评价.md)。
- [F-5记录](../../../日志/第一类实验（不加密度信息）/F-5-B-C-T1-RB.md)、[F-6记录](../../../日志/第一类实验（不加密度信息）/F-6-B-E-T0-RB.md)。
- configs/docking/sample-official-377793-test.yml、F-5.yml、F-6.yml；三份9-8契约和训练与运行README作为共同依据。
