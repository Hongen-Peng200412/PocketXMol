# Handoff: 官方测试完成，F-5/F-6继续排队

Date: 2026-09-12

## Current State

当前任务01a08f70-63cc-72e3-9d7a-4a04421931f2的官方对照部分已完成。官方冻结模型C0/C5/E推理、8进程CPU评价及ALL/CAP10/HF10_TO5九组汇总均完成，W&B在线上传成功。F-5为378587、F-6为378588，2026-09-12 08:09检查仍PENDING/Priority，尚无allocation目录，不能宣称训练已启动或重新提交。

本任务goal继续保持进行中：还要等待两项H100训练，各自正常停止后选实际best并完成规定测试与CPU评价。排队或稳定执行期间使用12或18次Start-Sleep -Seconds 300静默等待60或90分钟，不设heartbeat，不结束任务；H100首次被发现开始运行时通知用户。

## Completed

- 官方资源为374480_1，实际JobId377793，gnode05、单A100、8核。已完成的两条正式命令为 `bash 训练与运行/sh/sample_docking.sh official-377793-test` 和 `bash 训练与运行/sh/evaluate_docking.sh official-377793-test`。控制器out确认第二次执行成功，try_lock重新出现，after_lock保留。
- 实际冻结源码 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_dceb9298cebe/PocketXMol`，启动证据 `/home/penghongen/Feedback/PocketXMol/launches/377793/official_377793_test_job377793_20260911T173936`，源码基准b612032加本对话未提交配置。没有修改生产Python逻辑。
- 结果根 `/storage/penghongen/PocketXMol/sampling/official-377793/test/`，最终summary.json文件时间2026-09-12 07:47:34（UTC+8）；W&B为pencounkdual-111/PocketXmol_raw/runs/x8diqywv，状态online_completed。逐协议446份result.json、assessment.json和完整候选指标均在；每协议445个实例成功、22250个成功候选，9qkz/0的50个候选因原FeaturizePocket的unknown element in pocket预处理错误失败，失败始终保留在三个视图分母。
- 已只读核对三个协议的446个实例身份、候选编号0至49、50候选/100步/batch50、计数一致性及九个视图的冻结实例集合。直接从逐实例结果重算实例等权和PDB等权的Top-1/Top-5/50候选最佳成功率，与summary.json在1e-12内一致；每协议22250个有效RMSD，没有按原子编号匹配的备用计算，也没有新增候选评价错误。
- ALL按实例等权、RMSD严格小于2 Å的Top-1成功率，C0/C5/E分别44.84%/31.17%/63.90%。全部九组结果、PDB等权结果、相关系数/AUC有效数量、显存耗时和失败口径见独立官方日志。各协议446份assessment均有口袋核酸占比，其中7个大于0；后续六模型比较可直接读取这些逐实例结果。

## Decisions

所有本对话文件永远unstaged，禁止git add/commit，不处理他人修改或暂存，不推进Learn/CUMULATIVE。继续使用现有工作区，不另建worktree。配置与运行产物按F-5/F-6/official-377793独立命名，用户接受与另一个任务可能重复计算。

保留377793的after_lock和全部产物。其实际控制目录仍是 `/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/test_0_chain06/运行日志与统计/slurm/allocations/377793/`，try_lock在父目录；当前无需再次执行任何官方命令。374480_2不是授权目标，不操作；旧374480_0实际377521已被用户取消，旧产物保留。继承历史中的371591也不属于本任务控制范围。

## Next Actions

1. 继续等待378587/F-5和378588/F-6，单张H100、32核、after_hold。正式命令分别为 `bash 训练与运行/sh/train_docking.sh F-5` 和 `bash 训练与运行/sh/train_docking.sh F-6`；配置已存在并提交到队列，不重复申请。开始运行后核实各自实际release/launch/W&B id。
2. F-5对应B-C-T1-RB，F-6对应B-E-T0-RB；独立训练根分别为 `/storage/penghongen/PocketXMol/training/F-5/` 与F-6。batch72、累积1、bf16、原val/loss及已冻结调度停止规则不变。保留after_lock和全部检查点，不改其他任务。
3. 每项训练正常停止后，读取自己的实际最低val/loss检查点，创建明确的sample-F-5-test.yml或sample-F-6-test.yml。F-5只测试C0/C5并保留T1采样机制，F-6只测试E/T0；每实例50候选、100步、batch50，三个重叠测试视图共用候选。对应测试根为sampling/F-5、sampling/F-6，不用另一任务的产物。
4. 各自完成CPU评价和独立结果记录后再判断整个goal完成。训练期间保留val/loss验证；训练结束后不安排完整验证集采样，不启动密度实验。

## Files To Reopen

- [官方完整结果与执行记录](../../../日志/第一类实验（不加密度信息）/7-official.md)。
- [F-5记录](../../../日志/第一类实验（不加密度信息）/5-B-C-T1-RB.md)、[F-6记录](../../../日志/第一类实验（不加密度信息）/6-B-E-T0-RB.md)。
- configs/docking/F-5.yml、F-6.yml、sample-official-377793-test.yml。
- 2026-09-11-parallel-f5-f6-official.md保留提交背景；2026-09-11-official-moved-to-377793.md保留资源迁移背景。本文件替代它们的当前运行状态，不覆盖旧handoff。
