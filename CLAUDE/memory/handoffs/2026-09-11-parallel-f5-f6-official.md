# Handoff: F-5、F-6与377521官方测试的独立执行

Date: 2026-09-11

## Current State

本记录属于任务“核查PocketXMol执行前准备（3: 独立做另外半边）”，thread 01a08f70-63cc-72e3-9d7a-4a04421931f2。它从原对话继承上下文，但当前goal仅包括六模型中的第5、6个模型，以及官方测试。原对话01a08144-4f74-7813-bf84-0b99f1b97d99继续其自己的运行；不要沿继承历史重新控制371591或启动第1–4个模型。

F-5已提交为378587，F-6为378588，均单张H100、32核、after_hold、无pre_hold；提交后两次只读查看均PENDING/Priority。官方测试已按用户授权接管374480_0实际377521，gnode08单张A100/8CPU：写入独立PocketXMol运行命令后消耗实际try_lock，after_lock保留。最后一次检查尚无新的模型启动日志；进程树证实控制器PID86152正在等待原AdaLigand的create_release.sh（PID71563），其后代在执行find/sort/sha256sum，属于实际运行前冻结副本步骤。需继续核对模型启动，不能宣称已开始采样。

## Completed

- 已只读核实模型顺序、Slurm资源、实际控制路径及原预测完成状态。377521原控制目录为 `/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/test_0_chain06/运行日志与统计/slurm/allocations/377521/`，try_lock在父目录，after_lock在本目录。其它数组成员374480_1、374480_2及原371591不属于本对话控制范围。
- 新增F-5.yml、F-6.yml，分别复制B-C-T1-RB.yml和B-E-T0-RB.yml，仅W&B名称变化。训练输出使用training/F-5、training/F-6，W&B名称F-5_B-C-T1-RB、F-6_B-E-T0-RB，独立run id。暂保持原已验收15个DataLoader worker，申请CPU为32。
- 新增sample-official-377521-test.yml，保持model_name=official、官方训练配置/权重、protein分支、T0及C0/C5/E，只改变输出根sampling/official-377521和W&B名称official_377521_test。官方每协议ALL446、CAP10272、HF10_TO5227，三视图共享候选；50候选、100步、batch50。
- 主代理完成两遍配置自查；原make_config在Windows需Python -X utf8，默认GBK的首次失败未修改项目解析器。独立代理parallel_config_review完成两轮限定审查；官方失败产物说明已修正，最新unstaged规则经过窄核，无剩余问题。未新增或修改Python执行逻辑，不做第三轮扩大审查。
- 使用项目安全同步入口完成同步，保留远端文件。H100提交记录：`/storage/penghongen/PocketXMol/control/parallel-20260911/F-5_submit.json`及F-6_submit.json，master时间16:26:07/08，记录实际命令、作业编号和未提交源码来源。
- 官方接管记录：`/storage/penghongen/PocketXMol/control/377521/official_377521_test_start.json`，gnode08时间16:23:42，out/err读取起点17624/161；同目录run_cmd_377521_before_official_20260911.sh保存原命令，official_377521_test_run_cmd.sh保存新命令。操作前只有原控制器，无活跃Python预测；未用kill_lock/scancel，未删除旧产物。

## Decisions

用户允许两个对话做重复计算，冲突由用户协调；本对话用独立目录和W&B记录避免混写，不需要独立worktree。工作区中其他人修改的文件简单保留，不处理。

用户最新明确：**本对话所有新增及修改永远保持unstaged，禁止git add和git commit，不触碰其他人的暂存内容。** 当前基准提交f856ad8，分支codex/pxm-receptor-baselines；本次配置和日志是未提交文件，不能标为该commit已有内容。实际运行来源用基准、未提交文件清单、release/launch及配置快照共同说明。本对话不推进Learn/CUMULATIVE，也不代替原对话完成Git收口；新要求优先于skill提交默认流程。

训练继续既定72×1、bf16、原loss/置信度、每800更新val/loss、相对1%/patience5/第三次下降停止或40000上限。F-5动态C5训练、冻结C5监督验证、T1公式不变；F-6包络E/T0/RB不变。每个模型正常完成后读取自己的实际best，直接测试并CPU评价；不做训练后完整验证集采样，不重建共同科学资产。

排队也按用户最终要求完全静默60/90分钟，用多次Start-Sleep -Seconds 300组成；醒来发现H100分配后再通知用户及必要的原对话，不设置heartbeat、不结束active goal。此要求取代此前每5分钟查队列的建议。控制工具单次等待不超过60秒。

## Next Actions

1. 核对377521启动：实际原控制器在每次执行run_cmd前仍会先冻结AdaLigand来源，随后新run_cmd再为PocketXMol创建实际release/launch。不要修改原控制器、清理旧副本或重复消耗锁。进程树查询23539、69145均已完成，后一次看到sha256sum正在校验tests_output中的文件，仍有推进。开始60分钟静默等待后再核实启动；这些查询没有重启任务。
2. 官方模型严格加载成功后核对test/run.json、首批候选、实际release/launch及CPU链。输出 `/storage/penghongen/PocketXMol/sampling/official-377521/test/`。新动态命令顺序运行正式sample和evaluate入口，推理正常退出后自动使用同一allocation现有8核评价，after_hold继续保留卡。
3. H100按请求获批后直接训练F-5/F-6。记录各自actual release/launch/W&B id，分别跟踪正常停止、best、OOM及恢复状态；结束后固定自己的best再建立sample-F-5-test.yml或sample-F-6-test.yml，输出分别sampling/F-5及sampling/F-6。不要使用另一对话默认模型目录。
4. 两个H100、官方A100均使用已有冻结实例与语言/对称资产。任务稳定后按上述静默等待；有问题按已授权范围解决，科学变化另问用户。
5. 三项训练/推理/CPU评价全部收口后完成本goal、更新独立日志和重要handoff；本地文件继续unstaged。

## Files To Reopen

- [F-5日志](../../../日志/第一类实验（不加密度信息）/F-5-B-C-T1-RB.md)、[F-6日志](../../../日志/第一类实验（不加密度信息）/F-6-B-E-T0-RB.md)、[官方测试日志](../../../日志/第一类实验（不加密度信息）/official-377521-测试与评价.md)：正式命令与验收分开，包含路径和授权。
- configs/docking/F-5.yml、F-6.yml、sample-official-377521-test.yml：本对话新增文件，不编辑原模型配置。
- 训练与运行/README.md及runtime/create_release.sh、create_launch.sh：沿用正式入口和留证；与服务器交互/sync_code.ps1为已用的非删除同步入口。
- 三份9-8契约：仍为共同科学依据，本对话不修改其共享内容。
