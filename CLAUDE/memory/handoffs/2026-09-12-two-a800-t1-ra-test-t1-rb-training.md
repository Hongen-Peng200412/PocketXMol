# Handoff: 两张A800分别推进RA与RB实验

Date: 2026-09-12

## Current State

goal继续active，且用户已正式更新goal：本任务只负责六个无密度模型，共10个测试协议组合、30个视图汇总，不负责官方模型测评，不跟踪其他任务。用户2026-09-12新增明确授权：378693负责第5个B-C-T1-RB及第6个B-E-T0-RB，371591继续当前第2个B-C-T1-RA及第3个B-E-T0-RA、第4个B-C-T0-RB；每模型均包含训练、正式测试、评价和记录。两张卡独立推进，每张卡内部按上述步骤串行。评价使用相应A800作业已分配的CPU，不再另排纯CPU作业；当前评价配置8个进程，各卡实际16核CPU。

正确第1个B-C-T0-RA已完成重训及全部测试报告。371591正在生成T1-RA的C5 test，最近核对C0为446个实例／22300候选全部生成成功，C5为69个实例／3450候选，尚无候选错误、Traceback或OOM。378693已从pre_hold接入第5个T1-RB正式训练，官方参数加载、bf16、CUDA与W&B初始化成功，约20.7M参数全部可训练；启动验收已到80次更新，约1.04步／秒、lr=1e-4，无Traceback／OOM／reduce_batch／NaN。

不关注其它任务，不监控任务重叠；他人的工作区文件保留、不提交。训练后完整validation采样及评价已取消，旧产物保留历史；训练中的原val/loss调度与best保留。没有新增实验、修改共同资产或进入密度正式训练。

## 371591：T1-RA测试

- 节点gnode09，单张A800、16CPU，采样主进程57606。
- 正式命令：`bash 训练与运行/sh/sample_docking.sh B-C-T1-RA-test`。
- 实际release `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_55258ac14af2/PocketXMol`，源码ec06dbd加69a9507的新采样配置；后来a0e9589仅更新该YAML评价资源注释，运行副本保持原样。
- launch `/home/penghongen/Feedback/PocketXMol/launches/371591/sample_B-C-T1-RA_test_job371591_20260912T065551`，控制器第15次执行。
- 输出 `/storage/penghongen/PocketXMol/sampling/B-C-T1-RA/test/`，T1、RA、C0／C5、batch50、每实例50候选100步；两协议各446个实例，共用冻结种子和C5偏移。
- best `/storage/penghongen/PocketXMol/training/B-C-T1-RA/checkpoints/step=20800.ckpt`，C5 val/loss=2.831010341644287。训练22400步第三次下降正常停止，W&B nzkna4ow；28个定期检查点及last全保留，报告在training_summary_20260912.json。
- 启动记录 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T1-RA_test_start.json`，out／err起点83039931／82141。

推理全部完成并核对候选后，在同一371591、同一release接入正式命令 `bash 训练与运行/sh/evaluate_docking.sh B-C-T1-RA-test`。此脚本关闭CUDA、使用8个CPU评价进程；保存独立launch、命令和W&B记录，after_lock保留。旧handoff中的“另提交8核CPU作业”已由本次用户指示取代。完整报告写完后在371591从官方参数启动B-E-T0-RA，再按同样顺序完成B-C-T0-RB。

## 378693：T1-RB训练

- 节点gnode10，单张A800、16CPU，训练主进程50944；接入前只有等待pre_lock的控制器43832，无计算任务，因此没有使用kill_lock。
- 正式命令：`bash 训练与运行/sh/train_docking.sh B-C-T1-RB`。
- 实际release `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_fa0d957b2d3f/PocketXMol`，源码ec06dbd、原B-C-T1-RB.yml，无生产代码或科学配置改动，不吸收他人工作区文件。
- launch `/home/penghongen/Feedback/PocketXMol/launches/378693/train_B-C-T1-RB_job378693_20260912T111926`，控制器第1次执行。master请求时间11:20:47，保留各节点时钟原值。
- 训练根 `/storage/penghongen/PocketXMol/training/B-C-T1-RB/`，启动前确认不存在，从官方pocketxmol.ckpt开始，无--resume。
- W&B [wmgkgurr](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/wmgkgurr)，名称B-C-T1-RB；本地wandb/run-20260912_112017-wmgkgurr。
- 控制记录 `/storage/penghongen/PocketXMol/control/378693/train_B-C-T1-RB_start.json`，同目录保留新旧动态命令；out／err起点136／0。

训练保持72×1、bf16、15workers、AdamW1e-4、warmup0、每800更新验证；Plateau阈值1%、patience5、factor0.2，第三次实际下降立即停或最多40000步。T1训练动态一份完整delta选袋／定原点，高斯后加s*delta；val/loss固定C5，推理首步纯先验、后续-s*mean(Z)。训练正常完成后在Slurm内核实last的停止状态及实际raw best，再建立C0／C5 test配置。测试、同作业CPU评价及报告完成后，才在378693从官方参数训练第6个B-E-T0-RB并完成E test和评价。

## 资源与等待

两作业实际旧控制目录分别为 `/home/penghongen/Feedback/Pocket_Plus/allocations/371591/`、`.../378693/`。after_lock在各作业子目录，pre／try锁在父目录；保持after_lock，不scancel或释放资源。378693的pre_lock已按授权移除；两个任务运行时try和kill均不存在。执行前核对各自job、节点、进程和锁，不能串用。

两任务稳定后静默等待60／90分钟，分别12／18次Start-Sleep -Seconds 300，工具单次等待不超过60秒，不heartbeat、不结束goal。用户新消息及时处理。保存exec session id；工具host重置后轮询原session，不因此重启服务器任务。此前371591等待的本地timer 44689所属functions cell已终止，shell定时器本身会在11:32左右自然结束，不对应任何服务器任务。

## Git与记录

当前实现分支codex/pxm-receptor-baselines；a0e9589已提交两卡分工、A800内CPU评价及第5个模型登记。此前69a9507固定T1-RA best测试配置，adf9933记录实际启动。Learn/CUMULATIVE仍0412824，整个goal结束时才按双线流程完成学习历史与端点等价核验。仅提交明确属于本任务的文件。

模型实现及T0修复的两遍自查、两轮全面独立审查均已完成。新T1-RA测试配置经过主代理两遍自查和t1_test_config_review窄核；本次新增资源仅使用已有训练配置，不扩大代码审查。

## Files To Reopen

- [总日志](../../../日志/第一类实验（不加密度信息）/总日志&分析/总日志.md)、[计划执行映射](../../../日志/计划执行映射.md)。
- [T1-RA训练](../../../日志/第一类实验（不加密度信息）/2-B-C-T1-RA.md)、[T1-RA测试评价](../../../日志/第一类实验（不加密度信息）/2-B-C-T1-RA.md)、[T1-RB训练](../../../日志/第一类实验（不加密度信息）/5-B-C-T1-RB.md)。
- [已完成T0报告](../../../日志/第一类实验（不加密度信息）/1-B-C-T0-RA.md)及三份9-8契约、训练与运行/README.md。

正式与验收命令继续分开记录。源manifest实例字段为candidate_id，result.json为occurrence_id；下载大JSON时使用Console.Out的StringWriter捕获，避免SSH helper输出直接刷入聊天。
