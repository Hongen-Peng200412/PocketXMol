# Handoff: T1-RA报告完成，包络RA因空口袋停止并等待处理决定

Date: 2026-09-12

## Current State

当前goal只负责六个无密度模型的训练、规定测试与评价报告，合计10个测试协议组合、30个视图汇总。不负责官方测评，不跟踪其他任务。正确B-C-T0-RA及B-C-T1-RA的完整报告均已完成，所有旧产物保留。

两张获准A800各自串行完成训练→实际best完整测试→本作业内CPU评价与报告→下一模型。371591、gnode09的第3个B-E-T0-RA首次训练因空E口袋产生NaN已停止，正在做8核只读输入诊断，处理口径待用户回复，之后仍负责第4个B-C-T0-RB。378693、gnode10当前运行第5个B-C-T1-RB，随后第6个B-E-T0-RB。两卡均16CPU，评价沿用8个进程，不另申请CPU，不释放after_lock。

## Completed

B-C-T1-RA的20800步best已完成C0／C5各446实例、22300候选的推理和评价，全部候选有效、无资产或候选错误、RMSD原子编号回退0。ALL实例等权Top-1为59.19%／42.15%，Top-5为73.32%／51.57%，oracle为87.00%／60.54%。三视图、PDB等权、Spearman／AUC、7个核酸实例及与正确T0的同协议比较均已写入日志05，报告表格已逐项对照正式JSON核查。

正式根 `/storage/penghongen/PocketXMol/sampling/B-C-T1-RA/test/`，W&B [n9ddvnid](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/n9ddvnid)状态online_completed。评价控制器第16次执行成功，旧进程33502退出，after_lock保留；完成记录为control/371591/evaluate_B-C-T1-RA_test_complete_20260912.json。584条RDKit allene提示、0 Traceback，未改变评价规则。C0／C5累计实例评价时间22771.545／22744.151秒，不能当作并行墙钟时长。

报告整理副本位于tmp/pxm-20260912/B-C-T1-RA-test-results.json；T0对应副本和日志04保留。正式短命令、release、launch和完整证据均在各自日志中，报告及包络RA登记提交2a1bed8。

## 371591 B-E-T0-RA异常运行与诊断

正式命令：

```bash
bash 训练与运行/sh/train_docking.sh B-E-T0-RA
```

源码与配置仍为ec06dbd，运行release `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_fa0d957b2d3f/PocketXMol`。不从共享工作区吸收他人文件，不改现有B-E-T0-RA.yml。官方fresh初始化、72×1、bf16、15workers，E训练及原val/loss，center_translation=false。每800优化器更新验证，Plateau阈值1%、patience5、factor0.2、第三次实际下降停止，最多40000步，按原始最低val/loss选best。

master请求2026-09-12 16:43:45，控制器第17次执行，主进程9532，cgroup确认job_371591。launch `/home/penghongen/Feedback/PocketXMol/launches/371591/train_B-E-T0-RA_job371591_20260912T164054`，gnode09比master约慢3分钟。启动记录 `/storage/penghongen/PocketXMol/control/371591/train_B-E-T0-RA_start.json`，同目录保存train_B-E-T0-RA_run_cmd.sh，本次out／err起点83210385／132728。

训练根 `/storage/penghongen/PocketXMol/training/B-E-T0-RA/`，启动前确认不存在。W&B [so6e0mvy](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/so6e0mvy)，本地wandb/run-20260912_164135-so6e0mvy。官方权重加载、CUDA和bf16正常，但17步附近出现持续NaN；stderr明确dataset.py:149空E口袋求均值，原点及后续配体坐标变为NaN。

已按371591原kill_lock授权核对主进程与cgroup后停止，控制器第17次执行退出137，最后显示179步，进程9532消失，try_lock恢复、after_lock保留、kill_lock已处理；没有scancel。尚未到800步验证，checkpoints目录为空。停止证据 `/storage/penghongen/PocketXMol/control/371591/train_B-E-T0-RA_nan_stop_20260912.json`。此次异常运行全部产物保留，不能resume或作为有效模型。

只读独立代理envelope_nan_diagnosis已完成：准备阶段仅检查PDB整体标准受体非空，没有检查每个E口袋；RA没有错误删除核酸，E原点均值产生NaN即可解释污染。现有契约未授权E改用配体中心、扩大10 Å或纳入UNK，也未定义空E训练样本跳过规则。

本地诊断脚本tmp/pxm-20260912/audit_empty_envelope.py已写入 `/storage/penghongen/tmp/pocketxmol_empty_envelope_20260912/`，经srun --jobid=371591 --overlap --cpus-per-task=8和既定venv运行，CUDA隐藏。只扫描65290个train及781个validation，8092个PDB，直接用生产select_pocket；不读取test、不改任何清单。输出audit.log和empty_envelope_audit.json，当前需要收取完成计数。执行会话31501；保留session继续等待，不能重复exclusive写入脚本目录。

已经通过异步问题请用户决定：仅E训练及val/loss流跳过空口袋并记录身份，冻结文件不改；测试空E记录输入构造失败并保留分母；在独立目录从官方参数重训E，保留中心模型结果。用户尚未回复，不能先实施依赖此口径的代码或重训。

## 378693 当前B-C-T1-RB训练

同一ec06dbd／fa0d957b2d3f release，正式命令 `bash 训练与运行/sh/train_docking.sh B-C-T1-RB`。主进程50944，W&B [wmgkgurr](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/wmgkgurr)，训练根 `/storage/penghongen/PocketXMol/training/B-C-T1-RB/`。launch `/home/penghongen/Feedback/PocketXMol/launches/378693/train_B-C-T1-RB_job378693_20260912T111926`，第1次执行，out／err起点136／0。

最近核对已进入20222次更新，lr=4e-6，已实际下降两次；17600步原C5 val/loss=2.83541暂为最低，20000步为2.99427。此处只是中途观测，不能据此冻结best或假定22400步停止。训练结束后必须在Slurm内读取last及callback实际best。

## Next Actions

1. 收取空E训练／验证统计，补充完整身份和数量；等待异步问题答复后才实施相应最小修复。不能把B-E-T0-RA记作正常训练。诊断扫描与378693训练均正常计算时可以按约定静默等待，但问题继续待答，收到用户消息后及时处理中断。
2. 378693训练正常结束后，核对第三次下降或40000上限、last、实际best、W&B及日志。已用过的只读检查脚本在 `/storage/penghongen/tmp/pocketxmol_checkpoint_20260912/inspect_t1_ra.py`；改为RB的独立临时副本，必须替换模型根、job、run id，不能把RA的1130参数键断言用于RB。Torch加载在srun --jobid=378693 --overlap的CPU步骤内执行，普通SSH仅做轻量读写。
3. 为实际best建立sample-B-C-T1-RB-test.yml，固定RB／T1、C0／C5 test、batch50、50候选100步、8个CPU评价进程。两遍配置自查并按必要范围独立核查；使用原已审查release加这一配置建立隔离新release，严禁加入他人未提交文件。完整测试后同作业内评价、报告，再运行第6个B-E-T0-RB。
4. 371591的E问题明确并完成必要自查与独立核查后，从官方权重在独立目录重训，保留异常so6e0mvy。训练完成后核对实际best，只做E完整测试和评价，报告后再运行第4个B-C-T0-RB。保留原训练val/loss，不安排训练后完整验证集采样。两卡任务稳定后仍按60／90分钟、每次300秒分段静默等待，不heartbeat、不结束goal。
5. 所有六模型报告完成后收口双线Git。实现分支codex/pxm-receptor-baselines，共同Learn/CUMULATIVE基点0412824；实现端点与学习端点等价后再快进累计学习分支，不push。代码链两遍自查、两轮全面独立审查已关闭，运行和报告不扩大代码审查。

## Files To Reopen

- [T1-RA完整报告](../../../日志/第一类实验（不加密度信息）/05-B-C-T1-RA推理与评价.md)、[B-E-T0-RA日志](../../../日志/第一类实验（不加密度信息）/07-B-E-T0-RA.md)、[B-C-T1-RB日志](../../../日志/第一类实验（不加密度信息）/06-B-C-T1-RB.md)。
- [总日志](../../../日志/总日志.md)、[计划执行映射](../../../日志/计划执行映射.md)、三份9-8契约。

实际锁目录均为 `/home/penghongen/Feedback/Pocket_Plus/allocations/<job>/`，after和kill在子目录，try和pre在父目录。正式科学release／launch使用PocketXMol目录。现有YAML有CRLF，字节比较必须两边都保留原始字节或两边都归一化。保护他人未提交文件，不读取其状态、不commit。
