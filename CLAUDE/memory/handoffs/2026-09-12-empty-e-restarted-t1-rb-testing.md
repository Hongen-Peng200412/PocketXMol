# Handoff: 空E修复重训与T1-RB完整测试

Date: 2026-09-12

## 当前状态

本任务继续既有goal：只完成六个无密度模型的训练、完整测试、CPU评价和报告，不跟踪其他任务或官方模型测评。当前没有待用户决定的问题。已完成正确B-C-T0-RA和B-C-T1-RA的训练与完整报告；两张获准A800各自按“一个模型训练→best规定测试→CPU评价→报告→下一模型”串行推进。

- 371591、gnode09、A800／16CPU：B-E-T0-RA已按批准规则修复，从官方权重在 `/storage/penghongen/PocketXMol/training/B-E-T0-RA-nonemptyE/` 独立重训。主进程39120，W&B lukfzmf3，启动检查161步时所有损失有限；随后该卡完成E测试评价，再运行第4个B-C-T0-RB。
- 378693、gnode10、A800／16CPU：B-C-T1-RB训练已在22400步第三次学习率下降后正常停止，best为17600步，C5 val/loss=2.8354082107543945，W&B wmgkgurr。当前采样主进程26403正在使用该best完成C0/C5测试，之后用同卡CPU评价并写报告，再运行第6个B-E-T0-RB。
- 两项after_lock都保留；运行时try_lock应不存在。实际控制目录均为 `/home/penghongen/Feedback/Pocket_Plus/allocations/<job>/`，try_lock在父目录。不要把PocketXMol launch根当作这两个旧作业的控制目录，不使用scancel。

## 已完成与证据

用户批准：仅E训练及val/loss流跳过空口袋并记录身份；冻结文件不变，正式测试记录输入构造失败且保留实例和候选分母。完整只读审计65290训练、781验证，共46训练空E、0验证空E；E有效训练65244。全部46个身份列于日志07；完整JSON在 `/storage/penghongen/tmp/pocketxmol_empty_envelope_20260912/empty_envelope_audit.json`，本地副本为 `tmp/pxm-20260912/empty_envelope_audit.json`。

最小生产修改仅为docking/dataset.py：RA/RB空E在求原点前抛EmptyEnvelopePocketError，仅__iter__的train／validation捕获并跳过；正式采样直接索引，沿现有preprocess失败及评价分母路径处理。没有改变阈值、标准残基范围、中心模型、模型／loss／噪声器／采样器／评价器，也没有重建资产。

两遍主代理自查、两轮三类独立审查及提出问题的窄核全部通过。CPU回归22 passed／9 warnings；真实train的6j3z/11空E和5ftl/0非空混合验收，RA/RB各8次72×1、bf16优化器更新、原val/loss，组批、坐标、梯度及最终参数均有限。GPU报告为 `/storage/penghongen/tmp/pocketxmol_empty_envelope_fix_20260912/gpu_gate/results.json`，CPU及GPU日志位于同一临时根。均为验收，不是正式模型。

首次E-RA异常运行 `/storage/penghongen/PocketXMol/training/B-E-T0-RA/`、W&B so6e0mvy全部保留，约17步起NaN、179步附近通过kill_lock停止，尚无检查点。不得覆盖、续训或作为正确E结果。

## 正式运行入口与来源

E重训release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_43742cdf8166/PocketXMol`，基线ec06dbd加修复提交c9cc4a1。正式命令：

```bash
bash 训练与运行/sh/train_docking.sh B-E-T0-RA --logdir /storage/penghongen/PocketXMol/training/B-E-T0-RA-nonemptyE
```

2026-09-12 master 17:24:37启动，控制器第18次执行。启动记录为 `/storage/penghongen/PocketXMol/control/371591/train_B-E-T0-RA_nonemptyE_start.json`；out／err起点83348997／139462。实际launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/train_B-E-T0-RA_nonemptyE_job371591_20260912T172127`，节点时钟约慢3分钟。新W&B本地目录为训练根下wandb/run-20260912_172138-lukfzmf3。原B-E-T0-RA.yml参数保持：72×1、bf16、15 workers，每800优化器更新验证、Plateau相对阈值1%、patience5、第三次实际下降停，best取原val/loss最小值。

T1-RB测试release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_c59a88a03678/PocketXMol`，源码仍为ec06dbd，仅加入3001cf9中的明确测试配置和记录。正式采样命令：

```bash
bash 训练与运行/sh/sample_docking.sh B-C-T1-RB-test
```

每协议446实例、每实例50候选／100步、batch50，C0与冻结C5都保留T1后续中心相关重新加噪。输出为 `/storage/penghongen/PocketXMol/sampling/B-C-T1-RB/test/`。已核对run.json及首批完整候选，未见错误。2026-09-12 master 17:26:58启动，控制器第2次执行；启动记录为 `/storage/penghongen/PocketXMol/control/378693/sample_B-C-T1-RB_test_start.json`，out／err起点25698979／8263；launch为 `/home/penghongen/Feedback/PocketXMol/launches/378693/sample_B-C-T1-RB_test_job378693_20260912T172535`，节点时钟约慢1分20秒。

T1-RB训练完成后曾因W&B GraphQL超时等待后台关闭，最终自行收尾、控制器执行成功，无需kill_lock。检查点摘要为 `/storage/penghongen/PocketXMol/training/B-C-T1-RB/training_summary_20260912.json`，28个定期检查点与last保留。不要把网络收尾等待误记为训练失败。

只有全部C0/C5推理完成后，才在同一378693及同一测试release执行CPU评价，使用8 workers：

```bash
bash 训练与运行/sh/evaluate_docking.sh B-C-T1-RB-test
```

正式训练后的完整验证集采样和评价始终取消；原训练val/loss及best选择保留。不要新增实验或另申请CPU。

## 下一步

两项任务正常运行后静默等待60或90分钟，以多次Start-Sleep -Seconds 300组成；工具每次等待不超过60秒，期间不发消息、不设置heartbeat、不结束goal。醒来检查本任务两张卡的进展及实际错误，需要接续时按各自次序执行。

E训练完成后在该Slurm作业内核对实际best和停止状态，再建立独立E测试配置；第6个E-RB必须使用已修复Dataset的源码，不再使用未修复的fa0d957b2d3f训练E。T1-RB完成全测试后保留分母完成CPU评价、ALL／CAP10／HF10_TO5与核酸比例分析，报告格式可参照已完成日志04和05。

Git仍为持续任务实现分支codex/pxm-receptor-baselines，共同基点及Learn/CUMULATIVE为0412824。最终六模型报告完成后再重建学习线、证明端点等价并快进累计分支；不要在持续工作中重新请求起点审批。忽略其他任务的未提交文件，不提交、不修改、不跟踪其进展。

## 重新打开的文件

- [总日志](../../../日志/总日志.md)、[计划执行映射](../../../日志/计划执行映射.md)。
- [E-RA记录](../../../日志/第一类实验（不加密度信息）/3-B-E-T0-RA.md)：全部空E身份、审查验收、正式命令和异常／有效运行来源。
- [T1-RB记录](../../../日志/第一类实验（不加密度信息）/5-B-C-T1-RB.md)：实际best、测试配置、运行来源和后续报告。
- [T0-RA完整报告](../../../日志/第一类实验（不加密度信息）/1-B-C-T0-RA.md)、[T1-RA完整报告](../../../日志/第一类实验（不加密度信息）/2-B-C-T1-RA.md)。
