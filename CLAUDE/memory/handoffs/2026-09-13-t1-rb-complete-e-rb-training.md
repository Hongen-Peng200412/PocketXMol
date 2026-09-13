# Handoff: T1-RB报告完成、E-RA评价与E-RB训练

Date: 2026-09-13

## 当前目标与状态

继续当前goal，完成六个无密度模型的训练、规定完整测试、CPU评价与报告。不跟踪其他任务或官方模型评价，不增加实验。正确T0-RA、T1-RA、T1-RB三项已有完整报告；密度实验和最终核酸分支选择仍留给用户。

- 371591、gnode09、A800／16CPU：E-RA的有效重训及完整E候选生成已经完成，当前主进程37521和8个工作进程执行E测试CPU评价。评价报告完成后，运行第4个B-C-T0-RB。
- 378693、gnode10、A800／16CPU：T1-RB的训练、C0／C5测试和完整报告已经完成。第6个B-E-T0-RB已于master时间05:06:47接入，从官方权重重新训练，主进程57306、15个worker、W&B 423nfmpm。
- 两项after_lock保留，运行中try_lock不存在。控制目录均为 `/home/penghongen/Feedback/Pocket_Plus/allocations/<job>/`，try_lock在父目录；PocketXMol release／launch不是锁控制根。不使用scancel，不删除旧产物或after_lock。

## 已完成的T1-RB

完整记录见[日志06](../../../日志/第一类实验（不加密度信息）/5-B-C-T1-RB.md)。训练22400步按第三次学习率下降停止，原C5 val/loss最低为17600步、2.8354082107543945，训练W&B wmgkgurr。C0／C5各446实例、22300候选生成及评价全部完成，没有候选错误或RMSD按原子编号匹配的回退。ALL Top-1为58.97%／39.46%，Top-5为75.34%／50.22%，oracle为87.67%／60.54%。评价W&B [6wvylxgi](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/6wvylxgi)已在线上传。

正式结果根为 `/storage/penghongen/PocketXMol/sampling/B-C-T1-RB/test/`，本地完整整理副本 `tmp/pxm-20260913/B-C-T1-RB-test-results.json`。报告包括三个视图、两种加权、Spearman／AUC、7个含核酸实例及纯RNA9v7o/0、与T1-RA的逐实例比较。25个数值表格记录与JSON逐一核对通过。评价有591条既有RDKit allene提示，无Traceback；完整候选与评价完成证据位于 `/storage/penghongen/PocketXMol/control/378693/` 下sample_B-C-T1-RB_test_complete_20260913.json与evaluate_B-C-T1-RB_test_complete_20260913.json。

此前23:07提出的同卡并发资源可选问题未收到答复，采样已按原授权完整结束，没有调整资源或操作其他进程。有关耗时观测已保留日志06；不继续追踪其他任务，本次E-RB按既有378693授权执行。

## E-RA的CPU评价

有效训练根 `/storage/penghongen/PocketXMol/training/B-E-T0-RA-nonemptyE/`，W&B lukfzmf3，26400步停止，best21600、原E val/loss=1.6292482614517212。46个空E训练身份跳过、验证0个，旧异常E目录和so6e0mvy保留且无效。空E修复及全部验收见[日志07](../../../日志/第一类实验（不加密度信息）/3-B-E-T0-RA.md)。

E的446实例、22300候选全部生成成功，冻结身份和预算核对通过；逐实例耗时合计12606.271秒，推理合计12533.717秒。采样完成证据为 `/storage/penghongen/PocketXMol/control/371591/sample_B-E-T0-RA_test_complete_20260913.json`。实际采样／评价release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_1ce7d5426c18/PocketXMol`，包含c9cc4a1修复和aac0b89的E测试配置。

当前正式命令：

```bash
bash 训练与运行/sh/evaluate_docking.sh B-E-T0-RA-test
```

master时间05:02:17接入，控制器第20次执行，主进程37521及工作进程37595至37602，CUDA_VISIBLE_DEVICES为空。launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/evaluate_B-E-T0-RA_test_job371591_20260913T045916`；启动记录 `/storage/penghongen/PocketXMol/control/371591/evaluate_B-E-T0-RA_test_start.json`，out／err起点113848096／282173。输出根 `/storage/penghongen/PocketXMol/sampling/B-E-T0-RA-nonemptyE/test/`。评价完成后核对逐候选／逐实例／汇总和W&B，按日志06同样范围完成报告，再在371591启动第4个B-C-T0-RB。

## 新启动的E-RB

本次没有新增或修改生产函数、科学参数或资产，使用已通过两轮审查和真实RA／RB输入验收的 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_43742cdf8166/PocketXMol`。源码为ec06dbd加c9cc4a1，配置 `configs/docking/B-E-T0-RB.yml` 与E-RA仅核酸分支及W&B名称不同。登记提交d86a452，完整记录见[日志08](../../../日志/第一类实验（不加密度信息）/6-B-E-T0-RB.md)。

正式命令：

```bash
bash 训练与运行/sh/train_docking.sh B-E-T0-RB
```

独立根 `/storage/penghongen/PocketXMol/training/B-E-T0-RB/` 接入前确认不存在。没有--resume，日志确认从规定官方pocketxmol.ckpt加载；优化器、调度和W&B均独立。72×1、bf16、15 workers，每800次优化器更新原val/loss；Plateau阈值1%、patience5、factor0.2、第三次实际下降停，上限40000；best按原val/loss最低值选择。

master 05:06:47请求，控制器第4次执行；launch为 `/home/penghongen/Feedback/PocketXMol/launches/378693/train_B-E-T0-RB_job378693_20260913T050539`。启动记录 `/storage/penghongen/PocketXMol/control/378693/train_B-E-T0-RB_start.json`，out／err起点25869442／59362，保存新旧动态命令。配置SHA256为d4e03bf8a21553c2af786f3563a689e68b812ba5f58573368dd83aa8c556ef4b。

主进程57306，15个worker为57471至57485，均属于378693；W&B [423nfmpm](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/423nfmpm)，本地wandb/run-20260913_050609-423nfmpm。最初更新损失有限，空E身份6j40/273、6j3z/22已记录跳过。后续完整训练、原E验证、实际best的50候选／100步E测试以及同卡8进程CPU评价仍须完成。

## 持续执行

两张卡各自按“一个模型训练→实际best完整规定测试→同卡CPU评价→报告→下一模型”推进。训练后不做完整验证集采样；原训练val/loss和best选择保留。不根据测试结果重新选检查点或改变科学设置。

稳定后按要求静默等待60或90分钟，用多次Start-Sleep -Seconds 300组成；工具每次等待不超过60秒，期间不发消息、不设heartbeat、不结束goal。醒来只核对本任务两卡的进展和错误，再继续必要接续。

Git仍在codex/pxm-receptor-baselines，Learn/CUMULATIVE共同基点0412824。六模型和报告全部完成后再整理学习历史、证明端点等价并快进累计分支；不重复请求起点审批。忽略并保护外来未提交文件，只提交本任务明确文件。最新状态入口是[总日志](../../../日志/总日志.md)与[计划执行映射](../../../日志/计划执行映射.md)。
