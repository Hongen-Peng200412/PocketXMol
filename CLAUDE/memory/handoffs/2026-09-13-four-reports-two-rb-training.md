# Handoff: 四份完整报告完成、最后两个RB模型训练中

Date: 2026-09-13

## 当前目标与剩余工作

继续当前goal，完成既定六个无密度模型的训练、完整测试、CPU评价和报告。当前正确T0-RA、T1-RA、E-RA、T1-RB四个模型均已有完整报告；最后两个模型C-T0-RB、E-T0-RB正在两张获准A800分别训练。不要结束goal、设置heartbeat、增加实验或跟踪其他任务及官方模型测评。最终核酸分支和密度实验仍留给用户选择。

两张卡各自按“训练完成→实际best全部规定测试→同卡CPU评价→报告”执行。训练保留原val/loss和best选择，训练后不执行完整验证集采样。中心测试C0／C5，包络测试E；每实例每协议50候选、100步、batch优先50，冻结清单、种子和C5向量保持。评价使用该A800作业自带CPU中的8个工作进程。

## 371591上的C-T0-RB

gnode09，A800／16CPU。E-RA报告完成后，于2026-09-13 master时间06:14:42启动第4个B-C-T0-RB，控制器第21次执行。主进程7586，15个数据worker，独立W&B [hthglbuy](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/hthglbuy)。日志确认加载规定官方权重，没有--resume；启动28步时损失有限，无NaN／OOM。

正式命令：

```bash
bash 训练与运行/sh/train_docking.sh B-C-T0-RB
```

训练根 `/storage/penghongen/PocketXMol/training/B-C-T0-RB/`，保存配置与检查点分别在train_config／checkpoints。W&B本地目录wandb/run-20260913_061159-hthglbuy。启动记录 `/storage/penghongen/PocketXMol/control/371591/train_B-C-T0-RB_start.json`，out／err起点113849959／311189；新旧动态命令同目录保存。

实际launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/train_B-C-T0-RB_job371591_20260913T061132`，节点时钟约慢3分钟。本次配置SHA256为19809af372ac4c127efd8dab8c3850992cbb1164211a258eaebf4a4b0d6f8279，登记提交3101cff。完整记录见[日志09](../../../日志/第一类实验（不加密度信息）/4-B-C-T0-RB.md)。

训练和val/loss均为正确C0：真实中心选袋与定原点、无新增偏移及平移；后续T0+C5测试保持给定偏移中心且仅执行原T0噪声。模型最终输出只加回原点一次。不要把T0重新改成C5训练或打开mol_as_pocket_center。

## 378693上的E-T0-RB

gnode10，A800／16CPU。第5个T1-RB报告完成后，于master 05:06:47启动第6个B-E-T0-RB，控制器第4次执行。主进程57306、15个数据worker，W&B [423nfmpm](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/423nfmpm)。master 06:10检查约3820步，原E验证正常，损失有限，没有NaN／OOM；空E训练身份按批准规则跳过。

正式命令：

```bash
bash 训练与运行/sh/train_docking.sh B-E-T0-RB
```

训练根 `/storage/penghongen/PocketXMol/training/B-E-T0-RB/`，W&B本地目录wandb/run-20260913_050609-423nfmpm。启动记录 `/storage/penghongen/PocketXMol/control/378693/train_B-E-T0-RB_start.json`，out／err起点25869442／59362。launch为 `/home/penghongen/Feedback/PocketXMol/launches/378693/train_B-E-T0-RB_job378693_20260913T050539`，节点时钟约慢1分钟。配置SHA256为d4e03bf8a21553c2af786f3563a689e68b812ba5f58573368dd83aa8c556ef4b，登记提交d86a452。完整记录见[日志08](../../../日志/第一类实验（不加密度信息）/6-B-E-T0-RB.md)。

两项训练共用已审查的 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_43742cdf8166/PocketXMol`，基线ec06dbd加c9cc4a1空E修复。本次没有新增生产代码或资产重建。均为官方初始化、72×1、bf16、15 workers，AdamW1e-4、warmup0，每800次优化器更新原val/loss；Plateau相对阈值1%、patience5、factor0.2，第三次实际下降停，上限40000步，best按原loss最低值取。不要沿用更早的1000步验证等旧参数。

## 四项已完成报告

- [日志04：正确T0-RA](../../../日志/第一类实验（不加密度信息）/1-B-C-T0-RA.md)：best21600，训练根B-C-T0-RA-C0，C0／C5 ALL Top-1为60.09%／43.05%，评价W&B v4jilbdr。
- [日志05：T1-RA](../../../日志/第一类实验（不加密度信息）/2-B-C-T1-RA.md)：best20800，C0／C5 ALL Top-1为59.19%／42.15%，评价W&B n9ddvnid。
- [日志06：T1-RB](../../../日志/第一类实验（不加密度信息）/5-B-C-T1-RB.md)：best17600，C0／C5 ALL Top-1为58.97%／39.46%，评价W&B 6wvylxgi；全部逐候选指标及25个报告数值记录已核对，591条既有RDKit allene提示，无RMSD回退。
- [日志07：E-RA](../../../日志/第一类实验（不加密度信息）/3-B-E-T0-RA.md)：独立重训根B-E-T0-RA-nonemptyE，26400步停止，best21600、原E val/loss=1.6292482614517212；E ALL Top-1／Top-5／oracle为65.47%／80.94%／90.36%，评价W&B ajpum4py。全部逐候选指标及16个报告数值记录已核对，323条allene提示，无RMSD回退。

以上每个协议均446实例、22300候选生成与评价成功，无候选错误；三个视图、PDB等权、Spearman／AUC与7个含核酸实例含纯RNA9v7o/0均已记录。E-RA完成证据 `/storage/penghongen/PocketXMol/control/371591/evaluate_B-E-T0-RA_test_complete_20260913.json`，正式结果根 `/storage/penghongen/PocketXMol/sampling/B-E-T0-RA-nonemptyE/test/`；本地完整副本 `tmp/pxm-20260913/B-E-T0-RA-test-results.json`。T1-RB本地副本为同目录B-C-T1-RB-test-results.json。不要根据这些测试结果重选检查点或新增实验。

## 边界、等待与Git

两项after_lock保留；运行中try_lock不存在。实际控制目录是 `/home/penghongen/Feedback/Pocket_Plus/allocations/<job>/`，try_lock在父目录。科学运行来源是各自PocketXMol release／launch。不用scancel、不删除after_lock或旧产物。旧错误T0-RA、异常E-RA及旧validation候选全部保留且不作为有效结果。

仅E训练和val/loss在组批前跳过空口袋；完整只读审计为46训练、0验证，E有效训练65244，冻结文件不变。正式测试遇空E时保存失败及分母；此次E-RA测试全部输入成功。空E修复的两遍自查、两轮三类审查、22项CPU回归及真实RA／RB输入GPU验收已经完成，不重复扩大审查。

此前同卡并发的可选询问未获回复，相关T1-RB采样已正常完成，没有调整资源或处理其他进程；该历史观测在日志06，不继续追踪其他任务。

稳定状态静默等待60或90分钟，用多次Start-Sleep -Seconds 300组成，工具每次等待不超过60秒；期间不发消息、不设heartbeat、不结束goal。醒来只检查这两项授权任务。不要逐次验证或每个epoch更新handoff，等检查点审计、正式测试接续或训练完成等事件再更新。

Git保持codex/pxm-receptor-baselines，Learn/CUMULATIVE共同基点0412824。六模型完整报告完成后，再重建学习历史、核验端点等价并快进累计分支；当前不重新请求起点审批。忽略并保护其他任务未提交文件，只提交本任务明确文件。状态入口为[总日志](../../../日志/第一类实验（不加密度信息）/总日志&分析/总日志.md)与[计划执行映射](../../../日志/计划执行映射.md)。
