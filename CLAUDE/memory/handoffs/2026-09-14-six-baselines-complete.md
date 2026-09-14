# Handoff: 六个无密度模型的训练、测试与评价完成

Date: 2026-09-14

## 当前状态

六模型均已从规定官方初始参数完成有效训练、原val/loss选择best、完整测试及CPU评价，全部报告与总体比较见[总日志](../../../日志/第一类实验（不加密度信息）/总日志&分析/总日志.md)。中心模型各做C0和C5，包络模型做E，共10个模型／协议组合、30个重叠视图汇总；每协议446实例、每实例50候选和100步，全部223000个候选生成并评价成功。没有运行训练后完整验证集采样的新任务。

ALL Top-1成功率：T0-RA的C0／C5为60.09%／43.05%，T1-RA为59.19%／42.15%，T0-RB为59.64%／40.36%，T1-RB为58.97%／39.46%；E-RA与E-RB分别为65.47%和64.35%。Top-5、oracle、PDB等权、CAP10、HF10_TO5和全部含核酸实例结果见各模型日志前部。不同指标并未给出一致的核酸分支优劣，仍由用户选择后续分支和密度预算。

## 最后完成的模型与证据

[第4模型](../../../日志/第一类实验（不加密度信息）/4-B-C-T0-RB.md)有效训练根为 `/storage/penghongen/PocketXMol/training/B-C-T0-RB/`，W&B hthglbuy；31200次更新后第三次学习率下降停止，全部39份定期检查点及last保留，实际best21600，C0 val/loss=1.8065478801727295。

测试与评价根为 `/storage/penghongen/PocketXMol/sampling/B-C-T0-RB/test/`，测试release为07ea8b8a2406，配置提交933a49c。C0／C5各446实例、22300候选完成；CPU评价W&B为[d1osmk6j](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/d1osmk6j)，online_completed且无上传错误。2026-09-14 00:51 master确认主进程44073退出、控制器第23次执行成功。

完整逐候选审计为 `/storage/penghongen/PocketXMol/control/371591/evaluate_B-C-T0-RB_test_complete_20260914.json`：44600个候选的RMSD及self-ranking有限，排序、Top-1／Top-5／oracle和三视图分母一致，候选错误、RMSD按原子编号回退及Traceback均为0。604条RDKit allene提示已在报告说明，不作为604个独立失败。正式采样和评价命令均已执行，恢复上下文时不要重复启动。

第4模型和总体报告已完成两遍主检查、两轮同范围独立核查；数值表格、配对变化、AUC缺失、来源、分母、历史保留及引用全部通过。

## 保留规则与后续边界

371591负责的第1至4模型、378693负责的第5至6模型均完成。01:07 master只读确认两项A800资源仍在，各16核，after_lock与try_lock存在、kill_lock不存在；全部有效、错误及中断产物保留。旧错误T0的B-C-T0-RA和NaN的B-E-T0-RA只作历史，有效来源分别为B-C-T0-RA-C0与B-E-T0-RA-nonemptyE，未混入新总表。

使用当前七份实验日志结构：一模型一日志，前部有效结果、路径和W&B，后部执行证据及失败／中断历史；不恢复旧文件。官方及F-5／F-6仅沿用用户整理的既有记录，不增加本任务的跟踪或接管职责。科学部分没有待续的训练、采样或评价，不新增实验，密度阶段等待用户决定。

## Git收口与复查入口

保留实现分支codex/pxm-receptor-baselines，从共同基点0412824adbdd4e4f229b572d9a6bfde75d3fdbc9建立Learn/pxm-receptor-baselines，端点通过同范围验收后才快进Learn/CUMULATIVE。实际端点哈希、完整树与张量等价、CPU／GPU检查、分支和工作区最终状态见[双线Git核对记录](../../../tmp/pxm-20260914/dual_track_final.json)，以其status=complete表示收口完成；记录位于提交历史之外，不存在提交自引哈希问题，不推送远端。

验收范围、临时产物和正式命令的区分见[共同准备日志末节](../../../日志/实现与共同数据准备.md)；计划覆盖及密度未完成边界见[计划执行映射](../../../日志/计划执行映射.md)。如恢复时Git记录尚未complete，只接续已列出的收口验收，不重跑六模型科学任务。
