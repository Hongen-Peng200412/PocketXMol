# Handoff：D1中心训练完成并进入best测试

日期：2026-09-15

## 当前状态

四卡正式目标继续，不能关闭goal。D1中心在371591上正常完成训练并开始best的C0/C5测试；D2中心和包络分别在378693、379403_1并行训练。379402上的无密度中心C0/C5采样完成，CPU评价正在计算C5，尚未启动无密度包络。

## 已完成

D1中心训练30400次更新，第三次实际学习率下降时停止。完整last.ckpt确认decline_count3、stop_reason=plateau、best为step=20800.ckpt、val/loss=1.8294975757598877，最终优化器lr=8e-7；W&B 5qmpsxrs online finished。原训练源码、配置、全部检查点及产物保留。日志中的一处best小数抄写错误已按原始stdout和完整checkpoint纠正，不影响实际best选择。

新配置`configs/docking/sample-D1-C-T0-RA-test.yml`经过主代理自查、解析检查及一轮独立核查，密度输入从保存的训练配置派生，没有科学覆盖；C0/C5共用T0入口，50候选100步batch50、原FP32。提交`0d4ab9589ed186fe1f1cbf0764e748dcd5026472`，测试release为`/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_7efe284e12f4/PocketXMol`。他人在`docking/dataset.py`的未提交修改未提交、未带入release。

371591于15:52:18消费父目录try_lock，15:52:37正式开始。launch为`/home/penghongen/Feedback/PocketXMol/launches/371591/formal_D1-C-T0-RA_test_0d4ab95`，旧run_cmd已备份，after_lock保留。控制脚本和日志位置见D1实验日志。目标GPU有约946 MiB非本作业进程占用，保留原状，没有操作其他任务。

无密度中心采样于15:02:26正常退出0，随后同卡8进程评价。C0 ALL的446实例self-ranking top1成功率为0.6076233183856502，阈值RMSD＜2 Å；C5和完整W&B上传尚未完成，不能宣称全流程结束。八个CPU进程持续计算。

## 决策与下一步

按最新四卡映射逐模型完成训练、best测试与CPU评价：371591的D1中心之后为D1包络、D4中心、D4包络；378693的D2中心之后直接D3中心、D3包络，不等待D2包络；379403_1仅完成D2包络后保留等待；379402完成无密度中心全流程后才启动包络新训练，之后官方冻结对照。W&B online，不进行完整验证集采样，不扩展实验或I/O搜索。

D1正式命令为`bash 训练与运行/sh/sample_docking.sh D1-C-T0-RA-test`，采样退出0后自动`bash 训练与运行/sh/evaluate_docking.sh D1-C-T0-RA-test`。新测试根为`/storage/penghongen/PocketXMol/sampling/D1-C-T0-RA/test`，控制stdout／stderr在`/storage/penghongen/tmp/pxm_formal_execution_20260914/runs/D1-C-T0-RA-test/`。稳定后继续60分钟分段等待，取得新进度更新逐实验与所属总日志；不为常规检查点新建handoff。

## 重新打开的文件

- `日志/第二类实验（密度分支的训练）/1-D1.md`、`3-D2.md`及总日志。
- `日志/RA-T0第二次严格消融/1-B-C-T0-RA-SMILES.md`及总日志。
- `configs/docking/sample-D1-C-T0-RA-test.yml`。
- 本地`tmp/formal-execution-20260914/start-d1-test-371591.sh`和只读脚本`read_completed_d1.sh`。

实现分支为codex/formal-density-strict-ablation，累计学习基点仍d3c1ed5；全部正式目标完成后再进行双线Git等价收口。
