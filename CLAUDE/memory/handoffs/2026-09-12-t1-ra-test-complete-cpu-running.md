# Handoff: T1-RA测试候选完成，原A800内CPU评价中

Date: 2026-09-12

## Current State

用户已更新goal，当前只负责六个无密度模型及其10个测试协议组合、30个视图汇总；不负责官方测评，不跟踪其他任务。第1个正确B-C-T0-RA完整报告已完成。第2个B-C-T1-RA的C0／C5候选现已全部生成，在371591自带CPU内评价。第5个B-C-T1-RB同时在378693正常训练。

两张A800各自串行完成训练→实际best完整测试→CPU评价、报告→下一模型。371591位于gnode09，随后负责第3个B-E-T0-RA、第4个B-C-T0-RB；378693位于gnode10，随后负责第6个B-E-T0-RB。两卡各16CPU，评价使用既有8个进程，不再另申请纯CPU作业。保持各自after_lock，不scancel、不释放资源、不改无关任务。

## T1-RA候选完成

使用 `/storage/penghongen/PocketXMol/training/B-C-T1-RA/checkpoints/step=20800.ckpt`，训练在22400步第三次下降正常停止，C5最低val/loss=2.831010341644287，训练W&B nzkna4ow。

正式采样命令为 `bash 训练与运行/sh/sample_docking.sh B-C-T1-RA-test`，控制器第15次执行正常成功，进程57606退出，try恢复。输出 `/storage/penghongen/PocketXMol/sampling/B-C-T1-RA/test/`：C0、C5各446实例、22300候选，生成失败0；batch50、50候选100步、冻结种子与C5偏移保持。完整身份、object_key、种子、偏移、视图、候选编号及姿态／置信度文件核对通过。

完成记录 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T1-RA_test_complete_20260912.json`。实例耗时合计C0为13400.881秒、C5为13128.801秒；推理合计13282.236／13037.641秒；各44600次批量forward，峰值张量显存3209630720／3205350400字节。候选生成成功不代表姿态正确，评价尚待完成。

## 371591当前CPU评价

正式命令：

```bash
bash 训练与运行/sh/evaluate_docking.sh B-C-T1-RA-test
```

继续使用采样相同release `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_55258ac14af2/PocketXMol`，源码ec06dbd加69a9507明确best的测试配置；未吸收共享工作区文件。后来本地a0e9589仅改该配置的CPU资源注释，冻结运行副本不变。

评价主进程33502，cgroup为job_371591，CUDA_VISIBLE_DEVICES为空；主作业仍分配16CPU，程序使用8个评价进程。master请求2026-09-12 14:30:34，实际launch `/home/penghongen/Feedback/PocketXMol/launches/371591/evaluate_B-C-T1-RA_test_job371591_20260912T142745`，控制器第16次执行。节点约慢3分钟，保留各自时钟。

启动元数据 `/storage/penghongen/PocketXMol/control/371591/evaluate_B-C-T1-RA_test_start.json`，同目录保存evaluate_B-C-T1-RA_test_run_cmd.sh；本次out／err读取起点83208404／82237。评价直接写已有test根，W&B名称B-C-T1-RA_test，实际评价run id待最终上传记录确认。

## 378693当前训练

B-C-T1-RB继续使用ec06dbd的fa0d957b2d3f release，从官方参数开始，72×1、bf16、15workers；训练主进程50944，W&B [wmgkgurr](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/wmgkgurr)。正式命令 `bash 训练与运行/sh/train_docking.sh B-C-T1-RB`。训练根 `/storage/penghongen/PocketXMol/training/B-C-T1-RB/`。

launch `/home/penghongen/Feedback/PocketXMol/launches/378693/train_B-C-T1-RB_job378693_20260912T111926`，控制器第1次执行；控制记录位于control/378693/train_B-C-T1-RB_start.json，本次out／err起点136／0。14:27附近检查为11749更新、lr=2e-5，已记录8000步最低C5 val/loss=2.90767；这只是中途状态，不据此冻结best。没有Traceback、OOM、reduce_batch或NaN。

## Next Actions

1. 两项任务稳定后静默60／90分钟，使用12／18次Start-Sleep -Seconds 300分段；工具每次等待≤60秒，不heartbeat、不结束goal。保存当前exec session id，工具重置时继续同一session，不因此重启服务器任务。之前被新消息打断的旧本地timer 44689已于11:32正常结束并回收。
2. 371591评价完成后检查控制器、进程、summary／occurrences／逐实例assessment及W&B。生成完整三视图、实例与PDB等权、Spearman／AUC、核酸比例和失败报告，写入日志05并更新总记录；随后才在371591启动第3个B-E-T0-RA。
3. 378693训练按既定第三次下降／40000上限结束后，在Slurm内读取last和实际best核对停止状态；建立本模型C0／C5 test配置、生成完整候选、同作业CPU评价及报告，再运行第6个B-E-T0-RB。
4. 保护共同资产、旧错误T0／validation产物和他人未提交文件。当前代码两遍自查、两轮全面独立审查已闭合，运行和报告不再扩大代码审查。必要新配置只做相应核对。正式命令与验收命令分开记录。
5. 全部六模型报告完成后收口双线Git；实现分支codex/pxm-receptor-baselines，Learn/CUMULATIVE共同基点0412824。实现端点与学习端点等价后再推进累计学习分支；不push、不纳入他人文件。密度正式实验仍等待用户选择。

## Files To Reopen

- [T1-RA测试与评价日志05](../../../日志/第一类实验（不加密度信息）/2-B-C-T1-RA.md)、[T1-RB训练日志06](../../../日志/第一类实验（不加密度信息）/5-B-C-T1-RB.md)。
- [已完成T0报告](../../../日志/第一类实验（不加密度信息）/1-B-C-T0-RA.md)、[总日志](../../../日志/第一类实验（不加密度信息）/总日志&分析/总日志.md)、[计划执行映射](../../../日志/计划执行映射.md)。
- 三份9-8契约及训练与运行/README.md已按新goal收窄到本任务六模型范围。

实际旧控制目录均在 `/home/penghongen/Feedback/Pocket_Plus/allocations/<job>/`，after_lock在子目录，try／pre在父目录。不要把本项目launch目录当成锁目录。manifest实例字段是candidate_id，result.json为occurrence_id；大型JSON应在本地用Console.Out的StringWriter捕获再落盘，不直接刷入聊天。T0完整汇总副本仍在tmp/pxm-20260912/B-C-T0-RA-test-results.json。
