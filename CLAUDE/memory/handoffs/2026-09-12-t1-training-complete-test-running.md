# Handoff: T1-RA训练完成，best正在完整测试

Date: 2026-09-12

## Current State

goal仍active。正确中心T0-RA的训练、完整C0／C5 test、CPU评价、W&B上传及报告已完成。中心T1-RA训练现已正常结束，20800步best正在371591、gnode09、单张A800上进行完整C0／C5 test。采样主进程57606，模型严格加载成功，test/run.json确认center_translation=true、batch50、50候选、100步。首两个C0实例11jb/0、11jb/1均50／50成功、100次批量forward，约35／33秒，峰值张量显存约2.50 GB，无Traceback或CUDA OOM。

用户要求每模型训练→直接test→CPU评价与记录→下一模型。训练中的原val/loss用于调度和best；训练后完整validation采样及评价已取消，旧产物保留历史。只推进本goal，不跟踪其它任务或检查任务重叠。工作区他人文件保留、不提交。GPU仍仅使用371591，after_lock保留；CPU任务每个8核，纯CPU并发上限96核。密度正式实验等待最后的用户选择。

## Completed

T1-RA从官方参数初始化，源码ec06dbd，训练release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_fa0d957b2d3f/PocketXMol`，训练目录 `/storage/penghongen/PocketXMol/training/B-C-T1-RA/`，W&B [nzkna4ow](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/nzkna4ow)。旧控制器第14次执行正常成功；训练主进程16005已退出。

Slurm步骤371591.1读取last和best核实：global_step=last_validation_step=22400，stop_reason=plateau，decline_count=3，last优化器和调度器lr均8e-7。28个定期检查点及last全部保留。raw best为20800步、C5 val/loss=2.831010341644287，RA模型1130个model.参数键。调度器best仍2.8402891159057617，因为20800步的改善未达到1%相对阈值；22400步第三次下降符合既定规则。

只读核查脚本 `/storage/penghongen/tmp/pocketxmol_checkpoint_20260912/inspect_t1_ra.py`；报告 `/storage/penghongen/PocketXMol/training/B-C-T1-RA/training_summary_20260912.json`。验收命令单独记在日志03，不当作正式训练／推理命令。

新测试配置及完成记录提交69a9507。主代理按字段消费和注释做两遍自查，YAML解析及与T0配置的差异断言通过；独立代理t1_test_config_review只读窄核通过。生产函数未改，不重开已完成的两轮全面代码审查。

## Current Test Command And Sources

正式推理命令：

```bash
bash 训练与运行/sh/sample_docking.sh B-C-T1-RA-test
```

配置 `configs/docking/sample-B-C-T1-RA-test.yml` 指向本次保存训练配置和 `training/B-C-T1-RA/checkpoints/step=20800.ckpt`。RA、T1、C0／C5 test、batch50、50候选100步、冻结清单及种子保持。输出为 `/storage/penghongen/PocketXMol/sampling/B-C-T1-RA/test/`。

实际release `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_55258ac14af2/PocketXMol`，以ec06dbd的fa0d957b2d3f副本为基础，仅加入69a9507的新采样配置；服务器逐文件比较确认唯一差异为该新增YAML。临时装配目录 `/storage/penghongen/tmp/pocketxmol_t1_test_20260912/PocketXMol`。未复制共享工作区他人文件，也未改旧release。

实际launch `/home/penghongen/Feedback/PocketXMol/launches/371591/sample_B-C-T1-RA_test_job371591_20260912T065551`。master请求2026-09-12 06:58:52；gnode09约慢3分钟，保留各自时钟。旧控制器第15次执行。启动元数据 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T1-RA_test_start.json`，同目录保存sample_B-C-T1-RA_test_run_cmd.sh。本次out／err起点83039931／82141。

实际控制目录 `/home/penghongen/Feedback/Pocket_Plus/allocations/371591/`，after_lock_371591在此目录，try_lock_371591在其父目录。运行时try和kill不存在；不要套用PocketXMol新反馈目录去操作这个旧资源控制器。外层Pocket_Plus release承载控制器，实际科学代码固定为上面的PocketXMol release。

## Next Actions

1. 推理稳定后按用户要求静默等待60／90分钟，分别使用12／18次Start-Sleep -Seconds 300，工具每次等待不超过60秒，不heartbeat、不结束goal。保留本地exec session id；工具host重置后继续轮询同一session，不重启服务器任务。
2. 核对C0和C5各446个实例、22300候选。完整时核对冻结实例身份、种子、C5偏移、候选预算及文件；manifest的实例字段是candidate_id，result.json是occurrence_id。确认实际进程结束、控制器成功、try恢复且after保留。
3. 从同一55258ac14af2 release执行正式CPU评价命令：`bash 训练与运行/submit_task.sh --sh evaluate_docking.sh --resource cpu --cpus 8 -- B-C-T1-RA-test`。记录job和launch；W&B名称B-C-T1-RA_test。完成三视图、实例及PDB等权、置信度排序分析、核酸比例分析与逐实例失败记录后再训练下一个模型。不因测试结果修改其余冻结训练条件。
4. 完成goal剩余模型、报告与双线Git；当前实现分支codex/pxm-receptor-baselines，累计基点Learn/CUMULATIVE=0412824。最终实现端点与学习端点等价后推进累计学习分支；保护他人的未提交文件。暂不进入密度正式实验。

## Files To Reopen

- [T1训练与检查点核查](../../../日志/第一类实验（不加密度信息）/2-B-C-T1-RA.md)、[当前T1推理与评价](../../../日志/第一类实验（不加密度信息）/2-B-C-T1-RA.md)。
- [已完成T0报告](../../../日志/第一类实验（不加密度信息）/1-B-C-T0-RA.md)、日志/总日志.md、日志/计划执行映射.md及三份9-8契约。

本地T0汇总副本tmp/pxm-20260912/B-C-T0-RA-test-results.json包含两协议完整occurrences及summary。SSH helper通过Console.Out输出，PowerShell直接赋值不能捕获；大JSON应临时使用StringWriter接管Console.Out并在finally恢复后保存，避免刷入聊天。不得修改共享helper或输出凭据。
