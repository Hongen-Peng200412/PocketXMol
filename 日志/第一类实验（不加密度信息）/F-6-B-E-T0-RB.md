# F-6：包络T0、RB的独立训练与测试

本记录依据 [科学契约](../../想法/方案草稿/9-8-科学契约.md)、[工程细节](../../想法/方案草稿/9-8-工程与实现细节.md)及用户2026-09-11新增资源授权，执行既定六模型中的第6个模型B-E-T0-RB。任务由“核查PocketXMol执行前准备（3: 独立做另外半边）”负责；用户接受重复计算，各次输出独立。

## 正式配置与命令

配置 `configs/docking/F-6.yml` 复制B-E-T0-RB.yml，仅W&B显示名称为F-6_B-E-T0-RB。包络选择完整受体残基，残基重原子质量中心到任一真实配体重原子的距离严格小于10 Å；模型原点为实际输入受体重原子的坐标算术均值，始终关闭T1新增平移。使用RB核酸分支、官方初始参数和独立优化器及调度状态；batch72、累积1、bf16、15个DataLoader worker及每800更新原val/loss等参数不变。

在服务器 `/home/penghongen/My_Project/PocketXMol` 提交：

```bash
bash 训练与运行/submit_task.sh --sh train_docking.sh --resource h100 --gpus 1 --cpus 32 --after_hold --job-name F-6 -- F-6
```

该作业中的正式训练命令为：

```bash
bash 训练与运行/sh/train_docking.sh F-6
```

独立训练根 `/storage/penghongen/PocketXMol/training/F-6/`，首次配置train_config/F-6.yml；保留所有定期检查点和last。W&B使用pencounkdual-111/PocketXmol_raw、名称F-6_B-E-T0-RB，入口创建独立run id。另一对话的B-E-T0-RB产物不作为本任务恢复或测试来源。

训练完成后固定本任务最低原val/loss的实际best，再登记测试配置和正式命令；预定测试输出根 `/storage/penghongen/PocketXMol/sampling/F-6/`。只测试E，每实例50候选、100步、batch50，ALL/CAP10/HF10_TO5共用候选，随后8进程CPU评价并记录结果。保留训练中的val/loss，不运行训练后完整验证集采样或评价。

## 验收与执行状态

主代理第一遍用原make_config确认新增配置只改变W&B名称；第二遍核对包络说明、正式命令、独立输出根和检查点来源。配置检查使用本地Python -X utf8及stdin，不运行科学样本，不是正式实验命令。没有新增或修改Python函数。独立代理对三配置、三日志完成两轮限定审查；官方失败产物说明及最新unstaged规则已修正并关闭窄核，无剩余问题。

已于master时间2026-09-11 16:26:08提交，实际JobId为 **378588**，Slurm名称F-6，单张H100、32核、after_hold，无pre_hold。首次squeue为PENDING/Priority，实际release/launch和W&B id在启动后补记。提交记录 `/storage/penghongen/PocketXMol/control/parallel-20260911/F-6_submit.json` 保存请求、完整命令及返回的作业编号。安全同步已完成，未删除远端文件；配置来源为f856ad8对应原模型加本次未提交的F-6.yml。

本任务只申请一张H100、32核、after_hold；保留after_lock和全部产物。排队或稳定运行时静默等待60/90分钟，每段Start-Sleep -Seconds 300；醒来再检查并通知。用户随后明确本对话所有文件始终保持unstaged，不执行git add或git commit；其他人的修改和暂存内容不处理。记录基准提交f856ad8及实际release/launch，不能把本次未提交配置标成已提交版本。

## 计划与实现差异

并行执行及独立产物命名来自用户明确授权，科学定义与既定第6模型一致；训练、测试、CPU评价及报告尚未完成。
