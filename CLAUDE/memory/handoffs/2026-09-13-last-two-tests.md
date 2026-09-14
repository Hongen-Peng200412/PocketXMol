# Handoff: 六模型训练完成，最后两项完整测试已启动

Date: 2026-09-13

## 当前状态

既定六个无密度模型均已完成训练。第1、2、3、5模型已有完整测试及评价报告；第4个B-C-T0-RB正在371591／gnode09执行C0、C5测试，第6个B-E-T0-RB正在378693／gnode10执行E测试。两者均使用本模型的实际best21600，每实例每协议50候选、100步、batch50。15:01 master时间（UTC+8）的检查确认：第4模型C0完成2／446实例、100个候选，C5随后执行；第6模型E完成175／446实例、8750个候选，目前候选失败均为0。姿态指标尚待完整CPU评价，不能由推理成功率代替。

继续维护现行[总日志](../../../日志/第一类实验（不加密度信息）/总日志&分析/总日志.md)、[第4实验日志](../../../日志/第一类实验（不加密度信息）/4-B-C-T0-RB.md)、[第6实验日志](../../../日志/第一类实验（不加密度信息）/6-B-E-T0-RB.md)和[计划执行映射](../../../日志/计划执行映射.md)。遵守[AGENTS.md](../../../AGENTS.md)及[日志整理交接](2026-09-13-experiment-log-consolidation.md)：每实验只有一份日志，结果和有效来源在前，执行及失败历史在后；每次取得新进度同步实验日志与总日志，不恢复旧文件。普通进度检查不再写handoff。

## 完成事项与有效来源

第4模型在31200次更新后第三次学习率下降而正常停止，39份定期检查点及last保留，W&B hthglbuy上传完成。Slurm 371591.8的8核只读检查确认实际best21600的原C0 val/loss为1.8065478801727295，1236个模型参数键均有限，优化器及调度器最终学习率为8.000000000000002e-7。摘要位于 `/storage/penghongen/PocketXMol/training/B-C-T0-RB/training_summary_20260913.json`，本地副本为 `tmp/pxm-20260913/B-C-T0-RB-training-summary.json`。

第4模型测试配置提交933a49c，仅增加本模型身份、RB分支及明确best路径，生产Python不变；两遍主自查和同范围两轮独立审查均通过。测试源码为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_07ea8b8a2406/PocketXMol`，配置LF SHA256为5968646c8d56be437a0131df0110ab22bce611bd4d242d6a6b66b55478499650。master 14:58:53接入371591控制器第22次执行，主进程39974的Slurm归属已核对。实际launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/sample_B-C-T0-RB_test_job371591_20260913T145551`；启动记录为 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T0-RB_test_start.json`，同目录保留新旧动态命令。out／err读取起点为149921972／321399字节，避免混入此前训练日志。测试产物根为 `/storage/penghongen/PocketXMol/sampling/B-C-T0-RB/test/`。

第6模型训练在30400步正常停止，best21600的原E val/loss为1.6228482723236084，W&B 423nfmpm上传完成，38份定期检查点及last保留。详细训练结束证据见[此前阶段交接](2026-09-13-e-rb-best-testing.md)。测试源码为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_d3ffb62996bd/PocketXMol`，配置提交b666daa，LF SHA256为483efb6ee6c6340271b4a839340a0767ecc411a40c8d8bbc5269d0c3db2abbb0。378693控制器第5次执行，主进程33421，out／err读取起点60987569／208301字节；测试产物根为 `/storage/penghongen/PocketXMol/sampling/B-E-T0-RB/test/`。

## 正式命令与核查命令

以下两条正式采样命令已经分别在上述release和作业中执行，恢复上下文时不得重复启动：

```bash
bash 训练与运行/sh/sample_docking.sh B-C-T0-RB-test
bash 训练与运行/sh/sample_docking.sh B-E-T0-RB-test
```

各自全部规定协议完成、只读核对候选数量及身份后，在原作业的8个CPU进程执行对应正式评价。以下两条评价命令尚未执行：

```bash
bash 训练与运行/sh/evaluate_docking.sh B-C-T0-RB-test
bash 训练与运行/sh/evaluate_docking.sh B-E-T0-RB-test
```

测试、门控及只读检查是独立工作，不属于上述正式命令。本地检查点核查脚本为 `tmp/pxm-20260913/check_c_t0_rb_checkpoint.sh` 和 `check_e_rb_checkpoint.sh`；后续候选及评价检查可以参照同目录 `inspect_e_ra_sampling.py`、`inspect_t1_rb_sampling.py`、`inspect_e_ra_evaluation.py`、`inspect_t1_rb_evaluation.py`，改用本模型的身份、目录和独立检查输出，不重跑已完成模型。

## 决定与后续动作

继续固定科学条件：T0的C5测试保留实际给定偏移中心，后续只执行原高斯；E保持完整残基包络和实际受体重原子均值原点，关闭新增平移。空E仅在训练及原val/loss跳过并记录，正式测试失败保留分母。训练后直接测试，不安排完整验证集采样、不增加密度或其他实验。

两张A800的after_lock均保留，采样期间try_lock不存在，没有使用kill_lock或scancel。完成测试后，按实际锁协议依次接入各自CPU评价，保留所有检查点和已有产物。得到评价结果后，把三视图、PDB等权、原self-ranking表现和核酸占比分析写入各自实验日志前部，再汇总六模型结果；核酸分支选择留给用户。

当前实现分支为codex/pxm-receptor-baselines；接受用户日志备份900a50e和整理fac5d44。实现稳定且六模型报告完成后，仍需整理学习历史、核验实现与学习端点等价，并快进Learn/CUMULATIVE，不能现在标记goal完成。无须跟踪官方测评或其他并行任务；共享日志中已有记录保留即可。

稳定运行时静默等待60或90分钟，用12或18次 `Start-Sleep -Seconds 300` 分段完成，工具每次等待不超过60秒；用户输入到达时及时处理。醒来核查两项获准资源并继续推进，不设置heartbeat，不因进入等待而结束任务。
