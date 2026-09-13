# Handoff: 六模型推理完成，最后一项CPU评价运行中

Date: 2026-09-13

## 当前状态

既定六个无密度模型的训练和全部10个模型／协议组合推理均已完成。第1、2、3、5、6模型已有完整报告；第4个B-C-T0-RB已在371591／gnode09接续C0、C5的8进程CPU评价，是剩余唯一正式科学运行。378693完成第5、6模型后保留after_lock、try_lock和全部产物，不再安排新实验。

继续维护[总日志](../../../日志/总日志.md)、[第4实验日志](../../../日志/第一类实验（不加密度信息）/4-B-C-T0-RB.md)及[计划执行映射](../../../日志/计划执行映射.md)。每次检查取得新进度，同步实验与总日志；只有阶段结束或明确分叉才写handoff，不恢复旧实验文件。

## 完成的候选核对

第4模型使用本模型训练best21600，原C0 val/loss=1.8065478801727295。正式产物根 `/storage/penghongen/PocketXMol/sampling/B-C-T0-RB/test/`，C0、C5各446实例、22300候选全部成功，T0新增平移关闭，每实例50候选、100步、batch50保持。PID39974退出，采样控制器第22次成功。

只读核对实际配置、冻结身份、种子、视图、C5向量、候选编号、有限置信度及姿态文件通过。证据为 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T0-RB_test_complete_20260913.json`；脚本本地副本 `tmp/pxm-20260913/inspect_c_t0_rb_sampling.py`，经原作业8核CPU执行，不运行模型或计算RMSD。

C0／C5逐实例采样耗时累计13137.317372／13015.679950秒，模型推理累计13051.676773／12948.097865秒，各44600次批量forward；峰值张量显存3234062848／3227094528字节。累计耗时不是整个作业墙钟时间，未见Traceback、OOM、非有限数值或降低batch记录。

## 正式CPU评价已经启动

master时间2026-09-13 23:16:22保存新旧动态命令及启动记录，再移除恢复的try_lock，由371591控制器第23次接续评价。after_lock保留，没有使用kill_lock、删除候选或重复推理。正式命令已经执行，不得重新提交：

```bash
bash 训练与运行/sh/evaluate_docking.sh B-C-T0-RB-test
```

源码 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_07ea8b8a2406/PocketXMol`，生产来源c9cc4a1，测试配置提交933a49c，LF SHA256为5968646c8d56be437a0131df0110ab22bce611bd4d242d6a6b66b55478499650。实际launch `/home/penghongen/Feedback/PocketXMol/launches/371591/evaluate_B-C-T0-RB_test_job371591_20260913T231331`，节点时钟比master慢约3分钟。

启动记录 `/storage/penghongen/PocketXMol/control/371591/evaluate_B-C-T0-RB_test_start.json`，同目录保存新旧run_cmd。评价out／err读取起点150090468／321495字节，避免混入此前训练或采样。23:17 master检查无Traceback，主进程44073与8个子进程44162至44169已运行；Slurm cgroup为job_371591，CUDA_VISIBLE_DEVICES为空，OMP／MKL／OpenBLAS各1线程。首批逐实例指标尚未产生，评价W&B在最终汇总阶段创建，当前尚无run id。

## 后续工作

稳定运行期间继续按12或18次300秒静默等待60或90分钟，醒来核查本任务状态，不设置heartbeat。评价完整结束后核对控制器成功、PID退出、W&B上传及所有逐候选指标、排序和冻结分母；参考 `tmp/pxm-20260913/inspect_t1_rb_evaluation.py` 和 `report_t1_rb.py` 的只读检查，替换为本模型身份、实际W&B、进程和独立证据文件。已有其他模型不重跑、不覆盖。

完成第4实验日志前部报告及六模型总体报告，再进行双线Git收口：实现分支codex/pxm-receptor-baselines保留真实历史，从共同基点0412824重建学习线，先文档与契约，按依赖组织代码，全部测试最后；端点等价核验后快进Learn/CUMULATIVE，使其再次为按提交者时间唯一最新提交。当前Learn仍0412824，不能现在标记goal完成。保护外来未提交修改，不自动push，不改写运行依赖的历史。

不安排完整验证集采样或评价，不跟踪官方或其他任务，不新增密度实验。六模型结果交给用户后，由用户选择T／核酸分支和密度预算。
