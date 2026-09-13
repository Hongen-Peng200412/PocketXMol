# Handoff: E-RB完整推理通过核对并进入CPU评价

Date: 2026-09-13

## 当前状态

继续六个无密度模型的原goal。六项训练均完成，第1、2、3、5模型已有报告；第4模型仍在371591进行C0／C5推理，第6模型已在378693完成E推理并接续CPU评价。当前状态必须写入现行[第4实验日志](../../../日志/第一类实验（不加密度信息）/4-B-C-T0-RB.md)、[第6实验日志](../../../日志/第一类实验（不加密度信息）/6-B-E-T0-RB.md)及[总日志](../../../日志/总日志.md)，不恢复旧文件。普通进度检查不新建handoff。

## 已完成的推理与核对

第6模型使用本模型训练best21600，原E val/loss=1.6228482723236084。测试产物根 `/storage/penghongen/PocketXMol/sampling/B-E-T0-RB/test/`；446个实例、22300个候选全部生成成功。完整核对实际配置、冻结实例身份、种子、视图、候选编号及有限置信度通过，无候选错误、Traceback、OOM或降低batch记录。累计逐实例采样12638.163894秒、模型推理12564.934873秒，共44600次批量forward，峰值张量显存3250520064字节；累计耗时不等于作业墙钟时间。

378693采样控制器第5次执行成功，PID33421已退出，after_lock和恢复的try_lock存在。只读核对证据 `/storage/penghongen/PocketXMol/control/378693/sample_B-E-T0-RB_test_complete_20260913.json`；本地脚本 `tmp/pxm-20260913/inspect_e_rb_sampling.py` 和执行入口 `check_e_rb_sampling.sh`，在原作业8核CPU中运行，不重新生成或评价候选。

## 正式评价已启动

master时间2026-09-13 17:45:24保存新旧动态命令、启动记录并移除恢复的try_lock，378693控制器第6次开始正式评价。沿用源码 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_d3ffb62996bd/PocketXMol` 和b666daa测试配置，未修改生产代码或科学预算。正式命令已经执行，恢复上下文时不得重复提交：

```bash
bash 训练与运行/sh/evaluate_docking.sh B-E-T0-RB-test
```

实际launch为 `/home/penghongen/Feedback/PocketXMol/launches/378693/evaluate_B-E-T0-RB_test_job378693_20260913T174435`；启动记录 `/storage/penghongen/PocketXMol/control/378693/evaluate_B-E-T0-RB_test_start.json`，同目录保留新旧run_cmd。评价out／err读取起点61072223／208397字节，避免混入此前采样或训练日志。

主进程29594属于job_378693，CUDA_VISIBLE_DEVICES为空，OMP／MKL／OpenBLAS线程各1。17:48 master检查确认8个评价子进程29710至29717正在计算，各约100%单核CPU；尚无首批assessment.json，无Traceback。after_lock保留，try_lock与kill_lock不存在。评价W&B在汇总阶段创建，当前尚无评价run id。

## 下一步

继续当前评价。完成后检查进程退出、控制器成功、W&B上传完成及全部候选指标，核对Top-1／Top-5／oracle、冻结三视图和分母；把完整结果写入第6实验日志前部及总日志。既有 `tmp/pxm-20260913/inspect_e_ra_evaluation.py` 与 `report_e_ra.py` 可作为只读核查和报告脚本的参考，替换本模型身份、原进程、W&B和独立证据路径，不能重跑原E-RA产物。

第4模型17:42 master检查为C0完成349／446实例、17450个候选，失败0；C5按同一运行顺序接续，尚未开始。采样PID39974，源码07ea8b8a2406，产物 `/storage/penghongen/PocketXMol/sampling/B-C-T0-RB/test/`，out／err起点149921972／321399。完整C0／C5后才接续同作业内8进程评价，不安排完整验证集采样。

目前没有需要用户决策的故障。稳定期间继续以12或18次300秒分段静默等待60或90分钟；每次醒来有新进度就同步实验日志和总日志。保留两项A800资源及所有旧产物，不跟踪官方或其他任务。六模型全部报告及双线Git等价核验、Learn/CUMULATIVE快进仍待完成，goal保持active。
