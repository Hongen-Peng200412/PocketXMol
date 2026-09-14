# Handoff: E-RB报告完成，仅中心T0-RB尚待完整测评

Date: 2026-09-13

## 当前状态

六个无密度模型的训练全部完成，第1、2、3、5、6模型已有完整测试、CPU评价及报告。378693承担的第5、6模型已全部结束，after_lock和try_lock保留，全部训练与测试产物保留，不再为该资源安排新实验。371591继续第4个B-C-T0-RB：19:06 master检查C0完成446实例、22300候选；C5完成50／446实例、2500候选，目前失败0。

当前状态以[总日志](../../../日志/第一类实验（不加密度信息）/总日志&分析/总日志.md)、[第4实验日志](../../../日志/第一类实验（不加密度信息）/4-B-C-T0-RB.md)和[第6实验日志](../../../日志/第一类实验（不加密度信息）/6-B-E-T0-RB.md)为准，当前实现对应关系见[计划执行映射](../../../日志/计划执行映射.md)。沿用现行七份实验日志，不恢复旧文件；每次检查有新进度就更新对应实验与总日志，普通等待不写handoff。

## E-RB的有效结果和来源

有效训练根 `/storage/penghongen/PocketXMol/training/B-E-T0-RB/`，训练W&B 423nfmpm。30400更新后第三次学习率下降停止，实际best21600，原E val/loss=1.6228482723236084，38份定期检查点及last保留。

正式测试与评价根 `/storage/penghongen/PocketXMol/sampling/B-E-T0-RB/test/`，源码d3ffb62996bd、配置提交b666daa。446个实例、22300个候选全部生成并评价成功，原self-ranking、50候选及100步保持。评价W&B为[zykj1eit](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/zykj1eit)，online_completed且无上传错误。CPU评价控制器第6次成功，PID29594退出。

ALL的Top-1／Top-5／oracle为287／365／403个成功实例，即64.35%／81.84%／90.36%，Top-1平均RMSD为2.830 Å；CAP10为158／205／236，HF10_TO5为131／171／198。7个含核酸口袋的Top-1／Top-5／oracle成功数为5／5／7；纯RNA实例9v7o/0保留在正式分母，RMSD分别3.309／3.082／1.594 Å。三视图、PDB等权、评分表现与全部含核酸实例结果已写入第6日志前部，不据此替用户选择核酸分支。

完整只读检查证据 `/storage/penghongen/PocketXMol/control/378693/evaluate_B-E-T0-RB_test_complete_20260913.json`：22300个候选的RMSD和self-ranking均有限，候选错误、RMSD按编号回退及Traceback均为0。316条RDKit allene提示已说明，不能当作316个独立失败实例。逐实例评价耗时累计22419.738169秒，不是作业墙钟时间。

本地权威产物副本 `tmp/pxm-20260913/B-E-T0-RB-test-results.json` 包含服务器summary和全部E/occurrences；`report_e_rb.py`从该副本重算各视图汇总检查通过。主代理逐项核对18条数字表格内容、PDB数量、未被Top-1选中的116个成功候选池及W&B身份。正式评价命令与只读核查命令、release／launch和启动记录已分开登记在第6实验日志，恢复上下文不得重复启动评价。 两轮同范围独立报告核查已通过；首轮仅发现summary路径重复test层，修正后第二轮无新问题。工程细节和边界清单中的三处旧日志规则也已按用户最新指示同步，科学条件不变。

## 剩余工作

第4模型仍使用自己的best21600、原C0训练契约及T0的C0／C5测试机制。采样PID39974，371591／gnode09；源码 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_07ea8b8a2406/PocketXMol`，配置提交933a49c，产物根 `/storage/penghongen/PocketXMol/sampling/B-C-T0-RB/test/`。采样out／err读取起点149921972／321399字节；after_lock保留，try_lock不存在。

等待完整C5结束，核对两协议的实例身份和预算后，在371591的8个CPU进程执行以下正式评价命令。此命令尚未执行：

```bash
bash 训练与运行/sh/evaluate_docking.sh B-C-T0-RB-test
```

评价、逐候选核对与第4报告完成后，再汇总六模型结果并完成双线Git收口：保留真实实现历史，建立学习历史，核验端点等价后快进Learn/CUMULATIVE。本goal仍active，不安排完整验证集采样，不增加实验，不跟踪官方或其他任务。稳定等待按既定12或18次300秒静默执行，不设置heartbeat。
