# Handoff：D2中心训练完成，best测试开始

2026-09-16。总goal继续执行，尚未完成全部获准实验；本文件记录阶段节点，实时进度见逐实验日志与阶段总日志。

## D2中心训练结果

378693上的D2-C-T0-RA正常退出，20800次优化器更新在第三次学习率下降时停止。完整last.ckpt核对decline_count=3、stop_reason=plateau、last_validation_step=20800、最终优化器lr=8e-7，best为step=11200.ckpt，val/loss=1.783219575881958。训练36×2、bf16-mixed、从规定官方参数初始化，未发生OOM或手动改批；W&B 0t1vqqgk online finished。原源码、26份定期checkpoint、last及日志全部保留。

## 正式测试

新增`configs/docking/sample-D2-C-T0-RA-test.yml`读取真实保存训练配置及best11200，保持RA、D2、C0/C5、冻结SMILES清单、50候选×100步、batch50、8进程CPU及W&B online。主代理逐字段自查、服务器只读YAML解析与一轮独立审查批准，未改生产Python或重跑GPU预实验。提交为3f82cee3154ac74d9d80434e45f7a081fd6b5a9e。

11:04核对实际JobId378693、gnode10、16CPU、A800 UUID GPU-0e253751-cce3-c71d-c2df-a222fec3759b及控制器43832等待；after_lock和父目录try_lock存在、kill_lock不存在、新测试产物目录不存在。11:07:40备份旧控制命令并消费try_lock，11:07:55正式开始。release为`/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_d22a8cafa570/PocketXMol`，launch为`/home/penghongen/Feedback/PocketXMol/launches/378693/formal_D2-C-T0-RA_test_3f82cee`。

正式命令为`bash 训练与运行/sh/sample_docking.sh D2-C-T0-RA-test`，正常退出后自动执行`bash 训练与运行/sh/evaluate_docking.sh D2-C-T0-RA-test`。输出`/storage/penghongen/PocketXMol/sampling/D2-C-T0-RA/test`。本地`tmp/formal-execution-20260914/start-d2-center-test-378693.sh`只记录资源控制；服务器`/storage/penghongen/tmp/pxm_formal_execution_20260914/runs/D2-C-T0-RA-test/`保存stdout、stderr及previous_run_cmd，和正式命令分开。

## 其他任务与后续顺序

371591继续D1包络，之后D4中心→包络；379403_1继续D2包络，全流程后保留等待；378693本次C0/C5测试和CPU评价完成后直接D3中心→包络，不等待D2包络。379402继续官方冻结C0/C5/E测试，前面两个无密度复验全流程已完成。最新11:01／11:03核查：D1包络14623更新、best10400；D2包络13133更新、best11200；官方C0 230/446实例、11500候选成功。

W&B在线同步已从远端API核实。慢查询只改为有90秒观察上限的只读查询，不改变训练或online模式。稳定后继续约定分段等待，不设heartbeat；每次新进度更新实验与阶段总日志。双线Git最终收口仍待全部目标完成，保护他人未提交代码及文档。
