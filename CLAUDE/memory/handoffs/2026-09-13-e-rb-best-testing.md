# Handoff: E-RB训练完成并进入E测试

Date: 2026-09-13

## 当前目标与记录入口

继续原goal，完成既定六个无密度模型的训练、完整测试、CPU评价与报告。四个模型已经完成报告；第4个C-T0-RB仍在371591训练，第6个E-T0-RB已在378693完成训练并接续E测试。两张卡独立推进，原训练val/loss及best选择保留；训练后直接完整测试，不做完整验证集采样。

必须接受用户已提交的日志重整：每个实验只有一份日志，当前状态、产物、best、W&B及结果在前，失败／中断／并行尝试在后，不恢复旧文件。每次检查得到新进度，更新对应实验与[总日志](../../../日志/第一类实验（不加密度信息）/总日志&分析/总日志.md)的状态快照；阶段结束或明确分叉才写handoff，普通step或单次验证不新建handoff。当前规则见[AGENTS.md](../../../AGENTS.md)及[整理交接](2026-09-13-experiment-log-consolidation.md)。

## 第6个E-T0-RB：正在完整E测试

唯一实验日志为[6-B-E-T0-RB](../../../日志/第一类实验（不加密度信息）/6-B-E-T0-RB.md)。训练根 `/storage/penghongen/PocketXMol/training/B-E-T0-RB/`，训练W&B为423nfmpm。30400步原验证后第三次实际学习率下降，正常停止；38份定期检查点与last全部保留，训练日志记录46个不同空E训练身份、0个验证跳过，无NaN／OOM。主进程57306已退出并完成W&B上传。

在378693.3的8核CPU核对last与best：实际best为21600步，原E val/loss=1.6228482723236084，1236个model参数键均有限；最终优化器／调度器学习率8.000000000000002e-7，decline_count=3。调度器1%相对阈值使用的比较值1.6360849142074585与raw-loss best不同，测试必须使用21600步。完整摘要 `/storage/penghongen/PocketXMol/training/B-E-T0-RB/training_summary_20260913.json`，本地副本 `tmp/pxm-20260913/B-E-T0-RB-training-summary.json`。

正式命令已于master 2026-09-13 13:34:08接入378693控制器第5次执行：

```bash
bash 训练与运行/sh/sample_docking.sh B-E-T0-RB-test
```

配置 `configs/docking/sample-B-E-T0-RB-test.yml`，提交b666daa；RB、E、T0关闭新增平移、test446实例、每实例50候选×100步、batch50。源码使用 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_d3ffb62996bd/PocketXMol`，以已审查43742cdf8166源码副本增加本次测试配置，生产代码不变。配置LF SHA256为483efb6ee6c6340271b4a839340a0767ecc411a40c8d8bbc5269d0c3db2abbb0。主代理两遍配置／说明自查、独立代理同范围两轮只读核查均已通过，不再扩大已通过的空E代码审查。

实际launch为 `/home/penghongen/Feedback/PocketXMol/launches/378693/sample_B-E-T0-RB_test_job378693_20260913T133258`，节点时钟约比master慢1分钟。启动记录 `/storage/penghongen/PocketXMol/control/378693/sample_B-E-T0-RB_test_start.json`；同目录已保存新旧run_cmd。当前out／err起点60987569／208301，读取时不要混入训练日志。

采样主进程33421的cgroup已核对属于job_378693，gnode10，A800／16CPU。输出 `/storage/penghongen/PocketXMol/sampling/B-E-T0-RB/test/`，13:36 master检查run.json配置正确、3／446实例完成，11jb/0、1、2各50候选成功。前两实例32.81／31.98秒，各100次批量forward，峰值张量显存约2.21 GB。after_lock保留，try_lock不存在，无采样错误。

完整E候选完成后先核对已有产物及冻结预算，再用同一378693的8个CPU进程执行正式评价命令（尚未执行）：

```bash
bash 训练与运行/sh/evaluate_docking.sh B-E-T0-RB-test
```

不得另排完整验证集或新增实验。评价结束后将三视图、PDB等权、原self-ranking及核酸占比结果写入现有第6实验日志前部，并更新总日志。

## 第4个C-T0-RB：继续正确C0训练

唯一日志[4-B-C-T0-RB](../../../日志/第一类实验（不加密度信息）/4-B-C-T0-RB.md)。371591／gnode09，A800／16CPU，训练根 `/storage/penghongen/PocketXMol/training/B-C-T0-RB/`，W&B hthglbuy，主进程7586。正式训练源仍43742cdf8166，启动记录control/371591/train_B-C-T0-RB_start.json，out／err起点113849959／311189。

13:36 master检查27893步，lr4e-6，已下降两次；最近验证27200步loss1.99282，日志最低21600步1.80655仅为训练中观测，不预选最终best。已有34份定期检查点和last，无Traceback／OOM／非有限损失，after_lock保留、try_lock不存在。训练完成后在作业内核实实际last、best和停止状态，再建立明确best的C0／C5测试配置，完整推理后用同卡CPU评价，不承接任何其他模型参数。

训练仍为72×1、bf16、15 workers，AdamW1e-4／warmup0，每800次优化器更新计算原val/loss；Plateau相对阈值1%、patience5、factor0.2，第三次下降停，上限40000步。中心T0训练／val为C0，不抽中心偏移；T0+C5测试必须保留实际给定偏移中心并仅执行原高斯。

## 等待、范围与Git

实际锁仍在 `/home/penghongen/Feedback/Pocket_Plus/allocations/<job>/`，after_lock／kill_lock带作业号，try_lock在父目录。只操作两项已获准资源，不使用scancel，不删除after_lock或旧产物。此前错误T0、异常E和取消validation的产物均保留在对应实验末尾，不复用为有效基线。

稳定状态继续静默60或90分钟，以12／18次Start-Sleep -Seconds 300组成；工具单次等待不超过60秒，用户来信及时响应。没有heartbeat，不因进入等待而结束goal。醒来只检查本任务两张卡；官方和并行尝试仅沿用共享日志已有历史，不接管、不跟踪。

当前实现分支codex/pxm-receptor-baselines。用户日志备份及整理提交900a50e、fac5d44已接受，后续本任务提交914486a更新快照、b666daa保存E-RB测试配置与训练完成记录。Learn/CUMULATIVE仍为共同基点0412824；六模型全部报告完成后进行学习历史整理、端点等价核验和累计分支快进。保护其他任务文件，不把外来未提交修改混入提交。
