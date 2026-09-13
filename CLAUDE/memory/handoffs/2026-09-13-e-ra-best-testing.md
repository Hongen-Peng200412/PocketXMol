# Handoff: E-RA完整测试与T1-RB CPU评价

Date: 2026-09-13

## 当前任务与资源

继续当前goal，完成既定六个无密度模型的训练、完整测试、CPU评价与报告。不跟踪其他任务或官方模型测评，不增加实验。正确T0-RA和T1-RA的完整报告已经完成。每张卡按“一个模型训练→实际best规定测试→同卡CPU评价→报告→下一模型”执行。

- 371591、gnode09、A800／16CPU：E-RA有效重训已经完成，正在用21600步best执行完整E测试，实际采样进程52145。完成评价报告后进入第4个B-C-T0-RB。
- 378693、gnode10、A800／16CPU：T1-RB的C0／C5各446实例、22300候选已全部生成成功，原采样进程26403已退出。2026-09-13 master 02:56:31启动同卡8进程CPU评价，主进程10865；完整报告完成后再运行第6个B-E-T0-RB。
- 两项after_lock保留；运行中try_lock不存在。实际锁控制目录是 `/home/penghongen/Feedback/Pocket_Plus/allocations/<job>/`，try_lock位于其父目录。PocketXMol release／launch是科学代码来源，不是旧资源的锁控制根。不得用scancel或删除after_lock结束资源。

## E-RA训练完成证据

有效训练根为 `/storage/penghongen/PocketXMol/training/B-E-T0-RA-nonemptyE/`，W&B [lukfzmf3](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/lukfzmf3)。本次从官方权重重新初始化，不承接异常E参数。26400次优化器更新时第三次实际学习率下降触发停止，33个定期检查点与last均保留。原E val/loss最低值为1.6292482614517212，对应21600步best；1130个model参数键全部有限。

CPU核对报告为同训练根 `training_summary_20260913.json`，脚本 `/storage/penghongen/tmp/pocketxmol_checkpoint_20260913/inspect_e_ra.py`，在371591.5的8核CPU执行。最后验证步26400，decline_count=3，stop_reason=plateau，末次学习率8e-7。完整训练没有NaN／OOM，46个空E训练身份均有跳过警告，验证跳过0。

空E批准规则、全部46个身份、两遍自查、两轮三类审查、CPU 22 passed及RA／RB真实训练输入GPU验收详见[日志07](../../../日志/第一类实验（不加密度信息）/3-B-E-T0-RA.md)。生产修复c9cc4a1只修改Dataset；原阈值、标准残基范围、模型／loss／噪声及冻结清单不变。仅E训练和val/loss迭代跳过空输入，正式测试仍尝试每个实例，失败保留分母。旧异常E目录和W&B so6e0mvy全部保留，不作为有效结果。

## E测试正式命令与来源

配置 `configs/docking/sample-B-E-T0-RA-test.yml` 已通过两遍自查与两轮独立配置核对，提交aac0b89。使用独立重训保存配置及21600步best，RA、E、T0关闭新增平移。446个实例，每实例50候选／100步、batch50，不进行完整验证集采样。

正式推理命令：

```bash
bash 训练与运行/sh/sample_docking.sh B-E-T0-RA-test
```

2026-09-13 master 00:48:27接入，控制器第19次执行。实际源码为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_1ce7d5426c18/PocketXMol`，由已修复43742cdf8166加入明确测试配置和日志07构成。launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/sample_B-E-T0-RA_test_job371591_20260913T004527`，节点时钟约慢3分钟。

启动记录为 `/storage/penghongen/PocketXMol/control/371591/sample_B-E-T0-RA_test_start.json`，同目录保存新旧动态命令。本次out／err起点113763414／282077。实际test/run.json已确认正确检查点、RA、E、T0、50×100及batch50。输出根为 `/storage/penghongen/PocketXMol/sampling/B-E-T0-RA-nonemptyE/test/`。

全部E推理完成并核对后，继续在同一371591、同一release正式评价，8个CPU进程，入口自行关闭CUDA：

```bash
bash 训练与运行/sh/evaluate_docking.sh B-E-T0-RA-test
```

W&B名称B-E-T0-RA_test，实际评价run id启动后记录。不要把检查点核对或启动探查命令写成正式运行命令。

## T1-RB接续与待回复事项

T1-RB测试release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_c59a88a03678/PocketXMol`，训练W&B wmgkgurr，best17600、原C5 val/loss=2.8354082107543945。完整候选核对已通过，记录在 `/storage/penghongen/PocketXMol/control/378693/sample_B-C-T1-RB_test_complete_20260913.json`；两协议逐实例耗时合计13279.412／17366.621秒，候选、冻结身份与预算一致，没有失败。当前同卡CPU正式命令：

```bash
bash 训练与运行/sh/evaluate_docking.sh B-C-T1-RB-test
```

评价控制器第3次执行，CUDA_VISIBLE_DEVICES为空。启动记录为 `/storage/penghongen/PocketXMol/control/378693/evaluate_B-C-T1-RB_test_start.json`，out／err起点25867448／8359；launch为 `/home/penghongen/Feedback/PocketXMol/launches/378693/evaluate_B-C-T1-RB_test_job378693_20260913T025524`。代码和配置继续使用同一c59a88a03678，不另申请CPU，不重复生成候选。

2026-09-12约23:07已向用户提出可选资源问题，尚未收到答复：378693获配GPU出现本作业1个、作业外3个计算进程，部分近期C5实例较对应C0慢约3倍，可能存在并发竞争。没有GPU降频、I/O等待或采样失败证据；未查看其他任务内容或处理其进程。现有推理按原授权继续，不把用户未回复视为允许更改资源。具体数据及提问见[日志06](../../../日志/第一类实验（不加密度信息）/5-B-C-T1-RB.md)。

## 后续执行与记录

正常稳定运行后静默等待60或90分钟，使用多次Start-Sleep -Seconds 300，工具等待每次不超过60秒，期间不发消息、不设置heartbeat、不结束goal。醒来核对本任务两卡；出现实际问题及时处理或向用户说明。

评价报告沿用已完成日志04／05的范围：ALL／CAP10／HF10_TO5、实例等权与PDB等权、Top-1／Top-5／oracle、评分相关、核酸比例与纯核酸实例、失败分母和实际耗时。不得用不同训练条件的val/loss选最终科学优胜者。第6个E-RB必须使用c9cc4a1修复后的Dataset，不使用旧fa0d957b2d3f训练E。

Git仍在持续任务实现分支codex/pxm-receptor-baselines，Learn/CUMULATIVE共同基点0412824。仅提交本任务明确文件，保护并忽略外来未提交文件。六模型和报告完成后再整理学习线、核对端点等价并快进Learn/CUMULATIVE；不重新触发已通过的起点审批。

状态索引见[总日志](../../../日志/总日志.md)和[计划执行映射](../../../日志/计划执行映射.md)。本handoff只在训练完成、测试或评价接续时更新，不随每次状态探查更新。
