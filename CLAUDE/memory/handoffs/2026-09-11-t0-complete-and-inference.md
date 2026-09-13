# Handoff: 正确T0训练完成并开始配套完整推理

Date: 2026-09-11

## Current State

端到端goal仍active。用户最新更新明确要求逐模型完成：每个模型训练结束后，先完成其预定的完整验证／测试推理、评价和结果记录，再进行下一个模型。三份9-8契约、运行README、总日志和映射已同步该顺序。B-C-T1-RA只登记命令，没有提交；不能越过当前T0的结果收口直接训练它。

正确的B-C-T0-RA已经正常训练完成。当前使用它的21600步best，在既有371591、gnode09、单张A800内运行完整validation的C0／C5采样。正式命令为 `bash 训练与运行/sh/sample_docking.sh B-C-T0-RA-validation`；每实例50候选、100步、采样batch25。首两个实例5irx/0和5irx/1均完成50个候选，实际各200次批量forward；只有生成状态，尚没有完整RMSD或排名报告。

## Completed

- 正确T0训练源码693c7ec，独立产物 `/storage/penghongen/PocketXMol/training/B-C-T0-RA-C0/`，W&B `pencounkdual-111/PocketXmol_raw/ceqrh2ve`。从官方参数初始化，没有继承旧错误T0。31200步第三次下降后正常停止，fit进度耗时8小时8分8秒；last内global_step及last_validation_step均31200、decline_count=3、stop_reason=plateau、优化器／调度器lr均8e-7。最佳原C0验证损失1.7970343828201294，对应 `checkpoints/step=21600.ckpt`，best内部步数已核对。39个定期检查点及last全部保留。
- 已通过371591内8核Slurm步骤读取last和best，未在普通SSH会话加载模型。完整状态及验证损失保存在训练目录training_summary_20260911.json，检查脚本在 `/storage/penghongen/tmp/pocketxmol_checkpoint_20260911/inspect_corrected_t0.py`。最后W&B train/lr=4e-6是第三次下降前日志，以检查点停止状态为准。
- 原错误T0目录 `/storage/penghongen/PocketXMol/training/B-C-T0-RA/`、W&B 9wkyn4qn、源码、日志和检查点继续保留，不能作正确T0基线或新训练初始化。旧运行已按用户许可通过kill_lock停止，不scancel；详细错误及保留记录见日志01／02。
- 本次纠偏已完成两遍主代理自查、两轮三类独立审查及后续边界窄核；生产只改变Dataset动态C5条件和DataModule监督条件选择。CPU376811／376817均14项通过，新的72×1 T0 GPU短验收通过；T1公式及正式C0／C5输入能力保持。不再进行第三轮全面代码审查。
- 新两份实际best采样配置为sample-B-C-T0-RA-validation.yml和sample-B-C-T0-RA-test.yml，读取正确训练保存的配置，RA、center_translation=false、C0／C5、50×100、共同冻结资产根与官方预算一致。主代理两遍配置自查、原make_config解析及独立逻辑代理的本次配置窄核通过；没有新增或修改Python执行逻辑。配置与最新顺序提交d788edb。
- 用户后续注释／Docstring和空行改动按授权核对并保留：b1c9270包含sampling.py说明与models/sample.py函数定义阅读顺序；3ccd5c1包含evaluation.py说明与sampling.py空行。忽略Docstring后的执行AST和配置值未变。用户自行提交eaf1165的/tmp/忽略规则保留。

## Decisions

T0训练和监督验证为C0，不抽新增偏移；T1训练为动态C5、监督验证为冻结C5，完整同一delta选袋和定原点，原高斯后增加s*delta，s=1-level_dict['pos']。推理口袋和实际给定中心固定；首步纯先验，T0后续原高斯、T1后续额外-s*mean(Z)，最终+C一次。T0+C5与T1+C0均保留。六模型及官方评价矩阵不增加、删减或由成绩改变；密度正式实验仍等待用户选择。

训练配置已经冻结：72×1、bf16、AdamW初始1e-4、warmup0，800步原val/loss，Plateau相对1%、patience5、factor0.2、第三次下降或40000步停止。其余五模型仍从官方参数开始，不能根据先看到的T0测试成绩调整配置、种子、预算或筛选。共同数据不重建，train／validation／calibration／test为65290／781／361／446；测试视图ALL／CAP10／HF10_TO5为446／272／227。坏图和语言异常直接排除，纯核酸在共同有效清单内保留。

只有371591获得GPU使用授权，保留after_lock。CPU每任务8核、纯CPU并发≤96核；不要申请额外GPU、操作其它作业或删除服务器资产。单模型内允许完整validation采样完成后提交其CPU评价，并行执行该模型test采样；两集合结果全部收口后才训练下一模型。

## Next Actions

1. validation采样已进入正式循环，按用户要求静默等60或90分钟，使用多次Start-Sleep -Seconds 300，并让每次工具等待不超过60秒。不要heartbeat，不因稳定运行而结束goal。使用按开始时间加90分钟的截止时间循环，避免主机挂起后仍补足固定次数而过度延长。
2. 醒来读取当前采样新增out／err与result.json数量／失败阶段。实际控制根仍是 `/home/penghongen/Feedback/Pocket_Plus/allocations/371591/`，try_lock位于父目录；原控制器第12次执行、采样进程启动PID19347。当前正式release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_b3b44f062a12/PocketXMol`，launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/sample_B-C-T0-RA_validation_job371591_20260911T020258`。启动记录 `/storage/penghongen/PocketXMol/control/371591/sample_B-C-T0-RA_validation_start.json` 保存本次out／err起点56944267／73179；只读此后输出，避免旧训练日志混入。master和gnode09时钟相差约3分钟，跨主机时间不直接相减，耗时采用程序计时。
3. validation的C0和C5各781个实例全部尝试完成后，提交 `bash 训练与运行/submit_task.sh --sh evaluate_docking.sh --resource cpu --cpus 8 -- B-C-T0-RA-validation`。在相同GPU锁协议下，用新release／launch和已登记短命令 `bash 训练与运行/sh/sample_docking.sh B-C-T0-RA-test` 开始同模型test的C0／C5（各446个实例）。不要重建冻结偏移或候选种子，不因失败补生成。
4. test采样全部结束后再提交同格式8核CPU评价，参数改为B-C-T0-RA-test。记录W&B评价run、逐实例及汇总、失败分母、核酸占比与排名分析，完成本模型完整报告后才启动已登记的B-C-T1-RA。官方完整C0／C5／E对照仍属于原目标，尚未正式运行；不要遗漏。
5. 仅在正式提交、训练完成、推理评价收口等有意义事件更新handoff，避免每次验证或计数记流水账。当前继续codex/pxm-receptor-baselines实现分支，共同学习基点0412824；最终完整六模型与官方报告后组织学习分支、核验端点等价并快进Learn/CUMULATIVE。保护并按授权核实用户注释改动，不能混入其未授权逻辑修改。

## Files To Reopen

- [T0推理与评价记录](../../../日志/第一类实验（不加密度信息）/1-B-C-T0-RA.md)：正式命令、配置、预算、窄核及运行来源。
- [T0修复与重训记录](../../../日志/第一类实验（不加密度信息）/1-B-C-T0-RA.md)：纠偏、全部验收和正确训练停止／best证据。
- [T1-RA预登记](../../../日志/第一类实验（不加密度信息）/2-B-C-T1-RA.md)：当前未启动，等待T0推理评价收口。
- [总日志](../../../日志/总日志.md)、[计划映射](../../../日志/计划执行映射.md)、三份9-8规格与 [运行说明](../../../训练与运行/README.md)。
- 当前正式采样输出 `/storage/penghongen/PocketXMol/sampling/B-C-T0-RA-C0/`，validation和test分目录；采样完整配置在各自run.json中，模型身份B-C-T0-RA。结果格式见docking/sampling.py、docking/evaluation.py。
