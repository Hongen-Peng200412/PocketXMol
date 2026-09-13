# 377521：官方冻结模型的独立测试与评价

本记录依据 [科学契约](../../想法/方案草稿/9-8-科学契约.md)的官方对照与共同评价协议，以及用户2026-09-11对374480_0的接管授权。本次工作属于既定官方测试，运行由“核查PocketXMol执行前准备（3: 独立做另外半边）”负责；不改变另一对话的371591或其它数组成员。

本次尝试已由用户手动取消，当前官方测试转到 [374480_1实际377793](official-377793-测试与评价.md)。下文377521命令和资源仅作历史记录，不再执行。

## 资源与产物隔离

只读核实：用户点名374480_0的实际Slurm JobId为377521，gnode08，单张A100、8核，原任务为cryoatom2_test0。原控制目录为 `/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/test_0_chain06/运行日志与统计/slurm/allocations/377521/`；after_lock_377521在此目录，try_lock_377521在父目录。原预测已成功结束，控制器正在after_hold等待；该作业cgroup中未见活跃预测进程。执行前再次核实，再修改本作业动态命令并按实际try_lock协议启动；保留after_lock和全部旧产物，不触碰其它数组成员，也不使用scancel。

正式配置 `configs/docking/sample-official-377521-test.yml` 复制sample-official-test.yml，只将输出根改为 `/storage/penghongen/PocketXMol/sampling/official-377521`，W&B评价名称改为official_377521_test；model_name仍为official。官方只读权重为 `/storage/penghongen/PocketXMol_official_test/extracted/data/trained_models/pxm/checkpoints/pocketxmol.ckpt`，训练配置为同模型目录下train_config/train.yml。

测试实际落在上述输出根的test目录，C0/C5/E分别保存候选，每协议每实例50候选、100步、batch50。ALL有446个实例；CAP10和HF10_TO5分别272和227个，复用各协议已有候选，共九组评价汇总。输入保持官方标准蛋白特征；核酸参与共同受体碰撞评价。纯核酸实例按真实失败阶段及候选预算记录，不补生成、不从分母中静默删除。

## 正式命令

在该作业内的PocketXMol冻结release根目录，顺序运行：

```bash
bash 训练与运行/sh/sample_docking.sh official-377521-test
bash 训练与运行/sh/evaluate_docking.sh official-377521-test
```

第二条使用该allocation现有8核，入口关闭CUDA可见性，读取第一条的同一配置和候选；不另申请GPU。推理进程正常结束后才进入评价，after_hold继续保留资源。每实例保存candidates.json和result.json；成功候选写入poses.sdf，至少一个候选成功时才写confidence.npz。预处理失败时可能没有poses.sdf；零成功时result.json中的pose_file与confidence_file均为null。评价保留candidate_metrics.json、assessment.json、occurrences.json、summary.json，并以pencounkdual-111/PocketXmol_raw中的独立run汇报。

## 验收与执行状态

第一遍主代理自查用原make_config解析，确认只有output_root及W&B名称改变，C0/C5/E、test、50候选/100步及官方权重保持；第二遍核对实际作业编号、输入协议与三个视图的区别、CPU配额及输出隔离。没有改动Python执行逻辑，不重复此前完整模型审查或新增测试集门控。配置解析是本地Python stdin检查，不是上面的正式命令。独立代理完成两轮限定审查，修正失败实例文件说明，并窄核关闭最新unstaged规则，无剩余问题。

已于gnode08时间2026-09-11 16:23:42按实际try_lock协议请求运行。操作前重新核对Slurm资源、array身份和/proc进程，只见原控制器PID86152，无活跃Python预测；原run_cmd已备份为 `/storage/penghongen/PocketXMol/control/377521/run_cmd_377521_before_official_20260911.sh`。写入新动态命令后消耗try_lock，after_lock保持；未使用kill_lock或scancel，未删除旧产物。新命令先用bash -n检查语法，该检查是控制命令验收，不是正式测试。

启动记录 `/storage/penghongen/PocketXMol/control/377521/official_377521_test_start.json` 保存资源、原命令备份、输出根及out/err读取起点17624/161；同目录official_377521_test_run_cmd.sh保存新动态命令。节点和master时钟存在差异，原样注明时间来源，不用跨节点时间相减估计耗时。

启动初段尚无新增模型日志。按/proc核查后确认原控制器在执行AdaLigand的create_release.sh，子进程继续校验源码及tests_output中的文件；这是原控制器每次执行run_cmd前已有的步骤，尚未进入新PocketXMol命令。没有报错，不修改旧控制器或删除其文件；按用户要求静默等待60分钟后再确认实际模型加载和采样。

本次接管前的命令备份、启动记录和动态命令留在 `/storage/penghongen/PocketXMol/control/377521/`；实际release/launch和推理/评价状态在启动后补记。服务器源资产只读。按用户随后明确要求，本对话新增及修改文件永远保持unstaged，不执行git add或git commit，也不处理其他人的修改或暂存内容。运行来源为基准提交f856ad8加本次未提交文件，实际执行内容由release/launch与配置快照留证。

## 计划与实现差异

用户新增授权A100用于既定官方测试，并接受与另一对话重复计算；本次通过独立产物根避免混写。官方科学行为、三协议与三视图均不变。本次尝试未完成，后续使用用户最终指定的377793。

## 取消与保留证据

60分钟静默等待后发现Slurm作业已退出。用户随即说明，因怀疑gnode08异常亲自执行scancel 374480_0；sacct记录为CANCELLED by 1351，结束时间2026-09-11 17:27:37，batch进程退出码15。after_lock等活动控制文件由原控制器退出清理，本对话未删除这些锁或发出scancel。

本次实际PocketXMol release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_db3c806a5acc/PocketXMol`，launch为 `/home/penghongen/Feedback/PocketXMol/launches/377521/official_377521_test_job377521_20260911T162940`。run.json表明模型严格加载完成，但C0/C5/E均无完成的result.json；只见C0/11jb/0/poses.sdf为空文件。未生成完整候选，未运行CPU评价；保留该目录、release/launch和全部日志，不把本次尝试作为官方测试结果。
