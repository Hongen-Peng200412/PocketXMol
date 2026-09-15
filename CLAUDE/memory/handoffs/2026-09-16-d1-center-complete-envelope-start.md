# Handoff：D1中心收齐结果，两个包络模型进入下一阶段

日期：2026-09-16。正式目标继续执行，尚未完成全部实验；最新进度以实验日志及所属阶段总日志为准。

## 阶段成果

D1-C-T0-RA已完成训练、best20800的C0/C5完整测试及本资源8进程CPU评价。每协议446实例、22300候选，生成和RMSD评价全部成功；C0/C5实例等权Top-1分别58.52018%／40.80717%。完整结果在`/storage/penghongen/PocketXMol/sampling/D1-C-T0-RA/test/summary.json`，三个视图、PDB等权及逐实例记录均保留。评价2026-09-16 01:16:01正常结束，W&B cpitnsqq为online_completed、error=null。详细表格及原始路径已写入`日志/第二类实验（密度分支的训练）/1-D1.md`。

B-E-T0-RA-SMILES训练完成，31200更新、第三次实际学习率下降停止，best21600、val/loss=1.5245567560195923。完整last确认decline_count=3及最终优化器lr=8e-7，训练W&B 2uv4mdnt在线结束，全部检查点和历史产物保留。新测试入口`configs/docking/sample-B-E-T0-RA-SMILES-test.yml`经主代理自查与一轮独立审查批准，提交0f09526ea210b573071c1c601881c16412b7999a；没有修改生产Python或重复GPU验收。

## 当前正式任务

- 371591：D1-E-T0-RA于01:18:04开始，官方参数新初始化、72×1、bf16-mixed，W&B n7awsoa3 online。使用原已验收release e6173e5c817d，训练产物`/storage/penghongen/PocketXMol/training/D1-E-T0-RA`；初始加载尚未出现验证检查点。完成训练、best的E测试及CPU评价后依次D4中心、D4包络。
- 379402：无密度包络best21600的E测试于01:17:57开始，首实例50候选成功，随后自动8进程CPU评价。使用新release`/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_0beb73604a7d/PocketXMol`；输出`/storage/penghongen/PocketXMol/sampling/B-E-T0-RA-SMILES/test`。该流程结束后才执行官方冻结权重C0/C5/E，不训练官方对照。
- 378693：D2中心继续训练，01:13约15593更新、best11200／1.78322；全流程完成后D3中心→D3包络，不等待或运行D2包络。
- 379403_1：D2包络继续训练，01:13约7696更新、best7200／1.68269；完成best的E测试及CPU评价后保留资源等待，不追加实验。

## 命令与执行证据

正式命令为`bash 训练与运行/sh/train_docking.sh D1-E-T0-RA`、`bash 训练与运行/sh/sample_docking.sh B-E-T0-RA-SMILES-test`及采样成功后执行的`bash 训练与运行/sh/evaluate_docking.sh B-E-T0-RA-SMILES-test`。

本地`tmp/formal-execution-20260914/start-d1-envelope-371591.sh`与`start-envelope-test-379402.sh`仅负责资源控制，不是正式科学入口。服务器`/storage/penghongen/tmp/pxm_formal_execution_20260914/runs/<实验名>/`保存stdout、stderr和previous_run_cmd。两作业均先核对真实JobId/GPU/控制目录/旧任务正常退出，再消费父目录try_lock；after_lock、资源和所有旧产物保留。未重启其他两项训练。

## 后续纪律

维持既定科学、OOM配对降批、online W&B与四卡分工，不默认优化I/O。每次取得新进度更新逐实验及所属总日志，根索引只维护阶段；稳定后每60分钟分段等待，不设heartbeat。日志前部结果、后部尝试的组织不变。

实现分支仍为codex/formal-density-strict-ablation，Learn/CUMULATIVE基点d3c1ed5，全部正式任务完成后做双线等价收口。用户或其他代理的docking/dataset.py、models/density_backbone.py及其他无关未提交文件不审查、不提交、不混入release；本次release来自Git archive已提交端点。
