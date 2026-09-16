# Handoff：两项无密度复验完成，官方冻结对照开始

2026-09-16。总goal仍在执行，全部实验尚未完成；最新运行状态见各实验日志和所属总日志。

## 已完成

B-C-T0-RA-SMILES与B-E-T0-RA-SMILES均已完成官方参数初始化训练、最低val/loss的best测试、本资源CPU评价和W&B在线记录。中心C0/C5 Top-1分别60.76%／39.91%；包络E Top-1=63.00%、Top-5=79.15%、Oracle=89.69%，均为ALL实例等权、RMSD<2 Å、原self-ranking。两模型所有协议候选生成和RMSD评价均成功，原始结果、全部checkpoint、配置与日志保留。完整三视图和PDB等权表已写入`日志/RA-T0第二次严格消融/`对应日志。

包络best21600、val/loss1.5245567560195923；训练31200更新在第三次下降停止，训练W&B 2uv4mdnt已结束。E测试05:05:03结束，CPU评价05:58:22结束，评价W&B t5ckbgr9确认online_completed、error=null。完整结果在`/storage/penghongen/PocketXMol/sampling/B-E-T0-RA-SMILES/test/summary.json`。

## 已启动的官方对照

379402实际JobId不变、ArrayTaskId0、gnode09、A800物理GPU2 UUID GPU-adbf8fc8-5a4a-87e3-853b-c9cadcbdf74b、16CPU。核对旧任务结束、after_lock及父目录try_lock存在、kill_lock不存在、新输出不存在后，08:54:38派发，08:56:06正式开始。保持官方protein-only、精确SMILES、既有冻结清单/C5/种子，每协议50候选、100步、batch50、原FP32路径；失败输入保留评价分母，未新增适配。

正式命令为`bash 训练与运行/sh/sample_docking.sh official-SMILES-test`，正常退出后自动执行`bash 训练与运行/sh/evaluate_docking.sh official-SMILES-test`。配置`configs/docking/sample-official-SMILES-test.yml`为已经批准的现有入口，无新代码修改或重复GPU验收。

使用release`/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_0beb73604a7d/PocketXMol`，来源0f09526ea210b573071c1c601881c16412b7999a；launch为`/home/penghongen/Feedback/PocketXMol/launches/379402/formal_official-SMILES_test_0f09526`。输出`/storage/penghongen/PocketXMol/sampling/official-SMILES/test`。资源控制证据在本地`tmp/formal-execution-20260914/start-official-test-379402.sh`，stdout、stderr、previous_run_cmd在服务器`/storage/penghongen/tmp/pxm_formal_execution_20260914/runs/official-SMILES-test/`，与正式短命令分开。

## 其他任务与后续

371591上的D1中心全流程已完成，D1包络正在训练；378693继续D2中心，随后D3中心→D3包络，不等待D2包络；379403_1继续独立D2包络，完成后等待。最近08:52／08:54读数：D1包络11149更新、best10400；D2中心19804更新、best11200、lr4e-6；D2包络约11945更新、best11200。三项训练无OOM，损失有限。此次W&B远端只读API查询缓慢，仍在核查，不将观察超时认定任务停止，不切换offline。

继续按原科学、online记录及四卡职责完成全部模型。稳定运行分段等待，不设heartbeat；每次新进度更新逐实验与阶段总日志。实现分支与双线收口尚未结束，保护用户和其他代理无关修改，不把工作区改动混入冻结release。
