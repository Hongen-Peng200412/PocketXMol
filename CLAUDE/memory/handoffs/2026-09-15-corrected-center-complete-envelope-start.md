# Handoff：修正编码的无密度中心全流程完成，包络接续

日期：2026-09-15

## 当前状态

正式goal仍active。B-C-T0-RA-SMILES已完成训练、best C0/C5测试、CPU评价与W&B在线上传；379402已派发B-E-T0-RA-SMILES新训练，启动加载确认以包络实验日志为准。D1中心正在371591完成best测试，D2中心与包络分别在378693和379403_1继续训练。

## 已完成结果

无密度中心best为step=21600.ckpt，训练于31200更新、第三次下降时停止。C0/C5各446实例和22300候选，所有候选成功生成、完成RMSD及排名配对，全部实例评价成功；每协议44600次模型forward完成。原self-ranking、RMSD＜2 Å、实例等权ALL结果：C0 Top-1 60.76%、Top-5 69.96%、Oracle 83.63%；C5 Top-1 39.91%、Top-5 50.45%、Oracle 62.33%。CAP10和HF10_TO5以及PDB等权指标在中心日志前部表格和完整summary中保留。

正式采样15:02:26退出0，随即同卡8进程CPU评价，16:40:40评价进程退出0。完整结果为`/storage/penghongen/PocketXMol/sampling/B-C-T0-RA-SMILES/test/summary.json`，W&B为`pencounkdual-111/PocketXmol_raw/8ift0jbh`，状态online_completed，error为null；训练run miwl75au finished。逐实例、候选、置信度、原检查点和历史产物保留。评价时RDKit的allene立体化学提示保留于stderr，没有修改原评估定义或删样本。

## 包络接续

16:57确认379402仅控制器等待，after_lock和父目录try_lock存在、kill_lock不存在，GPU UUID GPU-adbf8fc8-5a4a-87e3-853b-c9cadcbdf74b空闲，B-E-T0-RA-SMILES训练目录尚不存在。16:58:48备份旧run_cmd后消费try_lock，沿真实协议派发，不用kill_lock或scancel。控制器开始准备该次运行副本，实际训练开始时间和W&B id仍需核对日志。

正式短命令为`bash 训练与运行/sh/train_docking.sh B-E-T0-RA-SMILES`。沿已验收配置E、RA＋T0、72×1、global72、bf16-mixed、W&B online，从官方pxm参数新初始化优化器和调度状态，不传resume。正式源码沿用`/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_e6173e5c817d/PocketXMol`（来源0c79d53），不混入工作区他人对dataset.py、density_backbone.py的未提交修改。

控制脚本`tmp/formal-execution-20260914/start-envelope-379402.sh`；远端执行证据根`/storage/penghongen/tmp/pxm_formal_execution_20260914/runs/B-E-T0-RA-SMILES/`，包含train.out、train.err、previous_run_cmd.sh。训练输出`/storage/penghongen/PocketXMol/training/B-E-T0-RA-SMILES`。本卡完成包络训练、best E测试和CPU评价后，再进行官方冻结C0/C5/E测试评价，官方不训练。

## 其余资源及下一步

371591继续D1中心测试→CPU评价→D1包络→D4中心→D4包络；378693继续D2中心全流程→D3中心→D3包络，不等待D2包络；379403_1仅完成D2包络全流程后保留等待。每模型必须完成自己的测试与评价后才进入后续模型，不增加实验。D1在16:56已有119个C0实例、5950候选生成成功；D2中心约11047更新、首次降lr至2e-5；D2包络3122更新、best2400。这些为阶段快照，后续读取实验日志最新状态。

继续观察实际进度、在线W&B、OOM与明确I/O退化；正常时按60分钟分段静默等待，不设置heartbeat。完整检查点恢复与降批沿现行授权，模型科学定义不变。当前实现分支codex/formal-density-strict-ablation，Learn/CUMULATIVE仍为d3c1ed5，全部正式目标完成后再双线等价收口。

## 重新打开的文件

- `日志/RA-T0第二次严格消融/1-B-C-T0-RA-SMILES.md`、`2-B-E-T0-RA-SMILES.md`、总日志。
- `日志/第二类实验（密度分支的训练）/1-D1.md`、`3-D2.md`、总日志。
- `tmp/formal-execution-20260914/B-C-T0-RA-SMILES-summary.json`为远端完整汇总的只读本地副本，填表脚本已核对预算、forward次数及成功分母。
