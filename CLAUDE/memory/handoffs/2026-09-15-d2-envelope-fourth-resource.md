# Handoff：D2包络移交第四张A800

日期：2026-09-15

## 当前状态

用户明确新增379403_1／gnode10，实际JobId379403、ArrayTaskId1，独立承担D2-E-T0-RA训练、best完整E测试及本卡CPU评价；完成后保留after_lock和资源等待，不安排其他任务。D2中心和包络共用既有3-D2.md，不新建实验日志。goal已恢复active且包含最新四卡分工。

## 已完成

只读核对真实Slurm资源、控制器PID3960、16CPU、物理GPU1（UUID GPU-f8f2f6b7-5bde-624a-fd2d-c0f4776730c9）。接管前GPU空闲，after_lock及父目录try_lock存在，旧校准入口已自行返回等待，D2-E训练产物目录不存在。

AGENTS、三份契约、运行README、计划执行映射及日志已更新，提交42a3a32只变更文档与状态。未修改生产Python或已验收配置，不重跑GPU预实验。D2-E沿用冻结release `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_e6173e5c817d/PocketXMol`，源码来源0c79d53。

11:34:18保存旧run_cmd后消费try_lock，after_lock保留；11:35:35实际正式启动。stdout确认读取规定官方pxm参数，stderr确认bf16-mixed、W&B online，run为`pencounkdual-111/PocketXmol_density/psr0c3ok`。起始36×2、global72；没有继承旧训练状态。launch为`/home/penghongen/Feedback/PocketXMol/launches/379403/formal_D2-E-T0-RA_0c79d53`。日志前部继续记录最新进度，不把本handoff当实时状态。

## 决策与下一步

- 371591：D1中心→D1包络→D4中心→D4包络。
- 378693：D2中心→D3中心→D3包络，不再运行D2包络，也不等待D2包络。
- 379403_1（实际379403）：仅D2包络全流程，完成后保留等待。
- 379402_0（实际379402）：无密度中心→包络→官方冻结对照。

每个模型结束训练后依据原val/loss选择真实best，中心C0/C5、包络E，50候选、100步、batch50，随后同卡CPU评价；不完整验证集采样。所有其它科学、OOM、I/O及在线W&B边界不变。新增资源不增加实验数量，D2-E仅本次一份独立运行。

启动确认期间，无密度中心C0已完成446实例、22300候选均成功，C5开始并完成前23实例；D1中心24285更新、best20800，D2中心15820训练批次（累积2）、best5600。这些是11:33／11:35快照，后续从实验日志读取最新状态。

## 重新打开的文件

- `日志/第二类实验（密度分支的训练）/3-D2.md`和总日志。
- `日志/RA-T0第二次严格消融/1-B-C-T0-RA-SMILES.md`和总日志。
- 本地资源控制脚本`tmp/formal-execution-20260914/start-d2-envelope-379403.sh`。
- 远端运行证据根`/storage/penghongen/tmp/pxm_formal_execution_20260914/runs/D2-E-T0-RA/`，含train.out、train.err、previous_run_cmd.sh。

正式命令只有`bash 训练与运行/sh/train_docking.sh D2-E-T0-RA`；只读scontrol、nvidia-smi与锁核查单列于实验日志。仍在实现分支codex/formal-density-strict-ablation，最终全部正式目标完成后再按既定双线Git收口。
