# Handoff: 第一次密度实验收口，`local_cov`正式训练继续

Date: 2026-09-18

## Current State

第一次密度实验实际执行的D1、D2中心与包络四个模型已经完成训练、best正式测试、CPU评价和日志收口；D3、D4按用户决定未启动。`local_cov`中心和包络继续使用同一不可变release`/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_ef9dd39/PocketXMol`正式训练。

2026-09-18 03:23，`local_cov`中心约5532次优化器更新，当前best为`step=4800.ckpt`、`val/loss=1.9076017141342163`；包络约4329次更新，当前best为`step=4000.ckpt`、`val/loss=1.650253415107727`。两条训练均无OOM、Traceback或非有限值。

## Completed

- D2包络训练在28800次优化器更新时正常停止，best为`step=25600.ckpt`、`val/loss=1.5142250061035156`。
- D2包络E推理完成446个实例、22300个候选，全部成功。
- D2包络CPU评价完成。ALL／CAP10／HF10_TO5 Top-1为70.63%／63.97%／64.76%；评价W&B `mf61xk6o`为`online_completed`，error为null。
- 379403保留`after_lock_379403`并返回父目录`try_lock_379403`，不再安排任务。
- `local_cov`首次36×2运行被提前停止的原因已确认：误把microbatch进度当作优化器更新步。原验证系统没有缺陷；当前72×1运行已在800、1600及后续间隔正常验证和保存检查点。

## Decisions

- D2包络完成后不向379403追加任务。
- `local_cov`继续72×1、配置全局批量72及全部既定科学参数，不因节点瞬时负载中断或改动训练。
- 每60或90分钟以多个300秒片段等待；取得新进度时同步逐实验日志与所属总日志。

## Next Actions

1. 继续监看378693中心与379402包络训练，核对每800次优化器更新的`val/loss`、best、学习率下降和停止原因。
2. 每个模型正常结束后从完整`last.ckpt`核对最终状态，使用真实best完成规定测试和CPU评价：中心C0／C5，包络E。
3. 完成三份`local_cov`日志、根阶段索引与有意义handoff。
4. 两个模型全流程结束后按双线Git规则从共同基点重建`Learn/local-cov`，验证端点等价，再推进`Learn/CUMULATIVE`。

## Files To Reopen

- `日志/第三次正式实验/local_cov/本轮实验日志.md`
- `日志/第三次正式实验/local_cov/C.md`
- `日志/第三次正式实验/local_cov/E.md`
- `日志/第二类实验（密度分支的训练）/3-D2.md`
- `想法/方案草稿/9-15.md`
- `训练与运行/README.md`
