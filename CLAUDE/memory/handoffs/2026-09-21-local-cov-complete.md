# Handoff: `local_cov` 正式实验与双线Git收口

Date: 2026-09-21

## Current State

`local_cov`中心与包络模型已经完成实现、审查、A800真实数据门控、正式训练、best完整测试、CPU评价和在线W&B记录。两条正式任务都使用不可变release`/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_ef9dd39/PocketXMol`；378693和379402均保留`after_lock`并回到`try_lock`等待，没有后续任务。

实现分支为`codex/local-cov`，学习分支为`Learn/local-cov`，共同基点为`d3c1ed548bff97dfe61ab4c052d0c53c47029f2f`。运行内容核验端点分别为`4154e01e320c57cd019278aaa1d8b4d2048fb9af`和`cee633132a4b1a16218eab6fd1c0606ab2b8b2cf`，完整tree均为`44434eefe40667e88d3b7433d8a1d06acd6efc1f`。

## Completed

- 实现80³外围密度裁块、固定106通道网格、六个去噪块逐层11³读取、六套局部卷积、共享LayerNorm和六套零初始化FiLMPlus。
- 删除第一次密度实验的D1至D4专属网络、配置和测试入口，保留RA＋T0、通用56维ALL通道、密度I/O和坐标几何。
- 本地`tests/test_local_cov.py`两端各9项通过；正式服务器环境25项数据、模型和局部模块测试通过；72×1真实A800门控包含优化器更新、验证、检查点恢复和小规模正式采样评价。
- 中心模型best为`step=24800.ckpt`、`val/loss=1.61287522315979`。C0／C5的ALL逐实例等权Top-1分别为73.77%／64.13%，评价W&B为`aqp4pl9u`。
- 包络模型best为`step=24800.ckpt`、`val/loss=1.4366419315338135`。E的ALL逐实例等权Top-1为78.03%，评价W&B为`okdl1eao`。
- D2包络既有任务也已完成训练、best E测试、CPU评价和第一次密度实验日志收口；D3、D4没有启动。
- `Learn/local-cov`按概念依赖重建，运行内容端点与实现线tree完全相同；`Learn/CUMULATIVE`在等价核验后快进。

## Decisions

- 后续科学构造仍固定RA＋T0；`local_cov`没有扩展为整块U-Net、全局注意力、`global_attn`、`selected`或额外评分器。
- 正式训练使用72×1、全局批量72、局部卷积分块4096，没有触发2048或24×3回退。
- 首次36×2运行没有验证检查点的原因是误读microbatch进度并提前停止，不是验证系统故障；失败运行的源码、W&B和产物均保留。
- 不删除服务器历史产物，不释放三张保留资源，不安排额外实验。

## Open Questions

当前授权范围没有阻塞项。若未来需要在中心C0/C5、包络E或第一次密度实验之间选择主模型，应由新的研究决定启动，不在本轮自动增加实验。

## Next Actions

新的实现工作应从最终`Learn/CUMULATIVE`开始。继续研究前先阅读三份`local_cov`日志中的完整指标、训练停止点和失败历史，并重新确认用户届时授权的模型与资源范围。

## Files To Reopen

- `想法/方案草稿/9-15.md`
- `日志/第三次正式实验/local_cov/本轮实验日志.md`
- `日志/第三次正式实验/local_cov/C.md`
- `日志/第三次正式实验/local_cov/E.md`
- `日志/计划执行映射.md`
- `models/density/local_cov.py`
