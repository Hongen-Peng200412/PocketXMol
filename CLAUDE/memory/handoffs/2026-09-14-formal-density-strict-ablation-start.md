# Handoff：正式密度与修正编码复验启动

日期：2026-09-14

## 当前状态

前置集成已经收口。用户接受官方输入与T0机制实质对齐及已披露GPU重复性、AdamW检查局限，不重跑整套预实验；正式goal已经由用户更新并恢复active。本节点只表示正式阶段启动，全部训练与测试尚未完成，不能关闭goal。

## 已完成

只读确认用户d3c1ed548bff97dfe61ab4c052d0c53c47029f2f仅新增两份空Markdown，且是Learn/CUMULATIVE及全仓唯一最新提交；主工作树干净，旧工作树运行缓存保留。实现分支为codex/formal-density-strict-ablation；0c79d53336f16f103227b967e596562f8a272f3e保存最新授权、资源及日志迁移，只有Markdown变更。旧第1／3／7日志历史正文与原提交一致，其复验准备记录完整迁入新目录。主代理自查、一轮独立文档审查及指定问题窄修完成，没有新生产代码改动或重复GPU检查。正式执行结束后仍须完成本轮学习线及端点等价收口，当前累计基点仍为d3c1ed5。

正式release为`/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_e6173e5c817d/PocketXMol`。来源归档与启动脚本证据位于`/storage/penghongen/tmp/pxm_formal_execution_20260914`及本地`tmp/formal-execution-20260914`。Windows归档的22个Shell文件仅恢复LF行尾后发布，其余源码未改；归档SHA-256为10cb86e3a84d71f0efa01fd6717aebe8c133d1c1488c765d342effec7a4eb741。

首批D1-C-T0-RA／371591、D2-C-T0-RA／378693、B-C-T0-RA-SMILES／379402已通过各自真实控制目录启动。保留after_lock和旧run_cmd副本，仅消费父目录try_lock。W&B分别为density项目5qmpsxrs、0t1vqqgk和raw项目miwl75au，全部online；启动确认结束时无密度中心235次更新、D1中心45次更新、D2中心17个训练批次（累积2），损失均有限且未见OOM。最新进度必须继续查下述实验日志，不能把本handoff当实时快照。

## 决策与资源

- 371591／gnode09：D1中心→D1包络→D4中心→D4包络。控制目录`/home/penghongen/Feedback/Pocket_Plus/allocations/371591`，GPU UUID `GPU-0e53cfe2-fd47-7b09-8bed-d71aa08d25a1`。
- 378693／gnode10：D2中心→D2包络→D3中心→D3包络。控制目录`/home/penghongen/Feedback/Pocket_Plus/allocations/378693`，GPU UUID `GPU-0e253751-cce3-c71d-c2df-a222fec3759b`。
- 379402_0（实际379402）／gnode09：无密度中心→包络→官方冻结对照。控制目录`/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration/运行日志与统计/slurm/allocations/379402`，GPU UUID `GPU-adbf8fc8-5a4a-87e3-853b-c9cadcbdf74b`。

每模型完成训练、原val/loss的best测试及本资源CPU评价后才进入下一模型。中心C0/C5、包络E，官方冻结对照C0/C5/E且不训练。三卡均16CPU，禁止操作无关任务，不使用scancel、不删除历史产物。

无密度／D1起始72×1，D4/D2/D3起始36×2。OOM成对降至36×2或24×3，配置global72；用户接受原裁批和累积梯度丢失，不增加补偿。现有完整checkpoint续训允许改变批量和累积，不需预先重写训练算法；24仍失败则报告。训练bf16-mixed，推理官方FP32，每实例每协议50候选100步，推理batch50。训练结束直接测试，不完整验证集采样。

默认不再优化I/O，仅实际发现比此前可比训练变慢才定位；D1旧利用率目标不构成优化触发。E密度裁块使用真实配体质心，模型原点仍为受体均值；D3标签冲突无需额外统计或处理。W&B始终online，不能自行offline。用户明确增撤资源后才调整排程，完整检查点恢复允许重做未保存更新。

## 下一步

确认三模型稳定更新并维护逐实验、阶段总日志。稳定期间按约定长等待，不设置heartbeat。训练完成后读取实际best，创建对应正式测试配置并执行短采样、评价命令；不能预填旧best。所有正式命令与核查／验收命令分开，失败和恢复留在同一实验日志后部。阶段完成或明确分叉才追加handoff。

## 重新打开的文件

- `日志/RA-T0第二次严格消融/总日志.md`及三份逐实验日志。
- `日志/第二类实验（密度分支的训练）/总日志.md`及四份D_i日志。
- `AGENTS.md`、三份`想法/方案草稿/9-8-*.md`、`日志/计划执行映射.md`。
