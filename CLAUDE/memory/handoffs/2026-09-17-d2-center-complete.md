# Handoff：D2中心完成，仅剩D2包络

日期：2026-09-17

## 当前状态

D2中心的best11200已完成C0/C5测试及CPU评价。各协议446实例、22300候选全部成功；ALL、实例等权、原self-ranking Top-1成功率为63.23%／33.41%。评价W&B bo5q8stp为online_completed，error=null。CPU评价05:55:08—07:33:52完成，完整结果已回填原D2日志，未重跑已完成部分。

09:29确认378693的实际控制文件after_lock_378693与父目录try_lock_378693存在，kill_lock_378693不存在；控制器43832已成功返回等待。保留资源与全部产物，不再追加任务。371591与379402在前次完成核查后同样保留等待。

## 剩余工作

仅379403_1（实际379403）上的D2包络继续：09:27为51401微批、约25700优化器更新，best及last25600，best val/loss=1.5142250061035156，lr=4e-6，OOM0；W&B psr0c3ok在线。训练完成后读取实际best与停止状态，按已批准入口创建best E测试配置，完成必要自查和独立审查，再正式测试及本资源CPU评价，最后保留after_lock并回到try_lock。不要提前假定当前best即最终best。

随后汇总本轮结果并完成双线Git收口。禁止D3、D4、local_cov或任何额外实验；不重复GPU前置验收或已完成实验。保护外部暂存9-15.md及其他代理改动；提交仅使用git commit --only显式路径，发布仅用已提交archive。

## 文件入口

- [D2实验与结果](../../../日志/第二类实验（密度分支的训练）/3-D2.md)
- [密度总日志](../../../日志/第二类实验（密度分支的训练）/总日志.md)
- [项目索引](../../../日志/总日志.md)
- [先前完成部分](2026-09-17-d1-and-strict-ablation-complete.md)

服务器完整汇总：/storage/penghongen/PocketXMol/sampling/D2-C-T0-RA/test/summary.json；本地只读副本：tmp/formal-execution-20260914/D2-C-T0-RA-summary.json。正式短命令仍保存在D2原日志，核查命令与发布脚本不作为正式命令。
