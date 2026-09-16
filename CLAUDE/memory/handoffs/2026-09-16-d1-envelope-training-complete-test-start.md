# Handoff：D1包络训练完成，进入best E测试

日期：2026-09-16

## 当前状态

D1-E-T0-RA已按第三次学习率下降停止，共25600个优化器更新；best为step20800、原val/loss=1.629565715789795。完整last检查点保留decline_count=3、stop_reason=plateau及最终optimizer_lr=8e-7，W&B n7awsoa3在线结束。旧训练配置、32个定期检查点、last、日志和产物全部保留。

## 已完成

新增sample-D1-E-T0-RA-test.yml，沿保存的D1包络训练配置读取密度参数，E、446个冻结测试实例、50候选、100步、batch50、官方FP32、CPU8评价及W&B online。主代理核对字段消费路径和实际checkpoint，远端既有venv解析通过，独立代理一轮审查批准；不重跑已接受的GPU预实验。

配置提交c259f45d65b1e8a13e08610fd2b4a96f29979817，仅包含本次YAML。发布使用git archive，不包含用户暂存的9-15.md或未提交代码；SHA256为81cdfd0edcffc9b1ccf767dba189799319d52eb6eca87a728a7ce89405ec83cd。release为`/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_f45e559e4773/PocketXMol`，仅归一22个Shell文件为LF。归档时间戳领先节点约160秒的tar提示来自已知本机与节点时差，提取成功。

18:55:13（gnode09）通过371591真实锁协议派发E测试→CPU评价；after_lock保留，原run_cmd备份到`/storage/penghongen/tmp/pxm_formal_execution_20260914/runs/D1-E-T0-RA-test/previous_run_cmd.sh`。同目录保存sample.out、sample.err和随后evaluate.out、evaluate.err。实际开始与首实例结果应继续在D1日志更新。

## 下一步与边界

继续D1-E测试评价、D2-C测试评价、D2-E训练及best测试评价、官方冻结C0/C5/E测试评价。D1中心与两个无密度RA＋T0微调已全流程完成，不重跑。每卡完成自身范围后保留after_lock、资源和历史产物，在try_lock等待。本轮禁止D3、D4、local_cov及额外实验。

正式短命令保存在D1日志；`read_completed_d1_envelope.sh`为只读checkpoint核查，`publish-d1-envelope-test.sh`和`start-d1-envelope-test-371591.sh`为临时发布及锁控制留证，不混作正式模型命令。仍需在全部本轮实验完成后做双线Git最终收口；保护并发暂存和未提交内容，提交显式限定本任务路径。

## 应重新打开

- [D1实验日志](../../../日志/第二类实验（密度分支的训练）/1-D1.md)
- [密度总日志](../../../日志/第二类实验（密度分支的训练）/总日志.md)
- [最新范围](2026-09-16-first-formal-scope-reduced.md)
- [测试配置](../../../configs/docking/sample-D1-E-T0-RA-test.yml)
