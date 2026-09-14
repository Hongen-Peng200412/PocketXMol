# Handoff：修正编码的无密度中心训练完成并进入测试

日期：2026-09-15

## 当前状态

正式阶段继续执行，尚未完成全部目标。379402上的B-C-T0-RA-SMILES训练正常结束，同卡已启动best的C0/C5完整测试，随后自动CPU评价。371591继续D1中心训练，378693继续D2中心训练。最新状态以逐实验及阶段总日志为准，不以本文件替代实时记录。

## 已完成

无密度中心从官方参数新初始化，72×1、bf16-mixed，31200次更新在第三次实际学习率下降时停止，未见OOM。完整last.ckpt确认decline_count=3、stop_reason=plateau、best为step=21600.ckpt、val/loss=1.72273850440979；训练W&B miwl75au已online finished。39份定期checkpoint、last、配置、旧源码与日志保留。最终优化器lr为8e-7，W&B最后一批训练lr=4e-6来自下降前，下降后未继续更新。

新入口`configs/docking/sample-B-C-T0-RA-SMILES-test.yml`读取本次训练保存的配置和best，保持RA、精确SMILES、C0/C5、50候选、100步、batch50及原FP32路径。主代理自查、本地配置解析、一轮独立审查完成，未重复GPU预实验。提交为`3af42af2866b947fdf2c6bee0e4d1dc2ff9e17da`，测试release为`/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_ea479744bb2a/PocketXMol`。没有改动生产Python代码。

379402按真实锁协议在07:38派发，07:39:34正式启动；launch为`/home/penghongen/Feedback/PocketXMol/launches/379402/formal_B-C-T0-RA-SMILES_test_3af42af`。控制目录沿用原AdaLigand allocation，after_lock保留，仅消费父目录try_lock。首次控制脚本语法检查发现缺少引号，未启动任务；修正后才派发，失败命令和原训练命令分开留存，详见实验日志。

## 决策与下一步

资源顺序不变：371591执行D1中心→D1包络→D4中心→D4包络；378693执行D2中心→D2包络→D3中心→D3包络；379402执行无密度中心→无密度包络→官方冻结C0/C5/E。每个模型完成训练、best测试与本资源CPU评价再进入下一模型。不完整验证集采样，不跟踪其他任务。

正式测试命令为`bash 训练与运行/sh/sample_docking.sh B-C-T0-RA-SMILES-test`，其退出0后执行`bash 训练与运行/sh/evaluate_docking.sh B-C-T0-RA-SMILES-test`。测试产物位于`/storage/penghongen/PocketXMol/sampling/B-C-T0-RA-SMILES/test`；执行stdout／stderr在`/storage/penghongen/tmp/pxm_formal_execution_20260914/runs/B-C-T0-RA-SMILES-test/`。完成后核对两个协议的既定测试实例、候选预算、失败分母、评价文件及在线W&B，再启动包络新训练。

当前不优化I/O，不改科学条件；OOM仅按已批准的成对降批规则处理。未来密度测试配置等待各自真实best，不提前填旧检查点。运行稳定后按约定分段长等待，不设heartbeat。goal工具当前返回blocked，但实际执行无阻塞，已通知用户在面板恢复；继续既有授权。实现分支仍为codex/formal-density-strict-ablation，Learn/CUMULATIVE为d3c1ed5，全部正式任务完成后仍需双线等价收口。

## 重新打开的文件

- `日志/RA-T0第二次严格消融/1-B-C-T0-RA-SMILES.md`及该目录总日志。
- `日志/第二类实验（密度分支的训练）/1-D1.md`、`3-D2.md`及总日志。
- `configs/docking/sample-B-C-T0-RA-SMILES-test.yml`。
- 本地`tmp/formal-execution-20260914/`中的只读核查、发布和资源控制脚本；这些不属于正式科学运行命令。
