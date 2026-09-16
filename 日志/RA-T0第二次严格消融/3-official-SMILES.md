# official-SMILES

当前状态（2026-09-16 08:56）：官方冻结权重C0/C5/E测试于08:56:06在379402正式开始；完成后自动执行本资源CPU评价，不训练。

| 项目 | 当前内容 |
|---|---|
| 资源 | 379402_0，实际JobId379402，gnode09，A800，16CPU |
| 配置 | configs/docking/sample-official-SMILES-test.yml |
| 冻结权重 | /storage/penghongen/PocketXMol_official_test/extracted/data/trained_models/pxm/checkpoints/pocketxmol.ckpt |
| 正式源码 | /home/penghongen/Feedback/PocketXMol/releases/PocketXMol_0beb73604a7d/PocketXMol；来源0f09526ea210b573071c1c601881c16412b7999a |
| 产物根 | /storage/penghongen/PocketXMol/sampling/official-SMILES/test |
| 测试协议 | C0/C5/E；每实例每协议50候选、100步，batch50 |
| best／W&B／指标 | official-SMILES：379402上08:54:38已派发C0/C5/E测试及后续CPU评价；官方冻结权重，不训练 |

仅使用规定官方冻结参数，不训练。

## 正式运行命令

以下命令已于2026-09-16 08:54:38派发；采样正常退出后自动进行CPU评价。

```bash
bash 训练与运行/sh/sample_docking.sh official-SMILES-test
bash 训练与运行/sh/evaluate_docking.sh official-SMILES-test
```

## 只读核查与验收依据

前置验收已被接受，不重跑。启动前核对真实作业、GPU UUID、控制目录、after_lock及try_lock；原命令独立备份。

本次复用已批准官方配置，没有新代码修改或重复GPU验收。只读解析核对protein分支、C0/C5/E、精确SMILES资产、既有冻结清单、50候选、100步、batch50、8CPU及W&B online；规定官方权重和train_config存在，输出目录此前不存在。官方受体输入保持protein-only，纯核酸等特殊实例沿既定流程忠实记录失败和评价分母，不另建模型适配。

## 本次正式执行记录

实际启动确认（08:56）：stdout记录开始时间08:56:06、job379402、冻结release及上述launch，stderr暂无错误。此次只使用官方权重推理，未建立或恢复训练优化器状态。

08:53核对实际JobId379402、ArrayTaskId0、gnode09、16CPU及A800 UUID GPU-adbf8fc8-5a4a-87e3-853b-c9cadcbdf74b，显存5MiB，控制器2383等待；after_lock和父目录try_lock存在、kill_lock不存在。原包络采样与评价已正常退出，完整结果及online_completed状态已记录。

08:54:38按`tmp/formal-execution-20260914/start-official-test-379402.sh`派发，只消费父目录try_lock，保留after_lock和资源。该脚本为资源控制证据，独立于上述正式短命令。原run_cmd备份于`/storage/penghongen/tmp/pxm_formal_execution_20260914/runs/official-SMILES-test/previous_run_cmd.sh`；同目录保存sample.out、sample.err及随后evaluate.out、evaluate.err。使用前项发布的冻结release 0beb73604a7d，不包含工作区无关修改；预定launch为`/home/penghongen/Feedback/PocketXMol/launches/379402/formal_official-SMILES_test_0f09526`。

## 之前的准备与尝试

以下为旧分工下尚未启动正式复验时的准备记录，其中378693接管Matcher是当时已发生的事实；当前职责已改为379402。旧编码实验结果继续留在[原历史日志](../第一类实验（不加密度信息）/7-official.md)。

最新执行边界：前置集成及验收收口后先汇报，正式复验等待用户再次明确允许；本次新运行尚未启动。
**当前：官方芳香键编码下的C0/C5/E复验已获准，前置验收完成，等待用户再次允许；尚未启动。** 使用378693和原官方冻结pxm权重，不训练模型。新产物根为`/storage/penghongen/PocketXMol/sampling/official-SMILES/test`，配置为`configs/docking/sample-official-SMILES-test.yml`；新W&B及指标尚未产生。

## 正式运行命令

用户再次明确允许后依次执行`bash 训练与运行/sh/sample_docking.sh official-SMILES-test`与`bash 训练与运行/sh/evaluate_docking.sh official-SMILES-test`，当前未执行。每协议446实例、每实例50候选、100步；使用已有冻结C5和视图。

核查（2026-09-16 08:52／08:54）：official-SMILES：379402上08:54:38已派发C0/C5/E测试及后续CPU评价；官方冻结权重，不训练。两次可见核查之间远端任务持续执行，未进行重启；本次按真实节点时间记录进度。无密度包络05:58:22已完成全流程、评价W&B t5ckbgr9 online_completed；官方冻结对照08:54:38派发。三项训练无OOM，损失有限。W&B只读API本次响应缓慢，尚在等待，未切换offline或认定训练停止。
