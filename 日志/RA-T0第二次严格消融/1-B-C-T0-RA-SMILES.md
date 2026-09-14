# B-C-T0-RA-SMILES

当前状态（2026-09-15，gnode09 06:23／gnode10 06:25）：29116次更新，best21600／val/loss约1.72274，last28800；lr=4e-6（已下降两次），W&B miwl75au online；尚无测试结果。

| 项目 | 当前内容 |
|---|---|
| 资源 | 379402_0，实际JobId379402，gnode09，A800，16CPU |
| 配置 | configs/docking/B-C-T0-RA-SMILES.yml |
| 产物根 | /storage/penghongen/PocketXMol/training/B-C-T0-RA-SMILES |
| 测试协议 | C0/C5；每实例每协议50候选、100步，batch50 |
| best／W&B／指标 | 29116次更新，best21600／val/loss约1.72274，last28800；lr=4e-6（已下降两次），W&B miwl75au online；测试未开始 |

所有新训练从规定官方参数初始化，重建优化器与调度状态；训练bf16-mixed，推理官方FP32张量路径。训练保留原val/loss与best选择，结束后直接完整测试，中心C0/C5、包络E；既有清单、视图、C5和种子不重建，每实例每协议50候选、100步，推理优先batch_size=50。W&B保持online，无法在线时报告，不自行切换offline。 无密度和D1起始72×1，D4/D2/D3起始36×2；OOM按72→36→24降批、累积相应1→2→3，配置global_batch_size保持72。接受原reduce_batch偶发裁批和累积梯度丢失，不增加补样或梯度补偿。仍无法运行则报告实际问题。

## 正式运行命令

以下训练命令已经执行，实际release、launch和开始时间见下文。

```bash
bash 训练与运行/sh/train_docking.sh B-C-T0-RA-SMILES
```

训练后依据实际best建立测试配置，记录短采样与评价命令，不使用旧模型best，也不完整验证集采样。

## 只读核查与验收依据

前置验收已被接受，不重跑。启动前核对真实作业、GPU UUID、控制目录、after_lock及try_lock；原命令独立备份。


## 本次正式启动记录

2026-09-14 22:02:58（gnode09本机UTC+8时间），沿作业379402真实控制目录保存旧run_cmd后，仅消费父目录try_lock启动；after_lock保留。当前批量72×1，官方初始参数、优化器与调度状态均从新训练开始。

- 正式命令：`bash 训练与运行/sh/train_docking.sh B-C-T0-RA-SMILES`。
- 实现提交：`0c79d53336f16f103227b967e596562f8a272f3e`；本次仅文档变更，源码与前置验收实现不变。
- release：`/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_e6173e5c817d/PocketXMol`。
- 实际launch：`/home/penghongen/Feedback/PocketXMol/launches/379402/formal_B-C-T0-RA-SMILES_0c79d53`，已由正式stdout确认。
- 本次stdout／stderr：`/storage/penghongen/tmp/pxm_formal_execution_20260914/runs/B-C-T0-RA-SMILES/train.out`及`train.err`。
- 旧控制命令：同目录`previous_run_cmd.sh`；新命令本地来源`tmp/formal-execution-20260914/start-379402.sh`。

服务器各节点时钟有差异，保留各自原始时间，不跨节点相减计算耗时。

## 之前的准备与尝试

以下为旧分工下尚未启动正式复验时的准备记录，其中378693接管Matcher是当时已发生的事实；当前职责已改为379402。旧编码实验结果继续留在[原历史日志](../第一类实验（不加密度信息）/1-B-C-T0-RA.md)。

最新执行边界：前置集成及验收收口后先汇报，正式复验等待用户再次明确允许；本次新运行尚未启动。
**当前：修正配体芳香键编码的复验已获准，前置验收完成，等待用户再次允许；尚未开始训练。** 本次从官方pxm权重重新初始化模型、优化器和调度状态，采用公共精确SMILES图，不恢复旧模板编码的checkpoint。资源为378693／gnode10／A800／16CPU；训练bf16-mixed，推理官方FP32张量路径。

| 当前复验字段 | 已确定内容 |
|---|---|
| 配置 | configs/docking/B-C-T0-RA-SMILES.yml |
| 训练产物根 | /storage/penghongen/PocketXMol/training/B-C-T0-RA-SMILES |
| 测试协议 | C0/C5；每实例每协议50候选、100步 |
| best／W&B／测试指标 | 尚未产生；旧值见后半部分，不充当本次结果 |

## 正式运行命令

用户再次明确允许后执行的短入口为 `bash 训练与运行/sh/train_docking.sh B-C-T0-RA-SMILES`，当前未执行。训练结束读取实际best后分别记录采样和评价命令，不生成完整验证集候选。

## 验收与资源控制记录

2026-09-14 19:52，主代理在gnode10确认378693正在执行Matcher的smiles_identity.experiment，进程组13179，随后按授权创建实际kill_lock。控制器已终止该进程组并回到try_lock；after_lock保留，GPU0空闲。接管证据根为/storage/penghongen/tmp/pxm_formal_smiles_20260914/takeover-378693。接管不删除Matcher文件或checkpoint，原运行命令另有副本。新链验收命令后续记录于[SMILES日志](../预实验（一 --二之间）/1-SMILES构图与GPU验收.md)。

启动确认：实际训练子进程开始于本节点22:03:54，stdout确认规定官方初始权重，stderr确认bf16-mixed及W&B online；没有继承预实验或旧模型状态。

最近核查：2026-09-14 gnode09 22:08:03／gnode10 22:09:41，三模型均持续产生有限训练损失，stdout／stderr未见OOM或异常退出；暂未产生首个验证检查点。已进入正常运行观察。

运行检查（23:09／23:11）：三模型累计OOM消息均为0，损失有限且学习率仍1e-4。W&B远端API确认三个run均running并持续收到训练指标。D1／D2／无密度中心显存快照分别75367／73531／61375 MiB，GPU利用率71%／100%／66%（单次快照，非窗口均值）。D1含验证平均约2.06秒／更新，未见相对前置约1.97秒明显退化；D2此时正在计算，不因利用率目标另做优化。已保存D1与无密度完整定期检查点及last；D2未到首个验证属于正常进度。只读命令包括目标GPU nvidia-smi、日志tail与OOM计数、checkpoint目录核查；在线指标脚本为tmp/formal-execution-20260914/read_wandb.sh，不是正式运行命令。

运行核查（2026-09-15 00:12／00:14）：三个run远端均running且持续接收指标，累计OOM为0，学习率均1e-4。D2已保存首次800更新完整检查点及last；D1 last3200，无密度中心last8000。GPU显存快照D1／D2／无密度分别75375／75687／61379 MiB，瞬时利用率99%／51%／77%。训练损失有限，吞吐未见下降趋势，未改变配置或进行I/O优化。继续60分钟分段等待。

核查（2026-09-15 01:14／01:16）：12046次更新，best10400／val/loss约1.95986，last12000，W&B miwl75au online。训练损失有限、学习率1e-4，累计OOM=0。W&B远端run为running并持续收到新指标。D1／D2／无密度中心显存75377／75687／63063 MiB，瞬时GPU利用率77%／54%／80%；吞吐维持原水平，未修改I/O或科学配置。继续60分钟分段等待。

核查（2026-09-15，gnode09 02:16／gnode10 02:18）：15740次更新，best10400／val/loss约1.95986，last15200；首次降学习率至2e-5后继续，W&B miwl75au online。三模型OOM累计0，损失有限，在线W&B持续收到指标；D1／D2／无密度显存75377／75687／63063 MiB，瞬时利用率100%／81%／75%。无密度第一次下降符合现行plateau规则，前两次下降后继续训练；没有手动改配置或提前停止。吞吐无明显退化，不进行I/O搜索。

核查（2026-09-15，gnode09 03:18／gnode10 03:20）：19152次更新，best10400／val/loss约1.95986，last18400；lr=2e-5（已下降一次），W&B miwl75au online。三模型OOM累计0，损失有限，W&B远端running且指标更新。D1／D2／无密度显存75377／75687／63063 MiB，瞬时利用率0%／100%／100%；单点利用率不等同于持续停顿，训练进度持续增加。没有观察到明确I/O退化，不修改配置，继续60分钟分段等待。

核查（2026-09-15，gnode09 04:20／gnode10 04:21）：22483次更新，best21600／val/loss约1.72274，last22400；lr=2e-5（已下降一次），W&B miwl75au online。三个W&B run均running并收到近期指标，累计OOM=0，训练损失有限。D1／D2／无密度显存75383／75687／63063 MiB，瞬时利用率0%／100%／100%；D1进度持续增长，不将单点利用率视为I/O异常。无手动配置修改，继续60分钟等待。

核查（2026-09-15，gnode09 05:22／gnode10 05:23）：25812次更新，best21600／val/loss约1.72274，last25600；lr=2e-5（已下降一次），W&B miwl75au online。三项W&B均running且指标更新，累计OOM=0，训练损失有限；D1／D2／无密度显存80261／75687／63063 MiB，瞬时利用率42%／100%／100%。未因显存缓存量自行降批，配置保持获准值；未见明确I/O退化，继续60分钟分段等待。

核查（2026-09-15，gnode09 06:23／gnode10 06:25）：29116次更新，best21600／val/loss约1.72274，last28800；lr=4e-6（已下降两次），W&B miwl75au online。三个W&B run均running且指标更新，累计OOM=0、训练损失有限。D1／D2／无密度显存80261／75687／63071 MiB，瞬时利用率51%／32%／73%。无密度第二次降lr后继续，第三次实际下降才停止；当前未提前停止、未选择测试候选。保持科学配置及60分钟分段等待。
