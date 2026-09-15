# B-E-T0-RA-SMILES

当前状态（2026-09-15 19:07／19:09）：B-E-T0-RA-SMILES：8113更新，best5600／约1.77278，last8000；lr=1e-4，W&B 2uv4mdnt online，尚无测试。

| 项目 | 当前内容 |
|---|---|
| 资源 | 379402_0，实际JobId379402，gnode09，A800，16CPU |
| 配置 | configs/docking/B-E-T0-RA-SMILES.yml |
| 产物根 | /storage/penghongen/PocketXMol/training/B-E-T0-RA-SMILES |
| 测试协议 | E；每实例每协议50候选、100步，batch50 |
| best／W&B／指标 | B-E-T0-RA-SMILES：8113更新，best5600／约1.77278，last8000；lr=1e-4，W&B 2uv4mdnt online，尚无测试 |

所有新训练从规定官方参数初始化，重建优化器与调度状态；训练bf16-mixed，推理官方FP32张量路径。训练保留原val/loss与best选择，结束后直接完整测试，中心C0/C5、包络E；既有清单、视图、C5和种子不重建，每实例每协议50候选、100步，推理优先batch_size=50。W&B保持online，无法在线时报告，不自行切换offline。 无密度和D1起始72×1，D4/D2/D3起始36×2；OOM按72→36→24降批、累积相应1→2→3，配置global_batch_size保持72。接受原reduce_batch偶发裁批和累积梯度丢失，不增加补样或梯度补偿。仍无法运行则报告实际问题。

## 正式运行命令

以下命令已于16:58:48派发；17:02:10正式开始，W&B 2uv4mdnt online。

```bash
bash 训练与运行/sh/train_docking.sh B-E-T0-RA-SMILES
```

训练后依据实际best建立测试配置，记录短采样与评价命令，不使用旧模型best，也不完整验证集采样。

## 只读核查与验收依据

前置验收已被接受，不重跑。启动前核对真实作业、GPU UUID、控制目录、after_lock及try_lock；原命令独立备份。

启动前核查（16:57）：379402仅控制器等待，实际GPU UUID GPU-adbf8fc8-5a4a-87e3-853b-c9cadcbdf74b，显存5 MiB、利用率0%；after_lock及父目录try_lock存在，kill_lock不存在。本包络训练目录尚不存在，已验收配置为E、72×1、global72、bf16-mixed、W&B online及规定官方初始权重；不传resume，不复用中心或旧包络状态。正式执行沿用已验收release e6173e5c817d（来源0c79d53），保留原控制命令及全部前序产物。

## 本次正式启动记录

2026-09-15 16:58:48（gnode09）备份原控制命令后仅消费父目录try_lock，after_lock和所有中心模型产物保留。正式命令为`bash 训练与运行/sh/train_docking.sh B-E-T0-RA-SMILES`，72×1、bf16-mixed、原E训练与val/loss，从规定官方参数新初始化优化器和调度状态。没有传resume，也没有修改已验收科学配置。

冻结release为`/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_e6173e5c817d/PocketXMol`（来源0c79d53）。本次控制脚本为`tmp/formal-execution-20260914/start-envelope-379402.sh`，日志与原命令副本在`/storage/penghongen/tmp/pxm_formal_execution_20260914/runs/B-E-T0-RA-SMILES/`，分别为train.out、train.err、previous_run_cmd.sh。实际launch为`/home/penghongen/Feedback/PocketXMol/launches/379402/formal_B-E-T0-RA-SMILES_0c79d53`，已由正式stdout确认；控制脚本属于资源证据，不与正式短训练命令混记。

## 之前的准备与尝试

以下为旧分工下尚未启动正式复验时的准备记录，其中378693接管Matcher是当时已发生的事实；当前职责已改为379402。旧编码实验结果继续留在[原历史日志](../第一类实验（不加密度信息）/3-B-E-T0-RA.md)。

最新执行边界：前置集成及验收收口后先汇报，正式复验等待用户再次明确允许；本次新运行尚未启动。
**当前：修正配体芳香键编码的复验已获准，前置验收完成，等待用户再次允许；尚未开始训练。** 本次从官方pxm权重重新初始化模型、优化器和调度状态，采用公共精确SMILES图，不恢复旧模板编码的checkpoint。资源为378693／gnode10／A800／16CPU；训练bf16-mixed，推理官方FP32张量路径。

| 当前复验字段 | 已确定内容 |
|---|---|
| 配置 | configs/docking/B-E-T0-RA-SMILES.yml |
| 训练产物根 | /storage/penghongen/PocketXMol/training/B-E-T0-RA-SMILES |
| 测试协议 | E；每实例每协议50候选、100步 |
| best／W&B／测试指标 | 尚未产生；旧值见后半部分，不充当本次结果 |

## 正式运行命令

用户再次明确允许后执行的短入口为 `bash 训练与运行/sh/train_docking.sh B-E-T0-RA-SMILES`，当前未执行。训练结束读取实际best后分别记录采样和评价命令，不生成完整验证集候选。

## 验收与资源控制记录

2026-09-14 19:52，主代理在gnode10确认378693正在执行Matcher的smiles_identity.experiment，进程组13179，随后按授权创建实际kill_lock。控制器已终止该进程组并回到try_lock；after_lock保留，GPU0空闲。接管证据根为/storage/penghongen/tmp/pxm_formal_smiles_20260914/takeover-378693。接管不删除Matcher文件或checkpoint，原运行命令另有副本。新链验收命令后续记录于[SMILES日志](../预实验（一 --二之间）/1-SMILES构图与GPU验收.md)。

启动确认（17:03）：stdout核对官方初始权重与本次launch，stderr确认bf16-mixed、在线W&B，31次更新损失有限。控制器准备运行副本约3分钟后正常启动，不属于模型I/O退化，未改变底层控制器或其他项目。沿用原空E口袋跳过规则，进入60分钟分段等待。

核查（2026-09-15 18:05／18:07）：B-E-T0-RA-SMILES：4048更新，best2400／约1.81572，last4000；lr=1e-4，W&B 2uv4mdnt online，尚无测试。三项训练持续产生有限损失，累计OOM均0；在线W&B持续记录。D1测试正常生成，无失败实例，未见明确I/O退化。继续60分钟分段等待，不改科学配置。

核查（2026-09-15 19:07／19:09）：B-E-T0-RA-SMILES：8113更新，best5600／约1.77278，last8000；lr=1e-4，W&B 2uv4mdnt online，尚无测试。三项训练持续产生有限损失，累计OOM均0；在线W&B持续记录。D1测试正常生成，无失败实例，未见明确I/O退化。继续60分钟分段等待，不改科学配置。
