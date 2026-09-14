# B-C-T0-RA-SMILES

当前状态（2026-09-14）：已获正式执行授权，尚未启动；中心模型优先启动。

| 项目 | 当前内容 |
|---|---|
| 资源 | 379402_0，实际JobId379402，gnode09，A800，16CPU |
| 配置 | configs/docking/B-C-T0-RA-SMILES.yml |
| 产物根 | /storage/penghongen/PocketXMol/training/B-C-T0-RA-SMILES |
| 测试协议 | C0/C5；每实例每协议50候选、100步，batch50 |
| best／W&B／指标 | 尚未产生 |

所有新训练从规定官方参数初始化，重建优化器与调度状态；训练bf16-mixed，推理官方FP32张量路径。训练保留原val/loss与best选择，结束后直接完整测试，中心C0/C5、包络E；既有清单、视图、C5和种子不重建，每实例每协议50候选、100步，推理优先batch_size=50。W&B保持online，无法在线时报告，不自行切换offline。 无密度和D1起始72×1，D4/D2/D3起始36×2；OOM按72→36→24降批、累积相应1→2→3，配置global_batch_size保持72。接受原reduce_batch偶发裁批和累积梯度丢失，不增加补样或梯度补偿。仍无法运行则报告实际问题。

## 正式运行命令

以下命令已获准，尚未执行；启动后补充实际release、launch和时间。

```bash
bash 训练与运行/sh/train_docking.sh B-C-T0-RA-SMILES
```

训练后依据实际best建立测试配置，记录短采样与评价命令，不使用旧模型best，也不完整验证集采样。

## 只读核查与验收依据

前置验收已被接受，不重跑。启动前核对真实作业、GPU UUID、控制目录、after_lock及try_lock；原命令独立备份。

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
