# B-E-T0-RB训练与配套测试

本记录对应既定六个无密度模型中的第6个实验。378693、gnode10的A800及16核CPU已完成第5个[T1-RB的训练、C0／C5测试和完整报告](06-B-C-T1-RB.md)，随后在同一资源从官方权重训练E-T0-RB，按实际best完成E测试及同卡CPU评价。371591继续E-RA的评价与后续第4个模型；本记录不涉及其他任务。

依据为[科学契约](../../想法/方案草稿/9-8-科学契约.md)、[工程细节](../../想法/方案草稿/9-8-工程与实现细节.md)和[边界清单](../../想法/方案草稿/9-8-边界与核查清单.md)。正式训练已从官方权重启动，主进程57306，独立W&B为423nfmpm；完整训练、best测试和评价仍须继续。

## 固定科学与训练条件

配置为 `configs/docking/B-E-T0-RB.yml`。包络E按标准受体残基重原子质量中心到任一真实配体重原子的距离严格小于10 Å选袋，模型原点为实际输入受体重原子的坐标算术均值。训练、val/loss和正式测试均为E，T0始终关闭新增整体平移。RB分别构建和编码蛋白／核酸受体图，从官方模型初始化后复制蛋白编码器参数给核酸编码器；原主体、loss、置信度头与原dock高斯链保持。

采用用户已批准的空E规则：仅训练和val/loss迭代在组批前跳过空口袋并警告记录身份，正式测试保留输入构造失败及全部分母。训练清单仍65290条，其中46个空E身份已完整审计，E有效训练实例65244；验证781条无空E。完整清单、对称排列和冻结C5向量均不重建。空E修复、46个身份和RA／RB真实训练输入验收见[日志07](07-B-E-T0-RA.md)。

训练从规定的官方 `pocketxmol.ckpt` 参数重新开始，不传--resume，不承接T1-RB或E-RA参数；优化器、调度器和W&B运行身份独立建立。配置为batch72、梯度累积1、bf16、15个数据worker，固定有效全局批量72。AdamW初始lr=1e-4、weight_decay=0.001、betas=0.99／0.999、eps=1e-8、warmup=0。每800次优化器更新沿原路径计算val/loss；Plateau相对改善阈值1%、patience5、factor0.2，第三次实际下降立即停止，上限40000步。best取所有定期检查点的最低原E val/loss，保留last和全部定期检查点。

## 来源与运行前核对

使用已修复的正式release `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_43742cdf8166/PocketXMol`，源码基线ec06dbd加空E修复c9cc4a1，与有效E-RA重训所用源码相同。该版本已完成两遍主代理自查、两轮三类独立审查、22项CPU回归以及RA／RB各8步的真实训练输入GPU验收。本次没有新增或修改生产函数，不重复扩大审查或验收范围。

配置解析核对：相对已完成E-RA有效训练的配置，只有model.nucleic_branch从RA变为RB、train.wandb.name变为B-E-T0-RB，其余字段完全一致。目标训练根已检查不存在；正式接入时再次确认，避免覆盖已有产物。只复用明确release，不复制共享工作区外来文件。

## 正式训练命令与产物

以下是本模型正式训练命令，检查脚本和产物核对命令另行记录：

```bash
bash 训练与运行/sh/train_docking.sh B-E-T0-RB
```

独立训练根为 `/storage/penghongen/PocketXMol/training/B-E-T0-RB/`，训练保存配置位于其train_config子目录，定期检查点及last位于checkpoints子目录。W&B使用pencounkdual-111/PocketXmol_raw，名称B-E-T0-RB，实际run id启动后登记。

资源控制目录仍为 `/home/penghongen/Feedback/Pocket_Plus/allocations/378693/`，在前一评价正常结束、try_lock恢复后保存新旧动态命令及启动记录，再移除try_lock开始训练。after_lock保留，不使用scancel，不删除旧训练、测试、W&B或日志。

本模型训练结束后直接用实际best完成446个实例的E测试，每实例50个候选、100步，推理batch优先50。之后用同一A800作业的8个CPU评价进程完成ALL／CAP10／HF10_TO5及核酸比例分析。明确best的采样配置和短正式命令在训练结束后登记；不安排完整验证集采样或新增实验。

## 正式接入

2026-09-13 master时间05:06:47保存新旧命令与启动记录，移除try_lock后由378693控制器第4次执行开始训练，after_lock保留。接入前再次确认E-RB目标根不存在、前一T1-RB评价正常结束且W&B上传完成，动态命令确实是本任务的T1-RB评价入口。没有使用kill_lock或修改已完成实验的产物。

本次仍使用43742cdf8166 release，正式记录提交为d86a452；训练配置SHA256为d4e03bf8a21553c2af786f3563a689e68b812ba5f58573368dd83aa8c556ef4b。实际launch为 `/home/penghongen/Feedback/PocketXMol/launches/378693/train_B-E-T0-RB_job378693_20260913T050539`。启动记录 `/storage/penghongen/PocketXMol/control/378693/train_B-E-T0-RB_start.json`，同目录保存train_B-E-T0-RB_run_cmd.sh和修改前的train_B-E-T0-RB_before_run_cmd.sh。累计out／err起点为25869442／59362，后续仅以这一区间记录本次训练。

实际主进程57306及15个数据worker均在job_378693，没有--resume；日志确认从规定官方pocketxmol.ckpt加载模型，bf16及全部模型参数参与训练。新W&B运行是[423nfmpm](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/423nfmpm)，本地记录位于训练根wandb/run-20260913_050609-423nfmpm。最初更新的原损失与置信度损失均有限，且6j40/273、6j3z/22等空E实例已有明确跳过警告。

正式启动检查推进至109次优化器更新，约1.04步／秒，lr=1e-4；日志没有非有限损失、Traceback、OOM或reduce_batch，已记录6个不同空E训练身份。这里只确认正式训练正常开始，完整训练和best选择仍须继续。

## 计划与实现差异

本实验沿用已批准的E-T0-RB配置与空E规则，没有新增科学条件。正式训练已接入，完整训练、测试及评价仍待完成。
