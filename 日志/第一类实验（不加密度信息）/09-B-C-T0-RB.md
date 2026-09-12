# B-C-T0-RB训练与配套测试

本记录对应既定六个无密度模型中的第4个实验。在371591、gnode09的A800及16核CPU完成[E-RA有效训练和完整测试报告](07-B-E-T0-RA.md)后，使用同一资源训练中心T0-RB，再完成实际best的C0／C5测试、同卡CPU评价和报告。378693独立推进第6个E-RB，本记录不涉及其他任务。

依据为[科学契约](../../想法/方案草稿/9-8-科学契约.md)、[工程细节](../../想法/方案草稿/9-8-工程与实现细节.md)和[边界清单](../../想法/方案草稿/9-8-边界与核查清单.md)。本次登记明确配置和正式命令，后续按实际产物记录训练及测试结果。

## 中心T0与RB契约

配置为 `configs/docking/B-C-T0-RB.yml`。训练及原val/loss均使用C0：给定中心和模型原点C为完整配体重原子几何中心g，按标准受体残基重原子质量中心到g的距离严格小于15 Å选取完整残基。配体和受体共同减去C，局部监督目标x*的几何中心为0；只执行原高斯 `x_in=x*+s*σ(N)*ε`，s为1-level_dict['pos']，沿用原GaussianExplodePrior的尺度和逐原子噪声。center_translation=false，训练不抽新增中心偏移、不增加整分子平移。

RB分别构建和编码标准蛋白与核酸受体图，从官方权重初始化后复制蛋白编码器参数给核酸编码器。保留原主体模型、loss、置信度头及其目标、同构重分配和原固定字段恢复。

正式测试沿用446个实例的C0和冻结C5条件。T0+C5保持实际给定的偏移中心选袋及定原点，后续仍仅用T0原高斯；首步使用原纯高斯先验，不强制候选实际质心归零。定位后采样器不读取真值中心或偏移，输出只加回实际模型原点一次，不进行最终质心对齐。

## 固定训练配置与来源

源码使用 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_43742cdf8166/PocketXMol`，基线ec06dbd已含中心T0纠偏，追加c9cc4a1仅处理空E数据流，不改变中心模型。中心T0完整路径验收、原官方权重及72×1训练验收已经通过；RA／RB模型接线和既有生产代码已完成两遍自查及两轮独立审查。本次没有新增或修改生产函数，不扩大科学范围或重复验收。

配置解析核对：相对同release的B-C-T0-RA.yml，只将model.nucleic_branch改为RB、train.wandb.name改为B-C-T0-RB，其余字段一致；pocket_mode=center、center_translation=false，实际DataModule据此选C0训练与监督验证。冻结训练清单65290、验证781、校准361、测试446保持，E专用空口袋跳过规则不改变中心数据流。目标训练根已确认不存在，正式接入时再次核对。

训练从规定官方pocketxmol.ckpt参数开始，不传--resume，独立创建AdamW、调度器及W&B身份。batch72、累积1、bf16、15个数据worker，有效全局批量72。AdamW初始lr=1e-4、weight_decay=0.001、betas=0.99／0.999、eps=1e-8、warmup=0；每800次优化器更新计算原val/loss，Plateau相对阈值1%、patience5、factor0.2，第三次实际下降立即停止，上限40000步。best按最低原C0 val/loss选择，last和全部定期检查点保留。

## 正式训练命令与产物

以下是正式训练命令，产物核对或验收命令另行记录：

```bash
bash 训练与运行/sh/train_docking.sh B-C-T0-RB
```

独立训练根为 `/storage/penghongen/PocketXMol/training/B-C-T0-RB/`，保存的训练配置及检查点分别位于train_config和checkpoints子目录。W&B使用pencounkdual-111/PocketXmol_raw、名称B-C-T0-RB，实际run id在启动后记录。

371591控制目录仍为 `/home/penghongen/Feedback/Pocket_Plus/allocations/371591/`。前一E评价正常结束且try_lock恢复后，保存新旧动态命令与启动记录，再移除try_lock运行。after_lock和此前有效及异常产物全部保留，不使用scancel。

训练完成后直接用实际best测试C0／C5，每实例每协议50候选、100步、batch优先50；之后用同一作业8个CPU评价进程完成三个视图和核酸比例分析。明确best的采样配置及短正式命令在训练结束后登记，不执行训练后完整验证集采样，不额外增加实验。

## 计划与实现差异

本实验沿用已纠正的中心T0契约和既定RB配置，没有新增科学开关、代码或资产重建。完整训练、规定测试与报告仍待完成。
