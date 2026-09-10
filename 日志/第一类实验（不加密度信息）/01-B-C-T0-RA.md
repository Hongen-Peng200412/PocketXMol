# B-C-T0-RA 正式训练

本实验是六个无密度模型中的第一个：中心口袋，T0保持原位置噪声，RA使用独立核酸投影和共同受体编码器。依据 [科学契约](../../想法/方案草稿/9-8-科学契约.md) 和 [工程细节](../../想法/方案草稿/9-8-工程与实现细节.md)；共同数据、环境和前置验收见 [准备记录](00-实现与共同数据准备.md)。本记录只保存正式实验，不把前述短检查的权重或候选用作实验产物。

## 来源与配置

- 配置：[B-C-T0-RA.yml](../../configs/docking/B-C-T0-RA.yml)。训练种子2023，中心训练使用每次重采样C5；原val/loss路径使用冻结C5和完整781个验证实例。
- 冻结清单：`/storage/penghongen/PocketXMol/data/`，有效freeze作业376632；训练65290、验证781，校准361与测试446均不进入训练或val/loss模型选择。
- 初始权重：`/storage/penghongen/PocketXMol_official_test/extracted/data/trained_models/pxm/checkpoints/pocketxmol.ckpt`。只继承官方模型参数，新建优化器；全部主体和新增受体参数共同训练。
- 环境：`/storage/penghongen/PocketXMol/runtime/venv/bin/python`，继承pxm_phase1的Torch2.6.0、Lightning2.6.0和RDKit2023.9.3，项目层补充W&B0.21.1。
- 资源：已获准的371591，gnode09，一张A800、16核；按用户最新要求优先单卡72、累积1、有效全局72、bf16，15个DataLoader worker。训练子进程文件句柄软上限65536，TMPDIR为`/storage/penghongen/tmp`。
- AdamW，lr=1e-4、warmup=0，原loss和置信度目标。每800个优化器更新计算原val/loss，按1%相对改善、patience=5、factor=0.2的原Plateau下降；第三次实际下降停止，最多40000更新。best取原验证损失最低者，保留last及全部定期检查点。
- W&B：`pencounkdual-111/PocketXmol_raw`，名称B-C-T0-RA，online。实际run id及链接在启动成功后补充，密钥不进入日志。

## 正式运行命令

在371591每次冻结的PocketXMol release根执行：

```bash
bash 训练与运行/sh/train_docking.sh B-C-T0-RA
```

输出根为 `/storage/penghongen/PocketXMol/training/B-C-T0-RA/`：src保存Python源码副本，train_config保存实际配置，checkpoints保存检查点，wandb保存运行记录。正式启动前已确认该实验目录不存在。

沿用原allocation控制器和锁目录 `/home/penghongen/Feedback/Pocket_Plus/allocations/371591/`；out与err仍在该目录。动态命令调用PocketXMol既有release／launch工具，实际版本、launch、开始状态与W&B信息在成功启动后补充。after_lock保留，同一卡按实验顺序使用。

## 当前结果与后续

共同数据和原36×2的必要CPU／GPU验收已通过。正式启动前，用户要求无密度优先72×1，六份配置已统一更新，须先完成这一资源设置的真实GPU验收；尚无正式训练指标或检查点。完成训练后核实停止原因、实际优化器步、最低val/loss及其best路径，再为该best执行完整候选validation和test。中心模型分别评价C0与C5，各实例50候选、100步；当前不提前填写best路径或最终成绩。
