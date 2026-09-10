# B-C-T0-RA 正式训练

本实验是六个无密度模型中的第一个：中心口袋，T0保持原位置噪声，RA使用独立核酸投影和共同受体编码器。依据 [科学契约](../../想法/方案草稿/9-8-科学契约.md) 和 [工程细节](../../想法/方案草稿/9-8-工程与实现细节.md)；共同数据、环境和前置验收见 [准备记录](00-实现与共同数据准备.md)。本记录只保存正式实验，不把前述短检查的权重或候选用作实验产物。

## 来源与配置

- 配置：[B-C-T0-RA.yml](../../configs/docking/B-C-T0-RA.yml)。训练种子2023，中心训练使用每次重采样C5；原val/loss路径使用冻结C5和完整781个验证实例。
- 冻结清单：`/storage/penghongen/PocketXMol/data/`，有效freeze作业376632；训练65290、验证781，校准361与测试446均不进入训练或val/loss模型选择。
- 初始权重：`/storage/penghongen/PocketXMol_official_test/extracted/data/trained_models/pxm/checkpoints/pocketxmol.ckpt`。只继承官方模型参数，新建优化器；全部主体和新增受体参数共同训练。
- 环境：`/storage/penghongen/PocketXMol/runtime/venv/bin/python`，继承pxm_phase1的Torch2.6.0、Lightning2.6.0和RDKit2023.9.3，项目层补充W&B0.21.1。
- 资源：已获准的371591，gnode09，一张A800、16核；按用户最新要求优先单卡72、累积1、有效全局72、bf16，15个DataLoader worker。训练子进程文件句柄软上限65536，TMPDIR为`/storage/penghongen/tmp`。
- AdamW，lr=1e-4、warmup=0，原loss和置信度目标。每800个优化器更新计算原val/loss，按1%相对改善、patience=5、factor=0.2的原Plateau下降；第三次实际下降停止，最多40000更新。best取原验证损失最低者，保留last及全部定期检查点。
- W&B：`pencounkdual-111/PocketXmol_raw`，名称B-C-T0-RA，online；实际run id为 `9wkyn4qn`，[在线记录](https://wandb.ai/pencounkdual-111/PocketXmol_raw/runs/9wkyn4qn)。密钥不进入日志。

## 正式运行命令

在371591每次冻结的PocketXMol release根执行：

```bash
bash 训练与运行/sh/train_docking.sh B-C-T0-RA
```

输出根为 `/storage/penghongen/PocketXMol/training/B-C-T0-RA/`：src保存Python源码副本，train_config保存实际配置，checkpoints保存检查点，wandb保存运行记录。正式启动前已确认该实验目录不存在。

沿用原allocation控制器和锁目录 `/home/penghongen/Feedback/Pocket_Plus/allocations/371591/`；out与err仍在该目录。动态命令调用PocketXMol既有release／launch工具。after_lock保留，同一卡按实验顺序使用。

## 实际启动

72×1真实GPU检查通过后，已在原371591的第9次执行启动本实验。计算节点生成的启动标识为 `train_B-C-T0-RA_job371591_20260910T112700`；实际release为 `/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_dbfde52856a9/PocketXMol`，launch为 `/home/penghongen/Feedback/PocketXMol/launches/371591/train_B-C-T0-RA_job371591_20260910T112700`。本地实现提交为 `136c621`；release还包含工作区原有的docking/preparation.py说明文字改动，未把这些改动擅自纳入提交。服务器源码目录的旧.git元数据不代表本次实现版本，以冻结release和本地记录追溯。

动态命令副本和启动留证分别保存为 `/storage/penghongen/PocketXMol/control/371591/train_B-C-T0-RA_run_cmd.sh`、同目录train_B-C-T0-RA_start.json。留证记录本次在累计out、err中的起始字节位置360594和54949，读取本次错误时不混入先前GPU检查的失败输出。脚本语法检查通过后才写入该作业run_cmd并释放try_lock；没有取消allocation，after_lock保留。

启动已确认加载官方参数、bf16混合精度、单GPU、全部20.2 M参数可训练。W&B在线创建成功，记录目录为 `/storage/penghongen/PocketXMol/training/B-C-T0-RA/wandb/run-20260910_112743-9wkyn4qn`，源码与实际配置副本已保存。首次稳定检查已到第70个优化器更新，lr仍为1e-4，未见本次OOM或批量裁减；尚未到第800步的首次完整验证。约1.08更新／秒只是启动阶段观察，不作全程耗时承诺。

## 当前结果与后续

共同数据、必要CPU／GPU验收及72×1真实资源检查均已通过，本实验正式训练中。完成训练后核实停止原因、实际优化器步、最低val/loss及其best路径，再为该best执行完整候选validation和test。中心模型分别评价C0与C5，各实例50候选、100步；当前不提前填写best路径或最终成绩。
