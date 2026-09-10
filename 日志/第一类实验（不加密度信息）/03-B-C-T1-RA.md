# B-C-T1-RA正式训练

本实验属于已批准的六个无密度模型，承接 [共同准备与实现](00-实现与共同数据准备.md) 和 [中心T0纠偏](02-T0中心契约修复与重训.md)。它使用RA核酸编码、中心T1机制；不从正确或错误T0的训练检查点续训。

## 固定训练条件

配置为 `configs/docking/B-C-T1-RA.yml`。训练每次抽一份半径0–5 Å均匀、方向球面均匀的delta，用完整g+delta选袋并定原点；原高斯后加入s*delta，监督目标和受体不跟随该新增平移。原val/loss读取验证清单冻结C5偏移及同一T1公式。g为完整配体重原子中心，s=1-level_dict['pos']。正式姿态评价仍保留C0与C5，推理首步为原纯高斯，后续保留中心相关重新加噪。

从规定官方pxm参数初始化，重新建立AdamW及调度状态；单张A800使用batch_size=72、累积1、bf16，名义全局批量72。初始lr=1e-4、warmup=0，每800个优化器更新计算原val/loss；Plateau相对阈值1%、patience=5、factor=0.2，第三次实际下降停止，上限40000步。最低原验证损失选择best，last和全部定期检查点保留。

## 正式运行命令与产物

在既有371591的独立冻结release中运行：

```bash
bash 训练与运行/sh/train_docking.sh B-C-T1-RA
```

产物目录为 `/storage/penghongen/PocketXMol/training/B-C-T1-RA/`，W&B账号／项目为 `pencounkdual-111/PocketXmol_raw`，新run独立创建。实际版本、release、launch和run id在确认启动后填写；本段落盘时尚未接入控制器。

运行沿用已通过自查、两轮独立审查及CPU／GPU验收的实现，不增加科学开关或重建共同资产。测试命令和验收结果见日志00及02，本文件上面的命令是正式训练命令。371591的after_lock保留，不申请额外GPU，不释放或删除旧T0产物。
