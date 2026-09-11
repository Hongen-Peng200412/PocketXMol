# B-C-T1-RA测试推理与评价

本实验承接 [中心T1-RA训练](03-B-C-T1-RA.md)。训练于22400次优化器更新后按第三次学习率下降停止，最低原C5 val/loss为2.831010341644287，对应20800步检查点。本次直接生成完整测试集C0、C5候选；训练期间原val/loss已经用于选best，不另做完整验证集采样。

## 固定输入与正式命令

配置为 `configs/docking/sample-B-C-T1-RA-test.yml`，使用训练时保存的 `/storage/penghongen/PocketXMol/training/B-C-T1-RA/train_config/B-C-T1-RA.yml` 和同训练根下 `checkpoints/step=20800.ckpt`。RA受体分支、中心T1机制保持：首步原纯高斯；后续原高斯后增加s乘以局部给定中心与预测重原子质心之差，s=1-level_dict['pos']。C0与C5只决定给定中心和固定口袋，均执行T1采样；定位后不再读取真值中心或真值偏移。

测试ALL清单446个实例，每协议每实例50个候选、100步、batch_size=50，沿用冻结候选种子和C5偏移。CAP10的272个实例和HF10_TO5的227个实例复用同一候选池。原置信度头、轨迹汇总、self-ranking与CPU评价口径均沿用已验收实现。

在371591所用独立release中执行正式测试推理：

```bash
bash 训练与运行/sh/sample_docking.sh B-C-T1-RA-test
```

推理全部完成后，从同一release提交8核CPU正式评价：

```bash
bash 训练与运行/submit_task.sh --sh evaluate_docking.sh --resource cpu --cpus 8 -- B-C-T1-RA-test
```

推理和评价产物为 `/storage/penghongen/PocketXMol/sampling/B-C-T1-RA/test/`；各协议的逐实例候选、姿态、评价文件与汇总均保留。W&B评价使用 `pencounkdual-111/PocketXmol_raw`，名称B-C-T1-RA_test，实际run id在评价启动后记录。GPU继续使用371591的A800，after_lock保留。

## 运行前核查

20800步best由Slurm内的训练检查点核查确认，具体命令及结果见训练日志。本次只增加明确best的测试配置，模型、特征化、噪声链、评价代码和共同资产均沿用既有版本。主代理第一遍按实际消费字段核对配置职责、保存配置和best路径，第二遍核对注释中的原点、实际噪声强度及T1步骤。未新增或修改生产函数。

配置验收：本地yaml.safe_load解析通过；与已完成的T0测试配置比较，仅模型名、保存训练配置、checkpoint、T1开关、输出根及W&B名称发生预期变化，其余字段完全一致；C0/C5 test、RA、batch50、50候选、100步及8核评价断言通过。独立代理t1_test_config_review完成一次仅针对本配置及本文的窄范围只读核查，确认实际字段消费与科学注释一致，无阻断问题。此前代码链的两轮全面独立审查已结束，本次不重复扩大范围。

本次release以已审查的ec06dbd运行副本为基础，只加入此测试配置；不从共享工作区复制其他代理的未提交文件。实际release、launch、启动元数据和完成情况随正式执行填写。
