# PocketXMol 学习材料索引

本目录保存面向代码学习者的当前文档、任务档案、执行计划和执行证据。仓库原有的 `docs/` 继续承担安装、运行和测试集复现说明；`talk/` 解释论文任务、数据契约和源码调用关系，不替代原有用户文档。

## 当前学习轮次

| 规划文档 | 当前状态 | 执行记录 | 覆盖任务 | 当前学习分支 |
|---|---|---|---|---|
| [分子构象生成与小分子对接学习计划](plans/conformation_docking_learning_plan.md) | 已完成 | [分子构象生成与小分子对接执行记录](logs/conformation_docking_learning_log.md) | 分子构象生成、小分子对接 | `Learn/conformation-docking` |

## 当前契约文档

| 文档 | 职责 | 当前状态 |
|---|---|---|
| [PocketXMol 统一建模总览](00_pocketxmol_unified_view.md) | 解释统一任务形式化、联合训练方法和整体能力来源 | 完成 |
| [输入、批字段与输出产物概览](01_input_output_overview.md) | 解释原始输入、批次字段、模型输出和落盘产物 | 完成 |
| [模块与调用结构概览](02_module_structure_overview.md) | 解释入口、数据变换、去噪器、GNN、采样和评测之间的调用关系 | 完成 |
| [指标、候选生成与排序](03_metrics_and_ranking.md) | 解释构象与对接指标、候选生成和排序协议 | 完成 |
| [任务档案：分子构象生成](tasks/molecular_conformation_generation.md) | 定义分子构象生成及其 PocketXMol 实现 | 完成 |
| [任务档案：小分子 docking](tasks/small_molecule_docking.md) | 定义小分子 docking 及其 PocketXMol 实现 | 完成 |

## Git 学习层次

本轮把一次 Git 提交视为一个连续阅读章节，而不是一种文件类型的集合。提交依次回答以下问题：

1. 任务范围、统一表示、输入输出、模块地图和评测口径是什么？
2. 外部结构怎样解析为单分子数据容器？
3. 数据集怎样读取样本并构造构象与 docking prompt？
4. 训练配置怎样装配数据、模型与损失？
5. 信息等级怎样驱动平移、旋转、扭转和自由坐标先验？
6. 任务 noiser 怎样加噪并把预测投影回允许的几何自由度？
7. 口袋条件等变网络怎样编码并更新节点、半边和坐标？
8. fixed prompt、距离、二面角和置信度怎样组成训练损失？
9. 采样配置与迭代循环怎样生成最终批状态和轨迹？
10. 批量/单复合物入口怎样重建并保存候选结构？
11. 构象 Coverage/Matching 与 docking RMSD 怎样计算？
12. 碰撞、立体化学和 PoseBusters 怎样验证 docking 候选？
13. tuned confidence 怎样与辅助检查共同完成最终排序？

可用下列命令按提交顺序阅读：

```powershell
git log --reverse --oneline df1cf84..Learn/conformation-docking
```

## 明确未覆盖的范围

本轮不覆盖线性肽对接、环肽对接、肽设计、SBDD、片段生成和 PROTAC 设计。源码中与这些任务共享的低层模块可以说明通用契约，但不展开这些任务的领域设置。
