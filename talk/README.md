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

本轮提交按“文档先于源码、上游先于下游”排列：范围与计划 → 统一模型与字段契约 → 任务与指标 → 数据/transform → 噪声/GNN → 损失/采样/评测。可用下列命令按顺序阅读：

```powershell
git log --reverse --oneline df1cf84..Learn/conformation-docking
```

## 明确未覆盖的范围

本轮不覆盖线性肽对接、环肽对接、肽设计、SBDD、片段生成和 PROTAC 设计。源码中与这些任务共享的低层模块可以说明通用契约，但不展开这些任务的领域设置。
