# Handoff: CA2标准测评与strongest-1端到端测评收口

Date: 2026-09-23

## Current State

CA2标准测评与strongest-1到PocketXMol的GT／CA2端到端测评均已完成。完整结果、正式命令、产物位置、W&B、失败口径与运行追溯见`日志/————最终测试————/标准测试/总体结果与实验追溯.md`和`日志/————最终测试————/端到端测试/总体结果与实验追溯.md`。

378587／hnode02与386151／hnode02均已返回`try_lock`并保留`after_lock`。不得在没有新授权时删除`after_lock`或向这两项资源追加任务。

## Completed

- CA2标准测评完成local_cov-C的C0/C5、local_cov-E的E以及官方冻结模型的C0/C5/E；详细三视图结果保存在三份标准测试日志。
- 端到端测评完成GT 280条和CA2 268条handoff的official-C、local_cov-C、C-C与C-C-E；主表统一使用77个PDB分母。
- 最终并行评价release为`/home/penghongen/Feedback/PocketXMol/releases/PocketXMol_f4d72bbea387/PocketXMol`。GT结果根为`/storage/penghongen/PocketXMol/final_evaluation/end_to_end/GT`，CA2结果根为`/storage/penghongen/PocketXMol/final_evaluation/end_to_end/CA2`。
- 两个结果根均保存`docking_results.jsonl`、`evaluation.json`和`wandb_run.json`；直接结果顺序已与各自`inputs/base.jsonl`精确核对。
- W&B在线运行分别为GT `z9ei2jzt`和CA2 `so7ostew`，项目为`pencounkdual-111/PocketXmol_final`。
- 最终评价使用32个进程。非测试validation实例`5irx/0`证明串行与多进程的四阶段状态、RMSD、Top-1、成功布尔值、汇总和输出顺序精确一致；证据位于`/storage/penghongen/PocketXMol/gates/end_to_end_parallel_validation/equivalence_v3`。
- 独立科学逻辑、Git布局和注释审查均已批准。实现线与学习线的最终运行树完全相同，tree为`59e9791cc38bf0f5234c4753da6068aa564a3c68`。

## Decisions

- strongest固定为strongest-1。official只执行一次C；local_cov报告单次C、C-C和C-C-E。
- 中间轮只检查精确预测SMILES明确声明的立体化学；未声明的潜在立体中心不判错。
- C-C-E的预测E完全复用现行包络逻辑，不读取真实配体质心、真实包络或冻结C5。
- 每个Stage3前景候选和背景候选消耗一次尝试；已知不属于Stage3的前景候选免费跳过。K=20只是报告位置，不是尝试上界。
- GT的`9qkz/0` official-C因`AssertionError: unknown element in pocket`产生50/50预处理失败候选；该失败保留计费候选和77个PDB分母，不做例外修复。

## Next Actions

本轮没有遗留运行任务。后续若需要新的统计分析，直接读取冻结的`evaluation.json`和`docking_results.jsonl`，不得重复GPU推理或改写现有候选产物。

## Files To Reopen

- `日志/————最终测试————/标准测试/总体结果与实验追溯.md`
- `日志/————最终测试————/端到端测试/总体结果与实验追溯.md`
- `想法/方案草稿/re-9-21.md`
- `docking/final_evaluation.py`
- `scripts/evaluate_end_to_end.py`
