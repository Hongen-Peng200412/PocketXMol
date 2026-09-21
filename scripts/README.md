# 最终测评脚本入口

`prepare_end_to_end.py` 从 Matcher strongest-1 正式产物建立初始中心清单, 或根据上一轮无真值 self-ranking 建立第二次中心和预测包络清单。`sample_end_to_end.py` 每次只运行一个明确阶段。`evaluate_end_to_end.py` 汇总四种方法并回填完整 Matcher 轨迹。`summarize_docking_thresholds.py` 从标准评价的连续 RMSD 生成严格小于 2 Å和3 Å的新版汇总, 不覆盖旧评价文件。

正式执行使用 `训练与运行/sh/` 下的短 shell 入口。这里的 Python 文件既可用于门控, 也由正式入口调用; 实际命令必须按实验日志中的“正式运行命令”和“测试、门控与只读核查命令”分开记录。
