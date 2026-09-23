"""评价一个受体条件的四种端到端方法并合并完整 Matcher 轨迹.

命令行接收一个 GT 或 CA2 端到端 YAML, 并可用 ``--workers`` 覆盖配置中的
``evaluation_workers``. 主逻辑由 ``docking.final_evaluation.evaluate_end_to_end`` 提供,
按 ``inputs/base.jsonl`` 的 handoff 记录并行评价. 配置的 ``output_root`` 下会写出三项
最终产物:
``docking_results.jsonl``: JSONL, 按 ``inputs/base.jsonl`` 原顺序保存逐候选四阶段结果.
``evaluation.json``: dict, 核心字段 ``summary`` 保存汇总表, ``traces`` 保存扩展 Matcher 轨迹.
``wandb_run.json``: dict, 核心字段 ``run_id``、``entity`` 和 ``project`` 保存 W&B 运行身份.
主进程和所有评价子进程都把数值库线程数固定为 1, 避免进程与线程重复并行; W&B 只由
主进程写入.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 端到端评价按 handoff 使用多进程; 主进程与子进程都禁止数值库在进程内再次扩展线程.
for thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[thread_variable] = "1"

from docking.final_evaluation import evaluate_end_to_end
from utils.misc import make_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评价strongest-1到PocketXMol端到端结果")
    parser.add_argument("config", help="端到端YAML配置路径")
    parser.add_argument(
        "--workers",
        type=int,
        help="评价进程数; 未指定时读取配置字段 evaluation_workers, 缺省为 1",
    )
    arguments = parser.parse_args()
    config = make_config(arguments.config)
    if arguments.workers is not None:
        config.evaluation_workers = arguments.workers
    evaluate_end_to_end(config)
