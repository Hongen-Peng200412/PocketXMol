"""评价一个受体条件的四种端到端方法并合并完整 Matcher 轨迹.

命令行接收一个 GT 或 CA2 端到端 YAML. 主逻辑由
``docking.final_evaluation.evaluate_end_to_end`` 提供, 在配置的 ``output_root`` 写
``docking_results.jsonl``、扩展 ``evaluation.json``、``wandb_run.json`` 和 W&B 汇报.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 端到端评价按 handoff 使用多进程；主进程与子进程都禁止数值库在进程内再次扩展线程.
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
        help="评价进程数；未指定时读取配置中的evaluation_workers，缺省为1",
    )
    arguments = parser.parse_args()
    config = make_config(arguments.config)
    if arguments.workers is not None:
        config.evaluation_workers = arguments.workers
    evaluate_end_to_end(config)
