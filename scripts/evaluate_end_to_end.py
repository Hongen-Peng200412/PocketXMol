"""评价一个受体条件的四种端到端方法并合并完整 Matcher 轨迹.

命令行接收一个 GT 或 CA2 端到端 YAML. 主逻辑由
``docking.final_evaluation.evaluate_end_to_end`` 提供, 在配置的 ``output_root`` 写
``docking_results.jsonl``、扩展 ``evaluation.json``、``wandb_run.json`` 和 W&B 汇报.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docking.final_evaluation import evaluate_end_to_end
from utils.misc import make_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评价strongest-1到PocketXMol端到端结果")
    parser.add_argument("config", help="端到端YAML配置路径")
    arguments = parser.parse_args()
    evaluate_end_to_end(make_config(arguments.config))
