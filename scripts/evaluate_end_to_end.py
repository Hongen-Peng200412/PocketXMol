"""评价一个受体条件的四种端到端方法并合并完整Matcher轨迹。"""

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
