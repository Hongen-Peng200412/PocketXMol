"""从标准候选连续RMSD汇总严格2 Å与3 Å结果。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docking.final_evaluation import summarize_docking_thresholds
from utils.misc import make_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="汇总标准docking多阈值结果")
    parser.add_argument("config", help="与采样和评价相同的YAML配置路径")
    arguments = parser.parse_args()
    summarize_docking_thresholds(make_config(arguments.config))

