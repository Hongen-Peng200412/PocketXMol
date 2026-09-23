"""从标准候选连续 RMSD 汇总严格 2 Å 与 3 Å 结果.

命令行接收与采样评价相同的 YAML. 主逻辑由
``docking.final_evaluation.summarize_docking_thresholds`` 提供, 只读取既有候选指标并在
``<output_root>/<split>/threshold_summary.json`` 写多协议、多视图、实例/PDB 等权汇总.
"""

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
