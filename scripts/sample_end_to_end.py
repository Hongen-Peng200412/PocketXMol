"""读取一个端到端冻结阶段清单并调用原PocketXMol采样循环。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docking.end_to_end import STAGES, sample_end_to_end_stage
from utils.misc import make_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="执行一个端到端PocketXMol阶段")
    parser.add_argument("config", help="端到端YAML配置路径")
    parser.add_argument("stage", choices=STAGES, help="本次唯一推理阶段")
    arguments = parser.parse_args()
    sample_end_to_end_stage(make_config(arguments.config), arguments.stage)
