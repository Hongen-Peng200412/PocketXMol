"""读取一个端到端冻结阶段清单并调用原 PocketXMol 采样循环.

命令行接收端到端 YAML 和唯一阶段名. 主逻辑由
``docking.end_to_end.sample_end_to_end_stage`` 提供, 在 official 或 local_cov 阶段目录保存
逐候选科学身份、50 个候选、世界坐标 SDF、原置信度、完成记录和无真值 ranking.
"""

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
