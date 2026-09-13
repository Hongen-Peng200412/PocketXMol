"""读取一份明确的采样配置并执行该模型的全部定位协议.

调用形式: python scripts/sample_docking.py <采样配置.yml>, 配置须明确填写实际 checkpoint 和输出目录.
逐实例产物与配置字段见 docking.sampling.sample_docking.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docking.sampling import sample_docking
from utils.misc import make_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按冻结 occurrence 清单生成 docking 候选")
    parser.add_argument("config", help="采样 YAML 配置路径")
    arguments = parser.parse_args()
    sample_docking(make_config(arguments.config))
