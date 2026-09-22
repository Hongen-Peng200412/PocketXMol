"""构造 strongest-1 端到端初始清单或按上一轮 Top-1 构造后续清单.

命令行接收端到端 YAML 和 ``initial``、``local-c2``、``local-e`` 之一. 主逻辑由
``docking.end_to_end`` 提供, 在 ``<output_root>/inputs`` 写对应阶段 JSONL; 后续清单只读取
上一轮无真值 ranking, 不读取沉积配体坐标.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docking.end_to_end import prepare_followup_records, prepare_initial_records
from utils.misc import make_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构造端到端冻结输入清单")
    parser.add_argument("config", help="端到端YAML配置路径")
    parser.add_argument(
        "stage",
        choices=("initial", "local-c2", "local-e"),
        help="initial 读取 Matcher, 后两项读取上一轮正式候选",
    )
    arguments = parser.parse_args()
    configuration = make_config(arguments.config)
    if arguments.stage == "initial":
        prepare_initial_records(configuration)
    else:
        prepare_followup_records(configuration, arguments.stage)
