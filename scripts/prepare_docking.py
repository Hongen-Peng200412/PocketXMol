"""读取共同数据准备配置, 调用一个明确阶段; 正式命令见训练与运行/README.md."""

import argparse
import sys
from pathlib import Path

import yaml
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docking.preparation import prepare


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备共同 docking 清单、对称排列和逐实例标签")
    parser.add_argument("config")
    parser.add_argument("stage", choices=["index", "objects", "samples", "freeze"])
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as stream:
        config = EasyDict(yaml.safe_load(stream))
    prepare(config, args.stage, args.shard_id, args.shard_count)
