"""读取与采样相同的配置评价既有候选; 验证汇总ALL, 测试输出三个重叠视图.

调用形式: python scripts/evaluate_docking.py <采样配置.yml>, 使用生成候选时的同一份配置.
本入口不生成候选, 不加载独立排序器, 不申请CPU资源.
"""

import argparse
import sys
from pathlib import Path

import yaml
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docking.evaluation import evaluate_docking


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评价冻结 docking 候选池")
    parser.add_argument("config", help="与采样一致的 YAML 配置路径")
    arguments = parser.parse_args()
    with Path(arguments.config).open(encoding="utf-8") as stream:
        config = EasyDict(yaml.safe_load(stream))
    evaluate_docking(config)
