"""定义最终测评 JSON 文件读写与端到端产物路径, 不加载模型或采样器."""

import json
from pathlib import Path


def read_jsonl(path):
    """读取非空 JSONL 记录并保持文件顺序."""
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    """在同目录写临时 JSON 后原子替换目标文件."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path, records):
    """在同目录写完整 JSONL 后原子替换目标文件."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def stage_input_path(config, stage):
    """返回当前受体条件下一个推理阶段的冻结 JSONL 路径."""
    return Path(config.output_root) / "inputs" / f"{stage}.jsonl"


def stage_output_root(config, stage):
    """返回 official 或 local_cov 的独立采样根."""
    family = "official" if stage == "official-c" else "local_cov"
    return Path(config.output_root) / family


def candidate_output_dir(config, stage, record):
    """返回一条 Matcher 候选在指定阶段的正式产物目录."""
    return (
        stage_output_root(config, stage)
        / "test"
        / stage
        / record["pdb_id"]
        / str(record["candidate_id"])
    )
