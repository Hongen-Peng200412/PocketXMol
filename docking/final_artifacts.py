"""定义最终测评 JSON/JSONL 的原子写入和端到端产物路径.

主要入口是 ``write_json``、``write_jsonl`` 和 ``candidate_output_dir``. 本模块只返回内存
记录或文件路径, 不加载模型、受体、密度或采样器. JSON 写入使用同目录临时文件替换,
避免中断时把半个文件识别为已完成产物.
"""

import json
from pathlib import Path


def read_jsonl(path):
    """读取一个 JSONL 文件中的非空记录并保持文件顺序.

    输入 ``path`` 是 JSONL 文件路径. 返回 list[dict], 每个元素对应一个非空物理行;
    空白行不产生记录. 本函数不排序、不去重、不修改字段.
    """
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    """以 UTF-8 缩进 JSON 原子写入一个结构化对象.

    输入 ``path`` 是目标文件路径, ``value`` 是可由 ``json.dumps`` 序列化的对象. 函数先
    写 ``<path.name>.tmp``, 禁止 NaN, 完成后替换目标文件; 父目录不存在时创建. 无返回值.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 与目标文件同目录的临时路径, 保证 replace 不跨文件系统.
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path, records):
    """以 UTF-8 JSONL 原子写入一组有序记录.

    输入 ``path`` 是目标文件路径, ``records`` 是可迭代的 JSON 对象; 每个元素写成一个
    物理行, 保持输入顺序并禁止 NaN. 函数先完整写入同目录临时文件再替换目标; 无返回值.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def stage_input_path(config, stage):
    """返回一个端到端阶段的冻结输入清单路径.

    ``config.output_root`` 决定当前 GT 或 CA2 端到端实验根, ``stage`` 是 official-c、
    local-c1、local-c2 或 local-e. 返回 ``<output_root>/inputs/<stage>.jsonl``.
    """
    return Path(config.output_root) / "inputs" / f"{stage}.jsonl"


def stage_output_root(config, stage):
    """返回一个阶段所属模型家族的独立采样根.

    official-c 返回 ``<output_root>/official``; local-c1、local-c2 和 local-e 返回
    ``<output_root>/local_cov``. 同一 local_cov 根内继续由阶段名隔离产物.
    """
    family = "official" if stage == "official-c" else "local_cov"
    return Path(config.output_root) / family


def candidate_output_dir(config, stage, record):
    """返回一条 Matcher 候选在指定阶段的正式产物目录.

    ``record.pdb_id`` 与 ``record.candidate_id`` 共同标识一个 handoff 候选. 返回目录为
    ``<stage_output_root>/test/<stage>/<pdb_id>/<candidate_id>``; 其中保存科学输入身份、
    候选 SDF、置信度、候选状态、结果完成标记和可选的无真值 ranking.
    """
    return (
        stage_output_root(config, stage)
        / "test"
        / stage
        / record["pdb_id"]
        / str(record["candidate_id"])
    )
