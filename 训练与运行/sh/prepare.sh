#!/usr/bin/env bash
set -euo pipefail

# 一个CPU任务运行一个明确准备阶段, 每任务8核; 只有objects/samples使用Slurm array分片.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$project_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
stage="${1:?请指定 index、objects、samples 或 freeze}"
shard_id="${SLURM_ARRAY_TASK_ID:-0}"
shard_count="${2:-${SLURM_ARRAY_TASK_COUNT:-1}}"
exec /storage/penghongen/PocketXMol/runtime/venv/bin/python scripts/prepare_docking.py configs/docking/prepare.yml "$stage" --shard-id "$shard_id" --shard-count "$shard_count"
