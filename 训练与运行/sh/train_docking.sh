#!/usr/bin/env bash
set -euo pipefail

# 每次只运行一个明确实验; 同一GPU顺序执行, 多张获准GPU由独立Slurm任务隔离.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$project_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# 为DataLoader多worker预取的共享张量预留文件句柄, 仅调整本次子进程软上限.
ulimit -Sn 65536
export TMPDIR=/storage/penghongen/tmp
mkdir -p "$TMPDIR"
experiment="${1:?请指定明确实验配置, 如 local_cov-C-T0-RA}"
shift
exec /storage/penghongen/PocketXMol/runtime/venv/bin/python scripts/train_pl.py "configs/docking/${experiment}.yml" --logdir "/storage/penghongen/PocketXMol/training/${experiment}" "$@"
