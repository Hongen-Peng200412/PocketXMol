#!/usr/bin/env bash
set -euo pipefail

# 仅用于CPU验收, 不是训练或推理命令; 使用本项目环境, 不读取held-out科学样本.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR="/storage/penghongen/tmp/pocketxmol_checks_${SLURM_JOB_ID:?本检查入口须经Slurm运行}"
mkdir -p "$TMPDIR"
exec /storage/penghongen/PocketXMol/runtime/venv/bin/python -m pytest -q -p no:cacheprovider "$@"
