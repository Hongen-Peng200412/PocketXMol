#!/usr/bin/env bash
set -euo pipefail

# 在8核CPU任务中评价既有候选, 不生成候选或申请GPU.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$project_root"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR=/storage/penghongen/tmp
mkdir -p "$TMPDIR"
run_name="${1:?请指定与采样相同的配置, 如 official-validation}"
exec /storage/penghongen/PocketXMol/runtime/venv/bin/python scripts/evaluate_docking.py "configs/docking/sample-${run_name}.yml"
