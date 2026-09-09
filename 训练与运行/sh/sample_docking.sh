#!/usr/bin/env bash
set -euo pipefail

# 在获准GPU allocation中执行一个明确模型/划分的完整候选生成.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$project_root"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_name="${1:?请指定明确采样配置, 如 official-validation}"
exec /storage/penghongen/PocketXMol/runtime/venv/bin/python scripts/sample_docking.py "configs/docking/sample-${run_name}.yml"
