#!/usr/bin/env bash
set -euo pipefail

# 从三组既有GT候选生成严格2/3 Å汇总，不重新前向。
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$project_root"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python_bin=/storage/penghongen/PocketXMol/runtime/venv/bin/python
for name in local_cov-C-GT local_cov-E-GT official-GT; do
    "$python_bin" scripts/summarize_docking_thresholds.py "configs/docking/report-${name}.yml"
done
