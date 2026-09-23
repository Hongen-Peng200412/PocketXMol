#!/usr/bin/env bash
set -euo pipefail

# 在获准GPU中完成一个CA2标准配置的采样、CPU评价和2/3 Å汇总。
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$project_root"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
name="${1:?请指定标准测评配置名称，如local_cov-C-CA2}"
config="configs/docking/sample-${name}-test.yml"
python_bin=/storage/penghongen/PocketXMol/runtime/venv/bin/python
"$python_bin" scripts/sample_docking.py "$config"
CUDA_VISIBLE_DEVICES='' "$python_bin" scripts/evaluate_docking.py "$config"
CUDA_VISIBLE_DEVICES='' "$python_bin" scripts/summarize_docking_thresholds.py "$config"
