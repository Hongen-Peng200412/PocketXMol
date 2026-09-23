#!/usr/bin/env bash
set -euo pipefail

# 在一个受体条件内顺序完成可恢复的清单、四个推理阶段和CPU评价。
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$project_root"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
name="${1:?请指定端到端配置名称，如end-to-end-GT}"
config="configs/docking/${name}.yml"
python_bin=/storage/penghongen/PocketXMol/runtime/venv/bin/python
"$python_bin" scripts/prepare_end_to_end.py "$config" initial
"$python_bin" scripts/sample_end_to_end.py "$config" official-c
"$python_bin" scripts/sample_end_to_end.py "$config" local-c1
CUDA_VISIBLE_DEVICES='' "$python_bin" scripts/prepare_end_to_end.py "$config" local-c2
"$python_bin" scripts/sample_end_to_end.py "$config" local-c2
CUDA_VISIBLE_DEVICES='' "$python_bin" scripts/prepare_end_to_end.py "$config" local-e
"$python_bin" scripts/sample_end_to_end.py "$config" local-e
CUDA_VISIBLE_DEVICES='' "$python_bin" scripts/evaluate_end_to_end.py "$config"
