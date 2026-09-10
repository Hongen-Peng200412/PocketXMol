#!/usr/bin/env bash
set -euo pipefail

# GPU验收入口, 不创建W&B正式run; CUDA_VISIBLE_DEVICES保留Slurm分配.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# 多进程通信在TMPDIR创建UNIX socket, 必须用短路径; pytest产物仍按每次launch单独保存.
export TMPDIR=/storage/penghongen/tmp
pytest_root="$TMPDIR/pocketxmol_gpu_checks_${TASK_RUN_STAMP:?本检查入口须在授权Slurm的正式launch内运行}"
mkdir -p "$pytest_root"
# 专用GPU入口不能把全部skip的pytest退出码0记成验收成功.
/storage/penghongen/PocketXMol/runtime/venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "GPU验收需要实际可见的CUDA GPU"'
# 可追加pytest筛选参数, 如-k RB只复测失败分支.
exec /storage/penghongen/PocketXMol/runtime/venv/bin/python -m pytest -q -s -p no:cacheprovider --basetemp "$pytest_root/pytest" --tb=short tests/test_docking_gpu.py "$@"
