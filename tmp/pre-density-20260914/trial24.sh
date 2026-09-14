#!/usr/bin/env bash
# 第24次也是最后一次改进：仅为当前只读memmap建议随机访问，不修改资产或系统缓存。
set -euo pipefail
if (( $(date +%s) >= 1789382216 )); then exit 124; fi
'/storage/penghongen/PocketXMol/runtime/venv/bin/python' -u tmp/pre-density-20260914/benchmark.py \
 --config configs/docking/D1-C-T0-RA.yml \
 --output /storage/penghongen/tmp/pocketxmol_density_20260914/benchmarks/24_D1_b72_w32_seed2030_madv_random \
 --batch 72 --workers 32 --seconds 240 --updates 60 --distance 1 \
 --sample-seed 2030 --madvise-random 1 --deadline 1789382216
