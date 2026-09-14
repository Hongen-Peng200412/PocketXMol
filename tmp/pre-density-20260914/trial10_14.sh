#!/usr/bin/env bash
# 同一冻结代码下比较global72批量；每个子进程独立保存失败与完整更新证据。
set -uo pipefail
python='/storage/penghongen/PocketXMol/runtime/venv/bin/python'
evidence='/storage/penghongen/tmp/pocketxmol_density_20260914'
"$python" -m pytest -q tests/test_density.py --disable-warnings | tee "$evidence/density_tests_gpu_medium.log"
for spec in '10 D1 24 24 120' '11 D1 36 24 120' '12 D1 72 24 120' '13 D4 36 16 150' '14 D4 72 16 150'; do
 read -r number mode batch updates seconds <<<"$spec"
 if (( $(date +%s) >= 1789382216 )); then exit 124; fi
 "$python" -u tmp/pre-density-20260914/benchmark.py --config "configs/docking/$mode-C-T0-RA.yml" --output "$evidence/benchmarks/${number}_${mode}_b${batch}_w32_full" --batch "$batch" --workers 32 --seconds "$seconds" --updates "$updates" --sample-seed 2024 --deadline 1789382216
 status=$?
 printf 'BATCH_COMPARISON number=%s mode=%s batch=%s exit=%s\n' "$number" "$mode" "$batch" "$status"
done
