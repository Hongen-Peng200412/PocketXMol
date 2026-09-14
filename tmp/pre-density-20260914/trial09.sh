#!/usr/bin/env bash
set -euo pipefail
python='/storage/penghongen/PocketXMol/runtime/venv/bin/python'
evidence='/storage/penghongen/tmp/pocketxmol_density_20260914'
if "$python" -u tmp/pre-density-20260914/readout_equivalence.py | tee "$evidence/readout_equivalence_fp32.log"; then
 reference=0
 label=batched_fp32
 "$python" -m pytest -q tests/test_density.py --disable-warnings | tee "$evidence/density_tests_gpu.log"
else
 reference=1
 label=reference
 printf 'FINAL_BATCHED_GATE_FAILED using_frozen_reference_and_ending_batched_optimization\n'
fi
exec "$python" -u tmp/pre-density-20260914/benchmark.py --config configs/docking/D4-C-T0-RA.yml --output "$evidence/benchmarks/09_D4_b24_w32_full_$label" --batch 24 --workers 32 --seconds 150 --updates 16 --readout-reference "$reference" --sample-seed 2024 --deadline 1789382216
