#!/usr/bin/env bash
set -euo pipefail
evidence='/storage/penghongen/tmp/pocketxmol_density_20260914'
python='/storage/penghongen/PocketXMol/runtime/venv/bin/python'
export PYTHONPATH="$evidence/deps:$PYTHONPATH"
if ! "$python" -c 'import einops'; then
 "$python" -m pip install --target "$evidence/deps" --no-deps einops==0.8.1
fi
"$python" -u tmp/pre-density-20260914/source_equivalence.py | tee "$evidence/source_equivalence.log"
"$python" -u tmp/pre-density-20260914/readout_equivalence.py | tee "$evidence/readout_equivalence.log"
"$python" -m pytest -q tests/test_density.py --disable-warnings | tee "$evidence/density_tests_gpu.log"
