#!/usr/bin/env bash
set -euo pipefail
/storage/penghongen/PocketXMol/runtime/venv/bin/python - <<'PY'
import glob,json
result={}
for path in sorted(glob.glob('/storage/penghongen/tmp/pocketxmol_density_20260914/benchmarks/*/result.json')):
 data=json.load(open(path))
 result[path]=data
print(json.dumps(result,ensure_ascii=False))
PY
