#!/usr/bin/env bash
set -euo pipefail
/storage/penghongen/PocketXMol/runtime/venv/bin/python - <<'PY'
import glob,json,statistics
for number in (15,16,17,18):
 for path in glob.glob(f'/storage/penghongen/tmp/pocketxmol_density_20260914/benchmarks/{number}_*/result.json'):
  data=json.load(open(path));updates=data['updates'][1:]
  print(json.dumps(dict(path=path,status=data['status'],warm_updates=len(updates),warm_mean=statistics.mean(x['total_seconds'] for x in updates) if updates else None,compute_mean=statistics.mean(sum(x[k] for k in ('forward_seconds','backward_seconds','optimizer_seconds')) for x in updates) if updates else None,parameters_finite=data.get('parameters_finite'),error=data.get('error'))))
PY
