#!/usr/bin/env bash
set -euo pipefail
/storage/penghongen/PocketXMol/runtime/venv/bin/python - <<'PY'
import glob,json,hashlib
result={}
for path in sorted(glob.glob('/storage/penghongen/tmp/pocketxmol_density_20260914/benchmarks/*/result.json')):
 data=json.load(open(path))
 ids=data.pop('sample_ids',[])
 data['sample_count']=len(ids)
 data['consumed_sample_prefix_sha256']=hashlib.sha256(json.dumps(ids[:len(data['updates'])*72]).encode()).hexdigest()
 result[path]=data
print(json.dumps(result,ensure_ascii=False))
PY
