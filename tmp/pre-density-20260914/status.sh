#!/usr/bin/env bash
set -euo pipefail
control='/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration/运行日志与统计/slurm/allocations/379402'
date -Is
ls -l "$control" "$control/../try_lock_379402" 2>/dev/null || true
tail -12 "$control/out"
tail -12 "$control/err"
/storage/penghongen/PocketXMol/runtime/venv/bin/python - <<'PY'
import glob,json
for path in sorted(glob.glob('/storage/penghongen/tmp/pocketxmol_density_20260914/benchmarks/*/result.json')):
 data=json.load(open(path));updates=data.get('updates',[])
 print('BENCHMARK',path,data['status'],'updates',len(updates),'last',updates[-1] if updates else data.get('error','pending'))
PY
