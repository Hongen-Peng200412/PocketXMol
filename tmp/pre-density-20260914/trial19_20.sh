#!/usr/bin/env bash
# 用超过预取容量的均匀实例窗口检查持续供数；不清除操作系统文件缓存。
set -uo pipefail
python='/storage/penghongen/PocketXMol/runtime/venv/bin/python'
evidence='/storage/penghongen/tmp/pocketxmol_density_20260914'
read -r batch distance < <("$python" - "$evidence" <<'PY'
import json,glob,sys,statistics
from pathlib import Path
root=Path(sys.argv[1]);comparisons={};batch=None
for mode,first in [('D1',15),('D4',17)]:
 values=[];runs=[]
 for number in (first,first+1):
  names=list((root/'benchmarks').glob(f'{number}_*/result.json'));assert len(names)==1,names
  data=json.load(open(names[0]));assert data['status']=='complete' and data['parameters_finite'],names[0]
  runs.append(data['updates'])
  if mode=='D1': batch=data['arguments']['batch']
 count=min(map(len,runs));assert count>5,(mode,count)
 values=[statistics.mean(x['total_seconds'] for x in updates[1:count]) for updates in runs]
 comparisons[mode]=dict(paired_warm_updates=count-1,plain_seconds=values[0],distance_seconds=values[1],overhead_fraction=values[1]/values[0]-1)
keep=all(item['overhead_fraction']<=.10 for item in comparisons.values())
(root/'distance_decision.json').write_text(json.dumps(dict(scope='preexperiment_not_formal',criterion='both_D1_D4_complete_optimizer_update_overhead_at_most_10_percent',keep_distance=keep,comparisons=comparisons),indent=2)+'\n')
print(batch,int(keep))
PY
)
test -n "$batch" && test -n "$distance" || exit 1
for spec in '19 15 2025' '20 32 2026'; do
 read -r number workers seed <<<"$spec"
 if (( $(date +%s) >= 1789382216 )); then exit 124; fi
 "$python" -u tmp/pre-density-20260914/benchmark.py --config configs/docking/D1-C-T0-RA.yml --output "$evidence/benchmarks/${number}_D1_b${batch}_w${workers}_seed${seed}_long" --batch "$batch" --workers "$workers" --seconds 240 --updates 60 --distance "$distance" --sample-seed "$seed" --deadline 1789382216
 status=$?
 printf 'IO_WINDOW number=%s workers=%s seed=%s exit=%s\n' "$number" "$workers" "$seed" "$status"
done
