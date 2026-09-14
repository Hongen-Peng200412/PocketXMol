#!/usr/bin/env bash
# 每种结构以已完成批量比较中最快且参数有限的组合，配对关闭/开启距离项。
set -uo pipefail
python='/storage/penghongen/PocketXMol/runtime/venv/bin/python'
evidence='/storage/penghongen/tmp/pocketxmol_density_20260914'
for mode in D1 D4; do
 batch=$("$python" - "$evidence" "$mode" <<'PY'
import json,glob,sys,statistics
root,mode=sys.argv[1:]
candidates=[]
for filename in glob.glob(root+'/benchmarks/*/result.json'):
 data=json.load(open(filename));name=filename.split('/')[-2]
 if int(name[:2]) not in (9,10,11,12,13,14) or data.get('mode')!=mode: continue
 if data['status']=='complete' and data.get('parameters_finite') and len(data['updates'])>5:
  candidates.append((statistics.mean(x['total_seconds'] for x in data['updates'][1:]),data['arguments']['batch']))
assert candidates,(mode,'no_stable_batch')
print(min(candidates)[1])
PY
 )
 test -n "$batch" || exit 1
 if [[ "$mode" == D1 ]]; then first=15;updates=24;else first=17;updates=16;fi
 for distance in 0 1; do
  number=$((first+distance))
  if (( $(date +%s) >= 1789382216 )); then exit 124; fi
  "$python" -u tmp/pre-density-20260914/benchmark.py --config "configs/docking/$mode-C-T0-RA.yml" --output "$evidence/benchmarks/${number}_${mode}_b${batch}_distance${distance}" --batch "$batch" --workers 32 --seconds 150 --updates "$updates" --distance "$distance" --sample-seed 2024 --deadline 1789382216
  status=$?
  printf 'DISTANCE_COMPARISON number=%s mode=%s batch=%s distance=%s exit=%s\n' "$number" "$mode" "$batch" "$distance" "$status"
 done
done
