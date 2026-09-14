#!/usr/bin/env bash
# 包络入口短检查, 再让已选D4中心组合运行超过预取容量的持续窗口。
set -uo pipefail
python='/storage/penghongen/PocketXMol/runtime/venv/bin/python'
evidence='/storage/penghongen/tmp/pocketxmol_density_20260914'
read -r d1_workers distance < <("$python" - "$evidence" <<'PY'
import json,sys,statistics
from pathlib import Path
root=Path(sys.argv[1]);candidates=[]
for number in (19,20):
 names=list((root/'benchmarks').glob(f'{number}_*/result.json'));assert len(names)==1,names
 data=json.load(open(names[0]))
 if data['status']=='complete' and data['parameters_finite'] and len(data['updates'])>32:
  tail=data['updates'][len(data['updates'])//2:]
  candidates.append((statistics.mean(x['total_seconds'] for x in tail),data['arguments']['workers']))
assert candidates,'D1持续窗口未形成稳定组合;先报告后再决定'
decision=json.load(open(root/'distance_decision.json'))
print(min(candidates)[1],int(decision['keep_distance']))
PY
)
test -n "$d1_workers" && test -n "$distance" || exit 1
for spec in "21 D1 E 72 $d1_workers 4 120 2027" '22 D4 E 36 32 4 180 2028' '23 D4 C 36 32 48 285 2029'; do
 read -r number mode context batch workers updates seconds seed <<<"$spec"
 if (( $(date +%s) >= 1789382216 )); then exit 124; fi
 "$python" -u tmp/pre-density-20260914/benchmark.py --config "configs/docking/$mode-$context-T0-RA.yml" --output "$evidence/benchmarks/${number}_${mode}_${context}_b${batch}_w${workers}_seed${seed}" --batch "$batch" --workers "$workers" --seconds "$seconds" --updates "$updates" --distance "$distance" --sample-seed "$seed" --deadline 1789382216
 status=$?
 printf 'FINAL_VALIDATION number=%s mode=%s context=%s exit=%s\n' "$number" "$mode" "$context" "$status"
done
