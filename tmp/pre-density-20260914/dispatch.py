"""生成本次资源的动态命令；只解锁已核实的379402，不触碰after/kill。"""
from pathlib import Path
import argparse,shlex
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
task=Path(__file__).resolve().parent
remote='/storage/penghongen/tmp/pocketxmol_density_20260914'
control='/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration/运行日志与统计/slurm/allocations/379402'
command=' '.join(shlex.quote(x) for x in a.command)
body=f'''#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR='{remote}/scratch'
mkdir -p "$TMPDIR"
release_project_root="$(bash '{remote}/source/训练与运行/runtime/create_release.sh' '{remote}/source' '{remote}/releases')"
export TASK_PROJECT_ROOT="$release_project_root"
export TASK_RUN_STAMP='{a.name}'
bash "$TASK_PROJECT_ROOT/训练与运行/runtime/create_launch.sh" '{remote}' '{a.name}' '{control}/run_cmd_379402.sh' 'tmp/pre-density-20260914/benchmark.py'
cd "$TASK_PROJECT_ROOT"
export PYTHONPATH='{remote}/deps':"$TASK_PROJECT_ROOT"
printf 'DENSITY_EXEC name={a.name} release=%s job=%s cuda=%s\\n' "$TASK_PROJECT_ROOT" "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
{command}
'''
script=f'''#!/usr/bin/env bash
set -euo pipefail
control='{control}'
test -f "$control/../try_lock_379402"
test -f "$control/after_lock_379402"
test ! -e "$control/kill_lock_379402"
mkdir -p '{remote}/control'
cp "$control/run_cmd_379402.sh" '{remote}/control/{a.name}_previous_run_cmd.sh'
cat > "$control/run_cmd_379402.sh" <<'PXM_DENSITY_COMMAND'
{body}PXM_DENSITY_COMMAND
cp "$control/run_cmd_379402.sh" '{remote}/control/{a.name}_run_cmd.sh'
chmod +x "$control/run_cmd_379402.sh"
rm -- "$control/../try_lock_379402"
printf 'DISPATCHED {a.name}\\n'
'''
(task/'dispatch.sh').write_text(script,encoding='utf-8',newline='\n')
