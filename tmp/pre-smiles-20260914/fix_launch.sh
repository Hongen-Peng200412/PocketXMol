set -euo pipefail
python3 - <<'PY'
from pathlib import Path
root=Path('/storage/penghongen/tmp/pxm_pre_smiles_20260914')
p=root/'code/训练与运行/runtime/create_launch.sh'
(root/'create_launch_lf.sh').write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
c=Path('/home/penghongen/Feedback/Pocket_Plus/allocations/371591/run_cmd_371591.sh')
(root/'launch_cpu_attempt1.sh').write_bytes(c.read_bytes())
c.write_text(c.read_text().replace('$TASK_PROJECT_ROOT/训练与运行/runtime/create_launch.sh','/storage/penghongen/tmp/pxm_pre_smiles_20260914/create_launch_lf.sh'))
PY
test -f /home/penghongen/Feedback/Pocket_Plus/allocations/371591/after_lock_371591
rm /home/penghongen/Feedback/Pocket_Plus/allocations/try_lock_371591
