#!/usr/bin/env bash
# 在指定节点只读核查控制器及本次GPU；不发信号、不修改锁。
set -euo pipefail
'/storage/penghongen/PocketXMol/runtime/venv/bin/python' - <<'PY'
import json,os,subprocess,time
from pathlib import Path
import psutil
control=Path('/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration/运行日志与统计/slurm/allocations/379402')
result={'checked_epoch':time.time(),'hostname':os.uname().nodename,'after_lock':(control/'after_lock_379402').exists(),'try_lock':(control.parent/'try_lock_379402').exists(),'kill_lock':(control/'kill_lock_379402').exists()}
try:
 controller=psutil.Process(2383)
 result['controller']={'pid':2383,'cmdline':controller.cmdline(),'status':controller.status(),'children':[{'pid':p.pid,'name':p.name(),'cmdline':p.cmdline()[:8]} for p in controller.children(recursive=True)]}
except psutil.Error as error:result['controller_error']=repr(error)
result['assigned_gpu']=subprocess.check_output(['nvidia-smi','--id=GPU-adbf8fc8-5a4a-87e3-853b-c9cadcbdf74b','--query-gpu=uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
all_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
result['assigned_gpu_processes']=[line for line in all_apps.splitlines() if 'GPU-adbf8fc8-5a4a-87e3-853b-c9cadcbdf74b' in line]
print(json.dumps(result,ensure_ascii=False))
PY
