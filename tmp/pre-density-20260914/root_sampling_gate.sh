#!/usr/bin/env bash
set -euo pipefail
evidence='/storage/penghongen/tmp/pocketxmol_density_20260914'
source_root="$evidence/sampling_gate_0a9c849/source"
release_root=$(bash "$source_root/训练与运行/runtime/create_release.sh" "$source_root" "$evidence/releases")
(
 export TASK_PROJECT_ROOT="$release_root" TASK_RUN_STAMP='density_sampling_gate_0a9c849'
 export PYTHONPATH="$evidence/deps:$release_root"
 bash "$release_root/训练与运行/runtime/create_launch.sh" "$evidence" "$TASK_RUN_STAMP" '/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration/运行日志与统计/slurm/allocations/379402/run_cmd_379402.sh' tests/test_density_sampling.py
 cd "$release_root"
 /storage/penghongen/PocketXMol/runtime/venv/bin/python -c 'import torch; assert torch.cuda.is_available() and torch.backends.cuda.is_flash_attention_available(); print("A800_REAL_FLASH_SAMPLING_GATE", torch.__version__, flush=True)'
 timeout --signal=TERM --kill-after=15s 300s /storage/penghongen/PocketXMol/runtime/venv/bin/python -m pytest tests/test_density_sampling.py -q | tee "$evidence/root_sampling_gate_0a9c849.log"
)
