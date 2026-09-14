set -euo pipefail
allocation=/home/penghongen/Feedback/Pocket_Plus/allocations/371591
ready=/home/penghongen/Feedback/Pocket_Plus/allocations/try_lock_371591
test -f "$ready"
test -f "$allocation/after_lock_371591"
test ! -f "$allocation/kill_lock_371591"
cp "$allocation/run_cmd_371591.sh" /storage/penghongen/tmp/pxm_pre_smiles_20260914/previous_run_cmd.sh
cat > "$allocation/run_cmd_371591.sh" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
export TASK_PROJECT_ROOT=/storage/penghongen/tmp/pxm_pre_smiles_20260914/code
export EXPERIMENT_FEEDBACK_ROOT=/home/penghongen/Feedback/PocketXMol
export TASK_RUN_STAMP="pre_smiles_cpu_${SLURM_JOB_ID}_$(date '+%Y%m%dT%H%M%S')"
export TASK_RESOURCE_TYPE=a800 TASK_NODE_COUNT=1 TASK_GPUS_PER_NODE=1 TASK_CPU_COUNT=16 TASK_ARRAY_SPEC=""
export TASK_LAUNCH_DIR
TASK_LAUNCH_DIR="$(bash "$TASK_PROJECT_ROOT/训练与运行/runtime/create_launch.sh" "$EXPERIMENT_FEEDBACK_ROOT" "$TASK_RUN_STAMP" /home/penghongen/Feedback/Pocket_Plus/allocations/371591/run_cmd_371591.sh tmp/pre-smiles-20260914/migrate.py)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$TASK_PROJECT_ROOT"
printf 'PRE_SMILES_CPU base=a78ae8f launch=%s\n' "$TASK_LAUNCH_DIR"
exec /storage/penghongen/PocketXMol/runtime/venv/bin/python -u tmp/pre-smiles-20260914/migrate.py > /storage/penghongen/tmp/pxm_pre_smiles_20260914/cpu.out 2> /storage/penghongen/tmp/pxm_pre_smiles_20260914/cpu.err
RUN
rm "$ready"
printf 'submitted 371591 pre_smiles_cpu\n'
