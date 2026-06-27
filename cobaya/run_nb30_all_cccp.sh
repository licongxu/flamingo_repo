#!/usr/bin/env bash
# nb30: run remaining CCCP chains in parallel on both GPUs.
# YY B135 is skipped if already converged (check progress file).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

REPO="/scratch/scratch-lxu/flamingo_repo/chains"
pids=()

start() {
  local gpu="$1"
  local script="$2"
  local log="$3"
  echo "=== $(date -u +%H:%M:%S) GPU${gpu} -> ${script} ==="
  CUDA_VISIBLE_DEVICES="${gpu}" nohup bash "${script}" >"logs/${log}" 2>&1 &
  pids+=("$!")
}

# YY B135: skip if converged with new omega_b run already done today
if grep -q "The run has converged" logs/run_yy_fullsky_B135_cccp.log 2>/dev/null; then
  echo "skip yy B135 (already converged)"
else
  start 0 run_yy_fullsky_B135_cccp.sh run_yy_fullsky_B135_cccp.log
fi

# CNC B135: skip if already running
if pgrep -f "configs/cnc_cosmo_arnaudB135_Y500c_cccp.yaml" >/dev/null; then
  echo "skip cnc B135 (already running pid $(pgrep -f 'configs/cnc_cosmo_arnaudB135_Y500c_cccp.yaml'))"
else
  start 1 run_cnc_cosmo_B135_cccp.sh run_cnc_cosmo_B135_cccp.log
fi

start 0 run_cnc_yy_combined_fullsky_B135_cccp.sh run_cnc_yy_combined_fullsky_B135_cccp.log
start 0 run_cnc_cosmo_B1_Y500c_cccp.sh run_cnc_cosmo_B1_Y500c_cccp.log
start 1 run_cnc_yy_combined_fullsky_B1_Y500c_cccp.sh run_cnc_yy_combined_fullsky_B1_Y500c_cccp.log

echo "=== launched PIDs: ${pids[*]} ==="
for pid in "${pids[@]}"; do
  wait "${pid}" || echo "job pid ${pid} failed"
done
echo "=== nb30 CCCP parallel batch done $(date -u) ==="
