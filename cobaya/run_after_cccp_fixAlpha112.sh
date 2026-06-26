#!/usr/bin/env bash
# Wait for all six B135 CCCP chains to converge, then launch fixAlpha112 variants.
set -e
cd "$(dirname "$0")"
REPO="/scratch/scratch-lxu/flamingo_repo/chains"
LOG="logs/wait_cccp_then_fixAlpha112.log"
mkdir -p logs

chains=(
  cnc_cosmo_arnaudB135_Y500c_cccp
  yy_fullsky_arnaudB135_Y500c_cccp
  cnc_yy_combined_fullsky_arnaudB135_Y500c_cccp
  cnc_cosmo_arnaudB135_Y500c_unrot_z0p35_cccp
  yy_unrot_z0p35_arnaudB135_Y500c_cccp
  cnc_yy_combined_unrot_z0p35_arnaudB135_Y500c_cccp
)

echo "=== $(date -u +%H:%M:%S) waiting for CCCP chains ===" | tee -a "$LOG"
while true; do
  done=0
  for d in "${chains[@]}"; do
    ckpt=$(ls "$REPO/$d"/*.checkpoint 2>/dev/null | head -1)
    if [ -f "$ckpt" ] && grep -q "converged: true" "$ckpt"; then
      done=$((done + 1))
    fi
  done
  echo "$(date -u +%H:%M:%S) converged $done / ${#chains[@]}" | tee -a "$LOG"
  if [ "$done" -eq "${#chains[@]}" ]; then
    break
  fi
  sleep 300
done

echo "=== $(date -u +%H:%M:%S) launching fixAlpha112 chains ===" | tee -a "$LOG"
for s in run_cnc_cosmo_B135_cccp_fixAlpha112.sh \
         run_cnc_yy_combined_fullsky_B135_cccp_fixAlpha112.sh \
         run_yy_fullsky_B135_cccp_fixAlpha112.sh \
         run_cnc_cosmo_B135_unrot_z0p35_cccp_fixAlpha112.sh \
         run_cnc_yy_combined_unrot_z0p35_B135_cccp_fixAlpha112.sh \
         run_yy_unrot_z0p35_B135_cccp_fixAlpha112.sh; do
  nohup bash "$s" >> "$LOG" 2>&1 &
done
echo "=== $(date -u +%H:%M:%S) all fixAlpha112 jobs submitted ===" | tee -a "$LOG"
