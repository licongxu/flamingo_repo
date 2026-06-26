#!/usr/bin/env bash
# Wait for CCCP chains (omega_b fixed, 1-b sampled) to finish, then launch B=1 fixOb runs.
set -e
REPO="/scratch/scratch-lxu/flamingo_repo"
CNC_CKPT="$REPO/chains/cnc_cosmo_arnaudB1_Y500c_cccp/cnc_cosmo.checkpoint"
COMB_CKPT="$REPO/chains/cnc_yy_combined_fullsky_arnaudB1_Y500c/combined.checkpoint"
LOG="$REPO/cobaya/logs/wait_then_fixObB1.log"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -u +%H:%M:%S) waiting for CCCP CNC + fullsky combined to finish ==="

_finished() {
  local ckpt="$1"
  [[ -f "$ckpt" ]] && grep -q 'converged: true' "$ckpt"
}

while true; do
  cnc_ok=0; comb_ok=0
  _finished "$CNC_CKPT" && cnc_ok=1
  _finished "$COMB_CKPT" && comb_ok=1
  if [[ $cnc_ok -eq 1 && $comb_ok -eq 1 ]]; then
    echo "=== $(date -u +%H:%M:%S) both CCCP chains converged — starting fixObB1 runs ==="
    break
  fi
  echo "$(date -u +%H:%M:%S)  CCCP CNC converged=$cnc_ok  CCCP combined converged=$comb_ok  (sleep 120s)"
  sleep 120
done

cd "$REPO/cobaya"
bash run_cnc_yy_combined_fullsky_B1_Y500c_fixObB1.sh &
bash run_cnc_cosmo_B1_Y500c_fixObB1.sh &
wait
echo "=== $(date -u +%H:%M:%S) fixObB1 runs finished ==="
