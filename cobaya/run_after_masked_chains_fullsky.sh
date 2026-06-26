#!/usr/bin/env bash
# Wait for CNC-only and masked CNC+tSZ chains to converge, then launch fullsky combined.
set -e
REPO="/scratch/scratch-lxu/flamingo_repo"
CNC_CKPT="$REPO/chains/cnc_cosmo_arnaudB1_Y500c/cnc_cosmo.checkpoint"
MASK_CKPT="$REPO/chains/cnc_yy_combined_arnaudB1_Y500c/combined.checkpoint"
LOG="$REPO/cobaya/logs/wait_then_fullsky_combined.log"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -u +%H:%M:%S) waiting for CNC-only + masked combined to converge ==="

_converged() {
  local ckpt="$1"
  [[ -f "$ckpt" ]] && grep -q 'converged: true' "$ckpt"
}

while true; do
  cnc_ok=0; mask_ok=0
  _converged "$CNC_CKPT" && cnc_ok=1
  _converged "$MASK_CKPT" && mask_ok=1
  if [[ $cnc_ok -eq 1 && $mask_ok -eq 1 ]]; then
    echo "=== $(date -u +%H:%M:%S) both chains converged — starting fullsky combined ==="
    break
  fi
  echo "$(date -u +%H:%M:%S)  CNC converged=$cnc_ok  masked-combined converged=$mask_ok  (sleep 120s)"
  sleep 120
done

# Ensure full-sky bandpowers exist (yang26-rot Y500c map).
if [[ ! -f "$REPO/data/bandpowers_arnaudB1_Y500c/Dl_yy_fullsky_binned_18_Y500c.txt" ]]; then
  echo "=== computing fullsky Y500c bandpowers ==="
  /scratch/scratch-lxu/venv/cmbagent_env/bin/python \
    "$REPO/scripts/compute_bandpowers_arnaudB1_Y500c.py" --mode fullsky
fi

cd "$REPO/cobaya"
bash run_cnc_yy_combined_fullsky_B1_Y500c.sh
