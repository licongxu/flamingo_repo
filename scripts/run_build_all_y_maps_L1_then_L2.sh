#!/usr/bin/env bash
# Build all FLAMINGO yang26-rotated Compton-y maps sequentially:
#   1) all L1_m9 variants (lc0, NSIDE=4096, shells 0..59)
#   2) all L2p8_m9 lightcones (lc0..7)
#
# One map at a time (CPU-bound healpy). Safe to restart: child scripts skip
# completed outputs and resume from checkpoints.
set -euo pipefail

source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate

REPO=/scratch/scratch-lxu/flamingo_repo
LOGDIR="${REPO}/cobaya/logs"
LOG="${LOGDIR}/run_build_all_y_maps_L1_then_L2.log"
mkdir -p "$LOGDIR"

exec > >(tee -a "$LOG") 2>&1

echo "$(date -Is) === y-maps: L1_m9 first, then L2p8_m9 ==="

echo "$(date -Is) === PHASE 1/2: L1_m9 y-maps ==="
"${REPO}/scripts/run_build_L1_m9_y_maps.sh"

echo "$(date -Is) === PHASE 2/2: L2p8_m9 y-maps ==="
"${REPO}/scripts/run_build_L2p8_m9_y_maps.sh"

echo "$(date -Is) === all y-maps done (L1 then L2) ==="
