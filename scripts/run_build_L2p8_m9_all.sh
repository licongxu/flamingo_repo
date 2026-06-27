#!/usr/bin/env bash
# Build L2p8_m9 lightcones 0..7: catalogues first, then y-maps.
set -euo pipefail

REPO=/scratch/scratch-lxu/flamingo_repo
LOGDIR="${REPO}/cobaya/logs"
mkdir -p "$LOGDIR"

echo "=== Phase 1: catalogues (lc0..lc7) ==="
"${REPO}/scripts/run_build_L2p8_m9_catalogues.sh" \
  2>&1 | tee "${LOGDIR}/run_build_L2p8_m9_catalogues_all.log"

echo "=== Phase 2: y-maps (lc0..lc7) ==="
"${REPO}/scripts/run_build_L2p8_m9_y_maps.sh" \
  2>&1 | tee "${LOGDIR}/run_build_L2p8_m9_y_maps_all.log"

echo "All L2p8_m9 lightcone builds done."
