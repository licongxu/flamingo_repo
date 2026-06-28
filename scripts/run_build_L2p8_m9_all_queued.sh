#!/usr/bin/env bash
# Wait for in-flight L2p8_m9 builds, then run catalogues + y-maps with current code.
set -euo pipefail

REPO=/scratch/scratch-lxu/flamingo_repo
LOGDIR="${REPO}/cobaya/logs"
LOG="${LOGDIR}/run_build_L2p8_m9_all_queued.log"
mkdir -p "$LOGDIR"

{
  echo "$(date -Is) queued L2p8_m9 build: waiting for in-flight jobs"
  "${REPO}/scripts/wait_for_flamingo_builds.sh" \
    'build_halo_lightcone_catalogue.py' \
    'build_y_map.py.*L2p8_m9' \
    'run_build_L2p8_m9_catalogues.sh' \
    'run_build_L2p8_m9_y_maps.sh'
  echo "$(date -Is) starting L2p8_m9 build (current code)"
  QUEUED_RUN=1 ALLOW_PARALLEL=1 "${REPO}/scripts/run_build_L2p8_m9_all.sh"
  echo "$(date -Is) queued L2p8_m9 build finished"
} >>"$LOG" 2>&1
