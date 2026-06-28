#!/usr/bin/env bash
# Queue an L2p8_m9 catalogue build to run after any in-flight build finishes.
# Does not kill running jobs. Pass through env, e.g.:
#   REBUILD_OBS="0" ./queue_l2p8_m9_catalogues.sh
set -euo pipefail

REPO=/scratch/scratch-lxu/flamingo_repo
LOGDIR="${REPO}/cobaya/logs"
QUEUE_LOG="${LOGDIR}/queue_l2p8_m9_catalogues.log"
mkdir -p "$LOGDIR"

{
  echo "$(date -Is) queued L2p8_m9 catalogue build (REBUILD_OBS=${REBUILD_OBS:-})"
  "${REPO}/scripts/wait_for_l2p8_catalogue_build.sh"
  QUEUED_RUN=1 REBUILD_OBS="${REBUILD_OBS:-}" \
    "${REPO}/scripts/run_build_L2p8_m9_catalogues.sh"
  echo "$(date -Is) queued catalogue build finished"
} >> "$QUEUE_LOG" 2>&1 &

echo "queued (log: ${QUEUE_LOG}, PID=$!)"
