#!/usr/bin/env bash
# Wait for in-flight L1_m9 builds, then run catalogues with the current code.
set -euo pipefail

REPO=/scratch/scratch-lxu/flamingo_repo
LOGDIR="${REPO}/cobaya/logs"
LOG="${LOGDIR}/run_build_L1_m9_catalogues_queued.log"
mkdir -p "$LOGDIR"

{
  echo "$(date -Is) queued L1_m9 catalogue build: waiting for in-flight jobs"
  "${REPO}/scripts/wait_for_flamingo_builds.sh" \
    'build_halo_lightcone_catalogue.py' \
    'run_build_L1_m9_catalogues.sh'
  echo "$(date -Is) starting L1_m9 catalogue build (current code)"
  QUEUED_RUN=1 "${REPO}/scripts/run_build_L1_m9_catalogues.sh"
  echo "$(date -Is) queued L1_m9 catalogue build finished"
} >>"$LOG" 2>&1
