#!/usr/bin/env bash
# Run FLAMINGO RDS builds: all catalogues in parallel || y-maps sequential.
# Catalogues: lightcone-first (build_halo_lightcone_catalogue.py), all L1 + L2 jobs.
# Y-maps: one-at-a-time (CPU-bound healpy).
#
# Safe to restart: individual build scripts use --resume / skip-if-complete.
set -euo pipefail

source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate

REPO=/scratch/scratch-lxu/flamingo_repo
LOGDIR="${REPO}/cobaya/logs"
MASTER_LOG="${LOGDIR}/run_build_flamingo_sequential.log"
CAT_LOG="${LOGDIR}/run_build_flamingo_catalogues_pipeline.log"
MAP_LOG="${LOGDIR}/run_build_flamingo_y_maps_pipeline.log"
mkdir -p "$LOGDIR"

log() {
  echo "$(date -Is) $*" | tee -a "$MASTER_LOG"
}

run_catalogues() {
  {
    echo "$(date -Is) === catalogues pipeline start (all parallel, lightcone-first) ==="
    "${REPO}/scripts/run_build_all_catalogues_parallel.sh"
    echo "$(date -Is) === catalogues pipeline done ==="
  } >>"$CAT_LOG" 2>&1
}

run_y_maps() {
  {
    echo "$(date -Is) === y-maps pipeline start ==="
    echo "$(date -Is) === PHASE: L1_m9 y-maps ==="
    "${REPO}/scripts/run_build_L1_m9_y_maps.sh"
    echo "$(date -Is) === PHASE: L2p8_m9 y-maps ==="
    "${REPO}/scripts/run_build_L2p8_m9_y_maps.sh"
    echo "$(date -Is) === y-maps pipeline done ==="
  } >>"$MAP_LOG" 2>&1
}

log "starting FLAMINGO build (all catalogues parallel || y-maps sequential)"

run_catalogues &
cat_pid=$!
run_y_maps &
map_pid=$!

log "catalogues pipeline PID=${cat_pid} log=${CAT_LOG}"
log "y-maps pipeline PID=${map_pid} log=${MAP_LOG}"

cat_status=0
map_status=0
wait "$cat_pid" || cat_status=$?
wait "$map_pid" || map_status=$?

if [[ "$cat_status" -ne 0 || "$map_status" -ne 0 ]]; then
  log "pipeline failed (catalogues exit=${cat_status}, y-maps exit=${map_status})"
  exit 1
fi

log "dual-pipeline FLAMINGO build finished"
