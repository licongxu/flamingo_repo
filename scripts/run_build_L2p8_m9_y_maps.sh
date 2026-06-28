#!/usr/bin/env bash
# Build yang26-rotated Compton-y maps (NSIDE=4096, z<=3 shells) for L2p8_m9 lightcones 0..7.
#
# Intended to run AFTER run_build_L1_m9_y_maps.sh (or via run_build_all_y_maps_L1_then_L2.sh).
# Do not launch in parallel with L1 y-map builds.
set -euo pipefail

source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate

REPO=/scratch/scratch-lxu/flamingo_repo
BASE=/rds/rds-lxu/flamingo/L2p8_m9
CKPT=/scratch/scratch-lxu/flamingo_map_build
LOGDIR="${REPO}/cobaya/logs"
mkdir -p "$CKPT" "$LOGDIR"

ymap_running() {
  pgrep -f "build_y_map.py.*--out ${1}" >/dev/null 2>&1
}

for obs in $(seq 0 7); do
  outdir="${BASE}/lightcone${obs}/healpix_map"
  out="${outdir}/y_unlensed_L2p8_m9_lc${obs}.fits"
  log="${LOGDIR}/build_y_map_L2p8_m9_lc${obs}.log"
  mkdir -p "$outdir"
  if [[ -f "$out" ]]; then
    echo "skip lc${obs}: already exists at ${out}"
    continue
  fi
  if ymap_running "${out}"; then
    echo "skip lc${obs}: y-map build already running"
    continue
  fi
  echo "=== building L2p8_m9 lc${obs} y-map -> ${out} ==="
  python "${REPO}/scripts/build_y_map.py" \
    --variant L2p8_m9 \
    --observer "${obs}" \
    --nside 4096 \
    --shell-max 60 \
    --out "${out}" \
    --ckpt-dir "${CKPT}" \
    2>&1 | tee "${log}"
done

echo "All L2p8_m9 y-maps done."
