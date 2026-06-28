#!/usr/bin/env bash
# Build M500c>=1e13, z<=3 halo-lightcone catalogues for L2p8_m9 lightcones 0..7.
# Lightcone-first join (build_halo_lightcone_catalogue.py); snaps 18..78.
set -euo pipefail

source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate

REPO=/scratch/scratch-lxu/flamingo_repo
BASE=/rds/rds-lxu/flamingo/L2p8_m9
LOGDIR="${REPO}/cobaya/logs"
mkdir -p "$LOGDIR"

BUILD_SCRIPT="${REPO}/scripts/build_halo_lightcone_catalogue.py"

catalogue_running() {
  pgrep -f "build_halo_lightcone_catalogue.py.*--out ${1}" >/dev/null 2>&1
}

catalogue_status() {
  python "${BUILD_SCRIPT}" \
    --variant L2p8_m9 \
    --observer "$2" \
    --out "$1" \
    --status
}

for obs in $(seq 0 7); do
  outdir="${BASE}/lightcone${obs}/catalogues"
  out="${outdir}/halo_catalogue_M500c_1e13_zlt3_L2p8_m9_yang26rot.csv"
  log="${LOGDIR}/build_catalogue_L2p8_m9_lc${obs}.log"
  mkdir -p "$outdir"

  if catalogue_running "${out}"; then
    echo "skip lc${obs}: build already running"
    continue
  fi
  status="$(catalogue_status "${out}" "${obs}")"
  if [[ "${status}" == "complete" ]]; then
    echo "skip lc${obs}: complete (${out})"
    continue
  fi

  echo "=== building L2p8_m9 lc${obs} catalogue -> ${out} (status=${status}) ==="
  python "${BUILD_SCRIPT}" \
    --variant L2p8_m9 \
    --observer "${obs}" \
    --out "${out}" \
    --no-y-columns \
    --resume \
    2>&1 | tee "${log}"
done

echo "All L2p8_m9 catalogues done."
