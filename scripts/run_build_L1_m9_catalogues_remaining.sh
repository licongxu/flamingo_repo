#!/usr/bin/env bash
# Build remaining L1_m9 catalogues (all variants except fiducial L1_m9).
# Uses lightcone-first join via build_halo_lightcone_catalogue.py.
set -euo pipefail

source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate

REPO=/scratch/scratch-lxu/flamingo_repo
OUTDIR=/rds/rds-lxu/flamingo/L1_m9/catalogues
LOGDIR="${REPO}/cobaya/logs"
BUILD_SCRIPT="${REPO}/scripts/build_halo_lightcone_catalogue.py"
mkdir -p "$OUTDIR" "$LOGDIR"

VARIANTS=(
  fgas+2sigma
  fgas-2sigma
  fgas-4sigma
  fgas-8sigma
  Mstar-1sigma
  Mstar-1sigma_fgas-4sigma
  Jet
  Jet_fgas-4sigma
)

catalogue_running() {
  pgrep -f "build_halo_lightcone_catalogue.py.*--out ${1}" >/dev/null 2>&1
}

catalogue_status() {
  python "${BUILD_SCRIPT}" \
    --parent L1_m9 \
    --variant "$2" \
    --out "$1" \
    --status
}

for variant in "${VARIANTS[@]}"; do
  out="${OUTDIR}/halo_catalogue_M500c_1e13_zlt3_${variant}_yang26rot.csv"
  log="${LOGDIR}/build_catalogue_L1_m9_${variant}.log"

  if catalogue_running "${out}"; then
    echo "skip ${variant}: build already running"
    continue
  fi
  status="$(catalogue_status "${out}" "${variant}")"
  if [[ "${status}" == "complete" ]]; then
    echo "skip ${variant}: complete (${out})"
    continue
  fi

  echo "=== building catalogue ${variant} -> ${out} (status=${status}) ==="
  python "${BUILD_SCRIPT}" \
    --parent L1_m9 \
    --variant "${variant}" \
    --out "${out}" \
    --no-y-columns \
    --resume \
    2>&1 | tee "${log}"
done

echo "Remaining L1_m9 catalogues done."
