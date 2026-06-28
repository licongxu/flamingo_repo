#!/usr/bin/env bash
# Build M500c>=1e13, z<=3 halo-lightcone catalogues for all L1_m9 variants.
# Lightcone-first join (build_halo_lightcone_catalogue.py); snaps 17..77.
set -euo pipefail

source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate

REPO=/scratch/scratch-lxu/flamingo_repo
OUTDIR=/rds/rds-lxu/flamingo/L1_m9/catalogues
LOGDIR="${REPO}/cobaya/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

VARIANTS=(
  L1_m9
  fgas+2sigma
  fgas-2sigma
  fgas-4sigma
  fgas-8sigma
  Mstar-1sigma
  Mstar-1sigma_fgas-4sigma
  Jet
  Jet_fgas-4sigma
)

BUILD_SCRIPT="${REPO}/scripts/build_halo_lightcone_catalogue.py"

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

build_variant() {
  local variant="$1"
  local out="${OUTDIR}/halo_catalogue_M500c_1e13_zlt3_${variant}_yang26rot.csv"
  local log="${LOGDIR}/build_catalogue_L1_m9_${variant}.log"
  local yflag=(--no-y-columns)
  # L1_m9 partial build already has Compton-Y columns; keep schema on resume.
  if [[ "${variant}" == "L1_m9" && -f "${out}" ]]; then
    yflag=()
  fi

  if catalogue_running "${out}"; then
    echo "skip ${variant}: build already running"
    return 0
  fi
  local status
  status="$(catalogue_status "${out}" "${variant}")"
  if [[ "${status}" == "complete" ]]; then
    echo "skip ${variant}: complete (${out})"
    return 0
  fi

  echo "=== building catalogue ${variant} -> ${out} (status=${status}) ==="
  python "${BUILD_SCRIPT}" \
    --parent L1_m9 \
    --variant "${variant}" \
    --out "${out}" \
    "${yflag[@]}" \
    --resume \
    2>&1 | tee "${log}"
}

for variant in "${VARIANTS[@]}"; do
  build_variant "${variant}"
done

# Backfill snap 17 for fiducial if an older run started at snap 18.
fb_out="${OUTDIR}/halo_catalogue_M500c_1e13_zlt3_L1_m9_yang26rot.csv"
if [[ -f "${fb_out}.progress.json" ]] && ! catalogue_running "${fb_out}"; then
  done17="$(
    python - <<PY
import json
from pathlib import Path
p = Path("${fb_out}.progress.json")
print(17 in set(json.loads(p.read_text()).get("completed_snapshots", [])))
PY
  )"
  if [[ "${done17}" != "True" ]]; then
    echo "=== backfill L1_m9 snap 17 (z=3.00) -> ${fb_out} ==="
    python "${BUILD_SCRIPT}" \
      --parent L1_m9 \
      --variant L1_m9 \
      --snap-start 17 \
      --snap-stop 17 \
      --out "${fb_out}" \
      --resume \
      2>&1 | tee -a "${LOGDIR}/build_catalogue_L1_m9_L1_m9.log"
  fi
fi

echo "All L1_m9 catalogues done."
