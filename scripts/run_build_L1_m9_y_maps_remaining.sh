#!/usr/bin/env bash
# Build remaining L1_m9 y-maps (all variants except fiducial L1_m9).
# Safe to run in parallel with an in-flight fiducial y-map build.
set -euo pipefail

source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate

REPO=/scratch/scratch-lxu/flamingo_repo
OUTDIR=/rds/rds-lxu/flamingo/L1_m9/maps
CKPT=/scratch/scratch-lxu/flamingo_map_build
LOGDIR="${REPO}/cobaya/logs"
mkdir -p "$OUTDIR" "$CKPT" "$LOGDIR"

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

ymap_running() {
  pgrep -f "build_y_map.py.*--out ${1}" >/dev/null 2>&1
}

for variant in "${VARIANTS[@]}"; do
  out="${OUTDIR}/y_unlensed_${variant}_lc0_nside4096.fits"
  log="${LOGDIR}/build_y_map_L1_m9_${variant}.log"
  if [[ -f "$out" ]]; then
    echo "skip ${variant}: already exists at ${out}"
    continue
  fi
  if ymap_running "${out}"; then
    echo "skip ${variant}: y-map build already running"
    continue
  fi
  echo "=== building ${variant} -> ${out} ==="
  python "${REPO}/scripts/build_y_map.py" \
    --parent L1_m9 \
    --variant "${variant}" \
    --observer 0 \
    --nside 4096 \
    --shell-max 60 \
    --out "${out}" \
    --ckpt-dir "${CKPT}" \
    2>&1 | tee "${log}"
done

echo "Remaining L1_m9 y-maps done."
