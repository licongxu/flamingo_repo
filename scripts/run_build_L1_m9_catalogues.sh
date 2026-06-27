#!/usr/bin/env bash
# Build M500c>=1e13, z<=3 halo-lightcone catalogues for all L1_m9 variants.
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

for variant in "${VARIANTS[@]}"; do
  out="${OUTDIR}/halo_catalogue_M500c_1e13_zlt3_${variant}_yang26rot.csv"
  log="${LOGDIR}/build_catalogue_L1_m9_${variant}.log"
  if [[ -f "$out" && -f "${out}.progress.json" ]]; then
    echo "skip ${variant}: already exists at ${out}"
    continue
  fi
  echo "=== building catalogue ${variant} -> ${out} ==="
  python "${REPO}/scripts/build_halo_catalogue_M500c_1e13_zlt3.py" \
    --parent L1_m9 \
    --variant "${variant}" \
    --out "${out}" \
    --resume \
    2>&1 | tee "${log}"
done

echo "All L1_m9 catalogues done."
