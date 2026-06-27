#!/usr/bin/env bash
# Build M500c>=1e13, z<3 halo-lightcone catalogues for L2p8_m9 lightcones 0..7.
set -euo pipefail

source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate

REPO=/scratch/scratch-lxu/flamingo_repo
BASE=/rds/rds-lxu/flamingo/L2p8_m9
LOGDIR="${REPO}/cobaya/logs"
mkdir -p "$LOGDIR"

for obs in $(seq 0 7); do
  outdir="${BASE}/lightcone${obs}/catalogues"
  out="${outdir}/halo_catalogue_M500c_1e13_zlt3_L2p8_m9_yang26rot.csv"
  log="${LOGDIR}/build_catalogue_L2p8_m9_lc${obs}.log"
  mkdir -p "$outdir"
  if [[ -f "$out" && -f "${out}.progress.json" ]]; then
    echo "skip lc${obs}: already exists at ${out}"
    continue
  fi
  echo "=== building L2p8_m9 lc${obs} catalogue -> ${out} ==="
  python "${REPO}/scripts/build_halo_catalogue_M500c_1e13_zlt3.py" \
    --variant L2p8_m9 \
    --observer "${obs}" \
    --out "${out}" \
    --resume \
    2>&1 | tee "${log}"
done

echo "All L2p8_m9 catalogues done."
