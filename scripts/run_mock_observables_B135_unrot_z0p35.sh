#!/usr/bin/env bash
# Full mock-observable pipeline: B=1.35, unrotated z<0.35 map volume.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "=== [1/4] CNC N(z,q) binning (z_max=0.35, 7 bins) ==="
python scripts/bin_cnc_arnaudB135_Y500c_unrot_z0p35.py

echo "=== [2/4] tSZ bandpowers: qgt5 ==="
python scripts/compute_bandpowers_arnaudB135_Y500c_unrot_z0p35.py --mode qgt5

echo "=== [3/4] tSZ bandpowers: fullsky ==="
python scripts/compute_bandpowers_arnaudB135_Y500c_unrot_z0p35.py --mode fullsky

echo "=== [4/4] theory covariance (GPU) ==="
export JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0
/scratch/scratch-lxu/venv/cmbagent_env/bin/python scripts/compute_arnaudB135_unrot_z0p35_full_cov.py

echo "=== DONE ==="
echo "CNC:       data/cnc/N2d_z_q_bin_arnaudB135_Y500c_unrot_z0p35.txt"
echo "Bandpowers: data/bandpowers_arnaudB135_Y500c_unrot_z0p35/"
echo "Covariance: data/theory_cov_arnaudB135_unrot_z0p35/"
