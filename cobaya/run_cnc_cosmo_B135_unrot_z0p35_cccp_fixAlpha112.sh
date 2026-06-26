#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/theory:$(pwd)/likelihood:${PYTHONPATH:-}"
export HMFAST_COBAYA_USE_GPU=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false COBAYA_NOMPI=1 CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg
mkdir -p logs
echo "=== $(date -u +%H:%M:%S) CNC cosmo B135 unrot z0.35 CCCP fixAlpha112 chain start ==="
/scratch/scratch-lxu/venv/cmbagent_env/bin/python -m cobaya run --force configs/cnc_cosmo_arnaudB135_Y500c_unrot_z0p35_cccp_fixAlpha112.yaml >logs/run_cnc_cosmo_B135_unrot_z0p35_cccp_fixAlpha112.log 2>&1
echo "=== $(date -u +%H:%M:%S) CNC cosmo B135 unrot z0.35 CCCP fixAlpha112 chain done exit=$? ==="
