#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/theory:$(pwd)/likelihood:${PYTHONPATH:-}"
export HMFAST_COBAYA_USE_GPU=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false COBAYA_NOMPI=1 CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg
mkdir -p logs
echo "=== $(date -u +%H:%M:%S) CNC+YY unrot z0.35 B135 CCCP chain start ==="
/scratch/scratch-lxu/venv/cmbagent_env/bin/python -m cobaya run --force configs/cnc_yy_combined_unrot_z0p35_arnaudB135_Y500c_cccp.yaml >logs/run_cnc_yy_combined_unrot_z0p35_B135_cccp.log 2>&1
echo "=== $(date -u +%H:%M:%S) CNC+YY unrot z0.35 B135 CCCP chain done exit=$? ==="
