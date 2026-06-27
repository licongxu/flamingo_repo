#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/theory:$(pwd)/likelihood:${PYTHONPATH:-}"
export HMFAST_COBAYA_USE_GPU=0 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false COBAYA_NOMPI=1 MPLBACKEND=Agg
mkdir -p logs
echo "=== $(date -u +%H:%M:%S) CNC cosmo B135 D3A nuisance chain start ==="
/scratch/scratch-lxu/venv/cmbagent_env/bin/python -m cobaya run --force configs/cnc_cosmo_arnaudB135_Y500c_d3aNuisance.yaml >logs/run_cnc_cosmo_B135_Y500c_d3aNuisance.log 2>&1
echo "=== $(date -u +%H:%M:%S) CNC cosmo B135 D3A nuisance chain done exit=$? ==="
