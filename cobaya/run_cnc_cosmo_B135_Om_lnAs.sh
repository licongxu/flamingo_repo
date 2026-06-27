#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/theory:$(pwd)/likelihood:${PYTHONPATH:-}"
export HMFAST_COBAYA_USE_GPU=1 XLA_PYTHON_CLIENT_PREALLOCATE=false COBAYA_NOMPI=1 CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg
mkdir -p logs
echo "=== $(date -u +%H:%M:%S) CNC cosmo B=1.35 Om_lnAs chain start ==="
/scratch/scratch-lxu/venv/cmbagent_env/bin/python -m cobaya run --force configs/cnc_cosmo_arnaudB135_Om_lnAs.yaml >logs/run_cnc_cosmo_B135_Om_lnAs.log 2>&1
echo "=== $(date -u +%H:%M:%S) CNC cosmo B=1.35 Om_lnAs chain done exit=$? ==="
