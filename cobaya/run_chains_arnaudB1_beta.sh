#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
VENV_PY=/scratch/scratch-lxu/venv/cmbagent_env/bin/python
export PYTHONPATH="$(pwd)/theory:$(pwd)/likelihood:${PYTHONPATH:-}"
export HMFAST_COBAYA_USE_GPU=1 XLA_PYTHON_CLIENT_PREALLOCATE=false COBAYA_NOMPI=1
export CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg
mkdir -p logs
for TAG in fullsky qgt5 qgt10 qgt20 qgt50; do
  echo "=== $(date -u +%H:%M:%S) start ${TAG} ==="
  "${VENV_PY}" -m cobaya run --force configs/cobaya_yy_${TAG}_arnaudB1_fitBeta.yaml >"logs/run_arnaudB1_beta_${TAG}.log" 2>&1
  echo "=== $(date -u +%H:%M:%S) done ${TAG} exit=$? ==="
done
echo "=== all 5 arnaudB1 fitBeta chains done ==="
