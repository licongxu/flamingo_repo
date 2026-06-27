#!/usr/bin/env bash
# Sequentially run the 5 pure-gNFW fit-(B, sigma_lnY) chains with the ell>100 cut
# (last 9 of 18 bins). Cosmology fixed to D3A; B uniform, sigma_lnY ~ N(0.173,0.023).
set -e
cd "$(dirname "$0")"
VENV_PY=/scratch/scratch-lxu/venv/cmbagent_env/bin/python
export PYTHONPATH="$(pwd)/theory:$(pwd)/likelihood:${PYTHONPATH:-}"
export HMFAST_COBAYA_USE_GPU=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export COBAYA_NOMPI=1
export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND=Agg
mkdir -p logs

for TAG in fullsky qgt5 qgt10 qgt20 qgt50; do
  CFG=configs/cobaya_yy_${TAG}_fitB_scatter_ellgt100.yaml
  LOG=logs/run_${TAG}.log
  echo "=== $(date -u +%H:%M:%S)  start ${TAG} ==="
  "${VENV_PY}" -m cobaya run --force "${CFG}" >"${LOG}" 2>&1
  echo "=== $(date -u +%H:%M:%S)  done ${TAG} exit=$? ==="
done
echo "=== all 5 scatter chains done ==="
