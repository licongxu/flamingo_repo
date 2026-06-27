#!/usr/bin/env bash
# Sequentially run the 5 pure-gNFW fit-beta chains (ell>100 cut). FIXED: D3A
# cosmology, B=1, sigma_lnY=0.173, P0/c500/alpha/gamma (Arnaud). SAMPLED: beta ~ U(3,8).
set -e
cd "$(dirname "$0")"
VENV_PY=/scratch/scratch-lxu/venv/cmbagent_env/bin/python
export PYTHONPATH="$(pwd)/theory:$(pwd)/likelihood:${PYTHONPATH:-}"
export HMFAST_COBAYA_USE_GPU=1 XLA_PYTHON_CLIENT_PREALLOCATE=false COBAYA_NOMPI=1
export CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg
mkdir -p logs
for TAG in fullsky qgt5 qgt10 qgt20 qgt50; do
  CFG=configs/cobaya_yy_${TAG}_fitBeta_ellgt100.yaml
  echo "=== $(date -u +%H:%M:%S) start ${TAG} ==="
  "${VENV_PY}" -m cobaya run --force "${CFG}" >"logs/run_beta_${TAG}.log" 2>&1
  echo "=== $(date -u +%H:%M:%S) done ${TAG} exit=$? ==="
done
echo "=== all 5 fitBeta chains done ==="
