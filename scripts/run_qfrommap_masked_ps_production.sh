#!/usr/bin/env bash
set -euo pipefail

source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export FLAMINGO_ROOT=/scratch/scratch-lxu/flamingo_repo

repo=/scratch/scratch-lxu/flamingo_repo
worktree="$repo/.worktrees/stable-halo-catalogue-rebuild"
bandpowers="$repo/data_paper/binned_bandpowers"

cd "$worktree"
python scripts/compute_l1_m9_feedback_bandpowers.py \
  --selection qfrommap --workers 8

l1_count=$(find "$bandpowers" -maxdepth 1 -type f \
  -name 'Dl_yy_L1_m9*masked*qfrommap*.txt' | wc -l)
if [[ "$l1_count" -ne 90 ]]; then
  echo "L1 completeness gate failed: $l1_count/90"
  exit 1
fi
echo "L1 completeness gate passed: $l1_count/90"

python scripts/compute_l2p8_m9_masked_ps_alpha_fixed_1p12.py \
  --selection qfrommap --workers 8

l2_count=$(find "$bandpowers" -maxdepth 1 -type f \
  -name 'Dl_yy_L2p8_m9_lc*_masked_*_qfrommap_*.txt' | wc -l)
if [[ "$l2_count" -ne 80 ]]; then
  echo "L2 completeness gate failed: $l2_count/80"
  exit 1
fi
echo "L2 completeness gate passed: $l2_count/80"

python scripts/plot_l1_l2p8_m9_masked_ps_comparison.py --selection qfrommap
