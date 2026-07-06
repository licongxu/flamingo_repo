# fable_5_plan

Planning area for the Fable 5 autonomous research run (execution goes in a fresh
`autoresearch/fable_5_v1/`; this folder is plan + seed diagnostics only).

- `PROJECT_PLAN.md`: the full project plan. Topic: joint cluster number counts +
  masked tSZ power spectrum analysis on FLAMINGO, with a validated joint covariance
  and an optimal masking threshold.
- `seed01_cnc_ps_cross_correlation.py`: seed analysis on cached products (no map access).
  Measures, across the 8 L2p8_m9 lightcone observers, (a) the counts x bandpower Pearson
  correlation and (b) the bandpower realization scatter vs the Knox floor, per masking cut.
- `seed01_fig.png|pdf`, `seed01_results.npz`, `seed01_summary.json`: seed outputs
  (git 6265b89). Headline: masking q > 5 shrinks the ell ~ 100 bandpower scatter from
  26% to 1.4% (factor ~ 340 in variance), near-Gaussian; the cross-correlation estimate is
  inconclusive with 8 shared-box observers and is deferred to the plan's Phase 3.

Reproduce: `source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate &&
python autoresearch/fable_5_plan/seed01_cnc_ps_cross_correlation.py`

## Session traces

The full Fable 5 autonomous research trace is published on HuggingFace:
<https://huggingface.co/datasets/licongxu/fable5-flamingo-research-trace>.

- `fable5_session_trace.jsonl`, `fable5_session_timeline.jsonl`: the main-session
  message and event trace.
- `trace_subagents/`: per-subagent traces (e.g. `explore_repo_subagent.jsonl`) with
  their `.meta.json` sidecars.
