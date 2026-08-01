# L1 q-from-map Plot Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a non-overwriting q-from-map mode to every L1 plotting entry point affected by the new masked bandpowers and generate the complete refreshed figure set.

**Architecture:** Keep legacy defaults unchanged and select q-from-map inputs with an explicit CLI option. Reuse existing plotting functions and styles, switch only input tags/metadata/title text/output stems, and validate the resulting files.

**Tech Stack:** Python, NumPy, Matplotlib, argparse, pytest.

## Global Constraints

- Preserve all existing alpha-fixed figures.
- New output stems end in `qfrommap`.
- Read masked spectra from `data_paper/binned_bandpowers` and reuse existing full-sky spectra.
- Do not recompute NaMaster bandpowers.
- Do not refresh theory, rotation, CNC, null-test, or Planck q > 6 figures.

---

### Task 1: Define and test selection-aware plot paths

**Files:**
- Create: `tests/test_l1_qfrommap_plots.py`
- Modify: `scripts/plot_l1_m9_masked_ps_alpha_fixed_1p12.py`
- Modify: `scripts/plot_l1_m9_feedback_bandpowers.py`
- Modify: `scripts/plot_l1_m9_feedback_bandpowers_qgt1.py`
- Modify: `scripts/plot_l1_m9_feedback_ratio_vs_q.py`
- Modify: `paper_results/figures.py`

**Interfaces:**
- Consumes: selection string `qfrommz_alpha_fixed_1p12` or `qfrommap`.
- Produces: selection-aware masked input paths and output stem suffixes.

- [ ] **Step 1: Write failing tests**

Add tests that import each module, call its path helper with `qfrommap`, and
assert the path contains `_qfrommap_`; also assert legacy defaults still contain
`_qfrommz_alpha_fixed_1p12_`.

- [ ] **Step 2: Verify the tests fail**

Run `pytest -q tests/test_l1_qfrommap_plots.py` and confirm selection-aware
arguments are not yet accepted.

- [ ] **Step 3: Implement the minimal CLI/configuration changes**

Add `--selection` to each entry point. Resolve `selection_tag`, display label,
metadata source, and output suffix once in `main()`, then pass those values into
the existing path and plotting helpers. For q-from-map fiducial legends, omit
legacy `f_sky` metadata when a cut was skipped and therefore is absent from the
combined run metadata; never substitute alpha-fixed metadata.

- [ ] **Step 4: Verify focused tests pass**

Run `pytest -q tests/test_l1_qfrommap_plots.py tests/test_feedback_ratio_vs_q.py`.

- [ ] **Step 5: Review the diff**

Run `git diff --check` and confirm no legacy filename or default behavior changed.

### Task 2: Generate and validate the L1 q-from-map figures

**Files:**
- Create: `figures/masked_ps/*qfrommap.{png,pdf}`
- Create: `figures/feedback/*qfrommap*.{png,pdf}`
- Create: `figures/paper/*qfrommap.{png,pdf}`

**Interfaces:**
- Consumes: all 90 completed L1 q-from-map bandpower files.
- Produces: non-empty PNG/PDF pairs for every q-dependent L1 figure family.

- [ ] **Step 1: Preflight inputs**

Count 90 L1 q-from-map text files and check both binnings exist for every
variant/cut combination.

- [ ] **Step 2: Run plotting entry points serially**

Run the four L1 scripts with `--selection qfrommap`, then run
`python -m paper_results.figures --selection qfrommap --figure fiducial` and
`--figure feedback`.

- [ ] **Step 3: Validate outputs**

List every expected q-from-map PNG/PDF pair, assert each file is non-empty, and
load PNG dimensions to catch truncated images.

- [ ] **Step 4: Confirm production health**

Check the L2 tmux worker count, watchdog error count, and available memory after
plotting.

- [ ] **Step 5: Run regression tests**

Run `pytest -q tests/test_l1_qfrommap_plots.py tests/test_feedback_ratio_vs_q.py`
and report the exact output inventory.
