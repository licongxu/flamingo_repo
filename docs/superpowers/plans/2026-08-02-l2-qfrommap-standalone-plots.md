# L2 q-from-map Standalone Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce standalone L2p8_m9 lightcone-0 masked tSZ figures from empirical aperture-q bandpowers.

**Architecture:** Extend the existing L2 plotting script with a selection argument and small path helpers. The legacy branch retains its paths; the q-from-map branch points to canonical lightcone-0 bandpowers and writes separately tagged figures.

**Tech Stack:** Python, NumPy, Matplotlib, pytest.

## Global Constraints

- Use fiducial L2p8_m9 `lightcone0`.
- Plot `q > 50, 20, 10, 5, 1`.
- Do not recompute NaMaster spectra.
- Do not overwrite legacy figures.

---

### Task 1: Selection-aware L2 plotting

**Files:**
- Modify: `scripts/plot_l2p8_m9_masked_ps_alpha_fixed_1p12.py`
- Modify: `tests/test_qfrommap_masked_ps.py`

**Interfaces:**
- Consumes: existing q-from-map bandpower text files and metadata JSON.
- Produces: `_masked_paths(tag, log, selection_tag)` and tagged output figures.

- [ ] **Step 1: Write the failing path test**

Assert that q-from-map resolves to `Dl_yy_L2p8_m9_lc0_masked_qgt5_qfrommap_binned_18.txt` and that the output stem ends in `_qfrommap`.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_qfrommap_masked_ps.py -k standalone_l2_plot` and expect failure because the current helper has no selection argument.

- [ ] **Step 3: Implement the minimal selection branch**

Add `--selection`, route q-from-map reads to the lightcone-0 files and metadata, adjust the title, and use `_qfrommap` output stems.

- [ ] **Step 4: Verify GREEN**

Run `pytest -q tests/test_qfrommap_masked_ps.py -k standalone_l2_plot` and expect one passing test.

- [ ] **Step 5: Render and validate outputs**

Run `python scripts/plot_l2p8_m9_masked_ps_alpha_fixed_1p12.py --selection qfrommap`, then verify all five input arrays are finite and both PNG files exceed 10 kB.

- [ ] **Step 6: Run focused regression tests**

Run `pytest -q tests/test_qfrommap_masked_ps.py tests/test_l1_qfrommap_plots.py tests/test_qfrommap_threshold_completion.py` and expect zero failures.

