# Repository Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repeated scientific script implementations with tested package functions and one canonical command per workflow, while preserving every numerical result and stored product.

**Architecture:** Exact mask, NaMaster, and bandpower operations move into focused modules under `src/flamingo/`. Executable scripts become thin orchestration layers and are grouped by catalogue, power-spectrum, inference, and figure workflows; duplicate fixed-cut commands are deleted instead of retained as wrappers.

**Tech Stack:** Python 3.12, NumPy, healpy, pandas, NaMaster (`pymaster`), pytest, Git SHA-256 checks.

## Global Constraints

- Activate `/scratch/scratch-lxu/venv/cmbagent_env/bin/activate` before every command.
- Theory calculations remain in `hmfast`; do not change theory implementations.
- Masses remain physical `M_sun`, never `M_sun/h`.
- Preserve bin edges, inclusive-edge behavior, log-bin construction, masks, apodization, pixel-window correction, units, filenames, and metadata meanings.
- Do not edit or regenerate tracked files under `data_paper/`, `figures/`, or `paper_results/`.
- Add no runtime dependency.

---

### Task 1: Baseline and characterization gates

**Files:**
- Create: `tests/test_bandpower_production.py`
- Create locally only: `/tmp/flamingo-ponytail-result-sha256.txt`

**Interfaces:**
- Consumes: current script formulas and tracked scientific products.
- Produces: characterization tests for `flamingo.powerspectra.bandpowers`.

- [ ] **Step 1: Record the baseline**

```bash
pytest -q
git ls-files -z data_paper figures paper_results | xargs -0 sha256sum > /tmp/flamingo-ponytail-result-sha256.txt
```

Expected: pytest exits 0 and the checksum file contains every tracked product.

- [ ] **Step 2: Write the failing bandpower test**

```python
import numpy as np
from flamingo.powerspectra.bandpowers import (
    PLANCK_ELL_EFF, PLANCK_ELL_MAX, PLANCK_ELL_MIN, bin_log_dl, bin_planck_dl,
)

def test_planck_binning_matches_existing_inclusive_formula():
    ell = np.arange(10_001, dtype=float)
    cl = 1.0 / np.maximum(ell, 1.0) ** 2
    dl = ell * (ell + 1.0) * cl / (2.0 * np.pi)
    expected = np.array([
        np.nanmean(dl[(ell >= lo) & (ell <= hi)])
        for lo, hi in zip(PLANCK_ELL_MIN, PLANCK_ELL_MAX, strict=True)
    ])
    assert np.array_equal(bin_planck_dl(ell, cl), expected)
    assert np.array_equal(PLANCK_ELL_EFF, (PLANCK_ELL_MIN + PLANCK_ELL_MAX - 1) / 2)

def test_log_binning_matches_existing_left_inclusive_formula():
    ell = np.arange(10_001, dtype=float)
    cl = 1.0 / np.maximum(ell, 1.0) ** 2
    centres, actual = bin_log_dl(ell, cl)
    edges = 10_000 * np.exp(-12 * 0.4) * np.exp(0.4 * np.arange(13))
    expected_cl = np.array([
        np.nanmean(cl[(ell >= lo) & ((ell <= hi) if i == 11 else (ell < hi))])
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True))
    ])
    expected_centres = np.sqrt(edges[:-1] * edges[1:])
    assert np.array_equal(centres, expected_centres)
    assert np.array_equal(actual, expected_centres * (expected_centres + 1) * expected_cl / (2 * np.pi))
```

- [ ] **Step 3: Verify RED**

Run: `pytest tests/test_bandpower_production.py -q`

Expected: collection fails because `flamingo.powerspectra.bandpowers` does not exist.

---

### Task 2: Canonical production kernels

**Files:**
- Create: `src/flamingo/powerspectra/bandpowers.py`
- Modify: `src/flamingo/powerspectra/namaster.py`
- Modify: `tests/test_bandpower_production.py`

**Interfaces:**
- Produces: `bin_planck_dl(ell, cl)`, `bin_log_dl(ell, cl, *, lmax=10000, dln_ell=0.4, n_bins=12)`, `write_bandpowers(path, ell, dl, header)`, and `decoupled_cl_per_ell(m, mask, pixwin2, *, lmax)`.

- [ ] **Step 1: Implement the minimum bandpower module**

```python
PLANCK_ELL_MIN = np.array([9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835])
PLANCK_ELL_MAX = np.array([12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835, 1085])
PLANCK_ELL_EFF = (PLANCK_ELL_MIN + PLANCK_ELL_MAX - 1) / 2

def bin_planck_dl(ell, cl):
    dl = ell * (ell + 1.0) * cl / (2.0 * np.pi)
    return np.array([np.nanmean(dl[(ell >= lo) & (ell <= hi)])
                     for lo, hi in zip(PLANCK_ELL_MIN, PLANCK_ELL_MAX, strict=True)])
```

Implement the tested log-bin formula and preserve the existing `np.savetxt` format exactly.

- [ ] **Step 2: Verify GREEN**

Run: `pytest tests/test_bandpower_production.py -q`

Expected: both bandpower tests pass.

- [ ] **Step 3: Add a failing per-ell NaMaster contract test**

Use small fake `NmtField`, `NmtBin`, and `NmtWorkspace` objects; assert mask-weighted monopole subtraction, placement at effective multipoles, and division by `pixwin2`.

Run: `pytest tests/test_bandpower_production.py -q`

Expected: FAIL because `decoupled_cl_per_ell` is absent.

- [ ] **Step 4: Implement the exact shared estimator**

```python
def decoupled_cl_per_ell(m, mask, pixwin2, *, lmax):
    monopole = float(np.sum(mask * m) / np.sum(mask))
    field = nmt.NmtField(mask, [m - monopole], lmax=lmax)
    bands = nmt.NmtBin.from_lmax_linear(lmax, nlb=1)
    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(field, field, bands)
    cl = workspace.decouple_cell(nmt.compute_coupled_cell(field, field))[0]
    result = np.full(lmax + 1, np.nan)
    result[bands.get_effective_ells().astype(int)] = cl
    return result / pixwin2
```

- [ ] **Step 5: Verify and commit**

```bash
pytest tests/test_bandpower_production.py tests/test_powerspectra.py -q
git add src/flamingo/powerspectra tests/test_bandpower_production.py
git commit -m "refactor: centralize bandpower production kernels"
```

---

### Task 3: Refactor production consumers

**Files:**
- Modify: `scripts/compute_l1_m9_feedback_bandpowers.py`
- Modify: `scripts/compute_l1_m9_masked_ps_alpha_fixed_1p12.py`
- Modify: `scripts/compute_l2p8_m9_masked_ps_alpha_fixed_1p12.py`
- Modify: `scripts/masking_radius_null_test.py`
- Modify: focused tests loading those scripts.

**Interfaces:**
- Consumes: Task 2 functions and `flamingo.masking.disc_mask(..., inclusive=False)`.
- Produces: the same arrays and files with no local estimator/binning copies.

- [ ] **Step 1: Add failing source-structure assertions**

```python
for path in PRODUCTION_SCRIPTS:
    source = path.read_text()
    assert "def binary_disc_mask(" not in source
    assert "def decoupled_cl_per_ell(" not in source
    assert "def bin_dl_18(" not in source
    assert "def bin_cl_log(" not in source
```

Run the focused test and confirm it fails on the current duplicate definitions.

- [ ] **Step 2: Replace local functions with direct shared calls**

Import Task 2 functions and constants. Replace local binary mask calls with `disc_mask(..., inclusive=False)`. Adjust the feedback caller to unpack `(_, dl_log)`; do not add delegation wrappers.

- [ ] **Step 3: Verify and commit**

```bash
pytest tests/test_qfrommap_masked_ps.py tests/test_qfrommap_threshold_completion.py tests/test_feedback_ratio_vs_q.py tests/test_powerspectra.py -q
git add src scripts tests
git commit -m "refactor: reuse shared masked-spectrum kernels"
```

---

### Task 4: Consolidate compute commands

**Files:**
- Replace: `scripts/compute_l1_m9_feedback_bandpowers.py` with the multi-q implementation.
- Delete: `scripts/compute_l1_m9_feedback_ratio_vs_q_bandpowers.py`
- Delete: `scripts/compute_l1_m9_feedback_bandpowers_qgt1.py`
- Modify: `scripts/compute_l1_m9_masked_ps_alpha_fixed_1p12.py`
- Modify: `scripts/compute_l2p8_m9_masked_ps_alpha_fixed_1p12.py`
- Delete: `scripts/compute_masked_ps_qgt6.py`
- Modify: related tests and checked-in command references.

**Interfaces:**
- Feedback command accepts `--q-cuts`, `--selection`, `--variants`, `--workers`, `--force`, and `--dry-run`.
- L1/L2 commands accept `--q-cuts`, retaining current thresholds as defaults.

- [ ] **Step 1: Update tests to the canonical paths and verify RED**

Route q>1 and q>6 cases through canonical commands. Expected failures mention old paths or missing `--q-cuts` support.

- [ ] **Step 2: Make multi-q feedback production canonical**

Move the multi-q implementation onto `compute_l1_m9_feedback_bandpowers.py`, replace dynamic script loading with package/config imports, and delete both superseded files. Preserve output naming exactly.

- [ ] **Step 3: Add arbitrary cuts to L1/L2 and delete q>6**

Use `nargs="+"`, `type=float`, and current cut defaults. Derive tags using the existing integer/decimal convention.

- [ ] **Step 4: Verify and commit**

```bash
pytest tests/test_feedback_ratio_vs_q.py tests/test_qfrommap_masked_ps.py tests/test_qfrommap_threshold_completion.py -q
! rg 'importlib\.util' scripts/compute_l1_m9_feedback_bandpowers.py scripts/compute_l1_m9_masked_ps_alpha_fixed_1p12.py scripts/compute_l2p8_m9_masked_ps_alpha_fixed_1p12.py
git add scripts tests README.md docs
git commit -m "refactor: consolidate masked-spectrum commands"
```

---

### Task 5: Consolidate plot and postprocess commands

**Files:**
- Modify: `scripts/plot_l1_m9_feedback_bandpowers.py`
- Delete: `scripts/plot_l1_m9_feedback_bandpowers_qgt1.py`
- Create: `scripts/postprocess_asz_alpha_fixed.py`
- Delete: `scripts/postprocess_l1_m9_asz_alpha_fixed.py`
- Delete: `scripts/postprocess_l2p8_m9_asz_alpha_fixed.py`
- Modify: relevant tests.

**Interfaces:**
- Plot command supports `--single-cut qgt1` while retaining the multi-panel default.
- Postprocess command requires `case` in `{L1_m9,L2p8_m9}` and preserves existing paths, JSON keys, text lines, labels, and figure stems.

- [ ] **Step 1: Add routing tests and verify RED**

Assert the canonical plot resolves qgt1 products and the postprocessor returns the exact existing L1/L2 case configuration.

- [ ] **Step 2: Fold qgt1 into the canonical plotter**

Reuse existing load/save helpers and preserve the old single-cut output names. Delete the old plot file.

- [ ] **Step 3: Implement one fixed-alpha postprocessor**

Use a two-entry dictionary for chain path, label, summary extras, and figure stem differences; retain one shared statistical body.

- [ ] **Step 4: Verify and commit**

```bash
pytest tests/test_l1_qfrommap_plots.py -q
git add scripts tests
git commit -m "refactor: consolidate plotting and postprocessing"
```

---

### Task 6: Organize entry points and verify the repository

**Files:**
- Move remaining commands into `scripts/catalogue/`, `scripts/powerspectra/`, `scripts/inference/`, and `scripts/figures/`.
- Modify: tests, README, docs, and shell launchers referencing old paths.
- Delete: completed `docs/superpowers/plans/` and `docs/superpowers/specs/` after execution; Git retains their history.

**Interfaces:**
- Each command has one domain-specific location.
- No script dynamically imports another script.

- [ ] **Step 1: Move commands and update every repository reference**

Use Git moves, preserve filenames, and do not leave compatibility wrappers.

- [ ] **Step 2: Run structure checks**

```bash
! rg 'importlib\.util' scripts --glob '*.py'
python -m compileall -q src scripts paper_results
```

- [ ] **Step 3: Run complete verification**

```bash
pytest -q
git diff --check
git ls-files -z data_paper figures paper_results | xargs -0 sha256sum > /tmp/flamingo-ponytail-result-sha256-after.txt
diff -u /tmp/flamingo-ponytail-result-sha256.txt /tmp/flamingo-ponytail-result-sha256-after.txt
git diff origin/ponytail_review..HEAD -- pyproject.toml
```

Expected: tests pass, checksums match, formatting is clean, and dependencies are unchanged.

- [ ] **Step 4: Review and commit**

```bash
git status --short
git diff --stat origin/ponytail_review..HEAD
git add src scripts tests README.md docs
git commit -m "refactor: organize scientific workflows"
```

Expected: no tracked scientific product appears in the diff.
