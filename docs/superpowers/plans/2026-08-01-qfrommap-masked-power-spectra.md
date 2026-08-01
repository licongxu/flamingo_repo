# Q-from-map Masked tSZ Power Spectra Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce collision-free q-from-map NaMaster masked tSZ bandpowers for all nine L1 maps and eight L2p8 maps, then plot the established L1-versus-L2 comparison.

**Architecture:** Parameterize the existing L1, L2, and comparison-plot scripts so their current q-from-mass defaults remain unchanged. A small selection configuration layer supplies the q column, catalogue path, product tag, and provenance; all mask construction, NaMaster estimation, binning, and plotting code remains shared with the established pipeline.

**Tech Stack:** Python, pandas, NumPy, healpy, NaMaster/pymaster, matplotlib, pytest.

## Global Constraints

- Preserve every existing bandpower, metadata file, NPZ, PNG, and PDF.
- The only scientific change is `q_from_aperture > q_cut` from q-from-map catalogues.
- Use q cuts `50, 20, 10, 5, 1`.
- Keep `R=max(4*theta_500, 20 arcmin)`, C2 apodization 0.25 degrees, masked-monopole subtraction, NaMaster `nlb=1`, `lmax=10000`, and pixel-window deconvolution unchanged.
- Write `_qfrommap_` in every new masked product name.
- L1 reads corrected staging catalogues; L2 reads audited canonical catalogues.
- Hide GPUs and disable JAX preallocation.
- Pilot runs serially with one map process and at most 8 OpenMP threads.
- Production may use two map workers (16 OpenMP threads total) only if the
  pilot peaks below 30 GiB RSS and the host remains lightly loaded; otherwise
  it remains serial. Never exceed two map workers.

---

### Task 1: Selection configuration and catalogue loading

**Files:**
- Create: `src/flamingo/powerspectra/q_selection.py`
- Modify: `scripts/compute_l1_m9_feedback_bandpowers.py`
- Test: `tests/test_qfrommap_masked_ps.py`

**Interfaces:**
- Produces: `QSelection(tag, q_column, l1_catalogue_dir, l2_root)` and `resolve_q_selection(name, *, l1_catalogue_dir=None, l2_root=None)`.
- Produces: `load_catalogue(path, q_column="q_from_mz")` with unchanged default behavior.

- [ ] **Step 1: Write failing tests** asserting that `qfrommap` resolves to `q_from_aperture`, the corrected L1 staging directory, canonical L2 root, and tag `qfrommap`; assert that the legacy mode remains `q_from_mz` and `qfrommz_alpha_fixed_1p12`.
- [ ] **Step 2: Run** `pytest -q tests/test_qfrommap_masked_ps.py` and verify failure because the configuration API does not exist.
- [ ] **Step 3: Implement** the immutable configuration and make `load_catalogue` accept a q-column argument without changing its default.
- [ ] **Step 4: Run** `pytest -q tests/test_qfrommap_masked_ps.py tests/test_aperture_snr.py` and require all pass.
- [ ] **Step 5: Commit** `src/flamingo/powerspectra/q_selection.py`, the loader change, and tests.

### Task 2: Parameterize L1 and L2 compute drivers

**Files:**
- Modify: `scripts/compute_l1_m9_feedback_ratio_vs_q_bandpowers.py`
- Modify: `scripts/compute_l2p8_m9_masked_ps_alpha_fixed_1p12.py`
- Test: `tests/test_qfrommap_masked_ps.py`

**Interfaces:**
- L1 CLI adds `--selection`, `--l1-catalogue-dir`, and preserves `--workers 1`.
- L2 CLI adds `--selection`, `--l2-root`, `--workers`, and retains repeatable `--lightcone`.
- Both drivers write distinct tag-qualified metadata and never reuse/link a legacy q-from-mass file in q-from-map mode.

- [ ] **Step 1: Add failing path/name tests** for fiducial and feedback L1 outputs plus L2 lightcone outputs in both selection modes.
- [ ] **Step 2: Run the focused tests** and verify the expected legacy hard-coding failures.
- [ ] **Step 3: Implement** selection-aware paths, q columns, headers, metadata, and skip logic while leaving estimator functions untouched.
- [ ] **Step 4: Run** `pytest -q tests/test_qfrommap_masked_ps.py tests/test_qfrommap_catalogues.py` and require all pass.
- [ ] **Step 5: Run** both CLIs with `--help` and a no-compute inventory/dry-run path, verifying the 9 L1 and 8 L2 jobs resolve to the intended catalogues and `_qfrommap_` outputs.
- [ ] **Step 6: Commit** the two compute-driver changes and tests.

### Task 3: Parameterize the established comparison plot

**Files:**
- Modify: `scripts/plot_l1_l2p8_m9_masked_ps_comparison.py`
- Test: `tests/test_qfrommap_masked_ps.py`

**Interfaces:**
- CLI adds `--selection {qfrommz_alpha_fixed_1p12,qfrommap}`.
- Produces `figures/masked_ps/l1_l2p8_m9_masked_ps_{binned_18,logbins}_qfrommap.{png,pdf}`.

- [ ] **Step 1: Add failing tests** for tag-specific bandpower paths, metadata paths, titles, and figure stems.
- [ ] **Step 2: Run the focused test** and verify it fails against the hard-coded legacy tag.
- [ ] **Step 3: Implement** the CLI/tag parameterization without changing axes, line styles, cuts, full-sky references, or ratio calculation.
- [ ] **Step 4: Run the focused and plotting tests**, then commit.

### Task 4: Resource-bounded pilot

**Files:**
- Create outputs only under `data_paper/binned_bandpowers/` and `figures/masked_ps/`.

**Interfaces:**
- Consumes corrected L1 fiducial and audited L2 lightcone0 q-from-map catalogues.
- Produces ten masked spectra per map (five cuts times two binnings), metadata, and two pilot comparison figures.

- [ ] **Step 1: Record idle baseline** with `ps`, `free -h`, and `nproc`; confirm no stale NaMaster jobs.
- [ ] **Step 2: Run rotation preflight on all 17 map/catalogue pairs** and require the yang26-rotated peak ratio to exceed the natural-coordinate ratio for every map.
- [ ] **Step 3: Run L1 fiducial serially** with `CUDA_VISIBLE_DEVICES=`, `JAX_PLATFORMS=cpu`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `OMP_NUM_THREADS=8`, and `--workers 1`.
- [ ] **Step 4: Check** all five masked counts against direct catalogue threshold counts, finite bandpowers, `0 < f_sky_eff <= f_sky_raw <= 1`, peak RSS, and runtime.
- [ ] **Step 5: Run L2 lightcone0 serially** under the same resource limits and repeat validation.
- [ ] **Step 6: Generate and visually inspect** the 18-bin and log-bin q-from-map pilot comparison figures.

### Task 5: All-map production and final verification

**Files:**
- Create remaining q-from-map bandpowers and metadata under `data_paper/binned_bandpowers/`.
- Create final figures under `figures/masked_ps/`.

**Interfaces:**
- Produces the complete matrix for 9 L1 maps, 8 L2 maps, 5 cuts, and 2 binnings.

- [ ] **Step 1: Choose one or two production workers** from pilot peak RSS and current host load; cap each worker at eight OpenMP threads and never exceed two workers.
- [ ] **Step 2: Start a resumable detached run** for the remaining eight L1 variants under that cap.
- [ ] **Step 3: After L1 succeeds, run the remaining seven L2 lightcones** with the same cap.
- [ ] **Step 4: Verify the expected 170 masked text files** exist (`17 maps * 5 cuts * 2 binnings`), are nonempty, finite, and have the expected 18 or 12 rows; verify metadata covers all 17 maps.
- [ ] **Step 5: Confirm** no legacy q-from-mass file checksum or modification time changed during the run.
- [ ] **Step 6: Regenerate** the two final comparison figures, inspect them, and run the full focused pytest suite.
- [ ] **Step 7: Report** counts, sky fractions, runtimes, peak RSS, output links, and any failed/skipped map without claiming completion unless every verification passes.
