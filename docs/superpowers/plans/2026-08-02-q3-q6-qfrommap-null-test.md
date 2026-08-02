# q=3/q=6 q-from-map and L1 Null-Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the missing q > 3/q > 6 aperture-selection bandpowers and rerun the L1 fiducial masking-radius null test with corrected q-from-map inputs.

**Architecture:** Reuse the existing selection-aware L1 production for both thresholds. Add the same explicit selection boundary to the standalone L2 q > 6 and null-test entry points, preserving legacy defaults and separating null caches by selection tag.

**Tech Stack:** Python, healpy, NaMaster/pymaster, NumPy, pandas, pytest, tmux.

## Global Constraints

- Use physical `M_sun` catalogue products and corrected rotated coordinates.
- Preserve every q-from-mass/redshift file and cache.
- Tag every new bandpower with `qfrommap`.
- Keep aggregate memory near or below half of the 376 GiB host.

---

### Task 1: Selection-aware paths

**Files:**
- Modify: `scripts/compute_masked_ps_qgt6.py`
- Modify: `scripts/masking_radius_null_test.py`
- Test: `tests/test_qfrommap_threshold_completion.py`

**Interfaces:**
- Consumes: `QSelection` from `resolve_q_selection()`.
- Produces: q-from-map catalogue paths, q column names, bandpower tags, and isolated null-cache paths.

- [ ] Write tests asserting L2 lightcone0 q > 6 uses the canonical q-from-map catalogue and that null caches resolve below `masking_radius_null_test/qfrommap`.
- [ ] Run `pytest -q tests/test_qfrommap_threshold_completion.py` and observe failure because the selection interfaces do not exist.
- [ ] Add `--selection` to both scripts, parameterize q columns and catalogue paths, and retain legacy defaults.
- [ ] Run the focused test again and require all tests to pass.

### Task 2: Production launch

**Files:**
- Create: `logs/qfrommap_q3_q6_20260802.log`
- Create: `logs/qfrommap_l1_null_test_20260802.log`
- Create: q-from-map bandpowers and null-test caches under `data_paper/`.

**Interfaces:**
- Consumes: corrected catalogues and existing full-sky y maps.
- Produces: two bandpower binnings per map/cut and one cache per null-test radius.

- [ ] Dry-run q > 3/q > 6 paths and verify the rotation sanity inputs.
- [ ] Launch L1 q > 3/q > 6 with eight workers and bounded OpenMP threads.
- [ ] Launch the L1 null-test q cuts serially in a persistent tmux session.
- [ ] Launch L2 lightcone0 q > 6 after the L1 threshold process frees capacity.
- [ ] Monitor output counts, errors, RSS, and tmux health.

### Task 3: Plot and validation

**Files:**
- Modify affected selection-aware plotting scripts only if required.
- Create q-from-map q > 3/q > 6 and null-test figures.

**Interfaces:**
- Consumes: completed Task 2 products.
- Produces: updated PNG/PDF figures without overwriting legacy figures.

- [ ] Verify every expected bandpower and cache is present and non-empty.
- [ ] Regenerate affected L1, Planck-comparison, and null-test figures.
- [ ] Run the full test suite and validate PNG/PDF integrity.
