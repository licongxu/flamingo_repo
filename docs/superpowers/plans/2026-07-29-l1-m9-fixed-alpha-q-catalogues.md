# L1_m9 Fixed-Alpha `q` Catalogues Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and verify nine `_qfrommz_alpha_fixed_1p12.csv` catalogues from the fixed-alpha custom-GNFW best fit.

**Architecture:** Reuse the tested `rewrite_catalogue` streaming boundary without changing production code. Construct one `SZScaling` with the authoritative fixed-alpha parameters, process each original catalogue atomically on GPU 1, then independently stream every source/output pair for full validation.

**Tech Stack:** Python 3.10+, pandas, NumPy, JAX, hmfast, pytest.

## Global Constraints

- Activate `/scratch/scratch-lxu/venv/cmbagent_env/bin/activate` before every command.
- Bind JAX/hmfast work to GPU 1 with `CUDA_VISIBLE_DEVICES=1`.
- Use physical catalogue masses in `M_sun`, not `M_sun/h`.
- Read `A_SZ = -4.0953238` from `chains/l1_m9_customgnfw_asz_alpha_fixed_1p12/best_fit.json`.
- Fix `alpha_SZ = 1.12` and `B = 1.41`.
- Retain D3A cosmology, `sigma_lnY = 0.173`, seed `20260630`, and SZiFi `immf6`.
- Write outputs ending exactly in `_qfrommz_alpha_fixed_1p12.csv`.
- Do not modify existing `_qfrommz.csv` or `_qfrommz_bestfit.csv` files.

---

### Task 1: Preflight the calculation

**Files:**
- Read: `chains/l1_m9_customgnfw_asz_alpha_fixed_1p12/best_fit.json`
- Read: `scripts/create_l1_m9_bestfit_q_catalogues.py`
- Test: `tests/test_bestfit_q_catalogues.py`

**Interfaces:**
- Consumes: chain best fit and the existing `rewrite_catalogue` implementation.
- Produces: verified parameters, source fingerprints, output targets, GPU selection, and storage check.

- [ ] **Step 1: Run the existing generator tests**

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
CUDA_VISIBLE_DEVICES=1 pytest -q tests/test_bestfit_q_catalogues.py tests/test_cnc.py
```

Expected: all tests pass.

- [ ] **Step 2: Verify the authoritative parameters**

Load `best_fit.json` and require:

```python
assert result["best_fit"]["A_SZ"] == -4.0953238
assert result["fixed"]["alpha_SZ"] == 1.12
assert result["fixed"]["B"] == 1.41
```

- [ ] **Step 3: Verify all sources and destinations**

For every name in `paper_results.config.VARIANTS`, require the original
`*_qfrommz.csv` source to exist and the corresponding
`*_qfrommz_alpha_fixed_1p12.csv` destination to be absent. Record every source
size and nanosecond modification time and require sufficient free storage for
all outputs plus one temporary catalogue and 1 GiB.

---

### Task 2: Generate the nine catalogues

**Files:**
- Read: nine `/rds/rds-lxu/flamingo/L1_m9/catalogues/*_qfrommz.csv` files
- Create: nine `/rds/rds-lxu/flamingo/L1_m9/catalogues/*_qfrommz_alpha_fixed_1p12.csv` files

**Interfaces:**
- Consumes: `rewrite_catalogue(source, output, scaling)` and the preflight results.
- Produces: nine complete fixed-alpha catalogues.

- [ ] **Step 1: Construct the scaling relation**

```python
base = SZScaling.calibrated(
    B=1.41,
    alpha_SZ=1.12,
    sigma_y0_file=config.SIGMA_Y0_FILE,
    skyfracs_file=config.SKYFRACS_FILE,
)
scaling = replace(base, A_SZ=-4.0953238)
```

- [ ] **Step 2: Generate each output atomically**

For each source, set:

```python
output = source.with_name(source.stem + "_alpha_fixed_1p12.csv")
```

Call `rewrite_catalogue` without `force`. Print the variant, row count, output
path, `q_min`, `q_max`, size, and elapsed time after each successful file.

---

### Task 3: Independently validate and audit completion

**Files:**
- Read: all nine source/output pairs
- Read: `scripts/create_l1_m9_bestfit_q_catalogues.py`

**Interfaces:**
- Consumes: generated catalogues and recorded source fingerprints.
- Produces: row-level validation evidence and completion report.

- [ ] **Step 1: Validate every row**

Stream each source/output pair with `float_precision="round_trip"` and require:

```python
assert source_columns == output_columns
assert source_rows == output_rows
assert np.array_equal(source["soap_index"], output["soap_index"])
assert all_non_q_columns_are_equal
assert np.all(np.isfinite(output["q_from_mz"]))
assert np.all(output["q_from_mz"] > 0)
```

- [ ] **Step 2: Validate direct hmfast samples**

For the first, middle, and final row of each catalogue, recompute `q` with the
fixed-alpha `SZScaling` and require agreement with the stored value at
`rtol=5e-15`, `atol=0`.

- [ ] **Step 3: Audit provenance and preservation**

Require all nine provenance headers to contain `A_SZ=-4.0953238`,
`alpha_SZ=1.12`, `B=1.41`, `sigma_lnY=0.173`, `seed=20260630`,
`cosmology=D3A`, and `noise=SZiFi-immf6`. Confirm every source fingerprint is
unchanged, all nine outputs exist, and no `*.tmp-*` file remains.

- [ ] **Step 4: Run the full repository suite**

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
CUDA_VISIBLE_DEVICES=1 pytest -q
```

Expected: the full suite passes with no failures.

- [ ] **Step 5: Report completion**

Report the authoritative parameters, all nine output names with row counts,
sizes, and `q` ranges, source preservation, direct-sample validation, and test
result. Do not add the external CSV files to Git.
