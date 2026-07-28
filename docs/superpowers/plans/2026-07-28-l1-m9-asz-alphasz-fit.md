# L1_m9 Custom-GNFW A_SZ and Alpha_SZ Fit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a converged GPU Cobaya MCMC fit of `A_SZ` and `alpha_SZ` to all 18 fiducial L1_m9 bandpowers using fixed D3A cosmology, custom GNFW theory with 1h+2h power, and the newly computed full covariance.

**Architecture:** A focused `flamingo.inference` module provides a testable Gaussian likelihood and a JIT-compiled hmfast Cobaya theory component. One YAML config writes the chain to the requested `chains/` folder, while one post-processing script extracts the best retained sample, posterior intervals, convergence diagnostics, and the two-parameter contour.

**Tech Stack:** Python 3.12, NumPy, JAX CUDA float64, hmfast, Cobaya 3.5.7, GetDist 1.7.3, matplotlib, pytest.

## Global Constraints

- Activate `/scratch/scratch-lxu/venv/cmbagent_env/bin/activate` before every command.
- Use all 18 rows of `data_paper/binned_bandpowers/Dl_yy_L1_m9_fullsky_binned_18.txt`.
- Multiply the observed second column by `1e-12` before comparison to theory.
- Use `data_paper/covariance/cov_full_L1_m9_fullsky_Dl_yy_binned_18.npy`.
- Fix the exact D3A cosmology from `flamingo.catalogue.frame.D3A_COSMOLOGY`.
- Use hmfast `ParametricGNFWPressureProfile` with shape `(8.13, 1.156, 1.062, 5.4807, 0.3292)`, `B=1`, no scatter boost, physical `M_sun`, and `hm_consistency=False`.
- Include both one-halo and two-halo power.
- Sample only `A_SZ` and `alpha_SZ` with the approved uniform priors.
- Run theory and MCMC on CUDA.
- Store all chain products under `chains/l1_m9_customgnfw_asz_alphasz/`.

---

### Task 1: Test and implement the Gaussian 18-bin likelihood

**Files:**
- Create: `src/flamingo/inference/__init__.py`
- Create: `src/flamingo/inference/l1_m9.py`
- Create: `tests/test_l1_m9_inference.py`

**Interfaces:**
- Produces: `load_bandpower_likelihood(data_file, covariance_file, data_scale=1e-12)`.
- Produces: `gaussian_loglike(observed, theory, inverse_covariance)`.
- Produces: Cobaya class `L1M9BandPowerLikelihood`.

- [ ] **Step 1: Add failing tests for scaling and the quadratic form**

Create `tests/test_l1_m9_inference.py` with temporary two-bin files. Verify
that values `[2, 5]` are loaded as `[2e-12, 5e-12]`, and verify the literal
quadratic form `-0.5 * (1^2/4 + 2^2/9)`.

```python
import numpy as np

from flamingo.inference.l1_m9 import (
    gaussian_loglike,
    load_bandpower_likelihood,
)


def test_load_bandpower_likelihood_applies_data_scale(tmp_path):
    data = tmp_path / "data.txt"
    covariance = tmp_path / "cov.npy"
    np.savetxt(data, [[10.0, 2.0], [20.0, 5.0]])
    np.save(covariance, np.diag([4e-24, 9e-24]))

    ell, observed, cov, inverse = load_bandpower_likelihood(
        data, covariance, data_scale=1e-12
    )

    np.testing.assert_array_equal(ell, [10.0, 20.0])
    np.testing.assert_array_equal(observed, [2e-12, 5e-12])
    np.testing.assert_array_equal(cov, np.diag([4e-24, 9e-24]))
    np.testing.assert_allclose(inverse @ cov, np.eye(2))


def test_gaussian_loglike_matches_literal_quadratic_form():
    observed = np.array([1.0, 2.0])
    theory = np.zeros(2)
    inverse = np.diag([1.0 / 4.0, 1.0 / 9.0])
    expected = -0.5 * (1.0 / 4.0 + 4.0 / 9.0)
    assert gaussian_loglike(observed, theory, inverse) == expected
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run `pytest -q tests/test_l1_m9_inference.py`; expect collection to fail
because `flamingo.inference.l1_m9` does not exist.

- [ ] **Step 3: Implement strict input loading, Gaussian loglike, and the Cobaya likelihood**

Implement `load_bandpower_likelihood` with shape, finiteness, symmetry, and
positive-definiteness checks. Implement `gaussian_loglike` as
`-0.5 * residual @ inverse @ residual`. Implement
`L1M9BandPowerLikelihood.initialize`, `get_requirements`, and `logp`; require
`Cl_sz`, sum its `1h` and `2h` arrays, and reject any non-finite or non-18-bin
theory.

- [ ] **Step 4: Run the focused tests and commit**

Run `pytest -q tests/test_l1_m9_inference.py`; expect 2 passed. Commit:

```bash
git add src/flamingo/inference tests/test_l1_m9_inference.py
git commit -m "feat: add L1_m9 Gaussian bandpower likelihood"
```

---

### Task 2: Test and implement the fixed-cosmology custom-GNFW theory

**Files:**
- Modify: `src/flamingo/inference/l1_m9.py`
- Modify: `tests/test_l1_m9_inference.py`

**Interfaces:**
- Produces: Cobaya class `L1M9CustomGNFWTheory`.
- Produces: `evaluate_bandpowers(A_SZ, alpha_SZ) -> {"1h", "2h"}`.

- [ ] **Step 1: Add a failing GPU smoke test**

Add a test that constructs `L1M9CustomGNFWTheory`, calls `initialize()`, then
calls `evaluate_bandpowers(-4.1, 1.12)`. Assert exact keys `{"1h", "2h"}`,
shape `(18,)`, finite positive values for both terms, and positive total.

- [ ] **Step 2: Run the smoke test and verify the missing-class failure**

Run the selected test with `CUDA_VISIBLE_DEVICES=1`; expect import failure
because `L1M9CustomGNFWTheory` is not defined.

- [ ] **Step 3: Implement the custom-GNFW theory**

Use the exact 18 bin arrays from the covariance calculation. In
`initialize()`, construct:

```python
HaloModel(
    cosmology=D3A_COSMOLOGY,
    mass_definition=MassDefinition(500, "critical"),
    convert_masses=True,
    hm_consistency=False,
)
```

Seed `ParametricGNFWPressureProfile` with `A_SZ=-4.1`,
`alpha_SZ=1.12`, `P0=8.13`, `c500=1.156`, `alpha=1.062`,
`beta=5.4807`, `gamma=0.3292`, and `B=1.0`. JIT a two-argument method that
updates only `A_SZ` and `alpha_SZ`, then evaluates `cl_1h` and `cl_2h`.
Convert and bin both terms independently. Cobaya `calculate()` stores them in
`state["Cl_sz"]`.

- [ ] **Step 4: Run focused and regression tests and commit**

Run:

```bash
CUDA_VISIBLE_DEVICES=1 pytest -q \
  tests/test_l1_m9_inference.py tests/test_l1_m9_covariance.py tests/test_theory.py
```

Commit:

```bash
git add src/flamingo/inference/l1_m9.py tests/test_l1_m9_inference.py
git commit -m "feat: add GPU custom-GNFW Cobaya theory"
```

---

### Task 3: Add and validate the Cobaya configuration

**Files:**
- Create: `cobaya/configs/l1_m9_customgnfw_asz_alphasz.yaml`

**Interfaces:**
- Consumes: the two Cobaya classes from Tasks 1 and 2.
- Produces: chain root `chains/l1_m9_customgnfw_asz_alphasz/chain`.

- [ ] **Step 1: Create the YAML**

Configure the exact absolute data and covariance paths. Set:

```yaml
params:
  A_SZ:
    prior: {min: -5.5, max: -3.0}
    ref: {dist: norm, loc: -4.1, scale: 0.08}
    proposal: 0.05
  alpha_SZ:
    prior: {min: 0.7, max: 1.5}
    ref: {dist: norm, loc: 1.12, scale: 0.05}
    proposal: 0.03
sampler:
  mcmc:
    Rminus1_stop: 0.01
    Rminus1_cl_stop: 0.05
    proposal_scale: 1.5
    learn_every: 20
    learn_proposal: true
    burn_in: 100
    max_tries: 100000
```

Set the absolute output root to the requested chain directory.

- [ ] **Step 2: Validate the config without sampling**

Use `cobaya.model.get_model` on the YAML and evaluate one log-posterior point
at `A_SZ=-4.1`, `alpha_SZ=1.12` on GPU. Require finite log-posterior and
finite component likelihood.

- [ ] **Step 3: Commit the config**

```bash
git add cobaya/configs/l1_m9_customgnfw_asz_alphasz.yaml
git commit -m "feat: configure L1_m9 custom-GNFW chain"
```

---

### Task 4: Run the GPU MCMC to convergence

**Files:**
- Create at runtime: `chains/l1_m9_customgnfw_asz_alphasz/chain.*`
- Create at runtime: `chains/l1_m9_customgnfw_asz_alphasz/run.log`

- [ ] **Step 1: Confirm an idle GPU**

Use `nvidia-smi`; select the idle device explicitly.

- [ ] **Step 2: Launch Cobaya**

Run:

```bash
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda \
  python -m cobaya run --force \
  cobaya/configs/l1_m9_customgnfw_asz_alphasz.yaml \
  > chains/l1_m9_customgnfw_asz_alphasz/run.log 2>&1
```

Poll the process without waits longer than 60 seconds and report convergence
progress. Require exit code 0.

- [ ] **Step 3: Verify convergence and chain integrity**

Read the final progress and checkpoint files. Require final
`Rminus1 <= 0.01`, finite samples, both parameters inside their prior bounds,
and nonzero sample count after discarding 30%.

---

### Task 5: Post-process, reproduce the best fit, and plot contours

**Files:**
- Create: `scripts/postprocess_l1_m9_asz_alphasz.py`
- Create at runtime: `chains/l1_m9_customgnfw_asz_alphasz/best_fit.json`
- Create at runtime: `chains/l1_m9_customgnfw_asz_alphasz/best_fit.txt`
- Create at runtime: `chains/l1_m9_customgnfw_asz_alphasz/posterior_summary.txt`
- Create at runtime: `chains/l1_m9_customgnfw_asz_alphasz/contour_A_SZ_alpha_SZ.png`
- Create at runtime: `chains/l1_m9_customgnfw_asz_alphasz/contour_A_SZ_alpha_SZ.pdf`

- [ ] **Step 1: Implement post-processing**

Load the chain with GetDist using `ignore_rows=0.3`. Identify Cobaya's
`chi2__...` derived column and take its minimum as the best retained sample.
Compute weighted means and central 68% limits for both parameters. Write JSON
and text summaries. Draw filled 68% and 95% contours, mark the best-fit point,
and save PNG and PDF.

- [ ] **Step 2: Run post-processing**

Run:

```bash
MPLBACKEND=Agg python scripts/postprocess_l1_m9_asz_alphasz.py
```

Require all five files to exist and be non-empty.

- [ ] **Step 3: Re-evaluate the best-fit chi-square**

Instantiate the theory and load the likelihood inputs. Evaluate the 1h+2h
theory at the JSON best-fit coordinates, compute `chi2` directly, and require
agreement with the recorded chain `chi2` within `1e-6`.

- [ ] **Step 4: Commit post-processing and run the full suite**

```bash
git add scripts/postprocess_l1_m9_asz_alphasz.py
git commit -m "feat: summarize L1_m9 custom-GNFW fit"
CUDA_VISIBLE_DEVICES=1 pytest -q
git diff --check
```

Report the numerical best fit, posterior intervals, final `Rminus1`, and
clickable paths to the chain directory and contour.
