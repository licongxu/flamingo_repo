# L1_m9 qfrommap Cobaya Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce converged GPU/hmfast Cobaya inference chains for the fiducial L1_m9 full-sky and `q > 50, 20, 10, 5` `_qfrommap` 18-bin spectra, using exact synthetic-data binning and covariances evaluated at a converged amplitude-only full-sky best fit.

**Architecture:** Put the canonical inclusive-bin mean and covariance operations in one inference utility module, then use them in both the Cobaya theory and a focused qfrommap covariance producer. A separate driver alternates an amplitude-only full-sky MCMC and covariance recomputation until stable; the final science runner reads that converged artifact and launches five isolated chains.

**Tech Stack:** Python 3, NumPy, JAX/CUDA, hmfast, Cobaya, GetDist, pytest.

## Global Constraints

- Activate `/scratch/scratch-lxu/venv/cmbagent_env/bin/activate` before every command.
- All halo masses are physical `M_sun`, never `M_sun/h`.
- All theory calculations use hmfast and production calculations require a CUDA JAX device.
- Use the synthetic-data bins `[9,12], [12,16], ..., [835,1085]`, inclusive at both edges.
- Use custom-GNFW `P0=8.13`, `c500=1.156`, `alpha=1.062`, `beta=5.4807`, `gamma=0.3292`, `alpha_SZ=1.12`, and `B=1.41` for the amplitude fit and covariance fixed point.
- Fix `sigma_lnY=0.173` in the amplitude-only fit and include the `q_cat -> infinity` moments in full-sky power and trispectrum.
- Final masked likelihood data must end in `_qfrommap_binned_18.txt`.
- Final covariances must be evaluated at the converged full-sky best-fit parameters.
- Preserve unrelated working-tree and index changes. Commits must use `git commit --only` with explicit paths.

---

### Task 1: Canonical synthetic-data bandpower and covariance operations

**Files:**
- Create: `src/flamingo/inference/bandpowers.py`
- Modify: `src/flamingo/inference/l1_m9.py:19-46`
- Modify: `tests/test_l1_m9_covariance.py`
- Modify: `tests/test_l1_m9_inference.py`

**Interfaces:**
- Produces: `ELL_MIN`, `ELL_MAX`, `ELL_EFF`, `bin_dl_uniform(ell, dl)`, `gaussian_bandpower_covariance(ell_cl, cl, f_sky)`, `trispectrum_bandpower_covariance(ell_tri, trispectrum_cl, f_sky)`, and `validate_covariance(covariance_g, covariance_t, covariance)`.
- Consumes: NumPy arrays in physical `C_ell` or `D_ell` units; no hidden `1e12` scaling.

- [ ] **Step 1: Add failing inclusive-bin and shared-edge tests**

```python
from flamingo.inference.bandpowers import (
    ELL_MAX,
    ELL_MIN,
    bin_dl_uniform,
    gaussian_bandpower_covariance,
)


def test_bin_dl_uniform_uses_both_inclusive_edges():
    ell = np.arange(9.0, 17.0)
    dl = ell.copy()
    got = bin_dl_uniform(ell, dl, ell_min=np.array([9, 12]), ell_max=np.array([12, 16]))
    np.testing.assert_allclose(got, [np.mean([9, 10, 11, 12]), np.mean([12, 13, 14, 15, 16])])


def test_gaussian_covariance_keeps_shared_edge():
    ell = np.geomspace(9.0, 16.0, 10)
    cl = np.full_like(ell, 3e-12)
    cov = gaussian_bandpower_covariance(
        ell, cl, 0.61, ell_min=np.array([9, 12]), ell_max=np.array([12, 16])
    )
    assert cov[0, 1] > 0.0
    assert cov[1, 0] == cov[0, 1]
```

- [ ] **Step 2: Run the focused tests and verify the missing module failure**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest -q tests/test_l1_m9_covariance.py tests/test_l1_m9_inference.py
```

Expected: collection fails because `flamingo.inference.bandpowers` does not exist.

- [ ] **Step 3: Port the reference covariance operations with physical-unit weights**

Implement `bandpowers.py` from `/scratch/scratch-lxu/tsz_cnc_paper_plots/tsz_only/bandpower_theory.py`. The core Gaussian loop must be:

```python
def gaussian_bandpower_covariance(ell_cl, cl, f_sky, *, ell_min=ELL_MIN, ell_max=ELL_MAX):
    covariance = np.zeros((len(ell_min), len(ell_min)), dtype=float)
    for row, (lo_row, hi_row) in enumerate(zip(ell_min, ell_max, strict=True)):
        ell_row = np.arange(lo_row, hi_row + 1, dtype=float)
        weight_row = ell_row * (ell_row + 1.0) / (2.0 * np.pi * ell_row.size)
        for col, (lo_col, hi_col) in enumerate(zip(ell_min, ell_max, strict=True)):
            lo, hi = max(lo_row, lo_col), min(hi_row, hi_col)
            if lo > hi:
                continue
            shared = np.arange(lo, hi + 1, dtype=float)
            cl_shared = np.interp(np.log(shared), np.log(ell_cl), cl)
            weight_col = np.arange(lo_col, hi_col + 1, dtype=float)
            weight_col = weight_col * (weight_col + 1.0) / (2.0 * np.pi * weight_col.size)
            covariance[row, col] = np.sum(
                weight_row[(shared - lo_row).astype(int)]
                * weight_col[(shared - lo_col).astype(int)]
                * 2.0 * cl_shared**2 / ((2.0 * shared + 1.0) * f_sky)
            )
    return covariance
```

Build the trispectrum operator by applying the same inclusive uniform `D_ell` weights to basis vectors and return `operator @ trispectrum_cl @ operator.T / (4*pi*f_sky)`.

- [ ] **Step 4: Replace the old upper-exclusive `_bin_dl` implementation**

In `l1_m9.py`, import the shared constants and make `_bin_dl` delegate to `bin_dl_uniform`. Preserve `_bin_dl` as a compatibility wrapper because `masked_ps.py` imports it.

```python
def _bin_dl(ell: np.ndarray, dl: np.ndarray) -> np.ndarray:
    return bin_dl_uniform(ell, dl)
```

- [ ] **Step 5: Verify exact agreement with the synthetic reference**

Add a test that dynamically imports `tsz_only/bandpower_theory.py`, divides its `1e12`-scaled covariance by `1e24`, and compares the complete matrix at `rtol=1e-13`.

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest -q tests/test_l1_m9_covariance.py tests/test_l1_m9_inference.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit only the task files**

```bash
git commit --only -m "fix: match synthetic tSZ bandpower covariance" \
  src/flamingo/inference/bandpowers.py \
  src/flamingo/inference/l1_m9.py \
  tests/test_l1_m9_covariance.py \
  tests/test_l1_m9_inference.py
```

---

### Task 2: Make full sky the scattered `q_cat -> infinity` theory limit

**Files:**
- Modify: `src/flamingo/inference/masked_ps.py:89-232`
- Create: `tests/test_masked_ps_inference.py`

**Interfaces:**
- Consumes: `MaskedTSZTheory.q_cat`, with `None` meaning infinity; sampled/fixed Cobaya parameters including explicit `B`.
- Produces: `MaskedTSZTheory.evaluate_bandpowers(**values)` using conditional moments `W1`, `W2`, and a full-sky limit consistent with analytic lognormal moments.

- [ ] **Step 1: Write failing tests for explicit B and full-sky moments**

```python
def test_masked_theory_declares_explicit_B():
    assert "B" in MaskedTSZTheory.params


@pytest.mark.parametrize("n_power, exponent", [(1, 0.5), (2, 2.0), (4, 8.0)])
def test_infinite_q_moment_matches_lognormal_limit(n_power, exponent):
    snr = jnp.array([[1.0, 10.0], [100.0, 1000.0]])
    sigma = 0.173
    got = conditional_An_undetected(
        snr, sigma_lnY=sigma, q_cat=np.inf, n_power=n_power, n_grid=512, nsig=8.0
    )
    np.testing.assert_allclose(got, np.exp(exponent * sigma**2), rtol=2e-6)
```

- [ ] **Step 2: Verify the tests fail before the theory change**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest -q tests/test_masked_ps_inference.py
```

Expected: the explicit `B` assertion fails and the current full-sky branch is shown not to apply moments.

- [ ] **Step 3: Unify finite-q and infinite-q theory evaluation**

Add `B` to `params`, use `values["B"]` in both `ParametricGNFWPressureProfile` and `build_snr_grid`, map `q_cat=None` to `np.inf`, and always construct `W1` and `W2`. Always call `cl_1h_masked(..., W2, k_damp=0.0)` and `cl_2h_masked(..., W1)` rather than using the unscattered `cl_1h`/`cl_2h` branch.

```python
q_cat = np.inf if self.q_cat is None else float(self.q_cat)
masks = {
    n: conditional_An_undetected(
        snr, sigma_lnY=values["sigma_lnY"], q_cat=q_cat,
        n_power=n, n_grid=SCATTER_GRID, nsig=SCATTER_NSIG,
    )
    for n in (1, 2)
}
```

- [ ] **Step 4: Add a GPU smoke test for the full-sky output**

Evaluate D3A inputs twice, at `q_cat=None` and a subclass/config with `q_cat=np.inf`, and require 18 finite positive bandpowers agreeing at `rtol=1e-10`.

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest -q tests/test_masked_ps_inference.py
```

Expected: all tests pass and JAX reports a CUDA device during the smoke test.

- [ ] **Step 5: Commit only the theory and test**

```bash
git commit --only -m "fix: include scatter in full-sky tSZ theory" \
  src/flamingo/inference/masked_ps.py \
  tests/test_masked_ps_inference.py
```

---

### Task 3: Produce exact qfrommap covariance components at an explicit parameter point

**Files:**
- Modify: `scripts/compute_l1_m9_customgnfw_bestfit_covariance.py`
- Modify: `tests/test_l1_m9_covariance.py`

**Interfaces:**
- Produces: `compute_covariances(a_sz: float, output_dir: Path, cases: tuple[str, ...]) -> dict`, plus `.npy` Gaussian, trispectrum, and full covariance components and one JSON metadata file.
- Consumes: an explicit `A_SZ`, fixed D3A/`alpha_SZ=1.12`/`B=1.41`/`sigma_lnY=0.173`, and exact qfrommap `f_sky` constants.

- [ ] **Step 1: Add failing routing, sky-fraction, and component tests**

```python
EXPECTED_F_SKY = {
    "fullsky": 1.0,
    "qgt50": 0.9972647840959187,
    "qgt20": 0.9858059150434103,
    "qgt10": 0.9435688947539681,
    "qgt5": 0.8595492784996496,
}


def test_qfrommap_f_sky_values_are_exact():
    assert QFROMMAP_F_SKY == EXPECTED_F_SKY


def test_covariance_component_assembly():
    result = build_covariance_18(raw_spectrum_fixture, 0.8)
    np.testing.assert_allclose(
        result["cov_full"], result["cov_gaussian"] + result["cov_trispectrum"]
    )
```

- [ ] **Step 2: Run the tests and verify the old qfrommz/approximation failures**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest -q tests/test_l1_m9_covariance.py
```

Expected: failures because the script loads qfrommz metadata and does not expose exact component covariance.

- [ ] **Step 3: Refactor the producer around explicit A_SZ and five cases**

Define:

```python
QFROMMAP_F_SKY = {
    "fullsky": 1.0,
    "qgt50": 0.9972647840959187,
    "qgt20": 0.9858059150434103,
    "qgt10": 0.9435688947539681,
    "qgt5": 0.8595492784996496,
}
Q_CUTS = {"qgt50": 50.0, "qgt20": 20.0, "qgt10": 10.0, "qgt5": 5.0}
```

For `fullsky`, call the same masked hmfast methods with `q_cat=np.inf` and `W1/W2/W4`. For masked cases, use the finite threshold. Replace the approximate `gaussian_covariance` and old `bin_trispectrum` calls with Task 1 utilities.

- [ ] **Step 4: Add a narrow command-line interface**

Support exactly:

```text
--a-sz FLOAT
--output-dir PATH
--case fullsky|qgt50|qgt20|qgt10|qgt5  (repeatable; defaults to all)
```

Reject CPU production unless `ALLOW_CPU=1` is explicitly set for unit tests. Metadata must include parameter values, q thresholds, qfrommap sky fractions, component diagnostics, source data filenames, and `jax_devices`.

- [ ] **Step 5: Verify a CPU-sized fixture and a GPU full-sky smoke computation**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest -q tests/test_l1_m9_covariance.py
python scripts/compute_l1_m9_customgnfw_bestfit_covariance.py \
  --a-sz -4.0953238 \
  --case fullsky \
  --output-dir /tmp/l1_m9_qfrommap_cov_smoke
```

Expected: tests pass; smoke metadata reports `cuda:0`; the full covariance is 18x18, symmetric, positive definite, and exactly reconstructs from saved components.

- [ ] **Step 6: Commit only the covariance producer and tests**

```bash
git commit --only -m "feat: compute exact qfrommap theory covariance" \
  scripts/compute_l1_m9_customgnfw_bestfit_covariance.py \
  tests/test_l1_m9_covariance.py
```

---

### Task 4: Iterate the amplitude-only full-sky MCMC and covariance

**Files:**
- Create: `scripts/iterate_l1_m9_asz_covariance.py`
- Create: `tests/test_l1_m9_asz_iteration.py`

**Interfaces:**
- Produces: `iterate(seed_a_sz: float, output_root: Path, max_iterations: int = 10) -> dict` and `chains/l1_m9_qfrommap_asz_covariance/converged.json`.
- Consumes: Task 3 `compute_covariances`, the full-sky 18-bin data, and `MaskedTSZTheory` with all parameters fixed except `A_SZ`.

- [ ] **Step 1: Write failing tests for the amplitude-only Cobaya info**

```python
def test_asz_fit_varies_only_asz(tmp_path):
    info = build_asz_info(tmp_path / "cov.npy", tmp_path / "chain")
    sampled = [name for name, cfg in info["params"].items() if isinstance(cfg, dict) and "prior" in cfg]
    assert sampled == ["A_SZ"]
    assert info["params"]["B"]["value"] == 1.41
    assert info["params"]["sigma_lnY"]["value"] == 0.173
    assert info["params"]["alpha_SZ"]["value"] == 1.12
```

Add a pure convergence test using injected covariance and chain callbacks:

```python
def test_iteration_stops_only_when_asz_and_covariance_are_stable(tmp_path):
    summary = iterate(
        -4.0953238, tmp_path, covariance_fn=fake_covariance,
        chain_fn=fake_chain, max_iterations=10,
    )
    assert summary["converged"] is True
    assert abs(summary["delta_A_SZ"]) < 1e-4
    assert summary["relative_covariance_change"] < 1e-3
```

- [ ] **Step 2: Run tests and verify the driver is absent**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest -q tests/test_l1_m9_asz_iteration.py
```

Expected: collection fails because the iteration driver does not exist.

- [ ] **Step 3: Implement deterministic per-iteration artifacts**

For iteration `NN`, write covariance under `iteration_NN/covariance/` and Cobaya output under `iteration_NN/chain/chain`. Use a broad `A_SZ` uniform prior `[-5.5, -3.0]`, reference `Normal(current_A_SZ, 0.02)`, and `Rminus1_stop=0.01`. Set D3A fixed values:

```python
FIXED = {
    "H0": 68.1,
    "omega_cdm": 0.11872788986038219,
    "omega_b": 0.022538784599999993,
    "n_s": 0.965,
    "sigma_8": 0.8025701499024616,
    "tau_reio": 0.0544,
    "alpha_SZ": 1.12,
    "sigma_lnY": 0.173,
    "B": 1.41,
}
```

Load the chain with `getdist.loadMCSamples(..., ignore_rows=0.3)` and define the best fit as the retained sample with minimum `chi2__*`.

- [ ] **Step 4: Implement both convergence checks and failure behavior**

After each fit, recompute covariance at the new best fit. Calculate:

```python
delta_a_sz = abs(new_a_sz - old_a_sz)
relative_covariance_change = np.linalg.norm(new_cov - old_cov) / np.linalg.norm(old_cov)
converged = delta_a_sz < 1e-4 and relative_covariance_change < 1e-3
```

If iteration 10 remains unstable, write `converged.json` with `converged: false` and raise `RuntimeError`; never label the last covariance final.

When both checks pass, call `compute_covariances` once more at the converged
`A_SZ` with `cases=("fullsky", "qgt50", "qgt20", "qgt10", "qgt5")` and
write those products under `final/covariance/`. Store the exact final metadata
path and all five covariance paths in `converged.json`.

- [ ] **Step 5: Run unit tests, then the production GPU fixed-point fit**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest -q tests/test_l1_m9_asz_iteration.py
CUDA_VISIBLE_DEVICES=0 python scripts/iterate_l1_m9_asz_covariance.py
```

Expected: `converged.json` reports both thresholds satisfied, records `B=1.41` and `sigma_lnY=0.173`, and points to a converged covariance computed at its own best-fit `A_SZ`.

- [ ] **Step 6: Commit code and small JSON summaries, not raw chains**

```bash
git commit --only -m "feat: iterate L1_m9 amplitude and covariance" \
  scripts/iterate_l1_m9_asz_covariance.py \
  tests/test_l1_m9_asz_iteration.py
```

---

### Task 5: Build the five final qfrommap science-chain configurations

**Files:**
- Modify: `scripts/run_masked_ps_chains.py:30-176`
- Create: `tests/test_run_masked_ps_chains.py`

**Interfaces:**
- Consumes: `converged.json`, final Task 3 covariance files, and five specified 18-bin data files.
- Produces: `build_info(case: str, rminus1: float = 0.01) -> dict` with exact D3A-centered priors and explicit fixed `B=1.41`.

- [ ] **Step 1: Write failing data-routing and prior tests**

```python
@pytest.mark.parametrize("case", ["qgt50", "qgt20", "qgt10", "qgt5"])
def test_masked_cases_use_qfrommap_data(case):
    data, _ = data_files(case)
    assert data.name == f"Dl_yy_L1_m9_masked_{case}_qfrommap_binned_18.txt"


def test_d3a_prior_centers_and_fixed_B(monkeypatch):
    params = parameters(a_sz_best_fit=-4.11)
    assert params["H0"]["prior"]["loc"] == 68.1
    assert params["n_s"]["prior"]["loc"] == 0.965
    assert params["omega_b"]["prior"]["loc"] == pytest.approx(0.0225387846)
    assert params["A_SZ"]["prior"] == {"dist": "norm", "loc": -4.11, "scale": 0.03}
    assert params["B"]["value"] == 1.41
```

Also evaluate the `omega_cdm` lambda at D3A inputs and require `0.11872788986038219`, proving the fixed neutrino density was subtracted.

- [ ] **Step 2: Run tests and verify current qfrommz/reference-center failures**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest -q tests/test_run_masked_ps_chains.py
```

Expected: failures on qfrommz filenames, reference cosmology centers, hard-coded old amplitude, and absent explicit B.

- [ ] **Step 3: Read the converged amplitude and final covariance paths**

Remove `A_SZ_BEST_FIT`. Load `chains/l1_m9_qfrommap_asz_covariance/converged.json`, require `converged is true`, and verify the final covariance metadata `A_SZ` equals the summary best fit before constructing any science-chain info.

- [ ] **Step 4: Implement the approved priors exactly**

Retain flat `sigma_8 in [0.6, 1.0]` and `Omega_m in [0.2, 0.5]`. Recenter only the prior-driven Gaussian cosmology parameters on D3A. Define CDM density as:

```python
"lambda Omega_m, H0, omega_b: "
"Omega_m * (H0 / 100.0)**2 - omega_b - 0.06 / 93.14"
```

Keep `alpha_SZ ~ Normal(1.12, 0.03)`, `sigma_lnY ~ Normal(0.173, 0.023)`, `tau_reio=0.0544`, and `B=1.41`.

- [ ] **Step 5: Add GPU and resolved-input gates**

Before calling Cobaya, require `jax.devices()[0].platform == "gpu"`. Write one preflight JSON containing devices, case-to-data/covariance mapping, best-fit amplitude, and resolved priors. Keep `resume=True` and isolated outputs under `chains/l1_m9_qfrommap_signal_only/{case}/chain`.

- [ ] **Step 6: Verify all five resolved configurations without sampling**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest -q tests/test_run_masked_ps_chains.py
python scripts/run_masked_ps_chains.py --dry-run
```

Expected: tests pass; dry run lists five `_qfrommap` data files, five matching final covariances, D3A centers, `B=1.41`, and a CUDA device.

- [ ] **Step 7: Commit only the runner and tests**

```bash
git commit --only -m "feat: configure L1_m9 qfrommap science chains" \
  scripts/run_masked_ps_chains.py \
  tests/test_run_masked_ps_chains.py
```

---

### Task 6: Run, monitor, and summarize all five production chains

**Files:**
- Create: `scripts/summarize_masked_ps_chains.py`
- Create: `tests/test_summarize_masked_ps_chains.py`
- Runtime outputs: `chains/l1_m9_qfrommap_signal_only/{fullsky,qgt50,qgt20,qgt10,qgt5}/`

**Interfaces:**
- Consumes: completed Cobaya chain roots and checkpoint YAML files.
- Produces: per-case `posterior_summary.json`, a combined `posterior_summary.json`, and a completion audit JSON.

- [ ] **Step 1: Write failing summary and convergence tests**

```python
def test_checkpoint_must_report_converged(tmp_path):
    checkpoint = tmp_path / "chain.checkpoint"
    checkpoint.write_text("sampler:\n  mcmc:\n    converged: false\n    Rminus1_last: 0.03\n")
    with pytest.raises(RuntimeError, match="not converged"):
        read_convergence(checkpoint)


def test_summary_contains_required_parameters(fake_getdist_samples):
    summary = summarize_samples(fake_getdist_samples)
    assert set(["H0", "sigma_8", "n_s", "omega_b", "Omega_m", "A_SZ", "alpha_SZ", "sigma_lnY", "S8"]) <= set(summary["parameters"])
```

- [ ] **Step 2: Run tests and verify the summarizer is absent**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest -q tests/test_summarize_masked_ps_chains.py
```

Expected: collection fails because the summarizer does not exist.

- [ ] **Step 3: Implement strict chain summarization**

Read each `chain.checkpoint` with `yaml.safe_load`, require `converged: true` and `Rminus1_last <= 0.01`, load samples with 30% burn-in, and write weighted mean, standard deviation, 68% interval, best sampled chi-square point, effective samples, and retained weight for every sampled/derived parameter.

- [ ] **Step 4: Run the five GPU chains**

Use one GPU per process and avoid oversubscribing either device. Start these
first two commands concurrently in separate terminals, and wait for both exit
codes:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/run_masked_ps_chains.py --case fullsky > logs/l1_m9_qfrommap_fullsky.log 2>&1
CUDA_VISIBLE_DEVICES=1 python scripts/run_masked_ps_chains.py --case qgt50 > logs/l1_m9_qfrommap_qgt50.log 2>&1
```

Then start these two commands concurrently and wait for both exit codes:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/run_masked_ps_chains.py --case qgt20 > logs/l1_m9_qfrommap_qgt20.log 2>&1
CUDA_VISIBLE_DEVICES=1 python scripts/run_masked_ps_chains.py --case qgt10 > logs/l1_m9_qfrommap_qgt10.log 2>&1
```

Finally run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/run_masked_ps_chains.py --case qgt5 > logs/l1_m9_qfrommap_qgt5.log 2>&1
```

Monitor actual process exit codes and checkpoint files; do not infer completion
from the presence of chain text files.

- [ ] **Step 5: Run the summarizer and complete requirement-by-requirement audit**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
python scripts/summarize_masked_ps_chains.py
pytest -q tests/test_l1_m9_covariance.py tests/test_l1_m9_inference.py \
  tests/test_masked_ps_inference.py tests/test_l1_m9_asz_iteration.py \
  tests/test_run_masked_ps_chains.py tests/test_summarize_masked_ps_chains.py
```

The audit JSON must prove: five correct data files, covariance-at-best-fit equality, qfrommap sky fractions, exact covariance reconstruction, positive definiteness, D3A Gaussian centers, wider amplitude prior, fixed `B=1.41`, CUDA execution, and five converged checkpoints.

- [ ] **Step 6: Commit the summarizer and tests only**

```bash
git commit --only -m "feat: summarize L1_m9 qfrommap chains" \
  scripts/summarize_masked_ps_chains.py \
  tests/test_summarize_masked_ps_chains.py
```
