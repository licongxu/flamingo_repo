# L1_m9 Best-Fit Custom-GNFW Power-Spectrum Plot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate reproducible PNG and PDF figures containing only the empirical full-sky L1_m9 tSZ bandpowers and the rerun chain's best-fit D3A custom-GNFW total prediction.

**Architecture:** Add a small public spectrum evaluator to the existing fixed-D3A custom-GNFW theory class, then use it from one focused plotting script. Test the numerical interface and plot inputs before rendering the final artifacts.

**Tech Stack:** Python, NumPy, JAX/hmfast, Matplotlib, pytest.

## Global Constraints

- Run every command after `source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate`.
- All theory calculations use hmfast.
- Halo masses remain in physical `M_sun`, never `M_sun/h`.
- Plot exactly two series: the 18 empirical L1_m9 bandpowers and the best-fit custom-GNFW total.
- Fix D3A cosmology, `A_SZ = -4.1095805`, `alpha_SZ = 0.97447729`, and `B = 1.41`.
- Do not plot the superseded posterior curve, the former `B = 1.0` fiducial curve, separate 1-halo or 2-halo curves, covariance error bars, or a ratio panel.

---

### Task 1: Public custom-GNFW spectrum evaluation and two-series figure

**Files:**
- Modify: `src/flamingo/inference/l1_m9.py`
- Create: `scripts/plot_l1_m9_fiducial_customgnfw.py`
- Modify: `tests/test_l1_m9_inference.py`
- Create at runtime: `figures/l1_m9_fullsky_bestfit_customgnfw.png`
- Create at runtime: `figures/l1_m9_fullsky_bestfit_customgnfw.pdf`

**Interfaces:**
- Consumes: `L1M9CustomGNFWTheory._evaluate_cl(A_SZ, alpha_SZ)` and the module's `ELL_SMOOTH`.
- Produces: `L1M9CustomGNFWTheory.evaluate_spectrum(A_SZ, alpha_SZ) -> dict[str, np.ndarray]` with keys `ell`, `1h`, `2h`, and `total`, all in `D_ell^yy`.
- Produces: `load_plot_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`, returning empirical ell, empirical displayed `D_ell / 1e-12`, theory ell, and theory displayed `D_ell / 1e-12`.

- [ ] **Step 1: Write failing numerical tests**

Add tests that initialize `L1M9CustomGNFWTheory`, evaluate
`(-4.1095805, 0.97447729)` with the profile fixed at `B = 1.41`, and assert:

```python
result = theory.evaluate_spectrum(-4.1095805, 0.97447729)
assert set(result) == {"ell", "1h", "2h", "total"}
assert result["ell"].shape == result["total"].shape
assert np.all(np.isfinite(result["total"]))
assert np.all(result["total"] > 0.0)
np.testing.assert_allclose(result["total"], result["1h"] + result["2h"])
```

- [ ] **Step 2: Run the targeted test and verify failure**

Run:

```bash
pytest tests/test_l1_m9_inference.py -q
```

Expected: failure because `evaluate_spectrum` does not exist.

- [ ] **Step 3: Add the minimal public spectrum evaluator**

Implement:

```python
def evaluate_spectrum(self, A_SZ: float, alpha_SZ: float) -> dict[str, np.ndarray]:
    cl_1h, cl_2h = self._evaluate_cl(float(A_SZ), float(alpha_SZ))
    prefactor = ELL_SMOOTH * (ELL_SMOOTH + 1.0) / (2.0 * np.pi)
    dl_1h = prefactor * np.asarray(cl_1h)
    dl_2h = prefactor * np.asarray(cl_2h)
    return {
        "ell": ELL_SMOOTH.copy(),
        "1h": dl_1h,
        "2h": dl_2h,
        "total": dl_1h + dl_2h,
    }
```

Refactor `evaluate_bandpowers` only enough to bin the public method's `1h`
and `2h` arrays, preserving all existing behavior.

- [ ] **Step 4: Run the targeted numerical tests**

Run:

```bash
pytest tests/test_l1_m9_inference.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Add the plotting script**

The script will:

```python
DATA_FILE = Path("data_paper/binned_bandpowers/Dl_yy_L1_m9_fullsky_binned_18.txt")
OUTPUT_STEM = Path("figures/l1_m9_fullsky_bestfit_customgnfw")
A_SZ = -4.1095805
ALPHA_SZ = 0.97447729
B = 1.41
DISPLAY_SCALE = 1e12
```

It will load all 18 two-column empirical rows, evaluate the public spectrum
method, check finite positive arrays and exact 1h+2h addition, and draw one
connected-marker empirical series plus one smooth theory series on logarithmic
axes. The legend will have exactly those two entries. It will save PNG and PDF.

- [ ] **Step 6: Run the script and inspect the artifacts**

Run:

```bash
MPLBACKEND=Agg python scripts/plot_l1_m9_fiducial_customgnfw.py
```

Expected: both output paths are printed, exist, and are nonempty. Open the PNG
for visual inspection and confirm two legend entries, readable labels, and no
unrequested component or ratio curves.

- [ ] **Step 7: Run focused and full verification**

Run:

```bash
pytest tests/test_l1_m9_inference.py -q
pytest -q
git diff --check
```

Expected: all tests pass and the diff check is clean.

- [ ] **Step 8: Commit the implementation**

```bash
git add src/flamingo/inference/l1_m9.py tests/test_l1_m9_inference.py \
  scripts/plot_l1_m9_fiducial_customgnfw.py \
  figures/l1_m9_fullsky_bestfit_customgnfw.png \
  figures/l1_m9_fullsky_bestfit_customgnfw.pdf \
  docs/superpowers/plans/2026-07-28-l1-m9-bestfit-customgnfw-plot.md
git commit -m "feat: plot L1_m9 best-fit custom-GNFW spectrum"
```
