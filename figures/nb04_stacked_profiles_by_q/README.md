# Notebook 04 — Empirical Stacked y-Profiles Split by Detection Significance q

**Source:** `notebooks/04_stacked_profiles_by_q.ipynb`
**Inputs:**
- the FLAMINGO **high-resolution** Compton-y map, `nside16384.fits` (NSIDE = 16384),
- the `y0q` Arnaud-B=1 catalogue for `q`, joined to the original catalogue for sky positions.

**Figures (2, notebook-rendered PNG):** `nb04_stack_by_q_log`, `nb04_bkg_subtracted_vs_raw_log`.
See note on resolution at the end.

---

## 1. Method

The **same fully empirical aperture-normalised estimator as notebook 03** (no pressure model, no
central `y_0`): in scaled radius `x = theta / theta_500`,
`ybar_{i,a} = mean(y_p)` (equal pixel weights), `y_norm,a = Y_500,a / (pi theta_500,a^2)` with
`Y_500,a = sum_{theta_p < theta_500} y_p Omega_pix`, `g_{i,a} = ybar_{i,a} / y_norm,a`, and the
equal-weight stack `fhat_i = mean_a g_{i,a}`. The Arnaud A10 projected GNFW reference is normalised
the same aperture way and annulus-averaged in the same `x` bins.

The new ingredient is **splitting by catalogue detection significance `q`** into **non-overlapping
(differential) bins**, so each stack is an independent sample. Because all stacks share the aperture
normalisation, shape differences are intrinsic, not amplitude offsets.

| q bin | N total | N used (cap 4000) | median theta_500 |
|---|---|---|---|
| `q > 30` | 26 | 26 | 15.7 arcmin |
| `10 < q < 30` | 510 | 510 | 8.6 arcmin |
| `5 < q < 10` | 1856 | 1856 | 5.7 arcmin |
| `2 < q < 5` | 11493 | 4000 (random subsample) | 3.8 arcmin |

## 2. Background-subtracted variant

A local background is removed **before** forming the profile: `b` = median `y` in a `3–5 theta_500`
ring around each cluster, subtracted from every pixel, so `g = (ybar_i - b) / (y_norm_raw - b)`. This
targets the correlated 2-halo term and the sky floor. The second figure overlays the
background-subtracted stacks (solid, markers) on the raw stacks (thin lines) and A10.

## 3. Key results

- **High-q (massive, well-resolved) clusters track A10 into the core** and have the cleanest inner
  profiles; **low-q clusters are noisier** at small `x` (their theta_500 approaches the pixel scale)
  and flatten toward the pixel/background floor.
- **All raw stacks show an excess over A10 in the outskirts** (`x > 1`) that grows toward lower `q`:
  the uncorrected correlated/2-halo signal plus sky floor. The `2<q<5` bin sits highest there.
- **Background subtraction removes the outskirt floor** (`x >~ 1`) and brings the stacks down toward
  the projected A10 shape, while leaving the cores essentially unchanged. Bins that go to zero or
  negative beyond the background ring drop off the log axis, as expected.

## 4. Reproduce / note

Reads the 3 GB high-resolution map and loops `query_disc` per cluster for each q bin (raw and
background-subtracted). The figures here are the **notebook-rendered PNGs** (inline resolution
~100 dpi), saved as the verified results; regenerate from the notebook as vector PDF + 300 dpi PNG
if a publication-grade version is needed.
