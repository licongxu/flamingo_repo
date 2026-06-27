# Notebook 03 — Empirical Stacked tSZ Pressure Profiles and GNFW Fits

**Source:** `notebooks/03_stacked_pressure_profiles.ipynb`
**Inputs:**
- the FLAMINGO **high-resolution** Compton-y map, `nside16384.fits` (NSIDE = 16384, pixel ~0.21 arcmin),
- the `M500c > 5e13` cluster catalogue (`M_500c_Msun`, `z`, `theta500_arcmin`, sky positions).

**Figures (5, notebook-rendered PNG):**
`nb03_massive_stack_log`, `nb03_small_stack_log`, `nb03_gnfw_param_variations`,
`nb03_gnfw_fits_massive_small`, `nb03_100k_stack_fit`. See note on resolution at the end.

---

## 1. The empirical, aperture-normalised estimator

We measure a **fully empirical** stacked Compton-y profile directly from the pixelized map, with
**no pressure-profile model** and **without** the central value `y_0` (not measurable on a pixelized
map). For each cluster `a`, working in scaled angular radius `x = theta / theta_500,a`:

- **Annular mean** in each scaled bin `i` (equal pixel weights `w_p = 1` on the equal-area HEALPix map):
  `ybar_{i,a} = mean( y_p : p in bin i )`.
- **Aperture normalisation** from the integrated Compton-Y inside `R_500`, converted to an aperture mean:
  `Y_500,a = sum_{theta_p < theta_500,a} y_p * Omega_pix`, `y_norm,a = Y_500,a / (pi theta_500,a^2)`.

The per-cluster dimensionless profile and the equal-weight stack are
`g_{i,a} = ybar_{i,a} / y_norm,a`, `fhat_i = mean_a g_{i,a}`. The plotted point is at the
area-weighted bin centre `x_i = (2/3)(hi^3 - lo^3)/(hi^2 - lo^2)`. Errors: shaded SEM
(`std / sqrt N`) for the stack and the 16–84 percentile band for the cluster-to-cluster scatter.

The **Arnaud A10** projected GNFW reference is processed the *same way*: projected along the line of
sight, divided by its **own** aperture mean inside `x < 1`, and annulus-averaged in the same `x`
bins, so the comparison is consistent (the amplitude `P_0` cancels under this normalisation).

## 2. Samples

| Sample | N | Mass / selection | median theta_500 |
|---|---|---|---|
| massive | 1000 | `M_500c in [7.1e14, 2.07e15]` (most massive in `0<z<3`) | 3.26 arcmin |
| small | 1000 | bootstrap of the non-massive halos, `M_500c ~ 5e13–6e14` | 0.95 arcmin |
| 100k | 100000 | the 100k most massive in `0<z<3` (`M_500c > 1.7e14`, `z_med=0.63`) | 1.62 arcmin |

## 3. GNFW fits

We fit the projected GNFW with **`c_500` and `beta` free** and `P_0, gamma, alpha` fixed at A10,
through the identical projection + aperture-normalisation + annulus-averaging pipeline, minimising
the unweighted log-space residual on `x in [0.01, 2.0]` by a refined 2-D grid search.

| Stack | `c_500` | `beta` | RMS residual | max\|resid\| |
|---|---|---|---|---|
| massive (1000) | **0.869** | **5.613** | 5.9% | 19.7% |
| small (1000) | **0.775** | **3.600** | 13.5% | 73.4% |
| 100k most massive | **1.275** | **3.800** | 5.8% | 18.9% |

(A10 reference: `c_500 = 1.177`, `beta = 5.4905`.)

## 4. Key results

- **Massive clusters follow a slightly more concentrated, similarly steep profile** than A10
  (`c_500 = 0.87` vs 1.18, `beta = 5.6` vs 5.49); the empirical stack sits below A10 in the core
  (A10 over-predicts the central rise on this aperture-normalised footing) and the best fit matches
  to ~6% RMS over `x < 2`.
- **Small-mass clusters are shallower in the outskirts** (`beta = 3.6`): they carry a strong excess
  above A10 beyond `x ~ 1`, the noisy inner bins (theta_500 near the ~0.2 arcmin pixel scale) inflate
  the residual to ~13% RMS. This outskirt excess is the correlated 2-halo / sky floor, addressed by
  the background subtraction in notebook 04.
- **The 100k-cluster stack** (a mass-weighted average dominated by intermediate masses) fits with
  `c_500 = 1.275`, `beta = 3.80` at 5.8% RMS, sitting between A10 and the small-mass shape, with the
  characteristic outskirt excess over A10 at `x > 1`.
- `nb03_gnfw_param_variations` shows the sensitivity of the aperture-normalised projected shape to
  each GNFW parameter (`c_500, gamma, alpha, beta`) against the massive stack.

## 5. Reproduce / note

The stacks read the 3 GB high-resolution map and loop `query_disc` per cluster; the 100k online
stack is the expensive step. The figures here are the **notebook-rendered PNGs** (inline resolution
~100 dpi), saved as the verified results. They can be regenerated as vector PDF + 300 dpi PNG from
the notebook if a publication-grade version is needed.
