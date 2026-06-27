# Notebook 05 — The tSZ Angular Power Spectrum: FLAMINGO Map vs hmfast

**Source:** `notebooks/05_tsz_power_spectrum.ipynb`
**Inputs:**
- the FLAMINGO **L2p8_m9** full-sky lightcone Compton-y map (`y_unlensed_L2p8_m9_lc0.fits`, NSIDE = 4096),
- the matched `y0q` cluster catalogue (columns `z`, `M_500c_Msun`, `Y_5R500c_Mpc2`, `r_comoving_Mpc`).

**Figure (1):** `nb05_tsz_power_spectrum` (`.pdf` vector + `.png` at 300 dpi). A sidecar
`nb05_manifest.json` records the git hash, resolution, and the key scalar results.

---

## 1. What this notebook does

It measures the **tSZ (thermal Sunyaev–Zel'dovich) angular power spectrum** of the FLAMINGO
hydro lightcone two independent ways and overlays them on a halo-model prediction:

1. **Full-sky map measurement** — the angular power spectrum `D_ell` of the simulated y-map,
   measured directly with `healpy.anafast`.
2. **Discrete catalogue summation** — the low-ell ("white", shot-noise) normalisation of the
   1-halo term obtained as a direct sum over the integrated Compton-Y of every catalogue cluster.
3. **hmfast halo model** — the analytic 1-halo + 2-halo prediction from the differentiable JAX
   halo model, using the Arnaud et al. (2010) universal pressure profile with hydrostatic bias
   `B = 1`, for comparison.

All three use the FLAMINGO **D3A cosmology** (`h = 0.681`).

---

## 2. Result 1 — full-sky tSZ power spectrum from the map (`anafast`)

The y-map is read at NSIDE = 4096 (monopole `<y> = 1.536e-6`). The pipeline is:

1. Subtract the monopole: `y -> y - <y>`.
2. `C_ell = anafast(y_nomono, lmax=6000)`.
3. Deconvolve the **pixel window function** and apply the sky fraction:
   `C_ell -> C_ell / (pixwin^2 * f_sky)`. Here `f_sky = 1.000` (the lightcone map is full-sky).
4. Form `D_ell = ell(ell+1) C_ell / 2pi` and average into 30 log-spaced ell bins over
   `10 <= ell <= 6000`.

These are the **crimson points** in the figure. They trace the canonical tSZ shape: a slow rise
toward a broad peak near `ell ~ 3000`, where the one-halo term of massive, low-redshift clusters
dominates.

## 3. Result 2 — discrete catalogue summation (low-ell 1-halo normalisation)

For `ell * theta_max << 1` a cluster is unresolved, so its harmonic-space profile tends to its
**integrated Compton-Y**, `tilde y_ell -> Y`. In that regime the 1-halo term becomes **white in
`C_ell`** (flat power), hence `D_ell ~ ell^2`. The amplitude is a pure sum/integral of `Y^2`.

The catalogue stores the *spherical* Compton-Y inside `5 R_500c` as `D_A^2 Y` in physical
`Mpc^2` (column `Y_5R500c_Mpc2`). The **angular** integrated y (solid-angle units, sr) is therefore

```
Y_ang = Y_5R500c_Mpc2 / D_A^2 ,   D_A = r_comoving / (1 + z)   [proper Mpc, matches D3A].
```

Two estimates of the white level `C_white` are computed:

**(a) Direct discrete summation over the catalogue** (the "discrete summation"):

```
C_white = (1 / 4pi) * sum_a  Y_ang,a^2          ->  2.936e-16
```

This is a single-realisation shot-noise sum over every cluster `a` in the lightcone, with no halo
model or theory `dn/dM` in the loop. It is the **dotted purple** curve, plotted as
`D_ell = ell(ell+1) C_white / 2pi` for `ell <~ 300` (where the unresolved approximation holds).

**(b) Tinker08-HMF reweighting** — replaces the single realisation by theory expectation. The
catalogue `<Y_ang^2>` is binned in `(log10 M_500c, z)`, then reweighted by the expected counts per
bin from the Tinker (2008) mass function and the D3A comoving volume element:

```
N_T08(M, z) = 4pi * (dV/dz/dOmega) * (dn/dlnM) * dlnM * dz
C_white = (1 / 4pi) * sum_bins  <Y_ang^2> * N_T08   ->  2.421e-16
```

This is the **dotted red** curve. The two `C_white` agree to ~20%, confirming that the catalogue
realisation is consistent with the Tinker08 expectation for the same `Y-M` content.

## 4. Result 3 — hmfast halo-model overlay

The analytic prediction uses the Arnaud et al. (2010) universal pressure profile,
`P0 = 8.403, c500 = 1.177, gamma = 0.3081, alpha = 1.0510, beta = 5.4905`, with hydrostatic-bias
`B = 1.0` (i.e. `(1-b) = 1`), through `GNFWPressureProfile` + `tSZTracer`, integrated by `HaloModel`
on grids `M in [1e11, 10^15.5] Msun`, `z in [1e-3, 3]`, `ell in [10, 6000]`:

- **black solid** — total `D_ell` (1-halo + 2-halo), Tinker08 HMF;
- **blue dashed** — 1-halo term;
- **orange dashed** — 2-halo term (sub-dominant everywhere, peaks ~`ell ~ 300`).

Reference value: `D_ell(ell ~ 3000) = 1.59e-12` for the Tinker08 total.

---

## 5. Takeaways

- The hmfast A10 `B = 1` halo model reproduces the FLAMINGO map power spectrum well across
  `ell ~ 100-6000`, with the 1-halo term dominating near the peak.
- The **discrete catalogue summation** `C_white = sum_a Y_ang^2 / 4pi = 2.94e-16` provides an
  independent, model-free anchor of the low-ell 1-halo amplitude, and is consistent (to ~20%) with
  both the Tinker08-reweighted estimate and the halo-model curve in the unresolved regime.
- The `D_ell ~ ell^2` slope of the white-normalisation curves is the expected unresolved-source
  scaling and is only shown for `ell <~ 300`, where `ell * theta_max < 1`.

---

## 6. Reproduce

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
# run the notebook via jupyter, or the standalone script that mirrors it cell-for-cell.
```

Key scalars are cached in `nb05_manifest.json` (NSIDE, LMAX, f_sky, `C_white` values, A10 params,
git hash, input paths). PNG is rasterised at 300 dpi from the same figure as the vector PDF.
