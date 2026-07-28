# L1_m9 Simple-GNFW Theory Covariance Design

## Goal

Compute the full-sky theory covariance for the 18 fiducial FLAMINGO
`L1_m9` Compton-y bandpowers in
`data_paper/binned_bandpowers/Dl_yy_L1_m9_fullsky_binned_18.txt`. The
covariance is the sum of its Gaussian and connected one-halo trispectrum
terms and will subsequently be used to fit `A_SZ` and `alpha_SZ`.

## Scope

This first deliverable computes and validates the covariance only. The later
Cobaya fit remains part of the overall task but must consume this validated
covariance rather than the older reference covariance.

The calculation covers all 18 bandpower bins, with effective multipoles from
10 to 959.5. It is full-sky, with `f_sky = 1`, and uses no instrumental-noise
or foreground covariance.

## Theory

Use the fixed FLAMINGO D3A cosmology from
`flamingo.catalogue.frame.D3A_COSMOLOGY`. Use hmfast's
`GNFWPressureProfile` with its established simple-GNFW defaults
(`P0 = 8.13`, `c500 = 1.156`, `alpha = 1.062`, `beta = 5.4807`,
`gamma = 0.3292`) and hydrostatic mass-bias parameter `B = 1`.

The theory spectrum entering the Gaussian covariance includes both the
one-halo and two-halo terms:

```text
D_ell = ell (ell + 1) (C_ell^1h + C_ell^2h) / (2 pi).
```

No additional lognormal pressure scatter is applied:
`sigma_lnY = 0`. In particular, the power spectrum and trispectrum receive
none of the `exp(n sigma_lnY^2)` boosts used in some reference calculations.

All halo masses are physical solar masses, never `M_sun/h`. The primary
integration grids are 64 logarithmic mass samples over
`1e11 <= M/M_sun <= 1e16`, 96 logarithmic redshift samples over
`0.005 <= z <= 3`, and a smooth logarithmic multipole grid spanning all 18
bins. Halo-model consistency counterterms remain disabled.

## Covariance

The Gaussian term is diagonal and follows the full-sky Knox expression for
each band:

```text
Cov_G(b,b) = 2 D_b^2 / [(2 ell_eff,b + 1) Delta_ell_b].
```

The non-Gaussian term is hmfast's connected one-halo angular trispectrum,
`HaloModel.trispectrum_1h`, converted from `C_ell` units to the product of two
`D_ell` factors and uniformly averaged over both integer-multipole axes of
each pair of bins.

The final full-sky covariance is

```text
Cov_full = Cov_G + T_binned / (4 pi).
```

## Implementation

Add one focused, GPU-capable script under `scripts/`. Reuse the established
bin edges from the input bandpower file's reference calculation and keep the
calculation independent of the measured bandpower amplitudes. The script
will use JAX float64 and will be run on an idle CUDA device.

Factor the pure NumPy binning and covariance-assembly operations into
testable functions. Tests will cover bin shapes, constant-field binning,
the diagonal Gaussian formula, and the exact component-sum identity.

## Outputs

Write the following files under `data_paper/covariance/`:

- `cov_gaussian_L1_m9_fullsky_Dl_yy_binned_18.npy`
- `trispectrum_L1_m9_fullsky_Dl_yy_binned_18.npy`
- `cov_full_L1_m9_fullsky_Dl_yy_binned_18.npy`
- `cov_full_L1_m9_fullsky_Dl_yy_binned_18.csv`
- `Dl_yy_simplegnfw_B1_theory_binned_18.txt`
- `covariance_L1_m9_fullsky_metadata.json`

The NumPy arrays retain raw dimensionless `D_ell` units, matching the
reference Cobaya likelihood. Metadata records the cosmology, profile,
integration grids, formula, GPU device, and validation diagnostics.

## Verification

Before the covariance is accepted:

1. The Gaussian, trispectrum, and full covariance arrays must each be
   finite and have shape `(18, 18)`.
2. The full covariance must equal
   `Cov_G + T_binned / (4 pi)` to floating-point precision.
3. The full covariance must be symmetric within numerical precision.
4. Its symmetrized eigenvalues must all be positive.
5. Its diagonal entries must be positive, and the connected term must not
   reduce any diagonal variance.
6. The saved fiducial theory spectrum must include both one-halo and
   two-halo contributions.
7. Runtime output and metadata must confirm CUDA execution.

Only after these checks pass will the later fixed-cosmology Cobaya fit use
this covariance.
