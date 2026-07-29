# Masking-Radius Null Test Design

## Goal

The paper's masked tSZ power spectra excise every detected cluster with a disc
of radius `4 * theta_500`. That factor of 4 is inherited from the synthetic
painted-map prescription and has never been validated against the maps
themselves. This test measures how much the measured spectrum actually depends
on it.

For a fixed detection threshold `q`, grow the masking radius from `0` upwards
and measure the Compton-y auto-spectrum at each step. The residual power should

1. fall steeply while the disc is still cutting into the cluster, then
2. flatten once the disc encloses essentially all of that cluster's tSZ signal.

Where the flattening starts is the smallest defensible masking radius. If it
starts well below `4 theta_500`, the production choice is safe; if the curve is
still falling at `4`, the paper is under-masking and the masked spectra retain
cluster signal they claim to have removed.

Repeated at several `q` so the answer is not specific to one cluster
population, and on two independent simulations (`L1_m9`, `L2p8_m9`) so it is
not specific to one box.

## Inputs

Fiducial lightcone-0 Compton-y maps (`nside = 4096`) and the yang26-rotated,
`alpha_SZ = 1.12` fixed-alpha `q` catalogues:

```text
/rds/rds-lxu/flamingo/L1_m9/maps/y_unlensed_L1_m9_lc0_nside4096.fits
/rds/rds-lxu/flamingo/L1_m9/catalogues/halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommz_alpha_fixed_1p12.csv

/rds/rds-lxu/flamingo/L2p8_m9/lightcone0/healpix_map/y_unlensed_L2p8_m9_lc0.fits
/rds/rds-lxu/flamingo/L2p8_m9/lightcone0/catalogues/halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_qfrommz_alpha_fixed_1p12.csv
```

## Grid

- `q > 20, 10, 5, 1` (the paper's headline cut is `q > 5`).
- `R = r / theta_500 = 0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8`.

`R = 0` is the unmasked map. It is run through NaMaster with a unit mask rather
than through `anafast`, so the baseline is estimated by exactly the same code
path as every masked point, and it doubles as the estimator validation.

The grid extends to `8 theta_500`, twice the production radius, so that the
plateau is bracketed rather than assumed.

## Deviation from the production prescription

Production uses `radius = max(4 * theta_500, 2 * FWHM)` with `FWHM = 10 arcmin`
-- a floor inherited from the painting truncation radius of the synthetic maps.

**This test drops the floor and uses exactly `R * theta_500`.** A 20 arcmin
floor exceeds `R * theta_500` for most clusters at small `R` (the median
`theta_500` is 7.6 arcmin at `q > 5`), so keeping it would pin the small-`R` end
of the sweep to a constant radius and flatten the very trend being measured.
The FLAMINGO maps are unsmoothed, so there is no beam in *these* data that the
floor would be correcting for.

The production points are therefore not recomputed; they are read from the
stored `..._alpha_fixed_1p12_logbins_dln0p4_lmax10000.txt` files and overlaid.

Everything else is the production pipeline unchanged: `theta_500 = R_500c /
D_A(z)` in the D3A cosmology, yang26-rotated positions, binary `query_disc`
mask, C2 apodization at 0.25 deg, mask-weighted monopole subtracted, NaMaster
MASTER per-ell to `lmax = 10000`, HEALPix pixel window deconvolved, and the two
standard binnings (18 Planck bins, 12 log bins of `Delta ln ell = 0.4`).

## Random-position control

The masked spectrum keeps falling at large `R` for two reasons that the sweep
alone cannot separate: real cluster signal being removed, and sky simply being
removed. MASTER decoupling is designed to undo the second, but at the
aggressive end (`q > 1` at `8 theta_500` keeps only half the sky) that is an
assumption worth measuring.

The control rebuilds every mask with the *same number of discs and the same
radius distribution*, but with centres drawn uniformly over the sphere so they
no longer sit on clusters. Any decrement it shows is estimator response to lost
area, not physics. Run for `L1_m9` at `q > 5` (headline cut) and `q > 1` (worst
area loss).

## Verification

1. **Estimator**: the `R = 0` unit-mask run must reproduce the stored full-sky
   bandpowers. Fails the run if the max fractional deviation exceeds `5e-3`.
2. **Rotation**: the existing `rotation_sanity` check must show rotated
   positions tracing tSZ peaks better than natural ones.
3. **Sky fraction**: `f_sky` at `R = 4, q > 5` must match the stored production
   metadata (`0.9615` for `L1_m9`).
4. **Trajectory**: the sweep at `R = 4` must land on the stored production
   masked bandpowers, up to the dropped 20 arcmin floor.

## Outputs

Per-point caches (resumable, one file per `(variant, q, R)`) plus the summary
table in `data_paper/masking_radius_null_test/`, and three figures in
`figures/`:

- `masking_radius_null_test` -- surviving power `D_ell(R) / D_ell(unmasked)`.
- `masking_radius_convergence` -- `D_ell(R) / D_ell(8 theta_500) - 1`, the
  fractional power still left to remove at radius `R`. This is the direct
  systematic-error budget for the choice of `4`.
- `masking_radius_random_control` -- real masking against the random control.

## Scripts

```text
scripts/masking_radius_null_test.py        # the sweep
scripts/masking_radius_random_control.py   # the control
scripts/plot_masking_radius_null_test.py   # figures + summary table
```
