# L1_m9 Best-Fit Custom-GNFW Power-Spectrum Plot Design

## Goal

Plot exactly two full-sky total tSZ power-spectrum series:

1. the 18 empirical fiducial FLAMINGO L1_m9 bandpowers; and
2. the hmfast custom-GNFW prediction at fixed D3A cosmology with the rerun
   chain's best fit: `A_SZ = -4.1095805`, `alpha_SZ = 0.97447729`, and
   `B = 1.41`.

The superseded posterior values `A_SZ = -4.37210` and
`alpha_SZ = 0.93232`, and the former fiducial curve at
`A_SZ = -4.094622`, `alpha_SZ = 1.12`, and `B = 1.0`, are not included.

## Inputs and computation

Read the empirical points from
`data_paper/binned_bandpowers/Dl_yy_L1_m9_fullsky_binned_18.txt`.
Evaluate the existing `L1M9CustomGNFWTheory` machinery on a dense multipole
grid, retaining its fixed custom-GNFW shape, physical-mass integration grid,
D3A cosmology, `B = 1.41`, and 1-halo plus 2-halo total.

Both series use
`D_ell^yy = ell (ell + 1) C_ell^yy / (2 pi)`. The data file's values are
stored in units of `1e-12`; the plotted values will use the file's displayed
scale and the theory will be converted to that same scale.

## Figure and outputs

Use logarithmic axes. Draw the empirical bandpowers as connected markers and
the best-fit custom-GNFW total as one smooth curve. Do not add component
curves, a ratio panel, covariance error bars, or posterior annotations.

Write reproducible PNG and PDF outputs under `figures/`, and keep the plotting
entry point in `scripts/`.

## Verification

Check that:

- the script runs in the repository virtual environment;
- both output files exist and are nonempty;
- both plotted arrays are finite and positive;
- the theory is exactly the sum of its 1-halo and 2-halo terms;
- the empirical series contains all 18 input bandpowers; and
- the rendered legend contains exactly the two requested series.
