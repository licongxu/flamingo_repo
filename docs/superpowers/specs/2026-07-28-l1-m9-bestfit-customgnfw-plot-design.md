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

## Original 18-point figure and outputs

Use logarithmic axes. Draw the empirical bandpowers as connected markers and
the best-fit custom-GNFW total as one smooth curve. Do not add component
curves, a ratio panel, or covariance error bars. Annotate the fixed best-fit
parameters without adding another plotted series.

Write reproducible PNG and PDF outputs under `figures/`, and keep the plotting
entry point in `scripts/`.

The original 18-point data file, its covariance, and the resulting first
figure must not be modified when producing the high-multipole figure.

## Independent high-multipole figure

Create a second figure from the original fiducial L1_m9 full-sky Compton-y
map. Recompute the unbinned full-sky spectrum through `ell = 10000`,
subtracting the monopole and deconvolving the squared HEALPix pixel window.
Do not derive this figure from, modify, or overwrite the 18-point likelihood
data.

Follow the high-multipole prescription in
`/home/lxu/scratch/tsz_cnc_paper_plots/kernel_analysis/noise_curve.py`:

- logarithmic edges with exact width `Delta ln ell = 0.4`;
- geometric bin centres;
- left-inclusive and right-exclusive bins, with the last right edge included;
- bin-averaged `C_ell`, converted to `D_ell` at each geometric centre; and
- 12 geometrically spaced theory samples per bin.

The upper panel covers `100 <= ell <= 10000` and contains the log-binned map
bandpowers, best-fit custom-GNFW total, and best-fit 1-halo term. Do not draw
the 2-halo term. The lower panel covers the identical full range and shows
map divided by best-fit total in every displayed bin.

Write this independent figure as
`figures/l1_m9_fullsky_bestfit_customgnfw_highell.png` and `.pdf`.

## Verification

Check that:

- the script runs in the repository virtual environment;
- both output files exist and are nonempty;
- both plotted arrays are finite and positive;
- the theory is exactly the sum of its 1-halo and 2-halo terms;
- the empirical series contains all 18 input bandpowers; and
- the rendered legend contains exactly the two requested series.

For the high-multipole figure, additionally check that:

- the source map is read-only and no existing data product changes;
- every displayed bin centre lies between `ell = 100` and `ell = 10000`;
- adjacent bin edges differ by `Delta ln ell = 0.4`;
- the data, total theory, 1-halo theory, and ratio are finite and positive;
- `total = 1h + 2h` numerically even though the 2-halo line is not drawn; and
- both panels span the same complete multipole range.
