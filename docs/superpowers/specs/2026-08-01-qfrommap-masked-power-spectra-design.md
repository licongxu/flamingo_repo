# Q-from-map Masked tSZ Power Spectra Design

## Objective

Recompute the existing FLAMINGO masked tSZ power-spectrum products for all
nine L1_m9 feedback maps and all eight L2p8_m9 lightcone maps, changing only
the cluster-selection observable from `q_from_mz` to `q_from_aperture` in the
`*_qfrommap.csv` catalogues. Preserve every existing bandpower and figure.

## Inputs

- L1 uses the nine corrected, stable-identity catalogues staged below
  `/rds/rds-lxu/flamingo/.hbt_join_fix_staging/20260731/L1_m9/catalogues`.
- L2p8 uses the eight audited canonical catalogues below
  `/rds/rds-lxu/flamingo/L2p8_m9/lightcone*/catalogues`.
- Each catalogue is paired with its existing yang26-rotated Compton-y map.
- Selection uses `q_from_aperture > q_cut` for
  `q_cut = 50, 20, 10, 5, 1`.

## Invariant estimator pipeline

The q-from-map run must reuse the existing NaMaster implementation without
changing scientific settings:

- mask radius `max(4 * theta_500, 2 * FWHM)` with `FWHM = 10 arcmin`;
- `theta_500 = R_500c / D_A(z)` in the D3A cosmology;
- binary HEALPix discs at the catalogue's yang26-rotated coordinates;
- C2 apodization of 0.25 degrees;
- mask-weighted monopole subtraction;
- NaMaster MASTER decoupling with `nlb=1`, `lmax=10000`;
- no beam correction and HEALPix pixel-window deconvolution;
- the existing 18 inclusive Planck bins and 12 logarithmic bins with
  `Delta ln ell = 0.4`.

The maps, aperture radii, cosmology, rotation convention, binning, estimator,
and plotting styles are unchanged. Only the q column and catalogue paths
change.

## Implementation approach

Parameterize the existing L1 multi-q and L2 multi-lightcone scripts with a
selection mode whose default remains `qfrommz_alpha_fixed_1p12`. The new
`qfrommap` mode selects `q_from_aperture`, resolves the L1 staging directory
explicitly, and writes a distinct `qfrommap` tag. Parameterize the existing
L1-versus-L2 plotting script with the same tag rather than duplicating the
NaMaster or plotting implementation.

This approach is preferred over copying the pipelines because it guarantees
that both selections execute the same estimator code and preserves the old
default behavior for reproducibility.

## Outputs

- New masked bandpowers live beside the existing paper bandpowers and contain
  `_qfrommap_` in every filename.
- New metadata records the exact catalogue, map, q column, cut, masked count,
  sky fractions, estimator settings, and runtime.
- Existing full-sky bandpowers are reused because they do not depend on q.
- New comparison figures use the existing two-panel masked/full-sky layout
  and end in `_qfrommap`; existing figures are neither overwritten nor moved.
- The main comparison follows the established convention: L1 fiducial solid,
  L2p8 lightcone0 dashed, with both 18-bin and logarithmic-bin figures.

## Execution and validation

1. Unit-test selection-mode parsing, catalogue path resolution, q-column
   choice, and collision-free output names.
2. Run an L1 fiducial plus L2p8 lightcone0 pilot for all five cuts.
3. Before any NaMaster run, require every catalogue's yang26-rotated
   coordinates to trace its paired map's tSZ peaks more strongly than the
   natural coordinates. A failed rotation sanity check blocks that map.
4. Require catalogue counts to match direct threshold counts and require all
   bandpowers/sky fractions to be finite and physical.
5. Run the remaining eight L1 variants and seven L2p8 lightcones with at most
   two map workers and eight OpenMP threads per worker. Use two workers only
   when the serial pilot peaks below 30 GiB RSS and the host remains lightly
   loaded; otherwise use one worker.
6. Verify the complete expected output matrix before plotting.
7. Generate the two new comparison figures and visually inspect them.

Failures leave completed q-from-map products resumable and never alter any
q-from-mz product.
