# L2 q-from-map standalone plots design

## Goal

Regenerate the standalone L2p8_m9 fiducial masked tSZ power-spectrum figures from the empirical aperture-q bandpowers.

## Inputs and scope

- Use the canonical fiducial `lightcone0` products.
- Plot the existing standard thresholds: `q > 50, 20, 10, 5, 1`.
- Read `Dl_yy_L2p8_m9_lc0_masked_qgt*_qfrommap_*.txt` and the matching lightcone-0 metadata.
- Continue using the existing L2 full-sky spectra as the unmasked reference.
- Do not recompute NaMaster bandpowers.

## Outputs

- `figures/masked_ps/l2p8_m9_masked_ps_binned_18_qfrommap.{png,pdf}`
- `figures/masked_ps/l2p8_m9_masked_ps_logbins_qfrommap.{png,pdf}`

Legacy `q_from_mz` figures retain their current names and are not overwritten.

## Verification

- A path-level regression test checks that q-from-map plotting selects the lightcone-0 input and tagged output.
- Run the plot script for both binning schemes.
- Confirm all five threshold inputs are finite and both PNG files are non-empty.

