# q=3/q=6 q-from-map and L1 Null-Test Design

## Goal

Complete the empirical-aperture selection products omitted from the original
production by measuring q > 3 and q > 6 bandpowers and rerunning the L1
fiducial masking-radius null test with the corrected q-from-map catalogue.

## Products

- L1: q > 3 and q > 6 bandpowers for all nine feedback prescriptions, in both
  standard binnings.
- L2p8: q > 6 bandpowers for lightcone0, for the Planck-comparison figure.
- L1 fiducial null test: q > 20, 10, 5, and 1 radius sweeps using
  `q_from_aperture`, corrected rotated positions, and the existing radius grid.
- Updated q-dependent figures after all required products exist.

## Data and Naming

All new products use the corrected q-from-map catalogues selected by
`resolve_q_selection("qfrommap")`, read `q_from_aperture`, and use
`theta_rot_rad`/`phi_rot_rad`. Bandpowers retain the `qfrommap` filename tag.
Null-test caches live under `data_paper/masking_radius_null_test/qfrommap` so no
legacy q-from-mass/redshift cache can be reused or overwritten.

## Resource Schedule

Run the nine L1 maps with eight map workers and limited OpenMP threads. Run only
one null-test q cut concurrently, then continue the remaining cuts serially.
Compute the L2 lightcone0 q > 6 product after a worker slot is available. Keep
aggregate resident memory below approximately half of the 376 GiB host.

## Validation

- Dry-run every input/output path before starting.
- Require the existing rotation sanity test to favor rotated coordinates.
- Check all new files are non-empty and have the expected selection tag.
- Run focused unit tests for legacy defaults and q-from-map paths.
- Monitor tmux sessions, error logs, worker RSS, and output counts.
