# L2 fiducial Y_5R500c backfill design

## Goal

Ensure the fiducial q-from-map catalogues expose the official FLAMINGO SOAP `Y_5R500c_Mpc2` property.

## Current state

- L1_m9 fiducial already contains `Y_5R500c_Mpc2`; it must remain unchanged.
- L2p8_m9 fiducial lightcone0 canonical q-from-map catalogue lacks the column.

## Data source and identity

For each L2 row, use `(snap, soap_index)` to read `SO/5xR_500_crit/ComptonY` from the matching official SOAP snapshot. Never join by CSV row number or sky position.

## Write safety

- Add exactly one column named `Y_5R500c_Mpc2` to the L2 canonical q-from-map CSV.
- Preserve every pre-existing CSV field text exactly and append the new value to each data row.
- Write to a temporary sibling, validate it, create a hard-link backup of the original, then replace the canonical pathname atomically.
- Do not modify positions, masses, redshifts, aperture fluxes, noise, or q.

## Verification

- Reject an input that already has the target column.
- Require finite, non-negative SOAP values for every row.
- Require output row count and identity digest to match the input.
- Require the prefix of every rewritten data row to equal the complete original data row.
- Reopen the final CSV and confirm `Y_5R500c_Mpc2` is present and finite.

