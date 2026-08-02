# L2 all-lightcone Y_5R500c backfill design

## Goal

Ensure every canonical L2p8_m9 lightcone q-from-map catalogue contains the official SOAP `Y_5R500c_Mpc2` column.

## Inventory

- lightcone0 and lightcone1 already contain the column and remain unchanged.
- lightcone2 through lightcone7 lack the column and require backfill.

## Implementation

Generalize the verified lc0 backfill utility with a required lightcone selector. The selected lightcone controls both the canonical CSV pathname and the official SOAP target key. Preserve the existing `(snap, soap_index)` join, exact old-row text verification, hard-link backup, and validated atomic replacement.

Run at most two lightcones concurrently. Each process has independent source, temporary, backup, and canonical paths; failure of one process cannot publish or roll back another.

## Verification

- Unit tests cover lightcone path/key selection and reject indices outside 0–7.
- Each newly published catalogue must preserve its complete old-row digest and identity digest.
- Final inventory must show `Y_5R500c_Mpc2` in all eight headers with finite non-negative samples.
- lightcone0 and lightcone1 checksums must remain unchanged.

