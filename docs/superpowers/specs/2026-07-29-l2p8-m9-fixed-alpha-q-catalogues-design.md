# L2p8_m9 Fixed-Alpha `q` Catalogues Design

## Goal

Create one new best-fit `q_from_mz` catalogue for each of the eight L2p8_m9
lightcones using:

- chain: `chains/L2p8_m9_customgnfw_asz_alpha_fixed_1p12_B1p41/best_fit.json`;
- `A_SZ = -4.0834373` (best fit);
- `alpha_SZ = 1.12` (fixed);
- `B = 1.41` (fixed);
- D3A cosmology.

Each output ends in `_qfrommz_alpha_fixed_1p12.csv`.

## Inputs and outputs

For every `lightconeN`, where `N` is zero through seven, read:

```text
/rds/rds-lxu/flamingo/L2p8_m9/lightconeN/catalogues/halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_qfrommz.csv
```

Write the sibling:

```text
/rds/rds-lxu/flamingo/L2p8_m9/lightconeN/catalogues/halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_qfrommz_alpha_fixed_1p12.csv
```

Preserve each source's header, row order, and every non-`q_from_mz` value.
Lightcone1 has 30 columns while the other lightcones currently have 26; each
output retains its own source schema. Existing catalogues must not be renamed,
moved, overwritten, or edited.

## Calculation

Reuse `flamingo.cnc.SZScaling` and the existing tested streamed writer.
Calculate `q_from_mz` from physical `M_500c_Msun`, redshift, and `soap_index`
using the parameters above, `sigma_lnY = 0.173`, scatter seed `20260630`, and
the Planck-like SZiFi `immf6` noise curve.

Bind JAX/hmfast work to GPU 1. Write each output through a unique temporary
sibling and atomically install it only after the complete lightcone succeeds.
Refuse a pre-existing destination.

## Verification

For all eight source/output pairs:

1. require equal headers and row counts;
2. require identical `soap_index` order and identical non-`q_from_mz` values
   using round-trip float parsing;
3. require all new `q_from_mz` values to be finite and positive;
4. confirm source size and nanosecond modification time are unchanged;
5. recompute the first, middle, and final rows directly with `SZScaling.q`;
6. verify provenance comments contain the exact fit and selection parameters;
7. confirm exactly eight outputs exist and no temporary output remains;
8. run the full repository test suite.
