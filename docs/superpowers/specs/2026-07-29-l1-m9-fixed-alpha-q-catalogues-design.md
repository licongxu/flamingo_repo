# L1_m9 Fixed-Alpha `q` Catalogues Design

## Goal

Create nine new L1_m9 halo catalogues using the fixed-alpha custom-GNFW chain:

- chain: `chains/l1_m9_customgnfw_asz_alpha_fixed_1p12/best_fit.json`;
- `A_SZ = -4.0953238` (best fit);
- `alpha_SZ = 1.12` (fixed);
- `B = 1.41` (fixed).

Each output ends in `_qfrommz_alpha_fixed_1p12.csv`.

## Inputs and preservation

For each variant in `paper_results.config.VARIANTS`, read the original
`*_qfrommz.csv` catalogue under
`/rds/rds-lxu/flamingo/L1_m9/catalogues/`. Preserve its header, row order, and
every non-`q_from_mz` value. Do not modify the original catalogues or the
existing `_qfrommz_bestfit.csv` catalogues.

## Calculation

Reuse `flamingo.cnc.SZScaling` and the existing streamed catalogue writer.
Calculate `q_from_mz` from physical `M_500c_Msun`, redshift, and `soap_index`
using the parameters above, D3A cosmology, `sigma_lnY = 0.173`, scatter seed
`20260630`, and the Planck-like SZiFi `immf6` noise curve.

Write each result to a unique temporary sibling and atomically install the
final output only after the complete calculation succeeds. Refuse pre-existing
destinations rather than overwriting them.

## Verification

For all nine source/output pairs:

1. require equal headers and row counts;
2. require identical `soap_index` order and identical non-`q_from_mz` values
   with round-trip float parsing;
3. require all new `q_from_mz` values to be finite and positive;
4. confirm source size and modification time are unchanged;
5. directly recompute first, middle, and last rows with `SZScaling.q`;
6. verify provenance comments contain the exact fit and selection parameters;
7. confirm no temporary output remains.
