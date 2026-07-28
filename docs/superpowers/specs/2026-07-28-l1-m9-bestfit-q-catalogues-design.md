# L1_m9 Best-Fit `q` Catalogues Design

## Goal

Create nine new L1_m9 halo catalogues whose `q_from_mz` values use the updated
custom-GNFW best-fit scaling relation:

- `A_SZ = -4.1095805`
- `alpha_SZ = 0.97447729`
- `B = 1.41`

The existing `_qfrommz.csv` catalogues are the fiducial `B = 1.35` products.
They must remain in place and must not be renamed, moved, overwritten, or
edited.

## Inputs and outputs

For every feedback variant in `paper_results.config.VARIANTS`, read:

```text
/rds/rds-lxu/flamingo/L1_m9/catalogues/halo_catalogue_M500c_5e13_zlt3_<variant>_yang26rot_qfrommz.csv
```

Write the corresponding new file:

```text
/rds/rds-lxu/flamingo/L1_m9/catalogues/halo_catalogue_M500c_5e13_zlt3_<variant>_yang26rot_qfrommz_bestfit.csv
```

The nine variants are `fgas+2sigma`, `L1_m9`, `fgas-2sigma`,
`fgas-4sigma`, `fgas-8sigma`, `Mstar-1sigma`,
`Mstar-1sigma_fgas-4sigma`, `Jet`, and `Jet_fgas-4sigma`.

## Calculation

Use the existing `flamingo.cnc.SZScaling` and hmfast parametric central
Compton-amplitude calculation. For each halo, calculate

```text
q_from_mz = y0(M_500c, z; A_SZ, alpha_SZ, B)
            * exp(deterministic_scatter(soap_index))
            / sigma_y0(theta_500(M_500c, z; B))
```

with:

- fixed D3A cosmology;
- physical catalogue masses in `M_sun`, not `M_sun/h`;
- `A_SZ = -4.1095805`;
- `alpha_SZ = 0.97447729`;
- `B = 1.41`;
- `sigma_lnY = 0.173`;
- scatter seed `20260630`;
- the existing Planck-like SZiFi `immf6` noise curve.

These are the same parametric `y0` conventions used by the fitted
`ParametricGNFWPressureProfile`. The custom-GNFW radial shape parameters do
not separately enter catalogue `q`: the matched-filter selection calculation
uses its fitted central amplitude and mass exponent.

## Streaming and file safety

Add one focused command-line script under `scripts/`. It streams each source
CSV in chunks so memory use is independent of catalogue size. For every
chunk, it preserves all source columns and their order and replaces only
`q_from_mz`.

Each output is first written to a temporary sibling file. The script renames
the temporary file to its final `_qfrommz_bestfit.csv` name only after the
entire catalogue succeeds. By default it refuses to overwrite an existing
final output; an explicit `--force` option permits regeneration. A failed run
must leave the fiducial input untouched and must not leave a partial file at
the final output path.

The output begins with provenance comments recording the source path and all
scaling-relation, scatter, noise, and cosmology settings.

## Verification

Automated tests cover output naming, parameter construction, preservation of
the source schema and non-`q` values, replacement of `q_from_mz`, and refusal
to overwrite without `--force`.

After generating all nine files:

1. Confirm that every expected source and output exists.
2. Confirm that each output has the same header and row count as its source.
3. Confirm that `soap_index` order is unchanged.
4. Confirm that all new `q_from_mz` values are finite and positive.
5. Recompute a deterministic sample directly with `SZScaling.q` and require
   agreement with the stored values to floating-point CSV precision.
6. Confirm that source file timestamps and sizes did not change during the
   run.

Report the nine output paths, their sizes and row counts, and the verification
result.
