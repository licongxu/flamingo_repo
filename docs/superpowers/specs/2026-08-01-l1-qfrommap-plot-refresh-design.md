# L1 q-from-map Plot Refresh Design

## Goal

Regenerate every existing L1 figure whose masked-spectrum input has changed from
the legacy mass/redshift-derived selection to the empirical aperture selection
(`q_from_aperture`). Preserve every legacy figure and label each new output with
`qfrommap`.

## Scope

Refresh these q-dependent L1 figure families in both PNG and PDF:

- fiducial masked spectra at q cuts 50, 20, 10, 5, and 1, for the 18-bin and
  logarithmic bandpowers;
- feedback-variant full-sky versus masked spectra at q > 5 and q > 1, for both
  binnings;
- feedback-to-fiducial ratios across the existing full-sky, q > 20, q > 10,
  and q > 5 curves, including the single-model and all-feedback layouts;
- the main paper fiducial and feedback figures that consume the same masked
  spectra.

Do not refresh theory-only, rotation-group, CNC, null-test, pixel-histogram, or
Planck q > 6 figures because the new L1 production did not create corresponding
inputs for them.

## Interface and Data Flow

Each affected plotting entry point accepts an explicit selection option. The
legacy selection remains the default so existing commands and filenames remain
stable. Selecting `qfrommap` changes only:

1. the masked-bandpower filename tag;
2. the metadata source;
3. the explanatory plot text; and
4. the output filename suffix.

Masked inputs come from `data_paper/binned_bandpowers`; existing full-sky inputs
remain in their current locations. The q-from-map metadata comes from
`L1_m9_feedback_multi_q_bandpowers_qfrommap_metadata.json`. No bandpower is
recomputed.

## Validation

- Assert all expected q-from-map inputs exist before plotting.
- Retain the existing ell-grid consistency checks.
- Run focused path/selection tests followed by the plotting commands.
- Confirm all expected PNG/PDF pairs exist, are non-empty, and have the
  `qfrommap` suffix.
- Confirm legacy alpha-fixed figures are unchanged.

## Resource Use

Plotting loads small text arrays and runs serially. It does not read HEALPix maps
or invoke NaMaster, so its memory and CPU footprint is negligible relative to
the active L2 production.
