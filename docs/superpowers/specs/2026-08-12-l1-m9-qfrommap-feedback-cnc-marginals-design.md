# L1_m9 q-from-map Feedback CNC Marginals

## Goal

Plot the paper-binned cluster number-count marginals for all nine L1_m9
feedback prescriptions using the aperture-derived detection significance.
Produce two separate figures: one for `N(q)` marginalized over redshift and
one for `N(z)` marginalized over detection significance.

## Inputs and selection

Read the canonical catalogues under
`/rds/rds-lxu/flamingo/L1_m9/catalogues` whose names end in
`_yang26rot_qfrommap.csv`. Use the `q_from_aperture` and `z` columns. Do not use
`data/nb40_l1_m9_feedback_cnc_qz.npz`, because that cache contains the older
mass-derived `q_from_mz` selection with `B = 1.35`.

Use all nine variants and their established order, labels, and colours from
`paper_results.config`:

- `fgas+2sigma`
- `L1_m9`
- `fgas-2sigma`
- `fgas-4sigma`
- `fgas-8sigma`
- `Mstar-1sigma`
- `Mstar-1sigma_fgas-4sigma`
- `Jet`
- `Jet_fgas-4sigma`

## Binning and marginals

Preserve the existing paper/CNC bounds and binning:

- 10 linear redshift bins over `0.005 <= z <= 1`;
- 5 logarithmic detection-significance bins over `5 <= q <= 40`.

For each feedback prescription, form one `10 x 5` `N(z, q)` histogram. Derive
both plotted series from that same histogram:

- `N(q)` is the sum over all 10 redshift bins;
- `N(z)` is the sum over all 5 detection-significance bins.

Objects outside either preserved bound do not contribute to either marginal.
No new persistent `N(z, q)` cache is required because the canonical catalogues
are already stored on disk and the histogram is inexpensive to reproduce.

## Figures

Create two separate step-histogram figures under `figures/feedback/`, each in
PNG and PDF form:

- `l1_m9_cnc_binned_Nq_qgt5_feedback_qfrommap.{png,pdf}`;
- `l1_m9_cnc_binned_Nz_qgt5_feedback_qfrommap.{png,pdf}`.

Overlay all nine feedback prescriptions in each figure. Use a logarithmic
horizontal axis for `N(q)` and a linear horizontal axis for `N(z)`. Use count
`N` on the vertical axes, no grid or diagnostic subtitle, and a compact legend
with the established feedback labels. The figures should follow the existing
paper plotting style and remain readable with nine curves.

## Implementation and verification

Add one focused plotting script that streams only `z` and
`q_from_aperture` from each CSV, constructs the common 2D histogram, derives
the marginals, and writes both figures. Keep bin construction, catalogue-path
resolution, histogram construction, and plotting callable independently so
tests can exercise them without reading the production catalogues.

Tests will verify:

1. all nine q-from-map catalogue paths and `q_from_aperture` are selected;
2. the exact redshift and detection-significance edges are used;
3. both marginals equal the corresponding sums of a known 2D histogram;
4. boundary and out-of-range objects are handled consistently;
5. both explicit q-from-map output stems are used;
6. the production script renders both PNG and PDF figures from the stored
   catalogues without missing inputs.
