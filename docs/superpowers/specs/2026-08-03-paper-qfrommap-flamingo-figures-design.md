# Publication-Quality FLAMINGO `_qfrommap` Figures Design

## Objective

Update the figures associated with the paper subsections
`Progressive masking on fiducial FLAMINGO simulation` and
`Impact of feedback on the masked power spectrum` so that every masked
FLAMINGO curve is sourced from the empirical `_qfrommap` bandpowers and every
paper-facing figure is publication quality.

The paper checkout is
`/scratch/scratch-lxu/tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8`.
The figure-generation code and data products are in
`/scratch/scratch-lxu/flamingo_repo`.

## Scope

The update covers three paper figures:

1. A new two-panel, full-width fiducial-masking and observational-comparison
   figure.
2. The full-sky versus masked baryonic-feedback power-spectrum figure.
3. The feedback-to-fiducial ratio versus masking-threshold figure.

The unfinished Planck-comparison prose will be connected to the new combined
fiducial figure. Unrelated scientific results, tables, and figures remain
unchanged.

## Data provenance

All masked FLAMINGO curves must read filenames containing `_qfrommap`.
Full-sky FLAMINGO curves have no selection catalogue and therefore retain their
selection-independent full-sky filenames.

The observational comparison uses:

- Bolliet et al. (2018) marginalized Planck tSZ measurements from
  `ref_package/TopoSZ/fig12 data/planck_sz_1712_00788v1.txt`.
- Rotti et al. (2021) masked Planck tSZ measurements from
  `ref_package/TopoSZ/fig12 data/szpowespectrum_measurement_urc_snr6_p18cmb_bf_fg_from_TSZ+P18_l_clyy_sigclyy_cib_ir_rs_cn.txt`.

The right-hand FLAMINGO masked curve must use
`Dl_yy_L1_m9_masked_qgt6_qfrommap_*`. The feedback figures must use the
corresponding per-variant `qfrommap` products.

## Figure 1: fiducial masking and observations

Replace the existing single-column fiducial figure with one `figure*` spanning
the page. The generated PDF contains two side-by-side axes and no lower ratio
row.

### Left panel

- Show the fiducial L1_m9 full-sky spectrum in black.
- Show `_qfrommap` masked spectra for `q > 50`, `q > 20`, `q > 10`, and
  `q > 5`.
- Use the twelve logarithmic bins with `Delta ln ell = 0.4` and
  `ell_max = 10000`.
- Include `N` and effective `f_sky` values from the `_qfrommap` metadata in
  the legend.
- Do not show a masked/full ratio panel.

### Right panel

- Show the fiducial L1_m9 full-sky FLAMINGO curve and Bolliet et al. (2018)
  measurements as one visually associated pair.
- Show the fiducial L1_m9 `q > 6` `_qfrommap` curve and Rotti et al. (2021)
  measurements as a second visually associated pair.
- Do not include L2p8_m9 or Tanimura et al. data.
- Use distinct line/marker forms in addition to color so the comparison
  remains legible in grayscale.
- Use the existing hybrid FLAMINGO curve convention: the 18 Planck-style bins
  through `ell = 959.5`, followed by logarithmic bins at higher multipoles.

The output stem is
`figures/planck_comparison/l1_m9_fiducial_masking_planck_qfrommap`, and both
PDF and PNG outputs are produced.

## Figure 2: baryonic-feedback spectra

Regenerate the feedback comparison using
`scripts/plot_l1_m9_feedback_bandpowers.py --selection qfrommap` as a
spectra-only `2 x 2` figure.

- Top left: full-sky spectra.
- Top right: spectra after masking clusters with `q > 20`.
- Bottom left: spectra after masking clusters with `q > 10`.
- Bottom right: spectra after masking clusters with `q > 5`.
- Show the fiducial model and all eight feedback variants in every panel.
- Use `_qfrommap` products for every masked curve.
- Do not include variant/fiducial ratio panels.
- Do not show theory-covariance error bars that are not defined for the
  `_qfrommap` selections.
- Remove diagnostic selection descriptions and the figure-level title; retain
  only the four short panel headings needed to identify each sky cut.
- Write the explicitly tagged output
  `l1_m9_feedback_ps_logbins_multiq_qfrommap.{pdf,png}`.

## Figure 3: feedback response versus masking threshold

Regenerate the eight-panel feedback-ratio figure using
`scripts/plot_l1_m9_feedback_ratio_vs_q.py --selection qfrommap`.

- Use the 18 inclusive Planck-style multipole bins, not the logarithmic bins.
- Show full sky, `q > 20`, `q > 10`, and `q > 5` curves.
- Use `_qfrommap` products for all masked curves.
- Show both custom-GNFW relative-error bands about unity: the full-sky band
  and the `q > 5` band.
- Load the `q > 5` band from the previously computed theory covariance,
  `data_paper/covariance/cov_full_L1_m9_customgnfw_bestfit_masked_qgt5_Dl_yy_binned_18.npy`.
- Remove the diagnostic figure-level title and subtitle.
- Ensure the top row, panel titles, axes, and bottom legend are not cropped.
- Write
  `l1_m9_all_feedback_ratio_vs_q_binned_18_qfrommap.{pdf,png}`.

## Publication styling

All three figures must:

- provide a vector PDF as the manuscript asset and a 300 dpi PNG for visual
  inspection;
- use serif/LaTeX typography consistent with the manuscript;
- use consistent `10^12 D_ell^{yy}` notation and logarithmic multipole axes;
- omit diagnostic suptitles, data-path annotations, and grid lines;
- use readable line widths, marker sizes, legends, and panel labels at final
  printed size;
- use colorblind-distinguishable colors with line styles or marker shapes as
  redundant encodings;
- avoid clipped labels, legends, and panel titles.

## Manuscript changes

Update `main.tex` as follows:

- Replace the current fiducial single-column figure with a `figure*` that
  includes `figs/l1_m9_fiducial_masking_planck_qfrommap.pdf` at
  `width=\textwidth`.
- Rewrite its caption with explicit `Left:` and `Right:` descriptions and the
  correct `_qfrommap` thresholds.
- Point the previously broken `Figure~\ref{fig:}` reference to
  `fig:flamingo_fiducial_masked_ps`, because the observational comparison is
  now the right panel of that figure.
- Update the opening threshold list in the fiducial subsection to match the
  left panel (`50, 20, 10, 5`) and mention the separate `q = 6` comparison in
  the right panel.
- Replace the two feedback figure filenames with their explicit `_qfrommap`
  filenames.
- Rewrite the feedback-spectrum caption around the `2 x 2` panel ordering,
  remove its ratio-panel description, and remove its theory-covariance
  error-bar claim.
- Retain and verify the feedback-ratio caption statement that the shaded
  relative-error bands are for full sky and `q > 5`.

Copy the three generated PDF and PNG pairs into the paper `figs/` directory
without deleting or overwriting legacy non-`qfrommap` assets.

## Verification

1. Add focused tests proving the combined figure selects the L1_m9
   `qgt6_qfrommap` path, excludes L2p8/Tanimura, and uses the four requested
   left-panel thresholds.
2. Test that publication outputs retain explicit `_qfrommap` stems and that
   the feedback plots select `_qfrommap` data.
3. Run the focused plotting tests before and after implementation.
4. Regenerate all three PDF/PNG pairs in the repository virtual environment.
5. Inspect the PNGs visually at final-size scale for clipping, legibility,
   consistent styling, and correct curve/data association.
6. Search the two subsection region in `main.tex` to confirm that every
   FLAMINGO figure reference uses a `_qfrommap` filename.
7. Compile the paper with the available LaTeX build command and confirm that
   the new full-width figure and both feedback figures render without missing
   assets or LaTeX errors.
