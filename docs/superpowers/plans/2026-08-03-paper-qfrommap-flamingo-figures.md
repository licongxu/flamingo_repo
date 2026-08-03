# Paper `_qfrommap` FLAMINGO Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate and integrate three publication-quality FLAMINGO paper figures whose masked curves all use `_qfrommap` products.

**Architecture:** Keep data selection and plotting in the three existing figure scripts, adding small explicit helpers that make the publication inputs and output names testable. Generate PDF/PNG pairs in `flamingo_repo`, copy them under explicit `_qfrommap` names into the Overleaf checkout, then update and compile `main.tex`.

**Tech Stack:** Python 3, NumPy, Matplotlib, pytest, LaTeX/latexmk, PDF/PNG figure assets.

## Global Constraints

- Activate `/scratch/scratch-lxu/venv/cmbagent_env/bin/activate` before every command.
- All masked FLAMINGO curves must come from filenames containing `_qfrommap`.
- Full-sky FLAMINGO curves remain selection-independent.
- Use physical mass units, `M_sun`, if any mass labels are touched.
- Produce vector PDFs and 300 dpi PNG previews.
- Use serif/LaTeX typography, no figure-level title or diagnostic subtitle, no grid, and no ratio panel except in the dedicated feedback-ratio figure.
- Preserve unrelated working-tree and staged changes; use `git commit --only` with exact task paths.
- Paper checkout: `/scratch/scratch-lxu/tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8`.

---

### Task 1: Combined Fiducial Masking and Planck Comparison Figure

**Files:**
- Modify: `tests/test_l1_qfrommap_plots.py`
- Modify: `scripts/plot_flamingo_planck_tszps.py`
- Generate: `figures/planck_comparison/l1_m9_fiducial_masking_planck_qfrommap.pdf`
- Generate: `figures/planck_comparison/l1_m9_fiducial_masking_planck_qfrommap.png`

**Interfaces:**
- Consumes: L1_m9 logarithmic full-sky bandpowers; L1_m9 `qgt50`, `qgt20`, `qgt10`, `qgt5`, and `qgt6` `_qfrommap` bandpowers; `_qfrommap` metadata; Bolliet et al. (2018) and Rotti et al. observational files.
- Produces: `LEFT_CUTS`, `_bin18_path(..., selection_tag)`, `_logbin_path(..., selection_tag)`, `output_stem(selection_tag)`, and the two-panel PDF/PNG pair.

- [ ] **Step 1: Add failing tests for exact publication inputs and output name**

Append tests that load `plot_flamingo_planck_tszps.py` and assert the requested four left-panel cuts, L1_m9 `q>6` `_qfrommap` path, and explicit output stem:

```python
def test_combined_fiducial_planck_plot_uses_requested_qfrommap_inputs():
    module = _load_script(
        "combined_fiducial_planck_qfrommap_test",
        "plot_flamingo_planck_tszps.py",
    )

    assert module.LEFT_CUTS == (
        (50.0, "qgt50"),
        (20.0, "qgt20"),
        (10.0, "qgt10"),
        (5.0, "qgt5"),
    )
    assert module._bin18_path(
        "L1_m9", masked=True, selection_tag="qfrommap"
    ) == MASKED_DATA / "Dl_yy_L1_m9_masked_qgt6_qfrommap_binned_18.txt"
    assert module.output_stem("qfrommap").name == (
        "l1_m9_fiducial_masking_planck_qfrommap"
    )


def test_combined_fiducial_planck_plot_excludes_extra_comparisons():
    module = _load_script(
        "combined_fiducial_planck_scope_test",
        "plot_flamingo_planck_tszps.py",
    )

    assert not hasattr(module, "TANIMURA2021")
    assert not hasattr(module, "VARIANTS")
```

- [ ] **Step 2: Run the focused tests and verify the new assertions fail**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest tests/test_l1_qfrommap_plots.py::test_combined_fiducial_planck_plot_uses_requested_qfrommap_inputs tests/test_l1_qfrommap_plots.py::test_combined_fiducial_planck_plot_excludes_extra_comparisons -v
```

Expected: FAIL because `LEFT_CUTS`, the selection-aware path signatures, and `output_stem` do not yet exist, while `TANIMURA2021` and `VARIANTS` still exist.

- [ ] **Step 3: Implement the selection-aware data helpers**

In `plot_flamingo_planck_tszps.py`, set the publication constants and make the masked paths explicit:

```python
import json

SELECTION_TAG = "qfrommap"
LEFT_CUTS = (
    (50.0, "qgt50"),
    (20.0, "qgt20"),
    (10.0, "qgt10"),
    (5.0, "qgt5"),
)
META = DATA / "L1_m9_feedback_multi_q_bandpowers_qfrommap_metadata.json"


def _bin18_path(
    variant: str,
    *,
    masked: bool,
    selection_tag: str = SELECTION_TAG,
) -> Path:
    if masked:
        token = "" if variant == "L1_m9" else f"_{variant}"
        return DATA / f"Dl_yy_L1_m9{token}_masked_qgt6_{selection_tag}_binned_18.txt"
    return DATA / f"Dl_yy_{variant}_fullsky_binned_18.txt"


def _logbin_path(
    variant: str,
    *,
    masked: bool,
    selection_tag: str = SELECTION_TAG,
) -> Path:
    if masked:
        token = "" if variant == "L1_m9" else f"_{variant}"
        return DATA / (
            f"Dl_yy_L1_m9{token}_masked_qgt6_{selection_tag}_"
            "logbins_dln0p4_lmax10000.txt"
        )
    return DATA / (
        f"Dl_yy_{variant}_fiducial_fullsky_"
        "logbins_dln0p4_lmax10000_pixwin_deconvolved.txt"
    )


def output_stem(selection_tag: str = SELECTION_TAG) -> Path:
    return FIGURES / f"l1_m9_fiducial_masking_planck_{selection_tag}"
```

Remove `TANIMURA2021` and the L2p8 `VARIANTS` loop. Keep `_planck_bolliet`, `_planck_masked_marginalised`, and the hybrid L1_m9 loader.

- [ ] **Step 4: Implement the two-panel, spectra-only publication layout**

Use one `1 x 2` figure with no suptitle or axis title. The left panel reads the L1_m9 logarithmic full-sky spectrum plus the four `LEFT_CUTS`; the right panel draws only the L1_m9 full-sky/q>6 hybrid curves and the two observational datasets. Use paired redundant encodings:

```python
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.8, 4.8))

full_color = "#0072B2"
masked_color = "#D55E00"

# Right panel pair 1: full sky + Bolliet et al. (2018).
ax_right.loglog(ell_full, dl_full, color=full_color, lw=2.2, label="FLAMINGO full sky")
ax_right.errorbar(
    ell_b, dl_b, yerr=sig_b, fmt="o", mfc="white", mec=full_color,
    ecolor=full_color, capsize=2.5, ms=4.5, label="Bolliet et al. (2018)",
)

# Right panel pair 2: q>6 + Rotti et al. (2021).
ax_right.loglog(
    ell_masked, dl_masked, color=masked_color, ls="--", lw=2.2,
    label=r"FLAMINGO $q>6$",
)
ax_right.errorbar(
    ell_r, dl_r, yerr=sig_r, fmt="s", mfc="white", mec=masked_color,
    ecolor=masked_color, capsize=2.5, ms=4.5, label="Rotti et al. (2021)",
)
```

Apply logarithmic axes, `10 <= ell <= 10000`, the same `10^12 D_ell^{yy}` ylabel, `(a)`/`(b)` panel tags, legends without frames, no grids, and LaTeX/serif rcParams. Save PDF and PNG with `dpi=300` and `bbox_inches="tight"`.

- [ ] **Step 5: Run the focused tests and generate the figure**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest tests/test_l1_qfrommap_plots.py -v
python scripts/plot_flamingo_planck_tszps.py
```

Expected: all focused tests PASS and both files under `figures/planck_comparison/l1_m9_fiducial_masking_planck_qfrommap.*` are rewritten.

- [ ] **Step 6: Commit only Task 1 source and tests**

Run:

```bash
git commit --only scripts/plot_flamingo_planck_tszps.py tests/test_l1_qfrommap_plots.py -m "feat: add qfrommap paper comparison figure"
```

Expected: the commit contains only those two text files; generated figures and unrelated staged changes remain outside the commit.

---

### Task 2: Four-Panel Baryonic-Feedback Spectra

**Files:**
- Modify: `tests/test_l1_qfrommap_plots.py`
- Modify: `scripts/plot_l1_m9_feedback_bandpowers.py`
- Generate: `figures/feedback/l1_m9_feedback_ps_logbins_multiq_qfrommap.pdf`
- Generate: `figures/feedback/l1_m9_feedback_ps_logbins_multiq_qfrommap.png`

**Interfaces:**
- Consumes: full-sky feedback products plus per-variant `qgt20`, `qgt10`, and `qgt5` `_qfrommap` products.
- Produces: `PANEL_CUTS`, `_path(..., cut_tag)`, `output_stem(...)`, and the `2 x 2` spectra-only PDF/PNG pair.

- [ ] **Step 1: Add failing tests for the panel cuts and data routing**

Replace the q>5-only expectations with:

```python
def test_feedback_multiq_plot_uses_requested_qfrommap_panels():
    module = _load_script(
        "l1_feedback_multiq_qfrommap_plot_test",
        "plot_l1_m9_feedback_bandpowers.py",
    )

    assert module.PANEL_CUTS == (
        ("full sky", None),
        (r"$q>20$", "qgt20"),
        (r"$q>10$", "qgt10"),
        (r"$q>5$", "qgt5"),
    )
    assert module._path(
        "Jet", masked=True, log=True, selection_tag="qfrommap", cut_tag="qgt20"
    ) == MASKED_DATA / (
        "Dl_yy_L1_m9_Jet_masked_qgt20_qfrommap_"
        "logbins_dln0p4_lmax10000.txt"
    )
    assert module.output_stem(log=True, selection_tag="qfrommap").name == (
        "l1_m9_feedback_ps_logbins_multiq_qfrommap"
    )
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest tests/test_l1_qfrommap_plots.py::test_feedback_multiq_plot_uses_requested_qfrommap_panels -v
```

Expected: FAIL because `PANEL_CUTS` and `cut_tag` do not exist and the current output stem is q>5-only.

- [ ] **Step 3: Generalize the path helper and output stem**

Implement:

```python
PANEL_CUTS = (
    ("full sky", None),
    (r"$q>20$", "qgt20"),
    (r"$q>10$", "qgt10"),
    (r"$q>5$", "qgt5"),
)


def _path(
    variant: str,
    *,
    masked: bool,
    log: bool,
    selection_tag: str | None = None,
    cut_tag: str = "qgt5",
) -> Path:
    selection_tag = TAG if selection_tag is None else selection_tag
    suffix = "logbins_dln0p4_lmax10000" if log else "binned_18"
    if not masked:
        return DATA / f"Dl_yy_L1_m9_{variant}_fullsky_{suffix}.txt"
    root = MASKED_DATA if selection_tag == "qfrommap" else DATA
    token = "" if selection_tag == "qfrommap" and variant == "fiducial" else f"_{variant}"
    return root / f"Dl_yy_L1_m9{token}_masked_{cut_tag}_{selection_tag}_{suffix}.txt"


def output_stem(*, log: bool, selection_tag: str | None = None) -> Path:
    selection_tag = TAG if selection_tag is None else selection_tag
    bin_tag = "logbins" if log else "binned_18"
    if selection_tag == "qfrommap":
        return FIGURES / f"l1_m9_feedback_ps_{bin_tag}_multiq_qfrommap"
    return FIGURES / f"l1_m9_feedback_ps_{bin_tag}_alpha_fixed_1p12"
```

- [ ] **Step 4: Replace the ratio layout with four spectra panels**

Rewrite `_figure` to use four equally sized axes:

```python
fig, axes = plt.subplots(
    2,
    2,
    figsize=(11.0, 7.8),
    sharex=True,
    sharey=True,
    gridspec_kw={"hspace": 0.10, "wspace": 0.08},
)

for index, (ax, (panel_label, cut_tag)) in enumerate(zip(axes.ravel(), PANEL_CUTS)):
    masked = cut_tag is not None
    for variant, color in zip(VARIANTS, variant_colors):
        ell, dl = _load(
            _path(
                variant,
                masked=masked,
                log=log,
                selection_tag=selection_tag,
                cut_tag=cut_tag or "qgt5",
            )
        )
        keep = (ell >= ell_range[0]) & (ell <= ell_range[1])
        ax.loglog(
            ell[keep], dl[keep], color=color,
            lw=2.2 if variant == "fiducial" else 1.4,
            label=LABELS[variant],
        )
    ax.set_title(panel_label)
```

Use ylabels only in the left column, xlabels only in the bottom row, one shared frameless legend, no ratio axes, no suptitle, no grid, LaTeX/serif rcParams, and `_save(..., dpi=300)`.

- [ ] **Step 5: Run tests and generate both binning diagnostics**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest tests/test_l1_qfrommap_plots.py -v
python scripts/plot_l1_m9_feedback_bandpowers.py --selection qfrommap
```

Expected: tests PASS; the logarithmic publication PDF/PNG pair exists at the exact `multiq_qfrommap` stem; all four panels render without a lower ratio row or overall title.

- [ ] **Step 6: Commit only Task 2 source and tests**

Run:

```bash
git commit --only scripts/plot_l1_m9_feedback_bandpowers.py tests/test_l1_qfrommap_plots.py -m "feat: add multi-cut qfrommap feedback figure"
```

---

### Task 3: 18-Bin Feedback Ratios with Full-Sky and q>5 Theory Bands

**Files:**
- Modify: `tests/test_l1_qfrommap_plots.py`
- Modify: `scripts/plot_l1_m9_feedback_ratio_vs_q.py`
- Generate: `figures/feedback/l1_m9_all_feedback_ratio_vs_q_binned_18_qfrommap.pdf`
- Generate: `figures/feedback/l1_m9_all_feedback_ratio_vs_q_binned_18_qfrommap.png`

**Interfaces:**
- Consumes: 18-bin full-sky/q>20/q>10/q>5 feedback ratios and the full-sky plus q>5 theory covariance matrices.
- Produces: `selection_error_band_kinds(selection_tag)`, a title-free 18-bin `_qfrommap` ratio PDF/PNG, and both shaded theory bands.

- [ ] **Step 1: Add failing tests for q>5 covariance routing and bands**

Add:

```python
def test_feedback_ratio_qfrommap_uses_fullsky_and_q5_theory_bands():
    module = _load_script(
        "l1_feedback_ratio_qfrommap_bands_test",
        "plot_l1_m9_feedback_ratio_vs_q.py",
    )

    kinds = module.selection_error_band_kinds("qfrommap")
    assert kinds == (("fullsky", "full sky"), (5.0, r"$q>5$"))
    assert module.cov_path_for_band(5.0, log=False).name == (
        "cov_full_L1_m9_customgnfw_bestfit_masked_qgt5_Dl_yy_binned_18.npy"
    )
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest tests/test_l1_qfrommap_plots.py::test_feedback_ratio_qfrommap_uses_fullsky_and_q5_theory_bands -v
```

Expected: FAIL because `selection_error_band_kinds` does not exist.

- [ ] **Step 3: Preserve both bands for qfrommap and add title control**

Implement:

```python
def selection_error_band_kinds(
    selection_tag: str,
) -> tuple[tuple[str | float, str], ...]:
    return (("fullsky", "full sky"), (5.0, r"$q>5$"))
```

In `main`, set `ERROR_BAND_KINDS = selection_error_band_kinds(TAG)` for both selections instead of reducing the qfrommap case to full sky only.

Add `show_title: bool = True` to `plot_all_feedback_ratio_vs_q`. Wrap `fig.suptitle(...)` in `if show_title:` and use:

```python
layout_top = 0.97 if show_title else 0.99
fig.tight_layout(rect=(0.0, 0.06, 1.0, layout_top))
```

For the 18-bin qfrommap call, pass `show_title=False`. Keep the eight panel titles and bottom legend; remove only the overall title/subtitle.

- [ ] **Step 4: Run tests and regenerate the 18-bin ratio figure**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest tests/test_l1_qfrommap_plots.py -v
python scripts/plot_l1_m9_feedback_ratio_vs_q.py --selection qfrommap
```

Expected: tests PASS; the binned-18 PDF/PNG pair is rewritten; the legend contains both `full sky 1 sigma` and `q>5 1 sigma`; no overall title is present; nothing is cropped.

- [ ] **Step 5: Commit only Task 3 source and tests**

Run:

```bash
git commit --only scripts/plot_l1_m9_feedback_ratio_vs_q.py tests/test_l1_qfrommap_plots.py -m "fix: restore q5 theory band in qfrommap ratios"
```

---

### Task 4: Visual and Artifact Verification

**Files:**
- Verify: the six generated files from Tasks 1-3.

**Interfaces:**
- Consumes: generated PNG/PDF pairs.
- Produces: evidence that all assets are present, 300 dpi where rasterized, vector PDFs are valid, and layouts are publication quality.

- [ ] **Step 1: Verify exact artifacts and PDF metadata**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
test -s figures/planck_comparison/l1_m9_fiducial_masking_planck_qfrommap.pdf
test -s figures/planck_comparison/l1_m9_fiducial_masking_planck_qfrommap.png
test -s figures/feedback/l1_m9_feedback_ps_logbins_multiq_qfrommap.pdf
test -s figures/feedback/l1_m9_feedback_ps_logbins_multiq_qfrommap.png
test -s figures/feedback/l1_m9_all_feedback_ratio_vs_q_binned_18_qfrommap.pdf
test -s figures/feedback/l1_m9_all_feedback_ratio_vs_q_binned_18_qfrommap.png
pdfinfo figures/planck_comparison/l1_m9_fiducial_masking_planck_qfrommap.pdf
pdfinfo figures/feedback/l1_m9_feedback_ps_logbins_multiq_qfrommap.pdf
pdfinfo figures/feedback/l1_m9_all_feedback_ratio_vs_q_binned_18_qfrommap.pdf
```

Expected: every `test -s` succeeds and each `pdfinfo` call reports one valid page.

- [ ] **Step 2: Visually inspect all three PNGs**

Open the three generated PNGs and verify:

- no main title or diagnostic subtitle;
- no clipped axes, panel headings, or legends;
- fiducial figure has exactly two side-by-side spectra axes and no ratio row;
- feedback spectra have exactly four panels ordered full sky, q>20, q>10, q>5;
- feedback ratio figure has eight panels and visibly distinct full-sky and q>5 bands;
- colors, markers, and line styles remain readable at page width.

- [ ] **Step 3: Run the complete focused test module once more**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest tests/test_l1_qfrommap_plots.py -v
```

Expected: PASS with no skipped or failed tests.

---

### Task 5: Integrate the Figures into the Paper

**Files:**
- Create: `/scratch/scratch-lxu/tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8/figs/l1_m9_fiducial_masking_planck_qfrommap.pdf`
- Create: `/scratch/scratch-lxu/tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8/figs/l1_m9_fiducial_masking_planck_qfrommap.png`
- Create: `/scratch/scratch-lxu/tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8/figs/l1_m9_feedback_ps_logbins_multiq_qfrommap.pdf`
- Create: `/scratch/scratch-lxu/tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8/figs/l1_m9_feedback_ps_logbins_multiq_qfrommap.png`
- Create: `/scratch/scratch-lxu/tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8/figs/l1_m9_all_feedback_ratio_vs_q_binned_18_qfrommap.pdf`
- Create: `/scratch/scratch-lxu/tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8/figs/l1_m9_all_feedback_ratio_vs_q_binned_18_qfrommap.png`
- Modify: `/scratch/scratch-lxu/tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8/main.tex`

**Interfaces:**
- Consumes: verified repository figure assets.
- Produces: explicit `_qfrommap` paper assets and synchronized figure blocks/prose.

- [ ] **Step 1: Copy the six verified assets with explicit filenames**

Run the six explicit `cp` commands from the repository root:

```bash
cp figures/planck_comparison/l1_m9_fiducial_masking_planck_qfrommap.pdf /scratch/scratch-lxu/tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8/figs/
cp figures/planck_comparison/l1_m9_fiducial_masking_planck_qfrommap.png /scratch/scratch-lxu/tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8/figs/
cp figures/feedback/l1_m9_feedback_ps_logbins_multiq_qfrommap.pdf /scratch/scratch-lxu/tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8/figs/
cp figures/feedback/l1_m9_feedback_ps_logbins_multiq_qfrommap.png /scratch/scratch-lxu/tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8/figs/
cp figures/feedback/l1_m9_all_feedback_ratio_vs_q_binned_18_qfrommap.pdf /scratch/scratch-lxu/tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8/figs/
cp figures/feedback/l1_m9_all_feedback_ratio_vs_q_binned_18_qfrommap.png /scratch/scratch-lxu/tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8/figs/
```

- [ ] **Step 2: Replace the fiducial figure block with the combined figure**

Use this structure in `main.tex`:

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figs/l1_m9_fiducial_masking_planck_qfrommap.pdf}
    \caption{\label{fig:flamingo_fiducial_masked_ps}%
    Fiducial FLAMINGO L1\_m9 tSZ power spectra from the empirical
    map-based selection.
    \emph{Left:} $10^{12}D_\ell^{yy}$ for the full-sky map and after masking
    halos with $q>50$, $20$, $10$, and $5$, measured in twelve logarithmic
    multipole bins ($\Delta\ln\ell=0.4$, $\ell_{\rm max}=10^4$).
    The legend reports the number of masked halos and effective sky fraction.
    \emph{Right:} the FLAMINGO full-sky and $q>6$ spectra compared with the
    Planck tSZ measurements of \citet{Bolliet_2018} and \citet{Rotti_2021},
    respectively.}
\end{figure*}
```

Replace `Figure~\ref{fig:}` in the Planck-comparison paragraph with `Figure~\ref{fig:flamingo_fiducial_masked_ps}`. Keep the opening left-panel threshold list at `50, 20, 10, 5` and describe the `q=6` curve only in the observational-comparison sentence.

- [ ] **Step 3: Replace the feedback-spectrum figure block**

Use:

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figs/l1_m9_feedback_ps_logbins_multiq_qfrommap.pdf}
    \caption{\label{fig:flamingo_feedback_ps}%
    FLAMINGO L1\_m9 tSZ power spectra for the fiducial model and eight
    feedback variants, measured in twelve logarithmic multipole bins
    ($\Delta\ln\ell=0.4$, $\ell_{\rm max}=10^4$) using the empirical
    map-based selection.
    \emph{Top left:} full sky.
    \emph{Top right:} $q>20$.
    \emph{Bottom left:} $q>10$.
    \emph{Bottom right:} $q>5$.
    Feedback differences grow toward high $\ell$ and remain visible after
    progressive masking.}
\end{figure*}
```

- [ ] **Step 4: Replace the feedback-ratio filename and binning text**

Use:

```latex
\includegraphics[width=\textwidth]{figs/l1_m9_all_feedback_ratio_vs_q_binned_18_qfrommap.pdf}
```

Change the caption body to:

```latex
Ratio of each non-fiducial L1\_m9 feedback variant to the fiducial
spectrum under the same sky cut (full sky, $q>20$, $q>10$, and $q>5$),
measured in the 18 inclusive Planck-style multipole bins.
Shaded bands about unity are the full-sky and $q>5$ theory $1\sigma$
relative errors ($\sigma_D/D_\ell^{\mathrm{fid}}$).
At high $\ell$ the ratios for different $q$ cuts converge, indicating that
the fractional feedback response of the residual spectrum is approximately
independent of the catalogue threshold.
```

- [ ] **Step 5: Verify subsection figure provenance before compiling**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
sed -n '1450,1585p' /scratch/scratch-lxu/tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8/main.tex | rg 'includegraphics|qfrommap|alpha_fixed|Figure~\\ref\{fig:'
```

Expected: all three relevant `includegraphics` paths contain `_qfrommap`; no `alpha_fixed` figure remains in these two subsection figure blocks; the Planck paragraph has a valid figure label.

- [ ] **Step 6: Commit only the paper integration files**

From the paper checkout, run:

```bash
git add main.tex \
  figs/l1_m9_fiducial_masking_planck_qfrommap.pdf \
  figs/l1_m9_fiducial_masking_planck_qfrommap.png \
  figs/l1_m9_feedback_ps_logbins_multiq_qfrommap.pdf \
  figs/l1_m9_feedback_ps_logbins_multiq_qfrommap.png \
  figs/l1_m9_all_feedback_ratio_vs_q_binned_18_qfrommap.pdf \
  figs/l1_m9_all_feedback_ratio_vs_q_binned_18_qfrommap.png
git commit -m "Update FLAMINGO figures to qfrommap"
```

Expected: the paper commit contains `main.tex` and exactly the six new figure assets.

---

### Task 6: Compile and Audit the Paper

**Files:**
- Verify: `/scratch/scratch-lxu/tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8/main.pdf`

**Interfaces:**
- Consumes: updated `main.tex` and copied PDF figures.
- Produces: a successfully compiled paper and completion evidence for every figure requirement.

- [ ] **Step 1: Compile with halt-on-error behavior**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Working directory:

```text
/scratch/scratch-lxu/tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8
```

Expected: exit code 0 and a rewritten non-empty `main.pdf`.

- [ ] **Step 2: Check the LaTeX log for relevant failures**

Run:

```bash
rg -n 'LaTeX Error|Undefined control sequence|File .* not found|multiply defined|undefined references' main.log
```

Expected: no missing-figure errors, undefined control sequences, or unresolved `fig:flamingo_fiducial_masked_ps` references. Existing unrelated warnings may be reported separately but do not justify hiding a new failure.

- [ ] **Step 3: Render and inspect the pages containing the three figures**

Render the compiled paper pages with:

```bash
mkdir -p /tmp/flamingo-paper-pages
pdftoppm -png -r 150 main.pdf /tmp/flamingo-paper-pages/page
```

Inspect the rendered pages containing the three FLAMINGO figures and confirm:

- the combined fiducial figure spans both columns and has two axes only;
- the feedback spectra figure is a readable `2 x 2` grid with no ratio row;
- the feedback ratio figure uses the 18-bin data and both shaded bands;
- no main plot titles appear;
- captions match their panels and no figure is clipped.

- [ ] **Step 4: Run the final requirement-by-requirement audit**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest tests/test_l1_qfrommap_plots.py -v
rg -n 'l1_m9_(fiducial_masking_planck|feedback_ps|all_feedback_ratio).*qfrommap' main.tex
git status --short
```

Expected: focused tests PASS; `main.tex` references exactly the three explicit `_qfrommap` figures; status shows no accidental modifications beyond pre-existing user work and expected build artifacts.
