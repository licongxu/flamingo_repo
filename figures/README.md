# Figures

Paper and analysis figures, grouped by topic. Regenerate with the scripts noted
in each subsection.

## `paper/`

Main paper figures from `python -m paper_results.figures`:

| File | Script |
|------|--------|
| `fiducial_masked_tsz_ps` | `--figure fiducial` |
| `feedback_tsz_ps` | `--figure feedback` |
| `fiducial_tsz_ps_vs_halo_model` | `--figure theory` |

## `rotation_groups/`

L1_m9 rotation-group shell decomposition from `python -m paper_results.rotation_groups`.

## `masked_ps/`

Masked tSZ bandpowers with alpha-fixed `q` catalogues (`B=1.12`):

| File prefix | Script |
|-------------|--------|
| `l1_m9_masked_ps_*` | `scripts/plot_l1_m9_masked_ps_alpha_fixed_1p12.py` |
| `l2p8_m9_masked_ps_*` | `scripts/plot_l2p8_m9_masked_ps_alpha_fixed_1p12.py` |
| `l1_l2p8_m9_masked_ps_*` | `scripts/plot_l1_l2p8_m9_masked_ps_comparison.py` |

## `feedback/`

L1_m9 feedback-variant bandpowers:

| File prefix | Script |
|-------------|--------|
| `l1_m9_feedback_ps_*_qgt1` | `scripts/plot_l1_m9_feedback_bandpowers_qgt1.py` |
| `l1_m9_feedback_ps_*` (q>5) | `scripts/plot_l1_m9_feedback_bandpowers.py` |

## `planck_comparison/`

| File | Script |
|------|--------|
| `flamingo_planck_tszps_qgt6` | `scripts/plot_flamingo_planck_tszps.py` |

## `masking_radius_null_test/`

Masking-radius sweep and random-position control:

| File | Script |
|------|--------|
| `masking_radius_null_test` | `scripts/plot_masking_radius_null_test.py` |
| `masking_radius_convergence` | same |
| `masking_radius_random_control` | same |

## `customgnfw/`

| File | Script |
|------|--------|
| `l1_m9_fullsky_bestfit_customgnfw_highell` | `scripts/plot_l1_m9_bestfit_customgnfw_highell.py` |
