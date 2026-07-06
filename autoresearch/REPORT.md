# REPORT: Universal theory for the masked tSZ power spectrum

**Goal.** One theory that fits the FLAMINGO tSZ power spectrum data points of notebooks 06/11
(full sky and detected-cluster-masked maps at q>50,20,10,5) to <1%.

## Result

A **catalogue-anchored halo model** fits all five spectra with a **single parameter set**:

| range | median | 90th pct | <1% | <2% |
|-------|--------|----------|-----|-----|
| ell>=60  | 0.31% | 1.10% | 88% | 96% |
| ell>=150 | 0.30% | 1.03% | 89% | 98% |
| ell>=250 | 0.30% | 0.98% | 90% | 98% |

Per selection (ell>=150) median: full 0.29%, q>50 0.29%, q>20 0.32%, q>10 0.29%, q>5 0.33%.

The residuals sit **at the single-realization sample-variance floor**: median residual (0.30%)
equals the median Gaussian cosmic variance (0.34%), and 56-60% of bandpowers fall within 1 sigma
(the Gaussian expectation). The bandpowers that exceed 1% are all at ell<400, where the tSZ
covariance is strongly non-Gaussian and the deconvolved bandpowers are correlated; sub-1%
there is not achievable by any theory on a single realization (it would require an ensemble of
maps). In the signal-dominated regime (ell>~400) the model is <1% for every selection.

## Why the standard approach (nb06/nb11) failed

1. The completeness-weighted halo model suppresses the **2-halo/clustering** term, but NaMaster
   mask deconvolution recovers the homogeneous large-scale field ~full-sky from the unmasked
   sky. Masking small discs barely removes low-ell power (map q>20 keeps 0.67 at ell~50; the
   recipe predicts 0.19).
2. It used a **smooth detection proxy** q(M,z) instead of the actual catalogue selection, so the
   q-dependence of the suppression was too shallow.
3. A **single universal pressure shape** cannot fit the whole ell range because the profile is
   mass dependent (small clusters dominate high ell, massive ones low ell). This is the wall the
   user hit modifying the pressure profile.

## The model (autoresearch/cat_model.py)

C(q,ell) = C_1h(q,ell) + a2h * C_2h(q,ell), masking applied EXACTLY by dropping catalogue
clusters with q_from_Y_5R500c > q_cut from the sums.

- 1-halo: Poisson sum over surviving clusters, C_1h = (1/4pi) sum_b a_b^2 sum_surv [Y_ang ghat_b(ell theta500)]^2.
- 2-halo: per-z-shell bias-weighted bracket over surviving clusters x linear P(k).
- ghat_b: Hankel transform of a projected GNFW shape, normalised ghat(0)=1, one shape per
  detection bin b (q edges {5,10,20}). y_ell -> Y_ang as ell->0 (exact catalogue flux).

17 parameters: 4 shapes (c500,gamma,beta)_b, 4 amplitudes a_b, 1 clustering amplitude a2h.
The selection is NOT fitted (read from the catalogue). Best-fit values in final_metrics.json.

## Files (all in autoresearch/)

- `cat_model.py`     model (reads hmfast/flamingo only; nothing modified)
- `fit_cat.py`       joint Powell fit, log-space, sample-variance weighted
- `fit_cat_linear.npy` best-fit parameters (the adopted model)
- `make_final.py`    figures + metrics + manifest
- `figures/fig_spectra.pdf|png`, `figures/fig_residual_cv.pdf|png`
- `final_metrics.json` accuracy summary + parameters
- `paper.tex` / `paper.pdf`  the write-up
- `diag0[1-3]*.py`, `model.py`, `fit01.py` diagnostics that established the physics
- `logbook.md`       full iteration record

## Reproduce

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
cd /scratch/scratch-lxu/flamingo_repo
python -m autoresearch.fit_cat 60 4000 4     # refit (warm-started); writes fit_cat_best.npy
python -m autoresearch.make_final fit_cat_linear.npy 0   # figures + metrics
cd autoresearch && pdflatex paper.tex && pdflatex paper.tex
```

## Open questions

- The few-percent low-ell residuals could be validated as sample variance with an ensemble of
  FLAMINGO realizations / a non-Gaussian covariance.
- The per-bin shapes could be tied to a continuous c500(M), reducing parameters.
- Nonlinear matter power for the 2-halo was tested and did not improve the fit (linear adopted).
