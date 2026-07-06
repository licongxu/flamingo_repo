# Joint Cluster Number Counts + Masked tSZ Power Spectrum on FLAMINGO: Project Plan

> **For agentic workers:** execute this plan phase by phase with `superpowers:executing-plans`
> (or subagent-driven execution), under the session contract of
> `.claude/rules/autonomous-operation.md`. All new work goes in a fresh run folder
> `autoresearch/fable_5_v1/` (this folder, `fable_5_plan/`, holds only the plan and seed
> diagnostics). Checkboxes (`- [ ]`) track progress; append one logbook block per iteration.

**Goal:** A publication-quality paper demonstrating, with the FLAMINGO simulations, that a
joint analysis of SZ cluster number counts (CNC) and the tSZ power spectrum with detected
clusters masked yields tighter and more robust cosmological constraints than either probe
alone, with a validated joint covariance (including the CNC x PS cross term) and an explicit
optimal masking threshold.

**Architecture:** Reuse the existing measurement pipeline (SOAP catalogues, NSIDE 4096 y-maps,
NaMaster bandpowers, cosmocnc-style Poisson counts likelihood, cobaya). Add three new pieces:
(i) an analytic joint covariance with a masked 1-halo trispectrum, a CNC x PS cross block, and
super-sample terms, validated against FLAMINGO lightcones; (ii) an inference-grade parametric
theory for the masked tSZ PS, calibrated against the catalogue-anchored model from
`autoresearch/opus_4.8_v1/`; (iii) a joint likelihood and an inference grid over masking
thresholds q_cut in {none, 50, 20, 10, 5}.

**Tech stack:** Python 3.12 in `/scratch/scratch-lxu/venv/cmbagent_env`; `flamingo` (this repo),
`hmfast` (JAX halo model, has `trispectrum_1h` / `trispectrum_1h_masked`), `pymaster`,
`cobaya` + GetDist, SLURM for anything > 10 min.

## Global constraints

- Follow `.claude/rules/autonomous-operation.md`: logbook every iteration, manifest per job,
  mandatory stop conditions, non-destructive defaults.
- Follow `.claude/rules/cluster-usage.md`: GPU first for JAX paths, memory arithmetic before
  > 4 GiB allocations, SLURM for jobs > 10 min, no big outputs inside the repo.
- Follow `.claude/rules/paper-writing.md` for all `.tex`: no em or en dashes, VLM figure loop,
  quantitative claims only.
- Never edit `autoresearch/opus_4.8_v1/` or any previous run folder. New outputs go to
  `autoresearch/fable_5_v1/` (small artifacts) and `/scratch/scratch-lxu/fable_5_v1_data/`
  (large arrays, chains; gitignored).
- Fiducial data: L2p8_m9 lightcone 0, yang26-rotated frame, catalogue
  `halo_catalogue_M500c_5e13_zlt3_y0q_arnaudB1_Y500c.csv` (B = 1 SNR, Y_R500c aperture),
  q-cuts {5, 10, 20, 50}, mask discs of 5 x theta_500, C1 apodization 0.5 deg,
  Delta_ell = 30, ell_max = 6000. Do not silently change these.
- The truth cosmology is FLAMINGO D3A; every posterior figure must show the D3A input value.

---

## 1. Scientific case

### 1.1 The idea

CNC and the tSZ PS constrain nearly the same combination of sigma_8 and Omega_m, but they are
usually analysed either separately or combined assuming independence. Both assumptions are
wrong in an interesting, exploitable way:

1. The full-sky tSZ PS is dominated at all ell by a modest number of massive, low-z clusters.
   Its covariance is therefore strongly non-Gaussian (1-halo trispectrum), and it is strongly
   correlated with the counts of exactly those clusters.
2. Masking the detected (q > q_cut) clusters removes that shared population from the PS. The
   counts keep the detected clusters' information (their number, in bins of SNR and z); the
   masked PS keeps the information of the undetected population (low-mass, high-z halos and
   diffuse gas). The two probes become nearly disjoint in halo mass, so (a) the cross
   covariance is suppressed, (b) the PS covariance becomes nearly Gaussian, and (c) the joint
   likelihood factorizes to a good approximation, which is exactly the regime in which naive
   probe combination is valid and cheap.
3. The two probes have different degeneracy directions in (scaling relation amplitude / mass
   bias B) vs (sigma_8, Omega_m), and different sensitivity to baryonic feedback, so the
   combination also breaks nuisance degeneracies.

FLAMINGO is the ideal testbed: we control the truth (D3A cosmology, hydro gas physics), we
have the full halo catalogue (so selection and completeness are exact), 8 lightcone observers,
and 9 L1_m9 feedback variants.

### 1.2 Headline claims to establish (each must end with a number)

- C1 (covariance): masking at q > 5 reduces the low-ell (ell ~ 100 to 300) bandpower
  realization variance by a factor ~ N (measure it; seed estimate below says ~ 20 to 100 in
  variance) and brings it within a factor ~ 2 of the Gaussian Knox floor; the CNC x PS
  correlation coefficient drops from r_max ~ (measure) to |r| < (measure) after masking.
- C2 (constraints): the joint CNC + masked-PS analysis improves sigma(S8) by X% over CNC
  alone and Y% over the naive CNC + full-sky PS combination at fixed systematics model;
  report the (Omega_m, sigma_8, B) figure of merit as a function of q_cut and identify the
  optimum.
- C3 (validity): ignoring the cross covariance biases the joint posterior by Z sigma for the
  full-sky combination but < 0.1 sigma (target) for the masked combination; this legitimizes
  the simple product likelihood used in practice, but only after masking.
- C4 (robustness): the masked PS + CNC combination shifts by < W sigma across the 9 L1_m9
  feedback variants, vs V sigma for the full-sky PS combination (or the reverse; whatever the
  data say, quantified).

### 1.3 Novelty and positioning (verify in Phase 1)

Joint CNC + tSZ PS analyses exist analytically and on Planck data (candidate anchors, to be
verified before citing: Hurier & Lacasa 2017 for the joint likelihood with cross covariance;
Salvati, Douspis & Aghanim 2018 for Planck counts + PS; Bolliet et al. 2018 for the tSZ PS
likelihood; Osato & Takada 2021 and Horowitz & Seljak 2017 for tSZ SSC/trispectrum and
peak-masked variance suppression). What does not exist, to our knowledge: an end-to-end
validation on a hydrodynamical full-sky simulation, an explicit masked cross covariance, and
an optimization of the masking threshold for the joint figure of merit. That gap is the paper.

---

## 2. What already exists (do not rebuild)

| Asset | Where | Status |
|---|---|---|
| y-maps NSIDE 4096, lc0-7 + 9 L1_m9 feedback variants | `/rds/.../L2p8_m9/`, `/rds/.../L1_m9/` | done |
| Catalogues (B=1 / B=1.35) x (Y_5R500c / Y_R500c), M > 5e13, z < 3 | `data/` + RDS | done |
| Masked bandpowers, 5 cuts: lc0 fiducial, 8 lightcones, 9 feedback variants | `data/bandpowers_*` | done |
| Counts: (q, z) binned lc0 (5 x 10), (M, z) binned lc0-7 (12 x 15) | `data/cnc/` | done |
| PS covariance, Gaussian + 1h trispectrum, 18 bins, 5 cuts | `data/theory_cov_arnaudB1/` | done, no SSC / no cross |
| CNC likelihood (Poisson, cosmocnc-style) + YY Gaussian ell-cut likelihood | `cobaya/likelihood/` | done |
| CNC, PS, and combined chains (independent likelihoods) | `chains/` (nb27, nb28-30) | done, no cross-cov |
| Catalogue-anchored masked-PS model, < 1% for all 5 selections | `autoresearch/opus_4.8_v1/` | done (17-param, not inference-grade) |
| hmfast: HaloModel, `trispectrum_1h`, `trispectrum_1h_masked`, emulators | external package | done |

Known issues to inherit consciously:
- The completeness-weighted parametric masked-PS theory (nb06) fails at low ell: NaMaster
  deconvolution restores the near-full-sky 2-halo clustering that the naive completeness
  recipe suppresses, and a smooth q(M, z) proxy is too shallow versus the real selection
  (documented in `autoresearch/opus_4.8_v1/REPORT.md`). Phase 4 must fix or fence this.
- CNC inference with a Tinker08 HMF biases S8 low on FLAMINGO (nb27 finding); the paper is a
  consistency test against D3A truth, so the HMF must be FLAMINGO-consistent (Phase 4).
- The 8 lightcones share one 2.8 Gpc box: they are not independent realizations. Any
  covariance measured across them is a lower bound on true cosmic variance and their common
  mode inflates cross-correlations. Empirical covariance therefore needs patch resampling on
  a single map plus analytic theory, with the 8 observers as a consistency check only.

---

## 3. Seed analysis (done, this folder)

`seed01_cnc_ps_cross_correlation.py` (results: `seed01_results.npz`, `seed01_fig.png|pdf`,
`seed01_summary.json`; all from cached products, git 6265b89):

- Variance suppression is dramatic and robust: fractional bandpower scatter across the 8
  observers at ell ~ 100 falls from 0.26 (full sky) to 0.014 (q > 5 masked), i.e. a factor
  ~ 18 in sigma, ~ 340 in variance, landing within a factor ~ 1.4 of the full-sky Knox floor
  (0.010). At ell ~ 450: 0.051 -> 0.008 (Knox 0.003). This is claim C1's variance half,
  already visible with 8 realizations.
- The counts x bandpower Pearson r across the 8 observers is inconclusive (N = 8, shared box,
  null band +/- 0.38): full-sky r ~ -0.5 to 0 (unphysical sign, sample noise or observer
  common mode), masked r ~ +0.4 to +0.8 (plausibly the shared-box common mode surviving in
  the low-variance masked spectra). Lesson: the cross-covariance claim cannot be established
  from the 8 observers; it needs the analytic model + patch resampling of Phase 3. Also the
  seed used M > 2e14 full-z counts as a crude "detected" proxy; Phase 2 rebuilds per-observer
  detected (q, z) counts.

---

## 4. Phases

Effort estimates assume the autonomous session cadence of the opus_4.8_v1 run. Anything
> 10 min of compute goes through SLURM with a manifest.

### Phase 0: Run scaffolding (0.5 day)

**Files:** create `autoresearch/fable_5_v1/{logbook.md,manifest.json,REPORT.md(stub)}`,
`/scratch/scratch-lxu/fable_5_v1_data/` for large outputs.

- [ ] Create the run folder and `manifest.json` with: git hash, hmfast hash, venv path,
      fiducial data-vector definition (Section "Global constraints" verbatim), compute budget
      (proposal: 40 GPU-hours, 300 CPU-core-hours, hard stop per autonomous-operation rule 3),
      and the D3A truth parameters.
- [ ] First logbook entry: link to this plan, seed01 artifacts.
- verify: `manifest.json` parses; logbook exists; nothing written outside the two new dirs.

### Phase 1: Literature grounding and positioning (1 day)

- [ ] Run the `deep-research` skill on: "joint cluster number counts and tSZ power spectrum
      cosmology; cross-covariance; masked / cluster-subtracted tSZ power spectrum; tSZ
      trispectrum and super-sample covariance; FLAMINGO SZ analyses". Deliverable:
      `fable_5_v1/lit_review.md` with verified BibTeX for every anchor in Section 1.3.
- [ ] Write the novelty paragraph (5 sentences max) stating exactly what is new relative to
      the three closest works, each with a citation. If a prior paper already did the
      simulation-validated masked joint analysis, STOP and reassess scope with the user.
- verify: every citation resolves on ADS/arXiv (no invented references); novelty paragraph
  names the closest prior work explicitly.

### Phase 2: Fiducial joint data vector (1-2 days)

The joint data vector is d = {N_ij (q_i x z_j counts), D_b (masked bandpowers, ell > ell_min)}
per q_cut. Counts always use the full detected sample (q > 5); the PS mask threshold varies.

- [ ] **Per-observer detected counts.** Write `fable_5_v1/build_multilc_qz_counts.py`: for
      each lc 0-7 load the per-lightcone catalogue (q_from_mz columns exist for all 8;
      Y-based q exists only for lc0, record this limitation), bin into the nb09 scheme
      (q edges {5, 10, 20, 50, inf} x 10 z bins to z = 1). Output
      `fable_5_v1/data/N2d_qz_lc{0..7}.npz`.
      verify: lc0 counts reproduce `data/cnc/N2d_z_q_bin_arnaudB1_Y500c.txt` totals when the
      same q definition is used (exact match required for the Y-based path; document the
      q_from_mz vs q_from_Y difference for the others: lc0 q>5 totals 2023 (Y-based) vs 1155
      (mz-based) are both already on disk, so the mapping must be stated in the paper).
- [ ] **Bandpower ell-range decision.** From `data/theory_cov_arnaudB1/` and opus_4.8_v1
      residuals, fix ell_min per theory validity (expected ell_min ~ 150; Phase 4 confirms)
      and rebin the Delta_ell = 30 bandpowers into the analysis bins (reuse the 18-bin
      scheme unless Phase 4 dictates otherwise). Script writes the frozen data vector with a
      content hash: `fable_5_v1/data/datavector_qcut{5,10,20,50,none}.npz`.
- verify: a single loader `fable_5_v1/datavector.py::load(qcut)` returns (N_ij, D_b, ell_b,
  meta) and a pytest-style check asserts shapes, finiteness, and the content hash.

### Phase 3: Joint covariance (the methodological core, 3-5 days)

Blocks of Cov(d, d), all with the selection applied exactly as in the data:

1. Counts x counts: Cov(N_i, N_j) = delta_ij N_i (Poisson) + S_ij (SSC:
   S_ij = integral dV b_i b_j sigma^2_b(V) with b_i the mean halo bias of bin i;
   implement with hmfast bias + the lightcone volume window per z bin).
2. PS x PS: Knox Gaussian (with f_sky,eff of the apodized mask) + masked 1-halo trispectrum
   (`hm.trispectrum_1h_masked`, extend `scripts/compute_arnaudB1_full_cov.py`) + PS SSC
   (response dC_ell/d delta_b times sigma^2_b; the response is dominated by the 1-halo term
   of the surviving halos, computable in the same halo-model pass).
3. Counts x PS (new): 1-halo/Poisson overlap term
   Cov(N_i, C_ell) = integral dz dV/dz integral dM dn/dMdz chi_i(M, z) phi_cut(M, z)
   |y_ell(M, z)|^2, where chi_i is the (soft, scatter-convolved) membership of bin i and
   phi_cut is the survival probability under the PS mask (1 - detection completeness at
   q_cut); note chi_i phi_cut ~ 0 when the PS mask removes the detected bin i population,
   which is the analytic statement of claim C1. Plus the SSC cross term
   b_i N_i x (dC_ell/d delta_b) sigma^2_b, which survives masking and sets the residual floor
   of r. Follow the Hurier & Lacasa (2017) structure (verify exact expressions in Phase 1).

- [ ] **Implement** `fable_5_v1/joint_covariance.py` (JAX where it touches hmfast; GPU; the
      (M, z) grids match `compute_arnaudB1_full_cov.py`, memory estimate before launch:
      the existing script's grids fit in < 4 GiB, the cross block adds one (n_bin_N x n_ell)
      contraction, negligible). Unit tests in the run folder: shapes, finiteness,
      positive-semidefiniteness of the assembled matrix, Poisson limit (cross block -> 0 as
      q_cut -> detection threshold from below with zero scatter).
- [ ] **Empirical validation, patch resampling.** `fable_5_v1/patch_covariance.py`: split the
      lc0 sky into N_p = 48 or 192 HEALPix superpixels (nside 2 or 4); per patch compute
      counts and a local pseudo-D_ell for ell > ~ 300 (patch size limits low ell; state
      this), or alternatively delete-one-patch jackknife with the full NaMaster workspace
      (more expensive: one decoupling per patch; if > 10 min total, SLURM). Deliverable:
      empirical corr(N, D_b) and Var(D_b) vs the analytic blocks, full sky and q > 5.
      SLURM estimate: 192 NaMaster runs x ~ 1 min at NSIDE 4096 ~ 3.5 h, `--mem` per the
      existing bandpower jobs, `--time=06:00:00`, CPU partition.
- [ ] **Consistency check** against the 8-observer scatter (seed01) and, if useful, the
      rotation-group / shell maps (`scripts/build_rotation_group_maps.py`) for extra
      pseudo-realizations at low z.
- verify: (i) analytic Var(D_b) within a factor 2 of patch estimate for all bins used in the
  likelihood, (ii) analytic corr(N_tot, D_b) and patch estimate agree in sign and magnitude
  trend vs q_cut, (iii) assembled joint covariance is PSD and its Cholesky succeeds, (iv) a
  figure `fig_cov_blocks` (correlation matrix, full sky vs q > 5) passes the VLM loop.
- Fallback if the analytic cross block disagrees with patches beyond factor ~ 2: use the
  patch-estimated cross block with Hartlap correction in the likelihood, and demote the
  analytic version to a cross-check appendix. The paper's claims survive either way.

### Phase 4: Inference-grade theory (3-4 days, highest technical risk)

- [ ] **CNC theory.** Reuse the nb27 cosmocnc-style Poisson likelihood but swap the HMF to a
      FLAMINGO-consistent one: first check nb08's measured HMF against Tinker08 and Bocquet16
      hydro; adopt whichever is within the counts' Poisson errors, else fit a 2-parameter
      HMF correction (amplitude + slope in nu) on the DMO catalogue (data exists,
      `data/dmo_L2p8m9/`) and freeze it. Document: this is a simulation-consistency choice,
      not a real-data choice.
      verify: at D3A truth the CNC theory reproduces the lc0 counts with chi2/dof ~ 1 across
      the (q, z) bins.
- [ ] **Masked PS theory.** Strategy in order of preference:
      (a) completeness-weighted halo model (hmfast masked methods) with the exact catalogue
      selection replacing the smooth q(M, z) proxy, restricted to ell >= ell_min where the
      1-halo term dominates and the known 2-halo deconvolution problem is < 1% (measure
      ell_min per q_cut using opus_4.8_v1 and the maps; expected ~ 150 to 300);
      (b) if (a) misses > 2% inside the adopted range, add a fixed multiplicative transfer
      function T(ell, q_cut) calibrated once at D3A from the catalogue-anchored model
      (opus_4.8_v1), applied to all cosmologies; test its cosmology dependence at +/- 5% in
      sigma_8 by recomputing the halo-model ratio (cheap, no maps needed);
      (c) hard fallback: drop the lowest bins until (a) passes; the joint constraints lose
      little because low-ell full-sky information is cosmic-variance limited anyway (the
      covariance from Phase 3 quantifies exactly how little).
      verify: theory vs lc0 masked bandpowers < 2% (target < 1%) for every q_cut inside the
      adopted ell range, with a residual figure through the VLM loop; and the SAME scaling
      relation parameters (Y*, alpha, sigma_lnY, B) feed both probes (one module, one truth).
- [ ] **Derivative sanity.** d ln C_ell / d ln sigma_8 and d ln N / d ln sigma_8 finite,
      smooth, and matching finite differences (JAX grad vs eps = 1e-3) at D3A.

### Phase 5: Fisher forecast to prune the grid (1 day, GPU, cheap)

- [ ] `fable_5_v1/fisher.py` over parameters (Omega_m, sigma_8, B or 1-b, sigma_lnY, alpha_SZ)
      using Phase 3 covariance and Phase 4 derivatives, for each q_cut x {CNC, PS, joint} x
      {with, without cross-cov}.
      Deliverable: `fig_fom_vs_qcut` (the paper's money forecast) + a table that fixes which
      MCMC runs are worth their compute. Expect ~ 25 Fisher evaluations, minutes on GPU.
- verify: Fisher sigma(S8) for CNC alone and PS alone bracket the existing nb27/nb25 MCMC
  posteriors within ~ 30% (else debug before any MCMC).

### Phase 6: MCMC inference grid (3-5 days wall time, mostly queue)

Runs (cobaya, `chains/` under `/scratch/scratch-lxu/fable_5_v1_data/chains/`, R-1 < 0.01,
each with a manifest; reuse existing likelihood classes, add
`joint_cnc_yy_likelihood.py` that consumes the Phase 3 full covariance):

| id | data | cross-cov | B prior | purpose |
|---|---|---|---|---|
| G1 | CNC only | n/a | fixed B=1 | baseline (exists, rerun with Phase 4 HMF) |
| G2 | PS full sky | n/a | fixed | baseline |
| G3a-d | PS masked q>{50,20,10,5} | n/a | fixed | masking ladder |
| G4a-d | joint, q_cut ladder | yes | fixed | the headline |
| G5 | joint, best q_cut | no | fixed | quantifies claim C3 |
| G6 | joint, best q_cut | yes | CCCP on 1-b | realistic-prior variant |
| G7 | joint, best q_cut | yes | U(1, 2) | agnostic-prior variant |

- [ ] Launch in dependency order G1-G3 then G4-G7; two consecutive failures of the same run
      is a mandatory stop. Runtime calibration: nb27-30 chains as reference; declare
      expected wall time per chain in the manifest before submitting.
- verify per run: R-1 < 0.01, no prior-boundary pileup (except where the prior is the point),
  posterior includes D3A truth within 2 sigma unless a bias is the documented finding.

### Phase 7: Systematics and robustness (2-3 days)

- [ ] **Feedback:** repeat the Fisher (not MCMC) analysis on the 9 L1_m9 variants using the
      cached `data/bandpowers_L1_m9_feedback/` + `data/nb40_l1_m9_feedback_cnc_qz.npz`;
      report the induced (Omega_m, sigma_8, B) shift vs q_cut (claim C4). MCMC only for the
      two extreme variants at the best q_cut.
- [ ] **Aperture and selection:** Y_R500c vs Y_5R500c catalogue, q_from_Y vs q_from_mz
      selection, at the Fisher level; one MCMC cross-check if any shift > 0.3 sigma.
- [ ] **Observer scatter:** repeat the best-q_cut joint MCMC on 2 more lightcones (or all 8
      at Fisher level); report the posterior-centre scatter vs the predicted parameter
      covariance.
- [ ] **z < 0.35 unrotated subset:** one joint run as an independent-frame sanity check
      (data cached from nb23/29).

### Phase 8: Key figures and tables (interleaved with 6-7)

Target figure list (every one through the mandatory VLM review loop, PDF + PNG >= 300 dpi,
color-blind safe, linestyle + marker redundancy):

1. `fig_maps`: y-map cutout full sky vs q > 5 masked (visual motivation).
2. `fig_datavector`: counts N(q, z) + bandpower ladder D_ell per q_cut with errors.
3. `fig_theory_residuals`: Phase 4 theory vs data per q_cut (the < 2% panel).
4. `fig_cov_blocks`: joint correlation matrix, full sky vs masked (claim C1).
5. `fig_variance_suppression`: seed01 right panel, upgraded with patch + analytic curves.
6. `fig_fom_vs_qcut`: Fisher + MCMC FoM and sigma(S8) vs q_cut, with/without cross-cov
   (claims C2 + C3, the money plot).
7. `fig_contours`: (Omega_m, sigma_8) and (S8, B) for G1, G2, G3d, G4-best, with D3A truth.
8. `fig_feedback`: parameter shifts across L1_m9 variants vs q_cut (claim C4).
Tables: data-vector definition; run grid + constraints; systematics budget.

### Phase 9: Paper, review, closure (3-4 days)

- [ ] Draft with the ARS pipeline (`academic-research-skills:academic-paper`), MNRAS style,
      structure: intro (Phase 1 text), FLAMINGO + data vector, joint covariance, theory,
      results (constraints + optimum q_cut), systematics, discussion (real-survey outlook:
      SO / CMB-S4 noise instead of SZiFi-immf6 as future work, one paragraph only).
- [ ] Punctuation gate: `grep -nP '\x{2014}|\x{2013}| -- ' fable_5_v1/paper/*.tex` returns
      nothing.
- [ ] Simulated peer review (`/ars-reviewer`), address all major points, one revision round,
      re-review; citation check (`/ars-citation-check`).
- [ ] Close out per autonomous-operation "what success looks like": final logbook entry,
      complete manifest, `REPORT.md` with setup, results, linked figures, open questions.

---

## 5. Risks and fallbacks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Parametric masked-PS theory can't reach 2% at low ell | high | Phase 4 ladder (exact selection -> frozen transfer -> drop bins); covariance tells us the information loss is small |
| Analytic cross-covariance formalism subtle (mask + deconvolution effects) | medium | patch-based empirical block as the primary, analytic as cross-check (explicit fallback in Phase 3) |
| 8 observers too correlated to validate anything | certain, known | patches on one map are the primary empirical estimator; observers are consistency only |
| Joint MCMC underconstrained / degenerate with B free | medium | Fisher first (Phase 5) prunes hopeless runs; CCCP-prior variant is the realistic headline |
| HMF mismatch reintroduces the S8 bias | medium | Phase 4 FLAMINGO-consistent HMF with an explicit chi2 gate before any chain |
| Scope creep (real-data forecast, bispectrum, CMB lensing...) | high | out of scope by decree; one discussion paragraph max |

## 6. Budget and stop conditions

- Compute budget (manifest-enforced): 40 GPU-hours + 300 CPU-core-hours + queue time.
  Estimated usage: Phase 3 patches ~ 100 CPU-h; Phase 5 < 1 GPU-h; Phase 6 ~ 7 chains x
  (2-6 h) CPU/GPU; Phase 7 ~ 30 Fisher + 4 chains. Well inside budget; if a single phase
  exceeds 2x its estimate, log and halve the batch per the hardware-envelope rule.
- All mandatory stop conditions of `.claude/rules/autonomous-operation.md` apply verbatim
  (two same-error failures, double OOM, budget, protected files, non-localized test
  regression). Additional scientific stop: if Phase 1 finds the paper already exists, or
  Phase 3 finds corr(N, D) ~ 1 even after masking (joint analysis pointless), stop and
  report to the user rather than continuing.

## 7. Success criteria

The run is done when: claims C1-C4 each have a number, a figure, and an artifact path; the
joint covariance is validated (Phase 3 verify items); all figures passed the VLM loop; the
paper draft survived one simulated-review round; `REPORT.md` + logbook + manifest complete;
and nothing outside `autoresearch/fable_5_v1/` + `/scratch/scratch-lxu/fable_5_v1_data/` was
modified.
