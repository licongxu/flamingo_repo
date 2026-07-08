# Project logbook

## 2026-07-08 02:33 UTC — L1_m9 rotation-group tSZ PS (nb40)

- git: cacd8fc (branch masking_clusters)
- Intent: replicate nb39 for L1_m9: 13 rotation-group shell-sum maps for all 9
  feedback variants, cached D_ell, notebook 40 with theory + feedback plots.
- Commands: `python scripts/build_rotation_group_maps.py --run <variant>
  --parent L1_m9 --no-cache-shells` (fiducial solo + 8 variants via xargs -P3);
  logs in `logs/rotgroups_<variant>.log`.
- Artifacts: /rds/rds-lxu/flamingo/L1_m9/maps/rotation_groups/ (117 FITS +
  9 JSON, ~190 GB), data/nb40_l1_m9_rotation_group_tsz_ps/<variant>.npz,
  notebooks/40_l1_m9_rotation_group_tsz_ps.ipynb,
  figures/nb40_l1_m9_rotation_group_tsz_ps/.
- Estimates: 1.61 GB per nside4096 float64 map; ~5 GB peak RSS per build
  process; ~17 s/shell solo, ~60-70 s/shell at 4 concurrent streams (~3 h total).
- Conclusion: pending (builds running).

## 2026-07-08 04:35 UTC — Resolved tSZ PS vs Fourier-space stacking (nb41)

- git: cacd8fc (branch masking_clusters)
- Intent: end-to-end empirical validation: resolved PS (full-sky minus q>5 masked
  NaMaster bandpowers, fiducial L1_m9, 1088 clusters) vs an empirical theory from
  per-cluster Fourier transforms of 5*theta500 gnomonic stamps (nb31 method, absolute
  units); no hmfast in the comparison.
- Commands: `python scripts/export_L1_m9_resolved_stack_yell.py --order {0,1}`
  (100 s each) and `--order 0 --randoms 3` (293 s); log
  logs/nb41_resolved_stack_yell.log; `python scripts/make_nb41.py` + nbconvert.
- Artifacts: data/nb41_resolved_stack_yell/L1_m9_qgt5_order{0,1}.npz,
  L1_m9_qgt5_order0_randoms3.npz (+ .json manifests),
  notebooks/41_resolved_tsz_ps_fourier_stack.ipynb (executed),
  figures/nb41_resolved_tsz_ps_fourier_stack/{resolved_vs_stack,
  stacked_fourier_profile,dc_vs_catalogue_Y}.{pdf,png}.
- Estimates: 1.6 GB map in RAM, ~2.5 GB peak; 0.1-0.3 s/cluster stamp+FFT.
- Findings: raw stamp sum biased high (aperture area 8.1% of sky carries background
  power; ratio up to 2.9 at ell>3000) - fixed with random-aperture background
  subtraction. Unweighted stacked-shape estimator destroyed by faint-cluster noise -
  fixed with y0^2-weighted stacking. Final: direct sum / resolved = 1.04 / 0.98 /
  0.90 and stacked shape / resolved = 1.14 / 1.03 / 1.16 in ell 300-1000 /
  1000-3000 / 3000-6000. Stamp DC modes match catalogue Y_5R500c/dA^2 (median 0.89).
- Conclusion: Fourier-space stacking validated end-to-end; stop (task complete).

## 2026-07-08 04:00 UTC — nb42 rotation-group stacked-profile theory (in progress)

- git: cacd8fc (uncommitted work); intent: replace A10 theory of nb39 with empirical
  Fourier-space stacked profiles per L2p8_m9 lc0 rotation group (goal session).
- Verified group maps are raw shell sums matching catalogue NATIVE coords
  (map/mean 300-1000x at native positions of top group-1 clusters, ~1 at rotated).
- Stage sample: stratified subsample of M500c>1e13 lc0 catalogue (24.78M rows),
  0.25 dex bins, cap 500/bin -> 17,770 clusters; data/nb42_rotgroup_stack_yell/sample_group*.npz
  (logs/nb42_rotgroup_sample.log, 61 s).
- First stamp run (incoherent only) aborted after early group-0 check: faint strata
  (logM<13.75, weights N/n up to 1000) are background-noise dominated; bin 13.0-13.25
  contributed -83% of D_3000 (pure noise). Artifact: early check output in session log.
- Fix: coherent phase-centred profile re_y added to export script; faint strata use
  N_b*(mean(re)^2 - Var/n)*R(u), R calibrated on bright strata. Validated on 12 brightest
  group-0 clusters: s^2/S = 1.00 (ell=100) .. 0.90 (ell=2000); E[bg_re]/signal ~ 1%.
- Full stamp run v2 launched (pid 1885842, ~0.40 s/cluster incl. 2 randoms,
  est. ~105 min); logs/nb42_rotgroup_stamp.log; outputs group{g}_order0.npz + manifests.
- Continue: build nb42 with hybrid estimator, VLM figure loop, final logbook entry.
- Outcome (2026-07-08 ~05:30 UTC): all 9 variants built (117 FITS + 9 JSON,
  176 GB, no stream errors); 9 npz spectra caches finite, D_3000(all groups):
  fiducial 1.55e-12, fgas+2sigma 1.62e-12 ... fgas-8sigma 1.09e-12 (expected
  feedback ordering). nb40 executed cleanly; 5 figures reviewed visually and
  accepted. Done.

## 2026-07-08 06:04 UTC — nb43 CNC x tSZ-PS covariance, L1_m9 patch resampling (in progress)

- git: 350d024 (uncommitted); intent: goal session - reproduce Hurier & Lacasa (2017)
  Fig 7 (no bispectrum) on L1_m9 fiducial: joint corr matrices [N_cl(z), C_ell] for
  full-sky and q>5-masked tSZ PS, last 9 ell bins (ell_eff 117-959.5).
- Prior session's nb43 used the JXPaint 1000-realization painted benchmark (synthetic
  hmfast fiducial, NOT L1_m9); goal explicitly says L1_m9 fiducial, nb40 convention.
  L1_m9 is a single sky, so the covariance now comes from patch resampling
  (192 HEALPix nside=4 patches, ~215 deg^2), per the joint-CNC+masked-PS plan memory;
  the painted MC is kept as a cross-check section.
- Commands: `python scripts/export_L1_m9_patch_cnc_ps.py` (bg task b2pk0e5mk,
  logs/nb43_patch_ensemble.log, ~4 s/patch, est ~15 min; calibrated on 2 patches first).
  Map degraded 4096->1024 (lmax 1085 needed), q>5 mask = 5xtheta500 discs C1 0.5deg
  (same as compute_masked_tsz_ps_L1_m9_feedback.py); estimator anafast((y-<y>_w)w)/<w^2>,
  pixwin-deconvolved. Peak mem ~3 GB (one 1.6 GB map read + nside-1024 work arrays).
- Sanity so far: N(q>5)=1088 matches nb40 exactly; N(q>5,z<1)=1079.
- Artifacts (so far): data/nb43_L1_m9_patch_cov/patch_ensemble.npz (checkpointed),
  scripts/export_L1_m9_patch_cnc_ps.py, rewritten scripts/make_nb43.py.
- Continue: execute nb43, VLM figure loop, final entry.

## 2026-07-08 06:40 UTC — nb42 rotation-group tSZ PS from stacked profiles: DONE

- git: 350d024 (pre-commit); goal session: replace nb39's A10 theory with empirical
  Fourier-space stacked profiles per L2p8_m9 lc0 rotation group.
- Data: stamp run v2 completed for all 5 groups (17,770 clusters + 2 randoms each,
  0.38-0.45 s/cluster, logs/nb42_rotgroup_stamp.log clean of OOM/errors);
  data/nb42_rotgroup_stack_yell/group{0..4}_order0.npz + JSON manifests +
  measured_dl_group{g}.npz caches.
- Estimators: hybrid stratified direct sum (incoherent T_b where band SNR>=5 or
  fully enumerated; coherent stacked-profile U_b = N_b(mean(s)^2 - Var/n)R(u)
  otherwise) and pure stacked-profile route (U_b everywhere). Shape route via
  per-cluster y0^2 amplitudes was abandoned: faint-strata y0 are background-inflated
  10-60x (diagnosed per stratum against T_b).
- Result (band medians, stack/measured): group0 1.09/1.05/1.00, group1 1.14/1.04/0.91,
  group2 1.05/1.18/0.95, group3 0.72/1.13/1.00, group4 0.39/0.78/0.86 in
  ell 300-1000/1000-3000/3000-6000; pure-stacked-profile route within a few % of
  the hybrid. A10/measured for comparison: group2 0.66-0.81, group3 0.43-0.64,
  group4 0.45-0.51. Residuals accounted: low-ell 2-halo (A10-2h curve closes gap),
  ell~1000-2000 neighbour-pair double count, group4 catalogue floor not converged
  (lowest stratum 20% of D_3000 and rising).
- Artifacts: notebooks/42_rotgroup_tsz_ps_fourier_stack.ipynb (executed, 0 errors),
  scripts/export_L2p8_rotgroup_stack_yell.py, scripts/make_nb42.py,
  figures/nb42_rotgroup_tsz_ps_fourier_stack/{rotgroup_dl_vs_stack_5panel,
  stacked_profiles_per_group,massbin_contributions_D3000}.{pdf,png} (VLM-reviewed).
- Conclusion: stacked-profile theory reproduces the group spectra at 1-halo scales
  (5-18% for ell>=1000, groups 0-3) and hugely improves on A10 at high z. Stop.
