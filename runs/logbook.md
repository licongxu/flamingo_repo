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

## 2026-07-08 07:35 UTC — nb43 CNC x tSZ-PS covariance, L1_m9 patch resampling (done)

- Commands: `python scripts/export_L1_m9_patch_cnc_ps.py` (bg b2pk0e5mk, 910 s, 192/192
  patches, log logs/nb43_patch_ensemble.log clean: 0 NaN/error/OOM);
  `python scripts/make_nb43.py` + nbconvert --execute (clean).
- Artifacts: data/nb43_L1_m9_patch_cov/patch_ensemble.npz (+.json manifest, git 350d024),
  notebooks/43_cnc_tsz_ps_covariance.ipynb (executed),
  figures/nb43_cnc_tsz_ps_covariance/{corrmat_L1m9_fullsky,corrmat_L1m9_masked,
  cross_corr_vs_ell,corrmat_painted_mc}.{pdf,png} (all through VLM review loop;
  stale cnc_tsz_ps_correlation.* from the superseded painted-only version removed).
- Anchors: sum_p N_z == nb40 Nz(L1_m9) exactly ([270 349 229 141 72 18], N(q>5)=1088);
  patch-mean D_ell / stored NaMaster bandpowers = 0.999-1.032 (full sky) and
  0.982-1.009 (masked q>5) over the 9 bins ell_eff 117-959.5.
- Findings: L1_m9 q>5 counts end at z~0.6, so 4 empty nb40 z bins dropped (NaN fix).
  Full-sky joint corr: cross block mean r=+0.135, max 0.377; corr[N_tot,C_ell]
  mean +0.38 rising with ell. Masked q>5: cross block mean +0.057;
  corr[N_tot,C_ell] mean +0.16 - residual patch-level LSS coupling (2-halo),
  absent (+0.016) in the 1000-realization unclustered painted MC cross-check.
- Conclusion: Hurier & Lacasa Fig 7 structure reproduced on L1_m9 fiducial (masking
  q>5 strongly decorrelates counts from the PS); done, stop.

## append_comptonY_to_catalogues start; tokens=['fgas+2sigma', 'fgas-2sigma', 'fgas-4sigma', 'fgas-8sigma', 'Jet', 'Jet_fgas-4sigma', 'Mstar-1sigma', 'Mstar-1sigma_fgas-4sigma']
[fgas+2sigma] reading snap+soap_index from halo_catalogue_M500c_1e13_zlt3_fgas+2sigma_yang26rot.csv (11.1 GB)
[fgas+2sigma] 26,088,053 rows read in 22s; streaming ComptonY
    fgas+2sigma snap 0017: 16,784 rows, 0 out-of-range, 15s
    fgas+2sigma snap 0018: 37,855 rows, 0 out-of-range, 14s
    fgas+2sigma snap 0019: 43,528 rows, 0 out-of-range, 15s
    fgas+2sigma snap 0020: 49,985 rows, 0 out-of-range, 15s
    fgas+2sigma snap 0021: 56,394 rows, 0 out-of-range, 14s
    fgas+2sigma snap 0022: 63,758 rows, 0 out-of-range, 15s
    fgas+2sigma snap 0023: 72,495 rows, 0 out-of-range, 15s
    fgas+2sigma snap 0024: 82,002 rows, 0 out-of-range, 15s
    fgas+2sigma snap 0025: 92,598 rows, 0 out-of-range, 16s
    fgas+2sigma snap 0026: 103,631 rows, 0 out-of-range, 15s
    fgas+2sigma snap 0027: 116,343 rows, 0 out-of-range, 16s
    fgas+2sigma snap 0028: 129,501 rows, 0 out-of-range, 15s
    fgas+2sigma snap 0029: 142,654 rows, 0 out-of-range, 16s
    fgas+2sigma snap 0030: 157,994 rows, 0 out-of-range, 16s
    fgas+2sigma snap 0031: 175,079 rows, 0 out-of-range, 17s
    fgas+2sigma snap 0032: 191,676 rows, 0 out-of-range, 17s
    fgas+2sigma snap 0033: 212,045 rows, 0 out-of-range, 18s
    fgas+2sigma snap 0034: 232,686 rows, 0 out-of-range, 17s
    fgas+2sigma snap 0035: 255,267 rows, 0 out-of-range, 17s
    fgas+2sigma snap 0036: 279,960 rows, 0 out-of-range, 17s
    fgas+2sigma snap 0037: 308,453 rows, 0 out-of-range, 18s
    fgas+2sigma snap 0038: 334,347 rows, 0 out-of-range, 18s
    fgas+2sigma snap 0039: 363,182 rows, 0 out-of-range, 18s
    fgas+2sigma snap 0040: 395,196 rows, 0 out-of-range, 19s
    fgas+2sigma snap 0041: 424,454 rows, 0 out-of-range, 19s
    fgas+2sigma snap 0042: 454,812 rows, 0 out-of-range, 19s
    fgas+2sigma snap 0043: 490,243 rows, 0 out-of-range, 19s
    fgas+2sigma snap 0044: 524,977 rows, 0 out-of-range, 18s
    fgas+2sigma snap 0045: 560,553 rows, 0 out-of-range, 18s
    fgas+2sigma snap 0046: 592,396 rows, 0 out-of-range, 19s
    fgas+2sigma snap 0047: 626,921 rows, 0 out-of-range, 19s
    fgas+2sigma snap 0048: 668,445 rows, 0 out-of-range, 19s
    fgas+2sigma snap 0049: 703,548 rows, 0 out-of-range, 18s
    fgas+2sigma snap 0050: 736,828 rows, 0 out-of-range, 19s
    fgas+2sigma snap 0051: 772,356 rows, 0 out-of-range, 19s
    fgas+2sigma snap 0052: 798,845 rows, 0 out-of-range, 19s
    fgas+2sigma snap 0053: 828,680 rows, 0 out-of-range, 18s
    fgas+2sigma snap 0054: 856,306 rows, 0 out-of-range, 20s
    fgas+2sigma snap 0055: 878,877 rows, 0 out-of-range, 19s
    fgas+2sigma snap 0056: 894,461 rows, 0 out-of-range, 20s
    fgas+2sigma snap 0057: 902,430 rows, 0 out-of-range, 21s
    fgas+2sigma snap 0058: 908,442 rows, 0 out-of-range, 20s
    fgas+2sigma snap 0059: 910,846 rows, 0 out-of-range, 19s
    fgas+2sigma snap 0060: 902,014 rows, 0 out-of-range, 20s
    fgas+2sigma snap 0061: 889,621 rows, 0 out-of-range, 19s
    fgas+2sigma snap 0062: 866,357 rows, 0 out-of-range, 19s
    fgas+2sigma snap 0063: 835,426 rows, 0 out-of-range, 18s
    fgas+2sigma snap 0064: 794,599 rows, 0 out-of-range, 19s
    fgas+2sigma snap 0065: 742,766 rows, 0 out-of-range, 19s
    fgas+2sigma snap 0066: 687,367 rows, 0 out-of-range, 19s
    fgas+2sigma snap 0067: 624,998 rows, 0 out-of-range, 18s
    fgas+2sigma snap 0068: 551,248 rows, 0 out-of-range, 18s
    fgas+2sigma snap 0069: 475,439 rows, 0 out-of-range, 19s
    fgas+2sigma snap 0070: 394,410 rows, 0 out-of-range, 18s
    fgas+2sigma snap 0071: 316,447 rows, 0 out-of-range, 18s
    fgas+2sigma snap 0072: 234,434 rows, 0 out-of-range, 20s
    fgas+2sigma snap 0073: 164,152 rows, 0 out-of-range, 17s
    fgas+2sigma snap 0074: 99,888 rows, 0 out-of-range, 17s
    fgas+2sigma snap 0075: 47,146 rows, 0 out-of-range, 17s
    fgas+2sigma snap 0076: 13,341 rows, 0 out-of-range, 17s
    fgas+2sigma snap 0077: 567 rows, 0 out-of-range, 12s
[fgas+2sigma] Y_500c finite 26,088,053/26,088,053; median_pos=1.179e-07
  wrote 26,088,053 data rows -> halo_catalogue_M500c_1e13_zlt3_fgas+2sigma_yang26rot.csv.tmp
[fgas+2sigma] DONE in 19.6 min -> /rds/rds-lxu/flamingo/L1_m9/catalogues/halo_catalogue_M500c_1e13_zlt3_fgas+2sigma_yang26rot.csv
[fgas-2sigma] reading snap+soap_index from halo_catalogue_M500c_1e13_zlt3_fgas-2sigma_yang26rot.csv (10.0 GB)
[fgas-2sigma] 23,655,236 rows read in 19s; streaming ComptonY
    fgas-2sigma snap 0017: 14,938 rows, 0 out-of-range, 13s
    fgas-2sigma snap 0018: 33,686 rows, 0 out-of-range, 14s
    fgas-2sigma snap 0019: 38,989 rows, 0 out-of-range, 15s
    fgas-2sigma snap 0020: 44,559 rows, 0 out-of-range, 15s
    fgas-2sigma snap 0021: 50,901 rows, 0 out-of-range, 16s
    fgas-2sigma snap 0022: 57,554 rows, 0 out-of-range, 16s
    fgas-2sigma snap 0023: 65,173 rows, 0 out-of-range, 15s
    fgas-2sigma snap 0024: 73,217 rows, 0 out-of-range, 15s
    fgas-2sigma snap 0025: 82,627 rows, 0 out-of-range, 16s
    fgas-2sigma snap 0026: 93,329 rows, 0 out-of-range, 17s
    fgas-2sigma snap 0027: 103,362 rows, 0 out-of-range, 17s
    fgas-2sigma snap 0028: 116,650 rows, 0 out-of-range, 16s
    fgas-2sigma snap 0029: 128,324 rows, 0 out-of-range, 18s
    fgas-2sigma snap 0030: 141,025 rows, 0 out-of-range, 18s
    fgas-2sigma snap 0031: 156,782 rows, 0 out-of-range, 18s
    fgas-2sigma snap 0032: 171,851 rows, 0 out-of-range, 18s
    fgas-2sigma snap 0033: 187,061 rows, 0 out-of-range, 18s
    fgas-2sigma snap 0034: 207,176 rows, 0 out-of-range, 18s
    fgas-2sigma snap 0035: 228,122 rows, 0 out-of-range, 19s
    fgas-2sigma snap 0036: 248,630 rows, 0 out-of-range, 18s
    fgas-2sigma snap 0037: 274,310 rows, 0 out-of-range, 20s
    fgas-2sigma snap 0038: 296,752 rows, 0 out-of-range, 19s
    fgas-2sigma snap 0039: 322,748 rows, 0 out-of-range, 20s
    fgas-2sigma snap 0040: 349,630 rows, 0 out-of-range, 20s
    fgas-2sigma snap 0041: 379,113 rows, 0 out-of-range, 19s
    fgas-2sigma snap 0042: 407,237 rows, 0 out-of-range, 19s
    fgas-2sigma snap 0043: 437,531 rows, 0 out-of-range, 18s
    fgas-2sigma snap 0044: 469,316 rows, 0 out-of-range, 20s
    fgas-2sigma snap 0045: 501,586 rows, 0 out-of-range, 19s
    fgas-2sigma snap 0046: 530,270 rows, 0 out-of-range, 19s
    fgas-2sigma snap 0047: 561,929 rows, 0 out-of-range, 19s
    fgas-2sigma snap 0048: 598,337 rows, 0 out-of-range, 20s
    fgas-2sigma snap 0049: 630,188 rows, 0 out-of-range, 19s
    fgas-2sigma snap 0050: 662,114 rows, 0 out-of-range, 19s
    fgas-2sigma snap 0051: 693,988 rows, 0 out-of-range, 20s
    fgas-2sigma snap 0052: 722,156 rows, 0 out-of-range, 21s
    fgas-2sigma snap 0053: 750,548 rows, 0 out-of-range, 20s
    fgas-2sigma snap 0054: 776,199 rows, 0 out-of-range, 22s
    fgas-2sigma snap 0055: 795,916 rows, 0 out-of-range, 20s
    fgas-2sigma snap 0056: 812,532 rows, 0 out-of-range, 20s
    fgas-2sigma snap 0057: 821,086 rows, 0 out-of-range, 19s
    fgas-2sigma snap 0058: 828,362 rows, 0 out-of-range, 20s
    fgas-2sigma snap 0059: 830,477 rows, 0 out-of-range, 20s
    fgas-2sigma snap 0060: 825,022 rows, 0 out-of-range, 20s
    fgas-2sigma snap 0061: 814,274 rows, 0 out-of-range, 19s
    fgas-2sigma snap 0062: 793,780 rows, 0 out-of-range, 20s
    fgas-2sigma snap 0063: 766,314 rows, 0 out-of-range, 18s
    fgas-2sigma snap 0064: 730,454 rows, 0 out-of-range, 20s
    fgas-2sigma snap 0065: 683,370 rows, 0 out-of-range, 23s
    fgas-2sigma snap 0066: 633,834 rows, 0 out-of-range, 20s
    fgas-2sigma snap 0067: 576,332 rows, 0 out-of-range, 20s
    fgas-2sigma snap 0068: 510,803 rows, 0 out-of-range, 21s
    fgas-2sigma snap 0069: 441,053 rows, 0 out-of-range, 19s
    fgas-2sigma snap 0070: 366,611 rows, 0 out-of-range, 19s
    fgas-2sigma snap 0071: 294,747 rows, 0 out-of-range, 20s
    fgas-2sigma snap 0072: 218,475 rows, 0 out-of-range, 20s
    fgas-2sigma snap 0073: 153,122 rows, 0 out-of-range, 18s
    fgas-2sigma snap 0074: 93,439 rows, 0 out-of-range, 19s
    fgas-2sigma snap 0075: 44,266 rows, 0 out-of-range, 19s
    fgas-2sigma snap 0076: 12,528 rows, 0 out-of-range, 20s
    fgas-2sigma snap 0077: 531 rows, 0 out-of-range, 12s
[fgas-2sigma] Y_500c finite 23,655,236/23,655,236; median_pos=8.207e-08
  wrote 23,655,236 data rows -> halo_catalogue_M500c_1e13_zlt3_fgas-2sigma_yang26rot.csv.tmp
[fgas-2sigma] DONE in 20.5 min -> /rds/rds-lxu/flamingo/L1_m9/catalogues/halo_catalogue_M500c_1e13_zlt3_fgas-2sigma_yang26rot.csv
[fgas-4sigma] reading snap+soap_index from halo_catalogue_M500c_1e13_zlt3_fgas-4sigma_yang26rot.csv (9.8 GB)
[fgas-4sigma] 23,144,434 rows read in 19s; streaming ComptonY
    fgas-4sigma snap 0017: 14,458 rows, 0 out-of-range, 17s
    fgas-4sigma snap 0018: 32,381 rows, 0 out-of-range, 15s
    fgas-4sigma snap 0019: 37,516 rows, 0 out-of-range, 15s
    fgas-4sigma snap 0020: 43,177 rows, 0 out-of-range, 15s
    fgas-4sigma snap 0021: 49,154 rows, 0 out-of-range, 17s
    fgas-4sigma snap 0022: 55,292 rows, 0 out-of-range, 14s
    fgas-4sigma snap 0023: 62,455 rows, 0 out-of-range, 17s
    fgas-4sigma snap 0024: 69,671 rows, 0 out-of-range, 15s
    fgas-4sigma snap 0025: 79,674 rows, 0 out-of-range, 19s
    fgas-4sigma snap 0026: 89,729 rows, 0 out-of-range, 16s
    fgas-4sigma snap 0027: 100,461 rows, 0 out-of-range, 17s
    fgas-4sigma snap 0028: 111,814 rows, 0 out-of-range, 18s
    fgas-4sigma snap 0029: 123,921 rows, 0 out-of-range, 18s
    fgas-4sigma snap 0030: 136,779 rows, 0 out-of-range, 16s
    fgas-4sigma snap 0031: 151,550 rows, 0 out-of-range, 18s
    fgas-4sigma snap 0032: 165,456 rows, 0 out-of-range, 18s
    fgas-4sigma snap 0033: 181,364 rows, 0 out-of-range, 20s
    fgas-4sigma snap 0034: 199,711 rows, 0 out-of-range, 19s
    fgas-4sigma snap 0035: 220,058 rows, 0 out-of-range, 18s
    fgas-4sigma snap 0036: 241,666 rows, 0 out-of-range, 20s
    fgas-4sigma snap 0037: 266,610 rows, 0 out-of-range, 20s
    fgas-4sigma snap 0038: 289,015 rows, 0 out-of-range, 20s
    fgas-4sigma snap 0039: 313,027 rows, 0 out-of-range, 21s
    fgas-4sigma snap 0040: 339,840 rows, 0 out-of-range, 20s
    fgas-4sigma snap 0041: 367,109 rows, 0 out-of-range, 20s
    fgas-4sigma snap 0042: 395,584 rows, 0 out-of-range, 20s
    fgas-4sigma snap 0043: 426,243 rows, 0 out-of-range, 21s
    fgas-4sigma snap 0044: 456,773 rows, 0 out-of-range, 22s
    fgas-4sigma snap 0045: 487,446 rows, 0 out-of-range, 19s
    fgas-4sigma snap 0046: 517,636 rows, 0 out-of-range, 20s
    fgas-4sigma snap 0047: 548,122 rows, 0 out-of-range, 22s
    fgas-4sigma snap 0048: 584,359 rows, 0 out-of-range, 21s
    fgas-4sigma snap 0049: 615,650 rows, 0 out-of-range, 26s
    fgas-4sigma snap 0050: 647,233 rows, 0 out-of-range, 19s
    fgas-4sigma snap 0051: 679,721 rows, 0 out-of-range, 23s
    fgas-4sigma snap 0052: 705,109 rows, 0 out-of-range, 20s
    fgas-4sigma snap 0053: 732,633 rows, 0 out-of-range, 19s
    fgas-4sigma snap 0054: 760,234 rows, 0 out-of-range, 20s
    fgas-4sigma snap 0055: 779,410 rows, 0 out-of-range, 22s
    fgas-4sigma snap 0056: 795,771 rows, 0 out-of-range, 21s
    fgas-4sigma snap 0057: 806,536 rows, 0 out-of-range, 19s
    fgas-4sigma snap 0058: 812,999 rows, 0 out-of-range, 21s
    fgas-4sigma snap 0059: 815,797 rows, 0 out-of-range, 22s
    fgas-4sigma snap 0060: 810,430 rows, 0 out-of-range, 21s
    fgas-4sigma snap 0061: 799,940 rows, 0 out-of-range, 18s
    fgas-4sigma snap 0062: 780,965 rows, 0 out-of-range, 21s
    fgas-4sigma snap 0063: 753,295 rows, 0 out-of-range, 22s
    fgas-4sigma snap 0064: 718,617 rows, 0 out-of-range, 20s
    fgas-4sigma snap 0065: 672,330 rows, 0 out-of-range, 20s
    fgas-4sigma snap 0066: 624,628 rows, 0 out-of-range, 19s
    fgas-4sigma snap 0067: 568,453 rows, 0 out-of-range, 19s
    fgas-4sigma snap 0068: 503,396 rows, 0 out-of-range, 21s
    fgas-4sigma snap 0069: 435,067 rows, 0 out-of-range, 19s
    fgas-4sigma snap 0070: 361,704 rows, 0 out-of-range, 20s
    fgas-4sigma snap 0071: 290,738 rows, 0 out-of-range, 18s
    fgas-4sigma snap 0072: 215,701 rows, 0 out-of-range, 19s
    fgas-4sigma snap 0073: 151,108 rows, 0 out-of-range, 19s
    fgas-4sigma snap 0074: 92,406 rows, 0 out-of-range, 19s
    fgas-4sigma snap 0075: 43,639 rows, 0 out-of-range, 17s
    fgas-4sigma snap 0076: 12,348 rows, 0 out-of-range, 17s
    fgas-4sigma snap 0077: 525 rows, 0 out-of-range, 11s
[fgas-4sigma] Y_500c finite 23,144,434/23,144,434; median_pos=7.043e-08
  wrote 23,144,434 data rows -> halo_catalogue_M500c_1e13_zlt3_fgas-4sigma_yang26rot.csv.tmp
[fgas-4sigma] DONE in 21.1 min -> /rds/rds-lxu/flamingo/L1_m9/catalogues/halo_catalogue_M500c_1e13_zlt3_fgas-4sigma_yang26rot.csv
[fgas-8sigma] reading snap+soap_index from halo_catalogue_M500c_1e13_zlt3_fgas-8sigma_yang26rot.csv (9.6 GB)
[fgas-8sigma] 22,545,218 rows read in 18s; streaming ComptonY
    fgas-8sigma snap 0017: 13,660 rows, 0 out-of-range, 17s
    fgas-8sigma snap 0018: 31,100 rows, 0 out-of-range, 15s
    fgas-8sigma snap 0019: 35,846 rows, 0 out-of-range, 15s
    fgas-8sigma snap 0020: 40,815 rows, 0 out-of-range, 16s
    fgas-8sigma snap 0021: 47,033 rows, 0 out-of-range, 16s
    fgas-8sigma snap 0022: 52,726 rows, 0 out-of-range, 17s
    fgas-8sigma snap 0023: 60,087 rows, 0 out-of-range, 17s
    fgas-8sigma snap 0024: 66,629 rows, 0 out-of-range, 15s
    fgas-8sigma snap 0025: 75,736 rows, 0 out-of-range, 17s
    fgas-8sigma snap 0026: 85,639 rows, 0 out-of-range, 16s
    fgas-8sigma snap 0027: 95,883 rows, 0 out-of-range, 17s
    fgas-8sigma snap 0028: 107,092 rows, 0 out-of-range, 18s
    fgas-8sigma snap 0029: 118,787 rows, 0 out-of-range, 19s
    fgas-8sigma snap 0030: 130,740 rows, 0 out-of-range, 18s
    fgas-8sigma snap 0031: 145,010 rows, 0 out-of-range, 17s
    fgas-8sigma snap 0032: 157,845 rows, 0 out-of-range, 18s
    fgas-8sigma snap 0033: 174,273 rows, 0 out-of-range, 19s
    fgas-8sigma snap 0034: 191,718 rows, 0 out-of-range, 19s
    fgas-8sigma snap 0035: 211,493 rows, 0 out-of-range, 18s
    fgas-8sigma snap 0036: 231,070 rows, 0 out-of-range, 18s
    fgas-8sigma snap 0037: 256,323 rows, 0 out-of-range, 20s
    fgas-8sigma snap 0038: 278,655 rows, 0 out-of-range, 20s
    fgas-8sigma snap 0039: 300,368 rows, 0 out-of-range, 19s
    fgas-8sigma snap 0040: 327,043 rows, 0 out-of-range, 22s
    fgas-8sigma snap 0041: 353,215 rows, 0 out-of-range, 18s
    fgas-8sigma snap 0042: 380,798 rows, 0 out-of-range, 19s
    fgas-8sigma snap 0043: 410,960 rows, 0 out-of-range, 20s
    fgas-8sigma snap 0044: 442,262 rows, 0 out-of-range, 18s
    fgas-8sigma snap 0045: 472,099 rows, 0 out-of-range, 19s
    fgas-8sigma snap 0046: 500,172 rows, 0 out-of-range, 21s
    fgas-8sigma snap 0047: 530,761 rows, 0 out-of-range, 20s
    fgas-8sigma snap 0048: 567,601 rows, 0 out-of-range, 18s
    fgas-8sigma snap 0049: 597,804 rows, 0 out-of-range, 21s
    fgas-8sigma snap 0050: 628,253 rows, 0 out-of-range, 19s
    fgas-8sigma snap 0051: 658,190 rows, 0 out-of-range, 22s
    fgas-8sigma snap 0052: 685,989 rows, 0 out-of-range, 22s
    fgas-8sigma snap 0053: 713,204 rows, 0 out-of-range, 21s
    fgas-8sigma snap 0054: 740,466 rows, 0 out-of-range, 19s
    fgas-8sigma snap 0055: 762,108 rows, 0 out-of-range, 14s
    fgas-8sigma snap 0056: 777,912 rows, 0 out-of-range, 20s
    fgas-8sigma snap 0057: 788,723 rows, 0 out-of-range, 21s
    fgas-8sigma snap 0058: 794,900 rows, 0 out-of-range, 21s
    fgas-8sigma snap 0059: 799,692 rows, 0 out-of-range, 19s
    fgas-8sigma snap 0060: 795,602 rows, 0 out-of-range, 19s
    fgas-8sigma snap 0061: 785,584 rows, 0 out-of-range, 20s
    fgas-8sigma snap 0062: 766,979 rows, 0 out-of-range, 18s
    fgas-8sigma snap 0063: 740,394 rows, 0 out-of-range, 14s
    fgas-8sigma snap 0064: 705,633 rows, 0 out-of-range, 20s
    fgas-8sigma snap 0065: 661,411 rows, 0 out-of-range, 19s
    fgas-8sigma snap 0066: 614,459 rows, 0 out-of-range, 14s
    fgas-8sigma snap 0067: 559,689 rows, 0 out-of-range, 19s
    fgas-8sigma snap 0068: 495,444 rows, 0 out-of-range, 18s
    fgas-8sigma snap 0069: 427,881 rows, 0 out-of-range, 18s
    fgas-8sigma snap 0070: 356,157 rows, 0 out-of-range, 22s
    fgas-8sigma snap 0071: 286,512 rows, 0 out-of-range, 18s
    fgas-8sigma snap 0072: 212,737 rows, 0 out-of-range, 14s
    fgas-8sigma snap 0073: 149,050 rows, 0 out-of-range, 19s
    fgas-8sigma snap 0074: 91,126 rows, 0 out-of-range, 19s
    fgas-8sigma snap 0075: 43,115 rows, 0 out-of-range, 19s
    fgas-8sigma snap 0076: 12,255 rows, 0 out-of-range, 20s
    fgas-8sigma snap 0077: 510 rows, 0 out-of-range, 15s
[fgas-8sigma] Y_500c finite 22,545,218/22,545,218; median_pos=5.751e-08
  wrote 22,545,218 data rows -> halo_catalogue_M500c_1e13_zlt3_fgas-8sigma_yang26rot.csv.tmp
[fgas-8sigma] DONE in 20.3 min -> /rds/rds-lxu/flamingo/L1_m9/catalogues/halo_catalogue_M500c_1e13_zlt3_fgas-8sigma_yang26rot.csv
[Jet] reading snap+soap_index from halo_catalogue_M500c_1e13_zlt3_Jet_yang26rot.csv (11.6 GB)
[Jet] 27,373,102 rows read in 22s; streaming ComptonY
    Jet snap 0017: 17,598 rows, 0 out-of-range, 15s
    Jet snap 0018: 39,926 rows, 0 out-of-range, 15s
    Jet snap 0019: 46,119 rows, 0 out-of-range, 15s
    Jet snap 0020: 52,581 rows, 0 out-of-range, 16s
    Jet snap 0021: 59,684 rows, 0 out-of-range, 15s
    Jet snap 0022: 67,212 rows, 0 out-of-range, 15s
    Jet snap 0023: 76,155 rows, 0 out-of-range, 16s
    Jet snap 0024: 86,003 rows, 0 out-of-range, 18s
    Jet snap 0025: 97,622 rows, 0 out-of-range, 16s
    Jet snap 0026: 108,755 rows, 0 out-of-range, 15s
    Jet snap 0027: 121,917 rows, 0 out-of-range, 15s
    Jet snap 0028: 136,418 rows, 0 out-of-range, 16s
    Jet snap 0029: 150,473 rows, 0 out-of-range, 17s
    Jet snap 0030: 165,807 rows, 0 out-of-range, 16s
    Jet snap 0031: 184,033 rows, 0 out-of-range, 20s
    Jet snap 0032: 200,750 rows, 0 out-of-range, 18s
    Jet snap 0033: 221,987 rows, 0 out-of-range, 18s
    Jet snap 0034: 244,602 rows, 0 out-of-range, 19s
    Jet snap 0035: 269,193 rows, 0 out-of-range, 18s
    Jet snap 0036: 294,738 rows, 0 out-of-range, 18s
    Jet snap 0037: 325,548 rows, 0 out-of-range, 20s
    Jet snap 0038: 351,919 rows, 0 out-of-range, 18s
    Jet snap 0039: 380,190 rows, 0 out-of-range, 19s
    Jet snap 0040: 413,294 rows, 0 out-of-range, 19s
    Jet snap 0041: 444,382 rows, 0 out-of-range, 18s
    Jet snap 0042: 477,353 rows, 0 out-of-range, 20s
    Jet snap 0043: 514,103 rows, 0 out-of-range, 19s
    Jet snap 0044: 550,032 rows, 0 out-of-range, 22s
    Jet snap 0045: 586,873 rows, 0 out-of-range, 20s
    Jet snap 0046: 622,343 rows, 0 out-of-range, 18s
    Jet snap 0047: 659,440 rows, 0 out-of-range, 21s
    Jet snap 0048: 701,887 rows, 0 out-of-range, 19s
    Jet snap 0049: 739,606 rows, 0 out-of-range, 21s
    Jet snap 0050: 774,822 rows, 0 out-of-range, 25s
    Jet snap 0051: 813,261 rows, 0 out-of-range, 19s
    Jet snap 0052: 843,096 rows, 0 out-of-range, 19s
    Jet snap 0053: 872,868 rows, 0 out-of-range, 19s
    Jet snap 0054: 899,043 rows, 0 out-of-range, 19s
    Jet snap 0055: 923,384 rows, 0 out-of-range, 20s
    Jet snap 0056: 940,088 rows, 0 out-of-range, 19s
    Jet snap 0057: 948,596 rows, 0 out-of-range, 19s
    Jet snap 0058: 954,301 rows, 0 out-of-range, 20s
    Jet snap 0059: 958,026 rows, 0 out-of-range, 19s
    Jet snap 0060: 949,133 rows, 0 out-of-range, 18s
    Jet snap 0061: 934,900 rows, 0 out-of-range, 22s
    Jet snap 0062: 910,098 rows, 0 out-of-range, 19s
    Jet snap 0063: 875,666 rows, 0 out-of-range, 18s
    Jet snap 0064: 833,728 rows, 0 out-of-range, 19s
    Jet snap 0065: 776,512 rows, 0 out-of-range, 19s
    Jet snap 0066: 718,093 rows, 0 out-of-range, 18s
    Jet snap 0067: 652,590 rows, 0 out-of-range, 20s
    Jet snap 0068: 574,724 rows, 0 out-of-range, 19s
    Jet snap 0069: 494,383 rows, 0 out-of-range, 18s
    Jet snap 0070: 409,888 rows, 0 out-of-range, 19s
    Jet snap 0071: 328,478 rows, 0 out-of-range, 20s
    Jet snap 0072: 242,751 rows, 0 out-of-range, 18s
    Jet snap 0073: 169,870 rows, 0 out-of-range, 20s
    Jet snap 0074: 103,298 rows, 0 out-of-range, 19s
    Jet snap 0075: 48,646 rows, 0 out-of-range, 18s
    Jet snap 0076: 13,730 rows, 0 out-of-range, 18s
    Jet snap 0077: 586 rows, 0 out-of-range, 12s
[Jet] Y_500c finite 27,373,102/27,373,102; median_pos=1.058e-07
  wrote 27,373,102 data rows -> halo_catalogue_M500c_1e13_zlt3_Jet_yang26rot.csv.tmp
[Jet] DONE in 20.5 min -> /rds/rds-lxu/flamingo/L1_m9/catalogues/halo_catalogue_M500c_1e13_zlt3_Jet_yang26rot.csv
[Jet_fgas-4sigma] reading snap+soap_index from halo_catalogue_M500c_1e13_zlt3_Jet_fgas-4sigma_yang26rot.csv (9.3 GB)
[Jet_fgas-4sigma] 21,890,977 rows read in 18s; streaming ComptonY
    Jet_fgas-4sigma snap 0017: 12,815 rows, 0 out-of-range, 15s
    Jet_fgas-4sigma snap 0018: 29,250 rows, 0 out-of-range, 15s
    Jet_fgas-4sigma snap 0019: 33,732 rows, 0 out-of-range, 15s
    Jet_fgas-4sigma snap 0020: 38,017 rows, 0 out-of-range, 14s
    Jet_fgas-4sigma snap 0021: 43,827 rows, 0 out-of-range, 15s
    Jet_fgas-4sigma snap 0022: 50,132 rows, 0 out-of-range, 15s
    Jet_fgas-4sigma snap 0023: 56,648 rows, 0 out-of-range, 14s
    Jet_fgas-4sigma snap 0024: 62,464 rows, 0 out-of-range, 18s
    Jet_fgas-4sigma snap 0025: 71,537 rows, 0 out-of-range, 15s
    Jet_fgas-4sigma snap 0026: 80,463 rows, 0 out-of-range, 17s
    Jet_fgas-4sigma snap 0027: 90,072 rows, 0 out-of-range, 17s
    Jet_fgas-4sigma snap 0028: 100,593 rows, 0 out-of-range, 15s
    Jet_fgas-4sigma snap 0029: 111,605 rows, 0 out-of-range, 16s
    Jet_fgas-4sigma snap 0030: 123,253 rows, 0 out-of-range, 19s
    Jet_fgas-4sigma snap 0031: 135,670 rows, 0 out-of-range, 16s
    Jet_fgas-4sigma snap 0032: 149,304 rows, 0 out-of-range, 17s
    Jet_fgas-4sigma snap 0033: 164,689 rows, 0 out-of-range, 19s
    Jet_fgas-4sigma snap 0034: 182,421 rows, 0 out-of-range, 18s
    Jet_fgas-4sigma snap 0035: 198,751 rows, 0 out-of-range, 18s
    Jet_fgas-4sigma snap 0036: 219,775 rows, 0 out-of-range, 20s
    Jet_fgas-4sigma snap 0037: 241,243 rows, 0 out-of-range, 19s
    Jet_fgas-4sigma snap 0038: 263,018 rows, 0 out-of-range, 18s
    Jet_fgas-4sigma snap 0039: 285,193 rows, 0 out-of-range, 19s
    Jet_fgas-4sigma snap 0040: 312,041 rows, 0 out-of-range, 20s
    Jet_fgas-4sigma snap 0041: 337,901 rows, 0 out-of-range, 19s
    Jet_fgas-4sigma snap 0042: 363,151 rows, 0 out-of-range, 19s
    Jet_fgas-4sigma snap 0043: 393,026 rows, 0 out-of-range, 18s
    Jet_fgas-4sigma snap 0044: 423,442 rows, 0 out-of-range, 19s
    Jet_fgas-4sigma snap 0045: 452,234 rows, 0 out-of-range, 20s
    Jet_fgas-4sigma snap 0046: 477,848 rows, 0 out-of-range, 20s
    Jet_fgas-4sigma snap 0047: 507,986 rows, 0 out-of-range, 19s
    Jet_fgas-4sigma snap 0048: 544,436 rows, 0 out-of-range, 19s
    Jet_fgas-4sigma snap 0049: 575,156 rows, 0 out-of-range, 20s
    Jet_fgas-4sigma snap 0050: 606,216 rows, 0 out-of-range, 18s
    Jet_fgas-4sigma snap 0051: 636,754 rows, 0 out-of-range, 20s
    Jet_fgas-4sigma snap 0052: 662,484 rows, 0 out-of-range, 19s
    Jet_fgas-4sigma snap 0053: 691,269 rows, 0 out-of-range, 21s
    Jet_fgas-4sigma snap 0054: 718,555 rows, 0 out-of-range, 18s
    Jet_fgas-4sigma snap 0055: 739,399 rows, 0 out-of-range, 19s
    Jet_fgas-4sigma snap 0056: 755,602 rows, 0 out-of-range, 18s
    Jet_fgas-4sigma snap 0057: 765,912 rows, 0 out-of-range, 19s
    Jet_fgas-4sigma snap 0058: 776,605 rows, 0 out-of-range, 18s
    Jet_fgas-4sigma snap 0059: 782,502 rows, 0 out-of-range, 20s
    Jet_fgas-4sigma snap 0060: 778,973 rows, 0 out-of-range, 20s
    Jet_fgas-4sigma snap 0061: 769,348 rows, 0 out-of-range, 19s
    Jet_fgas-4sigma snap 0062: 754,030 rows, 0 out-of-range, 19s
    Jet_fgas-4sigma snap 0063: 730,140 rows, 0 out-of-range, 20s
    Jet_fgas-4sigma snap 0064: 697,591 rows, 0 out-of-range, 20s
    Jet_fgas-4sigma snap 0065: 653,719 rows, 0 out-of-range, 19s
    Jet_fgas-4sigma snap 0066: 610,643 rows, 0 out-of-range, 18s
    Jet_fgas-4sigma snap 0067: 555,972 rows, 0 out-of-range, 19s
    Jet_fgas-4sigma snap 0068: 493,958 rows, 0 out-of-range, 19s
    Jet_fgas-4sigma snap 0069: 426,952 rows, 0 out-of-range, 18s
    Jet_fgas-4sigma snap 0070: 355,809 rows, 0 out-of-range, 19s
    Jet_fgas-4sigma snap 0071: 286,428 rows, 0 out-of-range, 18s
    Jet_fgas-4sigma snap 0072: 212,916 rows, 0 out-of-range, 18s
    Jet_fgas-4sigma snap 0073: 149,856 rows, 0 out-of-range, 19s
    Jet_fgas-4sigma snap 0074: 91,498 rows, 0 out-of-range, 17s
    Jet_fgas-4sigma snap 0075: 43,322 rows, 0 out-of-range, 17s
    Jet_fgas-4sigma snap 0076: 12,306 rows, 0 out-of-range, 17s
    Jet_fgas-4sigma snap 0077: 525 rows, 0 out-of-range, 11s
[Jet_fgas-4sigma] Y_500c finite 21,890,977/21,890,977; median_pos=7.101e-08
  wrote 21,890,977 data rows -> halo_catalogue_M500c_1e13_zlt3_Jet_fgas-4sigma_yang26rot.csv.tmp
[Jet_fgas-4sigma] DONE in 19.8 min -> /rds/rds-lxu/flamingo/L1_m9/catalogues/halo_catalogue_M500c_1e13_zlt3_Jet_fgas-4sigma_yang26rot.csv
[Mstar-1sigma] reading snap+soap_index from halo_catalogue_M500c_1e13_zlt3_Mstar-1sigma_yang26rot.csv (10.0 GB)
[Mstar-1sigma] 23,549,405 rows read in 19s; streaming ComptonY
    Mstar-1sigma snap 0017: 14,669 rows, 0 out-of-range, 15s
    Mstar-1sigma snap 0018: 33,277 rows, 0 out-of-range, 15s
    Mstar-1sigma snap 0019: 38,146 rows, 0 out-of-range, 15s
    Mstar-1sigma snap 0020: 43,699 rows, 0 out-of-range, 15s
    Mstar-1sigma snap 0021: 50,027 rows, 0 out-of-range, 15s
    Mstar-1sigma snap 0022: 56,188 rows, 0 out-of-range, 15s
    Mstar-1sigma snap 0023: 63,496 rows, 0 out-of-range, 15s
    Mstar-1sigma snap 0024: 72,306 rows, 0 out-of-range, 16s
    Mstar-1sigma snap 0025: 81,129 rows, 0 out-of-range, 15s
    Mstar-1sigma snap 0026: 91,400 rows, 0 out-of-range, 16s
    Mstar-1sigma snap 0027: 103,259 rows, 0 out-of-range, 15s
    Mstar-1sigma snap 0028: 115,062 rows, 0 out-of-range, 17s
    Mstar-1sigma snap 0029: 126,799 rows, 0 out-of-range, 16s
    Mstar-1sigma snap 0030: 139,823 rows, 0 out-of-range, 17s
    Mstar-1sigma snap 0031: 155,428 rows, 0 out-of-range, 17s
    Mstar-1sigma snap 0032: 169,210 rows, 0 out-of-range, 17s
    Mstar-1sigma snap 0033: 185,272 rows, 0 out-of-range, 18s
    Mstar-1sigma snap 0034: 205,842 rows, 0 out-of-range, 17s
    Mstar-1sigma snap 0035: 226,592 rows, 0 out-of-range, 17s
    Mstar-1sigma snap 0036: 247,944 rows, 0 out-of-range, 18s
    Mstar-1sigma snap 0037: 274,248 rows, 0 out-of-range, 17s
    Mstar-1sigma snap 0038: 294,990 rows, 0 out-of-range, 18s
    Mstar-1sigma snap 0039: 321,473 rows, 0 out-of-range, 19s
    Mstar-1sigma snap 0040: 347,493 rows, 0 out-of-range, 18s
    Mstar-1sigma snap 0041: 376,553 rows, 0 out-of-range, 18s
    Mstar-1sigma snap 0042: 404,567 rows, 0 out-of-range, 18s
    Mstar-1sigma snap 0043: 436,085 rows, 0 out-of-range, 19s
    Mstar-1sigma snap 0044: 466,624 rows, 0 out-of-range, 19s
    Mstar-1sigma snap 0045: 498,565 rows, 0 out-of-range, 18s
    Mstar-1sigma snap 0046: 527,738 rows, 0 out-of-range, 19s
    Mstar-1sigma snap 0047: 559,598 rows, 0 out-of-range, 19s
    Mstar-1sigma snap 0048: 596,825 rows, 0 out-of-range, 19s
    Mstar-1sigma snap 0049: 628,320 rows, 0 out-of-range, 19s
    Mstar-1sigma snap 0050: 660,615 rows, 0 out-of-range, 18s
    Mstar-1sigma snap 0051: 692,056 rows, 0 out-of-range, 18s
    Mstar-1sigma snap 0052: 719,771 rows, 0 out-of-range, 18s
    Mstar-1sigma snap 0053: 746,165 rows, 0 out-of-range, 20s
    Mstar-1sigma snap 0054: 771,737 rows, 0 out-of-range, 19s
    Mstar-1sigma snap 0055: 794,387 rows, 0 out-of-range, 20s
    Mstar-1sigma snap 0056: 810,020 rows, 0 out-of-range, 19s
    Mstar-1sigma snap 0057: 820,362 rows, 0 out-of-range, 19s
    Mstar-1sigma snap 0058: 825,983 rows, 0 out-of-range, 19s
    Mstar-1sigma snap 0059: 827,732 rows, 0 out-of-range, 19s
    Mstar-1sigma snap 0060: 822,216 rows, 0 out-of-range, 18s
    Mstar-1sigma snap 0061: 811,036 rows, 0 out-of-range, 19s
    Mstar-1sigma snap 0062: 790,665 rows, 0 out-of-range, 18s
    Mstar-1sigma snap 0063: 763,694 rows, 0 out-of-range, 19s
    Mstar-1sigma snap 0064: 727,794 rows, 0 out-of-range, 19s
    Mstar-1sigma snap 0065: 680,675 rows, 0 out-of-range, 19s
    Mstar-1sigma snap 0066: 631,981 rows, 0 out-of-range, 19s
    Mstar-1sigma snap 0067: 574,573 rows, 0 out-of-range, 18s
    Mstar-1sigma snap 0068: 508,936 rows, 0 out-of-range, 18s
    Mstar-1sigma snap 0069: 438,830 rows, 0 out-of-range, 19s
    Mstar-1sigma snap 0070: 364,567 rows, 0 out-of-range, 17s
    Mstar-1sigma snap 0071: 293,003 rows, 0 out-of-range, 17s
    Mstar-1sigma snap 0072: 217,526 rows, 0 out-of-range, 17s
    Mstar-1sigma snap 0073: 152,614 rows, 0 out-of-range, 20s
    Mstar-1sigma snap 0074: 92,886 rows, 0 out-of-range, 17s
    Mstar-1sigma snap 0075: 43,924 rows, 0 out-of-range, 18s
    Mstar-1sigma snap 0076: 12,481 rows, 0 out-of-range, 17s
    Mstar-1sigma snap 0077: 529 rows, 0 out-of-range, 11s
[Mstar-1sigma] Y_500c finite 23,549,405/23,549,405; median_pos=9.104e-08
  wrote 23,549,405 data rows -> halo_catalogue_M500c_1e13_zlt3_Mstar-1sigma_yang26rot.csv.tmp
[Mstar-1sigma] DONE in 19.4 min -> /rds/rds-lxu/flamingo/L1_m9/catalogues/halo_catalogue_M500c_1e13_zlt3_Mstar-1sigma_yang26rot.csv
[Mstar-1sigma_fgas-4sigma] reading snap+soap_index from halo_catalogue_M500c_1e13_zlt3_Mstar-1sigma_fgas-4sigma_yang26rot.csv (9.6 GB)
[Mstar-1sigma_fgas-4sigma] 22,619,206 rows read in 18s; streaming ComptonY
    Mstar-1sigma_fgas-4sigma snap 0017: 13,588 rows, 0 out-of-range, 14s
    Mstar-1sigma_fgas-4sigma snap 0018: 30,836 rows, 0 out-of-range, 14s
    Mstar-1sigma_fgas-4sigma snap 0019: 35,660 rows, 0 out-of-range, 14s
    Mstar-1sigma_fgas-4sigma snap 0020: 40,737 rows, 0 out-of-range, 14s
    Mstar-1sigma_fgas-4sigma snap 0021: 46,582 rows, 0 out-of-range, 14s
    Mstar-1sigma_fgas-4sigma snap 0022: 52,604 rows, 0 out-of-range, 15s
    Mstar-1sigma_fgas-4sigma snap 0023: 59,603 rows, 0 out-of-range, 16s
    Mstar-1sigma_fgas-4sigma snap 0024: 66,819 rows, 0 out-of-range, 16s
    Mstar-1sigma_fgas-4sigma snap 0025: 75,554 rows, 0 out-of-range, 15s
    Mstar-1sigma_fgas-4sigma snap 0026: 85,362 rows, 0 out-of-range, 16s
    Mstar-1sigma_fgas-4sigma snap 0027: 95,677 rows, 0 out-of-range, 16s
    Mstar-1sigma_fgas-4sigma snap 0028: 107,372 rows, 0 out-of-range, 16s
    Mstar-1sigma_fgas-4sigma snap 0029: 119,017 rows, 0 out-of-range, 17s
    Mstar-1sigma_fgas-4sigma snap 0030: 130,725 rows, 0 out-of-range, 16s
    Mstar-1sigma_fgas-4sigma snap 0031: 144,766 rows, 0 out-of-range, 17s
    Mstar-1sigma_fgas-4sigma snap 0032: 158,456 rows, 0 out-of-range, 17s
    Mstar-1sigma_fgas-4sigma snap 0033: 173,299 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0034: 192,904 rows, 0 out-of-range, 17s
    Mstar-1sigma_fgas-4sigma snap 0035: 212,372 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0036: 232,401 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0037: 257,666 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0038: 280,002 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0039: 303,625 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0040: 330,513 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0041: 355,581 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0042: 384,428 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0043: 413,529 rows, 0 out-of-range, 19s
    Mstar-1sigma_fgas-4sigma snap 0044: 445,350 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0045: 474,567 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0046: 502,935 rows, 0 out-of-range, 19s
    Mstar-1sigma_fgas-4sigma snap 0047: 534,897 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0048: 570,846 rows, 0 out-of-range, 19s
    Mstar-1sigma_fgas-4sigma snap 0049: 601,971 rows, 0 out-of-range, 20s
    Mstar-1sigma_fgas-4sigma snap 0050: 632,480 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0051: 664,498 rows, 0 out-of-range, 20s
    Mstar-1sigma_fgas-4sigma snap 0052: 690,450 rows, 0 out-of-range, 20s
    Mstar-1sigma_fgas-4sigma snap 0053: 717,242 rows, 0 out-of-range, 20s
    Mstar-1sigma_fgas-4sigma snap 0054: 744,987 rows, 0 out-of-range, 19s
    Mstar-1sigma_fgas-4sigma snap 0055: 766,452 rows, 0 out-of-range, 20s
    Mstar-1sigma_fgas-4sigma snap 0056: 780,340 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0057: 789,419 rows, 0 out-of-range, 19s
    Mstar-1sigma_fgas-4sigma snap 0058: 796,780 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0059: 801,706 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0060: 796,484 rows, 0 out-of-range, 19s
    Mstar-1sigma_fgas-4sigma snap 0061: 785,248 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0062: 767,042 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0063: 741,718 rows, 0 out-of-range, 19s
    Mstar-1sigma_fgas-4sigma snap 0064: 706,498 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0065: 661,474 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0066: 614,976 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0067: 559,195 rows, 0 out-of-range, 19s
    Mstar-1sigma_fgas-4sigma snap 0068: 495,149 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0069: 427,429 rows, 0 out-of-range, 19s
    Mstar-1sigma_fgas-4sigma snap 0070: 355,584 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0071: 286,043 rows, 0 out-of-range, 18s
    Mstar-1sigma_fgas-4sigma snap 0072: 212,263 rows, 0 out-of-range, 17s
    Mstar-1sigma_fgas-4sigma snap 0073: 148,827 rows, 0 out-of-range, 17s
    Mstar-1sigma_fgas-4sigma snap 0074: 91,068 rows, 0 out-of-range, 17s
    Mstar-1sigma_fgas-4sigma snap 0075: 42,931 rows, 0 out-of-range, 17s
    Mstar-1sigma_fgas-4sigma snap 0076: 12,162 rows, 0 out-of-range, 17s
    Mstar-1sigma_fgas-4sigma snap 0077: 517 rows, 0 out-of-range, 11s
[Mstar-1sigma_fgas-4sigma] Y_500c finite 22,619,206/22,619,206; median_pos=7.043e-08
  wrote 22,619,206 data rows -> halo_catalogue_M500c_1e13_zlt3_Mstar-1sigma_fgas-4sigma_yang26rot.csv.tmp
[Mstar-1sigma_fgas-4sigma] DONE in 19.4 min -> /rds/rds-lxu/flamingo/L1_m9/catalogues/halo_catalogue_M500c_1e13_zlt3_Mstar-1sigma_fgas-4sigma_yang26rot.csv
## append_comptonY_to_catalogues finished
