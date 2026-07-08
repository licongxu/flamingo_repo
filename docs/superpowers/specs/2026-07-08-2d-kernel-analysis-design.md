# 2D (z, M) kernel-analysis plots via hmfast (GPU)

Date: 2026-07-08. Autonomous background job; design self-approved per goal-hook
directive (user unavailable for interactive approval).

## Goal

Replace/extend the 1D kernel-analysis figures used in
`/scratch/scratch-lxu/ai_paper/combine_tsz_clusters/paper.tex` (Sec. `sec:kernels`)
with 2D maps in the (z, M) plane, styled after Osato & Nagai (2021) Fig. 3
(`ref_literature/plots/Osato_2021_fig3.png`): panels of
`d^2 X / (dln z dln M)` at fixed multipoles, for

1. full-sky tSZ power spectrum `C_ell^{tSZ,full}`,
2. masked (unresolved) tSZ power spectrum `C_ell^{tSZ,unres}` (q_cat = 5,
   sigma_lnY = 0.173),
3. resolved cluster number counts N (q > 5).

The original 1D figures (`/scratch/scratch-lxu/tsz_cnc_paper_plots/kernel_analysis/`)
were computed with `tszpower` (source module
`/scratch/scratch-lxu/tsz_cnc_scatter/kernel_analysis/kernels.py`). tszpower no
longer imports on this machine (classy_sz/TensorFlow CUDA_ERROR_INVALID_HANDLE on
Blackwell GPUs). The new computation must be fully hmfast, on GPU.

## Physics

All three observables share the double integral over the halo population:

- `d^2 C_ell / (dz dlnM) = dV/dz/dOmega(z) * W_y(z)^2 * dn/dlnM(M,z) * |y_ell(M,z)|^2 * S(M,z)`
- `d^2 N / (dz dlnM)     = 4*pi * dV/dz/dOmega(z) * dn/dlnM(M,z) * P_det(M,z)`

with `y_ell = W_y * u_k(k_ell)` under Limber (`k_ell = (ell+1/2)/chi(z)`), and the
selection weights

- full PS: `S = 1`
- masked PS: `S = <A^2 1(q_obs < q_cat)>` (log-normal scatter conditional moment,
  `conditional_An_undetected(..., n_power=2)`), which is the paper's stated
  masking formalism (Sec. masking) and the configuration validated to <1% against
  tszpower in `hmfast/benchmarks/plot_masked_match.py`.
  Note: the legacy 1D kernel scripts used `S = 1 - P_det` (n_power=0); the
  normalized kernels differ only mildly. The script exposes `n_power` as config.
- counts: `P_det = 1 - <1(q_obs < q_cat)>` = `1 - conditional_An_undetected(..., n_power=0)`,
  which reproduces tszpower's `_pdet_grid_from_qbar` analytically.

The Osato-style kernel is the log-log density, normalized to unit double integral:

`K(z, M) = z * d^2 X / (dz dlnM) / X`,  with `X = int dz dlnM d^2X/(dz dlnM)`.

## Configuration (matches the paper's existing 1D figures + validated benchmark)

- Cosmology: `lcdm:v1`, H0=67.66, omega_cdm=0.1193, omega_b=0.02242,
  ln1e10A_s=2.9718, n_s=0.9665, tau_reio=0.0544.
- HaloModel: `MassDefinition(500, "critical")`, T08 HMF, `convert_masses=True`
  (as in the validated benchmark).
- PS form factor: `GNFWPressureProfile(P0=8.130, c500=1.156, gamma=0.3292,
  alpha=1.0620, beta=5.4807, B=1.41)` (pure Arnaud A10, exactly the profile the
  original tszpower kernels used via `y_ell_interpolate`).
- SNR grid: `build_snr_grid(A_SZ=-4.2373, alpha_SZ=1.12, B=1.41)` with Planck
  SZiFi noise curve (immf6). Sanity check in-script: parametric y0 vs the
  closed-form Arnaud y0 (tszpower `compute_y0` port) should agree to a few %.
- q_cat = 5, sigma_lnY = 0.173.
- Grids: z geomspace(0.005, 3.0, 200); M_phys geomspace(5e13/h, 1e16/h, 200)
  (axis labels in Msun/h to match the paper); ell in {100, 500, 1000}.

## Implementation

1. **hmfast** (repo `/scratch/scratch-lxu/agent_dev/auto_research_agent/hmfast`,
   new branch `kernel2d`, leaving unrelated WIP in pressure.py untouched):
   add `HaloModel.cl_1h_integrand(tracer1, tracer2, l, m, z, mask_mz=None,
   k_damp=0.01)` returning the Limber 1-halo integrand
   `d^2 C_ell/(dz dlnM)` with shape (Nz, Nl, Nm), mirroring `cl_1h_masked`
   quadrature exactly (same u_k slices, damping, volume, kernels).
   Test (`tests/test_halo_model.py`): shape/finiteness + consistency:
   `trapz_z(trapz_lnM(integrand)) == cl_1h_masked` (allclose, rtol=1e-10) and
   `== cl_1h` for unit mask up to the consistency counterterm (k_damp matched,
   hm_consistency accounted).
2. **Plot script** `/scratch/scratch-lxu/tsz_cnc_paper_plots/kernel_analysis/kernels_2d.py`
   (colocated with the existing per-figure scripts, hmfast-only, GPU):
   - computes the three kernels on the grid;
   - Figure A `kernels2d_all_observables.{pdf,png}`: 1x3 panels at ell=500
     (full PS, masked PS, CNC), pcolormesh of K(z,M), log-log axes,
     overlaid red dashed q(M,z)=q_cat detection boundary (Osato-style curve
     overlay), per-panel colorbar, cividis/viridis colormap;
   - Figure B `kernels2d_per_ell.{pdf,png}`: rows ell={100,500,1000} x cols
     {full, masked} with the same overlay (CNC is ell-independent, so it is
     not repeated);
   - writes a sidecar JSON (hmfast git hash + dirty-file list, config, grids,
     runtime, device) per reproducibility rules;
   - cross-checks vs paper numbers: at ell=500 the M-marginal peak of the
     full PS kernel ~6e14 Msun/h, masked ~3e14 Msun/h, CNC ~1e15 Msun/h;
     <M>, <z> vs the values quoted in Sec. kernels.
3. **Figure review**: VLM loop on the PNGs until publication-ready.
4. **Bookkeeping**: flamingo_repo `runs/logbook.md` entry + heartbeat.

## Success criteria

- New hmfast test passes; existing fast tests unaffected.
- 1D marginals of the 2D kernels reproduce the published 1D kernel shapes/peaks
  (peak positions within a grid cell; <M>, <z> within ~10% of paper values,
  given the n_power=2 vs 1-P_det weighting difference for the masked PS).
- Figures pass the VLM review loop; PDFs + 300 dpi PNGs + sidecar JSON exist in
  `/scratch/scratch-lxu/tsz_cnc_paper_plots/kernel_analysis/`.
- Everything runs on GPU (JAX default device), wall time < 10 min, memory << 4 GiB
  (200*200*3 float64 grids ~ 1 MB; u_k slices (3, 200) per z step — trivial).
