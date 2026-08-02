# L1_m9 qfrommap Cobaya Inference Design

## Objective

Run signal-only Cobaya inference for the fiducial L1_m9 Compton-y map for
full sky and masks with `q > 50`, `q > 20`, `q > 10`, and `q > 5`. Use the
18-bin `_qfrommap` bandpowers, hmfast theory on GPU, a covariance evaluated at
the refitted full-sky theory, and the signal-only prior recipe recentered on
the FLAMINGO D3A cosmology where appropriate.

## Data and cases

The five likelihoods use these data products:

- full sky: `Dl_yy_L1_m9_fullsky_binned_18.txt`;
- masked: `Dl_yy_L1_m9_masked_qgt{50,20,10,5}_qfrommap_binned_18.txt`.

The masked covariance uses the effective sky fraction of the actual
`_qfrommap` mask:

| Case | `f_sky_eff` |
| --- | ---: |
| full sky | 1.0 |
| `qgt50` | 0.9972647840959187 |
| `qgt20` | 0.9858059150434103 |
| `qgt10` | 0.9435688947539681 |
| `qgt5` | 0.8595492784996496 |

## Theory

Use the existing custom-GNFW hmfast model with physical halo masses in
`M_sun`, explicit `M_500c`, and both one-halo and two-halo terms. Fix the
profile shape to `P0=8.13`, `c500=1.156`, `alpha=1.062`, `beta=5.4807`, and
`gamma=0.3292`.

Fix `B=1.41` everywhere: the preliminary amplitude fit, covariance
calculation, selection, and final chains. Fix `alpha_SZ=1.12` and
`sigma_lnY=0.173` in the preliminary full-sky amplitude fit.

Full sky is the `q_cat -> infinity` limit of the same selection theory used
for masked spectra. Consequently the unconditional lognormal moments are
included:

- one-halo power uses `exp(2 sigma_lnY^2)`;
- each two-halo bracket uses the first moment, giving a total factor
  `exp(sigma_lnY^2)`;
- the one-halo trispectrum uses `exp(8 sigma_lnY^2)`.

For finite `q_cat`, use `conditional_An_undetected` with `n_power=2`, `1`, and
`4` for the one-halo power, each two-halo bracket, and the trispectrum,
respectively. This makes the full-sky calculation continuous with the
`q_cat -> infinity` limit and matches the synthetic-data theory recipe.

## Bandpower convention and covariance

Match the synthetic-data 18-bin implementation exactly. Every Planck bin is
inclusive at both integer-multipole edges. For the theory mean, log-interpolate
positive `D_ell` to the integer multipoles, then take the uniform mean within
each inclusive bin.

For the covariance, port the operations in
`tsz_cnc_paper_plots/tsz_only/bandpower_theory.py`: interpolate `C_ell`
linearly in `log(ell)`, sum the per-multipole Gaussian variance with the exact
inclusive-bin `D_ell` weights, and project the one-halo trispectrum with that
reference covariance operator. Because adjacent inclusive bins share an edge,
retain their small Gaussian off-diagonal covariance. Assemble

`Cov = Cov_G + Cov_T`, with `Cov_T = T_bb' / (4 pi f_sky_eff)`.

Validate every covariance for shape, finite entries, component reconstruction,
symmetry, positive definiteness, and condition number. The current covariance
products must not be reused: they use upper-exclusive bins, an effective-ell
Knox approximation, and `_qfrommz` sky fractions.

## A_SZ fit and converged covariance

Iterate the amplitude-only full-sky fit and its theory covariance to a stable
fixed point:

1. Build a corrected seed full-sky covariance at the old amplitude
   `A_SZ=-4.0953238`, now including `sigma_lnY=0.173` scatter moments and the
   canonical inclusive-bin calculation.
2. Run a GPU Cobaya MCMC on the full-sky 18-bin data varying only `A_SZ`.
   Hold D3A cosmology, `B=1.41`, `alpha_SZ=1.12`, and
   `sigma_lnY=0.173` fixed. Use the broad amplitude bounds of the earlier
   calibration rather than imposing the later Gaussian prior.
3. Recompute the full-sky covariance at the resulting best-fit `A_SZ`, then
   repeat the amplitude-only MCMC with that covariance.
4. Stop when both `abs(delta A_SZ) < 1e-4` and the relative Frobenius-norm
   covariance change is below `1e-3`; fail explicitly if ten fit updates do
   not satisfy both checks.
5. Compute the final full-sky and four `_qfrommap` masked covariances at the
   converged full-sky best-fit parameters.
6. Center the final science-chain prior at that best fit:
   `A_SZ ~ Normal(A_SZ_best_fit, 0.03)`.

Record each iteration's chain, best-fit summary, covariance provenance, and
convergence metrics. Keep the converged covariance products separate from the
seed and intermediate products.

## Final science-chain priors

Preserve the reference signal-only prior widths and distinguish
prior-dominated parameters from parameters constrained by the tSZ spectrum.
Use:

- `H0 ~ Normal(68.1, 1.0)`;
- `n_s ~ Normal(0.965, 0.014)`;
- `omega_b ~ Normal(0.0225387846, 0.002)`;
- `sigma_8 ~ Uniform(0.6, 1.0)`, initialized near the D3A value
  `0.80257015`;
- `Omega_m ~ Uniform(0.2, 0.5)`, initialized near `0.306`;
- `A_SZ ~ Normal(A_SZ_best_fit, 0.03)`;
- `alpha_SZ ~ Normal(1.12, 0.03)`;
- `sigma_lnY ~ Normal(0.173, 0.023)`;
- `tau_reio = 0.0544` and `B = 1.41` fixed.

Derive `omega_cdm` from `Omega_m`, `H0`, and `omega_b` while accounting for
the fixed `0.06 eV` neutrino density used by D3A, so that the center is exactly
the D3A matter density rather than treating all non-baryonic matter as CDM.
Report derived `S8 = sigma_8 (Omega_m / 0.3)^0.5`.

## Execution and outputs

Require a CUDA JAX device before theory or covariance production. Run the
preliminary fit and all five final chains with hmfast on GPU. Keep independent
output directories for `fullsky`, `qgt50`, `qgt20`, `qgt10`, and `qgt5`, and
write the fully resolved Cobaya input for each case.

The final handoff includes covariance diagnostics, GPU provenance, the
refitted `A_SZ`, prior/configuration checks, Cobaya convergence diagnostics,
and posterior summaries for all five cases. A run is complete only if each
chain reaches the configured Gelman-Rubin stopping criterion and all expected
artifacts are readable.

## Verification

Automated tests must establish:

- inclusive binning agrees with a literal integer-multipole calculation;
- Gaussian shared-edge off-diagonals are retained;
- Gaussian and trispectrum projection agree with the synthetic-data
  `bandpower_theory.py` implementation;
- the `q_cat -> infinity` conditional moments reproduce the analytic
  full-sky scatter factors;
- all cases route to `_qfrommap` data and sky fractions;
- D3A-centered priors and fixed `B=1.41` appear in resolved Cobaya inputs;
- covariance matrices are finite, symmetric, reconstruct from their
  components, and positive definite;
- JAX reports a CUDA device for production runs.
