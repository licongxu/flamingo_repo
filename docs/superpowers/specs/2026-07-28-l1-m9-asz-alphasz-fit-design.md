# L1_m9 Custom-GNFW A_SZ and Alpha_SZ Fit Design

## Goal

Fit `A_SZ` and `alpha_SZ` to all 18 fiducial FLAMINGO `L1_m9` full-sky
Compton-y bandpowers with Cobaya. Fix the cosmology and every other pressure
profile parameter, evaluate the custom-GNFW theory with hmfast on a GPU, store
the chain under the repository's `chains/` directory, and report the best-fit
parameters with a two-dimensional contour plot.

## Inputs

The observed data vector is
`data_paper/binned_bandpowers/Dl_yy_L1_m9_fullsky_binned_18.txt`. Its second
column is `1e12 D_ell`; the likelihood converts it back to raw dimensionless
`D_ell` by multiplying by `1e-12`.

The likelihood uses the newly computed full covariance
`data_paper/covariance/cov_full_L1_m9_fullsky_Dl_yy_binned_18.npy`. This
covariance includes a Gaussian term computed from the simple-GNFW one-halo
plus two-halo spectrum and the connected one-halo trispectrum term. All 18
bins, from `ell_eff = 10` through `959.5`, enter the fit.

## Fixed Theory

Use the FLAMINGO D3A cosmology from
`flamingo.catalogue.frame.D3A_COSMOLOGY`, with its exact hmfast parameter
values:

- `H0 = 68.1`
- `omega_cdm = 0.11872788986038219`
- `omega_b = 0.022538784599999993`
- `ln(1e10 A_s) = 3.035173309489548`
- `n_s = 0.965`
- `tau_reio = 0.0544`
- `m_ncdm = 0.06 eV`

Use hmfast `ParametricGNFWPressureProfile`, the custom-GNFW profile exposing
`A_SZ` and `alpha_SZ`. Fix its radial shape to the same values used for the
covariance:

- `P0 = 8.13`
- `c500 = 1.156`
- `alpha = 1.062`
- `beta = 5.4807`
- `gamma = 0.3292`
- `B = 1`

Do not apply a separate lognormal-scatter boost. Use a `500c` halo-mass
definition, physical masses in `M_sun`, `convert_masses = True`, and
`hm_consistency = False`. Evaluate 64 logarithmic mass samples over
`1e11 <= M/M_sun <= 1e16`, 96 logarithmic redshift samples over
`0.005 <= z <= 3`, and a 50-point logarithmic multipole grid spanning the
18 bins.

The predicted bandpowers must contain both terms:

```text
D_ell = ell (ell + 1) (C_ell^1h + C_ell^2h) / (2 pi).
```

## Sampled Parameters

Only these two parameters vary:

```text
A_SZ     ~ Uniform(-5.5, -3.0)
alpha_SZ ~ Uniform(0.7, 1.5)
```

Initialize from Gaussian reference distributions
`A_SZ ~ Normal(-4.1, 0.08)` and `alpha_SZ ~ Normal(1.12, 0.05)`; these are
not priors. Use initial proposal widths `0.05` and `0.03`, respectively, and
allow Cobaya to learn the covariance.

## Cobaya Components

Add one minimal Cobaya `Theory` class. It constructs the fixed halo model and
custom-GNFW tracer once, JIT-compiles the two-parameter one-halo plus two-halo
calculation, bins each term using exactly the same 18-bin operator as the
covariance script, and provides `Cl_sz = {"1h", "2h"}`.

Add one Gaussian `Likelihood` class. It loads and validates the 18-row data
file and the positive-definite `18 x 18` covariance, checks that the theory
contains 18 finite values, and returns `-chi2/2`.

The Cobaya YAML fixes no hidden nuisance parameters because the theory class
requests only `A_SZ` and `alpha_SZ`. Use the MCMC sampler with adaptive
proposal learning and `Rminus1_stop = 0.01`.

## Outputs

The chain root is:

```text
chains/l1_m9_customgnfw_asz_alphasz/chain
```

Cobaya writes its ordinary chain, checkpoint, progress, covariance, input,
and updated YAML files there.

Post-processing writes:

- `chains/l1_m9_customgnfw_asz_alphasz/best_fit.json`
- `chains/l1_m9_customgnfw_asz_alphasz/best_fit.txt`
- `chains/l1_m9_customgnfw_asz_alphasz/contour_A_SZ_alpha_SZ.png`
- `chains/l1_m9_customgnfw_asz_alphasz/contour_A_SZ_alpha_SZ.pdf`
- `chains/l1_m9_customgnfw_asz_alphasz/posterior_summary.txt`

Discard the first 30% of the stored chain during post-processing. Define the
reported best fit as the remaining MCMC sample with the smallest `chi2`.
Because both priors are flat within their bounds, this is also the
maximum-likelihood retained sample. The contour plot shows filled 68% and 95%
credible regions and marks that best-fit sample.

## Verification

Before reporting results:

1. Unit tests must show that the likelihood rescales the observed
   `1e12 D_ell` data correctly and computes the expected quadratic form.
2. A theory smoke test must produce finite, positive 18-bin one-halo and
   two-halo spectra and verify that both terms are included.
3. The runtime log and Cobaya metadata must confirm CUDA execution.
4. The chain must stop with `Rminus1 <= 0.01` and contain finite retained
   samples away from both prior boundaries.
5. Re-evaluating the theory at the reported best fit must reproduce its
   recorded `chi2`.
6. The contour PNG and PDF must exist and be non-empty.
7. The final report must print the best-fit `A_SZ`, `alpha_SZ`, `chi2`, the
   posterior means and 68% intervals, and link the chain folder and contour.
