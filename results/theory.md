# Halo-model prediction for the total tSZ power spectrum

Produced by `python -m paper_results.figures --figure theory`, which calls
`flamingo.theory.cl_yy`. Figure: `figures/paper/fiducial_tsz_ps_vs_halo_model.pdf`.

## Settings

| | |
|---|---|
| cosmology | FLAMINGO D3A (`flamingo.catalogue.frame.D3A_COSMOLOGY`) |
| pressure profile | Arnaud et al. (2010) GNFW, `flamingo.profiles.gnfw.A10_PARAMS` |
| hydrostatic bias | `B = 1` |
| mass grid | 64 log-spaced points, `1e11 - 1e16 Msun` (**physical** Msun, not Msun/h) |
| redshift grid | 96 log-spaced points, `0.005 - 3` |
| terms | 1-halo + 2-halo |
| `hm_consistency` | `False` |

Nothing is fitted: the FLAMINGO maps are built from the simulated pressure
itself, so there is no observational mass calibration to undo and the
comparison is made directly at `B = 1`.

The grids are converged: varying the number of mass points (64 to 128), the
number of redshift points (96 to 192), and the lower redshift edge (0.005 to
0.001) moves the total by less than 0.1%.

### The `hm_consistency` counterterm

hmfast assigns all mass missing from the grid,
`1 - int dn/dlnm (m/rho_m) dlnm`, to a phantom halo population sitting at
`m_grid[0]` with that halo's pressure profile. On a grid starting at
`1e13 Msun` most of the matter budget is missing, and the counterterm then
**inflates `C_ell^1h` by a factor 3.3 at `ell = 6000`**. It is off by default,
and on the grid above the resolved halos carry nearly all the mass, so the
switch moves `C_ell^1h` by <0.1% (2-halo, which is ~2% of the total, by ~16%).
`tests/test_theory.py` guards this.

## Agreement with the FLAMINGO L1_m9 map

Ratio map / halo model over `100 <= ell <= 6000`, no free parameters:

| | ratio |
|---|---|
| median | 0.985 |
| min | 0.862 |
| max | 1.109 |

| `ell` | ratio |
|---|---|
| 197 | 1.072 |
| 496 | 0.882 |
| 1006 | 0.894 |
| 1996 | 0.959 |
| 2987 | 0.987 |
| 5956 | 0.982 |

The 2-halo term is subdominant throughout: ~2% of the total at `ell = 200`,
falling steeply, so the measured spectrum is essentially the 1-halo term.

## Equivalent custom-GNFW scaling-relation parameters at `B = 1`

The `y_0(M_500c, z)` amplitude used by the CNC selection function
(`flamingo.cnc.SZScaling`, a 1-1 reparametrisation of the same A10 profile):

    A_SZ     = -4.094622      (log10 amplitude)
    alpha_SZ =  1.120000      (= 2/3 + 0.12 + 1/3)

For reference the FLAMINGO default `B = 1.35` gives `A_SZ = -4.051178` with the
same `alpha_SZ`.
