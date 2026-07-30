# flamingo

General-purpose tools for processing FLAMINGO lightcone products: HEALPix maps,
catalogue handling, cluster masking, GNFW pressure profiles, and power spectra
(measured and halo-model theory).

This branch (`paper-results`) is the `source-code` backbone plus a thin,
reproducible pipeline that produces the paper's tSZ power-spectrum results.
See [Paper results](#paper-results) below; the rest of this README documents
the backbone package itself.

## Install

```bash
pip install -e .                 # core (numpy, healpy, pandas, hmfast, jax)
pip install -e ".[powerspectra]" # + NaMaster (pymaster) for mask-decoupled Cl
pip install -e ".[stamps]"       # + pixell for flat-sky stamps and FFT Cl
```

The bundled data lives under `data/`. Point at another copy with
`export FLAMINGO_ROOT=/path/to/tree` (same layout).

## Layout

| Subpackage / module          | Backend        | Contents                                                  |
|------------------------------|----------------|-----------------------------------------------------------|
| `flamingo.maps`              | NumPy / healpy | map I/O, sample at positions, neighbour-max sampling      |
| `flamingo.maps.stamps`       | pixell         | gnomonic (tangent-plane) stamps cut from HEALPix maps     |
| `flamingo.masking`           | NumPy / healpy | binary N×R500 disc masks, sky fraction                    |
| `flamingo.profiles`          | **JAX / GPU**  | GNFW (Arnaud default) + line-of-sight projection          |
| `flamingo.profiles`          | NumPy / healpy | radial / normalized / stacked map profiles                |
| `flamingo.catalogue`         | NumPy / hmfast | SOAP CSV loading, `theta_500`, `E(z)`, `D_A(z)`, rotation check|
| `flamingo.powerspectra`      | healpy         | full-sky `anafast` Cl + linear bandpower binning          |
| `flamingo.powerspectra.namaster` | pymaster   | apodization + mask-decoupled `D_ell` (optional)           |
| `flamingo.powerspectra.flat` | pixell         | flat-sky FFT Cl of a stamp (optional)                     |
| `flamingo.theory`            | hmfast / JAX   | halo-model `C_ell^yy` (1h + 2h), D3A cosmology default    |

The model kernels (`gnfw`, `projected_shape`, hmfast halo model) are JAX,
jittable/`vmap`-able, and enable float64. HEALPix-bound code stays NumPy
because `healpy` pixel queries are CPU-only.

## Quick start

```python
import numpy as np
from flamingo import paths
from flamingo.maps import read_map, sample_neighbour_max
from flamingo.catalogue import load_catalogue, theta_500
from flamingo.masking import disc_mask, fsky
from flamingo.profiles import gnfw, projected_shape

ymap = read_map(paths.HYDRO_MAP)            # NSIDE=4096 Compton-y map
df = load_catalogue(paths.HYDRO_CATALOGUE)  # yang26-rotated SOAP catalogue

th, ph = df["theta_rot_rad"].to_numpy(float), df["phi_rot_rad"].to_numpy(float)
y_max, y_central = sample_neighbour_max(ymap, th, ph)

t500 = theta_500(df["R_500c_Mpc"].to_numpy(float), df["z"].to_numpy(float))
mask = disc_mask(4096, th, ph, 5.0 * t500)  # 5*R500 discs
print("fsky:", fsky(mask))

x = np.linspace(0.01, 5, 100)
p = gnfw(x)                  # Arnaud A10 by default
y = projected_shape(x)       # line-of-sight projection, unity at centre
```

Use the **rotated** columns (`theta_rot_rad`, `phi_rot_rad`): the L2p8 map is in
the yang26-rotated frame.

### Power spectra

```python
# full sky (healpy)
from flamingo.powerspectra import full_sky_cl, bin_cl
ell, cl = full_sky_cl(ymap)
ell_eff, dl, cl_b = bin_cl(ell, cl, delta_ell=30)

# masked (NaMaster)
from flamingo.powerspectra.namaster import apodize, decoupled_dl
apod = apodize(mask, aperture_deg=0.5)
ell_eff, dl, cl = decoupled_dl(ymap, apod, delta_ell=30)

# flat-sky stamp (pixell)
from flamingo.maps.stamps import gnomonic_stamp, zero_outside_aperture
from flamingo.powerspectra.flat import stamp_cl
stamp = gnomonic_stamp(ymap, lon_deg=30.0, lat_deg=45.0,
                       width_rad=np.radians(2.0), res_rad=np.radians(0.005))
ell_eff, cl = stamp_cl(stamp)

# halo-model theory (hmfast; loads emulators on import)
from flamingo.theory import cl_yy, dl_of_cl
th = cl_yy(np.geomspace(10, 6000, 64), B=1.0)
dl = dl_of_cl(th["ell"], th["cl"])
```

## Tests

```bash
pytest -q
```

---

# Paper results

`paper_results/` is the pipeline behind the paper's tSZ power-spectrum figures.
It measures, for the FLAMINGO **L1_m9** lightcone-0 Compton-y maps:

1. the **total** (unmasked) tSZ power spectrum, and the **masked** spectrum
   after excising every cluster a Planck-like survey would detect at
   `q > 50, 20, 10, 5, 1`;
2. the same two spectra for **all nine feedback variants**, so the sensitivity
   of the diffuse tSZ signal to the feedback model can be read off directly.

Every physical operation is a call into `flamingo` (`src/`); the modules below
only select inputs, loop, and save. `paper_results/__init__.py` puts this
checkout's `src/` at the front of `sys.path`, so the pipeline always runs
against the source code on this branch regardless of what is pip-installed.

## Pipeline

| Step | Module | Uses from `flamingo` | Writes |
|------|--------|----------------------|--------|
| 1 | `paper_results.compute_q` | `cnc.SZScaling`, `catalogue.theta_500` | `q` catalogues (cache) |
| 2 | `paper_results.compute_ps` | `masking.disc_mask`, `powerspectra.namaster` | `results/bandpowers/` |
| 3 | `paper_results.figures` | — | `figures/` |

```bash
python -m paper_results.compute_q      # ~1 min, GPU
python -m paper_results.compute_ps     # ~1 h for 9 variants x 6 spectra
python -m paper_results.figures
```

Each step caches its output and skips work already done; pass `--force` to
recompute and `--variant NAME` (repeatable) to restrict to some variants.
All paths, thresholds and estimator settings live in `paper_results/config.py`.

## Step 1 — cluster detection significance

There is no cluster *finder* here: `q` is what the assumed SZ scaling relation
predicts a Planck-like survey would measure, given only `(M_500c, z)`:

$$q = \frac{y_0(M_{500c}/B,\,z)\;e^{\epsilon}}{\sigma_{y_0}(\theta_{500})},
\qquad \epsilon\sim\mathcal N(0,\sigma_{\ln Y}^2)$$

with the Arnaud et al. (2010) self-similar pressure profile, hydrostatic bias
`B = 1.35`, `alpha_SZ = 2/3 + 0.12 + 1/3`, intrinsic scatter
`sigma_lnY = 0.173`, and the Planck-like matched-filter noise curve
`sigma_y0(theta_500)` from szifi (`immf6`). This lives in
[`flamingo.cnc`](src/flamingo/cnc.py). The amplitude `A_SZ` is calibrated
against the analytic A10 profile at build time (`SZScaling.calibrated()`), and
the scatter is drawn deterministically from each halo's SOAP index so runs are
reproducible.

Two angular scales appear and must not be conflated:

* the **true** `theta_500 = R_500c / D_A(z)` from the catalogue sets the
  *masking radius*;
* the **hydrostatic** `theta_500(M_500c/B)` inside the scaling relation sets
  the *matched-filter noise* entering `q`.

## Step 2 — power spectra

Unmasked and masked spectra go through the *same* NaMaster estimator, so they
are directly comparable:

* binary mask: discs of radius `5 x theta_500` around every halo with `q` above
  the cut (`flamingo.masking.disc_mask`);
* C1 apodization, 0.5 deg (`flamingo.powerspectra.namaster.apodize`);
* mask-weighted monopole subtracted, mask-decoupled pseudo-Cl, linear
  `Delta ell = 30` bandpowers to `ell_max = 6000`, HEALPix pixel window
  deconvolved (`flamingo.powerspectra.namaster.decoupled_dl`).

The unmasked case runs through NaMaster with a unit mask rather than through
`anafast`, so both share identical binning and pixel-window handling.

## Outputs

```
results/bandpowers/<variant>.npz              ell, dl_fullsky, dl_masked, n_masked, fsky, settings
results/bandpowers/Dl_yy_<variant>_<tag>.txt  plain "ell  D_ell" columns, tag in {fullsky, qgt50 ... qgt1}
results/counts.md                             N(q > cut) per variant
figures/paper/fiducial_masked_tsz_ps.{pdf,png}      figure 1
figures/paper/feedback_tsz_ps.{pdf,png}             figure 2
```

## Inputs

Large inputs are read from outside the repo and are not version controlled:

| What | Where |
|------|-------|
| L1_m9 y-maps, `nside=4096` | `/rds/rds-lxu/flamingo/L1_m9/maps/y_unlensed_<variant>_lc0_nside4096.fits` |
| L1_m9 halo catalogues, `M_500c > 5e13`, `z < 3` | `/rds/rds-lxu/flamingo/L1_m9/catalogues/` |
| szifi matched-filter noise curves | `$FLAMINGO_ROOT/data/noise/` |

Override with `FLAMINGO_ROOT`, `L1M9_MAP_DIR` and `L1M9_CAT_DIR`.
