# flamingo

General-purpose tools for processing FLAMINGO lightcone products: HEALPix maps,
catalogue handling, cluster masking, GNFW pressure profiles, and power spectra
(measured and halo-model theory).

This branch (`source-code`) holds only the backbone package; the analysis
notebooks and task-specific scripts live on other branches.

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
