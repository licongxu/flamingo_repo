# Notebook 14 — SNR(q)-Masked 1-Halo tSZ Power Spectrum: Map vs Catalogue Sum vs Theory

**Source:** `notebooks/14_catalogue_tsz_power_spectrum.ipynb` (theory curves added from the
`data/nb09_tsz_theory_ps.npz` cache).
**Inputs:**
- `data/nb09_tsz_map_ps.npz` — NaMaster mask-decoupled map bandpowers (full sky + SNR cuts),
- `data/nb09_tsz_theory_ps.npz` — hmfast completeness-masked halo-model theory (full sky + SNR cuts),
- the `y0q` Arnaud-B=1 catalogue (`Y_5R500c_Mpc2`, `R_500c_Mpc`, `theta500_arcmin`, SNR `q`).

**Figure (1):** `nb14_snr_masked_tsz_ps` (`.pdf` vector + `.png` 300 dpi) with a sidecar
`nb14_manifest.json`. This is the SNR-masked completion of notebook 06.

---

## 1. What this figure shows

The thermal-SZ angular power spectrum after **masking clusters above a detection SNR threshold**
`q`, presented for the full sky and for survivor cuts `q < 50, 20, 10, 5`. Each `q`-cut is one
viridis colour; within a colour the line style encodes the quantity:

| Style | Quantity |
|---|---|
| open circles | **masked map** bandpowers (NaMaster, mode-coupling deconvolved, pixel-window corrected) |
| solid line | **catalogue discrete 1-halo sum** over the surviving halos (GNFW `B=1`) |
| dashed line | **hmfast halo-model theory**, completeness-masked at the same `q` |
| dotted line | **white plateau** `D_ell = ell(ell+1) C_white / 2pi`, the `ell -> 0` Poisson level |

The lower panel shows `model / map` for both the catalogue sum (solid) and the theory (dashed).

## 2. The three predictions

**Map (circles).** A `5 theta_500` disc is zeroed around every detected cluster (`q > q_cut`), the
mask is apodized (0.5 deg) and the bandpowers are estimated with NaMaster, then divided by the
HEALPix pixel window. `f_sky` falls from 0.999 (`q>50`) to 0.840 (`q>5`).

**Catalogue discrete 1-halo sum (solid).** The model-free summation over surviving halos
(`q < q_cut`):

```
C_ell^{1h} = (1 / 4pi) * sum_{i in surv} | y_ell(M_i, z_i) |^2 ,
y_ell = Y_ang,i * Ghat(ell theta_500,i) ,   Y_ang = Y_5R500c_Mpc2 / D_A^2  [sr],
```

with `Ghat(u)` the normalised 3D sinc transform of the fixed Arnaud GNFW shape inside `5 R_500c`.
As `ell -> 0`, `Ghat -> 1` and `C_ell -> C_white = sum Y_ang^2 / 4pi` (the dotted plateau).

**Halo-model theory (dashed).** The hmfast 1h+2h prediction (Tinker08 HMF, A10 GNFW `B=1`) with a
**completeness mask in `(M, z)`**: each halo is down-weighted by its probability of being *below*
threshold given the log-normal `Y-M` scatter (`sigma_lnY = 0.173`). This is a smooth probabilistic
selection, in contrast to the catalogue's hard `q < q_cut` survivor cut.

## 3. Key results

- **Full sky:** the catalogue 1-halo sum tracks the map to `median(cat/map) = 1.028` over
  `400 < ell < 6000`. The low-`ell` deficit (ratio `< 1` below `ell ~ 300`) is the **omitted 2-halo
  clustering term** in the discrete sum; the halo-model theory (which includes 2-halo) closes it.
- **Masking lowers power monotonically** with the threshold: removing the brightest clusters first
  (`q>50`) barely changes the spectrum, while `q>5` removes the most power. The white plateau drops
  by a factor ~23 from full sky (`C_white = 2.94e-16`) to `q<5` (`1.26e-17`), since the few highest-`q`
  objects carry most of `sum Y_ang^2`.
- **Theory vs catalogue at strong cuts:** the completeness theory removes slightly *more* power than
  the hard survivor cut at high `ell` (its probabilistic weighting also suppresses near-threshold
  objects), so the dashed theory sits a touch below the solid catalogue sum for `q<10, q<5`. Both
  bracket the masked map.

## 4. Reproduce

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
python <gen script>   # reads the two data/ caches; no map re-read, no GPU needed
```

The expensive map masking + NaMaster decoupling and the GPU theory are pre-computed into
`data/nb09_tsz_map_ps.npz` and `data/nb09_tsz_theory_ps.npz`; this figure is a cheap re-plot plus
the catalogue summation. Scalars (white plateau per cut, cat/map ratio) are in `nb14_manifest.json`.
