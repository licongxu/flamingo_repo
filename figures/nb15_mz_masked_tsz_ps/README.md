# Notebook 15 — (M_500c, z)-Masked 1-Halo tSZ Power Spectrum: Map vs Catalogue Sum vs Theory

**Source:** `notebooks/15_mz_masked_tsz_power_spectrum.ipynb` (halo-model theory added, computed on
GPU and cached to `data/nb15_mz_theory_ps.npz`).
**Inputs:**
- `data/nb15_mz_masked_ps.npz` — NaMaster mask-decoupled map bandpowers for each (M,z) cut,
- `data/nb15_mz_theory_ps.npz` — hmfast halo-model theory with the matching (M,z) top-hat mask,
- the `y0q` Arnaud-B=1 catalogue (`M_500c_Msun`, `z`, `Y_5R500c_Mpc2`, `R_500c_Mpc`, `theta500_arcmin`).

**Figure (1):** `nb15_mz_masked_tsz_ps` (`.pdf` vector + `.png` 300 dpi) with a sidecar
`nb15_manifest.json`. This is the (M,z)-masked completion of notebook 07.

---

## 1. What this figure shows

The same map vs catalogue-sum vs theory comparison as notebook 14, but the mask is now a **direct
top-hat in `(M_500c, z)`** instead of an SNR threshold. Two progressions, one per column:

- **Left, mask high mass:** remove `M_500c > M_cut` for `M_cut = 1e15, 5e14, 2e14 Msun`. Survivors are
  the lower-mass halos.
- **Right, mask low redshift:** remove `z < z_cut` for `z_cut = 0.01, 0.05, 0.1, 0.25`. Survivors are
  the more distant halos.

Each cut is one viridis colour; line style encodes the quantity (shared legend along the bottom):

| Style | Quantity |
|---|---|
| open circles | **masked map** bandpowers (NaMaster, pixel-window corrected) |
| solid line | **catalogue discrete 1-halo sum** over survivors (GNFW `B=1`) |
| dashed line | **hmfast halo-model theory** with the same `(M,z)` top-hat mask (1h+2h) |
| dotted line | **white plateau** `ell -> 0` (`sum_surv Y_ang^2 / 4pi`) |

The lower panels show `model / map` (catalogue: solid, theory: dashed).

## 2. The three predictions

**Map (circles).** A `5 theta_500` disc is zeroed around every masked cluster, the mask is apodized
(0.5 deg), and bandpowers are estimated with NaMaster, then divided by the HEALPix pixel window.

**Catalogue discrete 1-halo sum (solid).** Identical to notebook 14:
`C_ell^{1h} = (1/4pi) sum_{i in surv} |Y_ang,i Ghat(ell theta_500,i)|^2`, summed only over halos
*outside* the masked corner. As `ell -> 0` this tends to the white plateau `C_white`.

**Halo-model theory (dashed).** hmfast 1h+2h (Tinker08 HMF, A10 GNFW `B=1`) with `cl_*_masked` applied
on the native `(M, z)` grid: native masses are converted to `M_500c` and the integrand is zeroed
inside the same top-hat (`M_500c > M_cut`, or `z < z_cut`). Unlike the SNR case there is no
completeness scatter, so the theory mask is the exact `(M,z)` analogue of the catalogue survivor cut.

## 3. Key results

- **Mass masking removes intermediate/high-`ell` power sharply.** The massive clusters are rare (so
  little sky is lost) but carry the largest `Y_ang`, so dropping `M_500c > 2e14` cuts `D_ell` near the
  peak by more than half (`D_ell(3000)`: `1.59e-12 -> 6.3e-13` in theory). The catalogue/map ratio
  rises with the cut (`1.03 -> 1.19` for `M>2e14`), reflecting the growing relative size of the
  omitted 2-halo term once the dominant 1-halo objects are gone; the theory (which has 2-halo) sits
  closer to the map.
- **Redshift masking lowers the whole spectrum**, especially the lowest multipoles sourced by the
  largest-angular-size (nearby) clusters. The `z<0.25` cut brings `cat/map = 0.96` and visibly
  suppresses the low-`ell` end.
- **Theory tracks the masked map across `ell`.** Because the dashed theory includes the 2-halo
  clustering term, it removes the low-`ell` deficit that the 1-halo-only catalogue sum shows
  (ratio `< 1` below `ell ~ 300`), and both bracket the map at high `ell`.
- The white plateau falls monotonically with each cut (full sky `C_white = 2.94e-16`; `z<0.25`
  reaches `6.1e-18`), confirming the masks remove the expected `sum Y_ang^2` shot-noise level.

## 4. Reproduce

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
python <gen script>   # reads the map cache; computes (or loads cached) GPU theory
```

The map masking + NaMaster decoupling is cached in `data/nb15_mz_masked_ps.npz`. The halo-model
theory for each `(M,z)` cut is computed once on GPU and cached in `data/nb15_mz_theory_ps.npz`
(delete to force a recompute). Per-cut white-plateau and cat/map medians are in `nb15_manifest.json`.
