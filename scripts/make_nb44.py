"""One-off: assemble notebooks/44_resolved_tsz_ps_analytic_bg.ipynb."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# 44 - Resolved tSZ power spectrum from apodized Fourier stamps with a calibrated analytic background

Successor to notebook 41 (which is preserved unchanged). Same target: reproduce the
**resolved** (q>5) tSZ power spectrum of the L1_m9 fiducial map as a direct sum over
per-cluster Fourier profiles. Changes to the estimator, computed by
`scripts/export_L1_m9_resolved_stack_yell_apod.py`:

1. **Apodized aperture (mask) and patch.** The hard $5\\,\\theta_{500}$ disc of nb41 is
   replaced by $W_i(r) = 1$ for $r \\le 5\\,\\theta_{500}$ with a $\\cos^2$ taper to zero at
   $6\\,\\theta_{500}$. The tapered stamp is zero-padded into a $2048^2$ patch
   ($15.36^\\circ$ at $0.45'$), so the FFT never sees a hard edge on either the mask or
   the patch boundary.
2. **Analytic background (MASTER in forward mode).** The expected background pseudo-power
   under each cluster's window is predicted from the masked-sky spectrum,
   $$B_i(\\ell) = \\int \\frac{d^2\\ell'}{(2\\pi)^2}\\, |\\tilde W_i(\\ell-\\ell')|^2\\, C^{\\rm bg}(\\ell'),$$
   evaluated exactly on the stamp FFT grid by circular convolution. $C^{\\rm bg}$ is the
   decoupled NaMaster $q>5$ masked bandpower, power-law extrapolated beyond
   $\\ell = 5957$ and re-multiplied by the HEALPix pixel window.
3. **DC-offset term.** The stamps subtract the global monopole, but the cluster-free sky
   has a lower mean: $\\delta = \\bar y_{\\rm masked} - \\bar y = -4.25\\times10^{-8}$
   (`masked_mean_offset.json`). Every background stamp therefore carries a deterministic
   $\\delta^2\\,|\\tilde W_i(\\ell)|^2$ term, important at low $\\ell$ where
   $|\\tilde W_i|^2$ has support.
4. **Transfer calibration.** The convolution term relies on modeling the pixelization
   and resampling chain ($p_\\ell^2$, gnomonic regridding, bandpower extrapolation); this
   does not cancel as it does for the nb41 random-aperture subtraction. A single smooth
   transfer $\\hat T(\\ell)$ is calibrated MASTER-style on **half** of a shared 3600-stamp
   random-aperture ensemble and validated on the other half (Fig. 1). The calibrated
   background is $\\hat B_i(\\ell) = \\hat T(\\ell)\\, B_i(\\ell) + \\delta^2 |\\tilde W_i(\\ell)|^2$
   and the estimator
   $$\\hat C_\\ell = \\frac{1}{4\\pi\\, p_\\ell^2} \\sum_i \\left[\\langle|\\tilde y_i(\\ell)|^2\\rangle_\\varphi - \\hat B_i(\\ell)\\right].$$

Unlike nb41 (2-3 random apertures **per cluster**, noisy), the background here is
deterministic per cluster; the only stochastic input is the shared ensemble entering the
smooth $\\hat T$, whose noise is at the per-cent level after smoothing.

**Validation chain** (one claim, one artifact):

* GRF self-test (`--selftest`): 40 Gaussian skies with the background spectrum, windowed
  and compared to the prediction; median sim/pred = **0.998** over $300<\\ell<6000$
  (rerun with `python scripts/export_L1_m9_resolved_stack_yell_apod.py --selftest`).
* Split-ensemble validation (Fig. 1): $\\hat T$ fitted on even-index stamps, tested on
  odd-index stamps.
* Estimator comparison (Fig. 2): direct sum vs the resolved reference
  (full $-$ masked NaMaster) and vs the nb41 random-aperture estimator.

Method context: forward-mode MASTER (Hivon et al. 2002); random-point subtraction as
variance reduction (Singh et al. 2017); noise-bias-free split estimators
(Madhavacheril et al. 2020) motivate the roadmap for the faint-strata nb42 successor."""
))

cells.append(nbf.v4.new_code_cell(
"""import json
from pathlib import Path

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

REPO = Path("/scratch/scratch-lxu/flamingo_repo")
BP_NPZ = REPO / "data/bandpowers_L1_m9_feedback/masked_tsz_ps.npz"
APOD_NPZ = REPO / "data/nb44_resolved_stack_yell_apod/L1_m9_qgt5_order0_apod.npz"
ENS_NPZ = REPO / "data/nb44_resolved_stack_yell_apod/L1_m9_qgt5_order0_ensemble300.npz"
DC_JSON = REPO / "data/nb44_resolved_stack_yell_apod/masked_mean_offset.json"
OLD_NPZ = REPO / "data/nb41_resolved_stack_yell/L1_m9_qgt5_order0.npz"
OLD_RAND = REPO / "data/nb41_resolved_stack_yell/L1_m9_qgt5_order0_randoms3.npz"
FIG_DIR = REPO / "figures/nb44_resolved_tsz_ps_analytic_bg"
FIG_DIR.mkdir(parents=True, exist_ok=True)
NSIDE = 4096
ELL_MIN = 100

dc = json.loads(DC_JSON.read_text())
DELTA = dc["delta"]
print(f"masked-sky mean offset delta = {DELTA:.3e} (f_sky = {dc['f_sky_binary']:.4f})")
print(json.dumps(json.loads(APOD_NPZ.with_suffix(".json").read_text()), indent=2))"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 1. Transfer calibration and split-ensemble validation

3600 random apertures (12 $\\theta_{500}$ quantile nodes $\\times$ 300 centres outside the
binary $q>5$ discs), identical apodized-window pipeline. The DC term
$\\delta^2 |\\tilde W|^2$ is subtracted from the measurement, and the pooled ratio to the
convolution prediction over the **even-index half** defines $\\hat T(\\ell)$
(rebinned $\\Delta\\ell = 100$, Savitzky-Golay smoothed in $\\log\\ell$). The
**odd-index half** then tests the fully calibrated background. The background $y$ field
is heavy-tailed, so means with standard errors are the estimator-relevant statistics."""
))

cells.append(nbf.v4.new_code_cell(
"""ens = np.load(ENS_NPZ)
ell_st = ens["ell_b"]
theta = ens["theta500_arcmin"]
nodes = np.unique(theta)
n_ens = len(theta)

bg_dc_ens = DELTA**2 * ens["w2"]
meas_corr = ens["abs_y2"] - bg_dc_ens  # measurement minus deterministic DC term

REBIN = 4  # 25 -> 100 wide ell bins
def rebin(a):
    n = a.shape[-1] // REBIN * REBIN
    return a[..., :n].reshape(*a.shape[:-1], -1, REBIN).mean(axis=-1)

ell_rb = rebin(ell_st)
fit = np.arange(n_ens) % 2 == 0  # even half: fit; odd half: validate

T_rb = rebin(meas_corr[fit]).sum(axis=0) / rebin(ens["bg_pred"][fit]).sum(axis=0)
# smooth in log-ell so the low-ell rise is not underfit by a linear-ell window
good = ell_rb > 0
logl = np.geomspace(ell_rb[good][0], ell_rb[good][-1], 120)
T_logl = np.interp(logl, ell_rb[good], T_rb[good])
T_sm_logl = savgol_filter(T_logl, 21, 2)
T_sm = np.interp(ell_rb, logl, T_sm_logl)
T_hat = np.interp(ell_st, logl, T_sm_logl)

def bg_cal(bg_pred, w2):
    return T_hat * bg_pred + DELTA**2 * w2

sel_rb = (ell_rb > ELL_MIN) & (ell_rb < 6000)
val = ~fit
r_val = rebin(ens["abs_y2"][val]).sum(axis=0) / rebin(bg_cal(ens["bg_pred"], ens["w2"])[val]).sum(axis=0)
r_raw = rebin(ens["abs_y2"][val]).sum(axis=0) / rebin(ens["bg_pred"][val]).sum(axis=0)
print(f"validation half, pooled meas/calibrated ({ELL_MIN}<ell<6000): "
      f"mean = {r_val[sel_rb].mean():.3f}, median = {np.median(r_val[sel_rb]):.3f}")
print(f"  same without calibration (raw convolution): mean = {r_raw[sel_rb].mean():.3f}")
node_means = []
for node in nodes:
    m = val & (theta == node)
    rr = rebin(ens["abs_y2"][m]).mean(axis=0) / rebin(bg_cal(ens["bg_pred"], ens["w2"])[m][0])
    node_means.append(rr)
    print(f"  theta500 = {node:6.2f}': mean ratio = {rr[sel_rb].mean():.3f}")
node_means = np.array(node_means)"""
))

cells.append(nbf.v4.new_code_cell(
"""fig, (ax, axr) = plt.subplots(
    2, 1, figsize=(7.2, 7.0), sharex=True,
    gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.06},
)

ax.semilogx(ell_rb[sel_rb], T_rb[sel_rb], "o", ms=3.5, color="0.6",
            label="pooled ratio, fit half (1800 stamps)")
ax.semilogx(ell_rb[sel_rb], T_sm[sel_rb], color="C3", lw=2.0,
            label=r"smoothed transfer $\\hat T(\\ell)$")
ax.axhline(1.0, color="k", lw=0.8)
ax.set_ylabel(r"(measured $-$ DC) / convolution $B(\\ell)$")
ax.set_title("Transfer calibration (fit half) and validation (odd half)")
ax.legend(loc="upper right", frameon=True, fontsize=8)
ax.grid(True, which="both", alpha=0.25)

for i in range(len(nodes)):
    axr.semilogx(ell_rb[sel_rb], node_means[i][sel_rb], color="0.75", lw=0.7,
                 label=r"per-$\\theta_{500}$ node (150 stamps)" if i == 0 else None)
axr.semilogx(ell_rb[sel_rb], r_val[sel_rb], color="C0", lw=2.0,
             label="validation half, pooled (1800 stamps)")
axr.semilogx(ell_rb[sel_rb], r_raw[sel_rb], color="C1", lw=1.2, ls="--",
             label="validation half, uncalibrated")
axr.axhline(1.0, color="k", lw=0.8)
axr.axhspan(0.95, 1.05, color="0.9", zorder=0)
axr.set_ylim(0.5, 1.8)
axr.set_xlim(ELL_MIN, 6000)
axr.set_xlabel(r"multipole $\\ell$")
axr.set_ylabel("measured / calibrated bg")
axr.legend(loc="upper right", frameon=True, fontsize=8)
axr.grid(True, which="both", alpha=0.25)

for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"ensemble_vs_analytic_bg.{ext}", bbox_inches="tight")
plt.show()"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 2. Resolved power spectrum: calibrated-background direct sum vs reference

The estimator $\\hat D_\\ell$ (direct sum with per-cluster calibrated analytic background)
is compared to the resolved reference (full $-$ masked NaMaster bandpowers) and to the
nb41 estimator (hard aperture, 3 random apertures per cluster)."""
))

cells.append(nbf.v4.new_code_cell(
"""st = np.load(APOD_NPZ)
ell_st = st["ell_b"]
n_cl = st["abs_y2"].shape[0]

pw = hp.pixwin(NSIDE, lmax=int(ell_st[-1]) + 20)
pw_st = np.interp(ell_st, np.arange(pw.size), pw)

bg_i = bg_cal(st["bg_pred"], st["w2"])
fac = ell_st * (ell_st + 1.0) / (2.0 * np.pi)
dl_new = fac * (st["abs_y2"] - bg_i).sum(axis=0) / (4.0 * np.pi) / pw_st**2
dl_raw = fac * st["abs_y2"].sum(axis=0) / (4.0 * np.pi) / pw_st**2

old = np.load(OLD_NPZ)
old_r = np.load(OLD_RAND)
ell_old = old["ell_b"]
pw_old = np.interp(ell_old, np.arange(pw.size), pw)
fac_old = ell_old * (ell_old + 1.0) / (2.0 * np.pi)
dl_old = fac_old * (old["abs_y2"] - old_r["abs_y2"]).sum(axis=0) / (4.0 * np.pi) / pw_old**2

bp = np.load(BP_NPZ, allow_pickle=True)
variants = [str(v) for v in bp["variants"]]
ifid = variants.index("L1_m9")
icut = list(bp["q_cuts"]).index(5.0)
ellb_nmt = bp["ellb"]
dl_full = bp["dl_fullsky"][ifid]
dl_masked = bp["dl_masked"][ifid, icut]
dl_resolved = dl_full - dl_masked
sel_nmt = (ellb_nmt >= ELL_MIN) & (ellb_nmt <= 6000)

ell_n = ellb_nmt[sel_nmt]
dl_new_n = np.interp(ell_n, ell_st, dl_new)
dl_old_n = np.interp(ell_n, ell_old, dl_old)
dl_res_n = dl_resolved[sel_nmt]

for lo, hi in ((300, 1000), (1000, 3000), (3000, 6000)):
    b = (ell_n >= lo) & (ell_n < hi)
    print(f"ell {lo}-{hi}: median new/res = {np.median(dl_new_n[b]/dl_res_n[b]):.3f}, "
          f"old/res = {np.median(dl_old_n[b]/dl_res_n[b]):.3f}")"""
))

cells.append(nbf.v4.new_code_cell(
"""fig, (ax, axr) = plt.subplots(
    2, 1, figsize=(7.2, 7.4), sharex=True,
    gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.06},
)

ax.loglog(ellb_nmt[sel_nmt], dl_full[sel_nmt], color="0.6", lw=1.4, label="full sky (NaMaster)")
ax.loglog(ellb_nmt[sel_nmt], dl_masked[sel_nmt], color="0.6", lw=1.4, ls=":",
          label=r"masked $q>5$ (NaMaster)")
ax.loglog(ell_n, dl_res_n, "o", ms=4, mfc="none", color="k",
          label=r"resolved $=$ full $-$ masked")
ax.loglog(ell_st, dl_new, color="C1", lw=2.0,
          label="direct sum, apodized + calibrated analytic bg (this nb)")
ax.loglog(ell_old, dl_old, color="C0", lw=1.4, ls="--",
          label="direct sum, nb41 (hard aperture, 3 randoms)")
ax.loglog(ell_st, dl_raw, color="C1", lw=1.0, ls=":", alpha=0.7, label="raw (no bg subtraction)")
ax.set_ylabel(r"$D_\\ell = \\ell(\\ell+1)C_\\ell/2\\pi$  [$y^2$]")
ax.set_title(f"L1_m9 fiducial: resolved tSZ power ($q>5$, {n_cl} clusters)")
ax.legend(loc="lower center", frameon=True, fontsize=8)
ax.grid(True, which="both", alpha=0.25)

axr.axhline(1.0, color="k", lw=0.8)
axr.axhspan(0.95, 1.05, color="0.9", zorder=0)
axr.semilogx(ell_n, dl_new_n / dl_res_n, color="C1", lw=1.8,
             label="apodized + calibrated bg / resolved")
axr.semilogx(ell_n, dl_old_n / dl_res_n, color="C0", lw=1.2, ls="--", label="nb41 / resolved")
axr.set_xlim(ELL_MIN, 6000)
axr.set_ylim(0.5, 1.5)
axr.set_xlabel(r"multipole $\\ell$")
axr.set_ylabel("ratio")
axr.legend(loc="upper left", frameon=True, fontsize=8)
axr.grid(True, which="both", alpha=0.25)

for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"resolved_ps_analytic_bg.{ext}", bbox_inches="tight")
plt.show()"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Conclusions

* The raw MASTER-forward convolution alone is **not** sufficient: the measured background
  exceeds it by up to $+75\\%$ at $\\ell \\sim 150$ (resolved-wing leakage into the random
  apertures, sub-bandpower modes) and falls $20\\%$ below it at $\\ell \\sim 6000$
  (pixelization/resampling transfer, bandpower extrapolation); nb41's random apertures
  absorbed all of this implicitly.
* With the deterministic $\\delta^2|\\tilde W|^2$ term and one smooth shared
  $\\hat T(\\ell)$ (fit on 1800 stamps), the calibrated background is unbiased on the
  independent validation half: pooled mean $0.991$, median $1.003$ over
  $100 < \\ell < 6000$ (uncalibrated: $1.046$ with a strong $\\ell$ trend), while
  remaining noiseless per cluster.
* Estimator accuracy vs the resolved reference (median ratio): $1.06 / 0.96 / 0.89$ in
  $\\ell = 300$-$10^3 / 10^3$-$3\\times10^3 / 3\\times10^3$-$6\\times10^3$, matching nb41
  ($1.04 / 0.98 / 0.90$) with the per-cluster random-aperture noise removed. The
  remaining deviations are shared physics/conventions: the $5$-$6\\,\\theta_{500}$ taper
  collects extra outskirt/neighbour power at low $\\ell$ (and the reference is
  convention-ambiguous there by the NaMaster apodization ring); at high $\\ell$ the
  background is $85\\%$ of the total stamp power, so even a $1\\%$ background error moves
  the difference by $\\sim6\\%$.
* Per-cluster determinism is the operational win: subsets, mass/redshift strata, or
  single-cluster profiles can now be background-subtracted without re-measuring randoms,
  which is what the faint-strata nb42 successor needs.

Provenance: `.json` manifests next to the `.npz` inputs (git hash, config, runtimes),
`masked_mean_offset.json` for $\\delta$, and `runs/logbook.md` for the run record."""
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
out = Path(__file__).resolve().parents[1] / "notebooks/44_resolved_tsz_ps_analytic_bg.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
