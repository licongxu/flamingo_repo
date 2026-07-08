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
"""## 3. nb42 redo: L2p8 rotation groups with the calibrated analytic background

The nb42 analysis (five L2p8_m9 lc0 rotation-group maps, mass-stratified subsamples of
the deep $M_{500c} > 10^{13}\\,M_\\odot$ catalogue, hybrid stratified estimator) is
repeated with the nb44 machinery, produced by
`scripts/export_L2p8_rotgroup_stack_yell_apod.py` on the **identical** nb42 subsamples:

* apodized windows and patches (as in Part 1-2);
* per-cluster analytic background from each group map's own measured $C_\\ell$
  (re-pixel-windowed, extrapolated), calibrated by a per-group transfer
  $\\hat T_g(\\ell)$ from a 1200-stamp uniform-sphere random ensemble
  (split-half validated). Randoms are uniform and each map subtracts its own
  monopole, so the L1_m9 DC-offset term is zero here by construction;
* the nb42 hybrid estimator is unchanged except that the incoherent branch uses
  $|\\tilde y_i|^2 - \\hat T_g B_i$ instead of the 2-random-aperture subtraction;
  the coherent faint-strata branch (phase-centred stack, variance debias,
  $R(u)$ calibration) is identical."""
))

cells.append(nbf.v4.new_code_cell(
"""NB42_DIR = REPO / "data/nb42_rotgroup_stack_yell"
NB44G_DIR = REPO / "data/nb44_rotgroup_stack_yell_apod"
N_GROUPS = 5
SNR_MIN = 5.0

def log_bin(ell, dl, lmin=10, lmax=6000, nbin=30):
    edges = np.logspace(np.log10(lmin), np.log10(lmax), nbin + 1)
    idx = np.digitize(ell, edges) - 1
    lb, db = [], []
    for b in range(nbin):
        sel = (idx == b) & (ell >= lmin)
        if sel.any():
            lb.append(ell[sel].mean())
            db.append(dl[sel].mean())
    return np.array(lb), np.array(db)

def calibrate_R(u_samples, R_samples, u_eval):
    lo, hi = np.log10(2e-4), np.log10(150.0)
    edges = np.linspace(lo, hi, 36)
    lu = np.log10(np.clip(u_samples, 10**lo, 10**hi))
    idx = np.digitize(lu, edges) - 1
    cent, med = [], []
    for b in range(edges.size - 1):
        m = idx == b
        if m.sum() >= 5:
            cent.append(0.5 * (edges[b] + edges[b + 1]))
            med.append(np.median(R_samples[m]))
    cent, med = np.array(cent), np.array(med)
    R = np.interp(np.log10(np.clip(u_eval, 10**lo, 10**hi)), cent, med)
    return np.clip(R, 1.0, 6.0)

def fit_transfer(ens):
    \"\"\"Split-half validated smooth T-hat(ell); returns (T_hat, val_mean).\"\"\"
    n = ens["abs_y2"].shape[0]
    half = np.arange(n) % 2 == 0
    T_rb = rebin(ens["abs_y2"][half]).sum(axis=0) / rebin(ens["bg_pred"][half]).sum(axis=0)
    good = ell_rb > 0
    lg = np.geomspace(ell_rb[good][0], ell_rb[good][-1], 120)
    T_sm = savgol_filter(np.interp(lg, ell_rb[good], T_rb[good]), 21, 2)
    T_half = np.interp(ens["ell_b"], lg, T_sm)
    r_val = (rebin(ens["abs_y2"][~half]).sum(axis=0)
             / rebin(T_half * ens["bg_pred"])[~half].sum(axis=0))
    val_mean = r_val[(ell_rb > 100) & (ell_rb < 6000)].mean()
    # final transfer from the full ensemble
    T_rb = rebin(ens["abs_y2"]).sum(axis=0) / rebin(ens["bg_pred"]).sum(axis=0)
    T_sm = savgol_filter(np.interp(lg, ell_rb[good], T_rb[good]), 21, 2)
    return np.interp(ens["ell_b"], lg, T_sm), val_mean

def hybrid(st, sub):
    \"\"\"nb42 hybrid stratified estimator; `sub` = bg-subtracted incoherent power.\"\"\"
    ell_g = st["ell_b"]
    nell = ell_g.size
    pw_g = np.interp(ell_g, np.arange(pw.size), pw)
    norm = 1.0 / (4.0 * np.pi) / pw_g**2
    mbin = st["mbin"]
    N_b = st["N_bin"].astype(np.float64)
    n_arr = st["n_bin"].astype(np.float64)
    nbin_m = N_b.size
    re = st["re_y"]
    theta500_rad = st["theta500_arcmin"] / (180.0 * 60.0 / np.pi)

    T = np.zeros((nbin_m, nell)); sigT = np.full((nbin_m, nell), np.inf)
    P = np.zeros((nbin_m, nell)); sigP = np.full((nbin_m, nell), np.inf)
    theta_med = np.zeros(nbin_m)
    for b in range(nbin_m):
        m = mbin == b
        n = int(m.sum())
        if n == 0:
            continue
        theta_med[b] = np.median(theta500_rad[m])
        T[b] = N_b[b] * sub[m].mean(axis=0)
        if n >= 2:
            fpc = max(1.0 - n / N_b[b], 0.0)
            sigT[b] = N_b[b] * sub[m].std(axis=0, ddof=1) / np.sqrt(n) * np.sqrt(fpc)
            sbar = re[m].mean(axis=0)
            v = re[m].var(axis=0, ddof=1)
            P[b] = sbar**2 - v / n
            sigP[b] = np.sqrt(4.0 * sbar**2 * v / n + 2.0 * v**2 / (n * (n - 1)))

    band = (ell_g >= 300) & (ell_g <= 4000)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.nanmedian(np.where(sigT > 0, T / sigT, np.inf), axis=1)
    reliable = (np.isinf(snr) | (snr >= SNR_MIN)) & (n_arr > 0)
    u_s, R_s = [], []
    for b in np.where(reliable & (n_arr >= 2))[0]:
        ok = band & (P[b] > 0) & (T[b] > 0)
        u_s.append(ell_g[ok] * theta_med[b])
        R_s.append(T[b, ok] / (N_b[b] * P[b, ok]))
    u_s, R_s = np.concatenate(u_s), np.concatenate(R_s)

    contrib = np.zeros((nbin_m, nell)); err2 = np.zeros(nell)
    for b in range(nbin_m):
        if n_arr[b] == 0:
            continue
        if reliable[b]:
            contrib[b] = T[b]
            e = np.where(np.isfinite(sigT[b]), sigT[b], 0.0)
        else:
            Rb = calibrate_R(u_s, R_s, ell_g * theta_med[b])
            contrib[b] = N_b[b] * np.clip(P[b], 0.0, None) * Rb
            e = N_b[b] * sigP[b] * Rb
        err2 += e**2
    return contrib.sum(axis=0) * norm, np.sqrt(err2) * norm

groups = []
for g in range(N_GROUPS):
    st = np.load(NB44G_DIR / f"group{g}_apod.npz")
    ens = np.load(NB44G_DIR / f"group{g}_ensemble100.npz")
    old = np.load(NB42_DIR / f"group{g}_order0.npz")
    md = np.load(NB42_DIR / f"measured_dl_group{g}.npz")
    T_hat_g, val_mean = fit_transfer(ens)
    cl_new, cl_new_err = hybrid(st, st["abs_y2"] - T_hat_g * st["bg_pred"])
    cl_old, _ = hybrid(old, old["abs_y2"] - old["bg_y2"])
    ell_g = st["ell_b"]
    fac_g = ell_g * (ell_g + 1.0) / (2.0 * np.pi)
    ellb, dlb = log_bin(md["ell"], md["dl"])
    groups.append(dict(
        g=g, zlo=float(st["z_inner"]), zhi=float(st["z_outer"]), ell=ell_g,
        dl_new=fac_g * cl_new, dl_new_err=fac_g * cl_new_err, dl_old=fac_g * cl_old,
        ellb=ellb, dlb=dlb, val=val_mean,
    ))
    b = (ellb >= 300) & (ellb <= 4000)
    rn = np.interp(ellb, ell_g, fac_g * cl_new) / dlb
    ro = np.interp(ellb, ell_g, fac_g * cl_old) / dlb
    print(f"group {g} (z {st['z_inner']:.2f}-{st['z_outer']:.2f}): T-hat val mean = {val_mean:.3f}; "
          f"median ratio 300<ell<4000: new = {np.median(rn[b]):.3f}, nb42 = {np.median(ro[b]):.3f}")"""
))

cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.4), sharex=True, sharey=True)
for k, gr in enumerate(groups):
    ax = axes.ravel()[k]
    ax.loglog(gr["ellb"], gr["dlb"], "o", ms=4, mfc="none", color="k", label="measured map")
    sel = (gr["ell"] > 80) & (gr["ell"] < 6000)
    ax.loglog(gr["ell"][sel], gr["dl_new"][sel], color="C1", lw=1.8,
              label="hybrid stack, calibrated analytic bg")
    ax.fill_between(gr["ell"][sel], (gr["dl_new"] - gr["dl_new_err"])[sel],
                    (gr["dl_new"] + gr["dl_new_err"])[sel], color="C1", alpha=0.3, lw=0)
    ax.loglog(gr["ell"][sel], gr["dl_old"][sel], color="C0", lw=1.2, ls="--",
              label="hybrid stack, nb42 (2 randoms)")
    ax.set_title(f"group {gr['g']}:  $z = {gr['zlo']:.2f}$-${gr['zhi']:.2f}$", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(80, 6000)
    ax.set_ylim(1e-16, 3e-13)
axes.ravel()[5].axis("off")
h, l = axes.ravel()[0].get_legend_handles_labels()
axes.ravel()[5].legend(h, l, loc="center", frameon=True, fontsize=10)
for ax in axes[1]:
    ax.set_xlabel(r"multipole $\\ell$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$D_\\ell$  [$y^2$]")
fig.suptitle("L2p8_m9 rotation groups: stratified stack vs measured map power", y=0.98)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"rotgroup_spectra_analytic_bg.{ext}", bbox_inches="tight")
plt.show()"""
))

cells.append(nbf.v4.new_code_cell(
"""fig, (ax, axo) = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True,
                              gridspec_kw={"hspace": 0.06})
cmap = plt.get_cmap("viridis")
for k, gr in enumerate(groups):
    c = cmap(k / max(N_GROUPS - 1, 1))
    rn = np.interp(gr["ellb"], gr["ell"], gr["dl_new"]) / gr["dlb"]
    en = np.interp(gr["ellb"], gr["ell"], gr["dl_new_err"]) / gr["dlb"]
    ro = np.interp(gr["ellb"], gr["ell"], gr["dl_old"]) / gr["dlb"]
    ax.semilogx(gr["ellb"], rn, color=c, lw=1.8,
                label=f"group {gr['g']} ($z$ {gr['zlo']:.2f}-{gr['zhi']:.2f})")
    ax.fill_between(gr["ellb"], rn - en, rn + en, color=c, alpha=0.25, lw=0)
    axo.semilogx(gr["ellb"], ro, color=c, lw=1.4, ls="--")
for a, t in ((ax, "calibrated analytic bg (this nb)"), (axo, "nb42 (2 random apertures)")):
    a.axhline(1.0, color="k", lw=0.8)
    a.axhspan(0.9, 1.1, color="0.9", zorder=0)
    a.set_ylim(0.4, 1.6)
    a.set_ylabel(f"stack / measured\\n{t}", fontsize=10)
    a.grid(True, which="both", alpha=0.25)
ax.legend(loc="upper left", frameon=True, fontsize=8, ncol=2)
axo.set_xlim(100, 6000)
axo.set_xlabel(r"multipole $\\ell$")
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"rotgroup_ratio_analytic_bg.{ext}", bbox_inches="tight")
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
  single-cluster profiles can now be background-subtracted without re-measuring randoms.
* **Rotation groups (Part 3):** the nb42 hybrid estimator with the calibrated analytic
  background reproduces each group's measured $D_\\ell$ (Figs. 3-4, band medians printed
  above); the incoherent branch no longer carries per-cluster random-aperture noise, and
  the per-group $\\hat T_g$ split-half validation means are printed alongside.

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
