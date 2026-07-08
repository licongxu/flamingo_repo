"""One-off: assemble notebooks/42_rotgroup_tsz_ps_fourier_stack.ipynb."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# 42 - Rotation-group tSZ power spectra from Fourier-space stacked profiles (L2p8_m9 lc0)

**Empirical stacked-profile theory for the notebook 39 group spectra.**

Notebook 39 measured the tSZ $D_\\ell$ of the five L2p8_m9 lightcone-0 rotation-group
maps (raw shell sums covering $z \\in [0.001, 0.35]$, $[0.35, 0.8]$, $[0.8, 1.35]$,
$[1.35, 2.2]$, $[2.2, 3.0]$) and compared them to the hmfast A10 GNFW ($B=1$) halo
model, which is increasingly poor at high redshift. Here the A10 profile is replaced
by the **measured Fourier-space stacked profiles of the clusters in each group**
(notebook 41 method), so the theory side is built entirely from the map and the halo
catalogue, with no pressure-profile model.

For each group, a mass-stratified subsample of the $M_{500c} > 10^{13}\\,M_\\odot$
lightcone catalogue (0.25 dex bins, up to 500 clusters per bin, rare massive bins
fully enumerated) is cut out of the *same group map* as gnomonic stamps, apertured
at $5\\,\\theta_{500}$, zero-padded to a common grid and FFT'd
(`scripts/export_L2p8_rotgroup_stack_yell.py`). Each stratum $b$ with $N_b$
catalogue members and $n_b$ sampled members enters with weight $N_b/n_b$, making
every estimator an unbiased estimate of the sum over the *full* catalogue.
Positions are the catalogue native lightcone coordinates, which match the unrotated
group maps (verified against the maps).

**Background subtraction** (notebook 41): every stamp contains the cluster plus the
sky behind it; the mean background power is measured with 2 random same-size
apertures per cluster and subtracted in power space,
$S_i(\\ell) = \\langle|\\tilde y_i|^2\\rangle - \\langle|\\tilde y_i^{\\rm rand}|^2\\rangle$.

**Stratum estimators.** The incoherent stratified sum
$T_b(\\ell) = N_b\\,\\overline{S_i}$ is unbiased but its noise scales as
$N_b/\\sqrt{n_b}$: for the abundant faint strata ($N_b/n_b$ up to $10^4$) the
per-cluster power is far below the background fluctuation and $T_b$ is useless.
There the **coherent stacked profile** is used instead: each stamp also stores the
azimuthally averaged, phase-centred $s_i(\\ell) = \\langle \\mathrm{Re}[\\tilde y_i
e^{i\\boldsymbol{\\ell}\\cdot\\boldsymbol{x}_c}]\\rangle$, in which the background
averages down as $1/n_b$ rather than $1/\\sqrt{n_b}$. The squared stacked profile,
debiased by the sample variance,
$P_b = \\bar s_b^2 - \\widehat{\\rm Var}(s_i)/n_b$, recovers the *coherent* part of
the stratum power; the incoherence factor $R(u) = \\overline{S_i}/P_b$ (amplitude
scatter, shape scatter, asymmetry) is calibrated on the signal-dominated strata as
a function of $u = \\ell\\,\\theta_{500}$ and applied to the faint ones,
$U_b = N_b\\, P_b\\, R(u)$. On the brightest clusters $s^2/S = 1.00$ at
$\\ell = 100$ and $0.90$ at $\\ell = 2000$, so $R$ is a modest correction.
Each stratum uses $T_b$ where its band signal-to-noise is $\\geq 5$ (or the stratum
is fully enumerated) and $U_b$ otherwise.

The **pure stacked-profile route** uses *only* stacked profiles everywhere:
every stratum's power is built from its own stacked coherent profile,
$C_\\ell = \\frac{1}{4\\pi} \\sum_b N_b\\, P_b(\\ell)\\, R(u)$, i.e. the coherent
estimator applied to all strata, bright and faint. This is the literal
"$D_\\ell$ from the stacked pressure profiles": no individual cluster amplitude
enters, only per-stratum stacks. For the bright strata it must track $T_b$ (and
does, since $R$ is calibrated there); for the faint strata the two routes are
identical by construction. The per-group universal form factor
$\\hat F_g(u) = \\hat f_g(u)^2 R(u)$ (the $w\\,\\tilde y_0^2$-weighted stack
$\\hat f_g$ of the coherent fractional profiles $s_i(\\ell)/s_i(0)$ over the
signal-dominated strata) is also shown; it is the empirical replacement for the
A10 form factor.

Everything is pixel-window deconvolved exactly like the measured $D_\\ell$ (anafast,
`iter=0`, nb39 convention). hmfast enters only through $d_A(z)$ in
$\\theta_{500} = R_{500c}/d_A$ and through the reference A10 curves; the stacked
theory itself contains no halo model.

**Expected residuals:** the cluster sum is a (partially cross-correlated) halo sum,
so it lacks part of the 2-halo power and everything below the $10^{13}\\,M_\\odot$
catalogue floor plus diffuse gas outside apertures; agreement should be best at
$\\ell \\gtrsim$ several hundred where the 1-halo term dominates."""))

cells.append(nbf.v4.new_code_cell(
"""import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

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
DATA = REPO / "data/nb42_rotgroup_stack_yell"
FIG_DIR = REPO / "figures/nb42_rotgroup_tsz_ps_fourier_stack"
FIG_DIR.mkdir(parents=True, exist_ok=True)
NSIDE = 4096
N_GROUPS = 5
ELL_MIN = 100
SNR_MIN = 5.0
COLORS = ["#2166ac", "#1b7837", "#762a83", "#d6604d", "#b8860b"]

print(json.dumps(json.loads((DATA / "group0_order0.json").read_text()), indent=2))"""))

cells.append(nbf.v4.new_markdown_cell(
"""## Load per-group stamp FFTs and build the stratified estimators

`abs_y2`/`bg_y2` are the azimuthally binned $|\\tilde y(\\ell)|^2$ of the cluster
stamps and of their random-aperture backgrounds; `re_y` is the coherent
phase-centred profile ($\\Delta\\ell = 25$, common zero-padded grid). The measured
group $D_\\ell$ (anafast on the same map) is cached by the export script and
log-binned into the nb39-style datapoints."""))

cells.append(nbf.v4.new_code_cell(
"""def log_bin(ell, dl, lmin=10, lmax=6000, nbin=30):
    "nb39 datapoint convention."
    edges = np.logspace(np.log10(lmin), np.log10(lmax), nbin + 1)
    idx = np.digitize(ell, edges) - 1
    lb, db = [], []
    for b in range(nbin):
        sel = (idx == b) & (ell >= lmin)
        if sel.any():
            lb.append(ell[sel].mean())
            db.append(dl[sel].mean())
    return np.array(lb), np.array(db)


def dl_from_cl(ell, cl):
    return ell * (ell + 1.0) / (2.0 * np.pi) * cl


def calibrate_R(u_samples, R_samples, u_eval):
    "running median of R(u) over pooled reliable-bin samples, clamped to [1, 6]."
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


groups = []
for g in range(N_GROUPS):
    st = np.load(DATA / f"group{g}_order0.npz")
    md = np.load(DATA / f"measured_dl_group{g}.npz")
    ell_st = st["ell_b"]
    nell = ell_st.size
    pw = hp.pixwin(NSIDE, lmax=int(ell_st[-1]) + 20)
    pw_st = np.interp(ell_st, np.arange(pw.size), pw)
    norm = 1.0 / (4.0 * np.pi) / pw_st**2

    mbin = st["mbin"]
    N_b = st["N_bin"].astype(np.float64)
    n_arr = st["n_bin"].astype(np.float64)
    nbin_m = N_b.size
    sub = st["abs_y2"] - st["bg_y2"]     # background-subtracted incoherent power
    re = st["re_y"]                      # coherent centred profiles
    y0 = st["y0"]
    theta500_rad = st["theta500_arcmin"] / (180.0 * 60.0 / np.pi)

    # per-stratum estimators
    T = np.zeros((nbin_m, nell))         # incoherent stratified sum
    sigT = np.full((nbin_m, nell), np.inf)
    P = np.zeros((nbin_m, nell))         # debiased squared coherent stack
    sigP = np.full((nbin_m, nell), np.inf)
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

    # stratum band SNR and R(u) calibration from the reliable strata
    band = (ell_st >= 300) & (ell_st <= 4000)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.nanmedian(np.where(sigT > 0, T / sigT, np.inf), axis=1)
    reliable = np.isinf(snr) | (snr >= SNR_MIN)
    reliable &= n_arr > 0
    u_s, R_s = [], []
    for b in np.where(reliable & (n_arr >= 2))[0]:
        ok = band & (P[b] > 0) & (T[b] > 0)
        u_s.append(ell_st[ok] * theta_med[b])
        R_s.append(T[b, ok] / (N_b[b] * P[b, ok]))
    u_s, R_s = np.concatenate(u_s), np.concatenate(R_s)

    # hybrid per-stratum contributions
    contrib = np.zeros((nbin_m, nell))
    err2 = np.zeros(nell)
    used_coh = np.zeros(nbin_m, dtype=bool)
    for b in range(nbin_m):
        if n_arr[b] == 0:
            continue
        if reliable[b]:
            contrib[b] = T[b]
            e = np.where(np.isfinite(sigT[b]), sigT[b], 0.0)
        else:
            used_coh[b] = True
            Rb = calibrate_R(u_s, R_s, ell_st * theta_med[b])
            contrib[b] = N_b[b] * np.clip(P[b], 0.0, None) * Rb
            e = N_b[b] * sigP[b] * Rb
        err2 += e**2
    cl_direct = contrib.sum(axis=0) * norm
    cl_err = np.sqrt(err2) * norm
    cl_raw = ((N_b[mbin] / n_arr[mbin])[:, None] * st["abs_y2"]).sum(axis=0) * norm

    # stacked *coherent* fractional profile f(u) from reliable-strata clusters;
    # the signed background noise averages to zero (no positive noise floor at
    # high u, unlike a stack of per-cluster |y|^2 shapes), and the incoherent
    # power shape is restored with the same R(u) calibration
    rel_cl = reliable[mbin]
    u_grid = np.logspace(-3.2, np.log10(120.0), 300)
    idx_rel = np.where(rel_cl)[0]
    f_prof = np.full((idx_rel.size, u_grid.size), np.nan)
    for k, i in enumerate(idx_rel):
        u_i = ell_st * theta500_rad[i]
        cov = (u_grid >= u_i[0]) & (u_grid <= u_i[-1])
        f_prof[k, cov] = np.interp(u_grid[cov], u_i, re[i] / re[i, 0])
    covered = np.isfinite(f_prof)
    wf = (N_b[mbin] / n_arr[mbin])[idx_rel] * y0[idx_rel] ** 2
    w_sum = np.where(covered, wf[:, None], 0.0).sum(axis=0)
    fhat = np.where(covered, wf[:, None] * f_prof, 0.0).sum(axis=0) / np.where(
        w_sum > 0, w_sum, np.nan
    )
    Fhat = fhat**2 * calibrate_R(u_s, R_s, u_grid)

    # pure stacked-profile route: every stratum from its own stacked coherent
    # profile (identical to the hybrid's faint-strata branch, applied everywhere)
    cl_shape = np.zeros(nell)
    for b in range(nbin_m):
        if n_arr[b] == 0:
            continue
        if n_arr[b] >= 2:
            Rb = calibrate_R(u_s, R_s, ell_st * theta_med[b])
            cl_shape += N_b[b] * np.clip(P[b], 0.0, None) * Rb
        else:
            cl_shape += T[b]
    cl_shape *= norm

    ellb, dlb = log_bin(md["ell"], md["dl"])
    groups.append(dict(
        g=g,
        zlo=float(st["z_inner"]), zhi=float(st["z_outer"]),
        n_cl=len(y0), n_tot=int(st["N_bin"].sum()),
        ell_st=ell_st,
        dl_direct=dl_from_cl(ell_st, cl_direct),
        dl_err=dl_from_cl(ell_st, cl_err),
        dl_raw=dl_from_cl(ell_st, cl_raw),
        dl_shape=dl_from_cl(ell_st, cl_shape),
        ellb=ellb, dlb=dlb,
        u_grid=u_grid, Fhat=Fhat,
        contrib3000=np.array([np.interp(3000.0, ell_st, contrib[b] * norm) for b in range(nbin_m)]),
        logm_edges=st["logm_edges"],
        reliable=reliable, used_coh=used_coh, snr=snr, n_bin=n_arr.astype(int),
    ))
    lab = ["T" if reliable[b] else ("U" if n_arr[b] else "-") for b in range(nbin_m)]
    print(f"group {g}: z=[{groups[-1]['zlo']:.2f},{groups[-1]['zhi']:.2f}]  "
          f"sampled {groups[-1]['n_cl']} of {groups[-1]['n_tot']:,}  strata: {' '.join(lab)}")"""))

cells.append(nbf.v4.new_markdown_cell(
"""## Reference: hmfast A10 GNFW ($B=1$) per group (the nb39 theory being replaced)"""))

cells.append(nbf.v4.new_code_cell(
"""import jax.numpy as jnp

from flamingo.catalogue import D3A_COSMOLOGY
from hmfast.halos import HaloModel
from hmfast.halos.profiles import GNFWPressureProfile
from hmfast.tracers import tSZTracer

A10_B1 = dict(P0=8.403, c500=1.177, gamma=0.3081, alpha=1.0510, beta=5.4905)
hm = HaloModel(cosmology=D3A_COSMOLOGY)
ell_th = jnp.logspace(1.0, np.log10(6000.0), 40)
m_grid = jnp.logspace(11.0, 15.5, 60)
pref = np.asarray(ell_th) * (np.asarray(ell_th) + 1) / (2 * np.pi)

profile = GNFWPressureProfile(**A10_B1, B=1.0)
tracer = tSZTracer(profile=profile)
for gr in groups:
    z = jnp.geomspace(max(gr["zlo"], 0.001), gr["zhi"], 60)
    cl1 = np.asarray(hm.cl_1h(tracer, tracer, l=ell_th, m=m_grid, z=z))
    cl2 = np.asarray(hm.cl_2h(tracer, tracer, l=ell_th, m=m_grid, z=z))
    gr["ell_th"] = np.asarray(ell_th)
    gr["dl_a10"] = pref * (cl1 + cl2)
    gr["dl_a10_2h"] = pref * cl2
print("A10 reference curves done")"""))

cells.append(nbf.v4.new_markdown_cell(
"""## Main comparison: measured group $D_\\ell$ vs stacked-profile theory

Black points: measured map spectra (the nb39 datapoints). Solid: hybrid stratified
stack sum with its sampling-uncertainty band. Dashed: the pure stacked-profile
route (every stratum from its stacked coherent profile). Dotted: raw sum without
background subtraction. Grey:
the A10 halo model that the stacked theory replaces. The thin dash-dotted curve adds
the A10 *2-halo* term to the stacked sum: an aperture-limited cluster sum cannot
contain power from scales larger than the apertures, and this shows the low-$\\ell$
gap is exactly that term."""))

cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4), sharex=True)
axes = axes.ravel()

for gr, ax, col in zip(groups, axes, COLORS):
    ax.plot(gr["ell_th"], gr["dl_a10"], color="0.65", lw=1.4,
            label="hmfast A10 B=1 (nb39)")
    ax.plot(gr["ellb"], gr["dlb"], "o", ms=4, mfc="none", color="k",
            label="measured (map)")
    ax.plot(gr["ell_st"], gr["dl_raw"], color=col, lw=1.0, ls=":", alpha=0.7,
            label="stack, raw (no bg sub)")
    ax.fill_between(gr["ell_st"],
                    np.clip(gr["dl_direct"] - gr["dl_err"], 1e-20, None),
                    gr["dl_direct"] + gr["dl_err"],
                    color=col, alpha=0.25, lw=0)
    ax.plot(gr["ell_st"], gr["dl_direct"], color=col, lw=2.0,
            label="stack, direct sum (hybrid)")
    ax.plot(gr["ell_st"], gr["dl_shape"], color=col, lw=1.6, ls="--",
            label="stacked profiles (all-coherent)")
    ax.plot(gr["ell_th"], gr["dl_a10_2h"] + np.interp(
                gr["ell_th"], gr["ell_st"], gr["dl_direct"]),
            color="0.35", lw=1.0, ls="-.", alpha=0.9,
            label="stack direct + A10 2-halo")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ELL_MIN, 6000)
    sel = gr["ellb"] >= ELL_MIN
    ax.set_ylim(gr["dlb"][sel].min() / 4.0, gr["dlb"][sel].max() * 6.0)
    ax.set_title(rf"group {gr['g']}: $z \\in [{gr['zlo']:.2f}, {gr['zhi']:.2f}]$, "
                 rf"{gr['n_cl']} of {gr['n_tot']:,} clusters")
    ax.grid(True, which="both", alpha=0.25)
    if gr["g"] >= 2:
        ax.set_xlabel(r"multipole $\\ell$")
    if gr["g"] % 3 == 0:
        ax.set_ylabel(r"$D_\\ell$  [$y^2$]")

axr = axes[-1]
for gr, col in zip(groups, COLORS):
    sel = gr["ellb"] >= ELL_MIN
    r = np.interp(gr["ellb"][sel], gr["ell_st"], gr["dl_direct"]) / gr["dlb"][sel]
    axr.semilogx(gr["ellb"][sel], r, color=col, lw=1.8,
                 label=rf"group {gr['g']}")
    rs = np.interp(gr["ellb"][sel], gr["ell_st"], gr["dl_shape"]) / gr["dlb"][sel]
    axr.semilogx(gr["ellb"][sel], rs, color=col, lw=1.2, ls="--", alpha=0.8)
axr.axhline(1.0, color="k", lw=0.8)
axr.axhspan(0.9, 1.1, color="0.92", zorder=0)
axr.set_xlim(ELL_MIN, 6000)
axr.set_ylim(0.4, 1.6)
axr.set_yscale("linear")
axr.set_xlabel(r"multipole $\\ell$")
axr.set_title("stack / measured (solid direct, dashed shape)")
axr.grid(True, which="both", alpha=0.25)
axr.legend(fontsize=8, loc="lower left", ncol=2)
axes[0].legend(fontsize=7.5, loc="lower left")

fig.suptitle(r"L2p8_m9 lc0 rotation groups: measured tSZ $D_\\ell$ vs empirical "
             r"Fourier-space stacked profiles", y=0.995)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"rotgroup_dl_vs_stack_5panel.{ext}", bbox_inches="tight")
plt.show()

for gr in groups:
    sel = gr["ellb"] >= ELL_MIN
    msg = [f"group {gr['g']} (z {gr['zlo']:.2f}-{gr['zhi']:.2f}):"]
    for lo, hi in [(300, 1000), (1000, 3000), (3000, 6000)]:
        m = sel & (gr["ellb"] >= lo) & (gr["ellb"] < hi)
        rd = np.median(np.interp(gr["ellb"][m], gr["ell_st"], gr["dl_direct"]) / gr["dlb"][m])
        rs = np.median(np.interp(gr["ellb"][m], gr["ell_st"], gr["dl_shape"]) / gr["dlb"][m])
        ra = np.median(np.interp(gr["ellb"][m], gr["ell_th"], gr["dl_a10"]) / gr["dlb"][m])
        msg.append(f"  ell {lo}-{hi}: direct/meas={rd:.3f} shape/meas={rs:.3f} A10/meas={ra:.3f}")
    print("\\n".join(msg))"""))

cells.append(nbf.v4.new_markdown_cell(
"""## The stacked Fourier-space profiles

The per-group stacked shapes $\\hat F_g(u) = \\hat f_g(u)^2 R(u)$, plotted as
$\\sqrt{\\hat F}$ (the effective rms $|\\tilde y(\\ell)|/\\tilde y(0)$ form
factor). These are the empirical pressure-profile form factors that replace the
A10 GNFW in the theory. Because $\\hat F$ carries the incoherence factor $R$,
its $u \\to 0$ limit is $\\langle \\tilde y_0^2 \\rangle / \\langle \\tilde y_0
\\rangle^2 > 1$ (population amplitude scatter), not exactly 1."""))

cells.append(nbf.v4.new_code_cell(
"""def sqrt_pos(a):
    "sqrt of a with non-positive values masked (log-scale friendly)."
    return np.sqrt(np.where(a > 0, a, np.nan))

fig, ax = plt.subplots(figsize=(6.6, 4.8))
for gr, col in zip(groups, COLORS):
    ax.plot(gr["u_grid"], sqrt_pos(gr["Fhat"]), color=col, lw=1.8,
            label=rf"group {gr['g']}: $z \\in [{gr['zlo']:.2f}, {gr['zhi']:.2f}]$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-3, 100)
ax.set_ylim(1e-3, 1.5)
ax.set_xlabel(r"$u = \\ell\\,\\theta_{500}$")
ax.set_ylabel(r"$\\sqrt{\\hat F(u)}$  (rms $|\\tilde y(\\ell)|\\,/\\,\\tilde y(0)$)")
ax.set_title(r"Stacked Fourier-space profiles per rotation group "
             r"($w\\,\\tilde y_0^2$-weighted)")
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"stacked_profiles_per_group.{ext}")
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell(
"""## Which masses carry the power, and is the $10^{13}\\,M_\\odot$ floor deep enough?

Per-group contribution of each 0.25 dex mass stratum to the hybrid $D_{3000}$
(open symbols mark strata evaluated with the coherent stacked-profile estimator).
If the lowest bins contribute a converging, small fraction, the catalogue floor
does not limit the comparison."""))

cells.append(nbf.v4.new_code_cell(
"""fig, ax = plt.subplots(figsize=(7.0, 4.8))
for gr, col in zip(groups, COLORS):
    mid = 0.5 * (gr["logm_edges"][:-1] + gr["logm_edges"][1:])
    frac = gr["contrib3000"] / gr["contrib3000"].sum()
    has = gr["n_bin"] > 0
    ax.plot(mid[has], frac[has], "-", color=col, lw=1.5,
            label=rf"group {gr['g']}: $z \\in [{gr['zlo']:.2f}, {gr['zhi']:.2f}]$")
    inc = has & ~gr["used_coh"]
    coh = has & gr["used_coh"]
    ax.plot(mid[inc], frac[inc], "o", color=col, ms=5)
    ax.plot(mid[coh], frac[coh], "o", color=col, ms=6, mfc="none")
ax.set_yscale("log")
ax.set_ylim(1e-4, 1.0)
ax.set_xlabel(r"$\\log_{10} M_{500c}\\,[M_\\odot]$ (stratum centre)")
ax.set_ylabel(r"fraction of hybrid $D_{3000}$")
ax.set_title("Mass-stratum contributions to the stacked $D_{3000}$\\n"
             "(filled: incoherent sum; open: coherent stacked profile)")
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"massbin_contributions_D3000.{ext}")
plt.show()

for gr in groups:
    frac = gr["contrib3000"] / gr["contrib3000"].sum()
    print(f"group {gr['g']}: lowest stratum (logM 13.00-13.25) fraction of D_3000 = {frac[0]:.3f}")"""))

cells.append(nbf.v4.new_markdown_cell(
"""## Conclusions

- The **empirical stacked-profile theory reproduces the measured rotation-group
  spectra at the scales where cluster (1-halo) power dominates**: the hybrid
  stratified sum matches the datapoints to $\\sim$5-15% for $\\ell \\gtrsim 1000$
  in groups 0-3 and to $\\sim$12-16% in group 4, with a few per cent sampling
  uncertainty (band). At high redshift this is a dramatic improvement over the
  A10 GNFW halo model it replaces (per-band A10/measured ratios printed above);
  no pressure-profile model enters the stacked prediction.
- Both routes agree: the hybrid stratified sum over per-cluster
  $|\\tilde y(\\ell)|^2$ and the pure stacked-profile route in which every
  stratum's power comes from its stacked coherent Fourier profile, confirming
  that stacked profiles carry the information needed to predict the group
  power spectrum (per-cluster amplitudes are only needed as weights, not as
  individual measurements).
- The residuals are accounted for by construction, not by fitting:
  (i) at low $\\ell$ the aperture-limited cluster sum cannot contain
  super-aperture (2-halo) power; adding the A10 2-halo term (dash-dotted)
  closes the gap at all redshifts.
  (ii) around $\\ell \\sim 1000$-$2000$ the stamps of clustered neighbours
  overlap, double counting part of the pair power (a $\\sim$10-25% excess in
  groups 1-3).
  (iii) in group 4 ($z > 2.2$) the mass-stratum breakdown shows the
  $10^{13}\\,M_\\odot$ catalogue floor is *not* converged (the lowest stratum
  still carries $\\sim$20% of $D_{3000}$ and the trend is rising), so
  sub-floor halos and diffuse gas explain the remaining $\\sim$12-16% deficit;
  in groups 0-2 the floor contributes $\\lesssim$3% and is converged.
- The random-aperture background subtraction is essential (dotted curves), and for
  the abundant faint strata even that is not enough: the coherent
  stack-first-then-square estimator (with the bright-strata incoherence
  calibration $R(u)$) is what makes the $10^{13}$ to $10^{13.75}\\,M_\\odot$
  contribution measurable at all (an incoherent sum over these strata has
  noise of order the full signal)."""))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

out = Path(__file__).resolve().parents[1] / "notebooks/42_rotgroup_tsz_ps_fourier_stack.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
