"""Prior predictive S(k): sample from ACT Table 1 priors, no likelihood."""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from flamingo.inference.dmb_ps import (
    DMB_TABLE1_PRIORS,
    SAMPLED_DMB_PARAMS,
    evaluate_pk_suppression,
)

FIGURES = REPO / "figures"
DATA = Path("/rds/rds-lxu/flamingo/power_spectra")
H_FLAMINGO = 0.681


def load_flamingo_suppression(z_key: str = "z=0.00") -> tuple[np.ndarray, np.ndarray]:
    """S_FLAMINGO(k) = P_hydro(k) / P_DMO(k) at given z."""
    with h5py.File(DATA / "L1_m9.hdf5", "r") as f_hydro, h5py.File(
        DATA / "L1_m9_DMO.hdf5", "r"
    ) as f_dmo:
        k = np.asarray(f_hydro[f"{z_key}/k"][:], dtype=float)
        pk_hydro = np.asarray(f_hydro[f"{z_key}/P(k)"][:], dtype=float)
        pk_dmo = np.asarray(f_dmo[f"{z_key}/P(k)"][:], dtype=float)
    return k * H_FLAMINGO, pk_hydro / pk_dmo

# Prior params that enter S(k) (exclude sigma_lnY)
pk_params = [p for p in SAMPLED_DMB_PARAMS if p != "sigma_lnY"]
print(f"Prior predictive params ({len(pk_params)}): {pk_params}")

N_DRAWS = 500
rng = np.random.default_rng(20260811)

# Sample from uniform priors
prior_samples = []
for _ in range(N_DRAWS):
    s = {}
    for name in pk_params:
        pr = DMB_TABLE1_PRIORS[name]
        s[name] = float(rng.uniform(pr["min"], pr["max"]))
    prior_samples.append(s)

print(f"Sampled {N_DRAWS} from prior")

# Compute S(k) — first call compiles JIT, rest are fast
print("Computing S(k) for each prior draw...")
ratios = []
k = None
for i, params in enumerate(prior_samples):
    result = evaluate_pk_suppression(**params)
    if k is None:
        k = result["k_h"]
    ratios.append(result["ratio"])
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N_DRAWS} done", flush=True)

stack = np.vstack(ratios)
med = np.median(stack, axis=0)
lo68 = np.percentile(stack, 16, axis=0)
hi68 = np.percentile(stack, 84, axis=0)
lo95 = np.percentile(stack, 2.5, axis=0)
hi95 = np.percentile(stack, 97.5, axis=0)

# Also load posterior S(k) for comparison
print("\nLoading posterior S(k) from chain...")
from getdist import loadMCSamples
samples = loadMCSamples(
    str(REPO / "chains/l1_m9_qfrommap_dmb/fullsky/chain"),
    settings={"ignore_rows": 0.3},
)
names = samples.getParamNames().list()
weights = np.asarray(samples.weights, dtype=float)
weights = weights / weights.sum()

N_POST = 128
idx = rng.choice(
    samples.samples.shape[0],
    size=min(N_POST, samples.samples.shape[0]),
    replace=False,
    p=weights,
)
post_ratios = []
for i, row_idx in enumerate(idx):
    row = samples.samples[row_idx]
    params = {name: float(row[names.index(name)]) for name in pk_params}
    result = evaluate_pk_suppression(**params)
    post_ratios.append(result["ratio"])
    if (i + 1) % 32 == 0:
        print(f"  posterior {i+1}/{N_POST}", flush=True)

post_stack = np.vstack(post_ratios)
post_med = np.median(post_stack, axis=0)
post_lo68 = np.percentile(post_stack, 16, axis=0)
post_hi68 = np.percentile(post_stack, 84, axis=0)
post_lo95 = np.percentile(post_stack, 2.5, axis=0)
post_hi95 = np.percentile(post_stack, 97.5, axis=0)

# FLAMINGO official reference
k_flamingo, s_flamingo = load_flamingo_suppression("z=0.00")

# Plot: prior predictive vs posterior
fig, ax = plt.subplots(figsize=(7.5, 5.0))

# Prior predictive (red)
ax.fill_between(k, lo95, hi95, color="#e74c3c", alpha=0.15, label="Prior 95% CL", zorder=1)
ax.fill_between(k, lo68, hi68, color="#e74c3c", alpha=0.30, label="Prior 68% CL", zorder=2)
ax.plot(k, med, color="#c0392b", lw=2.0, ls="--", label="Prior median", zorder=3)

# Posterior (blue)
ax.fill_between(k, post_lo95, post_hi95, color="#4e79a7", alpha=0.15, label="Posterior 95% CL", zorder=4)
ax.fill_between(k, post_lo68, post_hi68, color="#4e79a7", alpha=0.30, label="Posterior 68% CL", zorder=5)
ax.plot(k, post_med, color="#1f4e79", lw=2.0, label="Posterior median", zorder=6)

# FLAMINGO official (orange)
ax.plot(
    k_flamingo,
    s_flamingo,
    color="#e67e22",
    lw=2.5,
    label=r"FLAMINGO $P_{\rm hydro}/P_{\rm DMO}$ ($z=0$, $L1\_m9$)",
    zorder=7,
)

ax.axhline(1.0, color="0.35", ls=":", lw=0.9, zorder=0)
ax.set_xscale("log")
ax.set_xlabel(r"$k\ (h\,{\rm Mpc}^{-1})$")
ax.set_ylabel(r"$P_{\rm DMB}(k)\,/\,P_{\rm NFW}(k)$")
ax.set_title(
    rf"Prior predictive vs posterior S(k) at $z=0$ "
    rf"(L1\_m9 full-sky tSZ PS; prior={N_DRAWS}, post={N_POST} draws)"
)
k_lo = max(float(k.min()), float(k_flamingo.min()))
k_hi = min(float(k.max()), float(k_flamingo.max()))
ax.set_xlim(k_lo, k_hi)
mask = (k_flamingo >= k_lo) & (k_flamingo <= k_hi)
y_lo = min(
    0.60,
    0.95 * float(np.nanmin(lo95)),
    0.95 * float(np.nanmin(s_flamingo[mask])) if mask.any() else 0.60,
)
ax.set_ylim(y_lo, 1.10)
ax.legend(frameon=False, loc="lower left", fontsize=8, ncol=2)
fig.tight_layout()

out_png = FIGURES / "dmb" / "dmb_sk_prior_vs_posterior.png"
out_pdf = FIGURES / "dmb" / "dmb_sk_prior_vs_posterior.pdf"
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, bbox_inches="tight", dpi=200)
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_png}")

# Print comparison
print(f"\n{'k':>8s} {'prior_med':>10s} {'prior_68':>16s} {'post_med':>10s} {'post_68':>16s} {'shrink':>8s}")
for kt in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
    ik = np.argmin(np.abs(k - kt))
    prior_w = hi68[ik] - lo68[ik]
    post_w = post_hi68[ik] - post_lo68[ik]
    shrink = post_w / prior_w if prior_w > 0 else 0
    print(
        f"{k[ik]:8.2f} {med[ik]:10.4f} [{lo68[ik]:.4f},{hi68[ik]:.4f}] "
        f"{post_med[ik]:10.4f} [{post_lo68[ik]:.4f},{post_hi68[ik]:.4f}] {shrink:8.2f}"
    )
