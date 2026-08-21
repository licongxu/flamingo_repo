"""Best-fit q>5 masked D_ell^yy vs L1_m9 data at the PS scalrel MAP."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.run_l1_m9_cnc_qgt5_chain import FIXED, HMFAST_SRC  # noqa: E402
from scripts.run_l1_m9_ps_qgt5_scalrel_chain import (  # noqa: E402
    CASE,
    CASES,
    CHAINS,
    chain_dir,
)
from scripts.run_masked_ps_chains import data_files, load_converged_artifacts  # noqa: E402

STEM = REPO / "figures" / "masked_ps" / "l1_m9_ps_qgt5_scalrel_bestfit_qfrommap"
STEM_ALL = REPO / "figures" / "masked_ps" / "l1_m9_ps_scalrel_bestfit_qfrommap"
PANELS = (
    ("fullsky", "full sky"),
    ("qgt50", r"$q>50$"),
    ("qgt20", r"$q>20$"),
    ("qgt10", r"$q>10$"),
    ("qgt5", r"$q>5$"),
)
COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
BURN_IN_FRACTION = 0.3
SCALREL = ("A_SZ", "alpha_SZ", "sigma_lnY")
ELL_MIN = 10.0
ELL_MAX = 959.5
PAPER_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 15,
    "legend.fontsize": 10,
    "text.latex.preamble": r"\usepackage{amsmath}",
}


def shot_noise_c(
    y_mpc2: np.ndarray,
    d_a: np.ndarray,
    q: np.ndarray | None = None,
    q_cut: float | None = None,
) -> float:
    """Poisson C_ell = sum (Y_Mpc2 / D_A^2)^2 / 4pi.

    Masked cases keep the unmasked remainder, ~(q > q_cut), matching
    paper_results.compute_ps.
    """
    y_sr2 = (np.asarray(y_mpc2, dtype=float) / np.asarray(d_a, dtype=float) ** 2) ** 2
    ok = np.isfinite(y_sr2)
    if q_cut is not None:
        q = np.asarray(q, dtype=float)
        ok &= ~(np.isfinite(q) & (q > q_cut))
    return float(np.sum(y_sr2[ok]) / (4.0 * np.pi))


def shot_noise_dl(ell: np.ndarray, c_shot: float) -> np.ndarray:
    """White D_ell = ell(ell+1) C_shot / 2pi."""
    ell = np.asarray(ell, dtype=float)
    return ell * (ell + 1.0) / (2.0 * np.pi) * float(c_shot)


def load_catalogue_shot_c(path: Path | None = None) -> dict[str, float]:
    """One catalogue pass: full-sky and remaining-cluster shot noise per q cut."""
    import pandas as pd
    from flamingo.catalogue.frame import angular_diameter_distance
    from flamingo.powerspectra.q_selection import resolve_q_selection

    selection = resolve_q_selection("qfrommap")
    path = path or (
        selection.l1_catalogue_dir
        / "halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommap.csv"
    )
    q_col = selection.q_column
    acc = {case: 0.0 for case in CASES}
    for chunk in pd.read_csv(
        path,
        comment="#",
        usecols=["z", "Y_5R500c_Mpc2", q_col],
        chunksize=1_000_000,
    ):
        z = chunk["z"].to_numpy(np.float64)
        y = chunk["Y_5R500c_Mpc2"].to_numpy(np.float64)
        q = chunk[q_col].to_numpy(np.float64)
        finite = np.isfinite(z) & np.isfinite(y)
        d_a = np.asarray(angular_diameter_distance(z[finite]), dtype=float)
        y_f = y[finite]
        q_f = q[finite]
        acc["fullsky"] += shot_noise_c(y_f, d_a)
        for case, q_cut in CASES.items():
            if q_cut is None:
                continue
            acc[case] += shot_noise_c(y_f, d_a, q=q_f, q_cut=q_cut)
    return acc


def map_params(chain_file: Path = CHAINS / "chain.1.txt") -> dict[str, float]:
    """Return the maximum-posterior sample after the GetDist burn-in fraction."""
    header = chain_file.read_text().splitlines()[0].lstrip("#").split()
    samples = np.loadtxt(chain_file)
    kept = samples[int(BURN_IN_FRACTION * len(samples)) :]
    row = kept[int(np.argmin(kept[:, header.index("minuslogpost")]))]
    return {name: float(row[header.index(name)]) for name in (*SCALREL, "chi2", "minuslogpost")}


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        out = stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out, flush=True)
    plt.close(fig)


def evaluate(
    params: dict[str, float],
    case: str = CASE,
    theory=None,
    shot_c: float | None = None,
) -> dict:
    """Evaluate masked 1h+2h theory and catalogue Poisson shot noise."""
    sys.path.insert(0, str(HMFAST_SRC))
    from flamingo.inference.l1_m9 import load_bandpower_likelihood
    from flamingo.inference.masked_ps import MaskedTSZTheory

    data, covariance = data_files(case, load_converged_artifacts()["covariance_paths"])
    ell, observed, cov, inverse = load_bandpower_likelihood(
        data, covariance, data_scale=1e-12
    )
    if theory is None:
        theory = MaskedTSZTheory({"q_cat": CASES[case]}, timing=True)
    else:
        theory.q_cat = CASES[case]
    terms = theory.evaluate_bandpowers(
        A_SZ=params["A_SZ"],
        alpha_SZ=params["alpha_SZ"],
        sigma_lnY=params["sigma_lnY"],
        **FIXED,
    )
    one = np.asarray(terms["1h"], dtype=float)
    two = np.asarray(terms["2h"], dtype=float)
    total = one + two
    residual = observed - total
    ell = np.asarray(ell, dtype=float)
    if shot_c is None:
        shot_c = load_catalogue_shot_c()[case]
    return {
        "ell": ell,
        "observed": observed,
        "err": np.sqrt(np.diag(cov)),
        "1h": one,
        "2h": two,
        "total": total,
        "shot_noise": shot_noise_dl(ell, shot_c),
        "chi2": float(residual @ inverse @ residual),
        "params": params,
    }


def build_figure(fit: dict, params: dict[str, float]) -> plt.Figure:
    plt.rcParams.update(PAPER_RC)
    scale = 1e12
    fig, axes = plt.subplots(
        2, 1, figsize=(6.8, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    axes[0].errorbar(
        fit["ell"], scale * fit["observed"], yerr=scale * fit["err"],
        fmt="o", color="#1b9e77", ms=4.0, capsize=2, zorder=3, label=r"L1\_m9 $q>5$",
    )
    axes[0].plot(fit["ell"], scale * fit["total"], color="#d95f02", lw=2.0, zorder=2, label=r"1h+2h MAP")
    axes[0].plot(fit["ell"], scale * fit["1h"], color="#d95f02", ls=":", lw=1.6, zorder=1, label=r"1h")
    axes[0].plot(fit["ell"], scale * fit["2h"], color="#d95f02", ls="--", lw=1.6, zorder=1, label=r"2h")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlim(ELL_MIN, ELL_MAX)
    axes[0].set_ylabel(r"$10^{12}D_\ell^{yy}$")
    axes[0].legend(loc="lower right", frameon=False)
    axes[0].set_title(
        rf"$A_{{\mathrm{{SZ}}}}={params['A_SZ']:.3f}$, "
        rf"$\alpha_{{\mathrm{{SZ}}}}={params['alpha_SZ']:.3f}$, "
        rf"$\sigma_{{\ln Y}}={params['sigma_lnY']:.3f}$"
        rf" ($\chi^2={fit['chi2']:.1f}/15$)"
    )
    pull = (fit["observed"] - fit["total"]) / fit["err"]
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].plot(fit["ell"], pull, "o-", color="#1b9e77", ms=3.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$\ell$")
    axes[1].set_ylabel(r"$(D-D^{\rm th})/\sigma$")
    fig.tight_layout()
    return fig


def build_all_figure(fits: dict[str, dict]) -> plt.Figure:
    from matplotlib.lines import Line2D

    plt.rcParams.update(PAPER_RC)
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    scale = 1e12
    for (case, label), color in zip(PANELS, COLORS):
        fit = fits[case]
        ax.errorbar(
            fit["ell"], scale * fit["observed"], yerr=scale * fit["err"],
            fmt="o", color=color, ms=3.8, capsize=2, zorder=3,
        )
        ax.plot(fit["ell"], scale * fit["total"], color=color, lw=2.0, zorder=2, label=label)
        ax.plot(fit["ell"], scale * fit["1h"], color=color, ls=":", lw=1.6, zorder=1)
        ax.plot(fit["ell"], scale * fit["2h"], color=color, ls="--", lw=1.6, zorder=1)
        ax.plot(
            fit["ell"], scale * fit["shot_noise"],
            color=color, ls="-.", lw=1.4, zorder=1,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ELL_MIN, ELL_MAX)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$10^{12}D_\ell^{yy}$")
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color="0.2", lw=2.0, ls="-"),
        Line2D([0], [0], color="0.2", lw=1.6, ls=":"),
        Line2D([0], [0], color="0.2", lw=1.6, ls="--"),
        Line2D([0], [0], color="0.2", lw=1.4, ls="-."),
    ]
    labels += [
        r"1h+2h MAP",
        r"1h",
        r"2h",
        r"shot noise ($\sum Y_{5R_{500c}}^2/4\pi$)",
    ]
    ax.legend(handles, labels, loc="lower right", frameon=False, handlelength=2.2)
    fig.tight_layout()
    return fig


def main() -> dict:
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["JAX_ENABLE_X64"] = "1"
    sys.path.insert(0, str(HMFAST_SRC))
    from flamingo.inference.masked_ps import MaskedTSZTheory

    theory = MaskedTSZTheory({"q_cat": None}, timing=True)
    shot_c = load_catalogue_shot_c()
    fits = {}
    for case, _ in PANELS:
        params = map_params(chain_dir(case) / "chain.1.txt")
        print(f"MAP {case}", {k: params[k] for k in SCALREL}, flush=True)
        print(f"  C_shot={shot_c[case]:.6e}", flush=True)
        fits[case] = evaluate(params, case, theory=theory, shot_c=shot_c[case])
        print(f"  chi2={fits[case]['chi2']:.3f}", flush=True)
    _save(build_all_figure(fits), STEM_ALL)
    _save(build_figure(fits["qgt5"], fits["qgt5"]["params"]), STEM)
    return {case: fit["chi2"] for case, fit in fits.items()}


if __name__ == "__main__":
    main()
