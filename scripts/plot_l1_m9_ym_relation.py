"""L1_m9 SOAP Y_5R500c–M_500c relation, with resolved q>5 clusters in red.

    python scripts/plot_l1_m9_ym_relation.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from flamingo.catalogue.frame import efunc  # noqa: E402
from flamingo.inference.masked_ps import B_HYDROSTATIC, GNFW_SHAPE  # noqa: E402
from flamingo.powerspectra.q_selection import resolve_q_selection  # noqa: E402
from paper_results.config import FIGURES_FEEDBACK  # noqa: E402

_QFROMMAP = resolve_q_selection("qfrommap")
CATALOGUE = _QFROMMAP.l1_catalogue_dir / (
    "halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommap.csv"
)
Q_COLUMN = _QFROMMAP.q_column
Q_MIN = 5.0
COLUMNS = ("z", "M_500c_Msun", "R_500c_Mpc", "Y_5R500c_Mpc2", Q_COLUMN)
H = 0.681
PIVOT = 0.7 * 3.0e14
SELF_SIM_ALPHA_SZ = 2.0 / 3.0 + 0.12 + 1.0 / 3.0
SELF_SIM_ALPHA_Y = SELF_SIM_ALPHA_SZ + 2.0 / 3.0
PAPER_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "mathtext.fontset": "cm",
    "text.latex.preamble": r"\usepackage{amsmath}",
}


def catalogue_path() -> Path:
    """Return the fiducial L1_m9 q-from-map catalogue used for CNC."""
    return CATALOGUE


def output_stem() -> Path:
    """Return the explicit q-from-map Y–M figure stem."""
    return FIGURES_FEEDBACK / "l1_m9_cnc_ym_relation_qgt5_qfrommap"


def load_ym(path: Path) -> pd.DataFrame:
    """Load the SOAP Y–M columns and drop non-positive masses or Y."""
    if not path.is_file():
        raise FileNotFoundError(f"missing catalogue: {path}")
    frame = pd.read_csv(path, comment="#", usecols=list(COLUMNS))
    mass = frame["M_500c_Msun"].to_numpy(np.float64)
    y = frame["Y_5R500c_Mpc2"].to_numpy(np.float64)
    r500 = frame["R_500c_Mpc"].to_numpy(np.float64)
    ok = (
        np.isfinite(mass)
        & np.isfinite(y)
        & np.isfinite(r500)
        & (mass > 0.0)
        & (y > 0.0)
        & (r500 > 0.0)
    )
    if not np.any(ok):
        raise ValueError("catalogue has no finite positive M_500c, R_500c, and Y_5R500c")
    return frame.loc[ok].copy()


def scaled_y(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Self-similar $Y E(z)^{-2/3}$ used in the FLAMINGO Y–M figures."""
    return np.asarray(y, dtype=np.float64) * efunc(z) ** (-2.0 / 3.0)


def gnfw_y_over_y0_r2(x_out: float = 5.0) -> float:
    """GNFW factor $Y_{\\rm sph}(<x_{\\rm out})/(y_0 R_{500}^2)$."""
    from scipy.integrate import quad

    shape = GNFW_SHAPE
    c500, gamma, alpha, beta = shape["c500"], shape["gamma"], shape["alpha"], shape["beta"]

    def p(x: float) -> float:
        cx = c500 * max(float(x), 1e-12)
        return cx ** (-gamma) * (1.0 + cx ** alpha) ** (-(beta - gamma) / alpha)

    i_sph = float(quad(lambda x: x**2 * p(x), 0.0, x_out, epsabs=1e-10)[0])
    i_los = float(quad(p, 0.0, np.inf, epsabs=1e-10, limit=200)[0])
    return float(2.0 * np.pi * i_sph / i_los)


def m_tilde(mass: np.ndarray, *, B: float = B_HYDROSTATIC) -> np.ndarray:
    """Planck/CNC mass argument $(M_{500c} h/B)/(0.7\\times 3\\times 10^{14} M_\\odot)$."""
    return (np.asarray(mass, dtype=np.float64) * H / B) / PIVOT


def ols_log10(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Ordinary least squares: $\\log_{10} y = a + b \\log_{10} x$."""
    lx = np.log10(np.asarray(x, dtype=np.float64))
    ly = np.log10(np.asarray(y, dtype=np.float64))
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (intercept + slope * lx)
    n = lx.size
    dof = n - 2
    var = float(np.sum(resid**2) / dof)
    sxx = float(np.sum((lx - lx.mean()) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_err": float(np.sqrt(var / sxx)),
        "intercept_err": float(np.sqrt(var * (1.0 / n + lx.mean() ** 2 / sxx))),
        "sigma_ln": float(np.std(resid, ddof=1) * np.log(10.0)),
        "n": int(n),
    }


def fit_ym(mass: np.ndarray, y_scaled: np.ndarray) -> dict[str, float]:
    """Fit $Y E^{-2/3}\\propto M^{\\alpha_Y}$. Self-similar $\\alpha_Y=5/3$, $\\alpha_{SZ}=\\alpha_Y-2/3$."""
    fit = ols_log10(mass, y_scaled)
    fit["alpha_Y"] = fit["slope"]
    fit["alpha_SZ"] = fit["slope"] - 2.0 / 3.0
    fit["alpha_SZ_err"] = fit["slope_err"]
    return fit


def fit_asz(
    mass: np.ndarray,
    y_sph: np.ndarray,
    r500: np.ndarray,
    z: np.ndarray,
    *,
    B: float = B_HYDROSTATIC,
    k: float | None = None,
) -> dict[str, float]:
    """OLS of the CNC $y_0(M,z)$ relation, with $y_0=Y_{5R_{500}}/(k R_{500}^2)$."""
    if k is None:
        k = gnfw_y_over_y0_r2()
    ez = efunc(z)
    y0 = np.asarray(y_sph, dtype=np.float64) / (
        k * np.asarray(r500, dtype=np.float64) ** 2
    )
    lhs = y0 / (ez**2 * (H / 0.7) ** (-0.5))
    fit = ols_log10(m_tilde(mass, B=B), lhs)
    fit["A_SZ"] = fit["intercept"]
    fit["A_SZ_err"] = fit["intercept_err"]
    fit["alpha_SZ"] = fit["slope"]
    fit["alpha_SZ_err"] = fit["slope_err"]
    fit["k"] = float(k)
    fit["B"] = float(B)
    return fit


def median_band(
    mass: np.ndarray,
    y: np.ndarray,
    *,
    dex: float = 0.1,
    min_count: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return log-mass bin centres and the 16/50/84 percentiles of $Y$."""
    logm = np.log10(np.asarray(mass, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64)
    lo = np.floor(logm.min() / dex) * dex
    hi = np.ceil(logm.max() / dex) * dex
    edges = np.arange(lo, hi + 0.5 * dex, dex)
    mids, med, p16, p84 = [], [], [], []
    for left, right in zip(edges[:-1], edges[1:]):
        sel = (logm >= left) & (logm < right)
        if int(sel.sum()) < min_count:
            continue
        vals = y[sel]
        mids.append(10.0 ** (0.5 * (left + right)))
        med.append(float(np.median(vals)))
        p16.append(float(np.percentile(vals, 16)))
        p84.append(float(np.percentile(vals, 84)))
    return (
        np.asarray(mids),
        np.asarray(med),
        np.asarray(p16),
        np.asarray(p84),
    )


def _count_label(n: int) -> str:
    return f"{n:,d}".replace(",", "{,}")


def _powerlaw(ax, mass, y, color, ls, zorder):
    fit = fit_ym(mass, y)
    grid = np.logspace(np.log10(mass.min()), np.log10(mass.max()), 200)
    ax.plot(
        grid,
        10.0 ** (fit["intercept"] + fit["slope"] * np.log10(grid)),
        color=color,
        ls=ls,
        lw=1.4,
        zorder=zorder,
    )
    return fit


def build_figure(mass: np.ndarray, y: np.ndarray, resolved: np.ndarray) -> plt.Figure:
    """Publication-style log–log Y–M: density + median band, $q>5$ in red."""
    plt.rcParams.update(PAPER_RC)
    resolved = np.asarray(resolved, dtype=bool)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), layout="constrained")
    ax.hexbin(
        mass,
        y,
        xscale="log",
        yscale="log",
        gridsize=55,
        cmap="Greys",
        mincnt=1,
        linewidths=0.0,
        rasterized=True,
        zorder=1,
    )
    mids, med, p16, p84 = median_band(mass, y)
    ax.fill_between(mids, p16, p84, color="0.35", alpha=0.28, zorder=2, lw=0)
    ax.plot(mids, med, color="k", lw=1.6, zorder=3)
    ax.plot(
        mass[resolved],
        y[resolved],
        ".",
        color="#d62728",
        ms=3.2,
        alpha=0.85,
        rasterized=True,
        zorder=4,
    )
    _powerlaw(ax, mass, y, "#1f77b4", "--", 5)
    _powerlaw(ax, mass[resolved], y[resolved], "#d62728", ":", 6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_{500c}\,[M_\odot]$")
    ax.set_ylabel(r"$Y_{5R_{500c}}\,E(z)^{-2/3}\,[{\rm Mpc}^{2}]$")
    n_res = int(resolved.sum())
    ax.legend(
        handles=[
            Patch(facecolor="0.75", edgecolor="none", label=r"all $M_{500c}>5\times 10^{13}\,M_\odot$"),
            Line2D([0], [0], color="k", lw=1.6, label=r"median $\pm 16$--$84$"),
            Line2D([0], [0], color="#1f77b4", ls="--", lw=1.4, label=r"OLS, all"),
            Line2D(
                [0],
                [0],
                color="#d62728",
                marker=".",
                ls="none",
                ms=8,
                label=rf"$q>5$ ($N={_count_label(n_res)}$)",
            ),
            Line2D([0], [0], color="#d62728", ls=":", lw=1.4, label=r"OLS, $q>5$"),
        ],
        loc="upper left",
        frameon=False,
    )
    return fig


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        output = stem.with_suffix(f".{suffix}")
        fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print("wrote", output, flush=True)
    plt.close(fig)


def _report(name: str, ym: dict[str, float], asz: dict[str, float]) -> None:
    print(
        f"{name}: N={ym['n']:,d}  "
        f"alpha_Y={ym['alpha_Y']:.4f}+/-{ym['slope_err']:.4f}  "
        f"(self-sim Y 5/3={5.0/3.0:.4f}, A10 {SELF_SIM_ALPHA_Y:.4f})  "
        f"A_SZ={asz['A_SZ']:.4f}+/-{asz['A_SZ_err']:.4f}  "
        f"alpha_SZ={asz['alpha_SZ']:.4f}+/-{asz['alpha_SZ_err']:.4f}  "
        f"(A10 {SELF_SIM_ALPHA_SZ:.2f})  "
        f"sigma_lnY={asz['sigma_ln']:.4f}",
        flush=True,
    )


def main() -> Path:
    path = catalogue_path()
    print("loading", path, flush=True)
    frame = load_ym(path)
    mass = frame["M_500c_Msun"].to_numpy(np.float64)
    y_sph = frame["Y_5R500c_Mpc2"].to_numpy(np.float64)
    r500 = frame["R_500c_Mpc"].to_numpy(np.float64)
    z = frame["z"].to_numpy(np.float64)
    y = scaled_y(y_sph, z)
    resolved = frame[Q_COLUMN].to_numpy(np.float64) > Q_MIN
    k = gnfw_y_over_y0_r2()
    print(f"N={len(frame):,d}  q>{Q_MIN:g}={int(resolved.sum()):,d}  k={k:.5f}", flush=True)
    samples = {
        "all": np.ones(mass.shape, dtype=bool),
        "q>5": resolved,
        "all M<1e14": mass < 1.0e14,
        "all M>=1e14": mass >= 1.0e14,
    }
    for name, sel in samples.items():
        _report(name, fit_ym(mass[sel], y[sel]), fit_asz(mass[sel], y_sph[sel], r500[sel], z[sel], k=k))
    stem = output_stem()
    save_figure(build_figure(mass, y, resolved), stem)
    return stem.with_suffix(".pdf")


if __name__ == "__main__":
    main()
