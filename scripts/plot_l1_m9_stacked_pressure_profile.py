"""L1_m9 stacked y^cyl profiles from raw map pixels (no subtraction).

One aperture pass per mass bin writes both the map-Y_500 shape stack and
the A10-predicted amplitude stack. Also writes the q>5 and top-1000
map-Y_500 stacks.

    python scripts/plot_l1_m9_stacked_pressure_profile.py
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from flamingo.powerspectra.q_selection import resolve_q_selection  # noqa: E402
from flamingo.profiles.stacking import (  # noqa: E402
    _reduce_stack,
    default_n_jobs,
    measure_profiles,
    stack_normalized,
)

YMAP = Path("/rds/rds-lxu/flamingo/L1_m9/maps/y_unlensed_L1_m9_lc0_nside4096.fits")
_QFROMMAP = resolve_q_selection("qfrommap")
CAT_DIR = _QFROMMAP.l1_catalogue_dir
M_MIN = 5.0e13
Q_MIN = 5.0
X_EDGES = np.linspace(0.0, 4.0, 17)
X_THEORY = np.geomspace(0.12, 4.0, 80)
MASS_EDGES = np.array([5.0e13, 8.0e13, 1.3e14, 2.2e14, 3.6e14, 6.0e14, np.inf])
COLUMNS = (
    "M_500c_Msun",
    "z",
    "R_500c_Mpc",
    "theta_rot_rad",
    "phi_rot_rad",
    "theta_500_arcmin",
    "q_from_aperture",
)
# Arnaud 2010 UPP + D3A; numpy so stacking can fork before JAX.
A10 = dict(P0=8.403, c500=1.177, gamma=0.3081, alpha=1.0510, beta=5.4905)
H = 0.681
OMEGA_M = 0.306
_SIGMA_T_CM2 = 6.6524587e-25
_M_E_C2_EV = 510998.95
_MPC_CM = 3.085677581e24
PAPER_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 8,
    "text.latex.preamble": r"\usepackage{amsmath}",
}


def catalogue_path() -> Path:
    return CAT_DIR / "halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommap.csv"


def output_stem() -> Path:
    return REPO / "figures" / "stack" / "l1_m9_fiducial_mgt5e13_massbins_ycyl_vs_a10"


def output_stem_map() -> Path:
    return REPO / "figures" / "stack" / "l1_m9_fiducial_mgt5e13_massbins_ycyl_stack"


def output_stem_qgt5() -> Path:
    return REPO / "figures" / "stack" / "l1_m9_fiducial_qgt5_ycyl_stack"


def output_stem_top(n: int) -> Path:
    return REPO / "figures" / "stack" / f"l1_m9_fiducial_top{n}_ycyl_stack"


def gnfw_p(x: np.ndarray) -> np.ndarray:
    cx = A10["c500"] * np.clip(np.asarray(x, dtype=float), 1e-6, None)
    return A10["P0"] * cx ** (-A10["gamma"]) * (1.0 + cx ** A10["alpha"]) ** (
        -(A10["beta"] - A10["gamma"]) / A10["alpha"]
    )


def efunc(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.sqrt(OMEGA_M * (1.0 + z) ** 3 + (1.0 - OMEGA_M))


def los_onesided(x: np.ndarray, s_max: float = 30.0, n_s: int = 4000) -> np.ndarray:
    x = np.atleast_1d(np.asarray(x, dtype=float))
    s = np.linspace(0.0, s_max, n_s)
    return np.trapezoid(gnfw_p(np.sqrt(x[:, None] ** 2 + s[None, :] ** 2)), s, axis=1)


@lru_cache(maxsize=1)
def a10_shape_aperture() -> tuple[float, float]:
    """Return (2 int_0^1 x los(x)/los(0) dx, los(0))."""
    xx = np.linspace(0.0, 1.0, 2000)
    los = los_onesided(xx)
    los0 = float(los_onesided(np.array([0.0]))[0])
    return float(2.0 * np.trapezoid(xx * (los / los0), xx)), los0


def a10_y_norm(m: np.ndarray, z: np.ndarray, r500_mpc: np.ndarray, B: float = 1.0) -> np.ndarray:
    """A10 cylindrical aperture mean Y_500^cyl / (pi theta_500^2)."""
    shape_ap, los0 = a10_shape_aperture()
    E = efunc(z)
    m = np.asarray(m, dtype=float)
    p500 = (
        1.65
        * (H / 0.7) ** 2
        * E ** (8.0 / 3.0)
        * ((m * H / B) / (0.7 * 3.0e14)) ** (2.0 / 3.0 + 0.12)
        * (0.7 / H) ** 1.5
    )
    y0 = 2.0 * (_SIGMA_T_CM2 / _M_E_C2_EV) * p500 * (np.asarray(r500_mpc, dtype=float) * _MPC_CM) * los0
    return y0 * shape_ap


def a10_cyl_at(x: np.ndarray) -> np.ndarray:
    los = los_onesided(np.asarray(x, dtype=float))
    xx = np.linspace(0.0, 1.0, 2000)
    yin = los_onesided(xx)
    return los / (2.0 * np.trapezoid(xx * yin, xx))


def mass_label(lo: float, hi: float) -> str:
    def _one(m: float) -> str:
        exp = int(np.floor(np.log10(m)))
        coef = m / 10.0**exp
        if np.isclose(coef, 1.0):
            return rf"10^{{{exp}}}"
        return rf"{coef:g}\times 10^{{{exp}}}"

    if not np.isfinite(hi):
        return rf"$M_{{500c}}>{_one(lo)}\,M_\odot$"
    return rf"${_one(lo)}$--${_one(hi)}\,M_\odot$"


def load_clusters(path: Path, m_min: float = M_MIN) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing catalogue: {path}")
    frame = pd.read_csv(path, comment="#", usecols=list(COLUMNS))
    mass = frame["M_500c_Msun"].to_numpy(np.float64)
    theta = frame["theta_500_arcmin"].to_numpy(np.float64)
    r500 = frame["R_500c_Mpc"].to_numpy(np.float64)
    ok = np.isfinite(frame.to_numpy(np.float64)).all(axis=1)
    ok &= mass >= m_min
    ok &= theta > 0.0
    ok &= r500 > 0.0
    selected = frame.loc[ok].copy()
    if selected.empty:
        raise ValueError(f"no clusters with M_500c > {m_min:g}")
    return selected


def split_mass_bins(
    frame: pd.DataFrame,
    edges: np.ndarray = MASS_EDGES,
) -> list[tuple[float, float, pd.DataFrame]]:
    mass = frame["M_500c_Msun"].to_numpy(np.float64)
    bins: list[tuple[float, float, pd.DataFrame]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (mass >= lo) if not np.isfinite(hi) else (mass >= lo) & (mass < hi)
        chunk = frame.loc[sel]
        if chunk.empty:
            continue
        bins.append((float(lo), float(hi), chunk.reset_index(drop=True)))
    if not bins:
        raise ValueError("no occupied mass bins")
    return bins


def measure_bin(ymap: np.ndarray, frame: pd.DataFrame, n_jobs: int | None = None):
    return measure_profiles(
        ymap,
        frame["theta_rot_rad"].to_numpy(np.float64),
        frame["phi_rot_rad"].to_numpy(np.float64),
        frame["theta_500_arcmin"].to_numpy(np.float64),
        X_EDGES,
        n_jobs=n_jobs,
    )


def a10_norms(frame: pd.DataFrame) -> np.ndarray:
    return a10_y_norm(
        frame["M_500c_Msun"].to_numpy(np.float64),
        frame["z"].to_numpy(np.float64),
        frame["R_500c_Mpc"].to_numpy(np.float64),
    )


def build_figure(
    panels: list[tuple[float, float, dict]],
    theory: np.ndarray,
    x_theory: np.ndarray,
    *,
    vs_a10: bool = True,
) -> plt.Figure:
    plt.rcParams.update(PAPER_RC)
    n = len(panels)
    ncols = 3 if n > 3 else max(n, 1)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.3 * ncols, 2.8 * nrows + 0.4),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (lo, hi, stacked) in zip(axes_flat, panels):
        ax.errorbar(
            stacked["x_mid"],
            stacked["fhat"],
            yerr=stacked["sem"],
            fmt="o",
            ms=3.5,
            color="0.15",
            ecolor="0.45",
            elinewidth=0.8,
            capsize=1.5,
            zorder=3,
            label=(
                rf"$N={stacked['n']}$, $A={stacked['A_median']:.2f}$"
                if vs_a10
                else rf"$N={stacked['n']}$"
            ),
        )
        ax.plot(
            x_theory,
            theory,
            color="#d95f02",
            lw=1.6,
            zorder=2,
            label=r"A10 ($A=1$)" if vs_a10 else r"A10 $y^{\rm cyl}$",
        )
        ax.axvline(1.0, color="0.6", lw=0.7, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(mass_label(lo, hi), fontsize=10)
        ax.legend(frameon=False, loc="upper right")
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.supxlabel(r"$\theta/\theta_{500}$")
    fig.supylabel(
        r"$y^{\rm cyl}(\theta)\big/\bigl[Y_{500}^{\rm A10,cyl}/(\pi\theta_{500}^{2})\bigr]$"
        if vs_a10
        else r"$y^{\rm cyl}(\theta)\big/\bigl[Y_{500}^{\rm cyl}/(\pi\theta_{500}^{2})\bigr]$"
    )
    return fig


def build_topn_figure(
    stacked: dict,
    n_clusters: int,
    theory: np.ndarray,
    x_theory: np.ndarray,
    *,
    label: str | None = None,
) -> plt.Figure:
    plt.rcParams.update(PAPER_RC)
    fig, ax = plt.subplots(figsize=(4.4, 3.6), layout="constrained")
    ax.errorbar(
        stacked["x_mid"],
        stacked["fhat"],
        yerr=stacked["sem"],
        fmt="o",
        ms=4,
        color="0.15",
        ecolor="0.45",
        elinewidth=0.8,
        capsize=1.5,
        zorder=3,
        label=label or rf"L1\_m9 fiducial, $q>5$ ($N={n_clusters}$)",
    )
    ax.plot(x_theory, theory, color="#d95f02", lw=1.6, zorder=2, label=r"A10 $y^{\rm cyl}$")
    ax.axvline(1.0, color="0.6", lw=0.7, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta/\theta_{500}$")
    ax.set_ylabel(r"$y^{\rm cyl}(\theta)\big/\bigl[Y_{500}^{\rm cyl}/(\pi\theta_{500}^{2})\bigr]$")
    ax.legend(frameon=False, loc="upper right")
    return fig


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        out = stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out, flush=True)
    plt.close(fig)


def read_ymap() -> np.ndarray:
    import healpy as hp

    print("reading y map", YMAP, flush=True)
    return np.asarray(hp.read_map(YMAP, dtype=np.float32), dtype=np.float32)


def main() -> Path:
    n_jobs = default_n_jobs()
    path = catalogue_path()
    print("loading", path, flush=True)
    clusters = load_clusters(path)
    bins = split_mass_bins(clusters)
    print(
        f"n_jobs={n_jobs}  N={len(clusters)}  bins: "
        + ", ".join(f"[{lo:.2e},{hi:.2e}) n={len(chunk)}" for lo, hi, chunk in bins),
        flush=True,
    )
    ymap = read_ymap()
    theory = a10_cyl_at(X_THEORY)
    map_panels = []
    a10_panels = []
    for lo, hi, chunk in bins:
        rows = measure_bin(ymap, chunk, n_jobs=n_jobs)
        s_map = _reduce_stack(rows, X_EDGES)
        s_a10 = _reduce_stack(rows, X_EDGES, a10_norms(chunk))
        print(
            f"stacking {mass_label(lo, hi)} N={s_map['n']}  "
            f"A_med={s_a10['A_median']:.3f}",
            flush=True,
        )
        map_panels.append((lo, hi, s_map))
        a10_panels.append((lo, hi, s_a10))
    _save(build_figure(map_panels, theory, X_THEORY, vs_a10=False), output_stem_map())
    stem = output_stem()
    _save(build_figure(a10_panels, theory, X_THEORY, vs_a10=True), stem)
    qgt5 = clusters.loc[clusters["q_from_aperture"].to_numpy(np.float64) > Q_MIN]
    print(f"q>{Q_MIN:g} N={len(qgt5)}", flush=True)
    stacked_q = stack_normalized(
        ymap,
        qgt5["theta_rot_rad"].to_numpy(np.float64),
        qgt5["phi_rot_rad"].to_numpy(np.float64),
        qgt5["theta_500_arcmin"].to_numpy(np.float64),
        X_EDGES,
        n_jobs=n_jobs,
    )
    print(f"stacking q>{Q_MIN:g}  N={stacked_q['n']}", flush=True)
    _save(build_topn_figure(stacked_q, stacked_q["n"], theory, X_THEORY), output_stem_qgt5())
    ranked = qgt5.sort_values("q_from_aperture", ascending=False)
    top = ranked.head(1000)
    stacked = stack_normalized(
        ymap,
        top["theta_rot_rad"].to_numpy(np.float64),
        top["phi_rot_rad"].to_numpy(np.float64),
        top["theta_500_arcmin"].to_numpy(np.float64),
        X_EDGES,
        n_jobs=n_jobs,
    )
    print(f"stacking top 1000 by q  N={stacked['n']}", flush=True)
    _save(
        build_topn_figure(
            stacked,
            stacked["n"],
            theory,
            X_THEORY,
            label=rf"L1\_m9 fiducial, top $1000$ by $q$",
        ),
        output_stem_top(1000),
    )
    return stem.with_suffix(".pdf")


if __name__ == "__main__":
    main()
