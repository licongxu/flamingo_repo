"""L1_m9 map Y^cyl = int y dOmega vs official SOAP spherical Y.

    python scripts/plot_l1_m9_y5r500_inferred_vs_truth.py
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from flamingo.aperture_snr import aperture_y500  # noqa: E402
from flamingo.inference.masked_ps import GNFW_SHAPE  # noqa: E402

CATALOGUE = Path(
    "/rds/rds-lxu/flamingo/L1_m9/catalogues/"
    "halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommap.csv"
)
YMAP = Path("/rds/rds-lxu/flamingo/L1_m9/maps/y_unlensed_L1_m9_lc0_nside4096.fits")
FIGURES = REPO / "figures" / "diagnostics"
Q_MIN = 5.0
ARCMIN_PER_RAD = 180.0 * 60.0 / np.pi
PAPER_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "legend.fontsize": 9,
    "text.latex.preamble": r"\usepackage{amsmath}",
}


def gnfw_p(x: np.ndarray, *, P0: float, c500: float, gamma: float, alpha: float, beta: float) -> np.ndarray:
    cx = c500 * np.clip(np.asarray(x, dtype=float), 1e-6, None)
    return P0 * cx ** (-gamma) * (1.0 + cx ** alpha) ** (-(beta - gamma) / alpha)


def sph_over_cyl(x_out: float, n: int = 4000, s_max: float = 30.0, **shape) -> float:
    """cyl→sph factor Y_sph(<x_out)/Y_cyl(<x_out) for an isolated GNFW.

    Amplitude cancels. The cylinder uses an infinite line of sight, so it
    includes the r > x_out caps that the sphere drops.
    """
    shape = {**GNFW_SHAPE, **shape, "P0": 1.0}
    b = np.linspace(0.0, x_out, n)
    s = np.linspace(0.0, s_max, n)
    x = np.linspace(0.0, x_out, n)
    i_sph = 4.0 * np.pi * np.trapezoid(x**2 * gnfw_p(x, **shape), x)
    los = 2.0 * np.trapezoid(gnfw_p(np.sqrt(b[:, None] ** 2 + s[None, :] ** 2), **shape), s, axis=1)
    i_cyl = 2.0 * np.pi * np.trapezoid(b * los, b)
    if not np.isfinite(i_sph) or not np.isfinite(i_cyl) or i_cyl <= 0.0:
        raise ValueError("GNFW aperture integrals must be finite and positive")
    return float(i_sph / i_cyl)


def y_arcmin2_to_mpc2(y_arcmin2: np.ndarray, r500: np.ndarray, theta500_arcmin: np.ndarray) -> np.ndarray:
    """Convert ∫ y dΩ from arcmin² to Mpc² with D_A = R_500 / theta_500."""
    return np.asarray(y_arcmin2, dtype=float) * (np.asarray(r500, dtype=float) / np.asarray(theta500_arcmin, dtype=float)) ** 2


def load_catalogue(path: Path = CATALOGUE) -> pd.DataFrame:
    cols = [
        "R_500c_Mpc",
        "theta_rot_rad",
        "phi_rot_rad",
        "theta_500_arcmin",
        "Y_500c_Mpc2",
        "Y_5R500c_Mpc2",
        "q_from_aperture",
    ]
    frame = pd.read_csv(path, comment="#", usecols=cols)
    for column in ("Y_500c_Mpc2", "Y_5R500c_Mpc2"):
        values = frame[column].to_numpy(np.float64)
        if not np.all(np.isfinite(values)) or not np.all(values > 0.0):
            raise ValueError(f"SOAP {column} must be finite and positive")
    return frame


def inferred_cyl_mpc2(frame: pd.DataFrame, ymap: np.ndarray, r_mult: float) -> np.ndarray:
    """Cylindrical Y = ∫_{θ < r_mult θ_500} y dΩ, in Mpc²."""
    radius = r_mult * frame["theta_500_arcmin"].to_numpy(np.float64) / ARCMIN_PER_RAD
    y_arcmin2, _ = aperture_y500(
        ymap,
        frame["theta_rot_rad"].to_numpy(np.float64),
        frame["phi_rot_rad"].to_numpy(np.float64),
        radius,
    )
    return y_arcmin2_to_mpc2(
        y_arcmin2,
        frame["R_500c_Mpc"].to_numpy(np.float64),
        frame["theta_500_arcmin"].to_numpy(np.float64),
    )


def inferred_sph_mpc2(frame: pd.DataFrame, ymap: np.ndarray, r_mult: float) -> np.ndarray:
    """Map Y^cyl converted to spherical with the GNFW cyl→sph factor."""
    return sph_over_cyl(r_mult) * inferred_cyl_mpc2(frame, ymap, r_mult)


def _series(ax, truth, inferred, color, label):
    ok = np.isfinite(truth) & np.isfinite(inferred) & (truth > 0.0) & (inferred > 0.0)
    x, y = truth[ok], inferred[ok]
    ax.plot(
        x,
        y,
        ".",
        ms=2.4,
        alpha=0.45,
        color=color,
        rasterized=True,
        zorder=2,
        label=rf"{label}, median ${np.median(y / x):.2f}$",
    )
    return x, y


def build_figure(truth_500, inf_500, truth_5r, inf_5r) -> plt.Figure:
    plt.rcParams.update(PAPER_RC)
    fig, ax = plt.subplots(figsize=(6.4, 6.0), layout="constrained")
    x500, y500 = _series(ax, truth_500, inf_500, "#1b9e77", rf"$Y_{{500c}}$, $f={sph_over_cyl(1.0):.3f}$")
    x5, y5 = _series(ax, truth_5r, inf_5r, "#d95f02", rf"$Y_{{5R_{{500}}}}$, $f={sph_over_cyl(5.0):.3f}$")
    lo = min(x500.min(), y500.min(), x5.min(), y5.min())
    hi = max(x500.max(), y500.max(), x5.max(), y5.max())
    ax.plot([lo, hi], [lo, hi], color="0.2", lw=1.0, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"SOAP $Y^{\rm sph}\,[{\rm Mpc}^{2}]$")
    ax.set_ylabel(r"map $Y^{\rm sph}=f_{\rm cyl\to sph}\,Y^{\rm cyl}\,[{\rm Mpc}^{2}]$")
    ax.legend(loc="upper left", frameon=False, markerscale=4)
    return fig


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        out = stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out, flush=True)
    plt.close(fig)


def main() -> Path:
    import healpy as hp

    print("loading catalogue", CATALOGUE, flush=True)
    frame = load_catalogue()
    frame = frame.loc[frame["q_from_aperture"].to_numpy(np.float64) > Q_MIN].copy()
    print(
        f"cyl→sph  R500={sph_over_cyl(1.0):.5f}  5R500={sph_over_cyl(5.0):.5f}; "
        f"N(q>{Q_MIN:g})={len(frame):,}",
        flush=True,
    )
    print("reading y map", YMAP, flush=True)
    ymap = np.asarray(hp.read_map(YMAP, dtype=np.float32), dtype=np.float32)
    ymap -= float(np.mean(ymap, dtype=np.float64))
    inf_500 = inferred_sph_mpc2(frame, ymap, 1.0)
    inf_5r = inferred_sph_mpc2(frame, ymap, 5.0)
    stem = FIGURES / "l1_m9_y5r500_inferred_vs_truth"
    _save(
        build_figure(
            frame["Y_500c_Mpc2"].to_numpy(np.float64),
            inf_500,
            frame["Y_5R500c_Mpc2"].to_numpy(np.float64),
            inf_5r,
        ),
        stem,
    )
    return stem.with_suffix(".pdf")


if __name__ == "__main__":
    main()
