"""L1_m9 Y_5R500 inferred vs SOAP truth.

Spherical SOAP Y_5R500 is compared to (1) a cylindrical Compton-y map
aperture at 5 R_500 converted to spherical with the custom-GNFW shape,
and (2) the spherical integral of the full-sky tSZ best-fit parametric
GNFW (A_SZ and P0).

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
from flamingo.catalogue.frame import efunc  # noqa: E402
from flamingo.inference.masked_ps import B_HYDROSTATIC, GNFW_SHAPE  # noqa: E402
from scripts.iterate_l1_m9_asz_covariance import FIXED  # noqa: E402

CATALOGUE = Path(
    "/rds/rds-lxu/flamingo/L1_m9/catalogues/"
    "halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommap.csv"
)
YMAP = Path("/rds/rds-lxu/flamingo/L1_m9/maps/y_unlensed_L1_m9_lc0_nside4096.fits")
FIGURES = REPO / "figures" / "diagnostics"
# Full-sky A_SZ-only best-likelihood sample (paint_l1_m9 / asz_only/fullsky).
A_SZ = -4.1075214
ALPHA_SZ = float(FIXED["alpha_SZ"])
B = float(B_HYDROSTATIC)
H0 = float(FIXED["H0"])
X_OUT = 5.0
Q_MIN = 5.0
I_LOS_A10 = 0.470502095
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


def sph_over_cyl(x_out: float = X_OUT, n: int = 4000, s_max: float = 30.0, **shape) -> float:
    """Y_sph(<x_out) / Y_cyl(<x_out) for an isolated GNFW; amplitude cancels."""
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


def i_sph_shape(x_out: float = X_OUT, n: int = 8000) -> float:
    x = np.linspace(0.0, x_out, n)
    return float(np.trapezoid(x**2 * gnfw_p(x, **{**GNFW_SHAPE, "P0": 1.0}), x))


def y0_parametric(m: np.ndarray, z: np.ndarray) -> np.ndarray:
    h = H0 / 100.0
    m_tilde = (np.asarray(m, dtype=float) * h / B) / (0.7 * 3e14)
    return (10.0 ** A_SZ) * m_tilde ** ALPHA_SZ * efunc(z) ** 2 * (h / 0.7) ** (-0.5)


def model_y5r500_mpc2(m: np.ndarray, z: np.ndarray, r500: np.ndarray) -> np.ndarray:
    """Spherical Y(<5 R_500) of the full-sky parametric GNFW, in Mpc^2."""
    return y0_parametric(m, z) * 2.0 * np.pi * np.asarray(r500, dtype=float) ** 2 * i_sph_shape() / I_LOS_A10


def y_arcmin2_to_mpc2(y_arcmin2: np.ndarray, r500: np.ndarray, theta500_arcmin: np.ndarray) -> np.ndarray:
    return np.asarray(y_arcmin2, dtype=float) * (np.asarray(r500, dtype=float) / np.asarray(theta500_arcmin, dtype=float)) ** 2


def load_catalogue(path: Path = CATALOGUE) -> pd.DataFrame:
    cols = [
        "z",
        "M_500c_Msun",
        "R_500c_Mpc",
        "theta_rot_rad",
        "phi_rot_rad",
        "theta_500_arcmin",
        "Y_5R500c_Mpc2",
        "q_from_aperture",
    ]
    frame = pd.read_csv(path, comment="#", usecols=cols)
    if not np.all(np.isfinite(frame["Y_5R500c_Mpc2"])) or not np.all(frame["Y_5R500c_Mpc2"] > 0.0):
        raise ValueError("SOAP Y_5R500c must be finite and positive")
    return frame


def map_y5r500_cyl_arcmin2(frame: pd.DataFrame, ymap: np.ndarray) -> np.ndarray:
    radius = 5.0 * frame["theta_500_arcmin"].to_numpy(np.float64) / ARCMIN_PER_RAD
    y500, _ = aperture_y500(
        ymap,
        frame["theta_rot_rad"].to_numpy(np.float64),
        frame["phi_rot_rad"].to_numpy(np.float64),
        radius,
    )
    return y500


def _panel(ax, truth, inferred, label):
    ok = np.isfinite(truth) & np.isfinite(inferred) & (truth > 0.0) & (inferred > 0.0)
    x, y = truth[ok], inferred[ok]
    ax.plot(x, y, ".", ms=2.4, alpha=0.4, color="#1b9e77", rasterized=True, zorder=2)
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], color="0.2", lw=1.0, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.text(0.04, 0.96, label, transform=ax.transAxes, va="top")
    ax.text(
        0.04,
        0.88,
        rf"median ratio ${np.median(y / x):.2f}$",
        transform=ax.transAxes,
        va="top",
    )


def build_figure(truth, y_map, y_model) -> plt.Figure:
    plt.rcParams.update(PAPER_RC)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), sharex=True, sharey=True, layout="constrained")
    _panel(axes[0], truth, y_map, rf"map $Y_{{\rm cyl}}\to Y_{{\rm sph}}$, $q>{Q_MIN:.0f}$")
    _panel(axes[1], truth, y_model, rf"GNFW $A_{{\rm SZ}}={A_SZ:.4f}$, $q>{Q_MIN:.0f}$")
    axes[0].set_xlabel(r"SOAP $Y_{5R_{500}}^{\rm sph}\,[{\rm Mpc}^{2}]$")
    axes[1].set_xlabel(r"SOAP $Y_{5R_{500}}^{\rm sph}\,[{\rm Mpc}^{2}]$")
    axes[0].set_ylabel(r"inferred $Y_{5R_{500}}\,[{\rm Mpc}^{2}]$")
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
    factor = sph_over_cyl()
    print(f"GNFW sph/cyl at 5 R500 = {factor:.5f}; N(q>{Q_MIN:g})={len(frame):,}", flush=True)
    y_model = model_y5r500_mpc2(
        frame["M_500c_Msun"].to_numpy(np.float64),
        frame["z"].to_numpy(np.float64),
        frame["R_500c_Mpc"].to_numpy(np.float64),
    )
    print("reading y map", YMAP, flush=True)
    ymap = np.asarray(hp.read_map(YMAP, dtype=np.float32), dtype=np.float32)
    ymap -= float(np.mean(ymap, dtype=np.float64))
    print(f"aperture 5 R500 for {len(frame):,} mean-subtracted map pixels", flush=True)
    y_cyl = map_y5r500_cyl_arcmin2(frame, ymap)
    y_map = factor * y_arcmin2_to_mpc2(
        y_cyl,
        frame["R_500c_Mpc"].to_numpy(np.float64),
        frame["theta_500_arcmin"].to_numpy(np.float64),
    )
    truth = frame["Y_5R500c_Mpc2"].to_numpy(np.float64)
    stem = FIGURES / "l1_m9_y5r500_inferred_vs_truth"
    _save(build_figure(truth, y_map, y_model), stem)
    return stem.with_suffix(".pdf")


if __name__ == "__main__":
    main()
