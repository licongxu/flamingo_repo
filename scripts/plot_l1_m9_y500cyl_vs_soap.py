"""L1_m9 GNFW-inferred Y_500^sph (no background) vs SOAP Y_500^sph.

Catalogue cylindrical aperture (raw HEALPix sum, no BG) converted with the
isolated-GNFW factor Y_sph(<R_500)/Y_cyl(<R_500).

    python scripts/plot_l1_m9_y500cyl_vs_soap.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.plot_l1_m9_y5r500_inferred_vs_truth import (  # noqa: E402
    CATALOGUE,
    FIGURES,
    PAPER_RC,
    Q_MIN,
    sph_over_cyl,
    y_arcmin2_to_mpc2,
)

COLUMNS = (
    "Y_500c_Mpc2",
    "Y_500cyl_arcmin2",
    "R_500c_Mpc",
    "theta_500_arcmin",
    "q_from_aperture",
)


def output_stem() -> Path:
    return FIGURES / "l1_m9_y500_inferred_sph_vs_soap_nobg"


def load_catalogue(path: Path = CATALOGUE) -> pd.DataFrame:
    frame = pd.read_csv(path, comment="#", usecols=list(COLUMNS))
    soap = frame["Y_500c_Mpc2"].to_numpy(np.float64)
    if not np.all(np.isfinite(soap)) or not np.all(soap > 0.0):
        raise ValueError("SOAP Y_500c_Mpc2 must be finite and positive")
    return frame


def cyl_mpc2(frame: pd.DataFrame) -> np.ndarray:
    """Catalogue Y_500^cyl = ∫_{θ<θ_500} y dΩ, in Mpc². No background term."""
    return y_arcmin2_to_mpc2(
        frame["Y_500cyl_arcmin2"].to_numpy(np.float64),
        frame["R_500c_Mpc"].to_numpy(np.float64),
        frame["theta_500_arcmin"].to_numpy(np.float64),
    )


def inferred_sph_mpc2(frame: pd.DataFrame) -> np.ndarray:
    """GNFW Y_sph = f_cyl→sph × Y_cyl, still no background term."""
    return sph_over_cyl(1.0) * cyl_mpc2(frame)


def build_figure(soap: np.ndarray, inferred: np.ndarray) -> plt.Figure:
    plt.rcParams.update(PAPER_RC)
    fig, ax = plt.subplots(figsize=(6.4, 6.0), layout="constrained")
    ok = np.isfinite(soap) & np.isfinite(inferred) & (soap > 0.0) & (inferred > 0.0)
    x, y = soap[ok], inferred[ok]
    f = sph_over_cyl(1.0)
    ax.plot(
        x,
        y,
        ".",
        ms=2.4,
        alpha=0.45,
        color="#1b9e77",
        rasterized=True,
        zorder=2,
        label=rf"$Y_{{500c}}$, $f={f:.3f}$, median ${np.median(y / x):.2f}$",
    )
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], color="0.2", lw=1.0, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"SOAP $Y_{500}^{\rm sph}\,[{\rm Mpc}^{2}]$")
    ax.set_ylabel(r"map $Y^{\rm sph}=f_{\rm cyl\to sph}\,Y^{\rm cyl}\,[{\rm Mpc}^{2}]$ (no BG)")
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
    print("loading catalogue", CATALOGUE, flush=True)
    frame = load_catalogue()
    frame = frame.loc[frame["q_from_aperture"].to_numpy(np.float64) > Q_MIN].copy()
    soap = frame["Y_500c_Mpc2"].to_numpy(np.float64)
    inferred = inferred_sph_mpc2(frame)
    ok = np.isfinite(inferred) & (inferred > 0.0)
    print(
        f"N={ok.sum():,} / {len(frame):,}  "
        f"f={sph_over_cyl(1.0):.5f}  "
        f"median Y_sph_inf/Y_SOAP={np.median(inferred[ok] / soap[ok]):.3f}",
        flush=True,
    )
    stem = output_stem()
    _save(build_figure(soap, inferred), stem)
    return stem.with_suffix(".pdf")


if __name__ == "__main__":
    main()
