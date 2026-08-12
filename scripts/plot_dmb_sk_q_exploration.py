"""Posterior S(k) vs FLAMINGO for all available DMB tSZ PS q-cuts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "figures" / "dmb"
FIGURES.mkdir(parents=True, exist_ok=True)
CHAINS = REPO / "chains" / "l1_m9_qfrommap_dmb"
DATA = Path("/rds/rds-lxu/flamingo/power_spectra")
H_FLAMINGO = 0.681

PAPER_RC = {
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
}

# High-q (milder mask) → low-q (stronger mask)
Q_CASES = [
    ("fullsky", "Full sky", "#4d4d4d"),
    ("qgt50", r"$q>50$", "#8c564b"),
    ("qgt20", r"$q>20$", "#e377c2"),
    ("qgt10", r"$q>10$", "#9467bd"),
    ("qgt5", r"$q>5$", "#1f4e79"),
]


def load_flamingo_suppression(z_key: str = "z=0.00") -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(DATA / "L1_m9.hdf5", "r") as fh, h5py.File(
        DATA / "L1_m9_DMO.hdf5", "r"
    ) as fd:
        k = np.asarray(fh[f"{z_key}/k"][:], dtype=float)
        return k * H_FLAMINGO, fh[f"{z_key}/P(k)"][:] / fd[f"{z_key}/P(k)"][:]


def load_posterior_band(case: str) -> dict[str, np.ndarray] | None:
    npz = CHAINS / case / "pk_suppression_posterior.npz"
    if not npz.is_file():
        return None
    b = np.load(npz)
    return {
        "k": np.asarray(b["k_h"], dtype=float),
        "med": np.asarray(b["ratio_median"], dtype=float),
        "lo68": np.asarray(b["ratio_p16"], dtype=float),
        "hi68": np.asarray(b["ratio_p84"], dtype=float),
        "lo95": np.asarray(b["ratio_p025"], dtype=float),
        "hi95": np.asarray(b["ratio_p975"], dtype=float),
    }


def plot_q_exploration(cases: list[tuple[str, str, str]]) -> dict[str, Path]:
    k_flam, s_flam = load_flamingo_suppression()
    available = [(c, lab, col) for c, lab, col in cases if load_posterior_band(c) is not None]
    if not available:
        raise FileNotFoundError(
            "No pk_suppression_posterior.npz found — run postprocess_dmb_masked_ps.py first."
        )

    products: dict[str, Path] = {}
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7.2, 5.0))

        ax.plot(
            k_flam,
            s_flam,
            color="#e67e22",
            lw=2.5,
            zorder=10,
            label=r"FLAMINGO $P_{\rm hydro}/P_{\rm DMO}$",
        )

        y_mins = []
        for case, label, color in available:
            band = load_posterior_band(case)
            assert band is not None
            k = band["k"]
            ax.fill_between(k, band["lo95"], band["hi95"], color=color, alpha=0.12, zorder=1)
            ax.fill_between(k, band["lo68"], band["hi68"], color=color, alpha=0.22, zorder=2)
            ax.plot(k, band["med"], color=color, lw=2.0, label=f"{label} posterior", zorder=5)
            y_mins.append(float(np.nanmin(band["lo95"])))

        ax.axhline(1.0, color="0.35", ls="--", lw=0.9, zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel(r"$k\ (h\,{\rm Mpc}^{-1})$")
        ax.set_ylabel(r"$P_{\rm DMB}(k)\,/\,P_{\rm NFW}(k)$")
        ax.set_title(r"Matter power suppression at $z=0$: posterior vs $q$-cut")
        k_lo = max(float(k.min()), float(k_flam.min()))
        k_hi = min(float(k.max()), float(k_flam.max()))
        ax.set_xlim(k_lo, k_hi)
        mask = (k_flam >= k_lo) & (k_flam <= k_hi)
        y_mins.append(0.95 * float(np.nanmin(s_flam[mask])) if mask.any() else 0.70)
        ax.set_ylim(min(0.70, *y_mins), 1.10)
        ax.legend(frameon=False, loc="lower left", ncol=1)
        fig.tight_layout()

        for suffix in ("png", "pdf"):
            path = FIGURES / f"dmb_sk_q_exploration.{suffix}"
            fig.savefig(path, bbox_inches="tight", dpi=200)
            products[suffix] = path
            print(f"Saved: {path}")
        plt.close(fig)

    print(f"\n{'case':>10s} {'S(0.1)':>8s} {'S(1)':>8s} {'S(3)':>8s} {'S(10)':>8s}")
    for case, label, _ in available:
        band = load_posterior_band(case)
        assert band is not None
        k = band["k"]
        vals = []
        for kt in [0.1, 1.0, 3.0, 10.0]:
            i = np.argmin(np.abs(k - kt))
            vals.append(band["med"][i])
        print(f"{case:>10s} {vals[0]:8.4f} {vals[1]:8.4f} {vals[2]:8.4f} {vals[3]:8.4f}")
    j = int(np.argmin(np.abs(k_flam - 0.1)))
    print(f"{'FLAMINGO':>10s} {s_flam[j]:8.4f}", end="")
    for kt in [1.0, 3.0, 10.0]:
        j = int(np.argmin(np.abs(k_flam - kt)))
        print(f" {s_flam[j]:8.4f}", end="")
    print()

    return products


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case",
        action="append",
        choices=[c for c, _, _ in Q_CASES],
        help="Cases to include (default: all with posterior npz)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.case:
        selected = {(c, lab, col) for c, lab, col in Q_CASES if c in args.case}
        cases = [x for x in Q_CASES if x[0] in {c for c, _, _ in selected}]
    else:
        cases = Q_CASES
    plot_q_exploration(cases)


if __name__ == "__main__":
    main()
