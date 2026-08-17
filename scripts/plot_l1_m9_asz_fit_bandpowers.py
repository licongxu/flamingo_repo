"""L1_m9 A_SZ fits: data vs 1h+2h theory, and a 1h/2h split.

Uses the existing A_SZ-only posterior summary. Does not swap measured P(k)
into the 2-halo term.

    python scripts/plot_l1_m9_asz_fit_bandpowers.py
"""
from __future__ import annotations

import json
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

from flamingo.inference.l1_m9 import load_bandpower_likelihood  # noqa: E402
from flamingo.inference.masked_ps import MaskedTSZTheory  # noqa: E402
from scripts.iterate_l1_m9_asz_covariance import FIXED  # noqa: E402
from scripts.run_masked_ps_asz_only_chains import CASES  # noqa: E402

FIGURES = REPO / "figures" / "masked_ps"
_CHAIN_CANDIDATES = (
    REPO / "chains" / "l1_m9_qfrommap_asz_only",
    Path("/scratch/scratch-lxu/flamingo_repo/chains/l1_m9_qfrommap_asz_only"),
)
ELL_MIN = 10.0
ELL_MAX = 959.5
PANELS = (
    ("fullsky", "full sky"),
    ("qgt50", r"$q>50$"),
    ("qgt20", r"$q>20$"),
    ("qgt10", r"$q>10$"),
    ("qgt5", r"$q>5$"),
)
COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
PAPER_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 15,
    "legend.fontsize": 10,
    "text.latex.preamble": r"\usepackage{amsmath}",
}


def chains_dir() -> Path:
    for path in _CHAIN_CANDIDATES:
        if (path / "posterior_summary.json").is_file():
            return path
    raise FileNotFoundError("missing l1_m9_qfrommap_asz_only/posterior_summary.json")


def _load_summary() -> dict:
    return json.loads((chains_dir() / "posterior_summary.json").read_text())["cases"]


def evaluate_terms(case: str, a_sz: float) -> dict:
    preflight = json.loads((chains_dir() / "preflight.json").read_text())
    cfg = preflight["cases"][case]
    ell, observed, covariance, _ = load_bandpower_likelihood(
        cfg["data_file"], cfg["covariance_file"], data_scale=1e-12
    )
    theory = MaskedTSZTheory({"q_cat": CASES[case]}, timing=True)
    terms = theory.evaluate_bandpowers(A_SZ=a_sz, **FIXED)
    one = np.asarray(terms["1h"], dtype=float)
    two = np.asarray(terms["2h"], dtype=float)
    return {
        "ell": np.asarray(ell, dtype=float),
        "observed": observed,
        "err": np.sqrt(np.diag(covariance)),
        "1h": one,
        "2h": two,
        "total": one + two,
    }


def _save(fig: plt.Figure, stem: Path) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        out = stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out, flush=True)
    plt.close(fig)


def build_data_theory_figure(fits: dict[str, dict]) -> plt.Figure:
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
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ELL_MIN, ELL_MAX)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$10^{12}D_\ell^{yy}$")
    ax.legend(loc="lower right", frameon=False, handlelength=2.0)
    fig.tight_layout()
    return fig


def build_1h_2h_figure(fits: dict[str, dict]) -> plt.Figure:
    from matplotlib.lines import Line2D

    plt.rcParams.update(PAPER_RC)
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    scale = 1e12
    for (case, label), color in zip(PANELS, COLORS):
        fit = fits[case]
        ax.plot(fit["ell"], scale * fit["observed"], "o", color=color, ms=3.8, zorder=3)
        ax.plot(fit["ell"], scale * fit["total"], color=color, lw=2.0, zorder=2, label=label)
        ax.plot(fit["ell"], scale * fit["1h"], color=color, ls=":", lw=1.6, zorder=1)
        ax.plot(fit["ell"], scale * fit["2h"], color=color, ls="--", lw=1.6, zorder=1)
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
    ]
    labels += [r"1h+2h", r"1h", r"2h"]
    ax.legend(handles, labels, loc="lower right", frameon=False, handlelength=2.2)
    fig.tight_layout()
    return fig


def main() -> Path:
    summaries = _load_summary()
    fits = {}
    for case, _ in PANELS:
        a_sz = summaries[case]["best_likelihood_sample"]["A_SZ"]
        print(f"evaluating {case} at A_SZ={a_sz:.5f}", flush=True)
        fits[case] = evaluate_terms(case, a_sz)
    _save(build_data_theory_figure(fits), FIGURES / "l1_m9_asz_fit_bandpowers_qfrommap")
    _save(build_1h_2h_figure(fits), FIGURES / "l1_m9_asz_fit_bandpowers_qfrommap_1h_2h")
    return FIGURES / "l1_m9_asz_fit_bandpowers_qfrommap_1h_2h.pdf"


if __name__ == "__main__":
    main()
