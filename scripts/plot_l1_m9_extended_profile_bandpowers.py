"""Fixed-A_SZ overlay: fiducial GNFW vs a shallower (more extended) outer slope.

No MCMC. The q-mask stays on the fiducial y0; only the Cl-profile beta changes.

    python scripts/plot_l1_m9_extended_profile_bandpowers.py
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
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from flamingo.inference.l1_m9 import load_bandpower_likelihood  # noqa: E402
from flamingo.inference.masked_ps import GNFW_SHAPE, MaskedTSZTheory  # noqa: E402
from scripts.iterate_l1_m9_asz_covariance import FIXED  # noqa: E402
from scripts.plot_l1_m9_asz_fit_bandpowers import (  # noqa: E402
    COLORS,
    ELL_MAX,
    ELL_MIN,
    FIGURES,
    PANELS,
    PAPER_RC,
    _load_summary,
    chains_dir,
)
from scripts.run_masked_ps_asz_only_chains import CASES  # noqa: E402

STEM = FIGURES / "l1_m9_extended_profile_qfrommap"
BETAS = (
    (float(GNFW_SHAPE["beta"]), "-", rf"fiducial $\beta={GNFW_SHAPE['beta']}$"),
    (4.0, "--", r"extended $\beta=4$"),
    (3.0, ":", r"extended $\beta=3$"),
)


def _load_data(case: str) -> dict:
    cfg = json.loads((chains_dir() / "preflight.json").read_text())["cases"][case]
    ell, observed, covariance, _ = load_bandpower_likelihood(
        cfg["data_file"], cfg["covariance_file"], data_scale=1e-12
    )
    return {
        "ell": np.asarray(ell, dtype=float),
        "observed": observed,
        "err": np.sqrt(np.diag(covariance)),
    }


def _evaluate(theory: MaskedTSZTheory, a_sz: float, beta: float) -> np.ndarray:
    theory._profile_seed = theory._profile_seed.update(beta=beta)
    terms = theory.evaluate_bandpowers(A_SZ=a_sz, **FIXED)
    return np.asarray(terms["1h"], dtype=float) + np.asarray(terms["2h"], dtype=float)


def build_figure(rows: dict[str, dict]) -> plt.Figure:
    plt.rcParams.update(PAPER_RC)
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    scale = 1e12
    for (case, label), color in zip(PANELS, COLORS):
        row = rows[case]
        ax.errorbar(
            row["ell"],
            scale * row["observed"],
            yerr=scale * row["err"],
            fmt="o",
            color=color,
            ms=3.8,
            capsize=2,
            zorder=3,
        )
        for beta, ls, _ in BETAS:
            ax.plot(
                row["ell"],
                scale * row["theory"][beta],
                color=color,
                ls=ls,
                lw=2.0,
                zorder=2,
                label=label if ls == "-" else None,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ELL_MIN, ELL_MAX)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$10^{12}D_\ell^{yy}$")
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color="0.2", lw=2.0, ls=ls) for _, ls, _ in BETAS
    ]
    labels += [name for _, _, name in BETAS]
    ax.legend(handles, labels, loc="lower right", frameon=False, handlelength=2.2)
    fig.tight_layout()
    return fig


def main() -> Path:
    summaries = _load_summary()
    data = {case: _load_data(case) for case, _ in PANELS}
    theory = MaskedTSZTheory({"q_cat": None}, timing=True)
    rows: dict[str, dict] = {}
    for case, _ in PANELS:
        a_sz = summaries[case]["best_likelihood_sample"]["A_SZ"]
        theory.q_cat = CASES[case]
        print(f"evaluating {case} at A_SZ={a_sz:.5f}", flush=True)
        row = dict(data[case])
        row["theory"] = {}
        for beta, _, name in BETAS:
            total = _evaluate(theory, a_sz, beta)
            row["theory"][beta] = total
            i0 = 0
            print(
                f"  {name}: 1e12 D(ell={row['ell'][i0]:.0f})="
                f"{1e12 * total[i0]:.4e}  data/theory="
                f"{row['observed'][i0] / total[i0]:.3f}",
                flush=True,
            )
        rows[case] = row
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig = build_figure(rows)
    for suffix in ("png", "pdf"):
        out = STEM.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out, flush=True)
    plt.close(fig)
    return STEM.with_suffix(".pdf")


if __name__ == "__main__":
    main()
