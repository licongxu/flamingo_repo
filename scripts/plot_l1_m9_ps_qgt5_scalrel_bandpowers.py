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
from scripts.run_l1_m9_ps_qgt5_scalrel_chain import CASE, CHAINS  # noqa: E402
from scripts.run_masked_ps_chains import data_files, load_converged_artifacts  # noqa: E402

STEM = REPO / "figures" / "masked_ps" / "l1_m9_ps_qgt5_scalrel_bestfit_qfrommap"
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


def evaluate(params: dict[str, float]) -> dict:
    """Evaluate masked 1h+2h theory at the MAP scalrel point."""
    sys.path.insert(0, str(HMFAST_SRC))
    from flamingo.inference.l1_m9 import load_bandpower_likelihood
    from flamingo.inference.masked_ps import MaskedTSZTheory

    data, covariance = data_files(CASE, load_converged_artifacts()["covariance_paths"])
    ell, observed, cov, inverse = load_bandpower_likelihood(
        data, covariance, data_scale=1e-12
    )
    theory = MaskedTSZTheory({"q_cat": 5.0}, timing=True)
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
    return {
        "ell": np.asarray(ell, dtype=float),
        "observed": observed,
        "err": np.sqrt(np.diag(cov)),
        "1h": one,
        "2h": two,
        "total": total,
        "chi2": float(residual @ inverse @ residual),
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


def main() -> dict:
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["JAX_ENABLE_X64"] = "1"
    params = map_params()
    print("MAP", {k: params[k] for k in (*SCALREL, "chi2")}, flush=True)
    fit = evaluate(params)
    print(f"evaluated chi2={fit['chi2']:.3f}", flush=True)
    _save(build_figure(fit, params), STEM)
    return {"params": params, "fit": {k: fit[k] for k in ("ell", "observed", "total", "chi2")}}


if __name__ == "__main__":
    main()
