"""Paper figure: L1_m9 A_SZ-only fits with Poisson white-noise limits.

For each sky cut (full sky and ``qfrommap`` thresholds), plot the 18
Planck-style bandpowers with error bars, the best-fit one-parameter halo
model, and the Poisson (white-noise) limit obtained from
:math:`\\sum Y_{5R_{500}}^2/(4\\pi)` in the :math:`\\ell\\to0` limit.

Run::

    python scripts/plot_l1_m9_asz_fit_bandpowers.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.run_masked_ps_asz_only_chains import CASES, CHAINS, FIXED  # noqa: E402
from scripts.summarize_masked_ps_asz_only import evaluate_fit  # noqa: E402

FIGURES = REPO / "figures" / "masked_ps"
SUMMARY = CHAINS / "posterior_summary.json"
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
    "axes.titlesize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "text.latex.preamble": r"\usepackage{amsmath}",
}


def _load_summary() -> dict:
    return json.loads(SUMMARY.read_text())["cases"]


def poisson_cl(a_sz: float, q_cat: float | None) -> float:
    """Return the white-noise :math:`C_\\ell^{yy}` from :math:`\\sum Y_{5R500}^2/(4\\pi)`.

    In the :math:`\\ell\\to0` limit the masked one-halo term reduces to a
    flat Poisson contribution with
    :math:`C_\\ell = (4\\pi)^{-1}\\int \\mathrm{d}\\ln M\\,\\mathrm{d}z\\,
    (\\mathrm{d}n/\\mathrm{d}\\ln M)(\\mathrm{d}V/\\mathrm{d}z\\,\\mathrm{d}\\Omega)\\,
    W_2(M,z)\\,Y_{5R500}^2/D_A(z)^4`.
    We evaluate that limit with the same hmfast grids and profile as the fit.
    """
    import jax.numpy as jnp
    from hmfast.halos import HaloModel
    from hmfast.halos.mass_definition import MassDefinition
    from hmfast.halos.profiles import ParametricGNFWPressureProfile
    from hmfast.tracers import tSZTracer
    from hmfast.tracers.tsz_completeness import (
        build_snr_grid,
        conditional_An_undetected,
        load_sigma_y0_curve,
    )

    from flamingo.catalogue.frame import D3A_COSMOLOGY
    from flamingo.cnc import FILTER_NAME, SIGMA_Y0_FILE, SKYFRACS_FILE
    from flamingo.inference.masked_ps import GNFW_SHAPE, MASS_GRID, REDSHIFT_GRID

    halo_model = HaloModel(
        cosmology=D3A_COSMOLOGY,
        mass_definition=MassDefinition(500, "critical"),
        convert_masses=True,
        hm_consistency=False,
    )
    profile = ParametricGNFWPressureProfile(
        A_SZ=a_sz,
        alpha_SZ=FIXED["alpha_SZ"],
        B=FIXED["B"],
        **GNFW_SHAPE,
    )
    tracer = tSZTracer(profile=profile)
    coefficients, _ = load_sigma_y0_curve(
        sigma_obj_file=str(SIGMA_Y0_FILE),
        skyfr_file=str(SKYFRACS_FILE),
        filter_name=FILTER_NAME,
    )
    mass = jnp.asarray(MASS_GRID)
    redshift = jnp.asarray(REDSHIFT_GRID)
    snr = build_snr_grid(
        halo_model,
        mass,
        redshift,
        a_sz,
        FIXED["alpha_SZ"],
        FIXED["B"],
        coeff=jnp.asarray(coefficients),
    )
    q = np.inf if q_cat is None else float(q_cat)
    mask2 = conditional_An_undetected(
        snr,
        sigma_lnY=FIXED["sigma_lnY"],
        q_cat=q,
        n_power=2,
        n_grid=512,
        nsig=8.0,
    )
    ell_ref = jnp.asarray([1e-4])
    return float(
        halo_model.cl_1h_masked(
            tracer,
            None,
            ell_ref,
            mass,
            redshift,
            mask2,
            k_damp=0.0,
        )[0]
    )


def poisson_dl(ell: np.ndarray, c_l: float) -> np.ndarray:
    """Convert a flat Poisson :math:`C_\\ell` to display units :math:`10^{12}D_\\ell``."""
    ell = np.asarray(ell, dtype=float)
    return ell * (ell + 1.0) / (2.0 * np.pi) * c_l * 1e12


def _fit_artifacts(summary: dict) -> dict:
    return {"A_SZ": summary["fit_diagnostics"]["covariance_A_SZ"]}


def build_figure(summaries: dict[str, dict]) -> plt.Figure:
    plt.rcParams.update(PAPER_RC)
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ell_line = np.geomspace(ELL_MIN, ELL_MAX, 200)
    scale = 1e12

    for (case, label), color in zip(PANELS, COLORS):
        summary = summaries[case]
        a_sz = summary["best_likelihood_sample"]["A_SZ"]
        fit = evaluate_fit(case, a_sz, artifacts=_fit_artifacts(summary))
        c_poisson = poisson_cl(a_sz, CASES[case])

        ax.plot(
            fit["ell"],
            scale * fit["observed"],
            linestyle="none",
            marker="o",
            color=color,
            ms=3.8,
            zorder=3,
        )
        ax.plot(
            fit["ell"],
            scale * fit["theory"],
            color=color,
            lw=2.0,
            zorder=2,
            label=label,
        )
        ax.plot(
            ell_line,
            poisson_dl(ell_line, c_poisson),
            color=color,
            ls="--",
            lw=1.6,
            zorder=1,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ELL_MIN, ELL_MAX)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$10^{12}D_\ell^{yy}$")
    ax.grid(False)
    ax.legend(loc="lower right", frameon=False, handlelength=2.0)
    fig.tight_layout()
    return fig


def _save(fig: plt.Figure, stem: Path) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        out = stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out.relative_to(REPO)}", flush=True)
    plt.close(fig)


def main() -> Path:
    if not SUMMARY.is_file():
        raise FileNotFoundError(
            f"missing {SUMMARY}; run summarize_masked_ps_asz_only.py first"
        )
    summaries = _load_summary()
    fig = build_figure(summaries)
    stem = FIGURES / "l1_m9_asz_fit_bandpowers_qfrommap"
    _save(fig, stem)
    return stem.with_suffix(".pdf")


if __name__ == "__main__":
    main()
