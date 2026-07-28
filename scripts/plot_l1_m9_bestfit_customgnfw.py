"""Plot L1_m9 full-sky tSZ bandpowers with the best-fit custom-GNFW model."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flamingo.inference.l1_m9 import L1M9CustomGNFWTheory

REPO = Path(__file__).resolve().parents[1]
DATA_FILE = REPO / "data_paper/binned_bandpowers/Dl_yy_L1_m9_fullsky_binned_18.txt"
OUTPUT_STEM = REPO / "figures/l1_m9_fullsky_bestfit_customgnfw"
A_SZ = -4.1095805
ALPHA_SZ = 0.97447729
B = 1.41
DISPLAY_SCALE = 1e12


def load_plot_data(
    data_file: str | Path = DATA_FILE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return empirical and best-fit total spectra in displayed 1e12 D_ell units."""
    empirical = np.loadtxt(data_file)
    if empirical.shape != (18, 2):
        raise ValueError(f"expected 18 two-column bandpowers, got {empirical.shape}")

    theory = L1M9CustomGNFWTheory()
    theory.initialize()
    spectrum = theory.evaluate_spectrum(A_SZ, ALPHA_SZ)
    np.testing.assert_allclose(
        spectrum["total"],
        spectrum["1h"] + spectrum["2h"],
        rtol=1e-14,
    )

    empirical_ell = np.asarray(empirical[:, 0], dtype=float)
    empirical_dl = np.asarray(empirical[:, 1], dtype=float)
    theory_ell = np.asarray(spectrum["ell"], dtype=float)
    theory_dl = np.asarray(spectrum["total"], dtype=float) * DISPLAY_SCALE
    for name, values in (
        ("empirical ell", empirical_ell),
        ("empirical D_ell", empirical_dl),
        ("theory ell", theory_ell),
        ("theory D_ell", theory_dl),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values")
        if not np.all(values > 0.0):
            raise ValueError(f"{name} contains non-positive values")
    return empirical_ell, empirical_dl, theory_ell, theory_dl


def make_figure(
    empirical_ell: np.ndarray,
    empirical_dl: np.ndarray,
    theory_ell: np.ndarray,
    theory_dl: np.ndarray,
) -> plt.Figure:
    """Draw the empirical full-sky spectrum and best-fit total theory."""
    figure, axis = plt.subplots(figsize=(6.6, 5.2))
    axis.loglog(
        empirical_ell,
        empirical_dl,
        color="black",
        marker="o",
        markersize=4.2,
        linewidth=1.6,
        label="FLAMINGO L1_m9 full sky",
        zorder=3,
    )
    axis.loglog(
        theory_ell,
        theory_dl,
        color="#c23b32",
        linewidth=2.2,
        label="customGNFW best fit",
    )
    axis.set_xlabel(r"multipole $\ell$")
    axis.set_ylabel(r"$10^{12}\,\ell(\ell+1)C_\ell^{yy}/(2\pi)$")
    axis.set_title("Full-sky total tSZ power spectrum (D3A)")
    axis.grid(which="both", alpha=0.25, linewidth=0.5)
    axis.legend(frameon=False)
    axis.text(
        0.04,
        0.96,
        (
            rf"$A_{{\rm SZ}}={A_SZ:.7f}$"
            "\n"
            rf"$\alpha_{{\rm SZ}}={ALPHA_SZ:.8f}$"
            "\n"
            rf"$B={B:.2f}$"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    figure.tight_layout()
    return figure


def save_figure(
    figure: plt.Figure,
    output_stem: str | Path = OUTPUT_STEM,
) -> tuple[Path, Path]:
    """Save the comparison figure as PNG and PDF."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_stem.with_suffix(".png"),
        output_stem.with_suffix(".pdf"),
    )
    for output in outputs:
        figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return outputs


if __name__ == "__main__":
    plot_data = load_plot_data()
    output_paths = save_figure(make_figure(*plot_data))
    for output_path in output_paths:
        print(f"wrote {output_path.relative_to(REPO)}", flush=True)
