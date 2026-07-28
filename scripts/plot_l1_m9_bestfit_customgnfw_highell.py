"""High-ell FLAMINGO L1_m9 tSZ spectrum with best-fit custom-GNFW theory."""

from __future__ import annotations

from pathlib import Path

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flamingo.inference.l1_m9 import L1M9CustomGNFWTheory
from flamingo.theory.clyy import cl_yy

if __package__:
    from scripts.plot_l1_m9_bestfit_customgnfw import (
        ALPHA_SZ,
        A_SZ,
        B,
        DISPLAY_SCALE,
        save_figure,
    )
else:
    from plot_l1_m9_bestfit_customgnfw import (
        ALPHA_SZ,
        A_SZ,
        B,
        DISPLAY_SCALE,
        save_figure,
    )


REPO = Path(__file__).resolve().parents[1]
MAP_FILE = Path("/rds/rds-lxu/flamingo/L1_m9/maps/y_unlensed_L1_m9_lc0_nside4096.fits")
OUTPUT_STEM = REPO / "figures/l1_m9_fullsky_bestfit_customgnfw_highell"
BANDPOWER_FILE = (
    REPO
    / "data_paper/binned_bandpowers/"
    "Dl_yy_L1_m9_fiducial_fullsky_logbins_dln0p4_lmax10000_"
    "pixwin_deconvolved.txt"
)
DISPLAY_RANGE = (100.0, 10000.0)
LMAX = 10000
DLN_ELL = 0.4
N_BINS = 12
ELL_EDGE_MIN = LMAX * np.exp(-N_BINS * DLN_ELL)


def make_log_bins(ell_min: float, ell_max: float, dln_ell: float) -> np.ndarray:
    """Return complete logarithmic bins of exact width ``dln_ell``."""
    n_bins = int(np.floor(np.log(ell_max / ell_min) / dln_ell + 1e-12))
    return ell_min * np.exp(dln_ell * np.arange(n_bins + 1))


def bin_cl_log(
    ell: np.ndarray,
    cl: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average C_ell in left-inclusive logarithmic bins."""
    ell = np.asarray(ell, dtype=float)
    cl = np.asarray(cl, dtype=float)
    edges = np.asarray(edges, dtype=float)
    centres = np.sqrt(edges[:-1] * edges[1:])
    binned = np.empty(centres.size, dtype=float)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == centres.size - 1:
            inside = (ell >= lower) & (ell <= upper)
        else:
            inside = (ell >= lower) & (ell < upper)
        if not np.any(inside):
            raise ValueError(f"logarithmic bin [{lower}, {upper}] is empty")
        binned[index] = np.mean(cl[inside])
    return centres, binned


def compute_theory_bandpowers(
    edges: np.ndarray,
    *,
    n_per_bin: int = 12,
) -> dict[str, np.ndarray]:
    """Evaluate and log-bin best-fit custom-GNFW D_ell terms."""
    chunks = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == len(edges) - 2:
            ell_bin = np.geomspace(lower, upper, n_per_bin)
        else:
            ell_bin = np.geomspace(lower, upper, n_per_bin + 1)[:-1]
        chunks.append(ell_bin)
    ell = np.concatenate(chunks)

    theory = L1M9CustomGNFWTheory()
    theory.initialize()
    spectrum = theory.evaluate_spectrum(A_SZ, ALPHA_SZ, ell=ell)
    prefactor = ell * (ell + 1.0) / (2.0 * np.pi)
    centres = np.sqrt(edges[:-1] * edges[1:])
    binned = {"ell": centres}
    for term in ("1h", "2h"):
        _, cl_binned = bin_cl_log(ell, spectrum[term] / prefactor, edges)
        centre_prefactor = centres * (centres + 1.0) / (2.0 * np.pi)
        binned[term] = centre_prefactor * cl_binned
    binned["total"] = binned["1h"] + binned["2h"]
    return binned


def compute_simple_gnfw_bandpowers(
    edges: np.ndarray,
    *,
    n_per_bin: int = 12,
) -> dict[str, np.ndarray]:
    """Evaluate and log-bin the D3A A10 GNFW total spectrum with B=1."""
    chunks = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == len(edges) - 2:
            ell_bin = np.geomspace(lower, upper, n_per_bin)
        else:
            ell_bin = np.geomspace(lower, upper, n_per_bin + 1)[:-1]
        chunks.append(ell_bin)
    ell = np.concatenate(chunks)

    spectrum = cl_yy(ell, B=1.0)
    centres, cl_binned = bin_cl_log(ell, spectrum["cl"], edges)
    centre_prefactor = centres * (centres + 1.0) / (2.0 * np.pi)
    return {"ell": centres, "total": centre_prefactor * cl_binned}


def compute_map_bandpowers(
    map_file: str | Path,
    edges: np.ndarray,
    *,
    lmax: int,
    deconvolve_pixwin: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure and logarithmically bin a full-sky HEALPix map."""
    sky_map = hp.read_map(map_file, dtype=np.float64)
    nside = hp.npix2nside(sky_map.size)
    cl = hp.anafast(sky_map - np.mean(sky_map), lmax=lmax)
    ell = np.arange(cl.size, dtype=float)
    if deconvolve_pixwin:
        pixel_window = hp.pixwin(nside, lmax=lmax)
        cl = cl / pixel_window**2
    centres, cl_binned = bin_cl_log(ell, cl, edges)
    prefactor = centres * (centres + 1.0) / (2.0 * np.pi)
    return centres, prefactor * cl_binned


def write_empirical_bandpowers(
    output_path: str | Path,
    ell: np.ndarray,
    displayed_dl: np.ndarray,
) -> None:
    """Write the independent high-ell empirical bandpowers."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "L1_m9 fiducial full-sky tSZ; independent high-ell logarithmic bins\n"
        "Delta ln ell = 0.4; ell_max = 10000; "
        "HEALPix Nside=4096 pixel window deconvolved\n"
        "ell_eff  1e12_D_ell_yy"
    )
    np.savetxt(output_path, np.column_stack((ell, displayed_dl)), header=header)


def make_high_ell_figure(
    ell: np.ndarray,
    map_dl: np.ndarray,
    total_dl: np.ndarray,
    one_halo_dl: np.ndarray,
    simple_gnfw_dl: np.ndarray,
) -> plt.Figure:
    """Draw the high-ell map/model comparison and full-range ratio."""
    figure, (upper, ratio) = plt.subplots(
        2,
        1,
        figsize=(6.6, 7.2),
        sharex=True,
        height_ratios=[3.0, 1.0],
        gridspec_kw={"hspace": 0.06},
    )
    upper.loglog(
        ell,
        map_dl,
        color="black",
        marker="o",
        markersize=4.2,
        linewidth=1.7,
        label="FLAMINGO L1_m9 full sky",
        zorder=3,
    )
    upper.loglog(
        ell,
        total_dl,
        color="#c23b32",
        linewidth=2.4,
        label="customGNFW best fit, total",
    )
    upper.loglog(
        ell,
        one_halo_dl,
        color="#c23b32",
        linewidth=1.5,
        linestyle="--",
        label="customGNFW best fit, 1-halo",
    )
    upper.loglog(
        ell,
        simple_gnfw_dl,
        color="#2878b5",
        linewidth=2.0,
        linestyle="-.",
        label="simple GNFW, B=1, total",
    )
    ratio.semilogx(ell, map_dl / total_dl, color="black", marker="o", markersize=4.0)
    ratio.axhline(1.0, color="#c23b32", linewidth=1.0)

    upper.set_ylabel(r"$10^{12}\,\ell(\ell+1)C_\ell^{yy}/(2\pi)$")
    ratio.set_ylabel("map / total")
    ratio.set_xlabel(r"multipole $\ell$")
    upper.set_title("Full-sky total tSZ power spectrum (D3A)")
    upper.legend(frameon=False, fontsize=8.5)
    upper.text(
        0.04,
        0.96,
        (
            rf"$A_{{\rm SZ}}={A_SZ:.7f}$"
            "\n"
            rf"$\alpha_{{\rm SZ}}={ALPHA_SZ:.8f}$"
            "\n"
            rf"$B={B:.2f}$"
            "\n"
            rf"$\Delta\ln\ell=0.4$"
        ),
        transform=upper.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    for axis in (upper, ratio):
        axis.set_xlim(*DISPLAY_RANGE)
        axis.grid(which="both", alpha=0.25, linewidth=0.5)
    return figure


if __name__ == "__main__":
    log_edges = make_log_bins(ELL_EDGE_MIN, LMAX, DLN_ELL)
    map_ell, measured_dl = compute_map_bandpowers(
        MAP_FILE,
        log_edges,
        lmax=LMAX,
        deconvolve_pixwin=True,
    )
    model = compute_theory_bandpowers(log_edges)
    simple_gnfw = compute_simple_gnfw_bandpowers(log_edges)
    np.testing.assert_allclose(map_ell, model["ell"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(map_ell, simple_gnfw["ell"], rtol=0.0, atol=0.0)
    displayed_map_dl = measured_dl * DISPLAY_SCALE
    write_empirical_bandpowers(BANDPOWER_FILE, map_ell, displayed_map_dl)
    figure = make_high_ell_figure(
        map_ell,
        displayed_map_dl,
        model["total"] * DISPLAY_SCALE,
        model["1h"] * DISPLAY_SCALE,
        simple_gnfw["total"] * DISPLAY_SCALE,
    )
    print(f"wrote {BANDPOWER_FILE.relative_to(REPO)}", flush=True)
    for output_path in save_figure(figure, OUTPUT_STEM):
        print(f"wrote {output_path.relative_to(REPO)}", flush=True)
