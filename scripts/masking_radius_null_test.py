"""Null test: how the masked tSZ power spectrum depends on the masking radius.

The paper masks every detected cluster with a disc of radius ``4 * theta_500``.
This script tests how much that choice matters: for a fixed detection threshold
``q``, it grows the disc radius from ``0`` (no masking) upwards and measures the
Compton-y auto-spectrum at each step. The residual power should fall steeply
while the disc is still inside the cluster and then plateau once the disc
encloses essentially all of the cluster's tSZ signal; where the plateau starts
is the smallest defensible masking radius.

Differences from :mod:`compute_l1_m9_masked_ps_alpha_fixed_1p12`, which
produces the paper's data products:

* the disc radius here is exactly ``r_mult * theta_500`` -- the production
  ``max(4 * theta_500, 2 * FWHM)`` floor is dropped, because a floor of
  20 arcmin would dominate the radius at small ``r_mult`` and flatten the very
  trend this test is measuring. The production points are recovered from the
  stored ``..._alpha_fixed_1p12_binned_18.txt`` files at analysis time;
* ``r_mult = 0`` is the unmasked map, run through NaMaster with a unit mask so
  that it is estimated identically to the masked cases. It doubles as the
  estimator validation against the stored full-sky bandpowers.

Everything else is the production pipeline unchanged: yang26-rotated positions,
``theta_500 = R_500c / D_A(z)`` in the D3A cosmology, binary ``query_disc``
mask, C2 apodization at 0.25 deg, mask-weighted monopole subtracted, NaMaster
MASTER per-ell to ``lmax=10000``, HEALPix pixel window deconvolved, and the two
standard binnings (18 Planck bins, 12 log bins of width ``Delta ln ell = 0.4``).

Each ``(q_cut, r_mult)`` point is cached as its own ``.npz``, so the sweep is
resumable and the two simulations can run concurrently.

Run::

    python scripts/masking_radius_null_test.py --variant L1_m9
    python scripts/masking_radius_null_test.py --variant L2p8_m9
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
import pymaster as nmt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from flamingo.catalogue import theta_500  # noqa: E402
from flamingo.catalogue.frame import rotation_sanity  # noqa: E402
from flamingo.powerspectra.q_selection import (  # noqa: E402
    QSelection,
    resolve_q_selection,
)

# --- Inputs -----------------------------------------------------------------

VARIANTS = {
    "L1_m9": {
        "map": Path("/rds/rds-lxu/flamingo/L1_m9/maps/y_unlensed_L1_m9_lc0_nside4096.fits"),
        "catalogue": Path(
            "/rds/rds-lxu/flamingo/L1_m9/catalogues/"
            "halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommz_alpha_fixed_1p12.csv"
        ),
        "fullsky_18": "Dl_yy_L1_m9_fullsky_binned_18.txt",
    },
    "L2p8_m9": {
        "map": Path(
            "/rds/rds-lxu/flamingo/L2p8_m9/lightcone0/healpix_map/y_unlensed_L2p8_m9_lc0.fits"
        ),
        "catalogue": Path(
            "/rds/rds-lxu/flamingo/L2p8_m9/lightcone0/catalogues/"
            "halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_qfrommz_alpha_fixed_1p12.csv"
        ),
        "fullsky_18": "Dl_yy_L2p8_m9_fullsky_binned_18.txt",
    },
}

TAG = "qfrommz_alpha_fixed_1p12"
BANDPOWERS = REPO / "data_paper" / "binned_bandpowers"
OUT_DIR = REPO / "data_paper" / "masking_radius_null_test"

# --- Sweep ------------------------------------------------------------------

#: Detection thresholds. The paper's headline cut is ``q > 5``.
Q_CUTS = [20.0, 10.0, 5.0, 1.0]

#: Masking radii in units of ``theta_500``. ``0`` is the unmasked map; the
#: production choice is ``4``; the grid runs well past it to expose the plateau.
R_MULTS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]

# --- Estimator (identical to the production scripts) -------------------------

APOSIZE_DEG = 0.25
APOTYPE = "C2"
LMAX = 10000

ELL_MIN = np.array([9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835])
ELL_MAX = np.array(
    [12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835, 1085]
)
ELL_EFF = np.array(
    [
        10.0,
        13.5,
        18.0,
        23.5,
        30.5,
        40.0,
        52.5,
        68.5,
        89.5,
        117.0,
        152.5,
        198.0,
        257.5,
        335.5,
        436.5,
        567.5,
        738.0,
        959.5,
    ]
)
N_LOG_BINS = 12
DLN_ELL = 0.4
LOG_EDGES = LMAX * np.exp(-N_LOG_BINS * DLN_ELL) * np.exp(DLN_ELL * np.arange(N_LOG_BINS + 1))

CHUNK = 1_000_000
COLUMNS = ["z", "R_500c_Mpc", "theta_rot_rad", "phi_rot_rad", "q_from_mz"]
SAMPLE_COLUMNS = ["theta_nat_rad", "phi_nat_rad"]


def q_column(selection_tag: str) -> str:
    return "q_from_aperture" if selection_tag == "qfrommap" else "q_from_mz"


def catalogue_path(variant: str, selection: QSelection) -> Path:
    if variant == "L1_m9":
        return selection.l1_catalogue_dir / (
            "halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_" f"{selection.tag}.csv"
        )
    return (
        selection.l2_root
        / "lightcone0/catalogues"
        / ("halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_" f"{selection.tag}.csv")
    )


def selection_output_dir(selection_tag: str) -> Path:
    return OUT_DIR / "qfrommap" if selection_tag == "qfrommap" else OUT_DIR


def load_catalogue(cat_file: Path, q_column_name: str = "q_from_mz") -> dict[str, np.ndarray]:
    """Stream the catalogue; keep only what the masking step needs."""
    cols: dict[str, list[np.ndarray]] = {"theta": [], "phi": [], "t500": [], "q": [], "nat": []}
    n_rows = 0
    rng = np.random.default_rng(0)
    requested_columns = [q_column_name if column == "q_from_mz" else column for column in COLUMNS]
    for chunk in pd.read_csv(
        cat_file,
        comment="#",
        usecols=requested_columns + SAMPLE_COLUMNS,
        chunksize=CHUNK,
    ):
        cols["theta"].append(chunk["theta_rot_rad"].to_numpy(np.float64))
        cols["phi"].append(chunk["phi_rot_rad"].to_numpy(np.float64))
        cols["q"].append(chunk[q_column_name].to_numpy(np.float64))
        cols["t500"].append(
            theta_500(chunk["R_500c_Mpc"].to_numpy(np.float64), chunk["z"].to_numpy(np.float64))
        )
        take = rng.random(len(chunk)) < (30000 / 1.53e6)
        if take.any():
            cols["nat"].append(
                chunk.loc[
                    take, ["theta_nat_rad", "phi_nat_rad", "theta_rot_rad", "phi_rot_rad"]
                ].to_numpy()
            )
        n_rows += len(chunk)
        print(f"  catalogue: {n_rows:,} rows", flush=True)
    out = {k: np.concatenate(v) for k, v in cols.items() if k != "nat"}
    out["nat_sample"] = np.concatenate(cols["nat"])
    print(f"  catalogue: {n_rows:,} halos total", flush=True)
    return out


def binary_disc_mask(
    nside: int, theta: np.ndarray, phi: np.ndarray, radius: np.ndarray
) -> np.ndarray:
    """Binary mask, 0 inside discs -- production convention (query_disc defaults)."""
    mask = np.ones(hp.nside2npix(nside), dtype=np.float64)
    for th, ph, rr in zip(theta, phi, radius):
        mask[hp.query_disc(nside, hp.ang2vec(th, ph), float(rr))] = 0.0
    return mask


def decoupled_cl_per_ell(ymap: np.ndarray, mask_apo: np.ndarray, pixwin2: np.ndarray) -> np.ndarray:
    """Pixwin-corrected decoupled C_ell at every ell=2..LMAX (NaMaster MASTER)."""
    w = mask_apo
    m = ymap - float(np.sum(w * ymap) / np.sum(w))
    field = nmt.NmtField(w, [m], lmax=LMAX)
    bins = nmt.NmtBin.from_lmax_linear(LMAX, nlb=1)
    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(field, field, bins)
    cl = workspace.decouple_cell(nmt.compute_coupled_cell(field, field))[0]

    ell_eff = bins.get_effective_ells().astype(int)
    cl_full = np.full(LMAX + 1, np.nan)
    cl_full[ell_eff] = cl
    return cl_full / pixwin2


def bin_dl_18(ell: np.ndarray, cl: np.ndarray) -> np.ndarray:
    """Uniform mean of D_ell over the inclusive Planck bins."""
    dl = ell * (ell + 1.0) * cl / (2.0 * np.pi)
    out = np.full(18, np.nan)
    for i in range(18):
        inside = (ell >= ELL_MIN[i]) & (ell <= ELL_MAX[i])
        out[i] = np.nanmean(dl[inside])
    return out


def bin_cl_log(ell: np.ndarray, cl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Uniform mean of C_ell in left-inclusive log bins, D_ell at geometric centres."""
    centres = np.sqrt(LOG_EDGES[:-1] * LOG_EDGES[1:])
    out = np.empty(N_LOG_BINS)
    for i, (lo, hi) in enumerate(zip(LOG_EDGES[:-1], LOG_EDGES[1:])):
        inside = (ell >= lo) & (ell <= hi) if i == N_LOG_BINS - 1 else (ell >= lo) & (ell < hi)
        if not np.any(inside):
            raise ValueError(f"log bin [{lo}, {hi}] is empty")
        out[i] = np.nanmean(cl[inside])
    return centres, centres * (centres + 1.0) * out / (2.0 * np.pi)


def point_path(
    variant: str,
    cut: float,
    r_mult: float,
    *,
    selection_tag: str = TAG,
) -> Path:
    """Cache file for one ``(q_cut, r_mult)`` point of the sweep."""
    output_dir = selection_output_dir(selection_tag)
    if r_mult == 0.0:
        return output_dir / f"{variant}_unmasked.npz"
    radius_tag = f"r{r_mult:g}".replace(".", "p")
    return output_dir / f"{variant}_qgt{cut:g}_{radius_tag}.npz"


def run_point(
    variant: str,
    ymap: np.ndarray,
    nside: int,
    pixwin2: np.ndarray,
    ell: np.ndarray,
    cat: dict[str, np.ndarray],
    cut: float,
    r_mult: float,
    *,
    selection_tag: str = TAG,
    force: bool = False,
) -> dict:
    """Measure the bandpowers for one detection threshold and masking radius."""
    out = point_path(variant, cut, r_mult, selection_tag=selection_tag)
    if out.exists() and not force:
        print(f"  cached {out.name}", flush=True)
        return dict(np.load(out))

    t0 = time.time()
    if r_mult == 0.0:
        n_masked, mask_apo = 0, np.ones_like(ymap)
        f_sky_raw = f_sky_eff = 1.0
        print("=== unmasked (unit mask) ===", flush=True)
    else:
        keep = cat["q"] > cut
        n_masked = int(keep.sum())
        print(f"=== q>{cut:g}, r={r_mult:g}*theta500: {n_masked:,} halos ===", flush=True)
        mask_bin = binary_disc_mask(
            nside, cat["theta"][keep], cat["phi"][keep], r_mult * cat["t500"][keep]
        )
        f_sky_raw = float(mask_bin.mean())
        mask_apo = nmt.mask_apodization(mask_bin, APOSIZE_DEG, apotype=APOTYPE)
        f_sky_eff = float(np.mean(mask_apo**2))
        del mask_bin
        print(
            f"  f_sky raw={f_sky_raw:.4f} eff={f_sky_eff:.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    cl = decoupled_cl_per_ell(ymap, mask_apo, pixwin2)
    del mask_apo
    dl_18 = bin_dl_18(ell, cl)
    ell_log, dl_12 = bin_cl_log(ell, cl)

    result = dict(
        variant=variant,
        q_cut=cut,
        r_mult=r_mult,
        n_masked=n_masked,
        f_sky_raw=f_sky_raw,
        f_sky_eff=f_sky_eff,
        ell_18=ELL_EFF,
        dl_18=dl_18 * 1e12,
        ell_log=ell_log,
        dl_12=dl_12 * 1e12,
        nside=nside,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **result)
    print(f"  wrote {out.name} ({time.time() - t0:.0f}s)", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--selection", choices=(TAG, "qfrommap"), default=TAG)
    parser.add_argument(
        "--q-cut",
        type=float,
        action="append",
        dest="q_cuts",
        help="restrict the sweep to this q threshold (repeatable); default all of Q_CUTS",
    )
    parser.add_argument("--force", action="store_true", help="ignore cached points")
    args = parser.parse_args()

    variant = args.variant
    selection = resolve_q_selection(args.selection)
    q_cuts = args.q_cuts or Q_CUTS
    spec = VARIANTS[variant]
    t0 = time.time()
    output_dir = selection_output_dir(selection.tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {variant}: map + catalogue ===", flush=True)
    ymap = hp.read_map(spec["map"], dtype=np.float64)
    nside = hp.npix2nside(ymap.size)
    cat_path = catalogue_path(variant, selection)
    cat = load_catalogue(cat_path, q_column(selection.tag))
    print(f"  nside={nside}, {cat['q'].size:,} halos, mean y={ymap.mean():.3e}", flush=True)

    sample = cat["nat_sample"]
    r_nat = rotation_sanity(ymap, sample[:, 0], sample[:, 1])
    r_rot = rotation_sanity(ymap, sample[:, 2], sample[:, 3])
    print(
        f"  rotation sanity: rot ratio={r_rot['ratio']:.2f}, nat ratio={r_nat['ratio']:.2f}",
        flush=True,
    )
    if r_rot["ratio"] <= r_nat["ratio"]:
        raise RuntimeError("rotated positions do not trace tSZ peaks better than natural ones")

    pixwin2 = hp.pixwin(nside, lmax=LMAX) ** 2
    ell = np.arange(LMAX + 1, dtype=float)

    unmasked = run_point(
        variant,
        ymap,
        nside,
        pixwin2,
        ell,
        cat,
        np.nan,
        0.0,
        selection_tag=selection.tag,
        force=args.force,
    )
    stored = np.loadtxt(BANDPOWERS / spec["fullsky_18"])[:, 1]
    max_dev = float(np.max(np.abs(np.asarray(unmasked["dl_18"]) / stored - 1.0)))
    print(f"  unit mask vs stored full sky: max|frac diff| = {max_dev:.3e}", flush=True)
    if max_dev > 5e-3:
        raise RuntimeError(f"unit-mask validation failed: max frac diff {max_dev:.3e}")

    for cut in q_cuts:
        for r_mult in R_MULTS:
            if r_mult == 0.0:
                continue
            run_point(
                variant,
                ymap,
                nside,
                pixwin2,
                ell,
                cat,
                cut,
                r_mult,
                selection_tag=selection.tag,
                force=args.force,
            )
            print(f"  [{time.time() - t0:.0f}s elapsed]", flush=True)

    meta = {
        "variant": variant,
        "map": str(spec["map"]),
        "catalogue": str(cat_path),
        "selection": selection.tag,
        "q_column": selection.q_column,
        "q_cuts": q_cuts,
        "r_mults": R_MULTS,
        "masking": {
            "radius": "r_mult * theta_500 (no floor; production uses max(4*theta500, 2*FWHM))",
            "theta500": "R_500c/D_A(z), D3A cosmology",
            "positions": "theta_rot_rad/phi_rot_rad (yang26-rotated frame)",
            "apodization": f"{APOTYPE} {APOSIZE_DEG} deg",
            "monopole": "mask-weighted, subtracted",
            "estimator": f"NaMaster MASTER, nlb=1, lmax={LMAX}, no beam, pixwin deconvolved",
        },
        "unit_mask_vs_stored_fullsky_max_frac_diff": max_dev,
        "runtime_seconds": time.time() - t0,
    }
    # Suffix the metadata file when the sweep was split across processes by
    # q-cut, so concurrent runs don't clobber each other's summary.
    suffix = "" if args.q_cuts is None else "_" + "_".join(f"q{c:g}" for c in q_cuts)
    with open(output_dir / f"{variant}_null_test_metadata{suffix}.json", "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
