"""Masked tSZ power spectra of the fiducial L2p8_m9 lightcones.

Same pipeline as :mod:`compute_l1_m9_masked_ps_alpha_fixed_1p12`, using each
lightcone's yang26-rotated catalogue with ``q_from_mz`` from the custom-GNFW
``alpha_SZ=1.12`` scaling relation.

All eight lightcones are processed and written per lightcone as
``Dl_yy_L2p8_m9_lc{i}_masked_...``; the paper figures use lightcone 0.

Run::

    python scripts/compute_l2p8_m9_masked_ps_alpha_fixed_1p12.py
    python scripts/compute_l2p8_m9_masked_ps_alpha_fixed_1p12.py --lightcone 3
"""
from __future__ import annotations

import argparse
import json
import os
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

L2P8_ROOT = Path("/rds/rds-lxu/flamingo/L2p8_m9")

#: Every lightcone of the fiducial L2p8_m9 box. They are independent
#: realisations, so all eight are processed; products are written per lightcone
#: and the paper figures use lightcone 0.
LIGHTCONES = tuple(range(8))

#: Products live in the shared data tree, not in whichever checkout runs this.
OUT_DIR = (
    Path(os.environ.get("FLAMINGO_ROOT", str(REPO))) / "data_paper" / "binned_bandpowers"
)

#: Stored lightcone-0 full-sky spectrum, used only as a pipeline sanity check.
FULLSKY_18_LC0 = OUT_DIR / "Dl_yy_L2p8_m9_fullsky_binned_18.txt"
TAG = "qfrommz_alpha_fixed_1p12"


def variant(lightcone: int) -> str:
    """Product-name stem for one lightcone."""
    return f"L2p8_m9_lc{lightcone}"


def map_file(lightcone: int) -> Path:
    """Compton-y map of one lightcone."""
    return (
        L2P8_ROOT
        / f"lightcone{lightcone}"
        / "healpix_map"
        / f"y_unlensed_L2p8_m9_lc{lightcone}.fits"
    )


def cat_file(lightcone: int) -> Path:
    """yang26-rotated ``q_from_mz`` catalogue of one lightcone."""
    return (
        L2P8_ROOT
        / f"lightcone{lightcone}"
        / "catalogues"
        / f"halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_{TAG}.csv"
    )

Q_CUTS = [50.0, 20.0, 10.0, 5.0, 1.0]
CUT_TAGS = ["qgt50", "qgt20", "qgt10", "qgt5", "qgt1"]

FWHM_ARCMIN = 10.0
R_MULT = 4.0
APOSIZE_DEG = 0.25
APOTYPE = "C2"

LMAX = 10000
ELL_MIN = np.array(
    [9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835]
)
ELL_MAX = np.array(
    [12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835, 1085]
)
ELL_EFF = np.array(
    [10.0, 13.5, 18.0, 23.5, 30.5, 40.0, 52.5, 68.5, 89.5, 117.0, 152.5, 198.0,
     257.5, 335.5, 436.5, 567.5, 738.0, 959.5]
)
N_LOG_BINS = 12
DLN_ELL = 0.4
LOG_EDGES = LMAX * np.exp(-N_LOG_BINS * DLN_ELL) * np.exp(DLN_ELL * np.arange(N_LOG_BINS + 1))

CHUNK = 1_000_000
COLUMNS = ["z", "R_500c_Mpc", "theta_rot_rad", "phi_rot_rad", "q_from_mz"]
SAMPLE_COLUMNS = ["theta_nat_rad", "phi_nat_rad"]
N_HALOS_EST = 1_555_547


def load_catalogue(cat_path: Path) -> dict[str, np.ndarray]:
    cols: dict[str, list[np.ndarray]] = {
        "theta": [], "phi": [], "t500": [], "q": [], "nat": [],
    }
    n_rows = 0
    rng = np.random.default_rng(0)
    for chunk in pd.read_csv(
        cat_path, comment="#", usecols=COLUMNS + SAMPLE_COLUMNS, chunksize=CHUNK
    ):
        cols["theta"].append(chunk["theta_rot_rad"].to_numpy(np.float64))
        cols["phi"].append(chunk["phi_rot_rad"].to_numpy(np.float64))
        cols["q"].append(chunk["q_from_mz"].to_numpy(np.float64))
        cols["t500"].append(
            theta_500(chunk["R_500c_Mpc"].to_numpy(np.float64), chunk["z"].to_numpy(np.float64))
        )
        take = rng.random(len(chunk)) < (30000 / N_HALOS_EST)
        if take.any():
            cols["nat"].append(
                chunk.loc[take, ["theta_nat_rad", "phi_nat_rad", "theta_rot_rad", "phi_rot_rad"]].to_numpy()
            )
        n_rows += len(chunk)
        print(f"  catalogue: {n_rows:,} rows", flush=True)
    out = {k: np.concatenate(v) for k, v in cols.items() if k != "nat"}
    out["nat_sample"] = np.concatenate(cols["nat"])
    print(f"  catalogue: {n_rows:,} halos total", flush=True)
    return out


def binary_disc_mask(nside: int, theta: np.ndarray, phi: np.ndarray, radius: np.ndarray) -> np.ndarray:
    mask = np.ones(hp.nside2npix(nside), dtype=np.float64)
    for th, ph, rr in zip(theta, phi, radius):
        mask[hp.query_disc(nside, hp.ang2vec(th, ph), float(rr))] = 0.0
    return mask


def decoupled_cl_per_ell(ymap: np.ndarray, mask_apo: np.ndarray, pixwin2: np.ndarray) -> np.ndarray:
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
    dl = ell * (ell + 1.0) * cl / (2.0 * np.pi)
    out = np.full(18, np.nan)
    for i in range(18):
        inside = (ell >= ELL_MIN[i]) & (ell <= ELL_MAX[i])
        out[i] = np.nanmean(dl[inside])
    return out


def bin_cl_log(ell: np.ndarray, cl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centres = np.sqrt(LOG_EDGES[:-1] * LOG_EDGES[1:])
    out = np.empty(N_LOG_BINS)
    for i, (lo, hi) in enumerate(zip(LOG_EDGES[:-1], LOG_EDGES[1:])):
        inside = (ell >= lo) & (ell <= hi) if i == N_LOG_BINS - 1 else (ell >= lo) & (ell < hi)
        if not np.any(inside):
            raise ValueError(f"log bin [{lo}, {hi}] is empty")
        out[i] = np.nanmean(cl[inside])
    return centres, centres * (centres + 1.0) * out / (2.0 * np.pi)


def write_bandpowers(path: Path, ell: np.ndarray, dl_1e12: np.ndarray, header: str) -> None:
    np.savetxt(path, np.column_stack([ell, dl_1e12]), fmt="%.6e", header=header)


def process_lightcone(lightcone: int) -> None:
    """Compute and write the masked bandpowers of one lightcone."""
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = variant(lightcone)
    map_path, cat_path = map_file(lightcone), cat_file(lightcone)
    print(f"\n########## {name} ##########", flush=True)

    print("=== map + catalogue ===", flush=True)
    ymap = hp.read_map(map_path, dtype=np.float64)
    nside = hp.npix2nside(ymap.size)
    cat = load_catalogue(cat_path)
    q, theta, phi, t500 = cat["q"], cat["theta"], cat["phi"], cat["t500"]
    print(f"  nside={nside}, {q.size:,} halos, mean y={ymap.mean():.3e}", flush=True)

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
    mask_floor = np.deg2rad(2.0 * FWHM_ARCMIN / 60.0)

    print("=== unit-mask validation vs stored full-sky 18-bin ===", flush=True)
    cl_unit = decoupled_cl_per_ell(ymap, np.ones_like(ymap), pixwin2)
    dl18_unit = bin_dl_18(ell, cl_unit)
    max_dev = None
    if lightcone == 0 and FULLSKY_18_LC0.is_file():
        stored = np.loadtxt(FULLSKY_18_LC0)[:, 1]
        max_dev = float(np.max(np.abs(dl18_unit * 1e12 / stored - 1.0)))
        print(f"  NaMaster unit mask vs stored anafast: max|frac diff| = {max_dev:.3e}", flush=True)
    write_bandpowers(
        OUT_DIR / f"Dl_yy_{name}_fullsky_binned_18.txt",
        ELL_EFF,
        dl18_unit * 1e12,
        header=f"{name} full-sky tSZ; NaMaster unit mask, pixwin deconvolved\nell_eff  1e12_D_ell_yy",
    )

    ell_log, dl12_fullsky = bin_cl_log(ell, cl_unit)
    write_bandpowers(
        OUT_DIR / f"Dl_yy_{name}_fullsky_logbins_dln0p4_lmax10000.txt",
        ell_log,
        dl12_fullsky * 1e12,
        header=(
            f"{name} full-sky tSZ; independent high-ell logarithmic bins\n"
            f"Delta ln ell = {DLN_ELL}; ell_max = {LMAX}; "
            f"HEALPix Nside={nside} pixel window deconvolved\n"
            "ell_eff  1e12_D_ell_yy"
        ),
    )
    print(f"  wrote {name} full-sky bandpowers", flush=True)

    metadata: dict[str, dict] = {}
    dl18_all, dl12_all = [], []
    for cut, tag in zip(Q_CUTS, CUT_TAGS):
        keep = q > cut
        radius = np.maximum(R_MULT * t500[keep], mask_floor)
        print(
            f"=== q>{cut:g}: masking {keep.sum():,} halos "
            f"(radius floor {np.rad2deg(mask_floor) * 60:.0f} arcmin) ===",
            flush=True,
        )
        mask_bin = binary_disc_mask(nside, theta[keep], phi[keep], radius)
        f_sky_raw = float(mask_bin.mean())
        mask_apo = nmt.mask_apodization(mask_bin, APOSIZE_DEG, apotype=APOTYPE)
        f_sky_eff = float(np.mean(mask_apo**2))
        del mask_bin
        print(
            f"  f_sky raw={f_sky_raw:.4f} eff={f_sky_eff:.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )

        cl_masked = decoupled_cl_per_ell(ymap, mask_apo, pixwin2)
        del mask_apo
        dl18 = bin_dl_18(ell, cl_masked)
        ell_log, dl12 = bin_cl_log(ell, cl_masked)
        dl18_all.append(dl18)
        dl12_all.append(dl12)

        header18 = (
            f"{name} masked tSZ, q>{cut:g} ({TAG}); synthetic-data masking: "
            f"r=max(4*theta500, 2x10arcmin), {APOTYPE} apodization {APOSIZE_DEG} deg, "
            "masked monopole subtracted, NaMaster MASTER per-ell, pixwin deconvolved; "
            "uniform mean over inclusive Planck bins\nell_eff  1e12_D_ell_yy"
        )
        header12 = (
            f"{name} masked tSZ, q>{cut:g} ({TAG}); same masking as the binned_18 "
            f"file; Delta ln ell = {DLN_ELL}; ell_max = {LMAX}; "
            f"HEALPix Nside={nside} pixel window deconvolved\nell_eff  1e12_D_ell_yy"
        )
        write_bandpowers(
            OUT_DIR / f"Dl_yy_{name}_masked_{tag}_{TAG}_binned_18.txt",
            ELL_EFF,
            dl18 * 1e12,
            header18,
        )
        write_bandpowers(
            OUT_DIR / f"Dl_yy_{name}_masked_{tag}_{TAG}_logbins_dln0p4_lmax10000.txt",
            ell_log,
            dl12 * 1e12,
            header12,
        )
        metadata[tag] = {
            "q_cut": cut,
            "n_masked": int(keep.sum()),
            "f_sky_raw": f_sky_raw,
            "f_sky_eff": f_sky_eff,
        }
        print(f"  wrote {tag} bandpowers ({time.time() - t0:.0f}s)", flush=True)

    np.savez(
        OUT_DIR / f"{name}_masked_{TAG}.npz",
        ell_eff_18=ELL_EFF,
        dl_18=np.stack(dl18_all),
        ell_log=ell_log,
        dl_12=np.stack(dl12_all),
        q_cuts=np.array(Q_CUTS),
        cut_tags=np.array(CUT_TAGS),
    )
    meta = {
        "map": str(map_path),
        "catalogue": str(cat_path),
        "masking": {
            "radius": "max(4*theta500, 2*FWHM), FWHM=10 arcmin",
            "theta500": "R_500c/D_A(z), D3A cosmology",
            "positions": "theta_rot_rad/phi_rot_rad (yang26-rotated frame)",
            "apodization": f"{APOTYPE} {APOSIZE_DEG} deg",
            "monopole": "mask-weighted, subtracted",
            "estimator": f"NaMaster MASTER, nlb=1, lmax={LMAX}, no beam, pixwin deconvolved",
        },
        "unit_mask_vs_stored_fullsky_max_frac_diff": max_dev,
        "cuts": metadata,
        "runtime_seconds": time.time() - t0,
    }
    with open(OUT_DIR / f"{name}_masked_{TAG}_metadata.json", "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"done {name} ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--lightcone", type=int, action="append", choices=LIGHTCONES,
        help="lightcone(s) to process (default: all eight)",
    )
    args = parser.parse_args()
    for lightcone in args.lightcone or LIGHTCONES:
        process_lightcone(lightcone)


if __name__ == "__main__":
    main()
