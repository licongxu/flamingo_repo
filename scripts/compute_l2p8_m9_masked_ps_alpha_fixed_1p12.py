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
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import healpy as hp
import numpy as np
import pandas as pd
import pymaster as nmt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from flamingo.catalogue import theta_500  # noqa: E402
from flamingo.catalogue.frame import rotation_sanity  # noqa: E402
from flamingo.masking import disc_mask  # noqa: E402
from flamingo.powerspectra.bandpowers import (  # noqa: E402
    PLANCK_ELL_EFF,
    bin_log_dl,
    bin_planck_dl,
    write_bandpowers,
)
from flamingo.powerspectra.namaster import decoupled_cl_per_ell  # noqa: E402
from flamingo.powerspectra.q_selection import (  # noqa: E402
    QSelection,
    resolve_q_selection,
)

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
DEFAULT_SELECTION = resolve_q_selection(TAG)


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


def cat_file(
    lightcone: int,
    selection: QSelection = DEFAULT_SELECTION,
) -> Path:
    """Yang26-rotated catalogue of one lightcone and q selection."""
    return (
        selection.l2_root
        / f"lightcone{lightcone}"
        / "catalogues"
        / (
            "halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_"
            f"{selection.tag}.csv"
        )
    )


def masked_out_paths(
    lightcone: int,
    cut_tag: str,
    selection_tag: str = TAG,
) -> tuple[Path, Path]:
    stem = f"Dl_yy_{variant(lightcone)}_masked_{cut_tag}_{selection_tag}"
    return (
        OUT_DIR / f"{stem}_binned_18.txt",
        OUT_DIR / f"{stem}_logbins_dln0p4_lmax10000.txt",
    )

Q_CUTS = [50.0, 20.0, 10.0, 5.0, 1.0]
CUT_TAGS = ["qgt50", "qgt20", "qgt10", "qgt5", "qgt1"]

FWHM_ARCMIN = 10.0
R_MULT = 4.0
APOSIZE_DEG = 0.25
APOTYPE = "C2"

LMAX = 10000
DLN_ELL = 0.4

CHUNK = 1_000_000
COLUMNS = ["z", "R_500c_Mpc", "theta_rot_rad", "phi_rot_rad", "q_from_mz"]
SAMPLE_COLUMNS = ["theta_nat_rad", "phi_nat_rad"]
N_HALOS_EST = 1_555_547


def load_catalogue(
    cat_path: Path,
    q_column: str = "q_from_mz",
) -> dict[str, np.ndarray]:
    cols: dict[str, list[np.ndarray]] = {
        "theta": [], "phi": [], "t500": [], "q": [], "nat": [],
    }
    n_rows = 0
    rng = np.random.default_rng(0)
    requested_columns = [
        q_column if column == "q_from_mz" else column for column in COLUMNS
    ]
    for chunk in pd.read_csv(
        cat_path,
        comment="#",
        usecols=requested_columns + SAMPLE_COLUMNS,
        chunksize=CHUNK,
    ):
        cols["theta"].append(chunk["theta_rot_rad"].to_numpy(np.float64))
        cols["phi"].append(chunk["phi_rot_rad"].to_numpy(np.float64))
        cols["q"].append(chunk[q_column].to_numpy(np.float64))
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


def process_lightcone(
    lightcone: int,
    selection: QSelection = DEFAULT_SELECTION,
    force: bool = False,
) -> None:
    """Compute and write the masked bandpowers of one lightcone."""
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = variant(lightcone)
    map_path, cat_path = map_file(lightcone), cat_file(lightcone, selection)
    print(f"\n########## {name} ##########", flush=True)

    print("=== map + catalogue ===", flush=True)
    ymap = hp.read_map(map_path, dtype=np.float64)
    nside = hp.npix2nside(ymap.size)
    cat = load_catalogue(cat_path, q_column=selection.q_column)
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

    max_dev = None
    if selection.tag == TAG:
        print("=== unit-mask validation vs stored full-sky 18-bin ===", flush=True)
        cl_unit = decoupled_cl_per_ell(ymap, np.ones_like(ymap), pixwin2, lmax=LMAX)
        dl18_unit = bin_planck_dl(ell, cl_unit)
        if lightcone == 0 and FULLSKY_18_LC0.is_file():
            stored = np.loadtxt(FULLSKY_18_LC0)[:, 1]
            max_dev = float(np.max(np.abs(dl18_unit * 1e12 / stored - 1.0)))
            print(f"  NaMaster unit mask vs stored anafast: max|frac diff| = {max_dev:.3e}", flush=True)
        write_bandpowers(
            OUT_DIR / f"Dl_yy_{name}_fullsky_binned_18.txt",
            PLANCK_ELL_EFF,
            dl18_unit * 1e12,
            header=f"{name} full-sky tSZ; NaMaster unit mask, pixwin deconvolved\nell_eff  1e12_D_ell_yy",
        )

        ell_log, dl12_fullsky = bin_log_dl(ell, cl_unit)
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
    else:
        print("=== reusing existing full-sky products ===", flush=True)

    metadata: dict[str, dict] = {}
    dl18_all, dl12_all = [], []
    ell_log, _ = bin_log_dl(ell, np.ones_like(ell))
    for cut, tag in zip(Q_CUTS, CUT_TAGS):
        output_18, output_log = masked_out_paths(
            lightcone, tag, selection.tag
        )
        if not force and output_18.is_file() and output_log.is_file():
            print(f"=== q>{cut:g}: existing outputs, skipped ===", flush=True)
            dl18_all.append(np.loadtxt(output_18)[:, 1] / 1e12)
            dl12_all.append(np.loadtxt(output_log)[:, 1] / 1e12)
            continue
        keep = q > cut
        radius = np.maximum(R_MULT * t500[keep], mask_floor)
        print(
            f"=== q>{cut:g}: masking {keep.sum():,} halos "
            f"(radius floor {np.rad2deg(mask_floor) * 60:.0f} arcmin) ===",
            flush=True,
        )
        mask_bin = disc_mask(nside, theta[keep], phi[keep], radius, inclusive=False)
        f_sky_raw = float(mask_bin.mean())
        mask_apo = nmt.mask_apodization(mask_bin, APOSIZE_DEG, apotype=APOTYPE)
        f_sky_eff = float(np.mean(mask_apo**2))
        del mask_bin
        print(
            f"  f_sky raw={f_sky_raw:.4f} eff={f_sky_eff:.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )

        cl_masked = decoupled_cl_per_ell(ymap, mask_apo, pixwin2, lmax=LMAX)
        del mask_apo
        dl18 = bin_planck_dl(ell, cl_masked)
        ell_log, dl12 = bin_log_dl(ell, cl_masked)
        dl18_all.append(dl18)
        dl12_all.append(dl12)

        header18 = (
            f"{name} masked tSZ, q>{cut:g} ({selection.tag}); synthetic-data masking: "
            f"r=max(4*theta500, 2x10arcmin), {APOTYPE} apodization {APOSIZE_DEG} deg, "
            "masked monopole subtracted, NaMaster MASTER per-ell, pixwin deconvolved; "
            "uniform mean over inclusive Planck bins\nell_eff  1e12_D_ell_yy"
        )
        header12 = (
            f"{name} masked tSZ, q>{cut:g} ({selection.tag}); same masking as the binned_18 "
            f"file; Delta ln ell = {DLN_ELL}; ell_max = {LMAX}; "
            f"HEALPix Nside={nside} pixel window deconvolved\nell_eff  1e12_D_ell_yy"
        )
        write_bandpowers(
            output_18,
            PLANCK_ELL_EFF,
            dl18 * 1e12,
            header18,
        )
        write_bandpowers(
            output_log,
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
        OUT_DIR / f"{name}_masked_{selection.tag}.npz",
        ell_eff_18=PLANCK_ELL_EFF,
        dl_18=np.stack(dl18_all),
        ell_log=ell_log,
        dl_12=np.stack(dl12_all),
        q_cuts=np.array(Q_CUTS),
        cut_tags=np.array(CUT_TAGS),
    )
    meta = {
        "map": str(map_path),
        "catalogue": str(cat_path),
        "q_column": selection.q_column,
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
    with open(
        OUT_DIR / f"{name}_masked_{selection.tag}_metadata.json", "w"
    ) as handle:
        json.dump(meta, handle, indent=2)
    print(f"done {name} ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--lightcone", type=int, action="append", choices=LIGHTCONES,
        help="lightcone(s) to process (default: all eight)",
    )
    parser.add_argument(
        "--selection",
        choices=(TAG, "qfrommap"),
        default=TAG,
    )
    parser.add_argument("--l2-root", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        choices=tuple(range(1, 9)),
        default=1,
        help="parallel lightcones (maximum 8, one per lightcone)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list inputs and outputs without reading maps",
    )
    args = parser.parse_args()
    selection = resolve_q_selection(args.selection, l2_root=args.l2_root)
    lightcones = tuple(args.lightcone or LIGHTCONES)
    if args.dry_run:
        for lightcone in lightcones:
            print(f"lightcone{lightcone}: map={map_file(lightcone)}")
            print(f"  catalogue={cat_file(lightcone, selection)}")
            for tag in CUT_TAGS:
                print(
                    f"  outputs={masked_out_paths(lightcone, tag, selection.tag)}"
                )
        return
    if args.workers == 1:
        for lightcone in lightcones:
            process_lightcone(lightcone, selection, args.force)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_lightcone, lightcone, selection, args.force): lightcone
            for lightcone in lightcones
        }
        for future in as_completed(futures):
            lightcone = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"lightcone{lightcone} FAILED: {exc}", flush=True)
                raise


if __name__ == "__main__":
    main()
