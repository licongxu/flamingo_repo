"""Paper-pipeline NaMaster multi-q bandpowers for all L1_m9 feedback variants.

* catalogues ``*_yang26rot_qfrommz_alpha_fixed_1p12.csv``
* mask ``max(4*theta_500, 2*FWHM)``, FWHM=10', C2 apod 0.25°
* NaMaster MASTER per-ell (``nlb=1``, ``lmax=10000``), pixwin deconvolved
* 18 Planck bins + 12 log bins (``Delta ln ell = 0.4``)

Outputs go to ``data_paper/binned_bandpowers/``:

* fiducial: ``Dl_yy_L1_m9_masked_{qtag}_{TAG}_{binning}.txt``
  (matches existing multi-q fiducial products)
* other variants: ``Dl_yy_L1_m9_{variant}_masked_{qtag}_{TAG}_{binning}.txt``

Existing files are skipped. Optional ``--workers N`` runs variants in parallel
(each NaMaster already uses OpenMP; keep workers modest).

Timing (empirical from prior qgt1 run): ~5–8 min wall per (variant × q cut)
with multi-threaded NaMaster. Missing set is typically q∈{50,20,10,3} for
the eight feedback variants + q=3 for fiducial → ~33 cuts.
With ``--workers 4`` and ``OMP_NUM_THREADS=8`` expect ~1–2 hours.

Run::

    export OMP_NUM_THREADS=8
    python scripts/powerspectra/compute_l1_m9_feedback_bandpowers.py --workers 4
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# NaMaster is CPU-only. Importing flamingo/hmfast pulls JAX, which otherwise
# pre-allocates almost all GPU RAM (useless for this job). Force CPU before
# any scientific imports so ProcessPool workers inherit the same setting.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import healpy as hp
import numpy as np
import pymaster as nmt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

DATA_ROOT = Path(os.environ.get("FLAMINGO_ROOT", str(REPO)))
OUT_DIR = DATA_ROOT / "data_paper" / "binned_bandpowers"
LEGACY_FB = REPO / "data_paper" / "feedback_bandpower"

from flamingo.catalogue import load_masking_catalogue as load_catalogue  # noqa: E402
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
    cut_tag,
    resolve_q_selection,
)

MAP_DIR = Path("/rds/rds-lxu/flamingo/L1_m9/maps")
TAG = "qfrommz_alpha_fixed_1p12"

# (variant name, map stem, catalogue stem)
VARIANTS = [
    ("fiducial", "L1_m9", "L1_m9"),
    ("fgas+2sigma", "fgas+2sigma", "fgas+2sigma"),
    ("fgas-2sigma", "fgas-2sigma", "fgas-2sigma"),
    ("fgas-4sigma", "fgas-4sigma", "fgas-4sigma"),
    ("fgas-8sigma", "fgas-8sigma", "fgas-8sigma"),
    ("Jet", "Jet", "Jet"),
    ("Jet_fgas-4sigma", "Jet_fgas-4sigma", "Jet_fgas-4sigma"),
    ("Mstar-1sigma", "Mstar-1sigma", "Mstar-1sigma"),
    ("Mstar-1sigma_fgas-4sigma", "Mstar-1sigma_fgas-4sigma", "Mstar-1sigma_fgas-4sigma"),
]

FWHM_ARCMIN = 10.0
R_MULT = 4.0
APOSIZE_DEG = 0.25
APOTYPE = "C2"
LMAX = 10000
DLN_ELL = 0.4

LEGACY_Q_CUTS = [50.0, 20.0, 10.0, 5.0, 3.0, 1.0]
QFROMMAP_Q_CUTS = [50.0, 20.0, 10.0, 5.0, 1.0]


def map_file(stem: str) -> Path:
    return MAP_DIR / f"y_unlensed_{stem}_lc0_nside4096.fits"


def default_q_cuts(selection_name: str) -> list[float]:
    """Default thresholds for the requested catalogue selection."""
    return list(QFROMMAP_Q_CUTS if selection_name == "qfrommap" else LEGACY_Q_CUTS)


def merge_metadata_payload(existing: dict, incremental: dict) -> dict:
    """Merge an incremental cut run without discarding earlier metadata."""
    merged = copy.deepcopy(existing)
    merged.update(
        {key: value for key, value in incremental.items() if key not in {"q_cuts", "variants"}}
    )
    merged["q_cuts"] = list(existing.get("q_cuts", []))
    for cut in incremental.get("q_cuts", []):
        if cut not in merged["q_cuts"]:
            merged["q_cuts"].append(cut)

    variants = copy.deepcopy(existing.get("variants", {}))
    for name, current in incremental.get("variants", {}).items():
        prior = variants.get(name, {})
        combined = {**prior, **current}
        combined["cuts"] = {
            **prior.get("cuts", {}),
            **current.get("cuts", {}),
        }
        variants[name] = combined
    merged["variants"] = variants
    return merged


def catalogue_path(cat_stem: str, selection: QSelection) -> Path:
    """Catalogue input for one L1 feedback prescription."""
    return selection.l1_catalogue_dir / (
        f"halo_catalogue_M500c_5e13_zlt3_{cat_stem}_yang26rot_" f"{selection.tag}.csv"
    )


def out_paths(
    variant: str,
    tag: str,
    selection_tag: str = TAG,
) -> tuple[Path, Path]:
    """Paper naming: fiducial uses L1_m9 without variant token."""
    if variant == "fiducial":
        stem = f"Dl_yy_L1_m9_masked_{tag}_{selection_tag}"
    else:
        stem = f"Dl_yy_L1_m9_{variant}_masked_{tag}_{selection_tag}"
    return (
        OUT_DIR / f"{stem}_binned_18.txt",
        OUT_DIR / f"{stem}_logbins_dln0p4_lmax10000.txt",
    )


def legacy_paths(variant: str, tag: str) -> tuple[Path, Path]:
    """Older location under feedback_bandpower (qgt1/qgt5 already there)."""
    stem = f"Dl_yy_L1_m9_{variant}_masked_{tag}_{TAG}"
    return (
        LEGACY_FB / f"{stem}_binned_18.txt",
        LEGACY_FB / f"{stem}_logbins_dln0p4_lmax10000.txt",
    )


def ensure_from_legacy(variant: str, tag: str) -> bool:
    """If outputs missing but legacy files exist, hard-link/copy into OUT_DIR."""
    p18, p12 = out_paths(variant, tag)
    if p18.is_file() and p12.is_file():
        return True
    l18, l12 = legacy_paths(variant, tag)
    if not (l18.is_file() and l12.is_file()):
        return False
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src, dst in ((l18, p18), (l12, p12)):
        if dst.is_file():
            continue
        try:
            os.link(src, dst)
        except OSError:
            dst.write_bytes(src.read_bytes())
    print(f"  linked legacy {variant} {tag} → {OUT_DIR.name}/", flush=True)
    return True


def write_cut(
    variant: str,
    q_cut: float,
    dl18: np.ndarray,
    ell_log: np.ndarray,
    dl12: np.ndarray,
    selection_tag: str = TAG,
) -> None:
    tag = cut_tag(q_cut)
    p18, p12 = out_paths(variant, tag, selection_tag)
    if variant == "fiducial":
        common = (
            f"L1_m9 fiducial masked tSZ, q>{q_cut:g} ({selection_tag}); "
            f"synthetic-data masking: r=max({R_MULT:g}*theta500, 2x{FWHM_ARCMIN:g}arcmin), "
            f"{APOTYPE} apodization {APOSIZE_DEG} deg, masked monopole subtracted, "
            f"NaMaster MASTER per-ell, pixwin deconvolved"
        )
    else:
        common = (
            f"L1_m9 {variant} lightcone-0 tSZ; HEALPix Nside=4096 pixel window deconvolved; "
            f"masked q>{q_cut:g} ({selection_tag}); r=max({R_MULT:g}*theta500, 2x{FWHM_ARCMIN:g}arcmin), "
            f"{APOTYPE} apodization {APOSIZE_DEG} deg, masked monopole subtracted, "
            f"NaMaster MASTER per-ell (nlb=1, lmax={LMAX})"
        )
    bin18_note = "uniform mean of D_ell over inclusive Planck bins"
    bin12_note = (
        f"uniform mean of C_ell in log bins, Delta ln ell = {DLN_ELL}, ell_max = {LMAX}"
    )
    cols = "ell_eff  1e12_D_ell_yy"
    write_bandpowers(p18, PLANCK_ELL_EFF, dl18 * 1e12, f"{common}; {bin18_note}\n{cols}")
    write_bandpowers(p12, ell_log, dl12 * 1e12, f"{common}; {bin12_note}\n{cols}")


def process_variant(
    variant: str,
    map_stem: str,
    cat_stem: str,
    q_cuts: list[float],
    force: bool,
    selection: QSelection,
) -> dict:
    """Load one map+catalogue; compute any missing q cuts. Returns metadata."""
    t0 = time.time()
    needed: list[float] = []
    for q in q_cuts:
        tag = cut_tag(q)
        p18, p12 = out_paths(variant, tag, selection.tag)
        if not force and p18.is_file() and p12.is_file():
            continue
        if not force and selection.tag == TAG and ensure_from_legacy(variant, tag):
            continue
        needed.append(q)

    meta: dict = {
        "variant": variant,
        "needed": [cut_tag(q) for q in needed],
        "skipped": [cut_tag(q) for q in q_cuts if q not in needed],
        "cuts": {},
    }
    if not needed:
        print(f"[{variant}] nothing to do (all present)", flush=True)
        meta["runtime_seconds"] = time.time() - t0
        return meta

    print(f"[{variant}] computing {meta['needed']} ({time.strftime('%H:%M:%S')})", flush=True)
    pixwin2 = hp.pixwin(4096, lmax=LMAX) ** 2
    ell = np.arange(LMAX + 1, dtype=float)
    mask_floor = np.deg2rad(2.0 * FWHM_ARCMIN / 60.0)

    ymap = hp.read_map(map_file(map_stem), dtype=np.float64)
    nside = hp.npix2nside(ymap.size)
    if nside != 4096:
        raise RuntimeError(f"{variant}: expected nside=4096, got {nside}")
    cat_path = catalogue_path(cat_stem, selection)
    cat = load_catalogue(cat_path, q_column=selection.q_column)
    qv, theta, phi, t500 = cat["q"], cat["theta"], cat["phi"], cat["t500"]
    print(
        f"[{variant}] catalogue {qv.size:,} from {cat_path.name}",
        flush=True,
    )
    sample = cat["nat_sample"]
    r_nat = rotation_sanity(ymap, sample[:, 0], sample[:, 1])
    r_rot = rotation_sanity(ymap, sample[:, 2], sample[:, 3])
    print(
        f"[{variant}] rotation sanity: rot={r_rot['ratio']:.2f}, " f"nat={r_nat['ratio']:.2f}",
        flush=True,
    )
    if r_rot["ratio"] <= r_nat["ratio"]:
        raise RuntimeError(
            f"{variant}: rotated positions do not trace tSZ peaks better than natural ones"
        )

    for q_cut in needed:
        tag = cut_tag(q_cut)
        t_cut = time.time()
        keep = qv > q_cut
        n_masked = int(keep.sum())
        radius = np.maximum(R_MULT * t500[keep], mask_floor)
        mask_bin = disc_mask(nside, theta[keep], phi[keep], radius, inclusive=False)
        f_sky_raw = float(mask_bin.mean())
        mask_apo = nmt.mask_apodization(mask_bin, APOSIZE_DEG, apotype=APOTYPE)
        f_sky_eff = float(np.mean(mask_apo**2))
        del mask_bin
        print(
            f"[{variant}] q>{q_cut:g}: N={n_masked:,} f_sky_raw={f_sky_raw:.4f} "
            f"eff={f_sky_eff:.4f} — NaMaster starting",
            flush=True,
        )
        cl = decoupled_cl_per_ell(ymap, mask_apo, pixwin2, lmax=LMAX)
        del mask_apo
        dl18 = bin_planck_dl(ell, cl)
        ell_log, dl12 = bin_log_dl(ell, cl)
        write_cut(variant, q_cut, dl18, ell_log, dl12, selection.tag)
        dt = time.time() - t_cut
        meta["cuts"][tag] = {
            "q_cut": q_cut,
            "n_masked": n_masked,
            "f_sky_raw": f_sky_raw,
            "f_sky_eff": f_sky_eff,
            "runtime_seconds": dt,
        }
        p18, _ = out_paths(variant, tag, selection.tag)
        print(
            f"[{variant}] wrote {p18.name} in {dt/60:.1f} min "
            f"(elapsed { (time.time()-t0)/60:.1f} min)",
            flush=True,
        )

    del ymap
    meta["runtime_seconds"] = time.time() - t0
    print(f"[{variant}] done in {meta['runtime_seconds']/60:.1f} min", flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--q-cuts",
        type=float,
        nargs="+",
        default=None,
        help="q thresholds (selection-specific default)",
    )
    parser.add_argument(
        "--selection",
        choices=(TAG, "qfrommap"),
        default=TAG,
        help="catalogue q selection (default: legacy q-from-mass/redshift)",
    )
    parser.add_argument("--l1-catalogue-dir", type=Path)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="subset of variant names (default: all nine)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel variant workers (default 1; try 3–4 with OMP_NUM_THREADS=8)",
    )
    parser.add_argument("--force", action="store_true", help="recompute even if present")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list inputs and outputs without reading maps",
    )
    args = parser.parse_args()
    selection = resolve_q_selection(args.selection, l1_catalogue_dir=args.l1_catalogue_dir)
    q_cuts = args.q_cuts or default_q_cuts(args.selection)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = VARIANTS
    if args.variants is not None:
        want = set(args.variants)
        selected = [v for v in VARIANTS if v[0] in want]
        missing = want - {v[0] for v in selected}
        if missing:
            raise SystemExit(f"unknown variants: {sorted(missing)}")

    if args.dry_run:
        for variant, map_stem, cat_stem in selected:
            print(f"{variant}: map={map_file(map_stem)}")
            print(f"  catalogue={catalogue_path(cat_stem, selection)}")
            for q in q_cuts:
                print(f"  outputs={out_paths(variant, cut_tag(q), selection.tag)}")
        return

    # Pre-link any legacy qgt1/qgt5 into binned_bandpowers so workers skip them.
    if not args.force and selection.tag == TAG:
        for variant, _, _ in selected:
            for q in q_cuts:
                ensure_from_legacy(variant, cut_tag(q))

    print(
        f"OUT_DIR={OUT_DIR}\n"
        f"variants={[v[0] for v in selected]}\n"
        f"selection={selection.tag} q_column={selection.q_column}\n"
        f"q_cuts={q_cuts}\n"
        f"workers={args.workers}  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}",
        flush=True,
    )

    t0 = time.time()
    results: list[dict] = []
    if args.workers <= 1:
        for variant, map_stem, cat_stem in selected:
            results.append(
                process_variant(variant, map_stem, cat_stem, q_cuts, args.force, selection)
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(
                    process_variant,
                    variant,
                    map_stem,
                    cat_stem,
                    q_cuts,
                    args.force,
                    selection,
                ): variant
                for variant, map_stem, cat_stem in selected
            }
            for fut in as_completed(futs):
                variant = futs[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:
                    print(f"[{variant}] FAILED: {exc}", flush=True)
                    raise

    meta_path = OUT_DIR / (f"L1_m9_feedback_multi_q_bandpowers_{selection.tag}_metadata.json")
    payload = {
        "out_dir": str(OUT_DIR),
        "catalogue_suffix": f"_yang26rot_{selection.tag}.csv",
        "q_column": selection.q_column,
        "masking": {
            "radius": f"max({R_MULT:g}*theta500, 2*FWHM), FWHM={FWHM_ARCMIN:g} arcmin",
            "apodization": f"{APOTYPE} {APOSIZE_DEG} deg",
            "estimator": f"NaMaster MASTER, nlb=1, lmax={LMAX}, no beam, pixwin deconvolved",
        },
        "q_cuts": list(q_cuts),
        "workers": args.workers,
        "variants": {r["variant"]: r for r in results},
        "runtime_seconds": time.time() - t0,
    }
    if meta_path.exists():
        with open(meta_path) as handle:
            payload = merge_metadata_payload(json.load(handle), payload)
    with open(meta_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {meta_path}", flush=True)
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
