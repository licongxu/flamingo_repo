"""Paper-pipeline NaMaster multi-q bandpowers for ALL L1_m9 feedback variants.

Same catalogue / mask / estimator as
``compute_l1_m9_feedback_bandpowers.py`` (q>5) and
``compute_l1_m9_feedback_bandpowers_qgt1.py``:

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
    python scripts/compute_l1_m9_feedback_ratio_vs_q_bandpowers.py --workers 4
"""
from __future__ import annotations

import argparse
import importlib.util
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

REPO = Path(__file__).resolve().parents[1]
FEEDBACK = REPO / "scripts" / "compute_l1_m9_feedback_bandpowers.py"
OUT_DIR = REPO / "data_paper" / "binned_bandpowers"
LEGACY_FB = REPO / "data_paper" / "feedback_bandpower"

spec = importlib.util.spec_from_file_location("feedback_bp", FEEDBACK)
mod = importlib.util.module_from_spec(spec)
sys.modules["feedback_bp"] = mod
spec.loader.exec_module(mod)

# All nine prescriptions (same as paper feedback compute).
VARIANTS = list(mod.VARIANTS)  # (name, map_stem, cat_stem)

DEFAULT_Q_CUTS = [50.0, 20.0, 10.0, 5.0, 3.0, 1.0]


def cut_tag(q: float) -> str:
    if float(q).is_integer():
        return f"qgt{int(q)}"
    return f"qgt{str(q).replace('.', 'p')}"


def out_paths(variant: str, tag: str) -> tuple[Path, Path]:
    """Paper naming: fiducial uses L1_m9 without variant token."""
    if variant == "fiducial":
        stem = f"Dl_yy_L1_m9_masked_{tag}_{mod.TAG}"
    else:
        stem = f"Dl_yy_L1_m9_{variant}_masked_{tag}_{mod.TAG}"
    return (
        OUT_DIR / f"{stem}_binned_18.txt",
        OUT_DIR / f"{stem}_logbins_dln0p4_lmax10000.txt",
    )


def legacy_paths(variant: str, tag: str) -> tuple[Path, Path]:
    """Older location under feedback_bandpower (qgt1/qgt5 already there)."""
    stem = f"Dl_yy_L1_m9_{variant}_masked_{tag}_{mod.TAG}"
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


def write_cut(variant: str, q_cut: float, dl18: np.ndarray, dl12: np.ndarray) -> None:
    tag = cut_tag(q_cut)
    p18, p12 = out_paths(variant, tag)
    if variant == "fiducial":
        common = (
            f"L1_m9 fiducial masked tSZ, q>{q_cut:g} ({mod.TAG}); "
            f"synthetic-data masking: r=max({mod.R_MULT:g}*theta500, 2x{mod.FWHM_ARCMIN:g}arcmin), "
            f"{mod.APOTYPE} apodization {mod.APOSIZE_DEG} deg, masked monopole subtracted, "
            f"NaMaster MASTER per-ell, pixwin deconvolved"
        )
    else:
        common = (
            f"L1_m9 {variant} lightcone-0 tSZ; HEALPix Nside=4096 pixel window deconvolved; "
            f"masked q>{q_cut:g} ({mod.TAG}); r=max({mod.R_MULT:g}*theta500, 2x{mod.FWHM_ARCMIN:g}arcmin), "
            f"{mod.APOTYPE} apodization {mod.APOSIZE_DEG} deg, masked monopole subtracted, "
            f"NaMaster MASTER per-ell (nlb=1, lmax={mod.LMAX})"
        )
    bin18_note = "uniform mean of D_ell over inclusive Planck bins"
    bin12_note = (
        f"uniform mean of C_ell in log bins, Delta ln ell = {mod.DLN_ELL}, ell_max = {mod.LMAX}"
    )
    cols = "ell_eff  1e12_D_ell_yy"
    mod.write_bandpowers(
        p18, mod.ELL_EFF, dl18 * 1e12, f"{common}; {bin18_note}\n{cols}"
    )
    mod.write_bandpowers(
        p12, mod.LOG_CENTRES, dl12 * 1e12, f"{common}; {bin12_note}\n{cols}"
    )


def process_variant(
    variant: str,
    map_stem: str,
    cat_stem: str,
    q_cuts: list[float],
    force: bool,
) -> dict:
    """Load one map+catalogue; compute any missing q cuts. Returns metadata."""
    t0 = time.time()
    needed: list[float] = []
    for q in q_cuts:
        tag = cut_tag(q)
        p18, p12 = out_paths(variant, tag)
        if not force and p18.is_file() and p12.is_file():
            continue
        if not force and ensure_from_legacy(variant, tag):
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
    pixwin2 = hp.pixwin(4096, lmax=mod.LMAX) ** 2
    ell = np.arange(mod.LMAX + 1, dtype=float)
    mask_floor = np.deg2rad(2.0 * mod.FWHM_ARCMIN / 60.0)

    ymap = hp.read_map(mod.map_file(map_stem), dtype=np.float64)
    nside = hp.npix2nside(ymap.size)
    if nside != 4096:
        raise RuntimeError(f"{variant}: expected nside=4096, got {nside}")
    cat = mod.load_catalogue(mod.cat_file(cat_stem))
    qv, theta, phi, t500 = cat["q"], cat["theta"], cat["phi"], cat["t500"]
    print(
        f"[{variant}] catalogue {qv.size:,} from {mod.cat_file(cat_stem).name}",
        flush=True,
    )

    for q_cut in needed:
        tag = cut_tag(q_cut)
        t_cut = time.time()
        keep = qv > q_cut
        n_masked = int(keep.sum())
        radius = np.maximum(mod.R_MULT * t500[keep], mask_floor)
        mask_bin = mod.binary_disc_mask(nside, theta[keep], phi[keep], radius)
        f_sky_raw = float(mask_bin.mean())
        mask_apo = nmt.mask_apodization(mask_bin, mod.APOSIZE_DEG, apotype=mod.APOTYPE)
        f_sky_eff = float(np.mean(mask_apo**2))
        del mask_bin
        print(
            f"[{variant}] q>{q_cut:g}: N={n_masked:,} f_sky_raw={f_sky_raw:.4f} "
            f"eff={f_sky_eff:.4f} — NaMaster starting",
            flush=True,
        )
        cl = mod.decoupled_cl_per_ell(ymap, mask_apo, pixwin2)
        del mask_apo
        dl18 = mod.bin_dl_18(ell, cl)
        dl12 = mod.bin_cl_log(ell, cl)
        write_cut(variant, q_cut, dl18, dl12)
        dt = time.time() - t_cut
        meta["cuts"][tag] = {
            "q_cut": q_cut,
            "n_masked": n_masked,
            "f_sky_raw": f_sky_raw,
            "f_sky_eff": f_sky_eff,
            "runtime_seconds": dt,
        }
        p18, _ = out_paths(variant, tag)
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
        default=DEFAULT_Q_CUTS,
        help="q thresholds (default: 50 20 10 5 3 1)",
    )
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
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = VARIANTS
    if args.variants is not None:
        want = set(args.variants)
        selected = [v for v in VARIANTS if v[0] in want]
        missing = want - {v[0] for v in selected}
        if missing:
            raise SystemExit(f"unknown variants: {sorted(missing)}")

    # Pre-link any legacy qgt1/qgt5 into binned_bandpowers so workers skip them.
    if not args.force:
        for variant, _, _ in selected:
            for q in args.q_cuts:
                ensure_from_legacy(variant, cut_tag(q))

    print(
        f"OUT_DIR={OUT_DIR}\n"
        f"variants={[v[0] for v in selected]}\n"
        f"q_cuts={args.q_cuts}\n"
        f"workers={args.workers}  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}",
        flush=True,
    )

    t0 = time.time()
    results: list[dict] = []
    if args.workers <= 1:
        for variant, map_stem, cat_stem in selected:
            results.append(
                process_variant(variant, map_stem, cat_stem, args.q_cuts, args.force)
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(
                    process_variant, variant, map_stem, cat_stem, args.q_cuts, args.force
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

    meta_path = OUT_DIR / "L1_m9_feedback_multi_q_bandpowers_metadata.json"
    payload = {
        "out_dir": str(OUT_DIR),
        "catalogue_suffix": "_yang26rot_qfrommz_alpha_fixed_1p12.csv",
        "masking": {
            "radius": f"max({mod.R_MULT:g}*theta500, 2*FWHM), FWHM={mod.FWHM_ARCMIN:g} arcmin",
            "apodization": f"{mod.APOTYPE} {mod.APOSIZE_DEG} deg",
            "estimator": f"NaMaster MASTER, nlb=1, lmax={mod.LMAX}, no beam, pixwin deconvolved",
        },
        "q_cuts": list(args.q_cuts),
        "workers": args.workers,
        "variants": {r["variant"]: r for r in results},
        "runtime_seconds": time.time() - t0,
    }
    with open(meta_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {meta_path.relative_to(REPO)}", flush=True)
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
