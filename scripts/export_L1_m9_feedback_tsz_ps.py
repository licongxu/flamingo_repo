"""Export full-sky tSZ D_ell for L1_m9 feedback y-maps (notebook 35 cache).

Computes anafast spectra with monopole subtracted (notebook 05 convention) and
writes ``data/nb35_l1_m9_feedback_tsz_ps.npz`` for fast reload in the notebook.
"""
from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
OUT = _REPO / "data/nb35_l1_m9_feedback_tsz_ps.npz"
MAP_DIR = Path("/rds/rds-lxu/flamingo/L1_m9/maps")
LMAX = 6000
ELL_MIN = 10
NBIN = 40

VARIANTS = [
    "L1_m9",
    "fgas+2sigma",
    "fgas-2sigma",
    "fgas-4sigma",
    "fgas-8sigma",
    "Mstar-1sigma",
    "Mstar-1sigma_fgas-4sigma",
    "Jet",
    "Jet_fgas-4sigma",
]


def map_path(variant: str) -> Path:
    return MAP_DIR / f"y_unlensed_{variant}_lc0_nside4096.fits"


def full_sky_dl(ymap: np.ndarray, *, lmax: int = LMAX) -> tuple[np.ndarray, np.ndarray, float, float]:
    nside = hp.npix2nside(ymap.size)
    f_sky = float(np.mean(ymap != 0.0))
    monopole = float(ymap.mean())
    y_nomono = ymap - monopole
    cl = hp.anafast(y_nomono, lmax=lmax)
    ell = np.arange(cl.size)
    pw = hp.pixwin(nside, lmax=lmax)
    cl = cl / (pw**2 * f_sky)
    dl = ell * (ell + 1) / (2 * np.pi) * cl
    return ell, dl, monopole, f_sky


def log_bin(ell: np.ndarray, dl: np.ndarray, *, lmin: int = ELL_MIN, lmax: int = LMAX, nbin: int = NBIN):
    edges = np.logspace(np.log10(lmin), np.log10(lmax), nbin + 1)
    idx = np.digitize(ell, edges) - 1
    lb, db = [], []
    for b in range(nbin):
        sel = (idx == b) & (ell >= lmin)
        if sel.any():
            lb.append(ell[sel].mean())
            db.append(dl[sel].mean())
    return np.array(lb), np.array(db)


def main() -> None:
    variants: list[str] = []
    ell_ref: np.ndarray | None = None
    dl_rows: list[np.ndarray] = []
    ell_b_rows: list[np.ndarray] = []
    dl_b_rows: list[np.ndarray] = []
    monopoles: list[float] = []
    f_skies: list[float] = []

    for variant in VARIANTS:
        path = map_path(variant)
        if not path.exists():
            print(f"skip {variant}: missing {path}")
            continue
        print(f"computing {path.name} ...", flush=True)
        ymap = hp.read_map(path, verbose=False)
        ell, dl, monopole, f_sky = full_sky_dl(ymap)
        ell_b, dl_b = log_bin(ell, dl)
        if ell_ref is None:
            ell_ref = ell
        elif not np.array_equal(ell, ell_ref):
            raise ValueError(f"ell grid mismatch for {variant}")
        variants.append(variant)
        dl_rows.append(dl)
        ell_b_rows.append(ell_b)
        dl_b_rows.append(dl_b)
        monopoles.append(monopole)
        f_skies.append(f_sky)
        print(
            f"  monopole={monopole:.4e}, D_ell(l~3000)={float(np.interp(3000, ell, dl)):.4e}",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        variants=np.array(variants),
        ell=ell_ref,
        dl=np.stack(dl_rows, axis=0),
        ell_b=np.stack(ell_b_rows, axis=0),
        dl_b=np.stack(dl_b_rows, axis=0),
        monopole=np.array(monopoles),
        f_sky=np.array(f_skies),
        lmax=LMAX,
        ell_min=ELL_MIN,
        nbin=NBIN,
        map_dir=str(MAP_DIR),
    )
    print(f"wrote {OUT} ({len(variants)} variants)")


if __name__ == "__main__":
    main()
