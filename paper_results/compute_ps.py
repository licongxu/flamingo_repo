"""Step 2: tSZ power spectra of the L1_m9 y-maps, unmasked and cluster-masked.

For each feedback variant this measures the Compton-y auto-spectrum of the
``nside=4096`` lightcone-0 map:

* **unmasked** -- the full-sky tSZ power spectrum;
* **masked** -- the same map after removing a disc of radius
  ``R_MASK x theta_500`` around every halo with ``q`` above each threshold in
  ``config.Q_CUTS``, i.e. the power that would remain after a Planck-like
  cluster survey excised its own detections.

Both use the identical NaMaster estimator (C1-apodized mask, mask-weighted
monopole removed, mask-decoupled pseudo-Cl, linear ``Delta ell`` bandpowers,
HEALPix pixel window deconvolved), so unmasked and masked bandpowers are
directly comparable. The full-sky case runs through NaMaster with a unit mask
rather than ``anafast`` for exactly that reason.

Requires the ``q`` cache from :mod:`paper_results.compute_q`.

Run::

    python -m paper_results.compute_ps
    python -m paper_results.compute_ps --variant L1_m9 --force
"""
from __future__ import annotations

import argparse
import time

import healpy as hp
import numpy as np

from . import config
from flamingo.masking import disc_mask, fsky
from flamingo.powerspectra.namaster import apodize, decoupled_dl


def _bandpowers(ymap: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-window-corrected bandpowers of ``ymap`` under an (apodized) mask."""
    ell, dl, _ = decoupled_dl(
        ymap,
        mask,
        delta_ell=config.DELTA_ELL,
        lmax=config.LMAX,
        deconvolve_pixwin=True,
    )
    return ell, dl


def compute_variant(variant: str, *, force: bool = False) -> dict:
    """Measure unmasked and ``q``-masked bandpowers for one feedback variant.

    Parameters
    ----------
    variant : str
        Feedback-variant name, e.g. ``"L1_m9"``.
    force : bool, optional
        Recompute even if the output file already exists.

    Returns
    -------
    dict
        ``ell``, ``dl_fullsky``, ``dl_masked`` (one row per cut), ``n_masked``
        and ``fsky`` (binary-mask sky fraction per cut).
    """
    out = config.BANDPOWERS / f"{variant}.npz"
    if out.exists() and not force:
        print(f"=== {variant}: cached {out.name} ===", flush=True)
        return dict(np.load(out, allow_pickle=True))

    mpath = config.map_path(variant)
    qpath = config.qcat_path(variant)
    for path in (mpath, qpath):
        if not path.exists():
            raise FileNotFoundError(f"missing input {path}")

    t0 = time.time()
    print(f"=== {variant} ===", flush=True)
    ymap = hp.read_map(mpath)
    nside = hp.npix2nside(ymap.size)

    qcat = np.load(qpath)
    q = qcat["q"]
    theta, phi, theta_500 = qcat["theta_rot_rad"], qcat["phi_rot_rad"], qcat["theta_500_rad"]
    print(f"  map nside={nside}, {q.size:,} halos, mean y={ymap.mean():.3e}", flush=True)

    ell, dl_fullsky = _bandpowers(ymap, None)
    print(f"  full sky done ({time.time() - t0:.0f}s)", flush=True)

    dl_masked, n_masked, fsky_binary = [], [], []
    for cut, tag in zip(config.Q_CUTS, config.CUT_TAGS):
        keep = np.isfinite(q) & (q > cut)
        mask = disc_mask(
            nside, theta[keep], phi[keep], config.R_MASK * theta_500[keep]
        )
        _, dl = _bandpowers(ymap, apodize(mask, aperture_deg=config.APOD_DEG))
        dl_masked.append(dl)
        n_masked.append(int(keep.sum()))
        fsky_binary.append(fsky(mask))
        print(
            f"  q>{cut:g}: masked {n_masked[-1]:,} halos, "
            f"f_sky={fsky_binary[-1]:.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    result = dict(
        variant=variant,
        ell=ell,
        dl_fullsky=dl_fullsky,
        dl_masked=np.stack(dl_masked),
        q_cuts=np.array(config.Q_CUTS),
        cut_tags=np.array(config.CUT_TAGS),
        n_masked=np.array(n_masked),
        fsky=np.array(fsky_binary),
        nside=nside,
        lmax=config.LMAX,
        delta_ell=config.DELTA_ELL,
        r_mask=config.R_MASK,
        apod_deg=config.APOD_DEG,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **result)
    _write_text(variant, ell, dl_fullsky, result["dl_masked"])
    print(f"  wrote {out.name} ({time.time() - t0:.0f}s)", flush=True)
    return result


def _write_text(
    variant: str, ell: np.ndarray, dl_fullsky: np.ndarray, dl_masked: np.ndarray
) -> None:
    """Also write plain ``ell D_ell`` columns, one file per variant and cut."""
    columns = [(config.FULLSKY_TAG, dl_fullsky)] + list(zip(config.CUT_TAGS, dl_masked))
    for tag, dl in columns:
        np.savetxt(
            config.BANDPOWERS / f"Dl_yy_{variant}_{tag}.txt",
            np.column_stack([ell, dl]),
            fmt="%.6e",
            header=f"{variant} {tag}\nell  D_ell = ell(ell+1)C_ell/2pi",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", action="append", help="variant(s) to process")
    parser.add_argument("--force", action="store_true", help="ignore cached outputs")
    args = parser.parse_args()

    results = [compute_variant(v, force=args.force) for v in (args.variant or config.VARIANTS)]

    header = "  ".join(f"q>{c:g}" for c in config.Q_CUTS)
    print(f"\n{'variant':28s}  masked halos / retained f_sky:  {header}")
    for r in results:
        counts = "  ".join(
            f"{n:,} ({f:.3f})" for n, f in zip(r["n_masked"], r["fsky"])
        )
        print(f"{str(r['variant']):28s}  {counts}", flush=True)


if __name__ == "__main__":
    main()
