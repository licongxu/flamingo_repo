"""Step 1: assign a CNC detection significance ``q`` to every halo.

For each L1_m9 feedback variant this streams the ``M_500c > 5e13 Msun``,
``z < 3`` halo catalogue, evaluates the SZ scaling relation of
:class:`flamingo.cnc.SZScaling` on ``(M_500c, z)``, and caches a compact array
of everything the masking step needs::

    theta_rot_rad, phi_rot_rad   sky position in the yang26-rotated frame
    theta_500_rad                true angular size, sets the masking radius
    q                            detection significance, sets which halos are masked

No map is touched: ``q`` follows from the assumed scaling relation alone, which
is what a Planck-like cluster survey would report for these halos.

Two angular sizes appear and must not be confused:

* the *true* ``theta_500 = R_500c / D_A(z)`` from the catalogue, which defines
  the physical extent of the tSZ signal and hence the masking radius;
* the *hydrostatic* ``theta_500(M_500c / B)`` internal to the scaling relation,
  which sets the matched-filter noise ``sigma_y0`` entering ``q``.

Run::

    python -m paper_results.compute_q
    python -m paper_results.compute_q --variant L1_m9 --force
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import config
from flamingo.catalogue import theta_500
from flamingo.cnc import SZScaling

#: Rows read per chunk when streaming the (multi-hundred-MB) catalogues.
CHUNK = 1_000_000

#: Catalogue columns the scaling relation and the masking step need.
COLUMNS = [
    "soap_index",
    "z",
    "M_500c_Msun",
    "R_500c_Mpc",
    "theta_rot_rad",
    "phi_rot_rad",
    "q_from_mz",
]


def compute_variant(variant: str, sz: SZScaling, *, force: bool = False) -> dict:
    """Compute and cache the ``q`` catalogue of one feedback variant.

    Parameters
    ----------
    variant : str
        Feedback-variant name, e.g. ``"L1_m9"`` or ``"Jet_fgas-4sigma"``.
    sz : flamingo.cnc.SZScaling
        Calibrated scaling relation.
    force : bool, optional
        Recompute even if the cache file already exists.

    Returns
    -------
    dict
        Summary with the number of halos above each cut in ``config.Q_CUTS``
        and the maximum fractional deviation from the ``q`` stored in the
        catalogue (a regression check against the original pipeline).
    """
    out = config.qcat_path(variant)
    if out.exists() and not force:
        cached = np.load(out)
        print(f"  {variant}: cached {out.name}", flush=True)
        return _summarise(variant, cached["q"], float(cached["max_frac_dev"]))

    src = config.catalogue_path(variant)
    if not src.exists():
        raise FileNotFoundError(f"missing catalogue {src}")

    t0 = time.time()
    cols: dict[str, list[np.ndarray]] = {k: [] for k in ("theta", "phi", "t500", "q", "q_ref")}
    n_rows = 0
    for chunk in pd.read_csv(src, comment="#", usecols=COLUMNS, chunksize=CHUNK):
        m = chunk["M_500c_Msun"].to_numpy(np.float64)
        z = chunk["z"].to_numpy(np.float64)
        cols["q"].append(sz.q(m, z, index=chunk["soap_index"].to_numpy(np.uint32)))
        cols["q_ref"].append(chunk["q_from_mz"].to_numpy(np.float64))
        cols["t500"].append(theta_500(chunk["R_500c_Mpc"].to_numpy(np.float64), z))
        cols["theta"].append(chunk["theta_rot_rad"].to_numpy(np.float64))
        cols["phi"].append(chunk["phi_rot_rad"].to_numpy(np.float64))
        n_rows += len(chunk)
        print(f"    {variant}: {n_rows:,} halos ({time.time() - t0:.0f}s)", flush=True)

    joined = {k: np.concatenate(v) for k, v in cols.items()}
    q, q_ref = joined["q"], joined["q_ref"]
    max_frac_dev = float(np.nanmax(np.abs(q / q_ref - 1.0)))

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        theta_rot_rad=joined["theta"],
        phi_rot_rad=joined["phi"],
        theta_500_rad=joined["t500"],
        q=q,
        variant=variant,
        B=sz.B,
        A_SZ=sz.A_SZ,
        alpha_SZ=sz.alpha_SZ,
        sigma_lnY=sz.sigma_lnY,
        source=str(src),
        max_frac_dev=max_frac_dev,
    )
    print(f"  {variant}: {n_rows:,} halos -> {out.name} ({time.time() - t0:.0f}s)", flush=True)
    return _summarise(variant, q, max_frac_dev)


def _summarise(variant: str, q: np.ndarray, max_frac_dev: float) -> dict:
    """Number of halos above each ``config.Q_CUTS`` threshold, plus the check."""
    return dict(
        variant=variant,
        n_halos=int(q.size),
        counts={cut: int(np.sum(q > cut)) for cut in config.Q_CUTS},
        max_frac_dev=max_frac_dev,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", action="append", help="variant(s) to process")
    parser.add_argument("--force", action="store_true", help="ignore cached outputs")
    args = parser.parse_args()

    sz = SZScaling.calibrated(
        sigma_y0_file=config.SIGMA_Y0_FILE, skyfracs_file=config.SKYFRACS_FILE
    )
    print(
        f"SZ scaling relation: B={sz.B}  A_SZ={sz.A_SZ:.6f}  "
        f"alpha_SZ={sz.alpha_SZ:.4f}  sigma_lnY={sz.sigma_lnY}",
        flush=True,
    )

    summaries = [
        compute_variant(v, sz, force=args.force) for v in (args.variant or config.VARIANTS)
    ]

    header = "  ".join(f"N(q>{c:g})" for c in config.Q_CUTS)
    print(f"\n{'variant':28s} {'N halos':>10s}  {header}   max|dq/q|")
    for s in summaries:
        counts = "  ".join(f"{s['counts'][c]:8,d}" for c in config.Q_CUTS)
        print(
            f"{s['variant']:28s} {s['n_halos']:10,d}  {counts}   {s['max_frac_dev']:.2e}",
            flush=True,
        )

    if not args.variant:
        write_counts_table(summaries)


def write_counts_table(summaries: list[dict]) -> None:
    """Write ``results/counts.md``: the number of detected clusters per variant."""
    lines = [
        "# Detected clusters per L1_m9 feedback variant",
        "",
        f"Halos with `M_500c > {config.M_MIN_MSUN:.0e} Msun`, `z < 3`, lightcone 0.",
        "`q` is the significance predicted by the SZ scaling relation of",
        "`flamingo.cnc.SZScaling` (custom-GNFW best fit A_SZ=-4.0953238, "
        "alpha_SZ=1.12, B=1.41, sigma_lnY=0.173, szifi immf6);",
        "clusters above each threshold are the ones masked in the power spectra.",
        "",
        "| variant | N halos | " + " | ".join(f"N(q>{c:g})" for c in config.Q_CUTS) + " |",
        "|---|---:|" + "---:|" * len(config.Q_CUTS),
    ]
    for s in summaries:
        counts = " | ".join(f"{s['counts'][c]:,d}" for c in config.Q_CUTS)
        lines.append(f"| `{s['variant']}` | {s['n_halos']:,d} | {counts} |")

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "counts.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {(config.RESULTS / 'counts.md').relative_to(config.REPO)}")


if __name__ == "__main__":
    main()
