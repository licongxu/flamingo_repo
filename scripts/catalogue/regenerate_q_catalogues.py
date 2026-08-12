"""Recompute ``q_from_mz`` in every catalogue after the M_500c selection fix.

``flamingo.cnc.SZScaling`` used to build its :class:`~hmfast.halos.HaloModel`
without declaring a mass definition, so hmfast defaulted to ``M_200c`` and
``compute_y0_parametric`` / ``compute_theta500_arcmin`` re-converted the
already-``M_500c`` catalogue masses down to ``M_500c`` a second time. Masses
came out ~0.67x too small and ``q`` ~0.46x too low, so every mask was cut at
the wrong threshold. With the fix the A10-anchored amplitude calibrates to
``A_SZ = -4.237656``, matching the reference chains.

This rewrites the ``q_from_mz`` column of every ``*_qfrommz*.csv`` in place
(atomically, keeping a ``.pre_massdef_fix.bak`` copy) across the nine L1_m9
feedback variants and every L2p8_m9 lightcone.

All catalogues now share **one** selection convention: the custom-GNFW best fit
to the L1_m9 full-sky tSZ power spectrum, ``A_SZ = -4.0953238``,
``alpha_SZ = 1.12``, ``B = 1.41``. That amplitude comes from a full-sky fit,
which never involves the selection, so it is unaffected by the bug -- and it is
the same ``B`` the pressure profile uses, so selection and signal share one
mass calibration. The ``*_qfrommz_bestfit.csv`` flavour is skipped: nothing
consumes it, and keeping it would carry a second convention forward.

Run::

    python scripts/catalogue/regenerate_q_catalogues.py --dry-run
    python scripts/catalogue/regenerate_q_catalogues.py
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from flamingo.cnc import SZScaling  # noqa: E402

#: Directories holding the yang26-rotated ``q_from_mz`` catalogues: the nine
#: L1_m9 feedback variants plus every L2p8_m9 lightcone.
FLAMINGO = Path("/rds/rds-lxu/flamingo")
CATALOGUE_DIRS = [FLAMINGO / "L1_m9" / "catalogues"] + sorted(
    FLAMINGO.glob("L2p8_m9/lightcone*/catalogues")
)

#: Flavours that nothing consumes any more; regenerating them would just carry
#: a second selection convention forward.
DEAD_SUFFIXES = ("_qfrommz_bestfit.csv",)

#: The single selection convention used everywhere: the custom-GNFW best fit to
#: the L1_m9 full-sky tSZ power spectrum, at fixed ``alpha_SZ`` and the same
#: hydrostatic bias as the pressure profile. Selection and signal therefore
#: share one mass calibration.
A_SZ = -4.0953238
ALPHA_SZ = 1.12
B = 1.41

CHUNK = 1_000_000
BACKUP_SUFFIX = ".pre_massdef_fix.bak"
FIX_MARKER = "q recomputed with M_500c mass definition"
_HEADER_VALUE = re.compile(r"(A_SZ|alpha_SZ)\s*=\s*(-?[\d.eE+-]+)")


def read_header(path: Path) -> tuple[list[str], dict[str, float]]:
    """Return the leading ``#`` comment lines and any ``A_SZ``/``alpha_SZ`` in them."""
    lines: list[str] = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            lines.append(line.rstrip("\n"))
    values = {k: float(v) for k, v in _HEADER_VALUE.findall(" ".join(lines))}
    return lines, values


def scaling_for(base: SZScaling) -> SZScaling:
    """The one selection convention, applied to every catalogue."""
    return replace(base, A_SZ=A_SZ, alpha_SZ=ALPHA_SZ, B=B)


def provenance(scaling: SZScaling, previous: list[str]) -> str:
    """Header for the rewritten catalogue, preserving the original description."""
    kept = [line for line in previous if not _HEADER_VALUE.search(line)]
    return (
        "\n".join(
            kept
            + [
                f"# A_SZ={scaling.A_SZ:.8g} alpha_SZ={scaling.alpha_SZ:.8g} "
                f"B={scaling.B:.8g}",
                f"# sigma_lnY={scaling.sigma_lnY:.8g} seed={scaling.seed} "
                "cosmology=D3A noise=SZiFi-immf6",
                f"# {FIX_MARKER} (selection fix); "
                "supersedes q values built with hmfast's default M_200c.",
            ]
        )
        + "\n"
    )


def rewrite(path: Path, scaling: SZScaling, *, backup: bool = True) -> dict:
    """Stream ``path`` and atomically replace it with recomputed ``q_from_mz``."""
    header, _ = read_header(path)
    before = path.stat()
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")

    rows = 0
    q_old_max = -np.inf
    q_new_max = -np.inf
    ratios: list[np.ndarray] = []
    started = time.time()
    try:
        with temporary.open("x") as handle:
            handle.write(provenance(scaling, header))

        first = True
        for chunk in pd.read_csv(
            path, comment="#", chunksize=CHUNK, float_precision="round_trip"
        ):
            for column in ("M_500c_Msun", "z", "soap_index", "q_from_mz"):
                if column not in chunk.columns:
                    raise ValueError(f"{path.name}: missing column {column}")
            q_old = chunk["q_from_mz"].to_numpy(np.float64)
            q = np.asarray(
                scaling.q(
                    chunk["M_500c_Msun"].to_numpy(np.float64),
                    chunk["z"].to_numpy(np.float64),
                    index=chunk["soap_index"].to_numpy(np.uint32),
                ),
                dtype=np.float64,
            )
            if q.shape != (len(chunk),):
                raise ValueError(f"q has shape {q.shape}, expected {(len(chunk),)}")
            if not np.all(np.isfinite(q)) or not np.all(q > 0):
                raise ValueError(f"{path.name}: non-finite or non-positive q")

            chunk["q_from_mz"] = q
            chunk.to_csv(
                temporary, mode="a", header=first, index=False, float_format="%.17g"
            )
            first = False
            rows += len(chunk)
            q_old_max = max(q_old_max, float(q_old.max()))
            q_new_max = max(q_new_max, float(q.max()))
            ratios.append((q / q_old)[:: max(1, len(q) // 1000)])
            print(f"    {path.name}: {rows:,} rows ({time.time()-started:.0f}s)", flush=True)

        if first:
            raise ValueError(f"{path.name}: no data rows")
        after = path.stat()
        if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
            raise RuntimeError(f"{path.name} changed while being read")

        # Never clobber an existing backup: on a re-run that would replace the
        # original q values with already-corrected ones.
        target = path.with_name(path.name + BACKUP_SUFFIX)
        if backup and not target.exists():
            shutil.copy2(path, target)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    ratio = np.concatenate(ratios)
    return {
        "file": path.name,
        "rows": rows,
        "A_SZ": scaling.A_SZ,
        "alpha_SZ": scaling.alpha_SZ,
        "median_q_ratio": float(np.median(ratio)),
        "q_max_old": q_old_max,
        "q_max_new": q_new_max,
        "seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="list work, change nothing")
    parser.add_argument("--no-backup", action="store_true", help="skip .bak copies")
    parser.add_argument("--only", action="append", help="substring filter on filenames")
    parser.add_argument("--force", action="store_true", help="redo already-regenerated files")
    args = parser.parse_args()

    scaling = scaling_for(SZScaling.calibrated(B=B))
    print(
        "single selection convention: "
        f"A_SZ={scaling.A_SZ:.8g} alpha_SZ={scaling.alpha_SZ:.8g} B={scaling.B:.8g} "
        f"sigma_lnY={scaling.sigma_lnY:.8g}"
    )

    paths = sorted(
        p
        for directory in CATALOGUE_DIRS
        if directory.is_dir()
        for p in directory.glob("*qfrommz*.csv")
        if not p.name.endswith(".bak") and not p.name.endswith(DEAD_SUFFIXES)
    )
    if args.only:
        paths = [p for p in paths if any(s in p.name for s in args.only)]
    print(f"{len(paths)} catalogue(s) to regenerate\n")

    summaries = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.parent.parent.name}/{path.name}", flush=True)
        if args.dry_run:
            continue
        if not args.force and any(FIX_MARKER in line for line in read_header(path)[0]):
            print("    already regenerated; skipping\n", flush=True)
            continue
        summaries.append(rewrite(path, scaling, backup=not args.no_backup))
        print(f"    -> median q_new/q_old = {summaries[-1]['median_q_ratio']:.4f}\n", flush=True)

    if summaries:
        print(f"\n{'file':70s} {'rows':>12s} {'q_new/q_old':>12s}")
        for s in summaries:
            print(f"{s['file']:70s} {s['rows']:12,d} {s['median_q_ratio']:12.4f}")


if __name__ == "__main__":
    main()
