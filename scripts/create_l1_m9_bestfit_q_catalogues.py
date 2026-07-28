#!/usr/bin/env python
"""Create L1_m9 catalogues with ``q_from_mz`` from the custom-GNFW best fit."""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from flamingo.cnc import SZScaling  # noqa: E402
from paper_results import config  # noqa: E402


A_SZ = -4.1095805
ALPHA_SZ = 0.97447729
B = 1.41
CHUNKSIZE = 1_000_000
REQUIRED_COLUMNS = {"soap_index", "z", "M_500c_Msun", "q_from_mz"}


@dataclass(frozen=True)
class CatalogueSummary:
    """Summary of one generated best-fit catalogue."""

    source: Path
    output: Path
    rows: int
    q_min: float
    q_max: float


def bestfit_scaling() -> SZScaling:
    """Build the fitted scaling relation with the existing selection settings."""
    base = SZScaling.calibrated(
        B=B,
        alpha_SZ=ALPHA_SZ,
        sigma_y0_file=config.SIGMA_Y0_FILE,
        skyfracs_file=config.SKYFRACS_FILE,
    )
    return replace(base, A_SZ=A_SZ)


def bestfit_path(source: Path) -> Path:
    """Return the sibling ``_qfrommz_bestfit.csv`` path."""
    if not source.name.endswith("_qfrommz.csv"):
        raise ValueError(f"not a q_from_mz catalogue: {source}")
    return source.with_name(source.name.removesuffix(".csv") + "_bestfit.csv")


def catalogue_path(catalogue_dir: Path, variant: str) -> Path:
    """Return the existing fiducial catalogue path for one feedback variant."""
    return catalogue_dir / (
        f"halo_catalogue_M500c_5e13_zlt3_{variant}_yang26rot_qfrommz.csv"
    )


def _provenance(source: Path, scaling: SZScaling) -> str:
    return "\n".join(
        [
            "# L1_m9 q_from_mz catalogue using the custom-GNFW best-fit scaling relation.",
            f"# source={source}",
            (
                f"# A_SZ={scaling.A_SZ:.8g} alpha_SZ={scaling.alpha_SZ:.8g} "
                f"B={scaling.B:.8g}"
            ),
            (
                f"# sigma_lnY={scaling.sigma_lnY:.8g} seed={scaling.seed} "
                "cosmology=D3A noise=SZiFi-immf6"
            ),
        ]
    ) + "\n"


def rewrite_catalogue(
    source: Path,
    output: Path,
    scaling: SZScaling,
    *,
    force: bool = False,
    chunksize: int = CHUNKSIZE,
) -> CatalogueSummary:
    """Stream ``source`` and atomically write it with recomputed ``q_from_mz``."""
    source = Path(source)
    output = Path(output)
    if output.exists() and not force:
        raise FileExistsError(f"output already exists: {output}")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    source_before = source.stat()
    temporary = output.with_name(
        f"{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    rows = 0
    q_min = np.inf
    q_max = -np.inf

    try:
        with temporary.open("x") as handle:
            handle.write(_provenance(source, scaling))

        first_chunk = True
        for chunk in pd.read_csv(
            source,
            comment="#",
            chunksize=chunksize,
            float_precision="round_trip",
        ):
            missing = REQUIRED_COLUMNS.difference(chunk.columns)
            if missing:
                raise ValueError(
                    f"{source} is missing required columns: {sorted(missing)}"
                )

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
                raise ValueError("q_from_mz contains non-finite or non-positive values")

            chunk["q_from_mz"] = q
            chunk.to_csv(
                temporary,
                mode="a",
                header=first_chunk,
                index=False,
                float_format="%.17g",
            )
            first_chunk = False
            rows += len(chunk)
            q_min = min(q_min, float(q.min()))
            q_max = max(q_max, float(q.max()))
            print(f"    {source.name}: {rows:,} rows", flush=True)

        if first_chunk:
            raise ValueError(f"catalogue contains no data rows: {source}")

        source_after = source.stat()
        if (
            source_after.st_size != source_before.st_size
            or source_after.st_mtime_ns != source_before.st_mtime_ns
        ):
            raise RuntimeError(f"source changed while it was being read: {source}")

        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return CatalogueSummary(
        source=source,
        output=output,
        rows=rows,
        q_min=q_min,
        q_max=q_max,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        choices=config.VARIANTS,
        help="feedback variant to process; repeat to select several",
    )
    parser.add_argument("--force", action="store_true", help="replace best-fit outputs")
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    parser.add_argument("--catalogue-dir", type=Path, default=config.CAT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    variants = args.variant or config.VARIANTS
    scaling = bestfit_scaling()
    print(
        f"Best-fit scaling: A_SZ={scaling.A_SZ:.8g} "
        f"alpha_SZ={scaling.alpha_SZ:.8g} B={scaling.B:.8g} "
        f"sigma_lnY={scaling.sigma_lnY:.8g} seed={scaling.seed}",
        flush=True,
    )

    for variant in variants:
        source = catalogue_path(args.catalogue_dir, variant)
        output = bestfit_path(source)
        started = time.time()
        summary = rewrite_catalogue(
            source,
            output,
            scaling,
            force=args.force,
            chunksize=args.chunksize,
        )
        print(
            f"  {variant}: {summary.rows:,} rows -> {summary.output} "
            f"q=[{summary.q_min:.8g}, {summary.q_max:.8g}] "
            f"({time.time() - started:.1f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
