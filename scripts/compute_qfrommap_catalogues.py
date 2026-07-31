#!/usr/bin/env python3
"""Compute raw-map aperture SNRs for canonical FLAMINGO catalogues."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
import time
import uuid

import healpy as hp
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from flamingo.aperture_snr import (  # noqa: E402
    catalogue_chunk_to_qfrommap,
    fit_sigma_y500,
)


FLAMINGO_ROOT = Path("/rds/rds-lxu/flamingo")
NOISE_FILE = REPO / "data/noise/sigma_Y500_dict_szifi.npy"
SKYFRACS_FILE = REPO / "data/noise/skyfracs_szifi_cosmology.npy"
L1_PREFIX = "halo_catalogue_M500c_5e13_zlt3_"
L1_SUFFIX = "_yang26rot_qfrommz.csv"
L2_SOURCE_NAME = (
    "halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_qfrommz.csv"
)


@dataclass(frozen=True)
class CatalogueJob:
    """One canonical catalogue, its matching y map, and derived output."""

    dataset: str
    label: str
    source: Path
    map_path: Path
    output: Path


def _output_path(source: Path) -> Path:
    return source.with_name(source.name.removesuffix("_qfrommz.csv") + "_qfrommap.csv")


def _discover_l1(root: Path) -> list[CatalogueJob]:
    catalogue_dir = root / "L1_m9/catalogues"
    jobs = []
    for source in catalogue_dir.glob(f"{L1_PREFIX}*{L1_SUFFIX}"):
        variant = source.name.removeprefix(L1_PREFIX).removesuffix(L1_SUFFIX)
        map_path = root / "L1_m9/maps" / f"y_unlensed_{variant}_lc0_nside4096.fits"
        jobs.append(CatalogueJob("l1", variant, source, map_path, _output_path(source)))
    return sorted(jobs, key=lambda job: job.label)


def _discover_l2(root: Path) -> list[CatalogueJob]:
    jobs = []
    pattern = root / "L2p8_m9"
    for source in pattern.glob(f"lightcone*/catalogues/{L2_SOURCE_NAME}"):
        lightcone = source.parents[1].name
        match = re.fullmatch(r"lightcone(\d+)", lightcone)
        if match is None:
            raise ValueError(f"malformed lightcone directory: {lightcone}")
        index = int(match.group(1))
        map_path = source.parents[1] / "healpix_map" / f"y_unlensed_L2p8_m9_lc{index}.fits"
        jobs.append(CatalogueJob("l2", lightcone, source, map_path, _output_path(source)))
    return sorted(jobs, key=lambda job: int(job.label.removeprefix("lightcone")))


def discover_jobs(
    root: Path = FLAMINGO_ROOT,
    *,
    dataset: str = "all",
    only: tuple[str, ...] = (),
) -> list[CatalogueJob]:
    """Discover canonical 5e13 q-from-mass/redshift catalogues and paired maps."""
    if dataset not in {"all", "l1", "l2"}:
        raise ValueError(f"unknown dataset: {dataset}")

    jobs = []
    if dataset in {"all", "l1"}:
        jobs.extend(_discover_l1(root))
    if dataset in {"all", "l2"}:
        jobs.extend(_discover_l2(root))

    if only:
        jobs = [
            job
            for job in jobs
            if any(term in job.label or term in str(job.source) for term in only)
        ]
    if not jobs:
        raise ValueError("no canonical catalogues matched the requested selection")

    for job in jobs:
        if not job.map_path.is_file():
            raise FileNotFoundError(
                f"missing y map for {job.source.name}: {job.map_path}"
            )
    return jobs


def read_header(path: Path) -> list[str]:
    """Return the leading comment lines without trailing newlines."""
    lines = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            lines.append(line.rstrip("\n"))
    return lines


def provenance(job: CatalogueJob, previous: list[str]) -> str:
    """Build a self-contained provenance header for a q-from-map output."""
    return (
        "\n".join(
            previous
            + [
                "# q_from_aperture from raw cylindrical HEALPix aperture sum; "
                "no background subtraction",
                f"# source_catalogue={job.source}",
                f"# source_y_map={job.map_path}",
                "# theta_500=R_500c/D_A(D3A); noise=SZiFi-immf6 "
                "sky-fraction average cubic log-log fit",
            ]
        )
        + "\n"
    )


def rewrite_catalogue(
    job: CatalogueJob,
    noise_coeff: np.ndarray,
    *,
    chunk_size: int = 100_000,
    force: bool = False,
    output: Path | None = None,
) -> dict[str, object]:
    """Stream one source catalogue and atomically install its q-from-map output."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    output_path = job.output if output is None else Path(output)
    if output_path.exists() and not force:
        raise FileExistsError(f"output already exists: {output_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_before = job.source.stat()
    ymap = np.asarray(hp.read_map(job.map_path, dtype=np.float32), dtype=np.float32)
    header = read_header(job.source)
    temporary = output_path.with_name(
        f"{output_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    thresholds = (1, 5, 10, 20, 50)
    q_counts = {threshold: 0 for threshold in thresholds}
    rows = 0
    zero_pixels = 0
    first = True
    started = time.monotonic()

    try:
        with temporary.open("x") as handle:
            handle.write(provenance(job, header))

        for chunk in pd.read_csv(
            job.source,
            comment="#",
            chunksize=chunk_size,
            float_precision="round_trip",
        ):
            output = catalogue_chunk_to_qfrommap(chunk, ymap, noise_coeff)
            output.to_csv(
                temporary,
                mode="a",
                header=first,
                index=False,
                float_format="%.17g",
            )
            first = False
            rows += len(output)
            zero_pixels += int((output["npix_in_aperture"] == 0).sum())
            q = output["q_from_aperture"].to_numpy(np.float64)
            for threshold in thresholds:
                q_counts[threshold] += int(np.count_nonzero(q > threshold))
            print(
                f"    {job.label}: {rows:,} rows ({time.monotonic() - started:.0f}s)",
                flush=True,
            )

        if first:
            raise ValueError(f"{job.source.name}: no data rows")
        source_after = job.source.stat()
        if (
            source_after.st_size != source_before.st_size
            or source_after.st_mtime_ns != source_before.st_mtime_ns
        ):
            raise RuntimeError(f"{job.source.name} changed while being read")
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "dataset": job.dataset,
        "label": job.label,
        "output": str(output_path),
        "rows": rows,
        "zero_pixels": zero_pixels,
        "q_counts": q_counts,
        "seconds": time.monotonic() - started,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flamingo-root", type=Path, default=FLAMINGO_ROOT)
    parser.add_argument("--noise-file", type=Path, default=NOISE_FILE)
    parser.add_argument("--skyfracs-file", type=Path, default=SKYFRACS_FILE)
    parser.add_argument("--dataset", choices=("all", "l1", "l2"), default="all")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="retain jobs whose label or source path contains this value (repeatable)",
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    """Run the validated production manifest."""
    args = _parser().parse_args()
    jobs = discover_jobs(
        args.flamingo_root,
        dataset=args.dataset,
        only=tuple(args.only),
    )
    if args.dry_run:
        for job in jobs:
            print(f"{job.dataset} {job.label}: {job.source} | {job.map_path} -> {job.output}")
        print(f"jobs={len(jobs)}")
        return 0

    coeff = fit_sigma_y500(args.noise_file, args.skyfracs_file)
    for job in jobs:
        summary = rewrite_catalogue(
            job,
            coeff,
            chunk_size=args.chunk_size,
            force=args.force,
        )
        counts = " ".join(
            f"q>{threshold}:{summary['q_counts'][threshold]:,}"
            for threshold in (1, 5, 10, 20, 50)
        )
        print(
            f"DONE {summary['label']}: rows={summary['rows']:,} "
            f"zero_pixels={summary['zero_pixels']:,} {counts} "
            f"seconds={summary['seconds']:.1f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
