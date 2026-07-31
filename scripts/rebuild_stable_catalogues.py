#!/usr/bin/env python3
"""Stage, validate, archive, and publish identity-safe FLAMINGO catalogues."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import sys
import time


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from flamingo.catalogue.portal import HdfstreamSnapshotSource  # noqa: E402
from flamingo.catalogue.rebuild import (  # noqa: E402
    build_base_catalogue,
    catalogue_targets,
    derive_q_catalogues,
    publish_file,
    validate_catalogue,
)


DEFAULT_ROOT = Path("/rds/rds-lxu/flamingo")
DEFAULT_STAGE = DEFAULT_ROOT / ".hbt_join_fix_staging/20260731"
ARCHIVE_NAME = "hbt_join_fix_20260731"
FLAVOURS = ("base", "q", "q_alpha", "qmap")


def _load_qmap_module():
    path = REPO / "scripts/compute_qfrommap_catalogues.py"
    spec = importlib.util.spec_from_file_location("stable_qfrommap", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load q-from-map producer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _selected_targets(root: Path, only: tuple[str, ...]):
    targets = catalogue_targets(root)
    if only:
        targets = tuple(
            target
            for target in targets
            if any(term in target.key for term in only)
        )
    if not targets:
        raise ValueError("no catalogue target matched --only")
    return targets


def _stage_base_path(target, root: Path, stage: Path) -> Path:
    return stage / target.canonical_csvs[0].relative_to(root)


def _stage_path(canonical: Path, root: Path, stage: Path) -> Path:
    return stage / canonical.relative_to(root)


def _snapshot_numbers(target, requested: tuple[int, ...] = ()) -> tuple[int, ...]:
    available = tuple(range(18, 78 if target.family == "l1" else 79))
    if not requested:
        return available
    invalid = sorted(set(requested).difference(available))
    if invalid:
        raise ValueError(f"snapshot(s) outside {target.family} range: {invalid}")
    return tuple(sorted(set(requested)))


def _build_one_base(target, root: Path, stage: Path, requested_snaps):
    source = HdfstreamSnapshotSource()
    radii, angles = source.shell_geometry(target)
    staged = _stage_base_path(target, root, stage)
    return build_base_catalogue(
        target,
        staged,
        source,
        snaps=_snapshot_numbers(target, requested_snaps),
        shell_radii_mpc=radii,
        angles=angles,
    )


def _validate_target(paths, target):
    summaries = [
        validate_catalogue(path, target, flavour)
        for path, flavour in zip(paths, FLAVOURS, strict=True)
    ]
    base, q, q_alpha, qmap = summaries
    expected_rows = base["selected_rows"]
    expected_digest = base["selected_identity_sha256"]
    for summary in (q, q_alpha, qmap):
        if summary["rows"] != expected_rows:
            raise ValueError(f"{target.key}: derived/base selected row count mismatch")
        if summary["identity_sha256"] != expected_digest:
            raise ValueError(f"{target.key}: derived/base identity digest mismatch")
    return summaries


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--snap", type=int, action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inventory")
    subparsers.add_parser("build-base")
    subparsers.add_parser("derive-q")
    subparsers.add_parser("derive-qmap")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--canonical", action="store_true")
    subparsers.add_parser("publish")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    targets = _selected_targets(args.root, tuple(args.only))
    if args.command == "inventory":
        print(f"{len(targets)} target(s), {4 * len(targets)} canonical CSV(s)")
        for target in targets:
            print(f"{target.key}: {target.catalogue_dir}")
        return 0

    if args.command in {"validate", "publish"}:
        all_summaries = []
        for target in targets:
            paths = (
                target.canonical_csvs
                if getattr(args, "canonical", False)
                else tuple(
                    _stage_path(path, args.root, args.stage)
                    for path in target.canonical_csvs
                )
            )
            summaries = _validate_target(paths, target)
            all_summaries.extend(summaries)
            print(
                f"VALID {target.key}: base={summaries[0]['rows']:,} "
                f"selected={summaries[0]['selected_rows']:,}",
                flush=True,
            )
        if args.command == "validate":
            print(f"VALIDATED {len(all_summaries)}/{4 * len(targets)} files")
            return 0

        archive_root = args.archive or args.root / "archive" / ARCHIVE_NAME
        manifest = archive_root / "manifest.json"
        for target in targets:
            for canonical in target.canonical_csvs:
                staged = _stage_path(canonical, args.root, args.stage)
                archived = archive_root / canonical.relative_to(args.root)
                publish_file(staged, canonical, archived, manifest)
                print(f"PUBLISHED {canonical} (archive={archived})", flush=True)
        return 0

    if args.command == "derive-q":
        from flamingo.cnc import SZScaling

        scaling = replace(
            SZScaling.calibrated(B=1.41),
            A_SZ=-4.0953238,
            alpha_SZ=1.12,
            B=1.41,
        )
        for index, target in enumerate(targets, start=1):
            base = _stage_base_path(target, args.root, args.stage)
            outputs = tuple(
                _stage_path(path, args.root, args.stage)
                for path in target.canonical_csvs[1:3]
            )
            summary = derive_q_catalogues(
                base, outputs, target.family, scaling
            )
            print(
                f"[{index}/{len(targets)}] {target.key}: "
                f"{summary['rows']:,} q rows",
                flush=True,
            )
        return 0

    if args.command == "derive-qmap":
        qmap = _load_qmap_module()
        data_root = Path(os.environ.get("FLAMINGO_ROOT", REPO))
        noise_coeff = qmap.fit_sigma_y500(
            data_root / "data/noise/sigma_Y500_dict_szifi.npy",
            data_root / "data/noise/skyfracs_szifi_cosmology.npy",
        )
        for index, target in enumerate(targets, start=1):
            source_path = _stage_path(
                target.canonical_csvs[1], args.root, args.stage
            )
            staged_output = _stage_path(
                target.canonical_csvs[3], args.root, args.stage
            )
            job = qmap.CatalogueJob(
                target.family,
                target.key,
                source_path,
                target.map_path,
                target.canonical_csvs[3],
            )
            summary = qmap.rewrite_catalogue(
                job,
                noise_coeff,
                force=True,
                output=staged_output,
            )
            print(
                f"[{index}/{len(targets)}] {target.key}: "
                f"{summary['rows']:,} aperture rows q>5="
                f"{summary['q_counts'][5]:,}",
                flush=True,
            )
        return 0

    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    started = time.monotonic()
    if args.workers == 1:
        completed = [
            (target, _build_one_base(target, args.root, args.stage, tuple(args.snap)))
            for target in targets
        ]
    else:
        completed = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _build_one_base,
                    target,
                    args.root,
                    args.stage,
                    tuple(args.snap),
                ): target
                for target in targets
            }
            for future in as_completed(futures):
                completed.append((futures[future], future.result()))
    for index, (target, summary) in enumerate(completed, start=1):
        print(
            f"[{index}/{len(targets)}] {target.key}: {summary['rows']:,} rows "
            f"-> {summary['path']}",
            flush=True,
        )
    print(f"base build elapsed={time.monotonic() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
