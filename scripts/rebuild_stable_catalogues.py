#!/usr/bin/env python3
"""Stage, validate, archive, and publish identity-safe FLAMINGO catalogues."""
from __future__ import annotations

import argparse
from dataclasses import replace
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
)


DEFAULT_ROOT = Path("/rds/rds-lxu/flamingo")
DEFAULT_STAGE = DEFAULT_ROOT / ".hbt_join_fix_staging/20260731"


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


def _snapshot_numbers(target) -> tuple[int, ...]:
    return tuple(range(17, 78 if target.family == "l1" else 79))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--only", action="append", default=[])
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inventory")
    subparsers.add_parser("build-base")
    subparsers.add_parser("derive-q")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    targets = _selected_targets(args.root, tuple(args.only))
    if args.command == "inventory":
        print(f"{len(targets)} target(s), {4 * len(targets)} canonical CSV(s)")
        for target in targets:
            print(f"{target.key}: {target.catalogue_dir}")
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

    source = HdfstreamSnapshotSource()
    for index, target in enumerate(targets, start=1):
        started = time.monotonic()
        staged = _stage_base_path(target, args.root, args.stage)
        radii, angles = source.shell_geometry(target)
        summary = build_base_catalogue(
            target,
            staged,
            source,
            snaps=_snapshot_numbers(target),
            shell_radii_mpc=radii,
            angles=angles,
        )
        print(
            f"[{index}/{len(targets)}] {target.key}: {summary['rows']:,} rows "
            f"in {time.monotonic() - started:.1f}s -> {staged}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
