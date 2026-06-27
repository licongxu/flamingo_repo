"""Return 0 if a resumed catalogue build has finished all snapshots in range."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flamingo_snapshots import snap_range_zle


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--parent", default=None)
    p.add_argument("--variant", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--z-max", type=float, default=3.0)
    p.add_argument("--z-min", type=float, default=0.0)
    args = p.parse_args()

    progress_path = args.out.with_suffix(args.out.suffix + ".progress.json")
    if not progress_path.exists():
        return 1

    start, stop = snap_range_zle(
        args.parent, args.variant, z_max=args.z_max, z_min=args.z_min
    )
    done = set(json.loads(progress_path.read_text()).get("completed_snapshots", []))
    needed = set(range(start, stop + 1))
    return 0 if needed <= done else 1


if __name__ == "__main__":
    sys.exit(main())
