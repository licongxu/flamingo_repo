"""Render Mollweide preview PNGs for the HEALPix maps in the manifest.

Usage::

    python scripts/make_previews.py [--index data/index.json] [--out data/previews]
                                    [--only SUBSTRING] [--max-size-gb 5] [--force]

Each map is degraded to nside 512 and plotted as log10(y) on a fixed
colour scale, saved as ``<file-id>.png``. Maps above ``--max-size-gb``
(the 25 GB nside16384 shell) are skipped. Existing PNGs are skipped
unless ``--force``; run in batches to stay under interactive time limits.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import healpy as hp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

WEBSITE_DIR = Path(__file__).resolve().parent.parent
PREVIEW_NSIDE = 512


def render(map_path: Path, out_png: Path, title: str) -> None:
    m = hp.read_map(map_path, dtype=np.float64)
    if hp.get_nside(m) > PREVIEW_NSIDE:
        m = hp.ud_grade(m, PREVIEW_NSIDE)
    logy = np.log10(np.clip(m, 1e-12, None))
    # Per-map percentile scale: keeps contrast comparable across feedback
    # variants whose overall y amplitude differs.
    vmin, vmax = np.round(np.percentile(logy, [1.0, 99.9]), 2)
    hp.mollview(
        logy,
        title=title,
        unit=r"$\log_{10}\,y$",
        min=vmin,
        max=vmax,
        cmap="viridis",
        xsize=1200,
    )
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close("all")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=WEBSITE_DIR / "data" / "index.json")
    ap.add_argument("--out", type=Path, default=WEBSITE_DIR / "data" / "previews")
    ap.add_argument("--only", default="", help="only maps whose relpath contains this")
    ap.add_argument("--max-size-gb", type=float, default=5.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    with open(args.index) as fh:
        index = json.load(fh)
    data_root = Path(index["data_root"])
    args.out.mkdir(parents=True, exist_ok=True)

    maps = [
        f
        for f in index["files"]
        if f["kind"] == "map"
        and args.only in f["relpath"]
        and f["size_bytes"] <= args.max_size_gb * 1e9
    ]
    n_done = 0
    for f in maps:
        out_png = args.out / f"{f['id']}.png"
        if out_png.exists() and not args.force:
            continue
        src = data_root / f["relpath"]
        if not src.is_file():
            print(f"skip (missing): {src}", file=sys.stderr)
            continue
        print(f"rendering {f['relpath']} -> {out_png.name}")
        render(src, out_png, f["filename"])
        n_done += 1
    print(f"done: {n_done} rendered, {len(maps) - n_done} skipped/cached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
