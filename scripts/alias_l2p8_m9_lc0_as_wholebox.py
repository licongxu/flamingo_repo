"""Publish the L2p8_m9 lightcone-0 masked bandpowers under the whole-box name.

``compute_l2p8_m9_masked_ps_alpha_fixed_1p12.py`` is per-lightcone and writes
``Dl_yy_L2p8_m9_lc<N>_masked_<tag>_...``. The paper figures
(``plot_l2p8_m9_masked_ps_alpha_fixed_1p12``,
``plot_l1_l2p8_m9_masked_ps_comparison``) and ``compute_masked_ps_qgt6`` all use
the un-indexed ``Dl_yy_L2p8_m9_masked_<tag>_...`` name and mean lightcone 0 by it
-- the L2p8 panels are labelled "(lc0)".

This script makes that convention explicit instead of leaving the whole-box
files as untracked leftovers from an earlier run. It hard-links lc0's products
to the whole-box name and merges lc0's ``cuts`` into the whole-box metadata,
preserving the ``qgt6`` entry that ``compute_masked_ps_qgt6`` writes there.

Run::

    python scripts/alias_l2p8_m9_lc0_as_wholebox.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data_paper" / "binned_bandpowers"
TAG = "qfrommz_alpha_fixed_1p12"
CUT_TAGS = ["qgt50", "qgt20", "qgt10", "qgt5", "qgt1"]
SUFFIXES = ["binned_18", "logbins_dln0p4_lmax10000"]


def main() -> None:
    linked = []
    for tag in CUT_TAGS:
        for suffix in SUFFIXES:
            src = DATA / f"Dl_yy_L2p8_m9_lc0_masked_{tag}_{TAG}_{suffix}.txt"
            dst = DATA / f"Dl_yy_L2p8_m9_masked_{tag}_{TAG}_{suffix}.txt"
            if not src.is_file():
                raise SystemExit(f"missing lc0 product: {src.relative_to(REPO)}")
            dst.unlink(missing_ok=True)
            dst.hardlink_to(src)
            linked.append(dst.name)

    lc0_meta = json.loads((DATA / f"L2p8_m9_lc0_masked_{TAG}_metadata.json").read_text())
    box_path = DATA / f"L2p8_m9_masked_{TAG}_metadata.json"
    box_meta = json.loads(box_path.read_text()) if box_path.is_file() else {}

    # Keep whatever is already there (qgt6, written by compute_masked_ps_qgt6)
    # and fold in lc0's five cuts.
    cuts = dict(box_meta.get("cuts", {}))
    cuts.update({t: lc0_meta["cuts"][t] for t in CUT_TAGS})
    box_meta.update(
        {
            "map": lc0_meta["map"],
            "catalogue": lc0_meta["catalogue"],
            "masking": lc0_meta["masking"],
            "cuts": cuts,
            "aliased_from": f"L2p8_m9_lc0_masked_{TAG}_metadata.json",
            "note": "whole-box L2p8_m9 products are lightcone 0; hard-linked by "
            "scripts/alias_l2p8_m9_lc0_as_wholebox.py",
        }
    )
    box_path.write_text(json.dumps(box_meta, indent=2, sort_keys=True))

    print(f"hard-linked {len(linked)} files to the whole-box name")
    print(f"wrote {box_path.relative_to(REPO)} with cuts: {sorted(cuts)}")


if __name__ == "__main__":
    main()
