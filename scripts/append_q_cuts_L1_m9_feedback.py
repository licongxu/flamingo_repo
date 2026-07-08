"""Append q>3 and q>1 masked tSZ bandpowers to the existing L1_m9 feedback npz.

Runs the *same* pipeline as ``compute_masked_tsz_ps_L1_m9_feedback.py`` (imported
and reused, not duplicated), but only for the two new cuts, then concatenates the
new cut columns onto ``masked_tsz_ps.npz`` (final cut order 50, 20, 10, 5, 3, 1).
The four existing cuts are left untouched.

Run:
    python scripts/append_q_cuts_L1_m9_feedback.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import compute_masked_tsz_ps_L1_m9_feedback as M  # noqa: E402

NEW_CUTS = [3.0, 1.0]
NEW_TAGS = ["qgt3", "qgt1"]


def main() -> None:
    if not M.OUT_NPZ.exists():
        raise FileNotFoundError(f"missing {M.OUT_NPZ}; run the 4-cut pipeline first")

    old = np.load(M.OUT_NPZ, allow_pickle=True)
    old_cuts = list(old["q_cuts"])
    for c in NEW_CUTS:
        if c in old_cuts:
            raise ValueError(f"q>{c:g} already present in {M.OUT_NPZ.name}")
    variants = [str(v) for v in old["variants"]]
    ellb_ref = old["ellb"]

    # restrict the reused pipeline to the two new cuts
    M.Q_CUTS = NEW_CUTS
    M.CUT_TAGS = NEW_TAGS
    cosmo = M.Cosmology(**M.D3A)

    dl_new: list[np.ndarray] = []
    n_new: list[np.ndarray] = []
    fsky_new: list[np.ndarray] = []
    for variant in variants:
        row = M.process_variant(variant, cosmo)
        if not np.allclose(row["ellb"], ellb_ref):
            raise ValueError(f"ell bin mismatch for {variant}")
        dl_new.append(row["dl_cuts"])          # (2, nbin)
        n_new.append(row["n_detected"])        # (2,)
        fsky_new.append(row["fsky_binary"])    # (2,)

    dl_new = np.stack(dl_new, axis=0)          # (nvar, 2, nbin)
    n_new = np.stack(n_new, axis=0)            # (nvar, 2)
    fsky_new = np.stack(fsky_new, axis=0)      # (nvar, 2)

    q_cuts = np.concatenate([old["q_cuts"], np.array(NEW_CUTS)])
    cut_tags = np.concatenate([old["cut_tags"], np.array(NEW_TAGS)])
    dl_masked = np.concatenate([old["dl_masked"], dl_new], axis=1)
    n_detected = np.concatenate([old["n_detected"], n_new], axis=1)
    fsky_binary = np.concatenate([old["fsky_binary"], fsky_new], axis=1)

    np.savez(
        M.OUT_NPZ,
        variants=np.array(variants),
        ellb=ellb_ref,
        q_cuts=q_cuts,
        cut_tags=cut_tags,
        dl_fullsky=old["dl_fullsky"],
        dl_masked=dl_masked,
        n_detected=n_detected,
        fsky_binary=fsky_binary,
        lmax=M.LMAX,
        delta_ell=M.DELL,
        r_mask=M.R_MASK,
        apod_deg=M.APOD_DEG,
        map_dir=str(M.MAP_DIR),
        cat_dir=str(M.CAT_DIR),
    )
    print(f"\nwrote {M.OUT_NPZ}  dl_masked shape={dl_masked.shape}  cuts={list(q_cuts)}")
    print("\n=== N(q>cut) per variant (appended cuts) ===")
    for iv, variant in enumerate(variants):
        parts = "  ".join(f"q>{int(q)}={n_new[iv, ic]:5d}" for ic, q in enumerate(NEW_CUTS))
        print(f"  {variant:28s}  {parts}")


if __name__ == "__main__":
    main()
