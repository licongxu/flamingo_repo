"""Catalogue selection modes for FLAMINGO masked power spectra."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


FLAMINGO_ROOT = Path("/rds/rds-lxu/flamingo")
CORRECTED_L1_STAGE = (
    FLAMINGO_ROOT
    / ".hbt_join_fix_staging/20260731/L1_m9/catalogues"
)


@dataclass(frozen=True)
class QSelection:
    """Input catalogue locations and q column for one masking selection."""

    tag: str
    q_column: str
    l1_catalogue_dir: Path
    l2_root: Path


def cut_tag(q_cut: float) -> str:
    """Filename tag for a q threshold, preserving the existing decimal convention."""
    if float(q_cut).is_integer():
        return f"qgt{int(q_cut)}"
    return f"qgt{str(q_cut).replace('.', 'p')}"


def cut_tags(q_cuts: list[float]) -> list[str]:
    return [cut_tag(q_cut) for q_cut in q_cuts]


def resolve_q_selection(
    name: str,
    *,
    l1_catalogue_dir: Path | None = None,
    l2_root: Path | None = None,
) -> QSelection:
    """Resolve the legacy or empirical-map q selection without side effects."""
    if name == "qfrommz_alpha_fixed_1p12":
        default_l1 = FLAMINGO_ROOT / "L1_m9/catalogues"
        q_column = "q_from_mz"
    elif name == "qfrommap":
        default_l1 = CORRECTED_L1_STAGE
        q_column = "q_from_aperture"
    else:
        raise ValueError(f"unknown q selection: {name}")
    return QSelection(
        tag=name,
        q_column=q_column,
        l1_catalogue_dir=(
            default_l1 if l1_catalogue_dir is None else Path(l1_catalogue_dir)
        ),
        l2_root=(
            FLAMINGO_ROOT / "L2p8_m9" if l2_root is None else Path(l2_root)
        ),
    )
