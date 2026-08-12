"""Paper figures and data products for the FLAMINGO masked-tSZ analysis.

Every module here is a thin driver: the physics (SZ scaling relation, disc
masks, pseudo-Cl estimation) lives in the ``flamingo`` package under ``src/``,
and these modules only choose inputs, loop over them, and save outputs.

The ``src/`` tree of *this checkout* is put at the front of ``sys.path`` so the
pipeline always runs against the source code committed on this branch, never
against whatever ``flamingo`` happens to be pip-installed in the environment.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
