"""Load and query the generated ``index.json`` manifest.

The manifest is produced by ``scripts/build_index.py`` and is the single
source of truth for what the portal serves; API requests never walk the
data tree directly.
"""

import json
from pathlib import Path
from typing import Any


class Manifest:
    """In-memory view of ``index.json`` with mtime-based auto-reload."""

    def __init__(self, index_path: Path):
        self.index_path = Path(index_path)
        self._loaded_mtime: float | None = None
        self._data: dict[str, Any] | None = None
        self._by_id: dict[str, dict[str, Any]] = {}

    @property
    def available(self) -> bool:
        return self.index_path.is_file()

    def _load_if_needed(self) -> None:
        if not self.available:
            self._data = None
            self._by_id = {}
            return
        mtime = self.index_path.stat().st_mtime
        if self._data is not None and mtime == self._loaded_mtime:
            return
        with open(self.index_path) as fh:
            self._data = json.load(fh)
        self._by_id = {f["id"]: f for f in self._data["files"]}
        self._loaded_mtime = mtime

    def raw(self) -> dict[str, Any] | None:
        self._load_if_needed()
        return self._data

    def files(
        self,
        kind: str | None = None,
        simulation: str | None = None,
        variant: str | None = None,
        lightcone: int | None = None,
    ) -> list[dict[str, Any]]:
        self._load_if_needed()
        if self._data is None:
            return []
        out = self._data["files"]
        if kind is not None:
            out = [f for f in out if f["kind"] == kind]
        if simulation is not None:
            out = [f for f in out if f["simulation"] == simulation]
        if variant is not None:
            out = [f for f in out if f["variant"] == variant]
        if lightcone is not None:
            out = [f for f in out if f["lightcone"] == lightcone]
        return out

    def get(self, file_id: str) -> dict[str, Any] | None:
        self._load_if_needed()
        return self._by_id.get(file_id)

    def summary(self) -> dict[str, Any]:
        self._load_if_needed()
        if self._data is None:
            return {}
        files = self._data["files"]
        by_kind: dict[str, dict[str, Any]] = {}
        for f in files:
            entry = by_kind.setdefault(f["kind"], {"count": 0, "size_bytes": 0})
            entry["count"] += 1
            entry["size_bytes"] += f["size_bytes"]
        return {
            "generated_utc": self._data.get("generated_utc"),
            "data_root": self._data.get("data_root"),
            "n_files": len(files),
            "total_size_bytes": sum(f["size_bytes"] for f in files),
            "by_kind": by_kind,
            "simulations": sorted({f["simulation"] for f in files}),
            "variants": sorted({f["variant"] for f in files}),
        }

    def tree(self) -> list[dict[str, Any]]:
        """Simulation -> variant/lightcone hierarchy for the browse page."""
        self._load_if_needed()
        if self._data is None:
            return []
        sims: dict[str, dict[str, Any]] = {}
        for f in self._data["files"]:
            sim = sims.setdefault(
                f["simulation"],
                {"simulation": f["simulation"], "variants": {}, "lightcones": set()},
            )
            var = sim["variants"].setdefault(
                f["variant"], {"variant": f["variant"], "count": 0, "size_bytes": 0}
            )
            var["count"] += 1
            var["size_bytes"] += f["size_bytes"]
            if f["lightcone"] is not None:
                sim["lightcones"].add(f["lightcone"])
        return [
            {
                "simulation": s["simulation"],
                "variants": sorted(s["variants"].values(), key=lambda v: v["variant"]),
                "lightcones": sorted(s["lightcones"]),
            }
            for s in sorted(sims.values(), key=lambda s: s["simulation"])
        ]
