"""DuckDB-backed query engine for the halo catalogue files.

Queries run against the CSV directly (DuckDB handles the ``#`` comment
header lines); if a ``.parquet`` sibling exists (see
``scripts/convert_parquet.py``) it is preferred, which makes the 9.9 GB
L2p8 catalogues interactive.

Safety: file paths are always bound as query parameters, column
identifiers are validated against the file header and double-quoted,
and every call runs on its own cursor of a shared in-memory DuckDB
connection (a raw connection is not thread-safe under FastAPI's
threadpool; cursors are, and avoid spawning a thread pool per request).
"""

import functools
import math
import threading
from pathlib import Path
from typing import Any

import duckdb

MAX_LIMIT = 100_000
DUCKDB_THREADS = 4  # bounded: this often runs on a shared login node

_conn_lock = threading.Lock()
_conn: duckdb.DuckDBPyConnection | None = None


def _cursor() -> duckdb.DuckDBPyConnection:
    global _conn
    with _conn_lock:
        if _conn is None:
            _conn = duckdb.connect()
            _conn.execute(f"SET threads = {DUCKDB_THREADS}")
        return _conn.cursor()

# Numeric columns exposed as range filters: query param stem -> column.
FILTERABLE = {
    "M500c": "M_500c_Msun",
    "z": "z",
    "q": "q_from_mz",
}


class CatalogueError(ValueError):
    """Raised for invalid query parameters (mapped to HTTP 422)."""


def _source_path(csv_path: Path) -> Path:
    parquet = csv_path.with_suffix(".parquet")
    return parquet if parquet.is_file() else csv_path


def _relation(path: Path) -> str:
    """FROM-clause fragment; the path itself is bound as a parameter."""
    if path.suffix == ".parquet":
        return "read_parquet(?)"
    return "read_csv(?, header=true, comment='#')"


def _quote(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _clean(value: Any) -> Any:
    """NaN/Inf are not valid JSON; map them to null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def columns(csv_path: Path) -> list[str]:
    src = _source_path(csv_path)
    return list(_columns_cached(str(src), src.stat().st_mtime))


@functools.lru_cache(maxsize=64)
def _columns_cached(src: str, mtime: float) -> tuple[str, ...]:
    cur = _cursor().execute(f"SELECT * FROM {_relation(Path(src))} LIMIT 0", [src])
    return tuple(d[0] for d in cur.description)


def stats(csv_path: Path) -> dict[str, Any]:
    """Row count and filterable-column ranges, cached per (file, mtime)."""
    src = _source_path(csv_path)
    return dict(_stats_cached(str(src), src.stat().st_mtime))


@functools.lru_cache(maxsize=64)
def _stats_cached(src: str, mtime: float) -> dict[str, Any]:
    cols = _columns_cached(src, mtime)
    present = {stem: col for stem, col in FILTERABLE.items() if col in cols}
    aggs = ", ".join(
        f"min({_quote(c)}) AS min_{s}, max({_quote(c)}) AS max_{s}" for s, c in present.items()
    )
    sql = f"SELECT count(*) AS n{', ' + aggs if aggs else ''} FROM {_relation(Path(src))}"
    row = _cursor().execute(sql, [src]).fetchone()
    names = ["n"] + [f"{m}_{s}" for s in present for m in ("min", "max")]
    out: dict[str, Any] = dict(zip(names, row))
    out["columns"] = list(cols)
    out["source"] = src
    return out


def query(
    csv_path: Path,
    select: list[str] | None = None,
    ranges: dict[str, tuple[float | None, float | None]] | None = None,
    sort: str | None = None,
    descending: bool = False,
    limit: int = 1000,
    offset: int = 0,
) -> dict[str, Any]:
    """Run a filtered query and return column-oriented JSON-ready data.

    Parameters
    ----------
    csv_path : Path
        Catalogue CSV path from the manifest (parquet sibling preferred).
    select : list of str, optional
        Columns to return; default all.
    ranges : dict, optional
        ``{param_stem: (min, max)}`` filters on `FILTERABLE` columns.
    sort : str, optional
        Column to order by.
    limit, offset : int
        Pagination; ``limit`` is capped at ``MAX_LIMIT``.
    """
    src = _source_path(csv_path)
    cols = columns(csv_path)

    if select:
        bad = [c for c in select if c not in cols]
        if bad:
            raise CatalogueError(f"unknown columns: {bad}")
    sel = ", ".join(_quote(c) for c in select) if select else "*"

    where: list[str] = []
    params: list[Any] = [str(src)]
    for stem, (lo, hi) in (ranges or {}).items():
        col = FILTERABLE.get(stem)
        if col is None or col not in cols:
            if lo is not None or hi is not None:
                raise CatalogueError(f"column for filter '{stem}' not in this catalogue")
            continue
        if lo is not None:
            where.append(f"{_quote(col)} >= ?")
            params.append(lo)
        if hi is not None:
            where.append(f"{_quote(col)} <= ?")
            params.append(hi)

    if sort is not None and sort not in cols:
        raise CatalogueError(f"unknown sort column: {sort}")
    if not 0 < limit <= MAX_LIMIT:
        raise CatalogueError(f"limit must be in (0, {MAX_LIMIT}]")
    if offset < 0:
        raise CatalogueError("offset must be >= 0")

    sql = f"SELECT {sel} FROM {_relation(src)}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if sort:
        sql += f" ORDER BY {_quote(sort)} {'DESC' if descending else 'ASC'}"
    sql += f" LIMIT {limit} OFFSET {offset}"

    cur = _cursor().execute(sql, params)
    out_cols = [d[0] for d in cur.description]
    rows = [[_clean(v) for v in row] for row in cur.fetchall()]
    return {"columns": out_cols, "rows": rows, "n_rows": len(rows)}
