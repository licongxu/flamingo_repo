#!/usr/bin/env python3
"""Execute all notebooks under notebooks/, saving outputs in place.

Uses nbclient with a long timeout. Patches text.usetex -> False in the
publication style block only (Unicode labels in many notebooks break pdflatex).
All other rcParams (dpi, fonts, ticks) are kept.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO / "notebooks"
LOG_DIR = REPO / "logs" / "notebook_runs"
TIMEOUT = int(os.environ.get("NBEXEC_TIMEOUT", "7200"))  # 2 h per notebook


def notebook_order(path: Path) -> tuple:
    name = path.stem
    if name.startswith("autoresearch"):
        return (999, name)
    m = re.match(r"(\d+)([a-z]?)_", name)
    if m:
        return (int(m.group(1)), m.group(2), name)
    return (1000, name)


def patch_usetex_for_execution(source: str) -> str:
    if "Publication-quality plot defaults" not in source:
        return source
    return source.replace('"text.usetex": True', '"text.usetex": False')


def patch_cell_sources(nb) -> None:
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        text = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        patched = patch_usetex_for_execution(text)
        if patched != text:
            cell.source = patched


def execute_one(path: Path) -> tuple[bool, str]:
    from nbclient import NotebookClient
    import nbformat

    with path.open() as f:
        nb = nbformat.read(f, as_version=4)

    patch_cell_sources(nb)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("FLAMINGO_ROOT", str(REPO))

    client = NotebookClient(
        nb,
        timeout=TIMEOUT,
        kernel_name=os.environ.get("NBEXEC_KERNEL", "hmfast_py311"),
        resources={"metadata": {"path": str(path.parent.resolve())}},
    )
    client.allow_errors = False
    client.execute()

    with path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    return True, "ok"


def already_ok(name: str, summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("notebook") == name and rec.get("status") == "ok":
            return True
    return False


def main() -> int:
    try:
        import nbclient  # noqa: F401
        import nbformat  # noqa: F401
    except ImportError as exc:
        print("Missing nbclient/nbformat:", exc, file=sys.stderr)
        return 1

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"), key=notebook_order)

    summary_path = LOG_DIR / "summary.jsonl"
    only_failed = os.environ.get("NBEXEC_ONLY_FAILED", "") == "1"
    failed_names = set()
    if only_failed and summary_path.exists():
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("status") == "fail":
                    failed_names.add(rec["notebook"])

    ok, fail, skip = 0, 0, 0

    with summary_path.open("a", encoding="utf-8") as slog:
        for nb_path in notebooks:
            if only_failed:
                if nb_path.name not in failed_names:
                    continue
            elif already_ok(nb_path.name, summary_path):
                skip += 1
                print(f"SKIP {nb_path.name} (already ok)", flush=True)
                continue
            t0 = time.time()
            print(f"\n{'=' * 72}\nEXEC {nb_path.name}\n{'=' * 72}", flush=True)
            log_file = LOG_DIR / f"{nb_path.stem}.log"
            try:
                with log_file.open("w", encoding="utf-8") as lf:
                    old_stdout, old_stderr = sys.stdout, sys.stderr
                    sys.stdout = lf
                    sys.stderr = lf
                    try:
                        success, msg = execute_one(nb_path)
                    finally:
                        sys.stdout, sys.stderr = old_stdout, old_stderr
                elapsed = time.time() - t0
                record = {
                    "notebook": nb_path.name,
                    "status": "ok",
                    "seconds": round(elapsed, 1),
                    "log": str(log_file),
                }
                ok += 1
                print(f"OK {nb_path.name} ({elapsed:.0f}s)", flush=True)
            except Exception as exc:
                elapsed = time.time() - t0
                record = {
                    "notebook": nb_path.name,
                    "status": "fail",
                    "seconds": round(elapsed, 1),
                    "error": str(exc),
                    "log": str(log_file),
                }
                fail += 1
                print(f"FAIL {nb_path.name}: {exc}", flush=True)
                traceback.print_exc()
            slog.write(json.dumps(record) + "\n")
            slog.flush()

    print(f"\nDone: {ok} ok, {fail} failed, {skip} skipped (log: {summary_path})")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
