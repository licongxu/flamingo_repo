"""Sync the FLAMINGO data tree to a Google Cloud Storage bucket.

Usage::

    python scripts/sync_to_gcs.py --bucket flamingo-internal            # prints the command
    python scripts/sync_to_gcs.py --bucket flamingo-internal --execute  # actually runs it

Object names mirror the relative paths in ``index.json``, so switching
the portal to the bucket is just::

    export FLAMINGO_WEB_STORAGE=gcs
    export FLAMINGO_GCS_BUCKET=flamingo-internal

Uses ``gcloud storage rsync`` (falls back to ``gsutil -m rsync``); both
skip already-uploaded objects, so re-running resumes safely. Requires
prior authentication (``gcloud auth login``).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = "/rds/rds-lxu/flamingo"
EXCLUDE_RE = r".*\.progress\.json$|.*\.pre_repair\.bak$"


def build_command(root: Path, bucket: str) -> list[str]:
    if shutil.which("gcloud"):
        return [
            "gcloud", "storage", "rsync", "--recursive",
            f"--exclude={EXCLUDE_RE}",
            str(root), f"gs://{bucket}",
        ]
    if shutil.which("gsutil"):
        return ["gsutil", "-m", "rsync", "-r", "-x", EXCLUDE_RE, str(root), f"gs://{bucket}"]
    raise RuntimeError("neither gcloud nor gsutil found on PATH")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    ap.add_argument("--bucket", required=True)
    ap.add_argument(
        "--execute",
        action="store_true",
        help="run the sync (default: print the command and exit)",
    )
    args = ap.parse_args(argv)

    cmd = build_command(args.root, args.bucket)
    print(" ".join(cmd))
    if not args.execute:
        print("dry run: pass --execute to upload (~330 GB; run inside tmux/SLURM)")
        return 0
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
