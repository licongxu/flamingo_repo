"""Settings for the FLAMINGO data portal, read from environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path

WEBSITE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = "/rds/rds-lxu/flamingo"


@dataclass
class Settings:
    """Runtime configuration.

    Parameters
    ----------
    data_root : Path
        Root of the FLAMINGO data tree (local filesystem).
    index_path : Path
        Path to the generated ``index.json`` manifest.
    previews_dir : Path
        Directory of pre-rendered map preview PNGs.
    storage_backend : str
        ``"local"`` streams files from ``data_root``; ``"gcs"`` redirects
        downloads to signed URLs in ``gcs_bucket``.
    gcs_bucket : str
        Google Cloud Storage bucket name (required when backend is ``"gcs"``).
    api_token : str
        Optional shared token; when set, ``/api`` routes require the
        ``X-API-Key`` header to match.
    """

    data_root: Path = field(default_factory=lambda: Path(DEFAULT_DATA_ROOT))
    index_path: Path = field(default_factory=lambda: WEBSITE_DIR / "data" / "index.json")
    previews_dir: Path = field(default_factory=lambda: WEBSITE_DIR / "data" / "previews")
    storage_backend: str = "local"
    gcs_bucket: str = ""
    api_token: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        return cls(
            data_root=Path(os.environ.get("FLAMINGO_WEB_DATA_ROOT", str(defaults.data_root))),
            index_path=Path(os.environ.get("FLAMINGO_WEB_INDEX", str(defaults.index_path))),
            previews_dir=Path(
                os.environ.get("FLAMINGO_WEB_PREVIEWS", str(defaults.previews_dir))
            ),
            storage_backend=os.environ.get("FLAMINGO_WEB_STORAGE", "local"),
            gcs_bucket=os.environ.get("FLAMINGO_GCS_BUCKET", ""),
            api_token=os.environ.get("FLAMINGO_WEB_TOKEN", ""),
        )
