"""Resolve file downloads against the configured storage backend.

``local`` streams straight from the data root (Starlette handles HTTP
range requests); ``gcs`` redirects to a V4 signed URL in the Google
bucket. There is deliberately no fallback to unsigned public URLs: a
signing failure raises `StorageError` (surfaced as HTTP 503) rather
than silently bypassing the portal's token auth via a public bucket.
"""

import datetime

from fastapi.responses import FileResponse, RedirectResponse

from .config import Settings

SIGNED_URL_LIFETIME = datetime.timedelta(hours=1)


class StorageError(RuntimeError):
    """Download could not be resolved (mapped to HTTP 503)."""


def download_response(settings: Settings, relpath: str, filename: str):
    if settings.storage_backend == "gcs":
        if not settings.gcs_bucket:
            raise StorageError("storage backend is 'gcs' but FLAMINGO_GCS_BUCKET is not set")
        return RedirectResponse(signed_gcs_url(settings.gcs_bucket, relpath), status_code=307)
    local = settings.data_root / relpath
    if not local.is_file():
        return None
    return FileResponse(local, filename=filename)


def signed_gcs_url(bucket_name: str, relpath: str) -> str:
    try:
        from google.cloud import storage

        client = storage.Client()
        blob = client.bucket(bucket_name).blob(relpath)
        return blob.generate_signed_url(version="v4", expiration=SIGNED_URL_LIFETIME)
    except Exception as exc:  # noqa: BLE001 - import/auth/network failures all end here
        raise StorageError(f"signed URL generation failed: {exc}") from exc
