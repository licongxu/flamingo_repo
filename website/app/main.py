"""FastAPI application: JSON API under ``/api/v1`` plus the static frontend.

Run with::

    uvicorn app.main:app --host 0.0.0.0 --port 8080

from the ``website/`` directory (see README.md).
"""

import csv
import io
import logging
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import catalogues as catq
from .config import Settings
from .manifest import Manifest
from .storage import StorageError, download_response

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    manifest = Manifest(settings.index_path)

    def require_token(request: Request) -> None:
        if settings.api_token and not secrets.compare_digest(
            request.headers.get("X-API-Key", ""), settings.api_token
        ):
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    api = FastAPI(
        title="FLAMINGO Data Portal API",
        description="Internal API for FLAMINGO lightcone catalogues and maps.",
        version="0.1.0",
        dependencies=[Depends(require_token)],
    )

    def manifest_or_503() -> Manifest:
        if not manifest.available:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"manifest not found at {settings.index_path}; "
                    "run: python scripts/build_index.py"
                ),
            )
        return manifest

    def file_or_404(file_id: str) -> dict:
        entry = manifest_or_503().get(file_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"no file with id {file_id!r}")
        return entry

    def catalogue_or_404(file_id: str) -> Path:
        entry = file_or_404(file_id)
        if entry["kind"] != "catalogue":
            raise HTTPException(status_code=404, detail=f"{file_id!r} is not a catalogue")
        path = settings.data_root / entry["relpath"]
        # The CSV may have been replaced by its parquet conversion; the query
        # engine prefers the parquet sibling anyway.
        if not (path.is_file() or path.with_suffix(".parquet").is_file()):
            raise HTTPException(status_code=503, detail=f"catalogue file missing: {path}")
        return path

    @api.get("/summary")
    def summary():
        return manifest_or_503().summary()

    @api.get("/simulations")
    def simulations():
        return manifest_or_503().tree()

    @api.get("/files")
    def files(
        kind: str | None = None,
        simulation: str | None = None,
        variant: str | None = None,
        lightcone: int | None = None,
    ):
        return manifest_or_503().files(
            kind=kind, simulation=simulation, variant=variant, lightcone=lightcone
        )

    @api.get("/files/{file_id}")
    def file_meta(file_id: str):
        return file_or_404(file_id)

    @api.get("/files/{file_id}/download")
    def file_download(file_id: str):
        entry = file_or_404(file_id)
        try:
            resp = download_response(settings, entry["relpath"], entry["filename"])
        except StorageError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if resp is None:
            raise HTTPException(status_code=503, detail="file missing on local storage")
        return resp

    @api.get("/catalogues")
    def list_catalogues():
        return manifest_or_503().files(kind="catalogue")

    @api.get("/catalogues/{file_id}/stats")
    def catalogue_stats(file_id: str):
        return catq.stats(catalogue_or_404(file_id))

    @api.get("/catalogues/{file_id}/query")
    def catalogue_query(
        file_id: str,
        min_M500c: float | None = None,
        max_M500c: float | None = None,
        min_z: float | None = None,
        max_z: float | None = None,
        min_q: float | None = None,
        max_q: float | None = None,
        columns: str | None = Query(None, description="comma-separated column names"),
        sort: str | None = None,
        desc: bool = False,
        limit: int = Query(1000, le=catq.MAX_LIMIT, gt=0),
        offset: int = Query(0, ge=0),
        format: str = Query("json", pattern="^(json|csv)$"),
    ):
        path = catalogue_or_404(file_id)
        select = [c.strip() for c in columns.split(",")] if columns else None
        try:
            result = catq.query(
                path,
                select=select,
                ranges={
                    "M500c": (min_M500c, max_M500c),
                    "z": (min_z, max_z),
                    "q": (min_q, max_q),
                },
                sort=sort,
                descending=desc,
                limit=limit,
                offset=offset,
            )
        except catq.CatalogueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if format == "csv":
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(result["columns"])
            writer.writerows(result["rows"])
            buf.seek(0)
            return StreamingResponse(
                buf,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={file_id}_query.csv"},
            )
        return result

    @api.get("/maps")
    def list_maps():
        entries = manifest_or_503().files(kind="map")
        out = []
        for e in entries:
            e = dict(e)
            png = settings.previews_dir / f"{e['id']}.png"
            e["preview_url"] = f"/previews/{e['id']}.png" if png.is_file() else None
            out.append(e)
        return out

    app.mount("/api/v1", api)
    settings.previews_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/previews", StaticFiles(directory=settings.previews_dir), name="previews")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    app.state.settings = settings
    app.state.manifest = manifest
    return app


app = create_app()
