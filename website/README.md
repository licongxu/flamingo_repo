# FLAMINGO Data Portal

Internal website + REST API for the FLAMINGO data products under
`/rds/rds-lxu/flamingo` (halo catalogues and Compton-y HEALPix maps across box
sizes, feedback prescriptions, and lightcone realizations).

Design and decisions: [docs/DESIGN.md](docs/DESIGN.md).

## Quick start

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pip install -r requirements.txt          # only duckdb/httpx are new
cd website

python scripts/build_index.py            # scan /rds -> data/index.json
python scripts/make_previews.py          # render map preview PNGs (batched; ~30 s/map)

uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Then open `http://<node>:8080` (collaborators without cluster access:
`ssh -L 8080:<node>:8080 <login-host>` and browse `http://localhost:8080`).

- `/` — overview, per-simulation inventory, filterable file list with downloads
- `/catalogues.html` — interactive halo-catalogue queries (mass / z / q cuts, CSV export)
- `/maps.html` — map gallery with Mollweide previews and FITS downloads
- `/api/v1/docs` — interactive OpenAPI documentation for the whole API

## API in 30 seconds

```bash
BASE=http://localhost:8080/api/v1
curl $BASE/summary                                   # inventory totals
curl $BASE/files?kind=map\&simulation=L1_m9          # file listing + ids
curl -OJ $BASE/files/<id>/download                   # download any file
curl "$BASE/catalogues/<id>/query?min_M500c=1e14&max_z=0.5&columns=z,M_500c_Msun&limit=100"
curl "$BASE/catalogues/<id>/query?min_q=5&format=csv" -o clusters.csv
```

From Python:

```python
import requests
base = "http://localhost:8080/api/v1"
cats = requests.get(f"{base}/catalogues").json()
cid = next(c["id"] for c in cats if c["simulation"] == "L1_m9" and c["variant"] == "fiducial")
rows = requests.get(f"{base}/catalogues/{cid}/query",
                    params={"min_M500c": 1e14, "max_z": 0.5, "limit": 10000}).json()
```

## Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `FLAMINGO_WEB_DATA_ROOT` | `/rds/rds-lxu/flamingo` | local data tree |
| `FLAMINGO_WEB_INDEX` | `website/data/index.json` | manifest location |
| `FLAMINGO_WEB_PREVIEWS` | `website/data/previews` | preview PNG dir |
| `FLAMINGO_WEB_STORAGE` | `local` | `local` (stream from disk) or `gcs` (signed URLs) |
| `FLAMINGO_GCS_BUCKET` | – | bucket name for `gcs` backend |
| `FLAMINGO_WEB_TOKEN` | – | if set, `/api` requires `X-API-Key: <token>` |

## Moving the data to the Google bucket

```bash
gcloud auth login                                     # once
python scripts/sync_to_gcs.py --bucket <name>         # prints the rsync command
python scripts/sync_to_gcs.py --bucket <name> --execute   # ~347 GB; run in tmux/SLURM
export FLAMINGO_WEB_STORAGE=gcs FLAMINGO_GCS_BUCKET=<name>
```

Object names mirror the manifest `relpath`, so nothing else changes: download
endpoints switch from streaming local files to 307-redirecting to signed URLs
(1 h expiry; falls back to public object URLs if no signing credentials).

## Big catalogues

Queries read the CSVs directly via DuckDB. The 9.9 GB L2p8 catalogues are
usable but slow that way; convert once for interactive queries (Parquet is
written next to the CSV in the data tree, and the API picks it up
automatically):

```bash
python scripts/convert_parquet.py --only L2p8_m9
```

## Maintenance

- Data changed on disk? `python scripts/build_index.py` (the running server
  picks up the new manifest automatically via mtime).
- New maps? `python scripts/make_previews.py` (skips existing PNGs; `--force`
  to re-render, `--only <substring>` to batch).
- Tests: `python -m pytest tests/ -q` (runs against a synthetic tree, no /rds access).
