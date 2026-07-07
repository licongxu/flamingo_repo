# FLAMINGO internal data website: design

Date: 2026-07-07. Branch: `website`. Author: Claude (autonomous session), for Licong Xu.

## Goal

An internal website for the FLAMINGO data products under `/rds/rds-lxu/flamingo`, per the
supervisor's requirement: "a website, with data on a Google bucket and an API that allows
interacting with the data". Collaborators should be able to browse what exists, query the halo
catalogues, preview the maps, and download files, without shell access to the cluster.

## Data inventory (as scanned 2026-07-07)

```
/rds/rds-lxu/flamingo/
  L1_m9/                       1 Gpc box, m9 resolution, 9 feedback variants   (~111 GB)
    catalogues/                halo_catalogue_M500c_{1e13,5e13}_zlt3_<variant>_yang26rot[_qfrommz].csv
    maps/                      y_unlensed_<variant>_lc0_nside4096.fits  (1.6 GB each)
  L2p8_m9/                     2.8 Gpc box, fiducial feedback, 8 lightcones    (~218 GB)
    lightcone{0..7}/
      catalogues/              1e13 full (9.9 GB) + 5e13 qfrommz (658 MB) CSVs
      healpix_map/             y_unlensed_L2p8_m9_lc{N}.fits (1.6 GB)
        rotation_groups/       (lc0 only) shell-grouped Compton-y maps + JSON manifest
        snapshots/             (lc0 only) per-shell maps incl. one nside16384 (25 GB)
```

Feedback variants (L1_m9): `L1_m9` (fiducial), `fgas+2sigma`, `fgas-2sigma`, `fgas-4sigma`,
`fgas-8sigma`, `Jet`, `Jet_fgas-4sigma`, `Mstar-1sigma`, `Mstar-1sigma_fgas-4sigma`.

Catalogue columns: `snap, soap_index, z, x/y/z_Mpc, r_comoving_Mpc, shell_idx,
theta/phi_nat_rad, lon/lat_nat_deg, theta/phi_rot_rad, lon/lat_rot_deg, x/y/z_rot_Mpc,
M_500c_Msun, M_200c_Msun, M_200m_Msun, R_500c_Mpc, R_200c_Mpc, R_200m_Mpc,
Y_500c_Mpc2, Y_500c_noAGN_Mpc2, Y_5R500c_Mpc2, Y_5R500c_noAGN_Mpc2` (+ `q_from_mz` in the
5e13 `qfrommz` files).

`.progress.json` and `.pre_repair.bak` files are build artifacts and are excluded from the index.

## Decisions (made autonomously; flagged for review)

1. **Stack: FastAPI + vanilla HTML/JS.** FastAPI is already in the venv, gives OpenAPI docs at
   `/docs` for free (the "API to interact with the data"), and serves the static frontend from the
   same process. No node/build step: this is an internal tool; simplicity first.
2. **Manifest-driven, not live-walking.** A scanner (`scripts/build_index.py`) walks the data tree
   once and writes `data/index.json` with per-file metadata (kind, simulation, feedback variant,
   lightcone, nside, mass cut, size, mtime). The API serves from this manifest, so requests never
   touch `/rds` for listing. Re-run the scanner when data changes.
3. **Catalogue queries via DuckDB.** The API exposes filtered queries (mass, redshift, sky
   position, column selection, sort, limit) over the catalogue CSVs. DuckDB reads CSV directly
   with predicate pushdown and is fast enough for the 658 MB files; for the 9.9 GB ones a
   one-off Parquet conversion (`scripts/convert_parquet.py`) makes queries interactive. The API
   prefers `<name>.parquet` next to the CSV when present.
4. **Two storage backends, `local` now, `gcs` when the bucket exists.**
   - `local` (default): the API streams files straight from `/rds` with HTTP range support.
   - `gcs`: download endpoints 307-redirect to V4 signed URLs (1 h expiry) in the bucket.
     A signing failure returns 503; there is deliberately no fallback to unsigned public
     URLs, which would silently bypass the token auth on a public bucket.
   Backend and bucket are set by env vars (`FLAMINGO_WEB_STORAGE`, `FLAMINGO_GCS_BUCKET`).
   `scripts/sync_to_gcs.py` uploads the tree to the bucket, preserving relative paths, so the
   object names match the manifest `relpath` exactly. The site works today without any bucket.
5. **Map previews are pre-rendered.** `scripts/make_previews.py` renders a Mollweide PNG per map
   (log10 y on a per-map 1st/99.9th-percentile colour scale, degraded to nside 512 for speed)
   into `data/previews/`. The gallery shows these;
   nobody renders a 1.6 GB FITS per page load. The nside16384 shell map is indexed but skipped
   for preview by a size guard.
6. **Auth: optional shared token.** If `FLAMINGO_WEB_TOKEN` is set, all `/api` data routes
   require `X-API-Key: <token>` (constant-time comparison). Known scope limits, accepted for an
   internal tool: the static pages, the pre-rendered previews under `/previews`, and the API
   docs at `/api/v1/docs` are not token-gated; only the data endpoints are. Unset means fully
   open (e.g. behind an SSH tunnel or institutional VPN). Anything stronger (OAuth/IAP) belongs
   to the deployment layer, not this codebase.
7. **Location `website/` at repo root**, as explicitly requested (overrides the usual
   "research under `autoresearch/`" rule). Generated artifacts (`data/index.json`, previews)
   live in `website/data/`; previews are small PNGs, no large outputs in the repo.
8. **Dependencies** are listed in `website/requirements.txt` (only `duckdb` was missing from the
   venv and has been installed). `pyproject.toml` is untouched, per the autonomous-operation rules.

## Alternatives considered

- **Static site generator (no API):** rejected; supervisor explicitly wants an API for
  interacting with the data, and catalogue querying needs a live backend.
- **Full JS framework (React/Next):** rejected; adds a build toolchain for an internal tool with
  three pages. Vanilla JS + `fetch` against the API is sufficient and maintainable.
- **Serve everything from GCS from day one:** rejected; the bucket does not exist yet. The
  storage abstraction makes the switch a config change plus one sync run.

## Architecture

```
website/
  app/
    config.py        env-var settings (data root, storage backend, bucket, token)
    manifest.py      load data/index.json, filter/list datasets
    catalogues.py    DuckDB/pandas query engine over catalogue files
    storage.py       local streaming vs GCS signed-URL download resolution
    main.py          FastAPI app: /api/v1 routes + static frontend mount
  scripts/
    build_index.py   scan the data tree -> data/index.json
    make_previews.py render Mollweide PNGs -> data/previews/
    convert_parquet.py  optional CSV -> Parquet for the big catalogues
    sync_to_gcs.py   upload data tree to the Google bucket
  static/            index.html, catalogues.html, maps.html, css/, js/
  data/              generated: index.json, previews/*.png
  tests/test_api.py  API tests against a synthetic mini data tree
  requirements.txt, README.md, docs/DESIGN.md (this file)
```

### API surface (`/api/v1`)

| Route | Purpose |
|---|---|
| `GET /summary` | totals: simulations, variants, file counts, sizes |
| `GET /simulations` | the simulation/variant/lightcone hierarchy |
| `GET /files` | flat file list with filters (`kind`, `simulation`, `variant`, `lightcone`) |
| `GET /files/{id}` | metadata for one file |
| `GET /files/{id}/download` | stream (local) or redirect to signed URL (gcs) |
| `GET /catalogues` | list queryable catalogues |
| `GET /catalogues/{id}/query` | filtered rows: `min_M500c, max_z, min_z, columns, sort, limit, offset`, JSON or CSV |
| `GET /catalogues/{id}/stats` | row count, column min/max for query building |
| `GET /maps` | list maps with preview URLs and metadata |
| `GET /api/v1/openapi.json`, `/api/v1/docs` | machine-readable schema + interactive docs |

### Error handling

Unknown ids give 404 with a JSON detail; bad query parameters give 422 (FastAPI validation);
query row limits are capped server-side (`limit <= 100000`) so nobody pulls 9.9 GB through
JSON. Missing manifest gives a 503 with the command to run.

### Testing

`tests/` builds a tiny synthetic tree (two fake catalogues, fake FITS-less map entries) in
`tmp_path`, generates a manifest with the real scanner, and exercises every endpoint with
FastAPI's `TestClient`. No test touches `/rds` or downloads anything.

## Deployment (documented in README)

```
uvicorn app.main:app --host 0.0.0.0 --port 8080   # on a cluster node
ssh -L 8080:node:8080 ...                          # collaborators tunnel in
```

GCS migration: create bucket, `python scripts/sync_to_gcs.py --bucket <name>`, set
`FLAMINGO_WEB_STORAGE=gcs`, `FLAMINGO_GCS_BUCKET=<name>`, restart.
