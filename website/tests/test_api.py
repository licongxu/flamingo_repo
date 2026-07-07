"""API tests against a synthetic mini data tree (never touches /rds)."""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

WEBSITE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEBSITE_DIR))
sys.path.insert(0, str(WEBSITE_DIR / "scripts"))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from build_index import build_index, classify  # noqa: E402

CAT_HEADER = (
    "snap,soap_index,z,theta_rot_rad,phi_rot_rad,M_500c_Msun,R_500c_Mpc,Y_5R500c_Mpc2"
)
CAT_Q_HEADER = CAT_HEADER + ",q_from_mz"


def _write_cat(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# FLAMINGO test catalogue.", "# Selection: test.", header, *rows]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="session")
def data_tree(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("flamingo")
    _write_cat(
        root / "L1_m9/catalogues/halo_catalogue_M500c_1e13_zlt3_Jet_yang26rot.csv",
        CAT_HEADER,
        [
            "10,1,0.10,0.5,1.0,2.0e13,0.6,1.1e-6",
            "11,2,0.50,0.7,2.0,5.0e14,1.2,3.3e-5",
            "12,3,1.20,0.9,3.0,1.5e14,0.9,8.0e-6",
        ],
    )
    _write_cat(
        root
        / "L2p8_m9/lightcone0/catalogues/halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_qfrommz.csv",
        CAT_Q_HEADER,
        [
            "10,4,0.20,0.4,1.5,8.0e13,0.7,2.0e-6,3.5",
            "11,5,0.90,0.6,2.5,3.0e14,1.0,1.5e-5,7.2",
        ],
    )
    # Skipped artifacts and non-catalogue files.
    (root / "L1_m9/catalogues/foo.csv.progress.json").write_text("{}")
    (root / "L1_m9/maps").mkdir(parents=True)
    (root / "L1_m9/maps/y_unlensed_Jet_lc0_nside4096.fits").write_bytes(b"\x00" * 64)
    (root / "L2p8_m9/lightcone0/healpix_map").mkdir(parents=True)
    (root / "L2p8_m9/lightcone0/healpix_map/y_unlensed_L2p8_m9_lc0.fits").write_bytes(
        b"\x00" * 64
    )
    return root


@pytest.fixture(scope="session")
def settings(data_tree, tmp_path_factory) -> Settings:
    work = tmp_path_factory.mktemp("website_data")
    index_path = work / "index.json"
    with open(index_path, "w") as fh:
        json.dump(build_index(data_tree), fh)
    return Settings(
        data_root=data_tree, index_path=index_path, previews_dir=work / "previews"
    )


@pytest.fixture(scope="session")
def client(settings) -> TestClient:
    return TestClient(create_app(settings))


def test_summary(client):
    s = client.get("/api/v1/summary").json()
    assert s["n_files"] == 4
    assert s["by_kind"]["catalogue"]["count"] == 2
    assert s["by_kind"]["map"]["count"] == 2
    assert s["simulations"] == ["L1_m9", "L2p8_m9"]
    assert "Jet" in s["variants"] and "fiducial" in s["variants"]


def test_simulations_tree(client):
    tree = client.get("/api/v1/simulations").json()
    assert [t["simulation"] for t in tree] == ["L1_m9", "L2p8_m9"]
    l2 = tree[1]
    assert l2["lightcones"] == [0]


def test_files_filters(client):
    r = client.get("/api/v1/files", params={"kind": "map", "simulation": "L1_m9"})
    files = r.json()
    assert len(files) == 1
    assert files[0]["variant"] == "Jet"
    assert files[0]["nside"] == 4096
    # progress.json artifact excluded
    all_files = client.get("/api/v1/files").json()
    assert not any("progress" in f["filename"] for f in all_files)


def test_file_meta_and_404(client):
    files = client.get("/api/v1/files").json()
    fid = files[0]["id"]
    assert client.get(f"/api/v1/files/{fid}").json()["id"] == fid
    assert client.get("/api/v1/files/nonexistent0").status_code == 404


def test_download_local(client):
    maps = client.get("/api/v1/files", params={"kind": "map"}).json()
    r = client.get(f"/api/v1/files/{maps[0]['id']}/download")
    assert r.status_code == 200
    assert r.content == b"\x00" * 64


def test_catalogue_stats(client):
    cats = client.get("/api/v1/catalogues").json()
    qcat = next(c for c in cats if c["has_q"])
    s = client.get(f"/api/v1/catalogues/{qcat['id']}/stats").json()
    assert s["n"] == 2
    assert s["min_q"] == pytest.approx(3.5)
    assert "M_500c_Msun" in s["columns"]


def test_catalogue_query_filters(client):
    cats = client.get("/api/v1/catalogues").json()
    cat = next(c for c in cats if c["simulation"] == "L1_m9")
    r = client.get(
        f"/api/v1/catalogues/{cat['id']}/query",
        params={"min_M500c": 1e14, "max_z": 1.0, "columns": "z,M_500c_Msun"},
    ).json()
    assert r["columns"] == ["z", "M_500c_Msun"]
    assert r["n_rows"] == 1
    assert r["rows"][0][1] == pytest.approx(5.0e14)


def test_catalogue_query_sort_and_csv(client):
    cats = client.get("/api/v1/catalogues").json()
    cat = next(c for c in cats if c["simulation"] == "L1_m9")
    r = client.get(
        f"/api/v1/catalogues/{cat['id']}/query",
        params={"sort": "M_500c_Msun", "desc": True, "limit": 2},
    ).json()
    masses = [row[r["columns"].index("M_500c_Msun")] for row in r["rows"]]
    assert masses == sorted(masses, reverse=True)

    csv_resp = client.get(
        f"/api/v1/catalogues/{cat['id']}/query", params={"format": "csv", "limit": 10}
    )
    assert csv_resp.status_code == 200
    assert csv_resp.headers["content-type"].startswith("text/csv")
    assert csv_resp.text.splitlines()[0].startswith("snap,")


def test_catalogue_query_validation(client):
    cats = client.get("/api/v1/catalogues").json()
    cat = cats[0]
    assert (
        client.get(
            f"/api/v1/catalogues/{cat['id']}/query", params={"columns": "nope"}
        ).status_code
        == 422
    )
    assert (
        client.get(
            f"/api/v1/catalogues/{cat['id']}/query", params={"sort": "1; DROP"}
        ).status_code
        == 422
    )
    assert (
        client.get(
            f"/api/v1/catalogues/{cat['id']}/query", params={"limit": 10_000_000}
        ).status_code
        == 422
    )


def test_q_filter_only_where_present(client):
    cats = client.get("/api/v1/catalogues").json()
    noq = next(c for c in cats if not c["has_q"])
    hasq = next(c for c in cats if c["has_q"])
    assert (
        client.get(
            f"/api/v1/catalogues/{noq['id']}/query", params={"min_q": 5}
        ).status_code
        == 422
    )
    r = client.get(f"/api/v1/catalogues/{hasq['id']}/query", params={"min_q": 5}).json()
    assert r["n_rows"] == 1


def test_maps_endpoint(client):
    maps = client.get("/api/v1/maps").json()
    assert len(maps) == 2
    assert all(m["preview_url"] is None for m in maps)  # no previews rendered in tests


def test_token_auth(settings):
    from dataclasses import replace

    secured = TestClient(create_app(replace(settings, api_token="s3cret")))
    assert secured.get("/api/v1/summary").status_code == 401
    ok = secured.get("/api/v1/summary", headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200


def test_missing_manifest_gives_503(settings, tmp_path):
    from dataclasses import replace

    broken = TestClient(create_app(replace(settings, index_path=tmp_path / "nope.json")))
    r = broken.get("/api/v1/summary")
    assert r.status_code == 503
    assert "build_index" in r.json()["detail"]


def test_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "FLAMINGO" in r.text


def test_query_offset_pagination(client):
    cats = client.get("/api/v1/catalogues").json()
    cat = next(c for c in cats if c["simulation"] == "L1_m9")
    url = f"/api/v1/catalogues/{cat['id']}/query"
    all_z = client.get(url, params={"sort": "z", "columns": "z"}).json()["rows"]
    page = client.get(
        url, params={"sort": "z", "columns": "z", "limit": 1, "offset": 1}
    ).json()["rows"]
    assert page == [all_z[1]]


def test_shell_and_rotation_group_classification():
    base = Path("L2p8_m9/lightcone0/healpix_map")
    e = classify(base / "snapshots/comptonY_L2p8_m9_lc0_nside4096_shell10.fits")
    assert e["kind"] == "map" and e["shell"] == 10 and e["nside"] == 4096
    assert e["variant"] == "fiducial" and e["lightcone"] == 0
    e = classify(
        base / "rotation_groups/comptonY_L2p8_m9_lc0_nside4096_rotgroup2_shells16-26.fits"
    )
    assert e["rotation_group"] == 2 and e["shell_range"] == "16-26" and e["shell"] is None
    assert classify(base / "rotation_groups/rotation_groups_L2p8_m9_lc0.json")["kind"] == "aux"
    assert classify(Path("stray_root_file.csv")) is None


def test_parquet_sibling_preferred_and_not_double_indexed(tmp_path):
    import duckdb

    root = tmp_path / "flamingo"
    csv = root / "L1_m9/catalogues/halo_catalogue_M500c_1e13_zlt3_Jet_yang26rot.csv"
    _write_cat(csv, CAT_HEADER, ["10,1,0.10,0.5,1.0,2.0e13,0.6,1.1e-6"])
    duckdb.connect().execute(
        "COPY (SELECT * FROM read_csv(?, header=true, comment='#')) "
        f"TO '{csv.with_suffix('.parquet')}' (FORMAT parquet)",
        [str(csv)],
    )
    index = build_index(root)
    cats = [f for f in index["files"] if f["kind"] == "catalogue"]
    assert len(cats) == 1  # parquet sibling is not a second catalogue

    index_path = tmp_path / "index.json"
    with open(index_path, "w") as fh:
        json.dump(index, fh)
    c = TestClient(
        create_app(Settings(data_root=root, index_path=index_path,
                            previews_dir=tmp_path / "previews"))
    )
    s = c.get(f"/api/v1/catalogues/{cats[0]['id']}/stats").json()
    assert s["source"].endswith(".parquet")
    assert s["n"] == 1
    r = c.get(f"/api/v1/catalogues/{cats[0]['id']}/query").json()
    assert r["n_rows"] == 1


def test_download_missing_file_gives_503(data_tree, settings, tmp_path):
    import shutil

    root = tmp_path / "flamingo"
    shutil.copytree(data_tree, root)
    index_path = tmp_path / "index.json"
    with open(index_path, "w") as fh:
        json.dump(build_index(root), fh)
    c = TestClient(
        create_app(Settings(data_root=root, index_path=index_path,
                            previews_dir=tmp_path / "previews"))
    )
    maps = c.get("/api/v1/files", params={"kind": "map"}).json()
    (root / maps[0]["relpath"]).unlink()
    assert c.get(f"/api/v1/files/{maps[0]['id']}/download").status_code == 503


def test_gcs_backend_redirects(settings, monkeypatch):
    from dataclasses import replace

    import app.storage as storage_mod

    monkeypatch.setattr(
        storage_mod, "signed_gcs_url", lambda bucket, rel: f"https://signed.test/{bucket}/{rel}"
    )
    c = TestClient(
        create_app(replace(settings, storage_backend="gcs", gcs_bucket="mybucket")),
        follow_redirects=False,
    )
    maps = c.get("/api/v1/files", params={"kind": "map"}).json()
    r = c.get(f"/api/v1/files/{maps[0]['id']}/download")
    assert r.status_code == 307
    assert r.headers["location"].startswith("https://signed.test/mybucket/")


def test_gcs_backend_without_bucket_gives_503(settings):
    from dataclasses import replace

    c = TestClient(create_app(replace(settings, storage_backend="gcs")))
    maps = c.get("/api/v1/files", params={"kind": "map"}).json()
    r = c.get(f"/api/v1/files/{maps[0]['id']}/download")
    assert r.status_code == 503
    assert "FLAMINGO_GCS_BUCKET" in r.json()["detail"]
