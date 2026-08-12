"""Tests for the streamed `_qfrommap.csv` production command."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import healpy as hp
import numpy as np
import pandas as pd
import pytest

from flamingo.aperture_snr import APERTURE_COLUMNS


SCRIPT = Path(__file__).parents[1] / "scripts/catalogue/compute_qfrommap_catalogues.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("compute_qfrommap_catalogues", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _l1_job(root: Path, variant: str = "L1_m9") -> tuple[Path, Path]:
    source = _touch(
        root
        / "L1_m9/catalogues"
        / f"halo_catalogue_M500c_5e13_zlt3_{variant}_yang26rot_qfrommz.csv"
    )
    map_path = _touch(
        root / "L1_m9/maps" / f"y_unlensed_{variant}_lc0_nside4096.fits"
    )
    return source, map_path


def _l2_job(root: Path, index: int = 0) -> tuple[Path, Path]:
    lightcone = root / f"L2p8_m9/lightcone{index}"
    source = _touch(
        lightcone
        / "catalogues/halo_catalogue_M500c_5e13_zlt3_"
        "L2p8_m9_yang26rot_qfrommz.csv"
    )
    map_path = _touch(
        lightcone / f"healpix_map/y_unlensed_L2p8_m9_lc{index}.fits"
    )
    return source, map_path


def test_discover_jobs_pairs_canonical_l1_and_l2_inputs(mod, tmp_path):
    """A wrong map/catalogue pairing would measure apertures at unrelated sky pixels."""
    l1_source, l1_map = _l1_job(tmp_path)
    l2_source, l2_map = _l2_job(tmp_path)

    jobs = mod.discover_jobs(tmp_path, dataset="all", only=())

    assert [(job.dataset, job.label) for job in jobs] == [
        ("l1", "L1_m9"),
        ("l2", "lightcone0"),
    ]
    assert (jobs[0].source, jobs[0].map_path) == (l1_source, l1_map)
    assert (jobs[1].source, jobs[1].map_path) == (l2_source, l2_map)
    assert jobs[0].output.name.endswith("_yang26rot_qfrommap.csv")
    assert jobs[1].output.name.endswith("_yang26rot_qfrommap.csv")


def test_discover_jobs_ignores_noncanonical_catalogue_families(mod, tmp_path):
    """Derived q catalogues and the 1e13 population must not multiply production jobs."""
    source, _ = _l1_job(tmp_path)
    directory = source.parent
    _touch(directory / source.name.replace(".csv", "_alpha_fixed_1p12.csv"))
    _touch(directory / source.name.replace("_qfrommz.csv", "_qfrommap.csv"))
    _touch(directory / source.name.replace("M500c_5e13", "M500c_1e13").replace("_qfrommz", ""))

    jobs = mod.discover_jobs(tmp_path, dataset="l1", only=())

    assert [job.source for job in jobs] == [source]


def test_discover_jobs_applies_dataset_and_repeatable_only_filters(mod, tmp_path):
    """Selection options must narrow work without changing canonical pairing."""
    _l1_job(tmp_path, "L1_m9")
    jet_source, _ = _l1_job(tmp_path, "Jet")
    _l2_job(tmp_path, 0)
    lc1_source, _ = _l2_job(tmp_path, 1)

    l1 = mod.discover_jobs(tmp_path, dataset="l1", only=("Jet",))
    l2 = mod.discover_jobs(tmp_path, dataset="l2", only=("lightcone1", "unused"))

    assert [job.source for job in l1] == [jet_source]
    assert [job.source for job in l2] == [lc1_source]


def test_discover_jobs_rejects_a_missing_map_before_writes(mod, tmp_path):
    """A partial manifest must fail before any catalogue can be installed."""
    source, map_path = _l1_job(tmp_path)
    map_path.unlink()

    with pytest.raises(FileNotFoundError, match=source.name):
        mod.discover_jobs(tmp_path, dataset="l1", only=())


def test_discover_jobs_rejects_malformed_lightcone_names(mod, tmp_path):
    """Silently coercing a malformed lightcone name could select the wrong map."""
    source = _touch(
        tmp_path
        / "L2p8_m9/lightconeX/catalogues"
        / "halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_qfrommz.csv"
    )

    with pytest.raises(ValueError, match="lightconeX"):
        mod.discover_jobs(tmp_path, dataset="l2", only=())


def test_discover_jobs_rejects_an_empty_selection(mod, tmp_path):
    """A typo in an `--only` filter must not look like a successful no-op."""
    _l1_job(tmp_path)

    with pytest.raises(ValueError, match="no canonical catalogues"):
        mod.discover_jobs(tmp_path, dataset="l1", only=("not-present",))


def _writer_job(mod, tmp_path: Path, frame: pd.DataFrame):
    source = tmp_path / "source_qfrommz.csv"
    source.write_text("# original catalogue\n# retained detail\n")
    frame.to_csv(source, mode="a", index=False)

    map_path = tmp_path / "map.fits"
    ymap = np.arange(hp.nside2npix(8), dtype=np.float32) * 1.0e-8
    hp.write_map(map_path, ymap, dtype=np.float32, overwrite=True)
    return mod.CatalogueJob(
        dataset="l1",
        label="test",
        source=source,
        map_path=map_path,
        output=tmp_path / "source_qfrommap.csv",
    )


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "snap": [20, 21],
            "soap_index": [7, 9],
            "z": [0.1, 0.2],
            "R_500c_Mpc": [80.0, 80.0],
            "theta_rot_rad": [1.0, 1.2],
            "phi_rot_rad": [2.0, -1.0],
            "q_from_mz": [99.0, 98.0],
        }
    )


def test_rewrite_catalogue_streams_schema_identity_and_provenance(mod, tmp_path):
    """The installed CSV must preserve identity while replacing parametric q."""
    source_frame = _valid_frame()
    job = _writer_job(mod, tmp_path, source_frame)
    coeff = np.array([np.log(2.0e-4)])
    source_before = job.source.read_bytes()

    summary = mod.rewrite_catalogue(job, coeff, chunk_size=1)

    output = pd.read_csv(job.output, comment="#")
    assert summary["rows"] == 2
    assert output.columns[-5:].tolist() == APERTURE_COLUMNS
    assert "q_from_mz" not in output
    assert output[["snap", "soap_index"]].equals(
        source_frame[["snap", "soap_index"]]
    )
    assert np.allclose(output["q_from_aperture"],
                       output["Y_500cyl_arcmin2"] / output["sigma_Y500_arcmin2"])
    assert job.source.read_bytes() == source_before
    header = "\n".join(mod.read_header(job.output))
    assert str(job.source) in header
    assert str(job.map_path) in header
    assert "# original catalogue" in header


def test_rewrite_catalogue_refuses_to_overwrite_without_force(mod, tmp_path):
    """An existing result is user data unless replacement is explicitly requested."""
    job = _writer_job(mod, tmp_path, _valid_frame())
    job.output.write_text("sentinel\n")

    with pytest.raises(FileExistsError, match=job.output.name):
        mod.rewrite_catalogue(job, np.array([0.0]))

    assert job.output.read_text() == "sentinel\n"


def test_rewrite_catalogue_can_write_staging_without_touching_canonical(mod, tmp_path):
    """Staged aperture work must not replace a canonical predecessor early."""
    job = _writer_job(mod, tmp_path, _valid_frame())
    job.output.write_text("sentinel\n")
    staged = tmp_path / "staging/source_qfrommap.csv"

    summary = mod.rewrite_catalogue(
        job,
        np.array([np.log(2.0e-4)]),
        chunk_size=1,
        output=staged,
    )

    assert Path(summary["output"]) == staged
    assert staged.is_file()
    assert job.output.read_text() == "sentinel\n"
    assert pd.read_csv(staged, comment="#").columns[-5:].tolist() == APERTURE_COLUMNS


def test_rewrite_catalogue_failure_preserves_output_and_removes_temporary(mod, tmp_path):
    """A late invalid chunk must never expose a partial or clobbered catalogue."""
    frame = _valid_frame()
    frame.loc[1, "z"] = np.nan
    job = _writer_job(mod, tmp_path, frame)
    job.output.write_text("sentinel\n")

    with pytest.raises(ValueError, match="column z"):
        mod.rewrite_catalogue(job, np.array([0.0]), chunk_size=1, force=True)

    assert job.output.read_text() == "sentinel\n"
    assert list(job.output.parent.glob(f"{job.output.name}.tmp-*")) == []


def test_main_dry_run_lists_jobs_without_loading_noise(mod, tmp_path, monkeypatch, capsys):
    """Manifest inspection must not perform expensive reads or writes."""
    source, map_path = _l1_job(tmp_path)
    monkeypatch.setattr(
        mod,
        "fit_sigma_y500",
        lambda *args, **kwargs: pytest.fail("dry-run loaded noise"),
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["compute_qfrommap_catalogues.py", "--flamingo-root", str(tmp_path), "--dry-run"],
    )

    assert mod.main() == 0

    output = capsys.readouterr().out
    assert str(source) in output
    assert str(map_path) in output
    assert "_qfrommap.csv" in output
    assert not source.with_name(source.name.replace("_qfrommz", "_qfrommap")).exists()


def test_main_passes_selection_and_writer_options(mod, tmp_path, monkeypatch):
    """CLI controls must reach discovery and every selected writer unchanged."""
    source, map_path = _l2_job(tmp_path, 1)
    job = mod.CatalogueJob("l2", "lightcone1", source, map_path, tmp_path / "out.csv")
    calls = []

    def fake_discover(root, *, dataset, only):
        calls.append(("discover", root, dataset, only))
        return [job]

    coeff = np.array([3.0, 2.0, 1.0])
    monkeypatch.setattr(mod, "discover_jobs", fake_discover)
    monkeypatch.setattr(mod, "fit_sigma_y500", lambda *args, **kwargs: coeff, raising=False)

    def fake_rewrite(actual_job, actual_coeff, *, chunk_size, force):
        calls.append(("rewrite", actual_job, actual_coeff, chunk_size, force))
        return {
            "label": actual_job.label,
            "rows": 2,
            "zero_pixels": 0,
            "q_counts": {1: 2, 5: 1, 10: 0, 20: 0, 50: 0},
            "seconds": 0.1,
        }

    monkeypatch.setattr(mod, "rewrite_catalogue", fake_rewrite)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_qfrommap_catalogues.py",
            "--flamingo-root",
            str(tmp_path),
            "--dataset",
            "l2",
            "--only",
            "lightcone1",
            "--only",
            "unused",
            "--chunk-size",
            "7",
            "--force",
        ],
    )

    assert mod.main() == 0
    assert calls[0] == ("discover", tmp_path, "l2", ("lightcone1", "unused"))
    assert calls[1][0:2] == ("rewrite", job)
    assert calls[1][2] is coeff
    assert calls[1][3:] == (7, True)
