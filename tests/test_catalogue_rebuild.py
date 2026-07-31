import json
import os
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
import pytest

from flamingo.catalogue.rebuild import (
    APERTURE_COLUMNS,
    L1_VARIANTS,
    CatalogueTarget,
    add_rotated_geometry,
    base_columns,
    build_base_catalogue,
    build_snapshot_frame,
    catalogue_targets,
    derive_q_catalogues,
    publish_file,
    q_columns,
    qmap_columns,
    validate_catalogue,
)


def test_catalogue_targets_cover_nine_l1_and_eight_l2_families(tmp_path):
    """Dropping a run would leave a stale canonical catalogue in production."""
    targets = catalogue_targets(tmp_path)

    assert len(targets) == 17
    assert sum(target.family == "l1" for target in targets) == 9
    assert sum(target.family == "l2" for target in targets) == 8
    assert {target.variant for target in targets if target.family == "l1"} == set(
        L1_VARIANTS
    )
    assert {target.lightcone for target in targets if target.family == "l2"} == set(
        range(8)
    )
    assert len({path for target in targets for path in target.canonical_csvs}) == 68


def test_catalogue_targets_preserve_canonical_paths_and_map_pairing(tmp_path):
    """A path or map mismatch would silently measure a different sky realisation."""
    targets = catalogue_targets(tmp_path)
    l1 = next(target for target in targets if target.key == "L1_m9/L1_m9")
    l2 = next(target for target in targets if target.key == "L2p8_m9/lightcone7")

    assert l1.catalogue_dir == tmp_path / "L1_m9/catalogues"
    assert l1.map_path == tmp_path / "L1_m9/maps/y_unlensed_L1_m9_lc0_nside4096.fits"
    assert l2.catalogue_dir == tmp_path / "L2p8_m9/lightcone7/catalogues"
    assert l2.map_path == (
        tmp_path / "L2p8_m9/lightcone7/healpix_map/y_unlensed_L2p8_m9_lc7.fits"
    )
    assert [path.name for path in l2.canonical_csvs] == [
        "halo_catalogue_M500c_1e13_zlt3_L2p8_m9_yang26rot.csv",
        "halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_qfrommz.csv",
        "halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_qfrommz_alpha_fixed_1p12.csv",
        "halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_qfrommap.csv",
    ]


def test_family_schemas_match_existing_reader_contracts():
    """Changing column order would break downstream streaming readers."""
    for family, base_count, q_count, qmap_count in (
        ("l1", 29, 30, 34),
        ("l2", 25, 26, 30),
    ):
        base = base_columns(family)
        q = q_columns(family)
        qmap = qmap_columns(family)

        assert len(base) == base_count
        assert len(q) == q_count
        assert len(qmap) == qmap_count
        assert q == (*base, "q_from_mz")
        assert qmap == (*base, *APERTURE_COLUMNS)
        assert base[:2] == ("snap", "soap_index")
        assert base[19:25] == (
            "M_500c_Msun",
            "M_200c_Msun",
            "M_200m_Msun",
            "R_500c_Mpc",
            "R_200c_Mpc",
            "R_200m_Mpc",
        )

    assert base_columns("l1")[-4:] == (
        "Y_500c_Mpc2",
        "Y_500c_noAGN_Mpc2",
        "Y_5R500c_Mpc2",
        "Y_5R500c_noAGN_Mpc2",
    )


def test_catalogue_targets_are_root_relative(tmp_path):
    """Tests and staging must not accidentally fall back to the production root."""
    assert all(
        Path(target.catalogue_dir).is_relative_to(tmp_path)
        for target in catalogue_targets(tmp_path)
    )


class FakeSnapshotSource:
    def __init__(self, *, healthy_hint: bool = False):
        self.healthy_hint = healthy_hint
        self.full_identity_reads = 0
        self.identity_reads = 0
        self.property_rows = None
        self.soap_hbt = np.array([100, 200, 300])
        self.lightcone_hbt = np.array([300, 100, 300])
        self.hint = np.array([2, 0, 2]) if healthy_hint else np.array([0, 1, 0])

    def read_lightcone_identity(self, target, snap):
        self.identity_reads += 1
        return self.lightcone_hbt.copy(), self.hint.copy()

    def read_soap_identity(self, target, snap, rows=None):
        if rows is None:
            self.full_identity_reads += 1
            return self.soap_hbt.copy()
        return self.soap_hbt[np.asarray(rows)]

    def read_lightcone_fields(self, target, snap, rows):
        rows = np.asarray(rows)
        redshift = np.array([0.1, 0.2, 0.1])
        position = np.array([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0], [4.0, 5.0, 6.0]])
        return redshift[rows], position[rows]

    def read_soap_scale_factor(self, target, snap):
        return 1.0 / 1.1

    def read_soap_fields(self, target, snap, rows):
        rows = np.asarray(rows)
        fields = {
            "is_central": np.array([True, True, True]),
            "m500": np.array([4_000.0, 2_000.0, 6_000.0], dtype=np.float32),
            "m200c": np.array([5_000.0, 3_000.0, 7_000.0], dtype=np.float32),
            "m200m": np.array([5_500.0, 3_500.0, 7_500.0], dtype=np.float32),
            "r500": np.array([0.8, 0.6, 1.1], dtype=np.float32),
            "r200c": np.array([1.0, 0.8, 1.32], dtype=np.float32),
            "r200m": np.array([1.1, 0.9, 1.43], dtype=np.float32),
            "y500": np.array([1.0, 2.0, 3.0]),
            "y500_noagn": np.array([0.9, 1.9, 2.9]),
            "y5r500": np.array([1.5, 2.5, 3.5]),
            "y5r500_noagn": np.array([1.4, 2.4, 3.4]),
        }
        return {name: values[rows] for name, values in fields.items()}

    def read_soap_selection_fields(self, target, snap, rows):
        fields = self.read_soap_fields(target, snap, rows)
        return {name: fields[name] for name in ("is_central", "m500")}

    def read_soap_property_fields(self, target, snap, rows):
        self.property_rows = np.asarray(rows).copy()
        fields = self.read_soap_fields(target, snap, rows)
        return {
            name: values
            for name, values in fields.items()
            if name not in {"is_central", "m500"}
        }


def _target(family="l1"):
    return CatalogueTarget(
        family=family,
        run="L1_m9" if family == "l1" else "L2p8_m9",
        variant="L1_m9" if family == "l1" else "L2p8_m9",
        lightcone=0,
        catalogue_dir=Path("/catalogues"),
        map_path=Path("/map.fits"),
    )


def test_snapshot_frame_falls_back_from_stale_hint_and_pairs_current_soap():
    """Trusting a stale L1 hint would attach the wrong mass to both sky copies."""
    source = FakeSnapshotSource(healthy_hint=False)

    frame = build_snapshot_frame(source, _target("l1"), 75, 5.0e13)

    assert source.full_identity_reads == 1
    assert source.property_rows.tolist() == [2]
    assert frame.columns.tolist() == [
        "snap",
        "soap_index",
        "z",
        "x_Mpc",
        "y_Mpc",
        "z_Mpc",
        "M_500c_Msun",
        "M_200c_Msun",
        "M_200m_Msun",
        "R_500c_Mpc",
        "R_200c_Mpc",
        "R_200m_Mpc",
        "Y_500c_Mpc2",
        "Y_500c_noAGN_Mpc2",
        "Y_5R500c_Mpc2",
        "Y_5R500c_noAGN_Mpc2",
    ]
    assert frame["soap_index"].tolist() == [2, 2]
    assert frame[["x_Mpc", "y_Mpc", "z_Mpc"]].to_numpy().tolist() == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
    assert frame["M_500c_Msun"].tolist() == [6.0e13, 6.0e13]
    expected_radius = float(np.float32(1.1)) / 1.1
    assert frame["R_500c_Mpc"].tolist() == [expected_radius, expected_radius]
    assert frame["Y_500c_Mpc2"].tolist() == [3.0, 3.0]


def test_snapshot_frame_uses_only_identity_verified_hint_for_l2():
    """A healthy row hint is a safe fast path only after stable-ID verification."""
    source = FakeSnapshotSource(healthy_hint=True)

    frame = build_snapshot_frame(source, _target("l2"), 76, 5.0e13)

    assert source.full_identity_reads == 0
    assert len(frame) == 2
    assert frame.columns.tolist() == [
        "snap",
        "soap_index",
        "z",
        "x_Mpc",
        "y_Mpc",
        "z_Mpc",
        "M_500c_Msun",
        "M_200c_Msun",
        "M_200m_Msun",
        "R_500c_Mpc",
        "R_200c_Mpc",
        "R_200m_Mpc",
    ]


def test_snapshot_frame_rejects_a_noncentral_massive_object():
    """Including a massive satellite would violate the catalogue selection."""
    source = FakeSnapshotSource(healthy_hint=True)
    original = source.read_soap_fields

    def noncentral(target, snap, rows):
        fields = original(target, snap, rows)
        fields["is_central"][:] = False
        return fields

    source.read_soap_fields = noncentral

    assert build_snapshot_frame(source, _target("l1"), 75, 5.0e13).empty


def test_add_rotated_geometry_matches_official_inverse_rotator():
    """Swapping theta/phi or dropping inv=True would move every map aperture."""
    frame = {
        "snap": [75],
        "soap_index": [2],
        "z": [0.1],
        "x_Mpc": [1.0],
        "y_Mpc": [0.0],
        "z_Mpc": [0.0],
        "M_500c_Msun": [6.0e13],
        "M_200c_Msun": [7.0e13],
        "M_200m_Msun": [7.5e13],
        "R_500c_Mpc": [1.0],
        "R_200c_Mpc": [1.2],
        "R_200m_Mpc": [1.3],
        "Y_500c_Mpc2": [3.0],
        "Y_500c_noAGN_Mpc2": [2.9],
        "Y_5R500c_Mpc2": [3.5],
        "Y_5R500c_noAGN_Mpc2": [3.4],
    }
    frame = pd.DataFrame(frame)
    angles = np.array([[0.3], [0.4]])

    output = add_rotated_geometry(frame, np.array([2.0]), angles, "l1")
    expected_theta, expected_phi = hp.Rotator(
        rot=[np.degrees(0.4), np.degrees(0.3)], inv=True
    )(np.pi / 2.0, 0.0)

    assert output.columns.tolist() == list(base_columns("l1"))
    assert output.loc[0, "shell_idx"] == 0
    assert np.isclose(output.loc[0, "theta_rot_rad"], expected_theta)
    assert np.isclose(output.loc[0, "phi_rot_rad"], expected_phi)
    assert np.isclose(output.loc[0, "x_rot_Mpc"], np.sin(expected_theta) * np.cos(expected_phi))


def test_base_writer_resumes_from_verified_snapshot_parts(tmp_path):
    """A retry must not redownload snapshots whose atomic parts already validate."""
    source = FakeSnapshotSource(healthy_hint=True)
    staged = tmp_path / "catalogues/base.csv"
    target = _target("l2")

    first = build_base_catalogue(
        target,
        staged,
        source,
        snaps=(75, 76),
        shell_radii_mpc=np.array([20.0]),
        angles=np.zeros((2, 1)),
        mass_cut_msun=5.0e13,
    )
    parts = sorted((staged.parent / f".{staged.name}.parts").glob("snap_*.csv"))
    part_bytes = [part.read_bytes() for part in parts]
    staged.unlink()
    source.identity_reads = 0

    second = build_base_catalogue(
        target,
        staged,
        source,
        snaps=(75, 76),
        shell_radii_mpc=np.array([20.0]),
        angles=np.zeros((2, 1)),
        mass_cut_msun=5.0e13,
    )

    assert first["rows"] == second["rows"] == 4
    assert source.identity_reads == 0
    assert [part.read_bytes() for part in parts] == part_bytes
    assert staged.is_file()
    assert pd.read_csv(staged, comment="#").columns.tolist() == list(base_columns("l2"))


class FakeScaling:
    def q(self, mass, redshift, *, index):
        return np.asarray(mass) / 1.0e13 + np.asarray(redshift)


def test_derive_q_selects_strict_m500_cut_and_preserves_family_schema(tmp_path):
    """Selecting the threshold row or changing schema would alter CNC counts/readers."""
    base = tmp_path / "base.csv"
    frame = pd.DataFrame(
        {
            name: np.zeros(2)
            for name in base_columns("l2")
        }
    )
    frame["snap"] = [75, 76]
    frame["soap_index"] = [7, 9]
    frame["z"] = [0.1, 0.2]
    frame["M_500c_Msun"] = [5.0e13, 6.0e13]
    frame.to_csv(base, index=False)
    outputs = (tmp_path / "q.csv", tmp_path / "q_alpha.csv")

    summary = derive_q_catalogues(
        base,
        outputs,
        "l2",
        FakeScaling(),
        chunk_size=1,
    )

    assert summary["rows"] == 1
    for output in outputs:
        result = pd.read_csv(output, comment="#")
        assert result.columns.tolist() == list(q_columns("l2"))
        assert result["soap_index"].tolist() == [9]
        assert result["q_from_mz"].tolist() == [6.2]


def _valid_catalogue_frame(family="l2", flavour="base"):
    columns = {
        "base": base_columns,
        "q": q_columns,
        "q_alpha": q_columns,
        "qmap": qmap_columns,
    }[flavour](family)
    frame = pd.DataFrame({name: np.ones(2) for name in columns})
    frame["snap"] = [75, 76]
    frame["soap_index"] = [7, 9]
    frame["z"] = [0.1, 0.2]
    frame["M_500c_Msun"] = [6.0e13, 7.0e13]
    if "q_from_mz" in frame:
        frame["q_from_mz"] = [5.0, 6.0]
    if "npix_in_aperture" in frame:
        frame["npix_in_aperture"] = [0, 2]
        frame["q_from_aperture"] = [0.0, 3.0]
    return frame


def test_validate_catalogue_reports_identity_digest_and_allows_zero_aperture(tmp_path):
    """Zero-pixel apertures are permitted, while row identity must stay auditable."""
    path = tmp_path / "qmap.csv"
    _valid_catalogue_frame(flavour="qmap").to_csv(path, index=False)

    summary = validate_catalogue(path, _target("l2"), "qmap", chunk_size=1)

    assert summary["rows"] == 2
    assert len(summary["identity_sha256"]) == 64
    assert summary["selected_rows"] == 2


def test_validate_catalogue_rejects_wrong_schema_before_publication(tmp_path):
    """A missing field must block publication rather than break readers later."""
    path = tmp_path / "bad.csv"
    _valid_catalogue_frame().drop(columns="R_500c_Mpc").to_csv(path, index=False)

    with pytest.raises(ValueError, match="schema"):
        validate_catalogue(path, _target("l2"), "base")


def test_publish_failure_rolls_archived_file_back(tmp_path, monkeypatch):
    """A failed staged install must restore the predecessor at its canonical path."""
    staged = tmp_path / "stage/catalogue.csv"
    canonical = tmp_path / "canonical/catalogue.csv"
    archive = tmp_path / "archive/canonical/catalogue.csv"
    manifest = tmp_path / "archive/manifest.json"
    staged.parent.mkdir(parents=True)
    canonical.parent.mkdir(parents=True)
    staged.write_text("new\n")
    canonical.write_text("old\n")
    real_replace = os.replace
    calls = 0

    def fail_second_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected staged install failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="injected"):
        publish_file(staged, canonical, archive, manifest)

    assert canonical.read_text() == "old\n"
    assert staged.read_text() == "new\n"
    assert not archive.exists()
    assert not manifest.exists()


def test_publish_archives_old_and_installs_new_at_same_name(tmp_path):
    """Successful publication must preserve the canonical basename and old bytes."""
    staged = tmp_path / "stage/catalogue.csv"
    canonical = tmp_path / "canonical/catalogue.csv"
    archive = tmp_path / "archive/canonical/catalogue.csv"
    manifest = tmp_path / "archive/manifest.json"
    staged.parent.mkdir(parents=True)
    canonical.parent.mkdir(parents=True)
    staged.write_text("new\n")
    canonical.write_text("old\n")

    record = publish_file(staged, canonical, archive, manifest)

    assert canonical.name == "catalogue.csv"
    assert canonical.read_text() == "new\n"
    assert archive.read_text() == "old\n"
    assert record["canonical"] == str(canonical)
    assert json.loads(manifest.read_text())["files"][str(canonical)]["status"] == "published"
