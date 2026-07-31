from pathlib import Path

import healpy as hp
import numpy as np

from flamingo.catalogue.rebuild import (
    APERTURE_COLUMNS,
    L1_VARIANTS,
    CatalogueTarget,
    add_rotated_geometry,
    base_columns,
    build_snapshot_frame,
    catalogue_targets,
    q_columns,
    qmap_columns,
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
        self.soap_hbt = np.array([100, 200, 300])
        self.lightcone_hbt = np.array([300, 100, 300])
        self.hint = np.array([2, 0, 2]) if healthy_hint else np.array([0, 1, 0])

    def read_lightcone_identity(self, target, snap):
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

    def read_soap_fields(self, target, snap, rows):
        rows = np.asarray(rows)
        fields = {
            "is_central": np.array([True, True, True]),
            "m500": np.array([4_000.0, 2_000.0, 6_000.0]),
            "m200c": np.array([5_000.0, 3_000.0, 7_000.0]),
            "m200m": np.array([5_500.0, 3_500.0, 7_500.0]),
            "r500": np.array([0.8, 0.6, 1.1]),
            "r200c": np.array([1.0, 0.8, 1.32]),
            "r200m": np.array([1.1, 0.9, 1.43]),
            "y500": np.array([1.0, 2.0, 3.0]),
            "y500_noagn": np.array([0.9, 1.9, 2.9]),
            "y5r500": np.array([1.5, 2.5, 3.5]),
            "y5r500_noagn": np.array([1.4, 2.4, 3.4]),
        }
        return {name: values[rows] for name, values in fields.items()}


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
    assert np.allclose(frame["R_500c_Mpc"], 1.0)
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
    import pandas as pd

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
