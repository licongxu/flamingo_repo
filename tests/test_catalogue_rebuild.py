from pathlib import Path

from flamingo.catalogue.rebuild import (
    APERTURE_COLUMNS,
    L1_VARIANTS,
    base_columns,
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
