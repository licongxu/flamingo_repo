from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from flamingo.powerspectra.q_selection import resolve_q_selection

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str, filename: str):
    path = REPO / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_l2_qgt6_uses_lightcone0_canonical_catalogue():
    module = _load_script(
        "l2_qgt6_qfrommap_test",
        "compute_l2p8_m9_masked_ps_alpha_fixed_1p12.py",
    )
    selection = resolve_q_selection("qfrommap")

    map_path = module.map_file(0)
    catalogue_path = module.cat_file(0, selection)

    assert map_path == Path(
        "/rds/rds-lxu/flamingo/L2p8_m9/lightcone0/healpix_map/" "y_unlensed_L2p8_m9_lc0.fits"
    )
    assert catalogue_path == selection.l2_root / "lightcone0/catalogues" / (
        "halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_qfrommap.csv"
    )
    assert module.cut_tags([6.0, 0.5]) == ["qgt6", "qgt0p5"]


def test_l1_qfrommap_null_test_uses_corrected_catalogue_and_isolated_cache():
    module = _load_script("null_test_qfrommap_test", "masking_radius_null_test.py")
    selection = resolve_q_selection("qfrommap")

    catalogue_path = module.catalogue_path("L1_m9", selection)
    cache_path = module.point_path("L1_m9", 5.0, 4.0, selection_tag="qfrommap")

    assert catalogue_path == selection.l1_catalogue_dir / (
        "halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommap.csv"
    )
    assert cache_path == REPO / "data_paper/masking_radius_null_test/qfrommap" / (
        "L1_m9_qgt5_r4.npz"
    )
    assert module.q_column("qfrommap") == "q_from_aperture"


def test_legacy_null_test_cache_location_is_unchanged():
    module = _load_script("null_test_legacy_path_test", "masking_radius_null_test.py")

    assert module.point_path("L1_m9", 5.0, 4.0) == REPO / (
        "data_paper/masking_radius_null_test/L1_m9_qgt5_r4.npz"
    )


def test_null_test_plot_routes_qfrommap_products_without_overwriting_legacy():
    module = _load_script("null_test_qfrommap_plot_test", "plot_masking_radius_null_test.py")

    assert module.selection_data_dir("qfrommap") == REPO / (
        "data_paper/masking_radius_null_test/qfrommap"
    )
    assert module.product_path("masking_radius_null_test", "qfrommap") == REPO / (
        "figures/masking_radius_null_test/masking_radius_null_test_qfrommap"
    )
    assert module.product_path("masking_radius_null_test", "qfrommz_alpha_fixed_1p12") == REPO / (
        "figures/masking_radius_null_test/masking_radius_null_test"
    )


def test_incremental_multi_q_metadata_preserves_existing_cuts():
    module = _load_script(
        "multi_q_metadata_merge_test",
        "compute_l1_m9_feedback_bandpowers.py",
    )
    existing = {
        "q_cuts": [50.0, 20.0, 10.0, 5.0, 1.0],
        "variants": {"fiducial": {"variant": "fiducial", "cuts": {"qgt5": {"n_masked": 3}}}},
    }
    incremental = {
        "q_cuts": [6.0, 3.0],
        "variants": {"fiducial": {"variant": "fiducial", "cuts": {"qgt3": {"n_masked": 9}}}},
    }

    merged = module.merge_metadata_payload(existing, incremental)

    assert merged["q_cuts"] == [50.0, 20.0, 10.0, 5.0, 1.0, 6.0, 3.0]
    assert set(merged["variants"]["fiducial"]["cuts"]) == {"qgt5", "qgt3"}
