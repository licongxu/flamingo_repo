from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from flamingo.powerspectra.q_selection import resolve_q_selection


REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_qfrommap_selection_uses_corrected_l1_and_canonical_l2_catalogues():
    selection = resolve_q_selection("qfrommap")

    assert selection.tag == "qfrommap"
    assert selection.q_column == "q_from_aperture"
    assert selection.l1_catalogue_dir == Path(
        "/rds/rds-lxu/flamingo/.hbt_join_fix_staging/20260731/"
        "L1_m9/catalogues"
    )
    assert selection.l2_root == Path("/rds/rds-lxu/flamingo/L2p8_m9")


def test_legacy_selection_defaults_are_unchanged():
    selection = resolve_q_selection("qfrommz_alpha_fixed_1p12")

    assert selection.tag == "qfrommz_alpha_fixed_1p12"
    assert selection.q_column == "q_from_mz"
    assert selection.l1_catalogue_dir == Path(
        "/rds/rds-lxu/flamingo/L1_m9/catalogues"
    )


def test_feedback_catalogue_loader_selects_requested_q_column(tmp_path):
    module = _load_script(
        "feedback_bandpowers_q_selection_test",
        "scripts/compute_l1_m9_feedback_bandpowers.py",
    )
    rows = 10_000
    path = tmp_path / "catalogue.csv"
    pd.DataFrame(
        {
            "z": np.full(rows, 0.2),
            "R_500c_Mpc": np.full(rows, 1.0),
            "theta_rot_rad": np.full(rows, 1.0),
            "phi_rot_rad": np.full(rows, 2.0),
            "theta_nat_rad": np.full(rows, 1.1),
            "phi_nat_rad": np.full(rows, 2.1),
            "q_from_mz": np.full(rows, 3.0),
            "q_from_aperture": np.full(rows, 7.0),
        }
    ).to_csv(path, index=False)

    legacy = module.load_catalogue(path)
    aperture = module.load_catalogue(path, q_column="q_from_aperture")

    assert np.all(legacy["q"] == 3.0)
    assert np.all(aperture["q"] == 7.0)


def test_l1_multi_q_paths_are_selection_tagged_and_use_corrected_catalogue():
    module = _load_script(
        "l1_multi_q_selection_test",
        "scripts/compute_l1_m9_feedback_bandpowers.py",
    )
    selection = resolve_q_selection("qfrommap")

    catalogue = module.catalogue_path("L1_m9", selection)
    output_18, output_log = module.out_paths(
        "fiducial", "qgt5", selection.tag
    )

    assert catalogue == selection.l1_catalogue_dir / (
        "halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommap.csv"
    )
    assert output_18.name == "Dl_yy_L1_m9_masked_qgt5_qfrommap_binned_18.txt"
    assert output_log.name.endswith("qfrommap_logbins_dln0p4_lmax10000.txt")


def test_l2_paths_are_selection_tagged_and_use_canonical_qfrommap_catalogue():
    module = _load_script(
        "l2_multi_q_selection_test",
        "scripts/compute_l2p8_m9_masked_ps_alpha_fixed_1p12.py",
    )
    selection = resolve_q_selection("qfrommap")

    catalogue = module.cat_file(3, selection)
    output_18, output_log = module.masked_out_paths(3, "qgt5", selection.tag)

    assert catalogue == selection.l2_root / "lightcone3/catalogues" / (
        "halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_qfrommap.csv"
    )
    assert output_18.name == (
        "Dl_yy_L2p8_m9_lc3_masked_qgt5_qfrommap_binned_18.txt"
    )
    assert output_log.name.endswith("qfrommap_logbins_dln0p4_lmax10000.txt")


def test_qfrommap_defaults_exclude_legacy_q3_cut():
    module = _load_script(
        "l1_multi_q_default_cuts_test",
        "scripts/compute_l1_m9_feedback_bandpowers.py",
    )

    assert module.default_q_cuts("qfrommap") == [50.0, 20.0, 10.0, 5.0, 1.0]
    assert module.default_q_cuts("qfrommz_alpha_fixed_1p12") == [
        50.0,
        20.0,
        10.0,
        5.0,
        3.0,
        1.0,
    ]


def test_comparison_plot_uses_l2_lightcone0_qfrommap_product(tmp_path):
    module = _load_script(
        "masked_ps_plot_selection_test",
        "scripts/plot_l1_l2p8_m9_masked_ps_comparison.py",
    )

    path = module._paths(
        "L2p8_m9_lc0",
        "qgt5",
        log=False,
        selection_tag="qfrommap",
        data=tmp_path,
    )

    assert path == tmp_path / (
        "Dl_yy_L2p8_m9_lc0_masked_qgt5_qfrommap_binned_18.txt"
    )


def test_standalone_l2_plot_routes_qfrommap_to_lightcone0(tmp_path):
    module = _load_script(
        "standalone_l2_qfrommap_input_test",
        "scripts/plot_l2p8_m9_masked_ps_alpha_fixed_1p12.py",
    )

    path = module._masked_paths(
        "qgt5", log=False, selection_tag="qfrommap", data=tmp_path
    )

    assert path == tmp_path / (
        "Dl_yy_L2p8_m9_lc0_masked_qgt5_qfrommap_binned_18.txt"
    )


def test_standalone_l2_plot_tags_qfrommap_output():
    module = _load_script(
        "standalone_l2_qfrommap_output_test",
        "scripts/plot_l2p8_m9_masked_ps_alpha_fixed_1p12.py",
    )

    assert module.output_stem(log=True, selection_tag="qfrommap").name == (
        "l2p8_m9_masked_ps_logbins_qfrommap"
    )
