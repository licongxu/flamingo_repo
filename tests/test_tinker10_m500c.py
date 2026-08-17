"""T10 / GNFW must take the input mass as M_500c. No silent 200c conversion."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from hmfast.halos import HaloModel
from hmfast.halos.mass_definition import MassDefinition, convert_m_delta
from flamingo.catalogue.frame import D3A_COSMOLOGY
from flamingo.inference.l1_m9 import L1M9CustomGNFWTheory
from flamingo.inference.masked_ps import MaskedTSZTheory
from flamingo.theory import clyy

REPO = Path(__file__).resolve().parents[1]
MASS_500C = MassDefinition(500, "critical")
MASS_200C = MassDefinition(200, "critical")


def _halo_model(mass_definition, convert_masses):
    return HaloModel(
        cosmology=D3A_COSMOLOGY,
        mass_definition=mass_definition,
        convert_masses=convert_masses,
        hm_consistency=False,
    )


def test_t10_ignores_convert_masses_at_m500c():
    mass = np.array([1.0e13, 1.0e14, 5.0e14])
    z = np.array([0.0])
    with_conv = _halo_model(MASS_500C, True)
    without = _halo_model(MASS_500C, False)
    b_on = np.asarray(with_conv.halo_bias.halo_bias(with_conv, mass, z))
    b_off = np.asarray(without.halo_bias.halo_bias(without, mass, z))
    np.testing.assert_allclose(b_on, b_off, rtol=1e-10, atol=0.0)


def test_convert_m_delta_is_identity_when_already_m500c():
    mass = np.array([1.0e14, 5.0e14])
    z = np.array([0.0, 0.5])
    halo_model = _halo_model(MASS_500C, True)
    concentration = halo_model.concentration.c_delta(halo_model, mass, z)
    converted = np.asarray(
        convert_m_delta(
            D3A_COSMOLOGY, mass, z, MASS_500C, MASS_500C, c_old=concentration
        )
    )
    expected = np.broadcast_to(mass[:, None], converted.shape)
    np.testing.assert_allclose(converted, expected, rtol=1e-10, atol=0.0)


def test_default_200c_would_shrink_an_m500c_input():
    """The trap: HaloModel defaults to 200c, so convert_m_delta treats M_500c as M_200c."""
    mass = np.array([1.0e14])
    z = np.array([0.0])
    halo_model = _halo_model(MASS_200C, True)
    concentration = halo_model.concentration.c_delta(halo_model, mass, z)
    treated_as_200c = np.asarray(
        convert_m_delta(
            D3A_COSMOLOGY, mass, z, MASS_200C, MASS_500C, c_old=concentration
        )
    )
    assert treated_as_200c.shape == (1, 1)
    assert 0.6 * mass[0] < treated_as_200c[0, 0] < 0.75 * mass[0]


def test_gnfw_keeps_m500c_when_halo_model_is_500c():
    mass = np.array([2.0e14])
    z = np.array([0.25])
    halo_model = _halo_model(MASS_500C, True)
    concentration = halo_model.concentration.c_delta(halo_model, mass, z)
    m500c = np.asarray(
        convert_m_delta(
            D3A_COSMOLOGY,
            mass,
            z,
            halo_model.mass_definition,
            MASS_500C,
            c_old=concentration,
        )
    )
    np.testing.assert_allclose(m500c[:, 0], mass, rtol=1e-10, atol=0.0)


def test_theory_paths_declare_m500c():
    sources = (
        Path(sys.modules[MaskedTSZTheory.__module__].__file__).read_text(),
        Path(sys.modules[L1M9CustomGNFWTheory.__module__].__file__).read_text(),
        Path(clyy.__file__).read_text(),
    )
    for source in sources:
        assert 'MassDefinition(500, "critical")' in source
        assert "convert_masses=True" in source


def test_bh_plot_script_uses_m500c_only():
    path = REPO / "scripts" / "plot_tinker10_vs_flamingo_bh.py"
    spec = importlib.util.spec_from_file_location("tinker10_bh_plot_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    source = path.read_text()
    assert module.MASS_500C.delta == 500
    assert module.MASS_500C.reference == "critical"
    assert "MassDefinition(200" not in source
    assert r"$M_{500c}" in source
