from __future__ import annotations

import numpy as np

from flamingo.powerspectra.bandpowers import bin_log_dl, bin_planck_dl
from flamingo.powerspectra.namaster import decoupled_cl_per_ell


def test_planck_binning_preserves_existing_values() -> None:
    ell = np.arange(10_001, dtype=float)
    cl = 1.0 / (ell + 1.0) ** 2

    actual = bin_planck_dl(ell, cl)

    expected = np.array(
        [
            0.14518251175343727,
            0.14844884966995683,
            0.1509297027127849,
            0.15274753937578472,
            0.15414859272620732,
            0.15529301626552913,
            0.1561885431326356,
            0.1568672259326779,
            0.15739487966828117,
            0.15780361945688176,
            0.158115303043043,
            0.15835261243299747,
            0.1585367724012185,
            0.15867989959347437,
            0.1587894665195588,
            0.1588736182442935,
            0.1589384762386342,
            0.15898837797881615,
        ]
    )
    assert np.array_equal(actual, expected)


def test_log_binning_preserves_existing_values() -> None:
    ell = np.arange(10_001, dtype=float)
    cl = 1.0 / (ell + 1.0) ** 2

    centres, actual = bin_log_dl(ell, cl)

    expected_centres = np.array(
        [
            100.51835744633574,
            149.95576820477694,
            223.7077185616558,
            333.7326996032606,
            497.87068367863907,
            742.7357821433383,
            1108.0315836233383,
            1652.9888822158644,
            2465.969639416063,
            3678.79441171442,
            5488.116360940261,
            8187.307530779816,
        ]
    )
    expected = np.array(
        [
            0.15748731962953866,
            0.15811194018503108,
            0.15797127138716946,
            0.15855560666876192,
            0.15876634199058465,
            0.15877214311622212,
            0.1589322939629026,
            0.15907419768940279,
            0.15914783310428526,
            0.15912585730684056,
            0.15911022727432234,
            0.15911966082335088,
        ]
    )
    assert np.array_equal(centres, expected_centres)
    assert np.array_equal(actual, expected)


def test_per_ell_namaster_preserves_existing_values() -> None:
    ymap = np.sin(np.arange(12 * 4**2, dtype=float) / 13.0)
    mask = np.ones_like(ymap)
    mask[:10] = 0.0

    actual = decoupled_cl_per_ell(ymap, mask, np.ones(9), lmax=8)

    expected = np.array(
        [
            np.nan,
            np.nan,
            0.17107448002673953,
            0.00963864035116405,
            0.07520041781748443,
            0.08166438389059635,
            0.22278208975521985,
            0.04071502499240823,
            0.03738128357457864,
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0.0, equal_nan=True)
