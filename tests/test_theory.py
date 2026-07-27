"""Smoke test for the hmfast halo-model C_ell^yy wrapper (loads emulators)."""
import numpy as np

from flamingo.theory import cl_yy, dl_of_cl


def test_cl_yy_smoke():
    ell = np.geomspace(100.0, 3000.0, 8)
    out = cl_yy(
        ell,
        m_grid=np.logspace(13.5, 15.0, 12),
        z_grid=np.geomspace(0.05, 1.5, 12),
    )
    assert set(out) == {"ell", "cl_1h", "cl_2h", "cl"}
    for key in ("cl_1h", "cl_2h", "cl"):
        assert out[key].shape == ell.shape
        assert np.all(np.isfinite(out[key]))
        assert np.all(out[key] > 0.0)
    assert np.allclose(out["cl"], out["cl_1h"] + out["cl_2h"])

    dl = dl_of_cl(out["ell"], out["cl"])
    assert np.all(dl > 0.0)


def test_default_grid_is_insensitive_to_the_consistency_counterterm():
    """The default mass grid must reach low enough to make the counterterm moot.

    hmfast dumps all the mass missing from ``m_grid`` onto a phantom population
    at ``m_grid[0]``, carrying that halo's pressure profile. On a grid starting
    at ``10^13 Msun`` this inflates ``C_ell^1h`` by a factor >3 at ``ell=6000``.
    On a converged grid the switch barely registers, which is what makes the
    prediction meaningful -- so guard it.
    """
    ell = np.geomspace(100.0, 1.0e4, 6)
    off = cl_yy(ell)["cl"]
    on = cl_yy(ell, hm_consistency=True)["cl"]
    assert np.all(np.abs(on / off - 1.0) < 0.01)
