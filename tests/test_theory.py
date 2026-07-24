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
