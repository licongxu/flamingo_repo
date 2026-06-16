"""Build a derived catalogue with a simple-GNFW (B=1) Compton-y0 and SNR.

Starting from the raw SOAP catalogue (only hdfstream-fetched quantities), this
adds, for every cluster:

    y0   = K * Y_5R500c_Mpc2 / R_500c_Mpc**2
    theta500_arcmin = R_500c / D_A(z)            (no hydrostatic bias, B = 1)
    sigma_y0(theta500)  from the szifi noise file (immf6, sky-averaged)
    q    = y0 / sigma_y0

``K`` is the geometric Y->y0 factor of the Arnaud/Nagai GNFW *shape* used by
hmfast's ``GNFWPressureProfile`` (B only enters the normalisation, so K is
B-independent; K = 1.563860). The Battaglia 2012 profile is intentionally
ignored.

Run:
    JAX_PLATFORMS=cpu python scripts/build_y0q_catalogue.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from hmfast.halos.profiles.pressure import GNFWPressureProfile  # noqa: E402
from flamingo.catalogue import frame  # noqa: E402

RAW = _REPO / "data/hydro_L2p8m9/catalogue/halo_catalogue_M500c_5e13_zlt3_soap_hdfstream_raw.csv"
OUT = _REPO / "data/hydro_L2p8m9/catalogue/halo_catalogue_M500c_5e13_zlt3_y0q_arnaudB1.csv"
NOISE_DIR = _REPO / "data/noise"

R_OUT = 5.0  # line-of-sight / aperture truncation in units of R_500c
ARCMIN_PER_RAD = 180.0 / np.pi * 60.0
THETA_MIN, THETA_MAX = 0.5, 32.0  # szifi noise-curve validity range [arcmin]


def gnfw_K(r_out: float = R_OUT) -> float:
    """Geometric factor K so that y0 = K * Y(<r_out R500) / R_500c**2.

    K = (2 int_0^r_out p(s) ds) / (4 pi int_0^r_out p(x) x^2 dx) for the
    dimensionless hmfast GNFW shape p(x) (B = 1; shape is B-independent).
    """
    prof = GNFWPressureProfile(B=1.0)
    P0, c500, alpha, beta, gamma = prof.P0, prof.c500, prof.alpha, prof.beta, prof.gamma

    def p(x: float) -> float:
        xc = c500 * max(x, 1e-15)
        return P0 * xc ** (-gamma) * (1.0 + xc ** alpha) ** ((gamma - beta) / alpha)

    i_los = 2.0 * quad(lambda s: p(s), 0.0, r_out, limit=400)[0]
    i_3d = quad(lambda x: p(x) * x * x, 0.0, r_out, limit=400)[0]
    return float(i_los / (4.0 * np.pi * i_3d))


def sigma_y0_curve(noise_dir: Path = NOISE_DIR, filter_name: str = "immf6", deg: int = 3):
    """Polynomial (in log-log) for sky-averaged sigma_y0(theta_arcmin)."""
    sigma_obj = np.load(noise_dir / "sigma_dict_szifi.npy", allow_pickle=True).item()
    skyfracs = np.load(noise_dir / "skyfracs_szifi_cosmology.npy").ravel()
    data = sigma_obj[filter_name]
    n_theta = len(np.asarray(next(iter(data.values()))))
    theta_grid = np.exp(np.linspace(np.log(THETA_MIN), np.log(THETA_MAX), n_theta))
    num = np.zeros(n_theta)
    den = 0.0
    for tile, arr in data.items():
        w = skyfracs[int(tile)]
        num += w * np.asarray(arr, dtype=float)
        den += w
    sigma_skyavg = num / max(den, 1e-300)
    coeff = np.polyfit(np.log(theta_grid), np.log(sigma_skyavg), deg=deg)

    def sigma(theta_arcmin: np.ndarray) -> np.ndarray:
        th = np.clip(np.asarray(theta_arcmin, dtype=float), THETA_MIN, THETA_MAX)
        return np.exp(np.polyval(coeff, np.log(th)))

    return sigma


def main() -> None:
    K = gnfw_K()
    print(f"GNFW geometric constant K = {K:.6f}  (y0 = K * Y_5R500c / R_500c**2)")

    df = pd.read_csv(RAW, comment="#")
    z = df["z"].to_numpy(float)
    R500 = df["R_500c_Mpc"].to_numpy(float)
    Y = df["Y_5R500c_Mpc2"].to_numpy(float)

    y0 = np.where((Y > 0) & (R500 > 0), K * Y / (R500 * R500), np.nan)

    # theta_500 with no hydrostatic bias (B = 1), hmfast D3A angular distance.
    theta500_arcmin = frame.theta_500(R500, z) * ARCMIN_PER_RAD

    sigma = sigma_y0_curve()
    sigma_y0 = sigma(theta500_arcmin)
    q = np.where((sigma_y0 > 0) & np.isfinite(y0), y0 / sigma_y0, np.nan)

    out = pd.DataFrame(
        {
            "snap": df["snap"].to_numpy(),
            "z": z,
            "M_500c_Msun": df["M_500c_Msun"].to_numpy(float),
            "R_500c_Mpc": R500,
            "Y_5R500c_Mpc2": Y,
            "theta500_arcmin": theta500_arcmin,
            "y0": y0,
            "sigma_y0": sigma_y0,
            "q": q,
        }
    )

    header = (
        "# FLAMINGO L2p8_m9 lightcone0: derived Compton-y0 and SNR (simple GNFW, B=1).\n"
        f"# y0 = {K:.6f} * Y_5R500c_Mpc2 / R_500c_Mpc**2  (Arnaud/Nagai GNFW shape via\n"
        "#   hmfast GNFWPressureProfile, R_OUT = 5 R_500c, no beam, B=1; B12 ignored).\n"
        "# theta500_arcmin = R_500c / D_A(z) [hmfast D3A cosmology, no bias].\n"
        "# sigma_y0 from szifi noise file (immf6, sky-averaged), looked up at theta500.\n"
        "# q = y0 / sigma_y0.  Built from halo_catalogue_M500c_5e13_zlt3_soap_hdfstream_raw.csv.\n"
    )
    with open(OUT, "w") as fh:
        fh.write(header)
        out.to_csv(fh, index=False, float_format="%.6e")

    print(f"rows={len(out)}  median y0={np.nanmedian(y0):.3e}  median q={np.nanmedian(q):.3f}")
    print(f"q>5: {int(np.nansum(q > 5))}   q>4.5: {int(np.nansum(q > 4.5))}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
