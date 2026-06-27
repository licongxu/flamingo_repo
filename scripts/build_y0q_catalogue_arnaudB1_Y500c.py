"""Build B=1 SNR catalogue using SOAP Y_500c (GNFW K at r_out=1).

Same conventions as ``build_y0q_catalogue.py`` (B=1, no hydrostatic bias),
but uses SOAP Compton-Y within R_500c and K(r_out=1).

Run:
    JAX_PLATFORMS=cpu python scripts/build_y0q_catalogue_arnaudB1_Y500c.py
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
OUT = _REPO / "data/hydro_L2p8m9/catalogue/halo_catalogue_M500c_5e13_zlt3_y0q_arnaudB1_Y500c.csv"
NOISE_DIR = _REPO / "data/noise"

R_OUT = 1.0
Y_COL = "Y_500c_Mpc2"
ARCMIN_PER_RAD = 180.0 / np.pi * 60.0
THETA_MIN, THETA_MAX = 0.5, 32.0


def gnfw_K(r_out: float = R_OUT) -> float:
    prof = GNFWPressureProfile(B=1.0)
    P0, c500, alpha, beta, gamma = prof.P0, prof.c500, prof.alpha, prof.beta, prof.gamma

    def p(x: float) -> float:
        xc = c500 * max(x, 1e-15)
        return P0 * xc ** (-gamma) * (1.0 + xc ** alpha) ** ((gamma - beta) / alpha)

    i_los = 2.0 * quad(lambda s: p(s), 0.0, r_out, limit=400)[0]
    i_3d = quad(lambda x: p(x) * x * x, 0.0, r_out, limit=400)[0]
    return float(i_los / (4.0 * np.pi * i_3d))


def sigma_y0_curve(noise_dir: Path = NOISE_DIR, filter_name: str = "immf6", deg: int = 3):
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
    print(f"GNFW K(r_out={R_OUT}) = {K:.6f}  (y0 = K * {Y_COL} / R_500c**2)")

    df = pd.read_csv(RAW, comment="#")
    z = df["z"].to_numpy(float)
    R500 = df["R_500c_Mpc"].to_numpy(float)
    Y = df[Y_COL].to_numpy(float)
    M500 = df["M_500c_Msun"].to_numpy(float)

    y0 = np.where((Y > 0) & (R500 > 0), K * Y / (R500 * R500), np.nan)
    theta500_arcmin = frame.theta_500(R500, z) * ARCMIN_PER_RAD

    sigma = sigma_y0_curve()
    sigma_y0 = sigma(theta500_arcmin)
    q = np.where((sigma_y0 > 0) & np.isfinite(y0), y0 / sigma_y0, np.nan)

    out = pd.DataFrame(
        {
            "z": z,
            "M_500c_Msun": M500,
            "theta500_arcmin": theta500_arcmin,
            "y0": y0,
            "sigma_y0": sigma_y0,
            "q": q,
        }
    )

    header = (
        "# FLAMINGO L2p8_m9: derived y0 and SNR (A10 GNFW, B=1, Y_R500c aperture).\n"
        f"# SOAP {Y_COL} with GNFW K(r_out={R_OUT}) = {K:.6f}:\n"
        f"# y0 = {K:.6f} * {Y_COL} / R_500c**2\n"
        "# theta500_arcmin = R_500c / D_A(z); sigma_y0 from szifi immf6.\n"
        "# q = y0 / sigma_y0.\n"
    )
    with open(OUT, "w") as fh:
        fh.write(header)
        out.to_csv(fh, index=False, float_format="%.6e")

    print(f"rows={len(out)}  median y0={np.nanmedian(y0):.3e}  median q={np.nanmedian(q):.3f}")
    print(f"q>5: {int(np.nansum(q > 5))}   q>4.5: {int(np.nansum(q > 4.5))}   q>6: {int(np.nansum(q > 6))}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
