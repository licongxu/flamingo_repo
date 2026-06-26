"""Build a derived catalogue with a simple-GNFW SNR assuming hydrostatic bias B=1.35.

This mirrors ``build_y0q_catalogue.py`` (which uses B=1) but recomputes the SNR
under the assumption that the SZ observable is the A10 prediction at the
*hydrostatic* mass M_500c / B instead of the true M_500c. Two effects, both
driven by B:

  * Amplitude (suppressed): the central Compton-y of the A10 self-similar
    profile scales as y0 ~ P_500(M/B) R_500(M/B) ~ (M/B)^(2/3+alpha_p)(M/B)^(1/3)
    = (M/B)^(1+alpha_p) = (M/B)^1.12 (alpha_p = 0.12). Anchoring on the real
    B=1 catalogue,  y0(B) = y0(B=1) * (1/B)**1.12.
  * Aperture: the matched-filter size uses the hydrostatic mass M_500c/B, so
    theta_500 ~ R_500c(M/B) = R_500c * (1/B)**(1/3) (R_500c ~ M^(1/3)). This is
    the cosmocnc_jax / Planck convention theta_500 ~ ((1-b) M)**(1/3) (theta_* =
    6.997 arcmin, bias_sz = 1-b = 1/B). The noise curve is read at theta500(B).

      y0(B)              = K * Y_5R500c_Mpc2 / R_500c**2 * (1/B)**1.12
      theta500_arcmin(B) = R_500c * (1/B)**(1/3) / D_A(z)
      sigma_y0(B)        = noise curve read at theta500(B)
      q(B)               = y0(B) / sigma_y0(B)

The real SOAP Y_5R500c (the empirical signal, with its scatter) and the
Arnaud/Nagai A10 GNFW shape factor K = 1.563860 are kept; B enters through the
self-similar amplitude suppression AND the (1-b) aperture rescaling, exactly as
in cosmocnc_jax. The catalogue has FEWER q>5 clusters than B=1.

Run:
    JAX_PLATFORMS=cpu python scripts/build_y0q_catalogue_arnaudB135.py
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

B = float(sys.argv[1]) if len(sys.argv) > 1 else 1.35  # hydrostatic mass bias
B_TAG = f"{B:.2f}".replace(".", "")  # e.g. 1.35 -> "135", 1.10 -> "110"

RAW = _REPO / "data/hydro_L2p8m9/catalogue/halo_catalogue_M500c_5e13_zlt3_soap_hdfstream_raw.csv"
OUT = _REPO / f"data/hydro_L2p8m9/catalogue/halo_catalogue_M500c_5e13_zlt3_y0q_arnaudB{B_TAG}.csv"
NOISE_DIR = _REPO / "data/noise"

R_OUT = 5.0  # line-of-sight / aperture truncation in units of R_500c
ARCMIN_PER_RAD = 180.0 / np.pi * 60.0
THETA_MIN, THETA_MAX = 0.5, 32.0  # szifi noise-curve validity range [arcmin]
ALPHA_P = 0.12  # A10 self-similar tilt; central y0 ~ (M/B)^(1+ALPHA_P)
GAMMA_Y0 = 1.0 + ALPHA_P  # = 1.12, exponent of the y0 ~ (M/B) self-similar prefactor


def gnfw_K(r_out: float = R_OUT) -> float:
    """Geometric factor K so that y0 = K * Y(<r_out R500) / R_500c**2 (B-independent)."""
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
    b_geom = (1.0 / B) ** (1.0 / 3.0)
    amp_factor = (1.0 / B) ** GAMMA_Y0  # self-similar prefactor with M -> M/B
    print(f"GNFW geometric constant K = {K:.6f}  (y0 = K * Y_5R500c / R_500c**2)")
    print(f"B = {B}:  amplitude factor (1/B)^{GAMMA_Y0} = {amp_factor:.6f}  (M -> M/B in P500*R500);"
          f"  theta factor (1/B)^(1/3) = {b_geom:.6f}  (hydrostatic R_500c, cosmocnc convention)")

    df = pd.read_csv(RAW, comment="#")
    z = df["z"].to_numpy(float)
    R500 = df["R_500c_Mpc"].to_numpy(float)
    Y = df["Y_5R500c_Mpc2"].to_numpy(float)
    M500 = df["M_500c_Msun"].to_numpy(float)

    # Amplitude: real B=1 y0 suppressed by the self-similar prefactor (M -> M/B).
    y0_b1 = np.where((Y > 0) & (R500 > 0), K * Y / (R500 * R500), np.nan)
    y0 = y0_b1 * amp_factor

    # Aperture: hydrostatic R_500c(B) = R_500c * (1/B)^(1/3); theta from hmfast D3A.
    R500_b = R500 * b_geom
    theta500_arcmin = frame.theta_500(R500_b, z) * ARCMIN_PER_RAD

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
        f"# FLAMINGO L2p8_m9 lightcone0: derived Compton-y0 and SNR (A10 GNFW, B={B}).\n"
        f"# A10 self-similar SZ amplitude at hydrostatic mass M_500c/B, B={B}:\n"
        f"# y0 = {K:.6f} * Y_5R500c_Mpc2 / R_500c**2 * (1/B)^{GAMMA_Y0}\n"
        "#   (real SOAP Y_5R500c kept; amplitude suppressed by the self-similar prefactor\n"
        "#    P500*R500 with M->M/B, exponent 1+alpha_p=1.12; Arnaud/Nagai A10 GNFW shape).\n"
        "# theta500_arcmin = R_500c*(1/B)^(1/3) / D_A(z) [hmfast D3A]: hydrostatic R_500c,\n"
        "#   cosmocnc/Planck convention theta_500 ~ ((1-b) M)^(1/3) with 1-b = 1/B.\n"
        "# sigma_y0 from szifi noise file (immf6, sky-averaged), looked up at theta500(B).\n"
        "# q = y0 / sigma_y0.  Built from halo_catalogue_M500c_5e13_zlt3_soap_hdfstream_raw.csv.\n"
    )
    with open(OUT, "w") as fh:
        fh.write(header)
        out.to_csv(fh, index=False, float_format="%.6e")

    print(f"rows={len(out)}  median y0={np.nanmedian(y0):.3e}  median q={np.nanmedian(q):.3f}")
    print(f"q>5: {int(np.nansum(q > 5))}   q>4.5: {int(np.nansum(q > 4.5))}   q>6: {int(np.nansum(q > 6))}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
