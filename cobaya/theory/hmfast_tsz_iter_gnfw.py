"""Cobaya theory: q_theory > Q masked tSZ D_ell^{yy} using the PURE
(non-parametric) Arnaud gNFW pressure profile.

Identical in structure to ``hmfast_tsz_iter_cprofile.HMFastTSZIterCProfile``,
except the electron-pressure profile is the bare
``hmfast.halos.profiles.GNFWPressureProfile`` (native gNFW shape parameters
P0, c500, alpha, beta, gamma plus hydrostatic mass bias B) WITHOUT the
parametric Compton-y0 multiplier (no A_SZ / alpha_SZ / beta_SZ rescaling).

The q_theory>Q masking still uses the iterative-SR completeness grid built
from (a_sz_sr, alpha_sr, b_sr); those remain fixed theory-side inputs.

Cobaya params exposed: sigma8, omega_cdm, omega_b, H0, n_s, tau_reio,
                       P0, c500, alpha, beta, gamma, B, sigma_lnY.
"""
from __future__ import annotations

import time

import numpy as np
import jax
import jax.numpy as jnp
from cobaya.theory import Theory

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.profiles import GNFWPressureProfile
from hmfast.tracers import tSZTracer
from hmfast.tracers.tsz_completeness import (
    build_snr_grid, conditional_An_undetected, load_sigma_y0_curve,
)

# Reuse the shared cosmology reparameterisation helpers, ell-binning and
# fiducial cosmology from the parametric-profile module. Importing it also
# performs the one-time JAX/GPU env setup and enables float64.
from hmfast_tsz_iter_cprofile import (
    _ELL_MIN, _ELL_MAX, _bin_to_18, _FID, _A_S_REF, _LN1E10_A_S_REF,
    _FID_B, _FID_SIGMA_LNY,
)

# Standard Arnaud (2010) gNFW shape fiducials, used only to seed the JIT
# warmup; the live values come from Cobaya at evaluation time.
_GNFW_P0 = 8.403
_GNFW_C500 = 1.177
_GNFW_ALPHA = 1.051
_GNFW_BETA = 5.4905
_GNFW_GAMMA = 0.3081


class HMFastTSZIterGNFW(Theory):
    output = ["Cl_sz"]
    params = {
        "sigma8": 0, "omega_cdm": 0, "omega_b": 0, "H0": 0, "n_s": 0,
        "tau_reio": 0, "P0": 0, "c500": 0, "alpha": 0, "beta": 0,
        "gamma": 0, "B": 0, "sigma_lnY": 0,
    }
    n_mass: int = 64
    n_z: int = 96
    log10_m_min: float = 10.0
    log10_m_max: float = 15.5
    z_min: float = 0.005
    z_max: float = 3.0
    n_ell_internal: int = 50
    q_cat: float = 5.0
    include_2h: bool = True
    # iterative-SR catalogue parameters defining q_theory > Q (per-cut fixed)
    a_sz_sr: float = -4.31
    alpha_sr: float = 1.12
    b_sr: float = 1.25

    def get_requirements(self):
        return {k: None for k in self.params}

    def initialize(self):
        self._ell_int = jnp.geomspace(
            float(_ELL_MIN[0]), float(_ELL_MAX[-1]), int(self.n_ell_internal))
        self._m = jnp.geomspace(10**self.log10_m_min, 10**self.log10_m_max, self.n_mass)
        self._z = jnp.geomspace(self.z_min, self.z_max, self.n_z)

        self._cosmo_seed = Cosmology(emulator_set="lcdm:v1")
        cosmo_fid = self._cosmo_seed.update(**_FID)
        self._hm_seed = HaloModel(
            cosmology=cosmo_fid,
            mass_definition=MassDefinition(500, "critical"),
            convert_masses=True,
        )
        self._prof_seed = GNFWPressureProfile(
            P0=_GNFW_P0, c500=_GNFW_C500, alpha=_GNFW_ALPHA,
            beta=_GNFW_BETA, gamma=_GNFW_GAMMA, B=_FID_B)
        self._tsz = tSZTracer(profile=self._prof_seed)

        # q-selection grid built ONCE at fiducial cosmology + iterative SR.
        coeff, _ = load_sigma_y0_curve()
        self._snr = build_snr_grid(
            self._hm_seed, self._m, self._z,
            float(self.a_sz_sr), float(self.alpha_sr), float(self.b_sr),
            coeff=coeff)

        self._eval_cl_masked = jax.jit(
            self._eval_cl_masked_impl,
            static_argnames=("include_2h_static", "q_cat_static"),
        )

        block = getattr(jax, "block_until_ready", lambda x: x)
        t0 = time.perf_counter()
        for dh, dp, dc in ((0.0, 0.0, 0.0), (0.1, 0.1, 0.05)):
            c1, c2 = self._eval_cl_masked(
                _FID["H0"] + dh, _FID["omega_cdm"], _FID["omega_b"],
                _FID["ln1e10A_s"], _FID["n_s"], _FID["tau_reio"],
                _GNFW_P0 + dp, _GNFW_C500 + dc, _GNFW_ALPHA, _GNFW_BETA,
                _GNFW_GAMMA, _FID_B, _FID_SIGMA_LNY,
                include_2h_static=bool(self.include_2h),
                q_cat_static=float(self.q_cat),
            )
            block(c1); block(c2)
        self.log.info(
            "HMFastTSZIterGNFW: q_cat=%g  iter SR (a_sz_sr=%.4f, alpha_sr=%.4f, b_sr=%.3f) "
            "  init+JIT warmup %.3fs",
            self.q_cat, self.a_sz_sr, self.alpha_sr, self.b_sr,
            time.perf_counter() - t0)
        super().initialize()

    def _eval_cl_masked_impl(self, H0, omega_cdm, omega_b, ln1e10A_s,
                             n_s, tau_reio, P0, c500, alpha, beta, gamma, B,
                             sigma_lnY, include_2h_static, q_cat_static):
        cosmo = self._cosmo_seed.update(
            H0=H0, omega_cdm=omega_cdm, omega_b=omega_b,
            ln1e10A_s=ln1e10A_s, n_s=n_s, tau_reio=tau_reio)
        hm = self._hm_seed.update(cosmology=cosmo)
        prof = self._prof_seed.update(
            P0=P0, c500=c500, alpha=alpha, beta=beta, gamma=gamma, B=B)
        tsz = self._tsz.update(profile=prof)
        mask_1h = conditional_An_undetected(
            self._snr, sigma_lnY=sigma_lnY, q_cat=q_cat_static, n_power=2)
        cl_1h = hm.cl_1h_masked(tsz, None, self._ell_int, self._m, self._z, mask_1h)
        if include_2h_static:
            mask_2h = conditional_An_undetected(
                self._snr, sigma_lnY=sigma_lnY, q_cat=q_cat_static, n_power=1)
            cl_2h = hm.cl_2h_masked(tsz, None, self._ell_int, self._m, self._z, mask_2h)
        else:
            cl_2h = jnp.zeros_like(cl_1h)
        return cl_1h, cl_2h

    def _sigma8_to_lnA_s(self, sigma8_target, H0, omega_cdm, omega_b, n_s, tau_reio):
        cosmo = self._cosmo_seed.update(
            H0=H0, omega_cdm=omega_cdm, omega_b=omega_b,
            ln1e10A_s=_LN1E10_A_S_REF, n_s=n_s, tau_reio=tau_reio)
        s8_ref = float(np.asarray(cosmo.sigma8(0.0)))
        A_s_new = _A_S_REF * (float(sigma8_target) / s8_ref) ** 2
        return float(np.log(1e10 * A_s_new))

    def calculate(self, state, want_derived=True, **p):
        t0 = time.perf_counter()
        omega_cdm = float(p["omega_cdm"]); omega_b = float(p["omega_b"])
        H0 = float(p["H0"]); n_s = float(p["n_s"])
        tau_reio = float(p["tau_reio"])
        ln1e10A_s = self._sigma8_to_lnA_s(
            float(p["sigma8"]), H0, omega_cdm, omega_b, n_s, tau_reio
        )
        P0 = float(p["P0"]); c500 = float(p["c500"]); alpha = float(p["alpha"])
        beta = float(p["beta"]); gamma = float(p["gamma"])
        sigma_lnY = float(p["sigma_lnY"]); B = float(p["B"])

        cl_1h, cl_2h = self._eval_cl_masked(
            H0, omega_cdm, omega_b, ln1e10A_s, n_s, tau_reio,
            P0, c500, alpha, beta, gamma, B, sigma_lnY,
            include_2h_static=bool(self.include_2h),
            q_cat_static=float(self.q_cat))
        block = getattr(jax, "block_until_ready", lambda x: x)
        block(cl_1h); block(cl_2h)
        ell_np = np.asarray(self._ell_int)
        Dl_1h = _bin_to_18(ell_np, np.asarray(cl_1h))
        Dl_2h = _bin_to_18(ell_np, np.asarray(cl_2h))

        state["Cl_sz"] = {
            "1h": np.asarray(Dl_1h, dtype=np.float64),
            "2h": np.asarray(Dl_2h, dtype=np.float64),
        }
        self._current_state = state
        self.log.info(
            "HMFastTSZIterGNFW: theory in %.3fs  mean(Dl)=%.3e",
            time.perf_counter() - t0, float(np.mean(Dl_1h + Dl_2h)))

    def get_Cl_sz(self):
        return self._current_state.get("Cl_sz", None)


class HMFastTSZIterGNFWLogA(HMFastTSZIterGNFW):
    """Pure-gNFW theory parametrised by ``logA = ln(10^10 A_s)`` sampled
    DIRECTLY (no sigma8 reparameterisation), enabling joint cosmology + B
    sampling. Everything else — masking/completeness, JIT eval, ell-binning —
    is inherited unchanged from :class:`HMFastTSZIterGNFW`.

    Cobaya params: logA, omega_cdm, omega_b, H0, n_s, tau_reio,
                   P0, c500, alpha, beta, gamma, B, sigma_lnY.
    """
    params = {
        "logA": 0, "omega_cdm": 0, "omega_b": 0, "H0": 0, "n_s": 0,
        "tau_reio": 0, "P0": 0, "c500": 0, "alpha": 0, "beta": 0,
        "gamma": 0, "B": 0, "sigma_lnY": 0,
    }

    def calculate(self, state, want_derived=True, **p):
        t0 = time.perf_counter()
        omega_cdm = float(p["omega_cdm"]); omega_b = float(p["omega_b"])
        H0 = float(p["H0"]); n_s = float(p["n_s"])
        tau_reio = float(p["tau_reio"])
        ln1e10A_s = float(p["logA"])          # sampled directly, no sigma8 map
        P0 = float(p["P0"]); c500 = float(p["c500"]); alpha = float(p["alpha"])
        beta = float(p["beta"]); gamma = float(p["gamma"])
        sigma_lnY = float(p["sigma_lnY"]); B = float(p["B"])

        cl_1h, cl_2h = self._eval_cl_masked(
            H0, omega_cdm, omega_b, ln1e10A_s, n_s, tau_reio,
            P0, c500, alpha, beta, gamma, B, sigma_lnY,
            include_2h_static=bool(self.include_2h),
            q_cat_static=float(self.q_cat))
        block = getattr(jax, "block_until_ready", lambda x: x)
        block(cl_1h); block(cl_2h)
        ell_np = np.asarray(self._ell_int)
        Dl_1h = _bin_to_18(ell_np, np.asarray(cl_1h))
        Dl_2h = _bin_to_18(ell_np, np.asarray(cl_2h))

        state["Cl_sz"] = {
            "1h": np.asarray(Dl_1h, dtype=np.float64),
            "2h": np.asarray(Dl_2h, dtype=np.float64),
        }
        self._current_state = state
        self.log.info(
            "HMFastTSZIterGNFWLogA: theory in %.3fs  mean(Dl)=%.3e",
            time.perf_counter() - t0, float(np.mean(Dl_1h + Dl_2h)))
