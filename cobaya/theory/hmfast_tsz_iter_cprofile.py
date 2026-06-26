"""Cobaya theory: q_theory > Q masked tSZ D_ell^{yy} for the iterative-SR
pipeline.

Variant of `hmfast_tsz_masked_cprofile.HMFastTSZMaskedCProfile` which makes
the catalogue-SR triple (A_SZ_SR, alpha_SR, B_SR) that defines the q_theory
boundary explicit theory-side inputs, so each Q-cut config can pin them to
its own converged iterative-SR fit. Everything else mirrors the obscat
theory module: sigma8 reparameterisation, JIT compile, 4-parameter SZ
profile sampling, lognormal-scatter conditional moments via
conditional_An_undetected with n_power={1,2}.

Sampled params: sigma8, omega_cdm, omega_b, H0, n_s, A_SZ, alpha_SZ,
                sigma_lnY, c_500, + fixed tau_reio, B.
"""
from __future__ import annotations

import os
import time

if os.environ.get("HMFAST_COBAYA_USE_GPU", "1").strip() not in ("1", "true", "True", "yes", "YES"):
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
if "xla_gpu_persistent_cache_dir" in os.environ.get("XLA_FLAGS", ""):
    os.environ.pop("XLA_FLAGS")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.10")

import numpy as np
import jax
import jax.numpy as jnp
from cobaya.theory import Theory

jax.config.update("jax_enable_x64", True)

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.mass_definition import MassDefinition, convert_m_delta
from hmfast.halos.profiles import ParametricGNFWPressureProfile, B12PressureProfile
from hmfast.tracers import tSZTracer
from hmfast.tracers.tsz_completeness import (
    build_snr_grid, conditional_An_undetected, load_sigma_y0_curve,
)
from hmfast.utils import Const
from jax.scipy.special import gammaln as _jax_gammaln

# Pivot mass for the mass-dependent gNFW concentration (Tier 2 fit).
_M_PIV_C500 = 3.0e14  # M_sun


class MassDependentGNFWPressureProfile(ParametricGNFWPressureProfile):
    """gNFW pressure profile with mass-dependent concentration:

        c_500(M_500c) = c_0 * (M_500c / 3e14)**beta_c

    Replaces the scalar `c500` of the parent class. All other gNFW shape
    parameters (alpha, beta, gamma, P0) and the parametric amplitude
    (A_SZ, alpha_SZ, B) are inherited unchanged. Registered as a JAX pytree
    so c_0, beta_c are JIT-traceable.
    """

    def __init__(self, x=None, A_SZ=-4.97, alpha_SZ=0.7867,
                 P0=8.130, c_0=1.177, beta_c=0.0,
                 alpha=1.0620, beta=5.4807, gamma=0.3292, B=1.4):
        # Parent expects a scalar c500; pass c_0 (the pivot-mass value) as the
        # placeholder so the parent's bookkeeping is consistent. We override
        # u_r to use c_0 + beta_c -> c_500(M) instead.
        super().__init__(x=x, A_SZ=A_SZ, alpha_SZ=alpha_SZ,
                         P0=P0, c500=c_0,
                         alpha=alpha, beta=beta, gamma=gamma, B=B)
        self.c_0 = c_0
        self.beta_c = beta_c

    def _tree_flatten(self):
        leaves = (self.A_SZ, self.alpha_SZ,
                  self.P0, self.c_0, self.beta_c,
                  self.alpha, self.beta, self.gamma, self.B)
        aux_data = (tuple(self._x.tolist()), self._hankel)
        return (leaves, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        x_tuple, hankel = aux_data
        obj = cls.__new__(cls)
        (obj.A_SZ, obj.alpha_SZ,
         obj.P0, obj.c_0, obj.beta_c,
         obj.alpha, obj.beta, obj.gamma, obj.B) = leaves
        obj.c500 = obj.c_0  # placeholder for parent attribute
        obj._x = np.array(x_tuple)
        obj._hankel = hankel
        return obj

    def update(self, A_SZ=None, alpha_SZ=None, P0=None,
               c_0=None, beta_c=None,
               alpha=None, beta=None, gamma=None, B=None, **kwargs):
        leaves, treedef = self._tree_flatten()
        new_leaves = (
            A_SZ if A_SZ is not None else self.A_SZ,
            alpha_SZ if alpha_SZ is not None else self.alpha_SZ,
            P0 if P0 is not None else self.P0,
            c_0 if c_0 is not None else self.c_0,
            beta_c if beta_c is not None else self.beta_c,
            alpha if alpha is not None else self.alpha,
            beta if beta is not None else self.beta,
            gamma if gamma is not None else self.gamma,
            B if B is not None else self.B,
        )
        return self._tree_unflatten(treedef, new_leaves)

    @jax.jit
    def u_r(self, halo_model, r, m, z):
        H0 = halo_model.cosmology.H0
        A_SZ, alpha_SZ = self.A_SZ, self.alpha_SZ
        c_0, beta_c = self.c_0, self.beta_c
        P0, alpha, beta, gamma, B = (
            self.P0, self.alpha, self.beta, self.gamma, self.B
        )
        r, m, z = jnp.atleast_1d(r), jnp.atleast_1d(m), jnp.atleast_1d(z)
        h = H0 / 100.0

        mass_def_old = halo_model.mass_definition
        mass_def_500c = MassDefinition(500, "critical")
        c_old = halo_model.concentration.c_delta(halo_model, m, z)
        m500c = convert_m_delta(halo_model.cosmology, m, z,
                                mass_def_old, mass_def_500c, c_old=c_old)
        # m500c shape: (Nm, Nz). Mass-dependent c_500(M) (Nm, Nz):
        c500_mz = c_0 * (m500c / _M_PIV_C500) ** beta_c

        r_500c = mass_def_500c.r_delta(halo_model.cosmology, m500c, z)
        x_500c = r[:, None, None] / ((1.0 + z[None, None, :]) * r_500c[None, :, :])

        H = jnp.atleast_1d(halo_model.cosmology.hubble_parameter(z))[None, None, :]
        E_z = H / H0
        m500c_tilde = (m500c * h / B)[None]

        P_500c_arnaud = (
            1.65 * (h / 0.7) ** 2 * E_z ** (8.0 / 3.0)
            * (m500c_tilde / (0.7 * 3e14)) ** (2.0 / 3.0 + 0.12)
            * (0.7 / h) ** 1.5
        )

        sigma_T_cm2 = 6.6524587e-25
        m_e_c2_eV = 510998.95
        shape_integral = 0.470502095
        mpc_to_cm = Const._Mpc_over_m_ * 100.0

        r_500c_cm = (r_500c * mpc_to_cm)[None, :, :]
        y0_orig = (
            2.0 * (sigma_T_cm2 / m_e_c2_eV)
            * P0 * P_500c_arnaud * r_500c_cm
            * shape_integral
        )
        y0_param = (
            (10.0 ** A_SZ)
            * (m500c_tilde / (0.7 * 3e14)) ** alpha_SZ
            * E_z ** 2
            * (h / 0.7) ** (-0.5)
        )
        ratio = y0_param / y0_orig

        # Mass-dependent c_500 broadcasts over the r axis only
        scaled_x = c500_mz[None, :, :] * x_500c  # (Nr, Nm, Nz)
        Pe_arnaud = (P_500c_arnaud * P0
                     * scaled_x ** (-gamma)
                     * (1.0 + scaled_x ** alpha) ** ((gamma - beta) / alpha))
        return Pe_arnaud * ratio


jax.tree_util.register_pytree_node(
    MassDependentGNFWPressureProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: MassDependentGNFWPressureProfile._tree_unflatten(
        aux_data, children),
)

_ELL_MIN = np.array([9, 12, 16, 21, 27, 35, 46, 60, 78,
                     102, 133, 173, 224, 292, 380, 494, 642, 835], dtype=int)
_ELL_MAX = np.array([12, 16, 21, 27, 35, 46, 60, 78, 102,
                     133, 173, 224, 292, 380, 494, 642, 835, 1085], dtype=int)
_ELL_EFF = np.array([10.0, 13.5, 18.0, 23.5, 30.5, 40.0, 52.5, 68.5, 89.5,
                     117.0, 152.5, 198.0, 257.5, 335.5, 436.5, 567.5, 738.0, 959.5])
_L_MAX = int(np.max(_ELL_MAX - _ELL_MIN))
_ELL_INT = _ELL_MIN[:, None] + np.arange(_L_MAX)[None, :]
_ELL_MASK = (_ELL_INT < _ELL_MAX[:, None]).astype(np.float64)


def _bin_to_18(ell_in, Cl_in):
    log_ell = np.log(np.asarray(ell_in))
    Cl_q = np.interp(np.log(_ELL_INT.astype(float)), log_ell, np.asarray(Cl_in))
    Dl_q = _ELL_INT * (_ELL_INT + 1.0) * Cl_q / (2.0 * np.pi)
    return (Dl_q * _ELL_MASK).sum(axis=1) / _ELL_MASK.sum(axis=1)


_A_S_REF = 2.1e-9
_LN1E10_A_S_REF = float(np.log(1e10 * _A_S_REF))
_FID = dict(H0=67.4, omega_cdm=0.12, omega_b=0.022, n_s=0.96, tau_reio=0.0544,
            ln1e10A_s=_LN1E10_A_S_REF)
_FID_A_SZ = -4.31
_FID_ALPHA_SZ = 1.12
_FID_B = 1.4
_FID_C_500 = 1.156
_FID_SIGMA_LNY = 0.17


class HMFastTSZIterCProfile(Theory):
    output = ["Cl_sz"]
    params = {
        "sigma8": 0, "omega_cdm": 0, "omega_b": 0, "H0": 0, "n_s": 0,
        "tau_reio": 0, "A_SZ": 0, "alpha_SZ": 0, "sigma_lnY": 0,
        "c_500": 0, "B": 0, "beta_SZ": 0,
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
        self._prof_seed = ParametricGNFWPressureProfile(
            A_SZ=_FID_A_SZ, alpha_SZ=_FID_ALPHA_SZ, c500=_FID_C_500, B=_FID_B)
        self._tsz = tSZTracer(profile=self._prof_seed)

        # SNR(q) coefficients + catalogue SR triple; grid rebuilt per cosmology.
        coeff, _ = load_sigma_y0_curve()
        self._snr_coeff = coeff
        self._snr_sr = (
            float(self.a_sz_sr), float(self.alpha_sr), float(self.b_sr))

        self._eval_cl_masked = jax.jit(
            self._eval_cl_masked_impl,
            static_argnames=("include_2h_static", "q_cat_static"),
        )

        block = getattr(jax, "block_until_ready", lambda x: x)
        t0 = time.perf_counter()
        for dh, ds, dc in ((0.0, 0.0, 0.0), (0.1, 0.01, 0.05)):
            c1, c2 = self._eval_cl_masked(
                _FID["H0"] + dh, _FID["omega_cdm"], _FID["omega_b"],
                _FID["ln1e10A_s"], _FID["n_s"], _FID["tau_reio"],
                _FID_A_SZ + ds, _FID_ALPHA_SZ + ds,
                2.0, _FID_C_500 + dc, _FID_B, _FID_SIGMA_LNY,
                include_2h_static=bool(self.include_2h),
                q_cat_static=float(self.q_cat),
            )
            block(c1); block(c2)
        self.log.info(
            "HMFastTSZIterCProfile: q_cat=%g  iter SR (A_SZ=%.4f, alpha=%.4f, B=%.3f) "
            "  init+JIT warmup %.3fs",
            self.q_cat, self.a_sz_sr, self.alpha_sr, self.b_sr,
            time.perf_counter() - t0)
        super().initialize()

    def _eval_cl_masked_impl(self, H0, omega_cdm, omega_b, ln1e10A_s,
                             n_s, tau_reio, A_SZ, alpha_SZ, beta_SZ, c_500, B,
                             sigma_lnY, include_2h_static, q_cat_static):
        cosmo = self._cosmo_seed.update(
            H0=H0, omega_cdm=omega_cdm, omega_b=omega_b,
            ln1e10A_s=ln1e10A_s, n_s=n_s, tau_reio=tau_reio)
        hm = self._hm_seed.update(cosmology=cosmo)
        a_sz, alpha_sr, b_sr = self._snr_sr
        snr = build_snr_grid(
            hm, self._m, self._z, a_sz, alpha_sr, b_sr, coeff=self._snr_coeff)
        prof = self._prof_seed.update(
            A_SZ=A_SZ, alpha_SZ=alpha_SZ, c500=c_500, B=B)
        tsz = self._tsz.update(profile=prof)
        mask_1h = conditional_An_undetected(
            snr, sigma_lnY=sigma_lnY, q_cat=q_cat_static, n_power=2)
        cl_1h = hm.cl_1h_masked(tsz, None, self._ell_int, self._m, self._z, mask_1h)
        if include_2h_static:
            mask_2h = conditional_An_undetected(
                snr, sigma_lnY=sigma_lnY, q_cat=q_cat_static, n_power=1)
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
        A_SZ = float(p["A_SZ"]); alpha_SZ = float(p["alpha_SZ"])
        beta_SZ = float(p["beta_SZ"])
        sigma_lnY = float(p["sigma_lnY"]); c_500 = float(p["c_500"])
        B = float(p["B"])

        cl_1h, cl_2h = self._eval_cl_masked(
            H0, omega_cdm, omega_b, ln1e10A_s, n_s, tau_reio,
            A_SZ, alpha_SZ, beta_SZ, c_500, B, sigma_lnY,
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
            "HMFastTSZIterCProfile: theory in %.3fs  mean(Dl)=%.3e",
            time.perf_counter() - t0, float(np.mean(Dl_1h + Dl_2h)))

    def get_Cl_sz(self):
        return self._current_state.get("Cl_sz", None)


class HMFastTSZIterCProfileMM(Theory):
    """Tier 2 variant: replaces scalar c_500 with mass-dependent

        c_500(M_500c) = c_0 * (M_500c / 3e14)**beta_c

    Sampled params (5): A_SZ, alpha_SZ, sigma_lnY, c_0, beta_c.
    Fixed (theory-side): cosmology block + B.
    """

    output = ["Cl_sz"]
    params = {
        "sigma8": 0, "omega_cdm": 0, "omega_b": 0, "H0": 0, "n_s": 0,
        "tau_reio": 0, "A_SZ": 0, "alpha_SZ": 0, "sigma_lnY": 0,
        "c_0": 0, "beta_c": 0, "B": 0,
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
        # Mass-dependent gNFW profile seed
        self._prof_seed = MassDependentGNFWPressureProfile(
            A_SZ=_FID_A_SZ, alpha_SZ=_FID_ALPHA_SZ,
            c_0=1.177, beta_c=0.0, B=_FID_B)
        self._tsz = tSZTracer(profile=self._prof_seed)

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
        for d in ((0.0, 0.0, 0.0, 0.0), (0.1, 0.01, 0.05, -0.1)):
            dh, ds, dc, db_ = d
            c1, c2 = self._eval_cl_masked(
                _FID["H0"] + dh, _FID["omega_cdm"], _FID["omega_b"],
                _FID["ln1e10A_s"], _FID["n_s"], _FID["tau_reio"],
                _FID_A_SZ + ds, _FID_ALPHA_SZ + ds,
                1.177 + dc, 0.0 + db_, _FID_B, _FID_SIGMA_LNY,
                include_2h_static=bool(self.include_2h),
                q_cat_static=float(self.q_cat),
            )
            block(c1); block(c2)
        self.log.info(
            "HMFastTSZIterCProfileMM: q_cat=%g  iter SR (%.4f, %.4f, %.3f)  "
            "init+JIT warmup %.3fs",
            self.q_cat, self.a_sz_sr, self.alpha_sr, self.b_sr,
            time.perf_counter() - t0)
        super().initialize()

    def _eval_cl_masked_impl(self, H0, omega_cdm, omega_b, ln1e10A_s,
                             n_s, tau_reio, A_SZ, alpha_SZ,
                             c_0, beta_c, B, sigma_lnY,
                             include_2h_static, q_cat_static):
        cosmo = self._cosmo_seed.update(
            H0=H0, omega_cdm=omega_cdm, omega_b=omega_b,
            ln1e10A_s=ln1e10A_s, n_s=n_s, tau_reio=tau_reio)
        hm = self._hm_seed.update(cosmology=cosmo)
        prof = self._prof_seed.update(
            A_SZ=A_SZ, alpha_SZ=alpha_SZ,
            c_0=c_0, beta_c=beta_c, B=B)
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
        A_SZ = float(p["A_SZ"]); alpha_SZ = float(p["alpha_SZ"])
        sigma_lnY = float(p["sigma_lnY"])
        c_0 = float(p["c_0"]); beta_c = float(p["beta_c"])
        B = float(p["B"])

        cl_1h, cl_2h = self._eval_cl_masked(
            H0, omega_cdm, omega_b, ln1e10A_s, n_s, tau_reio,
            A_SZ, alpha_SZ, c_0, beta_c, B, sigma_lnY,
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
            "HMFastTSZIterCProfileMM: theory in %.3fs  mean(Dl)=%.3e",
            time.perf_counter() - t0, float(np.mean(Dl_1h + Dl_2h)))

    def get_Cl_sz(self):
        return self._current_state.get("Cl_sz", None)


class HMFastTSZIterCProfileShape(Theory):
    """Tier 3 variant: scalar c_500 + free gNFW outer slope (gnfw_beta)
    and inner slope (gnfw_gamma). gNFW transition rate (gnfw_alpha) is
    held at the Arnaud value 1.062.

    Sampled params (6): A_SZ, alpha_SZ, sigma_lnY, c_500, gnfw_beta,
                        gnfw_gamma.
    Fixed (theory-side): cosmology block + B + gnfw_alpha.
    """

    output = ["Cl_sz"]
    params = {
        "sigma8": 0, "omega_cdm": 0, "omega_b": 0, "H0": 0, "n_s": 0,
        "tau_reio": 0, "A_SZ": 0, "alpha_SZ": 0, "sigma_lnY": 0,
        "c_500": 0, "gnfw_beta": 0, "gnfw_gamma": 0, "B": 0,
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
    a_sz_sr: float = -4.31
    alpha_sr: float = 1.12
    b_sr: float = 1.25
    # gNFW transition rate fixed at Arnaud 2010
    gnfw_alpha_fixed: float = 1.0620

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
        # Stock parametric gNFW (scalar c_500); shape params updated per step
        self._prof_seed = ParametricGNFWPressureProfile(
            A_SZ=_FID_A_SZ, alpha_SZ=_FID_ALPHA_SZ,
            c500=_FID_C_500, B=_FID_B,
            alpha=float(self.gnfw_alpha_fixed),
            beta=5.4807, gamma=0.3292)
        self._tsz = tSZTracer(profile=self._prof_seed)

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
        for dh, ds, dc, dbeta, dgamma in (
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.1, 0.01, 0.05, -1.0, 0.1),
        ):
            c1, c2 = self._eval_cl_masked(
                _FID["H0"] + dh, _FID["omega_cdm"], _FID["omega_b"],
                _FID["ln1e10A_s"], _FID["n_s"], _FID["tau_reio"],
                _FID_A_SZ + ds, _FID_ALPHA_SZ + ds,
                2.0, _FID_C_500 + dc, _FID_B, _FID_SIGMA_LNY,
                5.4807 + dbeta, 0.3292 + dgamma,
                include_2h_static=bool(self.include_2h),
                q_cat_static=float(self.q_cat),
            )
            block(c1); block(c2)
        self.log.info(
            "HMFastTSZIterCProfileShape: q_cat=%g  iter SR (%.4f, %.4f, %.3f)  "
            "init+JIT warmup %.3fs",
            self.q_cat, self.a_sz_sr, self.alpha_sr, self.b_sr,
            time.perf_counter() - t0)
        super().initialize()

    def _eval_cl_masked_impl(self, H0, omega_cdm, omega_b, ln1e10A_s,
                             n_s, tau_reio, A_SZ, alpha_SZ, c_500, B,
                             sigma_lnY, gnfw_beta, gnfw_gamma,
                             include_2h_static, q_cat_static):
        cosmo = self._cosmo_seed.update(
            H0=H0, omega_cdm=omega_cdm, omega_b=omega_b,
            ln1e10A_s=ln1e10A_s, n_s=n_s, tau_reio=tau_reio)
        hm = self._hm_seed.update(cosmology=cosmo)
        prof = self._prof_seed.update(
            A_SZ=A_SZ, alpha_SZ=alpha_SZ,
            beta_SZ=beta_SZ, c500=c_500, B=B,
            beta=gnfw_beta, gamma=gnfw_gamma)
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
        A_SZ = float(p["A_SZ"]); alpha_SZ = float(p["alpha_SZ"])
        sigma_lnY = float(p["sigma_lnY"])
        c_500 = float(p["c_500"])
        gnfw_beta = float(p["gnfw_beta"]); gnfw_gamma = float(p["gnfw_gamma"])
        B = float(p["B"])

        cl_1h, cl_2h = self._eval_cl_masked(
            H0, omega_cdm, omega_b, ln1e10A_s, n_s, tau_reio,
            A_SZ, alpha_SZ, beta_SZ, c_500, B, sigma_lnY,
            gnfw_beta, gnfw_gamma,
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
            "HMFastTSZIterCProfileShape: theory in %.3fs  mean(Dl)=%.3e",
            time.perf_counter() - t0, float(np.mean(Dl_1h + Dl_2h)))

    def get_Cl_sz(self):
        return self._current_state.get("Cl_sz", None)


# ============================================================================
# Battaglia 2012 profile with parametric (A_SZ, alpha_SZ) Y0--M amplitude
# ============================================================================

class ParametricB12PressureProfile(B12PressureProfile):
    """B12 pressure profile rescaled so that the integrated central Compton-y
    follows the parametric Y0--M relation

        y0_param(M_500c, z) = 10^A_SZ
                              (M_500c h / B / (0.7 * 3e14))^alpha_SZ
                              E(z)^2 (h/0.7)^(-0.5)

    The B12 SHAPE (xc, alpha, beta, gamma, mass and redshift exponents) is
    held entirely fixed at the Battaglia 2012 hydro-calibrated values.

    Implementation mirrors `ParametricGNFWPressureProfile`: at each (M, z) the
    natural B12 y0 is computed analytically and the radial profile is rescaled
    by ratio = y0_param / y0_B12.
    """

    def __init__(self, x=None, A_SZ=-4.31, alpha_SZ=1.12, B=1.4,
                 A_P0=18.1, A_xc=0.497, A_beta=4.35,
                 alpha_m_P0=0.154, alpha_m_xc=-0.00865, alpha_m_beta=0.0393,
                 alpha_z_P0=-0.758, alpha_z_xc=0.731, alpha_z_beta=0.415):
        super().__init__(x=x, A_P0=A_P0, A_xc=A_xc, A_beta=A_beta,
                         alpha_m_P0=alpha_m_P0, alpha_m_xc=alpha_m_xc,
                         alpha_m_beta=alpha_m_beta,
                         alpha_z_P0=alpha_z_P0, alpha_z_xc=alpha_z_xc,
                         alpha_z_beta=alpha_z_beta)
        self.A_SZ = A_SZ
        self.alpha_SZ = alpha_SZ
        self.B = B

    def _tree_flatten(self):
        b12_leaves, aux = super()._tree_flatten()
        leaves = (self.A_SZ, self.alpha_SZ, self.B) + b12_leaves
        return (leaves, aux)

    @classmethod
    def _tree_unflatten(cls, aux, leaves):
        obj = B12PressureProfile._tree_unflatten(aux, leaves[3:])
        obj.__class__ = cls
        obj.A_SZ = leaves[0]
        obj.alpha_SZ = leaves[1]
        obj.B = leaves[2]
        return obj

    def update(self, A_SZ=None, alpha_SZ=None, B=None,
               A_P0=None, A_xc=None, A_beta=None,
               alpha_m_P0=None, alpha_m_xc=None, alpha_m_beta=None,
               alpha_z_P0=None, alpha_z_xc=None, alpha_z_beta=None):
        # Build the full 12-tuple and unflatten directly to avoid the parent's
        # update() path mismatching leaves length.
        leaves, treedef = self._tree_flatten()
        new_leaves = (
            A_SZ if A_SZ is not None else self.A_SZ,
            alpha_SZ if alpha_SZ is not None else self.alpha_SZ,
            B if B is not None else self.B,
            A_P0 if A_P0 is not None else self.A_P0,
            A_xc if A_xc is not None else self.A_xc,
            A_beta if A_beta is not None else self.A_beta,
            alpha_m_P0 if alpha_m_P0 is not None else self.alpha_m_P0,
            alpha_m_xc if alpha_m_xc is not None else self.alpha_m_xc,
            alpha_m_beta if alpha_m_beta is not None else self.alpha_m_beta,
            alpha_z_P0 if alpha_z_P0 is not None else self.alpha_z_P0,
            alpha_z_xc if alpha_z_xc is not None else self.alpha_z_xc,
            alpha_z_beta if alpha_z_beta is not None else self.alpha_z_beta,
        )
        return self._tree_unflatten(treedef, new_leaves)

    @jax.jit
    def u_r(self, halo_model, r, m, z):
        Pe_B12 = B12PressureProfile.u_r.__wrapped__(self, halo_model, r, m, z)

        cparams = halo_model.cosmology._cosmo_params()
        h = cparams["h"]
        H0 = halo_model.cosmology.H0
        r_a = jnp.atleast_1d(r); m_a = jnp.atleast_1d(m); z_a = jnp.atleast_1d(z)

        mass_def_old = halo_model.mass_definition
        mass_def_200c = MassDefinition(200, "critical")
        mass_def_500c = MassDefinition(500, "critical")
        c_old = halo_model.concentration.c_delta(halo_model, m_a, z_a)
        m200c = convert_m_delta(halo_model.cosmology, m_a, z_a,
                                mass_def_old, mass_def_200c, c_old=c_old)
        m500c = convert_m_delta(halo_model.cosmology, m_a, z_a,
                                mass_def_old, mass_def_500c, c_old=c_old)
        r_200c_phys = mass_def_200c.r_delta(halo_model.cosmology, m200c, z_a)

        H = jnp.atleast_1d(halo_model.cosmology.hubble_parameter(z_a))
        E_z = H / H0

        # B12 shape params at (M, z); m200c is (Nm, Nz)
        mass_ratio = m200c / 1e14
        one_plus_z = (1.0 + z_a)[None, :]
        P0_B12 = self.A_P0 * mass_ratio**self.alpha_m_P0 * one_plus_z**self.alpha_z_P0
        xc_B12 = self.A_xc * mass_ratio**self.alpha_m_xc * one_plus_z**self.alpha_z_xc
        beta_B12 = self.A_beta * mass_ratio**self.alpha_m_beta * one_plus_z**self.alpha_z_beta

        # B12 P_200c (matches expression in B12PressureProfile.u_r line 699)
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
        r_200c_h = r_200c_phys * h
        P_200c = (m200c / r_200c_h) * f_b * 2.61051e-18 * H[None, :]**2  # (Nm, Nz)

        # y0_B12 = 2 (sigma_T/m_ec^2) * r_phys_cm * P_200c * P0 * xc * Beta(0.7, beta-0.7)
        sigma_T_cm2 = 6.6524587e-25
        m_e_c2_eV = 510998.95
        mpc_to_cm = Const._Mpc_over_m_ * 100.0
        r_200c_cm = r_200c_phys * mpc_to_cm
        # Beta(0.7, beta-0.7) via log-gamma; require beta > 0.7
        beta_arg = jnp.maximum(beta_B12 - 0.7, 1e-3)
        log_beta_fn = _jax_gammaln(0.7) + _jax_gammaln(beta_arg) - _jax_gammaln(beta_arg + 0.7)
        beta_fn = jnp.exp(log_beta_fn)

        y0_B12 = (2.0 * (sigma_T_cm2 / m_e_c2_eV)
                  * P_200c * P0_B12 * r_200c_cm * xc_B12 * beta_fn)  # (Nm, Nz)

        # y0_param using parametric SR
        m500c_tilde = m500c * h / self.B
        y0_param = ((10.0 ** self.A_SZ)
                    * (m500c_tilde / (0.7 * 3e14)) ** self.alpha_SZ
                    * E_z[None, :] ** 2
                    * (h / 0.7) ** (-0.5))  # (Nm, Nz)

        ratio = y0_param / y0_B12  # (Nm, Nz)

        return Pe_B12 * ratio[None, :, :]


jax.tree_util.register_pytree_node(
    ParametricB12PressureProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux, children: ParametricB12PressureProfile._tree_unflatten(aux, children),
)


class HMFastTSZIterCProfileB12(Theory):
    """Battaglia-2012 profile variant: B12 SHAPE entirely fixed at hydro-calibrated
    values; samples only the FLAMINGO-style Y0--M parameters
    (A_SZ, alpha_SZ, sigma_lnY).
    """
    output = ["Cl_sz"]
    params = {
        "sigma8": 0, "omega_cdm": 0, "omega_b": 0, "H0": 0, "n_s": 0,
        "tau_reio": 0, "A_SZ": 0, "alpha_SZ": 0, "sigma_lnY": 0, "B": 0,
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
        self._prof_seed = ParametricB12PressureProfile(
            A_SZ=_FID_A_SZ, alpha_SZ=_FID_ALPHA_SZ, B=_FID_B)
        self._tsz = tSZTracer(profile=self._prof_seed)

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
        for dh, ds in ((0.0, 0.0), (0.1, 0.01)):
            c1, c2 = self._eval_cl_masked(
                _FID["H0"] + dh, _FID["omega_cdm"], _FID["omega_b"],
                _FID["ln1e10A_s"], _FID["n_s"], _FID["tau_reio"],
                _FID_A_SZ + ds, _FID_ALPHA_SZ + ds, _FID_B, _FID_SIGMA_LNY,
                include_2h_static=bool(self.include_2h),
                q_cat_static=float(self.q_cat),
            )
            block(c1); block(c2)
        self.log.info(
            "HMFastTSZIterCProfileB12: q_cat=%g  iter SR (%.4f, %.4f, %.3f)  "
            "init+JIT warmup %.3fs",
            self.q_cat, self.a_sz_sr, self.alpha_sr, self.b_sr,
            time.perf_counter() - t0)
        super().initialize()

    def _eval_cl_masked_impl(self, H0, omega_cdm, omega_b, ln1e10A_s,
                             n_s, tau_reio, A_SZ, alpha_SZ, B, sigma_lnY,
                             include_2h_static, q_cat_static):
        cosmo = self._cosmo_seed.update(
            H0=H0, omega_cdm=omega_cdm, omega_b=omega_b,
            ln1e10A_s=ln1e10A_s, n_s=n_s, tau_reio=tau_reio)
        hm = self._hm_seed.update(cosmology=cosmo)
        prof = self._prof_seed.update(A_SZ=A_SZ, alpha_SZ=alpha_SZ, B=B)
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
        A_SZ = float(p["A_SZ"]); alpha_SZ = float(p["alpha_SZ"])
        sigma_lnY = float(p["sigma_lnY"])
        B = float(p["B"])

        cl_1h, cl_2h = self._eval_cl_masked(
            H0, omega_cdm, omega_b, ln1e10A_s, n_s, tau_reio,
            A_SZ, alpha_SZ, B, sigma_lnY,
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
            "HMFastTSZIterCProfileB12: theory in %.3fs  mean(Dl)=%.3e",
            time.perf_counter() - t0, float(np.mean(Dl_1h + Dl_2h)))

    def get_Cl_sz(self):
        return self._current_state.get("Cl_sz", None)


# ============================================================================
# Native Battaglia 2012 parameterization: fit P_0 normalisation parameters
# A_P0, alpha_m_P0, alpha_z_P0 directly. All other B12 shape parameters
# (A_xc, alpha_m_xc, alpha_z_xc, A_beta, alpha_m_beta, alpha_z_beta) are held
# FIXED at the hydro-calibrated Battaglia+2012 defaults. sigma_lnY remains
# free (lognormal scatter on top of the SR mean).
# ============================================================================

class HMFastTSZIterCProfileB12Native(Theory):
    """Plain B12PressureProfile (no A10-style rescaling). Sampled:

        A_P0          B12 P_0 amplitude at (M_200c=1e14, z=0)
        alpha_m_P0    P_0 mass exponent
        alpha_z_P0    P_0 redshift exponent
        sigma_lnY     lognormal Y-M scatter (1h boost via conditional_An)

    B12 fiducials (Battaglia+2012 Table 1):
        A_P0 = 18.1, alpha_m_P0 = 0.154, alpha_z_P0 = -0.758
        A_xc = 0.497, alpha_m_xc = -0.00865, alpha_z_xc = 0.731
        A_beta = 4.35, alpha_m_beta = 0.0393, alpha_z_beta = 0.415
    """
    output = ["Cl_sz"]
    params = {
        "sigma8": 0, "omega_cdm": 0, "omega_b": 0, "H0": 0, "n_s": 0,
        "tau_reio": 0,
        "A_P0": 0, "alpha_m_P0": 0, "alpha_z_P0": 0,
        "sigma_lnY": 0,
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
        # PLAIN B12 profile (native parameterization). Other shape params left
        # at Battaglia+12 defaults; only A_P0, alpha_m_P0, alpha_z_P0 vary.
        self._prof_seed = B12PressureProfile()
        self._tsz = tSZTracer(profile=self._prof_seed)

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
        for dh, ds in ((0.0, 0.0), (0.1, 0.01)):
            c1, c2 = self._eval_cl_masked(
                _FID["H0"] + dh, _FID["omega_cdm"], _FID["omega_b"],
                _FID["ln1e10A_s"], _FID["n_s"], _FID["tau_reio"],
                18.1, 0.154 + ds, -0.758, _FID_SIGMA_LNY,
                include_2h_static=bool(self.include_2h),
                q_cat_static=float(self.q_cat),
            )
            block(c1); block(c2)
        self.log.info(
            "HMFastTSZIterCProfileB12Native: q_cat=%g  iter SR (%.4f, %.4f, %.3f)  "
            "init+JIT warmup %.3fs",
            self.q_cat, self.a_sz_sr, self.alpha_sr, self.b_sr,
            time.perf_counter() - t0)
        super().initialize()

    def _eval_cl_masked_impl(self, H0, omega_cdm, omega_b, ln1e10A_s,
                             n_s, tau_reio, A_P0, alpha_m_P0, alpha_z_P0,
                             sigma_lnY,
                             include_2h_static, q_cat_static):
        cosmo = self._cosmo_seed.update(
            H0=H0, omega_cdm=omega_cdm, omega_b=omega_b,
            ln1e10A_s=ln1e10A_s, n_s=n_s, tau_reio=tau_reio)
        hm = self._hm_seed.update(cosmology=cosmo)
        # Only P_0 normalisation parameters are sampled; other B12 shape
        # parameters remain at the seed (hydro-calibrated B12 defaults).
        prof = self._prof_seed.update(
            A_P0=A_P0, alpha_m_P0=alpha_m_P0, alpha_z_P0=alpha_z_P0)
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
        A_P0 = float(p["A_P0"])
        alpha_m_P0 = float(p["alpha_m_P0"])
        alpha_z_P0 = float(p["alpha_z_P0"])
        sigma_lnY = float(p["sigma_lnY"])

        cl_1h, cl_2h = self._eval_cl_masked(
            H0, omega_cdm, omega_b, ln1e10A_s, n_s, tau_reio,
            A_P0, alpha_m_P0, alpha_z_P0, sigma_lnY,
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
            "HMFastTSZIterCProfileB12Native: theory in %.3fs  mean(Dl)=%.3e",
            time.perf_counter() - t0, float(np.mean(Dl_1h + Dl_2h)))

    def get_Cl_sz(self):
        return self._current_state.get("Cl_sz", None)
