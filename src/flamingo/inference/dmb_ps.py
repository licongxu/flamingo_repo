"""DMB (GODMAX BCM) theory for large-scale FLAMINGO L1_m9 masked tSZ bandpowers.

Architecture
------------
Full sky is the **masked** path with ``q_cat = +∞`` (same structure as
:class:`~flamingo.inference.masked_ps.MaskedTSZTheory`):

* 1-halo: ``cl_1h_masked`` × ``⟨A² 1(q_obs < q_cat)⟩`` (``n_power=2``)
* 2-halo: ``cl_2h_masked`` × ``⟨A 1(q_obs < q_cat)⟩`` per bracket (``n_power=1``)

Scatter only via :func:`~hmfast.tracers.tsz_completeness.conditional_An_undetected`.

For **finite** ``q_cat``, the mean-SNR field used in the scatter integral is the
**fixed** custom-GNFW parametric SNR at the prepared covariance point
(``A_SZ``, ``alpha_SZ=1.12``, ``B=1.41``). That matches the qfrommap / covariance
selection recipe; the **pressure** profile is pure DMB. DMB params do not
re-enter the selection (map-level masks are data-side).

Mass / cosmology
----------------
* tSZ halo model: explicit **``M_500c``** (same as GNFW masked-PS pipeline).
* Cosmology: **fixed FLAMINGO D3A** — To et al. (2024) and Dalal et al. (2026)
  fix a Planck cosmology for their tSZ/cluster DMB fits; we fix D3A for
  FLAMINGO maps.

Priors — ACT Table 1 knobs (Dalal et al. 2026)
-------------------------------------------------------------
Dalal et al. (2026) Table 1 flats (fiducial = prior midpoint):

* ``log10 Mc0`` → ``log10_Mc0``     flat(13, 15)
* ``νz``        → ``nu_z``          flat(-1, 1)
* ``μβ``        → ``mu_beta``       flat(0, 2)
* ``θej``       → ``theta_ej_0``    flat(2, 8)
* ``γ``         → ``gamma_rhogas``  flat(1, 4)
* ``δ``         → ``delta_rhogas``  flat(3, 11)
* ``αnt``       → ``alpha_nt``      flat(0.01, 0.4)
* ``nnt``       → ``n_nt``          flat(0.6, 1.0)   (ACT redshift-cap, eq. 2.12)
* ``η*``        → ``eta_star``      flat(0.15, 0.3)
* ``δη``        → ``delta_eta``     flat(0.05, 0.4)  with ``eta_cga = eta_star + delta_eta``
* ``σln y``     → ``sigma_lnY``     flat(0.1, 0.3)   (selection scatter; inert at q=∞)

**ACT eq. 2.12 non-thermal pressure** (implemented in hmfast via
``n_nt_zcap``):

    P_th/P_tot = max[0, 1 - α_nt * f(z) * (r/R_500c)^0.8]
    f(z) = min[(1+z)^0.5, (4^{-n_nt/α_nt} - 1)*tanh(0.5z) + 1]

The radial index is **fixed at 0.8**, redshift power at **0.5**, tanh
slope at **0.5**. ``n_nt`` controls the high-z cap ``4^{-n_nt/α_nt}``.
hmfast's GODMAX ``n_nt`` (radial slope) is fixed at 0.8 and **not sampled**;
the free ``n_nt`` maps to hmfast's ``n_nt_zcap``.

**Stellar amplitude:** papers use
``f_★ = 0.055 (2.5×10^{11} h^{-1} M_⊙ / M)^η``.
hmfast default is GODMAX ``A_starcga = 0.09``. We force ``A_starcga = 0.055``
so the stellar pivot matches ACT/To (``log10 M1 = 11.4`` is already the default).

P(k) suppression
----------------
Copy GODMAX ``setup_power_spectra_jit``:

``S = (P1h_dmb + b_dmb² P_lin) / (P1h_nfw + b_nfw² P_lin)``

Both windows are Hankel transforms of GODMAX densities divided by the
**same** ``Mtot`` (truncated NFW mass to ``16 R200``). The NFW counterpart
is :class:`~hmfast.halos.profiles.dmb.DMBNFWMatterProfile`, not analytic
``NFWMatterProfile`` (that one uses ``M200c`` and breaks ``S→1``).
``M_200c`` halo model, ``k`` in ``h Mpc^{-1}``, ``z=0``.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
from cobaya.theory import Theory

from ..catalogue.frame import D3A_COSMOLOGY
from ..cnc import FILTER_NAME, SIGMA_Y0_FILE, SKYFRACS_FILE
from .l1_m9 import (
    ELL_SMOOTH,
    MASS_GRID,
    REDSHIFT_GRID,
    _bin_dl,
    gaussian_loglike,
    load_bandpower_likelihood,
)
from .masked_ps import (
    SCATTER_GRID,
    SCATTER_NSIG,
    MaskedBandPowerLikelihood,
)

#: Fixed selection amplitude (covariance / qfrommap best-fit point).
SELECTION_A_SZ = -4.1075073
SELECTION_ALPHA_SZ = 1.12
SELECTION_B = 1.41

#: Profile knobs passed to DMBPressureProfile / DMBMatterProfile (no sigma_lnY).
PROFILE_PARAM_NAMES = (
    "theta_ej_0",
    "log10_Mc0",
    "mu_beta",
    "nu_z",
    "gamma_rhogas",
    "delta_rhogas",
    "alpha_nt",
    "n_nt",
    "n_nt_zcap",
    "eta_star",
    "eta_cga",  # set from eta_star + delta_eta
    "A_starcga",
    "beta_nt",
)

#: Free DMB knobs = ACT Table 1 (Dalal et al. 2026), 10 parameters.
#: ``n_nt`` is ACT's redshift-cap (eq. 2.12), mapped to hmfast ``n_nt_zcap``.
SAMPLED_DMB_PARAMS = (
    "log10_Mc0",
    "nu_z",
    "mu_beta",
    "theta_ej_0",
    "gamma_rhogas",
    "delta_rhogas",
    "alpha_nt",
    "n_nt",
    "eta_star",
    "delta_eta",
    "sigma_lnY",
)

ALL_FREE_PARAMS = SAMPLED_DMB_PARAMS

#: Fixed profile settings: GODMAX n_nt (radial slope) fixed at ACT's 0.8;
#: beta_nt (redshift power) unused in ACT mode but kept for GODMAX fallback.
#: A_starcga = paper f_★ pivot (not GODMAX 0.09).
FIXED_PROFILE = {
    "n_nt": 0.8,  # GODMAX radial slope (fixed; ACT radial index is 0.8)
    "beta_nt": 0.5,  # GODMAX redshift power (unused in ACT mode)
    "A_starcga": 0.055,  # ACT/To f_★ pivot (not GODMAX 0.09)
    "log10_M1_starcga": 11.4,  # 2.5e11 h^{-1} M_⊙
}

#: Subset used for P(k) draws (no A_yy, no sigma_lnY).
PRIMARY_DMB_PARAMS = ("theta_ej_0", "log10_Mc0", "mu_beta")

#: ACT Table 1 prior midpoints (paper fiducials = mean of prior).
#: ``n_nt`` is ACT's redshift-cap (eq. 2.12), mapped to hmfast ``n_nt_zcap``.
PRIMARY_DMB_DEFAULTS = {
    "log10_Mc0": 14.0,
    "nu_z": 0.0,
    "mu_beta": 1.0,
    "theta_ej_0": 5.0,
    "gamma_rhogas": 2.5,
    "delta_rhogas": 7.0,
    "alpha_nt": 0.205,
    "n_nt": 0.8,
    "eta_star": 0.225,
    "delta_eta": 0.225,  # → eta_cga = 0.45
    "sigma_lnY": 0.2,
}

#: ACT Table 1 flats (Dalal et al. 2026) — all 10 free parameters.
DMB_TABLE1_PRIORS = {
    "log10_Mc0": {"min": 11.0, "max": 15.0},
    "nu_z": {"min": -2.0, "max": 1.0},
    "mu_beta": {"min": 0.0, "max": 2.0},
    "theta_ej_0": {"min": 0.5, "max": 8.0},
    "gamma_rhogas": {"min": 0.5, "max": 4.0},
    "delta_rhogas": {"min": 1.0, "max": 11.0},
    "alpha_nt": {"min": 0.01, "max": 0.4},
    "n_nt": {"min": 0.6, "max": 1.0},
    "eta_star": {"min": 0.15, "max": 0.3},
    "delta_eta": {"min": 0.05, "max": 0.4},
    "sigma_lnY": {"min": 0.1, "max": 0.3},
}


def profile_kwargs_from_sampled(values: Mapping[str, float]) -> dict[str, float]:
    """Map Cobaya free params → DMBPressureProfile / DMBMatterProfile kwargs.

    Maps ACT Table-1 ``n_nt`` (redshift-cap, eq. 2.12) → hmfast ``n_nt_zcap``.
    GODMAX ``n_nt`` (radial slope) is fixed at 0.8 via FIXED_PROFILE.
    Optional overrides of fixed knobs are allowed only if passed explicitly
    in ``values`` (for GODMAX-like unit tests); free MCMC params never
    include them.
    """
    eta_star = float(values["eta_star"])
    delta_eta = float(values["delta_eta"])
    out = {
        "theta_ej_0": float(values["theta_ej_0"]),
        "log10_Mc0": float(values["log10_Mc0"]),
        "mu_beta": float(values["mu_beta"]),
        "nu_z": float(values["nu_z"]),
        "gamma_rhogas": float(values["gamma_rhogas"]),
        "delta_rhogas": float(values["delta_rhogas"]),
        "alpha_nt": float(values["alpha_nt"]),
        "eta_star": eta_star,
        "eta_cga": eta_star + delta_eta,
        # ACT n_nt (redshift-cap) → hmfast n_nt_zcap (triggers ACT eq. 2.12).
        "n_nt_zcap": float(values["n_nt"]),
        # Paper-aligned fixed profile (GODMAX radial slope + f_★ pivot).
        "n_nt": float(values.get("n_nt_radial", FIXED_PROFILE["n_nt"])),
        "beta_nt": float(values.get("beta_nt", FIXED_PROFILE["beta_nt"])),
        "A_starcga": float(values.get("A_starcga", FIXED_PROFILE["A_starcga"])),
        "log10_M1_starcga": float(
            values.get("log10_M1_starcga", FIXED_PROFILE["log10_M1_starcga"])
        ),
    }
    return out


D3A_VALUES = {
    "H0": 68.1,
    "omega_cdm": 0.11872788986038219,
    "omega_b": 0.022538784599999993,
    "n_s": 0.965,
    "sigma_8": 0.8025701499024616,
    "tau_reio": 0.0544,
}

NUM_POINTS_TRAPZ = 32
#: Redshift for suppression (Dalal et al. 2026 §5.1 use z=0).
PK_SUPPRESSION_Z = 0.0
#: k grid in **h Mpc⁻¹** as in Dalal et al. Fig. 6.
K_SUPPRESSION_H = np.geomspace(0.1, 10.0, 48)
#: Legacy physical-k grid (1/Mpc); prefer K_SUPPRESSION_H.
K_SUPPRESSION = None  # set below after h is known at call time
SIGMA_LNY = 0.2


def _halo_model_m500c(cosmology=None):
    from hmfast.halos import HaloModel
    from hmfast.halos.mass_definition import MassDefinition

    return HaloModel(
        cosmology=D3A_COSMOLOGY if cosmology is None else cosmology,
        mass_definition=MassDefinition(500, "critical"),
        convert_masses=True,
        hm_consistency=False,
    )


def _halo_model_m200c_matter(cosmology=None):
    """Matter P(k) comparison: M_200c with hm_consistency on.

    With shared 2h in ``evaluate_pk_suppression``, hm_consistency=True
    enforces S(k→0)→1 for both fiducial and posterior draws.  With
    hm_consistency=False the 1h term still contributes ~50% at k=0.1 h/Mpc
    and posterior samples can show unphysical S>1 even though the 2h term
    is shared.
    """
    from hmfast.halos import HaloModel
    from hmfast.halos.mass_definition import MassDefinition

    return HaloModel(
        cosmology=D3A_COSMOLOGY if cosmology is None else cosmology,
        mass_definition=MassDefinition(200, "critical"),
        convert_masses=True,
        hm_consistency=True,
    )


def _scatter_masks(snr, sigma_lnY: float, q_cat: float):
    from hmfast.tracers.tsz_completeness import conditional_An_undetected

    return {
        n: conditional_An_undetected(
            snr,
            sigma_lnY=float(sigma_lnY),
            q_cat=float(q_cat),
            n_power=n,
            n_grid=SCATTER_GRID,
            nsig=SCATTER_NSIG,
        )
        for n in (1, 2)
    }


def _selection_snr(halo_model, mass, redshift, noise_coeff):
    """Fixed GNFW parametric SNR at the prepared covariance A_SZ point."""
    from hmfast.tracers.tsz_completeness import build_snr_grid

    return build_snr_grid(
        halo_model,
        mass,
        redshift,
        SELECTION_A_SZ,
        SELECTION_ALPHA_SZ,
        SELECTION_B,
        coeff=noise_coeff,
    )


def evaluate_dmb_bandpowers(
    *,
    q_cat: float | None = None,
    num_points_trapz_int: int = NUM_POINTS_TRAPZ,
    cosmology=None,
    **sampled,
) -> dict[str, np.ndarray]:
    """Return binned 1h and 2h ``D_ell`` for DMB pressure at fixed D3A.

    Accepts ACT Table-1 free params (see ``PRIMARY_DMB_DEFAULTS``).
    ``q_cat=None`` → full sky (``+∞``). Finite ``q_cat`` uses fixed GNFW
    selection SNR (covariance point) for the scatter moments.
    """
    import jax.numpy as jnp
    from hmfast.halos.profiles import DMBPressureProfile
    from hmfast.tracers import tSZTracer
    from hmfast.tracers.tsz_completeness import load_sigma_y0_curve

    values = {**PRIMARY_DMB_DEFAULTS, **sampled}
    sigma_lnY = float(values["sigma_lnY"])
    halo_model = _halo_model_m500c(cosmology)
    profile = DMBPressureProfile(
        **profile_kwargs_from_sampled(values),
        num_points_trapz_int=int(num_points_trapz_int),
    )
    tracer = tSZTracer(profile=profile)

    mass = jnp.asarray(MASS_GRID)
    redshift = jnp.asarray(REDSHIFT_GRID)
    ell = jnp.asarray(ELL_SMOOTH)

    q = np.inf if q_cat is None else float(q_cat)
    if np.isfinite(q):
        coeff, _ = load_sigma_y0_curve(
            sigma_obj_file=str(SIGMA_Y0_FILE),
            skyfr_file=str(SKYFRACS_FILE),
            filter_name=FILTER_NAME,
        )
        snr = _selection_snr(halo_model, mass, redshift, jnp.asarray(coeff))
    else:
        # Values unused at q=∞; shape must match (Nm, Nz).
        snr = jnp.ones((mass.size, redshift.size), dtype=mass.dtype)

    masks = _scatter_masks(snr, sigma_lnY=sigma_lnY, q_cat=q)

    cl_1h = halo_model.cl_1h_masked(
        tracer, None, ell, mass, redshift, masks[2], k_damp=0.0
    )
    cl_2h = halo_model.cl_2h_masked(
        tracer, None, ell, mass, redshift, masks[1]
    )

    ell_np = np.asarray(ELL_SMOOTH, dtype=float)
    prefactor = ell_np * (ell_np + 1.0) / (2.0 * np.pi)
    return {
        "1h": _bin_dl(ell_np, prefactor * np.asarray(cl_1h, dtype=float)),
        "2h": _bin_dl(ell_np, prefactor * np.asarray(cl_2h, dtype=float)),
    }


#: Cache for the P(k) suppression machinery (JIT-compiled once, reused).
_pk_cache: dict = {}


def evaluate_pk_suppression(
    *,
    k_h: np.ndarray | None = None,
    z: float = PK_SUPPRESSION_Z,
    num_points_trapz_int: int = 48,
    cosmology=None,
    **sampled,
) -> dict[str, np.ndarray]:
    """Matter power suppression copied from GODMAX ``Pmm_sup_tot_mat``.

    GODMAX (``setup_power_spectra_jit``)::

        uk = FT(ρ / Mtot)                 # both DMB and truncated NFW
        P1h ∝ ∫ n(M) (Mtot uk)² dlnM
        Ptot = P1h + b(k)² P_lin          # b→1 at large scales
        S = Ptot_dmb / Ptot_nfw

    ``Mtot`` is the truncated-NFW mass to ``16 R200``, used for **both**
    profiles. Analytic ``NFWMatterProfile`` (``u(0)=M200/ρ̄``) is the wrong
    counterpart and was the source of the old ``amp_calib≈0.27`` hack.

    ``k`` is reported in **h Mpc⁻¹**. Mass limits:
    :math:`10^{12}–10^{16}\\,h^{-1}M_\\odot` in ``M_200c``.
    """
    import jax
    import jax.numpy as jnp
    from hmfast.halos.profiles import DMBMatterProfile, DMBNFWMatterProfile
    from hmfast.tracers.cmb_lensing import CMBLensingTracer

    cosmo = D3A_COSMOLOGY if cosmology is None else cosmology
    h = float(np.asarray(cosmo.H0)) / 100.0
    values = {**PRIMARY_DMB_DEFAULTS, **sampled}
    prof_kw = profile_kwargs_from_sampled(values)

    k_h_arr = np.asarray(K_SUPPRESSION_H if k_h is None else k_h, dtype=float)
    k_phys = jnp.asarray(k_h_arr * h)
    mass = jnp.asarray(np.geomspace(1e11 / h, 1e16 / h, 64))
    z_arr = jnp.asarray([float(z)])

    # Cache HaloModel + tracers so JAX compiles the kernel once.
    # Key on static args (num_points_trapz_int) that change shapes.
    cache_key = (int(num_points_trapz_int), float(z), k_h_arr.shape[0])
    if cache_key not in _pk_cache:
        halo_model = _halo_model_m200c_matter(cosmo)
        # Build base profiles with defaults (static shapes), then update
        base_kw = profile_kwargs_from_sampled(PRIMARY_DMB_DEFAULTS)
        base_kw["num_points_trapz_int"] = int(num_points_trapz_int)
        lens_nfw = CMBLensingTracer(profile=DMBNFWMatterProfile(**base_kw))
        lens_dmb = CMBLensingTracer(profile=DMBMatterProfile(**base_kw))
        _pk_cache[cache_key] = {
            "halo_model": halo_model,
            "lens_nfw": lens_nfw,
            "lens_dmb": lens_dmb,
            "k_phys": k_phys,
            "mass": mass,
            "z_arr": z_arr,
        }
    cache = _pk_cache[cache_key]

    # Update profiles with new DMB params (JIT reuses compiled kernel).
    lens_nfw = cache["lens_nfw"].update(profile=cache["lens_nfw"].profile.update(**prof_kw))
    lens_dmb = cache["lens_dmb"].update(profile=cache["lens_dmb"].profile.update(**prof_kw))

    # k_damp=0: GODMAX does not suppress the 1-halo term at low k.
    p1_nfw = np.squeeze(
        np.asarray(
            cache["halo_model"].pk_1h(
                lens_nfw, None, cache["k_phys"], cache["mass"], cache["z_arr"], k_damp=0.0
            ),
            dtype=float,
        )
    )
    p1_dmb = np.squeeze(
        np.asarray(
            cache["halo_model"].pk_1h(
                lens_dmb, None, cache["k_phys"], cache["mass"], cache["z_arr"], k_damp=0.0
            ),
            dtype=float,
        )
    )
    # Shared 2h from NFW tracer: enforces S(k→0)→1 regardless of DMB mass
    # normalization differences. This matches the original working version
    # where large-scale S(k) correctly approaches 1.
    p2 = np.squeeze(
        np.asarray(
            cache["halo_model"].pk_2h(
                lens_nfw, None, cache["k_phys"], cache["mass"], cache["z_arr"]
            ),
            dtype=float,
        )
    )

    denom = p1_nfw + p2
    numer = p1_dmb + p2
    ratio = np.divide(
        numer,
        denom,
        out=np.ones_like(numer),
        where=np.isfinite(denom) & (denom > 0),
    )
    n_lo = max(3, min(5, p1_nfw.size // 10))
    amp = float(np.median(p1_nfw[:n_lo] / np.maximum(p1_dmb[:n_lo], 1e-300)))

    return {
        "k_h": k_h_arr.copy(),
        "k": k_h_arr.copy(),
        "k_unit": "h/Mpc",
        "ratio": ratio,
        "pk_1h_nfw": p1_nfw,
        "pk_1h_dmb": p1_dmb,
        "pk_2h": p2,
        "pk_nfw": denom,
        "pk_dmb": numer,
        "amp_calib": np.asarray([amp]),
        "z": np.asarray([float(z)]),
    }


def fullsky_loglike(
    dmb_params: Mapping[str, float],
    *,
    data_file: str,
    covariance_file: str,
    data_scale: float = 1e-12,
    q_cat: float | None = None,
    **eval_kwargs,
) -> float:
    """Gaussian log-likelihood under a DMB parameter point (fixed D3A)."""
    _, observed, _, inverse = load_bandpower_likelihood(
        data_file, covariance_file, data_scale=data_scale
    )
    bp = evaluate_dmb_bandpowers(
        **{**PRIMARY_DMB_DEFAULTS, **dict(dmb_params)},
        q_cat=q_cat,
        **eval_kwargs,
    )
    total = np.asarray(bp["1h"], dtype=float) + np.asarray(bp["2h"], dtype=float)
    return gaussian_loglike(observed, total, inverse)


class DMBTSZTheory(Theory):
    """Fixed-D3A DMB masked tSZ bandpowers for Cobaya.

    Free: ACT Table-1 DMB knobs (10 params, Dalal et al. 2026) including
    ``n_nt`` (redshift-cap, eq. 2.12). Cosmology fixed at D3A.
    ``q_cat=None`` → full sky; finite → fixed GNFW SNR.
    """

    output = ["Cl_sz"]
    params = {name: None for name in ALL_FREE_PARAMS}
    q_cat: float | None = None
    num_points_trapz_int: int = NUM_POINTS_TRAPZ

    def get_requirements(self) -> dict:
        return {name: None for name in self.params}

    def initialize(self) -> None:
        import jax
        import jax.numpy as jnp
        from hmfast.halos.profiles import DMBPressureProfile
        from hmfast.tracers import tSZTracer
        from hmfast.tracers.tsz_completeness import load_sigma_y0_curve

        jax.config.update("jax_enable_x64", True)
        self._jnp = jnp
        self._halo_model = _halo_model_m500c()
        self._profile = DMBPressureProfile(
            num_points_trapz_int=int(self.num_points_trapz_int),
            **profile_kwargs_from_sampled(PRIMARY_DMB_DEFAULTS),
        )
        self._tracer = tSZTracer(profile=self._profile)
        self._mass = jnp.asarray(MASS_GRID)
        self._redshift = jnp.asarray(REDSHIFT_GRID)
        self._ell = jnp.asarray(ELL_SMOOTH)

        coeff, _ = load_sigma_y0_curve(
            sigma_obj_file=str(SIGMA_Y0_FILE),
            skyfr_file=str(SKYFRACS_FILE),
            filter_name=FILTER_NAME,
        )
        self._noise_coeff = jnp.asarray(coeff)
        q = np.inf if self.q_cat is None else float(self.q_cat)
        if np.isfinite(q):
            self._snr = _selection_snr(
                self._halo_model, self._mass, self._redshift, self._noise_coeff
            )
        else:
            self._snr = jnp.ones(
                (self._mass.size, self._redshift.size), dtype=self._mass.dtype
            )
        self._q = q
        self._current_state: dict = {}
        _ = self.evaluate_bandpowers(**PRIMARY_DMB_DEFAULTS)
        super().initialize()

    def evaluate_bandpowers(self, **values) -> dict[str, np.ndarray]:
        merged = {**PRIMARY_DMB_DEFAULTS, **{k: float(values[k]) for k in self.params}}
        profile = self._profile.update(**profile_kwargs_from_sampled(merged))
        tracer = self._tracer.update(profile=profile)
        masks = _scatter_masks(
            self._snr, sigma_lnY=float(merged["sigma_lnY"]), q_cat=self._q
        )
        cl_1h = self._halo_model.cl_1h_masked(
            tracer, None, self._ell, self._mass, self._redshift, masks[2], k_damp=0.0
        )
        cl_2h = self._halo_model.cl_2h_masked(
            tracer, None, self._ell, self._mass, self._redshift, masks[1]
        )
        ell = np.asarray(ELL_SMOOTH, dtype=float)
        prefactor = ell * (ell + 1.0) / (2.0 * np.pi)
        return {
            "1h": _bin_dl(ell, prefactor * np.asarray(cl_1h, dtype=float)),
            "2h": _bin_dl(ell, prefactor * np.asarray(cl_2h, dtype=float)),
        }

    def calculate(self, state: dict, want_derived: bool = True, **params_values) -> None:
        state["Cl_sz"] = self.evaluate_bandpowers(
            **{name: float(params_values[name]) for name in self.params}
        )
        self._current_state = state

    def get_Cl_sz(self) -> dict[str, np.ndarray] | None:
        return self._current_state.get("Cl_sz")


DMBBandPowerLikelihood = MaskedBandPowerLikelihood

__all__ = [
    "PRIMARY_DMB_PARAMS",
    "SAMPLED_DMB_PARAMS",
    "ALL_FREE_PARAMS",
    "FIXED_PROFILE",
    "PRIMARY_DMB_DEFAULTS",
    "DMB_TABLE1_PRIORS",
    "D3A_VALUES",
    "SELECTION_A_SZ",
    "SIGMA_LNY",
    "K_SUPPRESSION_H",
    "PK_SUPPRESSION_Z",
    "evaluate_dmb_bandpowers",
    "evaluate_pk_suppression",
    "fullsky_loglike",
    "DMBTSZTheory",
    "DMBBandPowerLikelihood",
]
