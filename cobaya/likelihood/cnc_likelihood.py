"""Cobaya likelihood for binned CNC of the FLAMINGO SOAP SNR-d3a catalogue.

Same Poisson chi^2 as cosmocnc_jax tutorial `cobaya_planck_scatter_jax.py`, but:
  * uses cosmology_tool="hmfast" (hmfast emulator HMF; faster and matches our tSZ PS chain)
  * accepts `sigma8` directly as the amplitude parameter (cosmocnc_jax has an
    internal sigma_8 -> A_s solver), no Cobaya-side rescaling needed
  * data file is the FLAMINGO 10x5 N(z, q) matrix in the local data/ dir.
"""
from __future__ import annotations

import os
import time

import numpy as np
from cobaya.likelihood import Likelihood

# JAX/TF init: same idiom as the cosmocnc_jax tutorial
os.environ["JAX_ENABLE_X64"] = "1"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")
if "xla_gpu_persistent_cache_dir" in os.environ.get("XLA_FLAGS", ""):
    os.environ.pop("XLA_FLAGS")
_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf  # noqa: E402
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_visible_devices
if os.environ.get("HMFAST_COBAYA_USE_GPU", "1").strip() in ("1", "true", "True", "yes", "YES"):
    os.environ["JAX_PLATFORMS"] = "cuda"

from cosmocnc_jax import cluster_number_counts  # noqa: E402
from cosmocnc_jax.params import (  # noqa: E402
    cnc_params_default, cosmo_params_default, scaling_relation_params_default,
)


class CNCBinnedFlamingoLikelihood(Likelihood):
    # Data / survey
    data_file: str = "data/N2d_z_q_bin.txt"
    survey_sr: str = (
        "/scratch/scratch-lxu/agent_dev/auto_research_agent/cosmocnc_jax/"
        "cosmocnc_jax/surveys/survey_sr_planck_sim.py"
    )
    survey_cat: str = (
        "/scratch/scratch-lxu/agent_dev/auto_research_agent/cosmocnc_jax/"
        "cosmocnc_jax/surveys/survey_cat_planck_sim.py"
    )
    tszsbi_noise_dir: str = "/scratch/scratch-lxu/tszsbi/noise_files"
    tszsbi_filter_name: str = "immf6"
    lambda_floor: float = 1.0e-12

    # Binning (must match data file shape)
    z_min: float = 0.005
    z_max: float = 1.0
    n_z_bins: int = 10
    q_min: float = 5.0
    q_max: float = 40.0
    n_q_bins: int = 5

    # Theory config
    n_points: int = 2048
    n_z: int = 50
    M_min: float = 1.0e14
    M_max: float = 1.0e16
    f_sky: float = 1.0
    M_pivot: float = 2.1e14

    # hmfast backend
    hmfast_path: str = "/scratch/scratch-lxu/agent_dev/auto_research_agent/hmfast/src"
    # "sigma_8" (sample sigma8) or "A_s" (sample ln10_10A_s / pass A_s)
    cosmo_amplitude_parameter: str = "sigma_8"

    def initialize(self):
        # Allow `data_file` relative to this likelihood module's directory.
        if not os.path.isabs(self.data_file):
            here = os.path.dirname(os.path.abspath(__file__))
            self.data_file = os.path.join(here, self.data_file)
        self.n_obs = np.loadtxt(self.data_file, dtype=float)
        if self.n_obs.shape != (self.n_z_bins, self.n_q_bins):
            raise ValueError(
                f"Observed count shape {self.n_obs.shape} != "
                f"(n_z_bins, n_q_bins)=({self.n_z_bins}, {self.n_q_bins})."
            )
        self.bin_edges_z = np.linspace(self.z_min, self.z_max, self.n_z_bins + 1)
        self.bin_edges_q = np.exp(
            np.linspace(np.log(self.q_min), np.log(self.q_max), self.n_q_bins + 1)
        )

        cnc_params = dict(cnc_params_default)
        cnc_params["survey_sr"] = self.survey_sr
        cnc_params["survey_cat"] = self.survey_cat
        cnc_params["tszsbi_noise_dir"] = self.tszsbi_noise_dir
        cnc_params["tszsbi_filter_name"] = self.tszsbi_filter_name

        cnc_params["load_catalogue"] = False
        cnc_params["likelihood_type"] = "binned"
        cnc_params["binned_lik_type"] = "z_and_obs_select"
        cnc_params["data_lik_from_abundance"] = False

        cnc_params["obs_select"] = "q_planck_sim"
        cnc_params["observables"] = [["q_planck_sim"]]
        cnc_params["obs_select_min"] = self.q_min
        cnc_params["obs_select_max"] = self.q_max
        cnc_params["z_min"] = self.z_min
        cnc_params["z_max"] = self.z_max
        cnc_params["bins_edges_z"] = self.bin_edges_z
        cnc_params["bins_edges_obs_select"] = self.bin_edges_q

        cnc_params["n_points"] = int(self.n_points)
        cnc_params["n_z"] = int(self.n_z)
        cnc_params["M_min"] = float(self.M_min)
        cnc_params["M_max"] = float(self.M_max)
        cnc_params["planck_sim_M_pivot"] = float(self.M_pivot)

        # *** hmfast backend (matches tSZ PS chain) ***
        cnc_params["cosmology_tool"] = "hmfast"
        cnc_params["hmfast_path"] = self.hmfast_path
        cnc_params["hmf_calc"] = "cnc"
        cnc_params["cosmo_param_density"] = "physical"
        amp = self.cosmo_amplitude_parameter
        if amp not in ("sigma_8", "A_s"):
            raise ValueError(f"cosmo_amplitude_parameter must be 'sigma_8' or 'A_s', got {amp!r}")
        cnc_params["cosmo_amplitude_parameter"] = amp
        cnc_params["cosmocnc_verbose"] = "none"

        self.cnc = cluster_number_counts(cnc_params=cnc_params)
        self.cosmo_base = dict(cosmo_params_default)
        self.scal_base = dict(scaling_relation_params_default)
        self.cnc.cosmo_params = dict(self.cosmo_base)
        self.cnc.scal_rel_params = dict(self.scal_base)
        self.cnc.initialise()
        self._eval_counter = 0
        self.log.info(
            "CNC: N_obs sum=%d, shape=%s, hmfast=%s",
            int(self.n_obs.sum()), self.n_obs.shape, self.hmfast_path,
        )

    def get_requirements(self):
        return {}

    def logp(self, **p):
        t0 = time.perf_counter()
        cosmo = dict(self.cosmo_base)
        scal = dict(self.scal_base)

        # Cobaya parameter names -> cosmocnc_jax keys
        if "h" in p:        cosmo["h"] = float(p["h"])
        if "H0" in p:       cosmo["h"] = float(p["H0"]) / 100.0
        if "omega_b" in p:  cosmo["Ob0h2"] = float(p["omega_b"])
        if "omega_cdm" in p:
            cosmo["Oc0h2"] = float(p["omega_cdm"])
        if "n_s" in p:      cosmo["n_s"] = float(p["n_s"])
        if "tau_reio" in p: cosmo["tau_reio"] = float(p["tau_reio"])
        if "m_nu" in p:     cosmo["m_nu"] = float(p["m_nu"])
        if "ln10_10A_s" in p:
            cosmo["A_s"] = float(np.exp(p["ln10_10A_s"]) / 1e10)
        elif "sigma8" in p:
            cosmo["sigma_8"] = float(p["sigma8"])

        # Scaling-relation: tSZ PS chain uses A_SZ, alpha_SZ, sigma_lnY, B
        if "A_SZ" in p:      scal["A_szifi"] = float(p["A_SZ"])
        if "alpha_SZ" in p:  scal["alpha_szifi"] = float(p["alpha_SZ"])
        if "sigma_lnY" in p: scal["sigma_lnq_szifi"] = float(p["sigma_lnY"])
        if "B" in p:         scal["bias_sz"] = 1.0 / float(p["B"])
        if "one_minus_b" in p: scal["bias_sz"] = float(p["one_minus_b"])

        self.cnc.update_params(cosmo, scal)
        _ = self.cnc.get_log_lik_binned()
        lam = np.asarray(self.cnc.n_binned, dtype=float)
        if lam.shape != self.n_obs.shape:
            raise ValueError(f"Theory bin shape {lam.shape} != obs shape {self.n_obs.shape}")
        lam = np.clip(lam, self.lambda_floor, None)
        log_like = float(np.sum(-lam + self.n_obs * np.log(lam)))
        self._eval_counter += 1
        self.log.info(
            "CNC %d  loglike=%.3f  N_theory=%.1f  N_obs=%d  dt=%.3fs",
            self._eval_counter, log_like, float(lam.sum()), int(self.n_obs.sum()),
            time.perf_counter() - t0,
        )
        return log_like
