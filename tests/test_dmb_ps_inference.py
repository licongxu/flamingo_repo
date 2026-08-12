"""Tests for DMB tSZ inference: ACT Table-1 (10 params) + ACT eq. 2.12 n_nt."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
FULLSKY_DATA = REPO / "data_paper" / "binned_bandpowers" / "Dl_yy_L1_m9_fullsky_binned_18.txt"
FULLSKY_COV = (
    REPO
    / "chains"
    / "l1_m9_qfrommap_asz_covariance"
    / "final"
    / "covariance"
    / "cov_full_L1_m9_customgnfw_qfrommap_fullsky_Dl_yy_binned_18.npy"
)


def test_act_table1_all_10_params():
    """ACT Table 1 (Dalal et al. 2026): 10 free params, no A_yy.

    n_nt is ACT's redshift-cap (eq. 2.12), mapped to hmfast n_nt_zcap.
    GODMAX n_nt (radial slope) is fixed at 0.8.
    """
    from flamingo.inference.dmb_ps import (
        ALL_FREE_PARAMS,
        DMB_TABLE1_PRIORS,
        FIXED_PROFILE,
        SAMPLED_DMB_PARAMS,
        profile_kwargs_from_sampled,
        PRIMARY_DMB_DEFAULTS,
    )

    assert "A_yy" not in ALL_FREE_PARAMS
    assert "A_yy" not in SAMPLED_DMB_PARAMS
    assert "A_yy" not in DMB_TABLE1_PRIORS
    assert "n_nt" in SAMPLED_DMB_PARAMS
    assert "n_nt" in ALL_FREE_PARAMS
    assert "n_nt" in DMB_TABLE1_PRIORS
    assert DMB_TABLE1_PRIORS["n_nt"] == {"min": 0.6, "max": 1.0}
    assert FIXED_PROFILE["n_nt"] == 0.8
    assert FIXED_PROFILE["A_starcga"] == 0.055
    # 10 DMB+baryon knobs + σ_lny = 11 total (no A_yy)
    assert len(SAMPLED_DMB_PARAMS) == 11
    assert len(ALL_FREE_PARAMS) == 11
    assert DMB_TABLE1_PRIORS["log10_Mc0"] == {"min": 13.0, "max": 15.0}

    kw = profile_kwargs_from_sampled(PRIMARY_DMB_DEFAULTS)
    # n_nt (ACT redshift-cap) maps to n_nt_zcap (triggers ACT eq. 2.12)
    assert kw["n_nt_zcap"] == 0.8
    # GODMAX n_nt (radial slope) is fixed at 0.8
    assert kw["n_nt"] == 0.8
    assert kw["A_starcga"] == 0.055
    assert kw["beta_nt"] == 0.5


def test_no_amplitude_param_exported():
    """AMPLITUDE_PARAM should no longer be exported."""
    import flamingo.inference.dmb_ps as mod

    assert not hasattr(mod, "AMPLITUDE_PARAM")


def test_pk_suppression_godmax_mtot():
    """GODMAX S(k): same-Mtot Hankel NFW, no analytic-NFW amp hack."""
    from flamingo.inference.dmb_ps import evaluate_pk_suppression

    result = evaluate_pk_suppression(
        theta_ej_0=4.0,
        log10_Mc0=14.83,
        mu_beta=0.21,
        gamma_rhogas=2.0,
        delta_rhogas=7.0,
        alpha_nt=0.18,
        n_nt=0.8,
        eta_star=0.3,
        delta_eta=0.3,
        A_starcga=0.09,
    )
    k, s = result["k_h"], result["ratio"]
    assert k[0] <= 0.15 and k[-1] >= 5.0
    # Same Mtot ⇒ P1h_nfw/P1h_dmb ≈ 1 at low k (old analytic-NFW bug was ~0.27).
    assert abs(float(result["amp_calib"][0]) - 1.0) < 0.08
    assert abs(s[0] - 1.0) < 0.03
    i1 = int(np.argmin(np.abs(k - 1.0)))
    assert 0.70 < s[i1] < 1.05, f"S(k=1)={s[i1]} not GODMAX-like"
    # Must not be the old 1h-only crash (~0.35).
    assert s.min() > 0.50


def test_prior_predictive_s_band_is_wide():
    """Large-scale tSZ should leave S(k) weakly constrained → prior S span is wide."""
    from flamingo.inference.dmb_ps import DMB_TABLE1_PRIORS, evaluate_pk_suppression

    rng = np.random.default_rng(0)
    stack = []
    for _ in range(25):
        p = {}
        for name, pr in DMB_TABLE1_PRIORS.items():
            if name == "sigma_lnY":
                continue  # not in S(k)
            p[name] = float(rng.uniform(pr["min"], pr["max"]))
        stack.append(evaluate_pk_suppression(**p)["ratio"])
    stack = np.vstack(stack)
    i1 = int(np.argmin(np.abs(evaluate_pk_suppression()["k_h"] - 1.0)))
    span = float(stack[:, i1].max() - stack[:, i1].min())
    assert span > 0.05, f"prior S span at k=1 only {span:.3f}"


def test_mcmc_frees_table1_10_params(tmp_path, monkeypatch):
    from scripts import run_dmb_masked_ps_chains as runner

    cov = FULLSKY_COV if FULLSKY_COV.is_file() else tmp_path / "cov.npy"
    if not Path(cov).is_file():
        np.save(cov, np.eye(18))
    artifacts = {
        "A_SZ": -4.1,
        "covariance_paths": {c: cov for c in runner.CASES},
        "metadata": {"A_SZ": -4.1},
        "summary": {"fixed_parameters": {}, "final_metadata_path": str(tmp_path / "m.json")},
    }
    monkeypatch.setattr(runner, "load_converged_artifacts", lambda: artifacts)
    monkeypatch.setattr(runner, "gpu_devices", lambda: ["cuda:0"])
    info = runner.build_info("fullsky", artifacts=artifacts, max_samples=3)
    free = [n for n, c in info["params"].items() if "prior" in c]
    assert "A_yy" not in free
    assert "n_nt" in free
    assert set(free) == set(runner.ALL_FREE_PARAMS)
    assert len(free) == 11
    assert runner.FIXED_PROFILE["n_nt"] == 0.8
    assert runner.FIXED_PROFILE["A_starcga"] == 0.055
