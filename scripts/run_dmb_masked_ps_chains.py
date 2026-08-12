"""Run Cobaya MCMC for DMB baryonic feedback on L1_m9 large-scale tSZ PS.

**Cosmology.** Fixed at FLAMINGO D3A (papers fix Planck; D3A for FLAMINGO maps).

**DMB priors.** ACT Table 1 (Dalal et al. 2026), all 10 free parameters.
``n_nt`` is ACT's redshift-cap (eq. 2.12), mapped to hmfast ``n_nt_zcap``;
hmfast's GODMAX ``n_nt`` (radial slope) is fixed at 0.8. ``A_starcga=0.055``
(paper f_★, not GODMAX 0.09). No A_yy — amplitude absorbed by feedback knobs.

**Selection.** Finite-q: fixed GNFW parametric SNR at covariance ``A_SZ``;
pressure is pure DMB.

Example::

    python scripts/run_dmb_masked_ps_chains.py --case fullsky --dry-run
    python scripts/run_dmb_masked_ps_chains.py --case fullsky
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.run_masked_ps_chains import (  # noqa: E402
    CASES,
    data_files,
    gpu_devices,
    load_converged_artifacts,
)

from flamingo.inference.dmb_ps import (  # noqa: E402
    ALL_FREE_PARAMS,
    D3A_VALUES,
    DMB_TABLE1_PRIORS,
    FIXED_PROFILE,
    PRIMARY_DMB_DEFAULTS,
    SAMPLED_DMB_PARAMS,
    SELECTION_A_SZ,
    SELECTION_ALPHA_SZ,
    SELECTION_B,
)

CHAINS = REPO / "chains" / "l1_m9_qfrommap_dmb"
PREFLIGHT_FILE = CHAINS / "preflight.json"

#: Cobaya latex + proposal/ref around ACT Table-1 prior midpoints.
_LATEX = {
    "log10_Mc0": r"\log_{10} M_{\mathrm{c},0}",
    "nu_z": r"\nu_z",
    "mu_beta": r"\mu_{\beta}",
    "theta_ej_0": r"\theta_{\mathrm{ej},0}",
    "gamma_rhogas": r"\gamma",
    "delta_rhogas": r"\delta",
    "alpha_nt": r"\alpha_{\mathrm{nt}}",
    "n_nt": r"n_{\mathrm{nt}}",
    "eta_star": r"\eta_{*}",
    "delta_eta": r"\delta\eta",
    "sigma_lnY": r"\sigma_{\ln Y}",
}

_PROPOSAL = {
    "log10_Mc0": 0.15,
    "nu_z": 0.15,
    "mu_beta": 0.15,
    "theta_ej_0": 0.4,
    "gamma_rhogas": 0.25,
    "delta_rhogas": 0.6,
    "alpha_nt": 0.04,
    "n_nt": 0.05,
    "eta_star": 0.02,
    "delta_eta": 0.04,
    "sigma_lnY": 0.02,
}


def parameters() -> dict:
    """ACT Table 1 DMB knobs (10 free params); cosmology fixed at D3A."""
    params = {}
    for name in ALL_FREE_PARAMS:
        prior = dict(DMB_TABLE1_PRIORS[name])
        mid = PRIMARY_DMB_DEFAULTS[name]
        prop = _PROPOSAL[name]
        params[name] = {
            "prior": prior,
            "ref": {"dist": "norm", "loc": mid, "scale": prop},
            "proposal": prop,
            "latex": _LATEX[name],
        }
    return params


PRIOR_SOURCE = (
    "ACT Table 1 (Dalal et al. 2026), all 10 free parameters. n_nt is ACT's "
    "redshift-cap (eq. 2.12), mapped to hmfast n_nt_zcap; hmfast GODMAX n_nt "
    "(radial slope) fixed at 0.8. A_starcga=0.055 (paper f_star, not GODMAX "
    "0.09). No A_yy. Cosmology fixed at FLAMINGO D3A."
)

def build_info(
    case: str,
    *,
    artifacts: dict | None = None,
    rminus1: float = 0.05,
    max_samples: int | None = None,
    burn_in: int = 50,
    covariance_override: str | Path | None = None,
    data_override: str | Path | None = None,
) -> dict:
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}; expected one of {tuple(CASES)}")
    artifacts = load_converged_artifacts() if artifacts is None else artifacts
    data, covariance = data_files(case, artifacts["covariance_paths"])
    if covariance_override is not None:
        covariance = Path(covariance_override)
    if data_override is not None:
        data = Path(data_override)
    for path in (data, covariance):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    sampler_mcmc: dict = {
        "Rminus1_stop": float(rminus1),
        "drag": False,
        "proposal_scale": 1.5,
        "learn_every": 40,
        "learn_proposal": True,
        "learn_proposal_Rminus1_max": 100.0,
        "learn_proposal_Rminus1_max_early": 100.0,
        "max_tries": 100000,
        "burn_in": int(burn_in),
    }
    if max_samples is not None:
        sampler_mcmc["max_samples"] = int(max_samples)

    return {
        "output": str(CHAINS / case / "chain"),
        "likelihood": {
            "flamingo.inference.masked_ps.MaskedBandPowerLikelihood": {
                "data_file": str(data),
                "covariance_file": str(covariance),
                "data_scale": 1e-12,
            }
        },
        "theory": {
            "flamingo.inference.dmb_ps.DMBTSZTheory": {
                "q_cat": CASES[case],
                "num_points_trapz_int": 32,
            }
        },
        "params": parameters(),
        "sampler": {"mcmc": sampler_mcmc},
        "timing": True,
        "resume": True,
        "seed": 20260811,
    }


def write_preflight(
    cases: tuple[str, ...],
    artifacts: dict,
    *,
    output_file: Path = PREFLIGHT_FILE,
    rminus1: float = 0.05,
    max_samples: int | None = None,
) -> dict:
    resolved = {}
    for case in cases:
        info = build_info(
            case, artifacts=artifacts, rminus1=rminus1, max_samples=max_samples
        )
        likelihood = next(iter(info["likelihood"].values()))
        free = [n for n, cfg in info["params"].items() if "prior" in cfg]
        resolved[case] = {
            "q_cat": CASES[case],
            "data_file": likelihood["data_file"],
            "covariance_file": likelihood["covariance_file"],
            "output": info["output"],
            "theory": next(iter(info["theory"].keys())),
            "free_params": free,
        }
    preflight = {
        "jax_devices": gpu_devices(),
        "cosmology": "D3A_fixed",
        "d3a_fixed": dict(D3A_VALUES),
        "prior_source": PRIOR_SOURCE,
        "dmb_table1_priors": {k: dict(v) for k, v in DMB_TABLE1_PRIORS.items()},
        "primary_dmb_defaults": dict(PRIMARY_DMB_DEFAULTS),
        "fixed_profile": dict(FIXED_PROFILE),
        "n_free_dmb": len(SAMPLED_DMB_PARAMS),
        "n_free_total": len(ALL_FREE_PARAMS),
        "selection": {
            "A_SZ": SELECTION_A_SZ,
            "alpha_SZ": SELECTION_ALPHA_SZ,
            "B": SELECTION_B,
            "note": (
                "Finite-q: fixed GNFW parametric SNR at covariance A_SZ; "
                "pressure is full DMB Table-1 model."
            ),
        },
        "mass_definition_tsz": "M_500c",
        "mass_definition_pk": "M_200c",
        "cases": resolved,
        "params": parameters(),
        "sampler": {"Rminus1_stop": rminus1, "max_samples": max_samples},
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    return preflight


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=tuple(CASES))
    parser.add_argument("--Rminus1-stop", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--covariance-override", type=Path, default=None,
                        help="Use this covariance file instead of the converged analytical one")
    parser.add_argument("--data-override", type=Path, default=None,
                        help="Use this data file instead of the default 18-bin one")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    selected_cases = tuple(args.case or ("fullsky",))
    artifacts = load_converged_artifacts()
    output_file = (
        PREFLIGHT_FILE
        if args.dry_run or len(selected_cases) != 1
        else CHAINS / selected_cases[0] / "preflight.json"
    )
    preflight = write_preflight(
        selected_cases,
        artifacts,
        output_file=output_file,
        rminus1=args.Rminus1_stop,
        max_samples=args.max_samples,
    )
    if args.dry_run:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    os.environ.setdefault("FLAMINGO_ROOT", str(REPO))
    from cobaya.run import run

    for case in selected_cases:
        print(f"\n=== DMB MCMC {case} (q_cat={CASES[case]}) ===", flush=True)
        print(f"devices: {gpu_devices()}", flush=True)
        print(
            f"n_free={len(ALL_FREE_PARAMS)} "
            f"(ACT Table 1, all 10 params; "
            f"n_nt→n_nt_zcap [ACT eq.2.12], "
            f"GODMAX n_nt fixed={FIXED_PROFILE['n_nt']}, "
            f"A_starcga={FIXED_PROFILE['A_starcga']})",
            flush=True,
        )
        print(f"prior source: {PRIOR_SOURCE}", flush=True)
        (CHAINS / case).mkdir(parents=True, exist_ok=True)
        run(
            build_info(
                case,
                artifacts=artifacts,
                rminus1=args.Rminus1_stop,
                max_samples=args.max_samples,
                burn_in=args.burn_in,
                covariance_override=args.covariance_override,
                data_override=args.data_override,
            )
        )


if __name__ == "__main__":
    main()
