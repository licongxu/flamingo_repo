"""Run five one-parameter GPU Cobaya chains for the L1_m9 qfrommap spectra."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.inference.iterate_l1_m9_asz_covariance import FIXED  # noqa: E402
from scripts.inference.run_masked_ps_chains import (  # noqa: E402
    CASES,
    data_files,
    gpu_devices,
    load_converged_artifacts,
)

CHAINS = REPO / "chains" / "l1_m9_qfrommap_asz_only"
PREFLIGHT_FILE = CHAINS / "preflight.json"
ASZ_PRIOR = {"dist": "norm", "loc": -4.09532, "scale": 0.03}


def parameters() -> dict:
    """Return D3A/profile parameters fixed except for ``A_SZ``."""
    params = {
        name: {"value": value}
        for name, value in FIXED.items()
    }
    params["A_SZ"] = {
        "prior": dict(ASZ_PRIOR),
        "ref": {"dist": "norm", "loc": ASZ_PRIOR["loc"], "scale": 0.01},
        "proposal": 0.01,
        "latex": r"A_\mathrm{SZ}",
    }
    return params


def _validate_covariance_parameter_point(artifacts: dict) -> None:
    """Require covariances evaluated at the approved full-sky best fit."""
    recorded = artifacts["summary"].get("fixed_parameters")
    if recorded != FIXED:
        raise ValueError("covariance fixed parameters do not match the approved D3A point")
    if float(artifacts["metadata"]["A_SZ"]) != float(artifacts["A_SZ"]):
        raise ValueError("covariance metadata and full-sky best-fit A_SZ disagree")


def build_info(
    case: str,
    *,
    artifacts: dict | None = None,
    rminus1: float = 0.005,
) -> dict:
    """Build one A_SZ-only configuration with a fixed q-specific covariance."""
    artifacts = load_converged_artifacts() if artifacts is None else artifacts
    _validate_covariance_parameter_point(artifacts)
    data, covariance = data_files(case, artifacts["covariance_paths"])
    for path in (data, covariance):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
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
            "flamingo.inference.masked_ps.MaskedTSZTheory": {"q_cat": CASES[case]}
        },
        "params": parameters(),
        "sampler": {
            "mcmc": {
                "Rminus1_stop": rminus1,
                "drag": False,
                "proposal_scale": 1.5,
                "learn_every": 40,
                "learn_proposal": True,
                "learn_proposal_Rminus1_max": 100.0,
                "learn_proposal_Rminus1_max_early": 100.0,
                "max_tries": 100000,
                "burn_in": 50,
            }
        },
        "timing": True,
        "resume": True,
        "seed": 20260802,
    }


def write_preflight(
    cases: tuple[str, ...],
    artifacts: dict,
    *,
    output_file: Path = PREFLIGHT_FILE,
) -> dict:
    """Record the fixed covariance point separately from the fit prior."""
    _validate_covariance_parameter_point(artifacts)
    resolved = {}
    for case in cases:
        info = build_info(case, artifacts=artifacts)
        likelihood = next(iter(info["likelihood"].values()))
        resolved[case] = {
            "q_cat": CASES[case],
            "data_file": likelihood["data_file"],
            "covariance_file": likelihood["covariance_file"],
            "output": info["output"],
        }
    preflight = {
        "jax_devices": gpu_devices(),
        "covariance_parameter_point": {
            **artifacts["summary"]["fixed_parameters"],
            "A_SZ": float(artifacts["A_SZ"]),
        },
        "covariance_metadata_path": artifacts["summary"]["final_metadata_path"],
        "fit_prior": {"A_SZ": dict(ASZ_PRIOR)},
        "cases": resolved,
        "params": parameters(),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    return preflight


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=tuple(CASES))
    parser.add_argument("--Rminus1-stop", type=float, default=0.005)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    selected_cases = tuple(args.case or CASES)
    artifacts = load_converged_artifacts()
    output_file = (
        PREFLIGHT_FILE
        if args.dry_run or len(selected_cases) != 1
        else CHAINS / selected_cases[0] / "preflight.json"
    )
    preflight = write_preflight(selected_cases, artifacts, output_file=output_file)
    if args.dry_run:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    os.environ.setdefault("FLAMINGO_ROOT", "/scratch/scratch-lxu/flamingo_repo")
    from cobaya.run import run

    for case in selected_cases:
        print(f"\n=== {case} (q_cat={CASES[case]}) ===", flush=True)
        (CHAINS / case).mkdir(parents=True, exist_ok=True)
        run(build_info(case, artifacts=artifacts, rminus1=args.Rminus1_stop))


if __name__ == "__main__":
    main()
