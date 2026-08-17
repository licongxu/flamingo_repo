"""q>5 masked tSZ chain with the CNC scalrel priors, D3A cosmology fixed."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.run_l1_m9_cnc_qgt5_chain import (  # noqa: E402
    FIXED,
    HMFAST_SRC,
    parameters as cnc_parameters,
)
from scripts.run_masked_ps_chains import data_files, load_converged_artifacts  # noqa: E402

CHAINS = REPO / "chains" / "l1_m9_qfrommap_ps_qgt5_scalrel"
CASE = "qgt5"
THEORY_PARAMS = (
    "H0",
    "omega_cdm",
    "omega_b",
    "n_s",
    "sigma_8",
    "tau_reio",
    "B",
    "A_SZ",
    "alpha_SZ",
    "sigma_lnY",
)


def parameters() -> dict:
    """Same sampled scalrel priors as the CNC q>5 chain; cosmology held at D3A."""
    return {name: cnc_parameters()[name] for name in THEORY_PARAMS}


def _validate_covariance_point(artifacts: dict) -> None:
    recorded = artifacts["summary"]["fixed_parameters"]
    for name, value in FIXED.items():
        if recorded[name] != value:
            raise ValueError(f"covariance {name}={recorded[name]!r} != D3A {value!r}")
    if float(artifacts["metadata"]["A_SZ"]) != float(artifacts["A_SZ"]):
        raise ValueError("covariance metadata and full-sky best-fit A_SZ disagree")


def build_info(
    *,
    artifacts: dict | None = None,
    rminus1: float = 0.01,
) -> dict:
    """q>5 bandpowers + the same covariance used by the A_SZ-only PS chains."""
    artifacts = load_converged_artifacts() if artifacts is None else artifacts
    _validate_covariance_point(artifacts)
    data, covariance = data_files(CASE, artifacts["covariance_paths"])
    for path in (data, covariance):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    return {
        "output": str(CHAINS / "chain"),
        "likelihood": {
            "flamingo.inference.masked_ps.MaskedBandPowerLikelihood": {
                "data_file": str(data),
                "covariance_file": str(covariance),
                "data_scale": 1e-12,
            }
        },
        "theory": {
            "flamingo.inference.masked_ps.MaskedTSZTheory": {"q_cat": 5.0}
        },
        "params": parameters(),
        "sampler": {
            "mcmc": {
                "Rminus1_stop": rminus1,
                "drag": False,
                "proposal_scale": 1.2,
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
        "seed": 20260817,
    }


def _prepare_runtime() -> None:
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("FLAMINGO_ROOT", str(REPO))
    sys.path.insert(0, str(HMFAST_SRC))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Rminus1-stop", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    artifacts = load_converged_artifacts()
    info = build_info(artifacts=artifacts, rminus1=args.Rminus1_stop)
    if args.dry_run:
        import json

        print(json.dumps(info, indent=2, sort_keys=True))
        return
    _prepare_runtime()
    CHAINS.mkdir(parents=True, exist_ok=True)
    from cobaya.run import run

    run(info)


if __name__ == "__main__":
    main()
