"""Iterate full-sky ``A_SZ`` MCMC fits and theory covariance to a fixed point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.powerspectra.compute_l1_m9_customgnfw_bestfit_covariance import (  # noqa: E402
    CASES,
    compute_covariances,
)

DEFAULT_OUTPUT_ROOT = REPO / "chains" / "l1_m9_qfrommap_asz_covariance"
FULLSKY_DATA = (
    REPO / "data_paper" / "binned_bandpowers" / "Dl_yy_L1_m9_fullsky_binned_18.txt"
)
DELTA_A_SZ_TOLERANCE = 1e-4
RELATIVE_COVARIANCE_TOLERANCE = 1e-3

FIXED = {
    "H0": 68.1,
    "omega_cdm": 0.11872788986038219,
    "omega_b": 0.022538784599999993,
    "n_s": 0.965,
    "sigma_8": 0.8025701499024616,
    "tau_reio": 0.0544,
    "alpha_SZ": 1.12,
    "sigma_lnY": 0.173,
    "B": 1.41,
}


def build_asz_info(
    covariance_file: Path,
    chain_root: Path,
    *,
    current_a_sz: float,
) -> dict:
    """Build an amplitude-only full-sky Cobaya configuration."""
    params = {name: {"value": value} for name, value in FIXED.items()}
    params["A_SZ"] = {
        "prior": {"min": -5.5, "max": -3.0},
        "ref": {"dist": "norm", "loc": float(current_a_sz), "scale": 0.02},
        "proposal": 0.005,
        "latex": r"A_\mathrm{SZ}",
    }
    return {
        "output": str(Path(chain_root)),
        "likelihood": {
            "flamingo.inference.masked_ps.MaskedBandPowerLikelihood": {
                "data_file": str(FULLSKY_DATA),
                "covariance_file": str(Path(covariance_file)),
                "data_scale": 1e-12,
            }
        },
        "theory": {
            "flamingo.inference.masked_ps.MaskedTSZTheory": {"q_cat": None}
        },
        "params": params,
        "sampler": {
            "mcmc": {
                "Rminus1_stop": 0.01,
                "drag": False,
                "proposal_scale": 1.5,
                "learn_every": 40,
                "learn_proposal": True,
                "max_tries": 100000,
                "burn_in": 20,
            }
        },
        "timing": True,
        "resume": True,
        "seed": 20260802,
    }


def _run_asz_chain(info: dict) -> dict[str, float]:
    from cobaya.run import run
    from getdist import loadMCSamples

    run(info)
    samples = loadMCSamples(info["output"], settings={"ignore_rows": 0.3})
    names = samples.getParamNames().list()
    a_sz_index = names.index("A_SZ")
    chi2_name = next(name for name in names if name.startswith("chi2__"))
    chi2_index = names.index(chi2_name)
    best_index = int(np.argmin(samples.samples[:, chi2_index]))
    return {
        "A_SZ": float(samples.samples[best_index, a_sz_index]),
        "chi2": float(samples.samples[best_index, chi2_index]),
    }


def _fullsky_covariance(result: dict) -> tuple[Path, np.ndarray]:
    covariance_path = Path(result["cases"]["fullsky"]["paths"]["cov_full"])
    return covariance_path, np.asarray(np.load(covariance_path), dtype=float)


def _write_summary(output_root: Path, summary: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "converged.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def iterate(
    seed_a_sz: float,
    output_root: Path,
    max_iterations: int = 10,
    *,
    covariance_fn: Callable = compute_covariances,
    chain_fn: Callable = _run_asz_chain,
) -> dict:
    """Alternate covariance calculation and amplitude-only MCMC until stable."""
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    current_a_sz = float(seed_a_sz)
    history: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        iteration_dir = output_root / f"iteration_{iteration:02d}"
        input_covariance_result = covariance_fn(
            current_a_sz,
            iteration_dir / "covariance",
            cases=("fullsky",),
        )
        input_covariance_path, input_covariance = _fullsky_covariance(
            input_covariance_result
        )
        info = build_asz_info(
            input_covariance_path,
            iteration_dir / "chain" / "chain",
            current_a_sz=current_a_sz,
        )
        best_fit = chain_fn(info)
        new_a_sz = float(best_fit["A_SZ"])
        bestfit_covariance_result = covariance_fn(
            new_a_sz,
            iteration_dir / "bestfit_covariance",
            cases=("fullsky",),
        )
        bestfit_covariance_path, bestfit_covariance = _fullsky_covariance(
            bestfit_covariance_result
        )
        delta_a_sz = abs(new_a_sz - current_a_sz)
        covariance_norm = float(np.linalg.norm(input_covariance))
        relative_covariance_change = float(
            np.linalg.norm(bestfit_covariance - input_covariance) / covariance_norm
        )
        converged = (
            delta_a_sz < DELTA_A_SZ_TOLERANCE
            and relative_covariance_change < RELATIVE_COVARIANCE_TOLERANCE
        )
        history.append(
            {
                "iteration": iteration,
                "input_A_SZ": current_a_sz,
                "best_fit": {key: float(value) for key, value in best_fit.items()},
                "delta_A_SZ": delta_a_sz,
                "relative_covariance_change": relative_covariance_change,
                "input_covariance_path": str(input_covariance_path),
                "bestfit_covariance_path": str(bestfit_covariance_path),
                "converged": converged,
            }
        )

        if converged:
            final_result = covariance_fn(
                new_a_sz,
                output_root / "final" / "covariance",
                cases=CASES,
            )
            final_covariance_paths = {
                case: result["paths"]["cov_full"]
                for case, result in final_result["cases"].items()
            }
            summary = {
                "converged": True,
                "iterations_completed": iteration,
                "best_fit": {key: float(value) for key, value in best_fit.items()},
                "delta_A_SZ": delta_a_sz,
                "relative_covariance_change": relative_covariance_change,
                "thresholds": {
                    "delta_A_SZ": DELTA_A_SZ_TOLERANCE,
                    "relative_covariance_change": RELATIVE_COVARIANCE_TOLERANCE,
                },
                "fixed_parameters": FIXED,
                "history": history,
                "final_metadata_path": final_result["metadata_path"],
                "final_covariance_paths": final_covariance_paths,
            }
            _write_summary(output_root, summary)
            return summary
        current_a_sz = new_a_sz

    summary = {
        "converged": False,
        "iterations_completed": max_iterations,
        "best_fit": history[-1]["best_fit"],
        "delta_A_SZ": history[-1]["delta_A_SZ"],
        "relative_covariance_change": history[-1]["relative_covariance_change"],
        "thresholds": {
            "delta_A_SZ": DELTA_A_SZ_TOLERANCE,
            "relative_covariance_change": RELATIVE_COVARIANCE_TOLERANCE,
        },
        "fixed_parameters": FIXED,
        "history": history,
    }
    _write_summary(output_root, summary)
    raise RuntimeError(
        f"A_SZ/covariance fixed point did not converge after {max_iterations} iterations"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-a-sz", type=float, default=-4.0953238)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-iterations", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    summary = iterate(args.seed_a_sz, args.output_root, args.max_iterations)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
