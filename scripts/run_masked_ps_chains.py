"""Run five GPU Cobaya chains on the L1_m9 18-bin qfrommap spectra."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BANDPOWERS = REPO / "data_paper" / "binned_bandpowers"
ASZ_COVARIANCE_ROOT = REPO / "chains" / "l1_m9_qfrommap_asz_covariance"
CONVERGED_FILE = ASZ_COVARIANCE_ROOT / "converged.json"
CHAINS = REPO / "chains" / "l1_m9_qfrommap_signal_only"
PREFLIGHT_FILE = CHAINS / "preflight.json"

CASES: dict[str, float | None] = {
    "fullsky": None,
    "qgt50": 50.0,
    "qgt20": 20.0,
    "qgt10": 10.0,
    "qgt5": 5.0,
}


def omega_cdm_from_omegam(Omega_m: float, H0: float, omega_b: float) -> float:
    """Convert total matter density to CDM density with one 0.06-eV neutrino."""
    return Omega_m * (H0 / 100.0) ** 2 - omega_b - 0.06 / 93.14


def _default_covariance_path(case: str) -> Path:
    stem = f"L1_m9_customgnfw_qfrommap_{case}_Dl_yy_binned_18"
    return ASZ_COVARIANCE_ROOT / "final" / "covariance" / f"cov_full_{stem}.npy"


def data_files(
    case: str,
    covariance_paths: dict[str, Path] | None = None,
) -> tuple[Path, Path]:
    """Return the exact observed spectrum and converged covariance for a case."""
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}; expected one of {tuple(CASES)}")
    if case == "fullsky":
        data = BANDPOWERS / "Dl_yy_L1_m9_fullsky_binned_18.txt"
    else:
        data = BANDPOWERS / f"Dl_yy_L1_m9_masked_{case}_qfrommap_binned_18.txt"
    covariance = (
        _default_covariance_path(case)
        if covariance_paths is None
        else Path(covariance_paths[case])
    )
    return data, covariance


def load_converged_artifacts() -> dict:
    """Load and cross-check the fixed-point summary and final covariance metadata."""
    if not CONVERGED_FILE.is_file():
        raise FileNotFoundError(CONVERGED_FILE)
    summary = json.loads(CONVERGED_FILE.read_text())
    if summary.get("converged") is not True:
        raise RuntimeError(f"A_SZ/covariance iteration is not converged: {CONVERGED_FILE}")
    metadata_path = Path(summary["final_metadata_path"])
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    best_fit = float(summary["best_fit"]["A_SZ"])
    covariance_a_sz = float(metadata["A_SZ"])
    if covariance_a_sz != best_fit:
        raise ValueError(
            f"final covariance A_SZ={covariance_a_sz} does not match best-fit A_SZ={best_fit}"
        )

    covariance_paths: dict[str, Path] = {}
    for case in CASES:
        metadata_path_for_case = Path(metadata["cases"][case]["paths"]["cov_full"])
        summary_path_for_case = Path(summary["final_covariance_paths"][case])
        if metadata_path_for_case.resolve() != summary_path_for_case.resolve():
            raise ValueError(f"conflicting final covariance paths for {case}")
        if not metadata_path_for_case.is_file():
            raise FileNotFoundError(metadata_path_for_case)
        covariance = np.asarray(np.load(metadata_path_for_case), dtype=float)
        if covariance.shape != (18, 18):
            raise ValueError(f"{case} covariance has shape {covariance.shape}")
        np.linalg.cholesky(covariance)
        covariance_paths[case] = metadata_path_for_case
    return {
        "A_SZ": best_fit,
        "covariance_paths": covariance_paths,
        "summary": summary,
        "metadata": metadata,
    }


def parameters(a_sz_best_fit: float) -> dict:
    """Approved science parameters and D3A-centred priors."""
    return {
        "H0": {
            "prior": {"dist": "norm", "loc": 68.1, "scale": 1.0},
            "ref": {"dist": "norm", "loc": 68.1, "scale": 1.0},
            "proposal": 0.6,
            "latex": r"H_0",
        },
        "sigma_8": {
            "prior": {"min": 0.6, "max": 1.0},
            "ref": {"dist": "norm", "loc": 0.8025701499024616, "scale": 0.02},
            "proposal": 0.02,
            "latex": r"\sigma_8",
        },
        "n_s": {
            "prior": {"dist": "norm", "loc": 0.965, "scale": 0.014},
            "ref": {"dist": "norm", "loc": 0.965, "scale": 0.014},
            "proposal": 0.014,
            "latex": r"n_\mathrm{s}",
        },
        "omega_b": {
            "prior": {
                "dist": "norm",
                "loc": 0.022538784599999993,
                "scale": 0.002,
            },
            "ref": {
                "dist": "norm",
                "loc": 0.022538784599999993,
                "scale": 0.002,
            },
            "proposal": 0.002,
            "latex": r"\Omega_\mathrm{b} h^2",
        },
        "Omega_m": {
            "prior": {"min": 0.2, "max": 0.5},
            "ref": {"dist": "norm", "loc": 0.306, "scale": 0.02},
            "proposal": 0.02,
            "latex": r"\Omega_m",
        },
        "omega_cdm": {
            "value": (
                "lambda Omega_m, H0, omega_b: "
                "Omega_m * (H0 / 100.0)**2 - omega_b - 0.06 / 93.14"
            ),
            "latex": r"\Omega_\mathrm{c} h^2",
        },
        "tau_reio": {"value": 0.0544, "latex": r"\tau_\mathrm{reio}"},
        "A_SZ": {
            "prior": {"dist": "norm", "loc": float(a_sz_best_fit), "scale": 0.03},
            "ref": {"dist": "norm", "loc": float(a_sz_best_fit), "scale": 0.01},
            "proposal": 0.01,
            "latex": r"A_\mathrm{SZ}",
        },
        "alpha_SZ": {
            "prior": {"dist": "norm", "loc": 1.12, "scale": 0.03},
            "ref": {"dist": "norm", "loc": 1.12, "scale": 0.01},
            "proposal": 0.01,
            "latex": r"\alpha_\mathrm{SZ}",
        },
        "sigma_lnY": {
            "prior": {"dist": "norm", "loc": 0.173, "scale": 0.023},
            "ref": {"dist": "norm", "loc": 0.173, "scale": 0.023},
            "proposal": 0.01,
            "latex": r"\sigma_{\ln Y}",
        },
        "B": {"value": 1.41, "latex": r"B"},
        "S8": {
            "derived": "lambda sigma_8, Omega_m: sigma_8 * (Omega_m / 0.3)**0.5",
            "latex": r"S_8",
        },
    }


def build_info(
    case: str,
    *,
    rminus1: float = 0.01,
    artifacts: dict | None = None,
) -> dict:
    """Assemble one resolved Cobaya configuration from converged artifacts."""
    artifacts = load_converged_artifacts() if artifacts is None else artifacts
    data, covariance = data_files(case, artifacts["covariance_paths"])
    for path in (data, covariance):
        if not path.is_file():
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
        "params": parameters(artifacts["A_SZ"]),
        "sampler": {
            "mcmc": {
                "Rminus1_stop": rminus1,
                "drag": False,
                "proposal_scale": 1.2,
                "learn_every": 40,
                "learn_proposal": True,
                "max_tries": 100000,
                "burn_in": 50,
            }
        },
        "timing": True,
        "resume": True,
    }


def gpu_devices() -> list[str]:
    import jax

    devices = jax.devices()
    if not devices or devices[0].platform != "gpu":
        raise RuntimeError(f"CUDA device required, got {devices}")
    return [str(device) for device in devices]


def write_preflight(cases: tuple[str, ...], artifacts: dict) -> dict:
    """Resolve and record every production input before sampling."""
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
        "best_fit_A_SZ": artifacts["A_SZ"],
        "covariance_metadata_A_SZ": artifacts["metadata"]["A_SZ"],
        "covariance_metadata_path": artifacts["summary"]["final_metadata_path"],
        "cases": resolved,
        "params": parameters(artifacts["A_SZ"]),
    }
    CHAINS.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_FILE.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    return preflight


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=tuple(CASES))
    parser.add_argument("--Rminus1-stop", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    selected_cases = tuple(args.case or CASES)
    artifacts = load_converged_artifacts()
    preflight = write_preflight(selected_cases, artifacts)
    if args.dry_run:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    os.environ.setdefault("FLAMINGO_ROOT", "/scratch/scratch-lxu/flamingo_repo")
    from cobaya.run import run

    for case in selected_cases:
        print(f"\n=== {case} (q_cat={CASES[case]}) ===", flush=True)
        (CHAINS / case).mkdir(parents=True, exist_ok=True)
        run(build_info(case, rminus1=args.Rminus1_stop, artifacts=artifacts))


if __name__ == "__main__":
    main()
