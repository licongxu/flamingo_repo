"""Run a fixed-cosmology CNC chain on L1_m9 qfrommap q>5 counts."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

CNC_GPU = Path("/scratch/scratch-lxu/tsz_cnc_inference_gpu")
HMFAST_SRC = Path(
    "/scratch/scratch-lxu/agent_dev/auto_research_agent/hmfast-fix-2h-pk-units/src"
)
COSMOCNC = Path("/scratch/scratch-lxu/agent_dev/auto_research_agent/cosmocnc_jax")
DATA_FILE = REPO / "data_paper" / "cnc" / "N2d_z_q_L1_m9_qfrommap_qgt5.txt"
CHAINS = REPO / "chains" / "l1_m9_qfrommap_cnc_qgt5"

# Same D3A point as the qfrommap A_SZ-only power-spectrum chains.
FIXED = {
    "H0": 68.1,
    "omega_cdm": 0.11872788986038219,
    "omega_b": 0.022538784599999993,
    "n_s": 0.965,
    "sigma_8": 0.8025701499024616,
    "tau_reio": 0.0544,
    "B": 1.41,
}
OMEGA_M = 0.306
M_NU = 0.06

LIKELIHOOD = "modules.likelihood_cnc.CNCBinnedPlanckScatterLikelihood"
THEORY = {
    "survey_sr": str(COSMOCNC / "cosmocnc_jax/surveys/survey_sr_planck_sim.py"),
    "survey_cat": str(COSMOCNC / "cosmocnc_jax/surveys/survey_cat_planck_sim.py"),
    "tszsbi_noise_dir": "/scratch/scratch-lxu/tszsbi/noise_files",
    "tszsbi_filter_name": "immf6",
    "lambda_floor": 1.0e-12,
    "hmfast_path": str(HMFAST_SRC),
    "hmfast_emulator_set": "lcdm:v1",
    "z_min": 0.005,
    "z_max": 1.0,
    "n_z_bins": 10,
    "q_min": 5.0,
    "q_max": 40.0,
    "n_q_bins": 5,
    "n_points": 2048,
    "n_z": 50,
    "M_min": 1.0e14,
    "M_max": 1.0e16,
    "f_sky": 1.0,
    "M_pivot": 2.1e14,
}


def write_counts(
    output: Path = DATA_FILE,
    catalogue: Path | None = None,
) -> np.ndarray:
    """Bin the L1_m9 qfrommap catalogue onto the paper ``(z, q)`` grid."""
    from scripts.plot_l1_m9_binned_cnc import bin_cnc, catalogue_path

    counts = bin_cnc(catalogue_path("L1_m9") if catalogue is None else catalogue)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, counts)
    return counts


def parameters() -> dict:
    """Fix D3A cosmology; sample ``A_SZ``, ``alpha_SZ``, ``sigma_lnY``."""
    params = {name: {"value": value} for name, value in FIXED.items()}
    params["Omega_m"] = {"value": OMEGA_M}
    params["m_nu"] = {"value": M_NU}
    params["one_minus_b"] = {"value": 1.0 / FIXED["B"], "latex": r"1-b"}
    params["A_SZ"] = {
        "prior": {"min": -4.41, "max": -4.00},
        "ref": {"dist": "norm", "loc": -4.205, "scale": 0.02},
        "proposal": 0.02,
        "latex": r"A_\mathrm{SZ}",
    }
    params["alpha_SZ"] = {
        "prior": {"min": 0.8, "max": 1.2},
        "ref": {"dist": "norm", "loc": 1.0, "scale": 0.02},
        "proposal": 0.02,
        "latex": r"\alpha_\mathrm{SZ}",
    }
    params["sigma_lnY"] = {
        "prior": {"dist": "norm", "loc": 0.173, "scale": 0.05},
        "ref": {"dist": "norm", "loc": 0.173, "scale": 0.05},
        "proposal": 0.01,
        "latex": r"\sigma_{\ln Y}",
    }
    return params


def build_info(*, rminus1: float = 0.01, data_file: Path = DATA_FILE) -> dict:
    """Cobaya info matching case03 CNC theory, with D3A cosmology held fixed."""
    params = parameters()
    return {
        "output": str(CHAINS / "chain"),
        "likelihood": {
            LIKELIHOOD: {
                **THEORY,
                "data_file": str(data_file),
                "input_params": list(params),
            }
        },
        "params": params,
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
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(HMFAST_SRC))
    sys.path.insert(0, str(CNC_GPU))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Rminus1-stop", type=float, default=0.01)
    parser.add_argument("--write-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.write_data or not DATA_FILE.is_file():
        counts = write_counts()
        print(f"wrote {DATA_FILE} shape={counts.shape} N={int(counts.sum())}", flush=True)
        if args.write_data:
            return
    info = build_info(rminus1=args.Rminus1_stop)
    if args.dry_run:
        import json

        print(json.dumps(info, indent=2, sort_keys=True))
        return
    _prepare_runtime()
    CHAINS.mkdir(parents=True, exist_ok=True)
    if args.eval:
        from cobaya.model import get_model

        model_info = {k: v for k, v in info.items() if k not in ("output", "sampler")}
        model = get_model(model_info)
        point = {
            "A_SZ": -4.205,
            "alpha_SZ": 1.0,
            "sigma_lnY": 0.173,
        }
        print("eval point:", point, flush=True)
        print("loglikes:", model.loglikes(point, as_dict=True)[0], flush=True)
        return
    from cobaya.run import run

    run(info)


if __name__ == "__main__":
    main()
