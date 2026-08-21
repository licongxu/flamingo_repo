"""Run a fixed-cosmology CNC chain on L1_m9 qfrommap q>5 counts."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# CUDA snapshots the device list at process start; an empty
# CUDA_VISIBLE_DEVICES hides every GPU. Pin GPU 1 before numpy/JAX/TF.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.pop("JAX_PLATFORMS", None)

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

CNC_GPU = Path("/scratch/scratch-lxu/tsz_cnc_inference_gpu")
HMFAST_SRC = Path(
    "/scratch/scratch-lxu/agent_dev/auto_research_agent/hmfast-fix-2h-pk-units/src"
)
COSMOCNC = Path("/scratch/scratch-lxu/agent_dev/auto_research_agent/cosmocnc_jax")
YMAP = Path("/rds/rds-lxu/flamingo/L1_m9/maps/y_unlensed_L1_m9_lc0_nside4096.fits")
DATA_DIR = REPO / "data_paper" / "cnc"
DATA_FILE = DATA_DIR / "N2d_z_q_L1_m9_qfrommap_qgt5.txt"
# Previous run at chains/l1_m9_qfrommap_cnc_qgt5 used alpha_SZ in [0.8, 1.2]
# and piled on the floor. This directory is a fresh chain with an open prior.
CHAINS = REPO / "chains" / "l1_m9_qfrommap_cnc_qgt5_openprior"
ARCMIN2_PER_SR = (180.0 * 60.0 / np.pi) ** 2

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
Y500_NOISE = REPO / "data" / "noise" / "sigma_Y500_dict_szifi.npy"
Y_MONOPOLE_L1_M9 = 1.502772e-6


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


def q_aperture_submono(
    y_cyl: np.ndarray,
    npix: np.ndarray,
    sigma: np.ndarray,
    ybar: float,
    pixarea_sr: float,
) -> np.ndarray:
    """Aperture SNR after subtracting the map monopole from each disc."""
    area = np.asarray(npix, dtype=np.float64) * float(pixarea_sr) * ARCMIN2_PER_SR
    return (
        np.asarray(y_cyl, dtype=np.float64) - float(ybar) * area
    ) / np.asarray(sigma, dtype=np.float64)


def write_counts_submono(
    output: Path,
    catalogue: Path,
    ybar: float,
    pixarea_sr: float,
    chunksize: int = 1_000_000,
) -> np.ndarray:
    """Bin monopole-subtracted map-aperture ``q`` onto the paper ``(z, q)`` grid."""
    import pandas as pd

    from scripts.plot_l1_m9_binned_cnc import Q_EDGES, Z_EDGES

    counts = np.zeros((len(Z_EDGES) - 1, len(Q_EDGES) - 1), dtype=np.int64)
    for chunk in pd.read_csv(
        catalogue,
        comment="#",
        usecols=["z", "Y_500cyl_arcmin2", "sigma_Y500_arcmin2", "npix_in_aperture"],
        chunksize=chunksize,
    ):
        q = q_aperture_submono(
            chunk["Y_500cyl_arcmin2"].to_numpy(np.float64),
            chunk["npix_in_aperture"].to_numpy(np.float64),
            chunk["sigma_Y500_arcmin2"].to_numpy(np.float64),
            ybar,
            pixarea_sr,
        )
        chunk_counts, _, _ = np.histogram2d(
            chunk["z"].to_numpy(np.float64),
            q,
            bins=[Z_EDGES, Q_EDGES],
        )
        counts += chunk_counts.astype(np.int64)
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
        "prior": {"min": 0.5, "max": 1.3},
        "ref": {"dist": "norm", "loc": 0.94, "scale": 0.05},
        "proposal": 0.05,
        "latex": r"\alpha_\mathrm{SZ}",
    }
    params["sigma_lnY"] = {
        "prior": {"dist": "norm", "loc": 0.173, "scale": 0.05},
        "ref": {"dist": "norm", "loc": 0.173, "scale": 0.05},
        "proposal": 0.01,
        "latex": r"\sigma_{\ln Y}",
    }
    return params


def build_info(
    *,
    rminus1: float = 0.01,
    data_file: Path = DATA_FILE,
    output: Path | None = None,
    add_monopole: bool = False,
    y_monopole: float = Y_MONOPOLE_L1_M9,
) -> dict:
    """Cobaya info matching case03 CNC theory, with D3A cosmology held fixed."""
    params = parameters()
    likelihood = {
        **THEORY,
        "data_file": str(data_file),
        "input_params": list(params),
    }
    if add_monopole:
        likelihood["add_monopole"] = True
        likelihood["y_monopole"] = float(y_monopole)
        likelihood["tszsbi_sigma_Y500_file"] = str(Y500_NOISE)
    return {
        "output": str(CHAINS / "chain" if output is None else output),
        "likelihood": {LIKELIHOOD: likelihood},
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
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if gpu in ("", "none", "None"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ.pop("JAX_PLATFORMS", None)
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
    parser.add_argument(
        "--submono",
        action="store_true",
        help="Bin monopole-subtracted map-aperture q and run that chain.",
    )
    parser.add_argument(
        "--add-monopole",
        action="store_true",
        help="Keep raw aperture q; add map mean to CNC theory q.",
    )
    parser.add_argument(
        "--y-monopole",
        type=float,
        default=Y_MONOPOLE_L1_M9,
        help="Compton-y map mean used when --add-monopole is set.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.submono and args.add_monopole:
        raise ValueError("use either --submono (data) or --add-monopole (theory), not both")
    data_file = DATA_FILE
    chains = CHAINS
    if args.submono:
        data_file = DATA_DIR / "N2d_z_q_L1_m9_qfrommap_submono_qgt5.txt"
        chains = REPO / "chains" / "l1_m9_qfrommap_submono_cnc_qgt5"
    if args.add_monopole:
        chains = REPO / "chains" / "l1_m9_qfrommap_cnc_qgt5_addmono"
    if args.write_data or not data_file.is_file():
        if args.submono:
            import healpy as hp

            from scripts.plot_l1_m9_binned_cnc import catalogue_path

            print("reading y map for monopole", YMAP, flush=True)
            ymap = np.asarray(hp.read_map(YMAP, dtype=np.float32), dtype=np.float64)
            ybar = float(ymap.mean())
            pixarea = float(hp.nside2pixarea(hp.npix2nside(ymap.size)))
            print(f"monopole <y>={ybar:.6e}  pixarea={pixarea:.6e} sr", flush=True)
            counts = write_counts_submono(
                data_file, catalogue_path("L1_m9"), ybar, pixarea
            )
        else:
            counts = write_counts(data_file)
        print(f"wrote {data_file} shape={counts.shape} N={int(counts.sum())}", flush=True)
        if args.write_data:
            return
    info = build_info(
        rminus1=args.Rminus1_stop,
        data_file=data_file,
        output=chains / "chain",
        add_monopole=args.add_monopole,
        y_monopole=args.y_monopole,
    )
    if args.submono or args.add_monopole:
        info["resume"] = False
    if args.dry_run:
        import json

        print(json.dumps(info, indent=2, sort_keys=True))
        return
    _prepare_runtime()
    chains.mkdir(parents=True, exist_ok=True)
    if args.eval:
        from cobaya.model import get_model

        model_info = {k: v for k, v in info.items() if k not in ("output", "sampler")}
        model = get_model(model_info)
        point = {
            "A_SZ": -4.205,
            "alpha_SZ": 0.94,
            "sigma_lnY": 0.173,
        }
        print("eval point:", point, flush=True)
        print("loglikes:", model.loglikes(point, as_dict=True)[0], flush=True)
        return
    from cobaya.run import run

    run(info)


if __name__ == "__main__":
    main()
