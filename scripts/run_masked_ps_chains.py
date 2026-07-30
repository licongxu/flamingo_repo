"""Signal-only Cobaya chains on the L1_m9 masked tSZ bandpowers.

One chain per masking threshold -- total (full-sky) tSZ and ``q > 50, 20, 10,
5`` -- on the fiducial 18-bin bandpowers, sampling cosmology with the priors of
the reference signal-only runs in
``/home/lxu/scratch/tsz_cnc_paper_plots/chains/case0{1,2}_tsz_*_signal_only_real1``.

Two deliberate departures from that reference:

* the theory includes **both** halo terms. The reference is 1-halo only because
  its synthetic maps were painted 1-halo only; the FLAMINGO maps contain both.
  The two terms carry *different* mask weights (see
  :mod:`flamingo.inference.masked_ps`).
* ``A_SZ`` is centred on the L1_m9 full-sky best fit rather than the reference's
  synthetic truth, and it actually enters the pressure profile (in the reference
  it was an inert nuisance for the power-spectrum-only cases).

Run::

    python scripts/run_masked_ps_chains.py --case fullsky
    python scripts/run_masked_ps_chains.py            # all five, in sequence
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DATA = Path("/scratch/scratch-lxu/flamingo_repo/data_paper")
BANDPOWERS = DATA / "binned_bandpowers"
COVARIANCE = DATA / "covariance"
CHAINS = REPO / "chains" / "masked_ps_signal_only"

#: Amplitude of the custom-GNFW best fit to the full-sky spectrum, which also
#: defines the selection that built the masks.
A_SZ_BEST_FIT = -4.0953238

#: ``None`` marks the unmasked total-tSZ case.
CASES: dict[str, float | None] = {
    "fullsky": None,
    "qgt50": 50.0,
    "qgt20": 20.0,
    "qgt10": 10.0,
    "qgt5": 5.0,
}


def data_files(case: str) -> tuple[Path, Path]:
    """Bandpower and covariance paths for one case."""
    if case == "fullsky":
        return (
            BANDPOWERS / "Dl_yy_L1_m9_fullsky_binned_18.txt",
            COVARIANCE / "cov_full_L1_m9_customgnfw_bestfit_fullsky_Dl_yy_binned_18.npy",
        )
    return (
        BANDPOWERS / f"Dl_yy_L1_m9_masked_{case}_qfrommz_alpha_fixed_1p12_binned_18.txt",
        COVARIANCE
        / f"cov_full_L1_m9_customgnfw_bestfit_masked_{case}_Dl_yy_binned_18.npy",
    )


def parameters() -> dict:
    """Sampled parameters and priors, matching the reference signal-only runs."""
    return {
        "H0": {
            "prior": {"dist": "norm", "loc": 67.66, "scale": 1.0},
            "ref": {"dist": "norm", "loc": 67.66, "scale": 1.0},
            "proposal": 0.6,
            "latex": r"H_0",
        },
        "sigma_8": {
            "prior": {"min": 0.6, "max": 1.0},
            "ref": {"dist": "norm", "loc": 0.80, "scale": 0.02},
            "proposal": 0.02,
            "latex": r"\sigma_8",
        },
        "n_s": {
            "prior": {"dist": "norm", "loc": 0.9665, "scale": 0.014},
            "ref": {"dist": "norm", "loc": 0.9665, "scale": 0.014},
            "proposal": 0.014,
            "latex": r"n_\mathrm{s}",
        },
        "omega_b": {
            "prior": {"dist": "norm", "loc": 0.02242, "scale": 0.002},
            "ref": {"dist": "norm", "loc": 0.02242, "scale": 0.002},
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
            "value": "lambda Omega_m, H0, omega_b: Omega_m * (H0 / 100.0)**2 - omega_b",
            "latex": r"\Omega_\mathrm{c} h^2",
        },
        "tau_reio": {"value": 0.0544, "latex": r"\tau_{reio}"},
        "A_SZ": {
            "prior": {"dist": "norm", "loc": A_SZ_BEST_FIT, "scale": 0.03},
            "ref": {"dist": "norm", "loc": A_SZ_BEST_FIT, "scale": 0.01},
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
        "S8": {
            "derived": "lambda sigma_8, Omega_m: sigma_8 * (Omega_m / 0.3)**0.5",
            "latex": r"S_8",
        },
    }


def build_info(case: str, *, rminus1: float = 0.02) -> dict:
    """Assemble the Cobaya input dictionary for one case."""
    data_file, covariance_file = data_files(case)
    for path in (data_file, covariance_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "output": str(CHAINS / case / "chain"),
        "likelihood": {
            "flamingo.inference.masked_ps.MaskedBandPowerLikelihood": {
                "data_file": str(data_file),
                "covariance_file": str(covariance_file),
                "data_scale": 1.0e-12,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", action="append", choices=sorted(CASES), help="case(s) to run")
    parser.add_argument("--Rminus1-stop", type=float, default=0.02)
    args = parser.parse_args()

    os.environ.setdefault("FLAMINGO_ROOT", "/scratch/scratch-lxu/flamingo_repo")
    from cobaya.run import run

    for case in args.case or list(CASES):
        print(f"\n=== {case} (q_cat={CASES[case]}) ===", flush=True)
        (CHAINS / case).mkdir(parents=True, exist_ok=True)
        run(build_info(case, rminus1=args.Rminus1_stop))


if __name__ == "__main__":
    main()
