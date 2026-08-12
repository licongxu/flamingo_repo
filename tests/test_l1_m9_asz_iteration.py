import json
from pathlib import Path

import numpy as np

from scripts.inference.iterate_l1_m9_asz_covariance import build_asz_info, iterate


def test_asz_fit_varies_only_asz(tmp_path):
    info = build_asz_info(
        tmp_path / "cov.npy",
        tmp_path / "chain",
        current_a_sz=-4.1,
    )
    sampled = [
        name
        for name, config in info["params"].items()
        if isinstance(config, dict) and "prior" in config
    ]

    assert sampled == ["A_SZ"]
    assert info["params"]["B"]["value"] == 1.41
    assert info["params"]["sigma_lnY"]["value"] == 0.173
    assert info["params"]["alpha_SZ"]["value"] == 1.12
    assert info["params"]["H0"]["value"] == 68.1
    assert info["params"]["A_SZ"]["prior"] == {"min": -5.5, "max": -3.0}
    assert "Rminus1_cl_stop" not in info["sampler"]["mcmc"]


def test_iteration_stops_only_when_asz_and_covariance_are_stable(tmp_path):
    fitted_values = iter([-4.1, -4.10005])

    def fake_covariance(a_sz: float, output_dir: Path, cases: tuple[str, ...]):
        output_dir.mkdir(parents=True, exist_ok=True)
        case_results = {}
        for case in cases:
            covariance = np.eye(18) * np.exp(a_sz)
            covariance_path = output_dir / f"cov_full_{case}.npy"
            np.save(covariance_path, covariance)
            case_results[case] = {
                "paths": {"cov_full": str(covariance_path)},
                "diagnostics": {"min_eigenvalue": float(np.exp(a_sz))},
            }
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps({"A_SZ": a_sz}) + "\n")
        return {
            "A_SZ": a_sz,
            "metadata_path": str(metadata_path),
            "cases": case_results,
        }

    def fake_chain(info: dict):
        return {"A_SZ": next(fitted_values), "chi2": 4.0}

    summary = iterate(
        -4.0,
        tmp_path,
        covariance_fn=fake_covariance,
        chain_fn=fake_chain,
        max_iterations=10,
    )

    assert summary["converged"] is True
    assert summary["iterations_completed"] == 2
    assert abs(summary["delta_A_SZ"]) < 1e-4
    assert summary["relative_covariance_change"] < 1e-3
    assert summary["best_fit"]["A_SZ"] == -4.10005
    final_metadata = json.loads(Path(summary["final_metadata_path"]).read_text())
    assert final_metadata["A_SZ"] == summary["best_fit"]["A_SZ"]
    assert set(summary["final_covariance_paths"]) == {
        "fullsky",
        "qgt50",
        "qgt20",
        "qgt10",
        "qgt5",
    }
