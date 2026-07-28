# L1_m9 Simple-GNFW Theory Covariance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute and validate the 18-bin full-sky Gaussian plus connected one-halo trispectrum covariance for the fiducial L1_m9 bandpowers.

**Architecture:** A focused script owns the established Planck-like bin definitions, pure NumPy binning/covariance helpers, hmfast GPU theory evaluation, validation, and artifact writing. Unit tests exercise the pure numerical helpers before the GPU calculation is implemented; the final runtime audit reloads every saved artifact independently.

**Tech Stack:** Python 3.12, NumPy, JAX CUDA float64, hmfast, pytest.

## Global Constraints

- Activate `/scratch/scratch-lxu/venv/cmbagent_env/bin/activate` before every command.
- Use the fixed FLAMINGO D3A cosmology from `flamingo.catalogue.frame.D3A_COSMOLOGY`.
- Use hmfast `GNFWPressureProfile(P0=8.13, c500=1.156, alpha=1.062, beta=5.4807, gamma=0.3292, B=1.0)`.
- Apply no lognormal pressure scatter or scatter boost: `sigma_lnY = 0`.
- Include both one-halo and two-halo power in the Gaussian term.
- Use the connected one-halo trispectrum for the non-Gaussian term.
- Use physical masses in `M_sun`, never `M_sun/h`.
- Use all 18 bins with `10 <= ell_eff <= 959.5` and `f_sky = 1`.
- Write covariance artifacts under `data_paper/covariance/`.
- Execute the production calculation on CUDA.

---

### Task 1: Test and implement the covariance numerics

**Files:**
- Create: `scripts/compute_l1_m9_simplegnfw_covariance.py`
- Create: `tests/test_l1_m9_covariance.py`

**Interfaces:**
- Produces: `bin_dl(ell, dl) -> np.ndarray` of shape `(18,)`.
- Produces: `bin_trispectrum(ell, trispectrum_cl) -> np.ndarray` of shape `(18, 18)`.
- Produces: `gaussian_covariance(dl_binned, fsky=1.0) -> np.ndarray`.
- Produces: `assemble_covariance(cov_gaussian, trispectrum_binned, fsky=1.0) -> np.ndarray`.

- [ ] **Step 1: Write the failing pure-numerics tests**

Create `tests/test_l1_m9_covariance.py`:

```python
import numpy as np

from scripts.compute_l1_m9_simplegnfw_covariance import (
    ELL_EFF,
    ELL_MAX,
    ELL_MIN,
    assemble_covariance,
    bin_dl,
    bin_trispectrum,
    gaussian_covariance,
)


def test_bin_dl_preserves_a_constant():
    ell = np.geomspace(9.0, 1085.0, 80)
    got = bin_dl(ell, np.full(ell.shape, 3.25))
    np.testing.assert_allclose(got, 3.25, rtol=0.0, atol=1e-14)


def test_bin_trispectrum_is_symmetric_and_has_18_bins():
    ell = np.geomspace(9.0, 1085.0, 40)
    trispectrum_cl = np.outer(ell**-1.2, ell**-1.2)
    got = bin_trispectrum(ell, trispectrum_cl)
    assert got.shape == (18, 18)
    np.testing.assert_allclose(got, got.T, rtol=1e-12, atol=0.0)
    assert np.all(got > 0.0)


def test_gaussian_covariance_matches_fullsky_knox_formula():
    dl = np.linspace(0.2, 1.9, 18) * 1e-12
    got = gaussian_covariance(dl)
    expected_diag = 2.0 * dl**2 / (
        (2.0 * ELL_EFF + 1.0) * (ELL_MAX - ELL_MIN)
    )
    np.testing.assert_allclose(np.diag(got), expected_diag)
    np.testing.assert_array_equal(got - np.diag(np.diag(got)), 0.0)


def test_assemble_covariance_adds_trispectrum_over_4pi():
    gaussian = np.eye(18) * 2.0
    trispectrum = np.full((18, 18), 3.0)
    got = assemble_covariance(gaussian, trispectrum)
    np.testing.assert_allclose(got, gaussian + trispectrum / (4.0 * np.pi))
```

- [ ] **Step 2: Run the tests and verify the expected import failure**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest -q tests/test_l1_m9_covariance.py
```

Expected: collection fails because
`scripts.compute_l1_m9_simplegnfw_covariance` does not exist.

- [ ] **Step 3: Implement the bin definitions and pure numerical helpers**

Create `scripts/compute_l1_m9_simplegnfw_covariance.py` with the environment
configuration before JAX imports and these definitions:

```python
"""Full-sky L1_m9 theory covariance from simple GNFW hmfast theory."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from hmfast.halos import HaloModel
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.profiles import GNFWPressureProfile
from hmfast.tracers import tSZTracer

from flamingo.catalogue.frame import D3A_COSMOLOGY


REPO = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO / "data_paper" / "covariance"

ELL_MIN = np.array(
    [9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380,
     494, 642, 835],
    dtype=int,
)
ELL_MAX = np.array(
    [12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494,
     642, 835, 1085],
    dtype=int,
)
ELL_EFF = np.array(
    [10.0, 13.5, 18.0, 23.5, 30.5, 40.0, 52.5, 68.5, 89.5, 117.0,
     152.5, 198.0, 257.5, 335.5, 436.5, 567.5, 738.0, 959.5],
)
_BIN_WIDTH_MAX = int(np.max(ELL_MAX - ELL_MIN))
_ELL_INTEGER = ELL_MIN[:, None] + np.arange(_BIN_WIDTH_MAX)[None, :]
_ELL_MASK = _ELL_INTEGER < ELL_MAX[:, None]


def bin_dl(ell: np.ndarray, dl: np.ndarray) -> np.ndarray:
    """Uniformly average a smooth D_ell curve over the 18 integer-ell bins."""
    ell = np.asarray(ell, dtype=float)
    dl = np.asarray(dl, dtype=float)
    sampled = np.empty(_ELL_INTEGER.shape, dtype=float)
    for index in range(18):
        sampled[index] = np.interp(
            np.log(_ELL_INTEGER[index]),
            np.log(ell),
            dl,
        )
    return np.sum(sampled * _ELL_MASK, axis=1) / np.sum(_ELL_MASK, axis=1)


def bin_trispectrum(
    ell: np.ndarray,
    trispectrum_cl: np.ndarray,
) -> np.ndarray:
    """Convert T_ell,ell' from C units and average over both bin axes."""
    ell = np.asarray(ell, dtype=float)
    trispectrum_cl = np.asarray(trispectrum_cl, dtype=float)
    log_ell = np.log(ell)
    log_integer = np.log(_ELL_INTEGER.astype(float))

    first_axis = np.empty((*_ELL_INTEGER.shape, ell.size), dtype=float)
    for column in range(ell.size):
        first_axis[..., column] = np.interp(
            log_integer, log_ell, trispectrum_cl[:, column]
        )

    interpolated = np.empty((*_ELL_INTEGER.shape, *_ELL_INTEGER.shape))
    for first_bin in range(18):
        for first_ell in range(_BIN_WIDTH_MAX):
            interpolated[first_bin, first_ell] = np.interp(
                log_integer,
                log_ell,
                first_axis[first_bin, first_ell],
            )

    dl_factor = _ELL_INTEGER * (_ELL_INTEGER + 1.0) / (2.0 * np.pi)
    trispectrum_dl = (
        interpolated
        * dl_factor[:, :, None, None]
        * dl_factor[None, None, :, :]
    )
    pair_mask = _ELL_MASK[:, :, None, None] * _ELL_MASK[None, None, :, :]
    return np.sum(trispectrum_dl * pair_mask, axis=(1, 3)) / np.sum(
        pair_mask, axis=(1, 3)
    )


def gaussian_covariance(
    dl_binned: np.ndarray,
    fsky: float = 1.0,
) -> np.ndarray:
    """Return the diagonal binned Knox covariance."""
    dl_binned = np.asarray(dl_binned, dtype=float)
    diagonal = 2.0 * dl_binned**2 / (
        (2.0 * ELL_EFF + 1.0) * (ELL_MAX - ELL_MIN) * fsky
    )
    return np.diag(diagonal)


def assemble_covariance(
    cov_gaussian: np.ndarray,
    trispectrum_binned: np.ndarray,
    fsky: float = 1.0,
) -> np.ndarray:
    """Combine Gaussian and connected one-halo terms."""
    return (
        np.asarray(cov_gaussian)
        + np.asarray(trispectrum_binned) / (4.0 * np.pi * fsky)
    )
```

- [ ] **Step 4: Run the focused tests and verify they pass**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
CUDA_VISIBLE_DEVICES=1 pytest -q tests/test_l1_m9_covariance.py
```

Expected: `4 passed`.

- [ ] **Step 5: Commit the numerical helpers and tests**

```bash
git add scripts/compute_l1_m9_simplegnfw_covariance.py \
  tests/test_l1_m9_covariance.py
git commit -m "test: define L1_m9 covariance numerics"
```

---

### Task 2: Implement and test artifact validation

**Files:**
- Modify: `scripts/compute_l1_m9_simplegnfw_covariance.py`
- Modify: `tests/test_l1_m9_covariance.py`

**Interfaces:**
- Consumes: the four numerical helpers from Task 1.
- Produces: `validate_covariance(cov_gaussian, trispectrum_binned, cov_full) -> dict[str, float]`.
- Produces: diagnostics containing `max_component_residual`, `max_asymmetry`, and `min_eigenvalue`.

- [ ] **Step 1: Add a failing validation test**

Append:

```python
from scripts.compute_l1_m9_simplegnfw_covariance import validate_covariance


def test_validate_covariance_reports_a_positive_definite_component_sum():
    gaussian = np.eye(18) * 2.0
    trispectrum = np.ones((18, 18))
    full = assemble_covariance(gaussian, trispectrum)
    diagnostics = validate_covariance(gaussian, trispectrum, full)
    assert diagnostics["max_component_residual"] == 0.0
    assert diagnostics["max_asymmetry"] == 0.0
    assert diagnostics["min_eigenvalue"] > 0.0
```

- [ ] **Step 2: Run the focused test and verify the expected import failure**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
CUDA_VISIBLE_DEVICES=1 pytest -q \
  tests/test_l1_m9_covariance.py::test_validate_covariance_reports_a_positive_definite_component_sum
```

Expected: collection fails because `validate_covariance` is not defined.

- [ ] **Step 3: Add strict validation**

Add:

```python
def validate_covariance(
    cov_gaussian: np.ndarray,
    trispectrum_binned: np.ndarray,
    cov_full: np.ndarray,
) -> dict[str, float]:
    """Validate shapes, finiteness, assembly, symmetry, and definiteness."""
    arrays = {
        "cov_gaussian": np.asarray(cov_gaussian, dtype=float),
        "trispectrum_binned": np.asarray(trispectrum_binned, dtype=float),
        "cov_full": np.asarray(cov_full, dtype=float),
    }
    for name, array in arrays.items():
        if array.shape != (18, 18):
            raise ValueError(f"{name} has shape {array.shape}, expected (18, 18)")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite entries")

    expected = assemble_covariance(
        arrays["cov_gaussian"], arrays["trispectrum_binned"]
    )
    component_residual = float(np.max(np.abs(arrays["cov_full"] - expected)))
    scale = float(np.max(np.abs(expected)))
    if component_residual > 1e-12 * max(scale, np.finfo(float).tiny):
        raise ValueError("full covariance does not equal Gaussian + T/(4 pi)")

    asymmetry = float(
        np.max(np.abs(arrays["cov_full"] - arrays["cov_full"].T))
    )
    if asymmetry > 1e-12 * max(scale, np.finfo(float).tiny):
        raise ValueError("full covariance is not symmetric")
    if np.any(np.diag(arrays["cov_gaussian"]) <= 0.0):
        raise ValueError("Gaussian covariance diagonal is not positive")
    if np.any(np.diag(arrays["trispectrum_binned"]) < 0.0):
        raise ValueError("trispectrum has a negative diagonal")

    eigenvalues = np.linalg.eigvalsh(
        0.5 * (arrays["cov_full"] + arrays["cov_full"].T)
    )
    min_eigenvalue = float(eigenvalues.min())
    if min_eigenvalue <= 0.0:
        raise ValueError("full covariance is not positive definite")
    return {
        "max_component_residual": component_residual,
        "max_asymmetry": asymmetry,
        "min_eigenvalue": min_eigenvalue,
        "max_eigenvalue": float(eigenvalues.max()),
        "condition_number": float(eigenvalues.max() / min_eigenvalue),
    }
```

- [ ] **Step 4: Run the complete focused test file**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
CUDA_VISIBLE_DEVICES=1 pytest -q tests/test_l1_m9_covariance.py
```

Expected: `5 passed`.

- [ ] **Step 5: Commit validation**

```bash
git add scripts/compute_l1_m9_simplegnfw_covariance.py \
  tests/test_l1_m9_covariance.py
git commit -m "feat: validate L1_m9 covariance assembly"
```

---

### Task 3: Add the GPU hmfast calculation and artifact writer

**Files:**
- Modify: `scripts/compute_l1_m9_simplegnfw_covariance.py`

**Interfaces:**
- Consumes: fixed D3A cosmology, simple-GNFW constants, and Task 1 helpers.
- Produces: `compute_covariance() -> dict[str, np.ndarray]`.
- Produces: `write_outputs(result) -> dict`.
- Writes the six artifacts named in the approved design.

- [ ] **Step 1: Implement the fixed theory and output constants**

Add:

```python
F_SKY = 1.0
MASS_GRID = np.geomspace(1e11, 1e16, 64)
REDSHIFT_GRID = np.geomspace(0.005, 3.0, 96)
ELL_SMOOTH = np.geomspace(9.0, 1085.0, 50)
PROFILE_PARAMETERS = {
    "P0": 8.13,
    "c500": 1.156,
    "alpha": 1.062,
    "beta": 5.4807,
    "gamma": 0.3292,
    "B": 1.0,
}
```

- [ ] **Step 2: Implement the GPU theory calculation**

Add:

```python
def compute_covariance() -> dict[str, np.ndarray]:
    """Evaluate simple-GNFW power and covariance components with hmfast."""
    devices = jax.devices()
    if not devices or devices[0].platform != "gpu":
        raise RuntimeError(f"CUDA device required, got {devices}")

    halo_model = HaloModel(
        cosmology=D3A_COSMOLOGY,
        mass_definition=MassDefinition(500, "critical"),
        convert_masses=True,
        hm_consistency=False,
    )
    profile = GNFWPressureProfile(**PROFILE_PARAMETERS)
    tracer = tSZTracer(profile=profile)
    ell = jnp.asarray(ELL_SMOOTH)
    mass = jnp.asarray(MASS_GRID)
    redshift = jnp.asarray(REDSHIFT_GRID)

    cl_1h = np.asarray(halo_model.cl_1h(tracer, None, ell, mass, redshift))
    cl_2h = np.asarray(halo_model.cl_2h(tracer, None, ell, mass, redshift))
    prefactor = ELL_SMOOTH * (ELL_SMOOTH + 1.0) / (2.0 * np.pi)
    dl_1h = bin_dl(ELL_SMOOTH, prefactor * cl_1h)
    dl_2h = bin_dl(ELL_SMOOTH, prefactor * cl_2h)
    dl_total = dl_1h + dl_2h

    trispectrum_cl = np.asarray(
        halo_model.trispectrum_1h(
            tracer, None, ell, ell, mass, redshift
        )
    )
    trispectrum_binned = bin_trispectrum(ELL_SMOOTH, trispectrum_cl)
    cov_gaussian = gaussian_covariance(dl_total, F_SKY)
    cov_full = assemble_covariance(
        cov_gaussian, trispectrum_binned, F_SKY
    )
    return {
        "dl_1h": dl_1h,
        "dl_2h": dl_2h,
        "dl_total": dl_total,
        "cov_gaussian": cov_gaussian,
        "trispectrum_binned": trispectrum_binned,
        "cov_full": cov_full,
    }
```

- [ ] **Step 3: Implement artifact writing and metadata**

Add:

```python
def write_outputs(
    result: dict[str, np.ndarray],
    runtime_seconds: float,
) -> dict:
    """Validate and save the theory bandpowers and covariance artifacts."""
    diagnostics = validate_covariance(
        result["cov_gaussian"],
        result["trispectrum_binned"],
        result["cov_full"],
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(
        OUTPUT_DIR / "cov_gaussian_L1_m9_fullsky_Dl_yy_binned_18.npy",
        result["cov_gaussian"],
    )
    np.save(
        OUTPUT_DIR / "trispectrum_L1_m9_fullsky_Dl_yy_binned_18.npy",
        result["trispectrum_binned"],
    )
    np.save(
        OUTPUT_DIR / "cov_full_L1_m9_fullsky_Dl_yy_binned_18.npy",
        result["cov_full"],
    )
    np.savetxt(
        OUTPUT_DIR / "cov_full_L1_m9_fullsky_Dl_yy_binned_18.csv",
        result["cov_full"],
        delimiter=",",
        fmt="%.16e",
    )
    np.savetxt(
        OUTPUT_DIR / "Dl_yy_simplegnfw_B1_theory_binned_18.txt",
        np.column_stack(
            [ELL_EFF, result["dl_1h"], result["dl_2h"], result["dl_total"]]
        ),
        header="ell_eff  D_ell_1h  D_ell_2h  D_ell_total",
        fmt="%.16e",
    )

    cosmology = {
        key: float(getattr(D3A_COSMOLOGY, key))
        for key in (
            "H0", "omega_cdm", "omega_b", "ln1e10A_s",
            "n_s", "tau_reio", "m_ncdm",
        )
    }
    metadata = {
        "data_file": str(
            REPO / "data_paper/binned_bandpowers/"
            "Dl_yy_L1_m9_fullsky_binned_18.txt"
        ),
        "cosmology": cosmology,
        "pressure_profile": "hmfast.GNFWPressureProfile",
        "profile_parameters": PROFILE_PARAMETERS,
        "sigma_lnY": 0.0,
        "f_sky": F_SKY,
        "mass_units": "physical M_sun",
        "mass_grid": {
            "minimum": float(MASS_GRID[0]),
            "maximum": float(MASS_GRID[-1]),
            "count": int(MASS_GRID.size),
        },
        "redshift_grid": {
            "minimum": float(REDSHIFT_GRID[0]),
            "maximum": float(REDSHIFT_GRID[-1]),
            "count": int(REDSHIFT_GRID.size),
        },
        "ell_grid_count": int(ELL_SMOOTH.size),
        "ell_effective": ELL_EFF.tolist(),
        "gaussian_power_terms": ["1h", "2h"],
        "non_gaussian_term": "connected 1h trispectrum / (4 pi f_sky)",
        "jax_devices": [str(device) for device in jax.devices()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runtime_seconds": float(runtime_seconds),
        "diagnostics": diagnostics,
    }
    with (
        OUTPUT_DIR / "covariance_L1_m9_fullsky_metadata.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata


def main() -> None:
    started = time.perf_counter()
    result = compute_covariance()
    metadata = write_outputs(result, time.perf_counter() - started)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print("Gaussian sigma:", np.sqrt(np.diag(result["cov_gaussian"])))
    print("Full sigma:", np.sqrt(np.diag(result["cov_full"])))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all focused unit tests**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
CUDA_VISIBLE_DEVICES=1 pytest -q tests/test_l1_m9_covariance.py
```

Expected: `5 passed`.

- [ ] **Step 5: Commit the GPU calculation**

```bash
git add scripts/compute_l1_m9_simplegnfw_covariance.py
git commit -m "feat: compute simple-GNFW L1_m9 covariance"
```

---

### Task 4: Run the production calculation and audit its outputs

**Files:**
- Create at runtime: `data_paper/covariance/cov_gaussian_L1_m9_fullsky_Dl_yy_binned_18.npy`
- Create at runtime: `data_paper/covariance/trispectrum_L1_m9_fullsky_Dl_yy_binned_18.npy`
- Create at runtime: `data_paper/covariance/cov_full_L1_m9_fullsky_Dl_yy_binned_18.npy`
- Create at runtime: `data_paper/covariance/cov_full_L1_m9_fullsky_Dl_yy_binned_18.csv`
- Create at runtime: `data_paper/covariance/Dl_yy_simplegnfw_B1_theory_binned_18.txt`
- Create at runtime: `data_paper/covariance/covariance_L1_m9_fullsky_metadata.json`

**Interfaces:**
- Consumes: the GPU script from Task 3.
- Produces: a validated covariance ready for the later Cobaya likelihood.

- [ ] **Step 1: Confirm GPU 1 remains available**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu \
  --format=csv,noheader
```

Expected: GPU 1 has enough free memory for the 64-by-96 hmfast grids. If GPU
1 is occupied, select the idle GPU explicitly rather than contending with
another process.

- [ ] **Step 2: Run the covariance calculation on the selected GPU**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda \
  python scripts/compute_l1_m9_simplegnfw_covariance.py
```

Expected: exit code 0; metadata lists a CUDA device; all six artifacts are
written under `data_paper/covariance/`.

- [ ] **Step 3: Independently reload and audit all output artifacts**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
python - <<'PY'
import json
from pathlib import Path

import numpy as np

root = Path("data_paper/covariance")
g = np.load(root / "cov_gaussian_L1_m9_fullsky_Dl_yy_binned_18.npy")
t = np.load(root / "trispectrum_L1_m9_fullsky_Dl_yy_binned_18.npy")
f = np.load(root / "cov_full_L1_m9_fullsky_Dl_yy_binned_18.npy")
csv = np.loadtxt(
    root / "cov_full_L1_m9_fullsky_Dl_yy_binned_18.csv", delimiter=","
)
theory = np.loadtxt(root / "Dl_yy_simplegnfw_B1_theory_binned_18.txt")
metadata = json.loads(
    (root / "covariance_L1_m9_fullsky_metadata.json").read_text()
)

assert g.shape == t.shape == f.shape == csv.shape == (18, 18)
assert theory.shape == (18, 4)
assert all(np.all(np.isfinite(x)) for x in (g, t, f, csv, theory))
np.testing.assert_allclose(f, g + t / (4.0 * np.pi), rtol=1e-14, atol=0.0)
np.testing.assert_allclose(csv, f, rtol=1e-14, atol=0.0)
np.testing.assert_allclose(f, f.T, rtol=1e-12, atol=0.0)
np.testing.assert_allclose(theory[:, 3], theory[:, 1] + theory[:, 2])
assert np.all(theory[:, 1:4] > 0.0)
assert np.linalg.eigvalsh(0.5 * (f + f.T)).min() > 0.0
assert metadata["sigma_lnY"] == 0.0
assert metadata["profile_parameters"]["B"] == 1.0
assert metadata["gaussian_power_terms"] == ["1h", "2h"]
assert any("cuda" in item.lower() for item in metadata["jax_devices"])
print("min eigenvalue:", np.linalg.eigvalsh(f).min())
print("condition number:", np.linalg.cond(f))
print("Gaussian sigma:", np.sqrt(np.diag(g)))
print("Full sigma:", np.sqrt(np.diag(f)))
PY
```

Expected: all assertions pass and the diagnostic values are finite.

- [ ] **Step 4: Run the relevant regression tests**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
CUDA_VISIBLE_DEVICES=1 pytest -q tests/test_l1_m9_covariance.py tests/test_theory.py
git diff --check
git status --short
```

Expected: all tests pass, `git diff --check` is silent, and only the intended
code plus the user's pre-existing changes appear in status.

- [ ] **Step 5: Record the covariance handoff**

Report the full covariance path, component paths, profile/cosmology
conventions, minimum eigenvalue, condition number, and the 18 full standard
deviations. Keep the overall goal active and proceed to the separately
designed fixed-cosmology Cobaya fit only after this covariance milestone is
verified.
