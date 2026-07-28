# L1_m9 Best-Fit `q` Catalogues Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate nine new `_qfrommz_bestfit.csv` catalogues with the updated custom-GNFW scaling relation while preserving the existing fiducial `B = 1.35` catalogues unchanged.

**Architecture:** A focused Python command streams each existing reduced CSV through pandas, evaluates `flamingo.cnc.SZScaling.q` on paired physical `(M_500c, z)` values, and atomically installs a sibling output after validation. Unit tests use a tiny real CSV and an injected deterministic scaling object; final validation streams the large source/output pairs.

**Tech Stack:** Python 3.10+, pandas, NumPy, JAX, hmfast, pytest.

## Global Constraints

- Activate `/scratch/scratch-lxu/venv/cmbagent_env/bin/activate` before every command.
- All theory calculations use `hmfast`.
- Catalogue masses are physical `M_sun`, not `M_sun/h`.
- Use `A_SZ = -4.1095805`, `alpha_SZ = 0.97447729`, and `B = 1.41`.
- Retain D3A cosmology, `sigma_lnY = 0.173`, scatter seed `20260630`, and the Planck-like SZiFi `immf6` noise curve.
- Never rename, move, overwrite, or edit an existing `_qfrommz.csv` fiducial `B = 1.35` catalogue.
- New outputs end exactly in `_qfrommz_bestfit.csv`.
- Preserve each source's header, row order, and every non-`q_from_mz` value.

---

### Task 1: Test and implement the streaming best-fit catalogue writer

**Files:**
- Create: `scripts/create_l1_m9_bestfit_q_catalogues.py`
- Create: `tests/test_bestfit_q_catalogues.py`

**Interfaces:**
- Consumes: `flamingo.cnc.SZScaling`, the two noise files in `paper_results.config`, and source CSV paths.
- Produces: `bestfit_scaling() -> SZScaling`.
- Produces: `bestfit_path(source: pathlib.Path) -> pathlib.Path`.
- Produces: `rewrite_catalogue(source: pathlib.Path, output: pathlib.Path, scaling: SZScaling, *, force: bool = False, chunksize: int = 1_000_000) -> CatalogueSummary`.
- Produces: CLI options `--variant` (repeatable), `--force`, `--chunksize`, and `--catalogue-dir`.

- [ ] **Step 1: Write the failing naming and parameter tests**

Create `tests/test_bestfit_q_catalogues.py` with imports loaded from the script
through `importlib.util.spec_from_file_location`, then add:

```python
def test_bestfit_path_adds_suffix_before_csv(tmp_path):
    source = tmp_path / "halo_qfrommz.csv"
    assert module.bestfit_path(source) == tmp_path / "halo_qfrommz_bestfit.csv"


def test_bestfit_scaling_uses_rerun_parameters():
    scaling = module.bestfit_scaling()
    assert scaling.A_SZ == pytest.approx(-4.1095805)
    assert scaling.alpha_SZ == pytest.approx(0.97447729)
    assert scaling.B == pytest.approx(1.41)
    assert scaling.sigma_lnY == pytest.approx(0.173)
    assert scaling.seed == 20260630
```

- [ ] **Step 2: Run the tests and verify the expected failure**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest -q tests/test_bestfit_q_catalogues.py
```

Expected: collection fails because
`scripts/create_l1_m9_bestfit_q_catalogues.py` does not exist.

- [ ] **Step 3: Add failing streamed-rewrite and overwrite-safety tests**

Use this two-row source fixture:

```python
source.write_text(
    "# fiducial B=1.35\n"
    "soap_index,z,M_500c_Msun,R_500c_Mpc,q_from_mz\n"
    "10,0.2,1e14,0.7,99\n"
    "20,0.4,2e14,0.9,88\n"
)
```

Define a real deterministic test object:

```python
class FakeScaling:
    A_SZ = -4.1095805
    alpha_SZ = 0.97447729
    B = 1.41
    sigma_lnY = 0.173
    seed = 20260630

    def q(self, mass, redshift, *, index):
        return mass / 1e14 + redshift + index / 100
```

Assert that `rewrite_catalogue`:

1. writes two rows with unchanged column order;
2. leaves all non-`q_from_mz` columns equal to the source;
3. writes `q_from_mz == [1.3, 2.6]`;
4. records both the source and best-fit parameters in leading `#` comments;
5. raises `FileExistsError` on a second call without `force=True`;
6. removes its temporary sibling if `FakeScaling.q` raises.

- [ ] **Step 4: Implement the minimum streaming command**

Create `scripts/create_l1_m9_bestfit_q_catalogues.py` with:

```python
A_SZ = -4.1095805
ALPHA_SZ = 0.97447729
B = 1.41
CHUNKSIZE = 1_000_000


@dataclass(frozen=True)
class CatalogueSummary:
    source: Path
    output: Path
    rows: int
    q_min: float
    q_max: float


def bestfit_scaling() -> SZScaling:
    base = SZScaling.calibrated(
        B=B,
        alpha_SZ=ALPHA_SZ,
        sigma_y0_file=config.SIGMA_Y0_FILE,
        skyfracs_file=config.SKYFRACS_FILE,
    )
    return replace(base, A_SZ=A_SZ)


def bestfit_path(source: Path) -> Path:
    if not source.name.endswith("_qfrommz.csv"):
        raise ValueError(f"not a q_from_mz catalogue: {source}")
    return source.with_name(source.name.removesuffix(".csv") + "_bestfit.csv")
```

`rewrite_catalogue` must:

1. require `soap_index`, `z`, `M_500c_Msun`, and `q_from_mz`;
2. capture the source size and nanosecond modification time;
3. refuse an existing output unless `force=True`;
4. write provenance comments and chunks to a unique temporary sibling;
5. evaluate `scaling.q(mass, redshift, index=soap_index)` elementwise;
6. reject non-finite or non-positive results;
7. write CSV floating values with `float_format="%.17g"`;
8. verify the accumulated output row count and `q` range;
9. verify the source size and modification time remain unchanged;
10. install with `os.replace(temp, output)` only after success;
11. delete only its explicit temporary path on an exception.

The CLI must iterate the nine exact names from `config.VARIANTS`, allow a
repeatable subset through `--variant`, and print source, output, row count,
`q_min`, `q_max`, and elapsed time.

- [ ] **Step 5: Run the focused tests until green**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest -q tests/test_bestfit_q_catalogues.py
```

Expected: all tests pass with no warnings.

- [ ] **Step 6: Run the existing CNC regression tests**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest -q tests/test_cnc.py tests/test_bestfit_q_catalogues.py
```

Expected: all tests pass with no warnings.

- [ ] **Step 7: Commit the tested command**

```bash
git add scripts/create_l1_m9_bestfit_q_catalogues.py tests/test_bestfit_q_catalogues.py
git commit -m "feat: add best-fit q catalogue generator"
```

---

### Task 2: Generate and validate the fiducial checkpoint

**Files:**
- Read: `/rds/rds-lxu/flamingo/L1_m9/catalogues/halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommz.csv`
- Create: `/rds/rds-lxu/flamingo/L1_m9/catalogues/halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommz_bestfit.csv`

**Interfaces:**
- Consumes: the Task 1 CLI and the fiducial `B = 1.35` catalogue.
- Produces: one complete, independently verified fiducial `B = 1.41` catalogue.

- [ ] **Step 1: Perform read-only preflight checks**

Record source size and modification time, check that the final output does not
already exist, and require at least twice the source size plus 1 GiB free on
the target filesystem. Stop without deleting or overwriting anything if these
checks fail.

- [ ] **Step 2: Generate only the `L1_m9` output**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
python scripts/create_l1_m9_bestfit_q_catalogues.py --variant L1_m9
```

Expected: one `_qfrommz_bestfit.csv` file is atomically installed and the
summary reports finite positive `q`.

- [ ] **Step 3: Independently validate source/output equivalence**

Stream the source and output together in equal-sized chunks. Require:

```python
assert source_columns == output_columns
assert source_rows == output_rows
assert np.array_equal(source["soap_index"], output["soap_index"])
for column in source_columns:
    if column != "q_from_mz":
        assert np.allclose(source[column], output[column], rtol=0, atol=0, equal_nan=True)
assert np.all(np.isfinite(output["q_from_mz"]))
assert np.all(output["q_from_mz"] > 0)
```

Sample the first, middle, and final rows and require their stored `q_from_mz`
to agree with a fresh `bestfit_scaling().q(...)` call at `rtol=5e-15`.
Recheck that the source size and modification time match the preflight values.

---

### Task 3: Generate and validate the eight feedback outputs

**Files:**
- Read: the eight remaining `_qfrommz.csv` catalogues under `/rds/rds-lxu/flamingo/L1_m9/catalogues`
- Create: the eight corresponding `_qfrommz_bestfit.csv` catalogues

**Interfaces:**
- Consumes: the checkpointed Task 1 command and Task 2 validation procedure.
- Produces: the complete nine-variant best-fit catalogue set.

- [ ] **Step 1: Preflight all remaining outputs**

Require all eight sources to exist, all eight final outputs to be absent, and
enough free space for the expected outputs plus temporary files. Record every
source size and nanosecond modification time.

- [ ] **Step 2: Generate the remaining variants**

Run the Task 1 command without `--force`, selecting:

```text
fgas+2sigma
fgas-2sigma
fgas-4sigma
fgas-8sigma
Mstar-1sigma
Mstar-1sigma_fgas-4sigma
Jet
Jet_fgas-4sigma
```

Expected: each variant is atomically installed only after its streamed
calculation succeeds.

- [ ] **Step 3: Validate every source/output pair**

Apply the Task 2 streamed validation to all eight pairs. Require exact
`soap_index` ordering and non-`q` values, equal row counts and headers, finite
positive best-fit `q`, and direct hmfast agreement for the three sampled rows.
Confirm all recorded source sizes and modification times remain unchanged.

- [ ] **Step 4: Run the full automated test suite**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest -q
```

Expected: the full suite passes with no new failures.

- [ ] **Step 5: Report the generated products**

Report all nine absolute output paths with their row counts and sizes, the
best-fit parameter provenance, validation results, and confirmation that the
fiducial `B = 1.35` sources are unchanged. Do not commit the external CSV
outputs to Git.
