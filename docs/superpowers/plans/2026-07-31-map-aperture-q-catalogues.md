# Map-Aperture q Catalogues Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a repository-native pipeline that writes `_qfrommap.csv` catalogues for all nine L1_m9 variants and all eight L2p8_m9 lightcones.

**Architecture:** A focused `flamingo.aperture_snr` module owns the tested numerical operations: sky-averaged `sigma_Y500`, raw HEALPix aperture sums, and dataframe enrichment. An importable CLI script owns canonical dataset discovery, streamed CSV reads, provenance, atomic installation, summaries, and the 17-job production run.

**Tech Stack:** Python 3.12, NumPy, pandas, healpy, hmfast through `flamingo.catalogue.theta_500`, pytest.

## Global Constraints

- Run every Python command after `source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate`.
- Process only canonical `M_500c > 5e13 M_sun`, `z < 3`, `_yang26rot_qfrommz.csv` inputs.
- Process nine L1_m9 lightcone-0 feedback variants and L2p8_m9 lightcones 0 through 7.
- Use physical `M_sun` and Mpc units; never introduce `M_sun/h`.
- Use yang26-rotated positions and true `theta_500 = R_500c / D_A(z)` from `flamingo.catalogue.theta_500`.
- Sum raw map pixels with no background subtraction, beam correction, pixel-window correction, or fractional boundary pixels.
- Use HEALPix RING pixel centres; zero-pixel apertures have zero signal and are valid.
- Use `data/noise/sigma_Y500_dict_szifi.npy`, `data/noise/skyfracs_szifi_cosmology.npy`, `immf6`, a sky-fraction-weighted arithmetic mean, and a cubic log-log fit over 0.5 to 32 arcmin.
- Preserve source row order and all source columns except `q_from_mz`.
- Append `theta_500_arcmin`, `Y_500cyl_arcmin2`, `sigma_Y500_arcmin2`, `npix_in_aperture`, and `q_from_aperture` in that order.
- Never modify an input catalogue. Install each output atomically only after its complete source succeeds.
- Keep large maps, input catalogues, noise arrays, and `_qfrommap.csv` products untracked.

---

### Task 1: Noise fit and raw HEALPix aperture integrator

**Files:**
- Create: `src/flamingo/aperture_snr.py`
- Create: `tests/test_aperture_snr.py`

**Interfaces:**
- Consumes: nested SZiFi noise dictionaries, sky-fraction arrays, an in-memory HEALPix map, and per-object angular coordinates/radii.
- Produces: `fit_sigma_y500(...) -> np.ndarray`, `sigma_y500_from_theta(...) -> np.ndarray`, and `aperture_y500(...) -> tuple[np.ndarray, np.ndarray]`.

- [ ] **Step 1: Write failing tests for the noise fit**

Add a test that builds two 25-sample tile curves on the standard theta grid,
saves them under `immf6`, assigns weights 1 and 3, and checks the evaluated
fit against their weighted mean:

```python
def test_fit_sigma_y500_uses_hmfast_sky_weighting(tmp_path):
    theta = np.geomspace(0.5, 32.0, 25)
    base = 2.0e-4 * theta**1.5
    noise = {"immf6": {0: base, 1: 2.0 * base}}
    noise_file = tmp_path / "noise.npy"
    sky_file = tmp_path / "sky.npy"
    np.save(noise_file, noise)
    np.save(sky_file, np.array([1.0, 3.0]))

    coeff = fit_sigma_y500(noise_file, sky_file)
    actual = sigma_y500_from_theta(theta, coeff)

    assert np.allclose(actual, 1.75 * base, rtol=1e-12)
```

Also assert that a missing filter, non-positive curve value, mismatched tile
length, or out-of-range sky-fraction tile index raises `ValueError` with the
offending condition in the message.

- [ ] **Step 2: Run the noise tests and confirm the RED state**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest tests/test_aperture_snr.py -k sigma -v
```

Expected: collection fails because `flamingo.aperture_snr` does not exist.

- [ ] **Step 3: Implement the minimal noise API**

Create `src/flamingo/aperture_snr.py` with these signatures:

```python
def fit_sigma_y500(
    noise_file: str | Path,
    skyfracs_file: str | Path,
    *,
    filter_name: str = "immf6",
    theta_min_arcmin: float = 0.5,
    theta_max_arcmin: float = 32.0,
    poly_deg: int = 3,
) -> np.ndarray: ...

def sigma_y500_from_theta(
    theta_500_arcmin: np.ndarray,
    coeff: np.ndarray,
) -> np.ndarray: ...
```

Implement the same arithmetic weighting and `np.polyfit(log(theta),
log(sigma), 3)` sequence as hmfast. Validate all arrays before fitting and
evaluate with `np.exp(np.polyval(coeff, np.log(theta)))` without clipping.

- [ ] **Step 4: Run the noise tests and confirm GREEN**

Run `pytest tests/test_aperture_snr.py -k sigma -v`.

Expected: all selected tests pass.

- [ ] **Step 5: Write failing tests for aperture integration**

Use an `nside=8` RING map with distinct pixel values. Obtain expected pixels
with `hp.query_disc`, then assert float64 integration and pixel counts:

```python
def test_aperture_y500_sums_ring_pixel_centres():
    nside = 8
    ymap = np.arange(hp.nside2npix(nside), dtype=np.float32) * 1e-8
    theta = np.array([1.0])
    phi = np.array([2.0])
    radius = np.array([0.2])
    pix = hp.query_disc(nside, hp.ang2vec(theta[0], phi[0]), radius[0])
    conversion = (180.0 * 60.0 / np.pi) ** 2

    y500, npix = aperture_y500(ymap, theta, phi, radius)

    expected = ymap[pix].sum(dtype=np.float64) * hp.nside2pixarea(nside) * conversion
    assert y500 == pytest.approx([expected])
    assert np.array_equal(npix, [len(pix)])
```

Add cases for an off-centre radius small enough to contain no pixel centre,
shape mismatch, non-finite inputs, and non-positive radii.

- [ ] **Step 6: Run aperture tests and confirm the RED state**

Run `pytest tests/test_aperture_snr.py -k aperture -v`.

Expected: failures because `aperture_y500` is not implemented.

- [ ] **Step 7: Implement the minimal aperture API**

Add:

```python
def aperture_y500(
    ymap: np.ndarray,
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
    theta_500_rad: np.ndarray,
    *,
    nest: bool = False,
) -> tuple[np.ndarray, np.ndarray]: ...
```

Infer `nside`, compute one pixel solid angle, loop over the objects, call
`hp.query_disc`, accumulate each selected map sum with `dtype=np.float64`,
convert sr to arcmin squared, and store pixel counts as `int32`.

- [ ] **Step 8: Run the complete module tests**

Run `pytest tests/test_aperture_snr.py -v`.

Expected: all tests pass with no warnings.

- [ ] **Step 9: Commit Task 1 only**

```bash
git add src/flamingo/aperture_snr.py tests/test_aperture_snr.py
git commit src/flamingo/aperture_snr.py tests/test_aperture_snr.py \
  -m "feat: add empirical aperture SNR kernels"
```

---

### Task 2: Catalogue chunk transformation

**Files:**
- Modify: `src/flamingo/aperture_snr.py`
- Modify: `tests/test_aperture_snr.py`

**Interfaces:**
- Consumes: `aperture_y500`, `sigma_y500_from_theta`, an input dataframe, an in-memory map, fitted coefficients, and optionally a deterministic theta function for tests.
- Produces: `catalogue_chunk_to_qfrommap(...) -> pd.DataFrame` with the exact output schema.

- [ ] **Step 1: Write a failing schema and identity test**

Construct a two-row dataframe with identifiers, required geometry, and
`q_from_mz`. Supply a deterministic `theta500_fn` and compare the output to a
direct call of the Task 1 kernels:

```python
def test_catalogue_chunk_replaces_qfrommz_with_aperture_columns():
    frame = pd.DataFrame({
        "snap": [20, 21],
        "soap_index": [7, 9],
        "z": [0.2, 0.4],
        "R_500c_Mpc": [0.8, 0.9],
        "theta_rot_rad": [1.0, 1.2],
        "phi_rot_rad": [2.0, -1.0],
        "q_from_mz": [99.0, 98.0],
    })
    radii = np.array([0.2, 0.15])
    theta500_fn = lambda radius, redshift: radii

    out = catalogue_chunk_to_qfrommap(
        frame, ymap, coeff, theta500_fn=theta500_fn
    )

    assert "q_from_mz" not in out
    assert out.columns[-5:].tolist() == APERTURE_COLUMNS
    assert out[["snap", "soap_index"]].to_numpy().tolist() == [[20, 7], [21, 9]]
    assert np.allclose(out["q_from_aperture"],
                       out["Y_500cyl_arcmin2"] / out["sigma_Y500_arcmin2"])
```

Add tests requiring `z`, `R_500c_Mpc`, `theta_rot_rad`, `phi_rot_rad`, and
`q_from_mz`, and rejecting non-finite or non-positive derived quantities while
allowing finite negative aperture signal and q.

- [ ] **Step 2: Run the chunk tests and confirm RED**

Run `pytest tests/test_aperture_snr.py -k catalogue_chunk -v`.

Expected: failure because `catalogue_chunk_to_qfrommap` is missing.

- [ ] **Step 3: Implement the dataframe transformation**

Add constants and the function:

```python
APERTURE_COLUMNS = [
    "theta_500_arcmin",
    "Y_500cyl_arcmin2",
    "sigma_Y500_arcmin2",
    "npix_in_aperture",
    "q_from_aperture",
]

def catalogue_chunk_to_qfrommap(
    frame: pd.DataFrame,
    ymap: np.ndarray,
    noise_coeff: np.ndarray,
    *,
    theta500_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = theta_500,
) -> pd.DataFrame: ...
```

Copy the frame, compute true theta in radians and arcminutes, call the Task 1
kernels, drop `q_from_mz`, append the five columns, and validate finiteness,
positive theta/noise, shapes, and exact column placement.

- [ ] **Step 4: Run Task 2 and all module tests**

Run `pytest tests/test_aperture_snr.py -v`.

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2 only**

```bash
git add src/flamingo/aperture_snr.py tests/test_aperture_snr.py
git commit src/flamingo/aperture_snr.py tests/test_aperture_snr.py \
  -m "feat: transform catalogue chunks to aperture q"
```

---

### Task 3: Canonical job discovery and atomic streamed writer

**Files:**
- Create: `scripts/compute_qfrommap_catalogues.py`
- Create: `tests/test_qfrommap_catalogues.py`

**Interfaces:**
- Consumes: `catalogue_chunk_to_qfrommap`, `fit_sigma_y500`, canonical FLAMINGO directory layouts, map FITS files, and commented source CSVs.
- Produces: immutable `CatalogueJob`, `discover_jobs(...)`, `rewrite_catalogue(...)`, provenance comments, CLI summaries, and atomic `_qfrommap.csv` outputs.

- [ ] **Step 1: Write failing discovery tests**

Load the script with `importlib.util.spec_from_file_location`. Build temporary
L1 and L2 directory trees with empty canonical source/map files and assert:

```python
jobs = mod.discover_jobs(tmp_path, dataset="all", only=())
assert [(j.dataset, j.label) for j in jobs] == [
    ("l1", "L1_m9"),
    ("l2", "lightcone0"),
]
assert jobs[0].output.name.endswith("_yang26rot_qfrommap.csv")
assert jobs[1].map_path.name == "y_unlensed_L2p8_m9_lc0.fits"
```

Include `_alpha_fixed_1p12.csv`, `_qfrommap.csv`, and `1e13` distractors and
prove they are ignored. Add failures for missing maps, malformed lightcone
names, duplicate canonical inputs, and no selected jobs. Verify `dataset` and
repeatable `only` filtering.

- [ ] **Step 2: Run discovery tests and confirm RED**

Run `pytest tests/test_qfrommap_catalogues.py -k discover -v`.

Expected: failure because the script does not exist.

- [ ] **Step 3: Implement job discovery**

Define:

```python
@dataclass(frozen=True)
class CatalogueJob:
    dataset: str
    label: str
    source: Path
    map_path: Path
    output: Path

def discover_jobs(
    flamingo_root: Path,
    *,
    dataset: str = "all",
    only: tuple[str, ...] = (),
) -> list[CatalogueJob]: ...
```

Use exact glob patterns from the spec, derive names with anchored prefix and
suffix removal, validate every selected triple, sort L1 by variant and L2 by
integer lightcone index, then concatenate L1 before L2.

- [ ] **Step 4: Run discovery tests and confirm GREEN**

Run `pytest tests/test_qfrommap_catalogues.py -k discover -v`.

Expected: all discovery tests pass.

- [ ] **Step 5: Write failing atomic-writer tests**

Build a small valid RING FITS map and a commented two-row CSV. Use a linear
noise coefficient representing constant positive noise. Assert:

```python
summary = mod.rewrite_catalogue(job, coeff, chunk_size=1)
out = pd.read_csv(job.output, comment="#")
assert summary["rows"] == 2
assert out.columns[-5:].tolist() == APERTURE_COLUMNS
assert "q_from_mz" not in out
assert out[["snap", "soap_index"]].equals(source[["snap", "soap_index"]])
```

Verify provenance contains the source and map paths. Verify an existing output
raises `FileExistsError` without force. For cleanup, make the second CSV chunk
contain a non-finite required value, run with `force=True` over an existing
sentinel output, assert the sentinel is unchanged, and assert no
`*.tmp-*` sibling remains.

- [ ] **Step 6: Run writer tests and confirm RED**

Run `pytest tests/test_qfrommap_catalogues.py -k rewrite -v`.

Expected: failure because `rewrite_catalogue` is missing.

- [ ] **Step 7: Implement provenance and the atomic writer**

Add:

```python
def read_header(path: Path) -> list[str]: ...

def provenance(job: CatalogueJob, previous: list[str]) -> str: ...

def rewrite_catalogue(
    job: CatalogueJob,
    noise_coeff: np.ndarray,
    *,
    chunk_size: int = 100_000,
    force: bool = False,
) -> dict[str, object]: ...
```

Read the map once as float32. Create a unique temporary sibling with exclusive
mode, write provenance, stream `pd.read_csv(..., comment="#",
float_precision="round_trip")`, transform and append each chunk with
`float_format="%.17g"`, update counts for q cuts `(1, 5, 10, 20, 50)`, and
verify non-empty input plus unchanged source stat before `os.replace`.
Unlink the temporary in `except BaseException` and re-raise.

- [ ] **Step 8: Implement and test the CLI**

Add `main()` arguments `--flamingo-root`, `--dataset`, repeatable `--only`,
`--chunk-size`, `--dry-run`, and `--force`. Default the root to
`/rds/rds-lxu/flamingo`, noise paths to repository `data/noise`, dataset to
`all`, and chunk size to `100_000`. Discover and validate all jobs before
loading noise or writing. Print every triple on dry-run and one summary row per
completed job.

Test `main()` by monkeypatching `sys.argv` and `rewrite_catalogue` so dry-run
performs no writes and selected execution passes the parsed options.

- [ ] **Step 9: Run all focused tests**

Run:

```bash
pytest tests/test_aperture_snr.py tests/test_qfrommap_catalogues.py -v
```

Expected: all tests pass with no warnings.

- [ ] **Step 10: Commit Task 3 only**

```bash
git add scripts/compute_qfrommap_catalogues.py tests/test_qfrommap_catalogues.py
git commit scripts/compute_qfrommap_catalogues.py tests/test_qfrommap_catalogues.py \
  -m "feat: add atomic q-from-map catalogue pipeline"
```

---

### Task 4: Repository regression and real fiducial validation

**Files:**
- Modify only if a failure requires an in-scope correction:
  `src/flamingo/aperture_snr.py`, `scripts/compute_qfrommap_catalogues.py`,
  `tests/test_aperture_snr.py`, `tests/test_qfrommap_catalogues.py`

**Interfaces:**
- Consumes: the complete Task 1-3 pipeline and real fiducial L1 RDS inputs.
- Produces: a clean repository test result, a validated dry-run job manifest,
  and a fiducial `_qfrommap.csv` matching the established counts.

- [ ] **Step 1: Run the complete repository test suite**

Run:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pytest -q
```

Expected: all repository tests pass. If a relevant failure appears, first add
or tighten a focused failing regression test, confirm RED, make the smallest
in-scope correction, and rerun both focused and full suites.

- [ ] **Step 2: Verify the complete production manifest without writes**

Run:

```bash
python scripts/compute_qfrommap_catalogues.py --dry-run
```

Expected: exactly 17 triples: nine L1 jobs followed by L2 lightcones 0-7, each
with an existing map and a distinct output ending `_qfrommap.csv`.

- [ ] **Step 3: Run the fiducial L1 catalogue**

Run:

```bash
/usr/bin/time -v python scripts/compute_qfrommap_catalogues.py \
  --dataset l1 --only L1_m9_yang26rot --chunk-size 100000
```

Expected: one complete output installed beside the fiducial source and no
temporary sibling left behind.

- [ ] **Step 4: Stream-validate the fiducial output**

Read source/output in paired chunks and assert equal row count, equal
`snap`/`soap_index`, exact output schema, and finite empirical values. Count
`q_from_aperture` thresholds and require:

```python
{50: 3, 20: 38, 10: 271, 5: 1336, 1: 25402}
```

If these fail, treat it as a bug: write a failing test that isolates the
discrepancy before changing production code.

- [ ] **Step 5: Commit any Task 4 correction only after RED-GREEN verification**

If no correction was required, do not create an empty commit. Otherwise use a
path-limited commit containing only the focused fix and regression test.

---

### Task 5: Full L1 and L2 production run and completion audit

**Files:**
- Create externally: 8 remaining L1 `_qfrommap.csv` products under
  `/rds/rds-lxu/flamingo/L1_m9/catalogues`
- Create externally: 8 L2 `_qfrommap.csv` products, one in each
  `/rds/rds-lxu/flamingo/L2p8_m9/lightconeN/catalogues`
- Do not modify repository source unless verification exposes a tested bug.

**Interfaces:**
- Consumes: the verified CLI and all canonical RDS source/map pairs.
- Produces: all 17 requested output catalogues and a requirement-by-requirement audit.

- [ ] **Step 1: Run every remaining production job**

Because the fiducial L1 output already exists, run with `--force` so the
default all-job command has one consistent behavior:

```bash
/usr/bin/time -v python scripts/compute_qfrommap_catalogues.py \
  --dataset all --chunk-size 100000 --force
```

Capture per-job rows, elapsed seconds, zero-pixel count, q-threshold counts,
overall wall/CPU time, and peak memory.

- [ ] **Step 2: Audit all 17 source/output pairs independently**

Use a read-only streaming verifier separate from `rewrite_catalogue`. For each
pair require:

```python
assert output.exists()
assert output_columns == [c for c in source_columns if c != "q_from_mz"] + APERTURE_COLUMNS
assert source_rows == output_rows
assert np.array_equal(source_ids, output_ids)
assert np.isfinite(output[APERTURE_COLUMNS].to_numpy()).all()
assert (output["theta_500_arcmin"] > 0).all()
assert (output["sigma_Y500_arcmin2"] > 0).all()
assert np.allclose(
    output["q_from_aperture"],
    output["Y_500cyl_arcmin2"] / output["sigma_Y500_arcmin2"],
    rtol=2e-15,
    atol=0.0,
)
```

Accumulate and report row totals, zero-pixel totals, and q-threshold counts for
each catalogue. Require exactly 17 passing pairs and zero temporary files.

- [ ] **Step 3: Run final code verification**

Run:

```bash
pytest tests/test_aperture_snr.py tests/test_qfrommap_catalogues.py -v
pytest -q
git diff --check
```

Expected: focused and full tests pass, and no whitespace errors occur in the
task-owned files.

- [ ] **Step 4: Review task-owned diff and external outputs**

Run path-limited status/diff commands so pre-existing unrelated worktree
changes are not confused with this feature:

```bash
git status --short -- \
  src/flamingo/aperture_snr.py \
  scripts/compute_qfrommap_catalogues.py \
  tests/test_aperture_snr.py \
  tests/test_qfrommap_catalogues.py \
  docs/superpowers/specs/2026-07-31-map-aperture-q-catalogues-design.md \
  docs/superpowers/plans/2026-07-31-map-aperture-q-catalogues.md
git log --oneline -5
```

Expected: only intentional task files appear, every source change is committed,
and all 17 external outputs remain untracked.
