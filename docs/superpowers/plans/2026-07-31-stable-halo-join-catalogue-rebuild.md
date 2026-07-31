# Stable Halo Join Catalogue Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild and publish all 68 canonical L1_m9 and L2p8_m9 catalogue CSVs through a stable `HaloCatalogueIndex` join while archiving every predecessor under a mirrored timestamped tree.

**Architecture:** A tested catalogue core defines targets, stable joins, family-specific schemas, rotation geometry, validation, and transactional publication. A resumable command streams one snapshot at a time into a staging tree, derives both q-from-mz flavours and q-from-map, validates each file, then moves the predecessor to the archive and installs the staged file at the unchanged canonical path.

**Tech Stack:** Python 3.12, NumPy, pandas, hdfstream, healpy, pytest, existing `flamingo.cnc` and `flamingo.aperture_snr` modules.

## Global Constraints

- Run every command after `source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate`.
- Use physical masses in `M_sun`, never `M_sun/h`.
- Never join a halo lightcone to SOAP by `InputHalos/SOAPIndex`.
- Preserve every canonical CSV filename and family-specific column order.
- Build and validate in staging before moving any canonical file.
- Archive predecessors below `/rds/rds-lxu/flamingo/archive/hbt_join_fix_20260731/` and never overwrite an archive entry.
- Do not modify maps, masked spectra, bandpowers, covariance products, or plots in this phase.
- Use only the official per-shell rotation convention matching the corresponding map builder.
- Do not touch unrelated changes already present on `paper-results`.

---

### Task 1: Target Matrix and Family Schemas

**Files:**
- Create: `src/flamingo/catalogue/rebuild.py`
- Create: `tests/test_catalogue_rebuild.py`

**Interfaces:**
- Produces: `CatalogueTarget`, `catalogue_targets(root)`, `base_columns(family)`, `q_columns(family)`, and `qmap_columns(family)`.
- Consumes: canonical path conventions already used by `scripts/compute_qfrommap_catalogues.py`.

- [ ] **Step 1: Write the failing target-matrix test**

```python
def test_catalogue_targets_cover_nine_l1_and_eight_l2_families(tmp_path):
    targets = catalogue_targets(tmp_path)
    assert len(targets) == 17
    assert sum(t.family == "l1" for t in targets) == 9
    assert sum(t.family == "l2" for t in targets) == 8
    assert len({p for t in targets for p in t.canonical_csvs}) == 68
```

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest -q tests/test_catalogue_rebuild.py::test_catalogue_targets_cover_nine_l1_and_eight_l2_families`

Expected: FAIL because `rebuild.py` does not yet expose the target API.

- [ ] **Step 3: Implement immutable target metadata and exact schemas**

```python
@dataclass(frozen=True)
class CatalogueTarget:
    family: Literal["l1", "l2"]
    run: str
    variant: str
    lightcone: int
    catalogue_dir: Path
    map_path: Path
```

Define the nine L1 variants, eight L2 lightcones, canonical filenames, and the
29/30/34-column L1 versus 25/26/30-column L2 schemas exactly.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest -q tests/test_catalogue_rebuild.py`

Expected: PASS.

- [ ] **Step 5: Commit only Task 1 files**

```bash
git commit --only src/flamingo/catalogue/rebuild.py tests/test_catalogue_rebuild.py -m "feat: define stable catalogue rebuild targets"
```

### Task 2: Stable Identity Resolution

**Files:**
- Modify: `src/flamingo/catalogue/join.py`
- Modify: `src/flamingo/catalogue/rebuild.py`
- Modify: `tests/test_catalogue_join.py`
- Modify: `tests/test_catalogue_rebuild.py`

**Interfaces:**
- Consumes: `match_hbt_indices(lightcone_hbt, selected_soap_hbt, selected_soap_rows)`.
- Produces: `resolve_lightcone_rows(lightcone_hbt, soap_hbt) -> (lightcone_rows, soap_rows)` with duplicate-SOAP-identity rejection and periodic-copy expansion.

- [ ] **Step 1: Write failing tests for permutation, missing identity, and periodic copies**

```python
def test_resolve_lightcone_rows_uses_identity_not_stale_row_number():
    lc = np.array([30, 10, 30, 40])
    soap = np.array([10, 40, 30])
    lc_rows, soap_rows = resolve_lightcone_rows(lc, soap)
    assert lc_rows.tolist() == [0, 1, 2, 3]
    assert soap_rows.tolist() == [2, 0, 2, 1]
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_catalogue_join.py tests/test_catalogue_rebuild.py -k 'resolve or duplicate or periodic'`

Expected: FAIL on the missing generalized resolver.

- [ ] **Step 3: Implement sort/search identity resolution**

Use a stable sorted lookup, reject duplicate identities in the SOAP snapshot,
and retain every matching lightcone copy. Do not inspect lightcone SOAPIndex.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/test_catalogue_join.py tests/test_catalogue_rebuild.py`

Expected: PASS.

- [ ] **Step 5: Commit only Task 2 files**

```bash
git commit --only src/flamingo/catalogue/join.py src/flamingo/catalogue/rebuild.py tests/test_catalogue_join.py tests/test_catalogue_rebuild.py -m "fix: resolve halo lightcones by stable identity"
```

### Task 3: Family-Specific Snapshot Frames and Rotation

**Files:**
- Modify: `src/flamingo/catalogue/rebuild.py`
- Modify: `tests/test_catalogue_rebuild.py`

**Interfaces:**
- Produces: `build_snapshot_frame(source, target, snap, mass_cut_msun)` and `add_rotated_geometry(frame, shell_radii, angles)`.
- Consumes: official lightcone centres/redshifts, current SOAP fields, and `resolve_lightcone_rows`.

- [ ] **Step 1: Write failing tests for source pairing, physical units, schema, and rotation convention**

```python
def test_snapshot_frame_pairs_lightcone_geometry_with_current_soap_quantities(fake_source):
    frame = build_snapshot_frame(fake_source, TARGET, 75, 1e13)
    assert frame.loc[0, "M_500c_Msun"] == 6e13
    assert frame.loc[0, ["x_Mpc", "y_Mpc", "z_Mpc"]].tolist() == [1.0, 2.0, 3.0]
    assert frame.loc[0, "soap_index"] == 2
```

Also compare `add_rotated_geometry` against direct
`hp.Rotator(rot=[degrees(phi), degrees(theta)], inv=True)` for L1 and L2.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_catalogue_rebuild.py -k 'snapshot or rotation or units or schema'`

Expected: FAIL because snapshot construction is missing.

- [ ] **Step 3: Implement sparse field reads and exact output ordering**

Read full snapshot identity only where required for L1. For L2, validate the
released row hint against stable identity before using it as a fast path.
Convert SOAP mass values by `1e10` to physical `M_sun`; convert comoving SOAP
radii to physical Mpc with `a=1/(1+z)`. Include L1 integrated-Y fields only.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/test_catalogue_rebuild.py`

Expected: PASS.

- [ ] **Step 5: Commit only Task 3 files**

```bash
git commit --only src/flamingo/catalogue/rebuild.py tests/test_catalogue_rebuild.py -m "feat: build identity-safe snapshot catalogue frames"
```

### Task 4: Resumable Staged Base Builder

**Files:**
- Create: `scripts/rebuild_stable_catalogues.py`
- Modify: `src/flamingo/catalogue/rebuild.py`
- Modify: `tests/test_catalogue_rebuild.py`

**Interfaces:**
- Produces CLI commands `inventory`, `build-base`, `derive-q`, `derive-qmap`, `validate`, `publish`, and `run`.
- Produces one progress JSON per target and one global manifest.

- [ ] **Step 1: Write failing tests for snapshot streaming and resume**

```python
def test_base_writer_resumes_only_after_verified_snapshot_part(tmp_path, fake_source):
    build_base(TARGET, tmp_path, fake_source, snaps=(17, 18))
    first = (tmp_path / "parts/0017.csv").read_bytes()
    build_base(TARGET, tmp_path, fake_source, snaps=(17, 18))
    assert (tmp_path / "parts/0017.csv").read_bytes() == first
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_catalogue_rebuild.py -k 'base_writer or resume or progress'`

Expected: FAIL because staged writing is missing.

- [ ] **Step 3: Implement per-snapshot parts and deterministic assembly**

Write each snapshot to a temporary part, validate it, atomically install the
part, record its digest and counts, then assemble parts in snapshot order into
the staged canonical filename. A failed remote request leaves completed parts
reusable and the canonical file untouched.

- [ ] **Step 4: Run tests and CLI dry-run**

Run: `pytest -q tests/test_catalogue_rebuild.py && python scripts/rebuild_stable_catalogues.py inventory`

Expected: tests PASS and inventory lists 17 targets/68 canonical files without writes.

- [ ] **Step 5: Commit only Task 4 files**

```bash
git commit --only scripts/rebuild_stable_catalogues.py src/flamingo/catalogue/rebuild.py tests/test_catalogue_rebuild.py -m "feat: stage resumable catalogue rebuilds"
```

### Task 5: q-from-mz Derived Catalogues

**Files:**
- Modify: `src/flamingo/catalogue/rebuild.py`
- Modify: `scripts/rebuild_stable_catalogues.py`
- Modify: `tests/test_catalogue_rebuild.py`

**Interfaces:**
- Produces: `derive_q_catalogues(base_path, outputs, scaling)`.
- Consumes: `SZScaling` with `A_SZ=-4.0953238`, `alpha_SZ=1.12`, `B=1.41`.

- [ ] **Step 1: Write a failing streamed-derivation test**

```python
def test_derive_q_selects_m500_and_preserves_family_schema(tmp_path):
    summary = derive_q_catalogues(BASE, OUTPUTS, SCALING, chunk_size=1)
    assert summary.rows == 1
    assert read_columns(OUTPUTS[0]) == q_columns("l2")
```

- [ ] **Step 2: Run test and verify RED**

Run: `pytest -q tests/test_catalogue_rebuild.py -k derive_q`

Expected: FAIL because the derived writer is missing.

- [ ] **Step 3: Implement one-pass dual-output derivation**

Stream the staged base once, apply `M_500c > 5e13 M_sun`, compute deterministic
q, and write both existing q-from-mz filenames with exact comments and schema.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/test_catalogue_rebuild.py tests/test_feedback_ratio_vs_q.py`

Expected: PASS.

- [ ] **Step 5: Commit only Task 5 files**

```bash
git commit --only src/flamingo/catalogue/rebuild.py scripts/rebuild_stable_catalogues.py tests/test_catalogue_rebuild.py -m "feat: derive canonical q from mass catalogues"
```

### Task 6: Staged q-from-map Catalogues

**Files:**
- Modify: `scripts/compute_qfrommap_catalogues.py`
- Modify: `scripts/rebuild_stable_catalogues.py`
- Modify: `tests/test_qfrommap_catalogues.py`
- Modify: `tests/test_catalogue_rebuild.py`

**Interfaces:**
- Extends `rewrite_catalogue` with an explicit output path while retaining its current canonical default.
- Consumes: staged q-from-mz source, canonical map, existing noise curve, and no-background aperture logic.

- [ ] **Step 1: Write a failing explicit-staging-output test**

```python
def test_rewrite_catalogue_can_write_staging_without_touching_canonical(mod, tmp_path):
    summary = mod.rewrite_catalogue(job, coeff, output=tmp_path / "stage/out.csv")
    assert Path(summary["output"]).is_file()
    assert job.output.read_text() == "sentinel\n"
```

- [ ] **Step 2: Run test and verify RED**

Run: `pytest -q tests/test_qfrommap_catalogues.py -k staging`

Expected: FAIL because `output=` is unsupported.

- [ ] **Step 3: Implement explicit staged output and orchestration**

Retain the current aperture calculation unchanged. Load each map once per
target, stream the staged q catalogue, validate aperture columns, and install
only into staging.

- [ ] **Step 4: Run aperture and catalogue tests**

Run: `pytest -q tests/test_aperture_snr.py tests/test_qfrommap_catalogues.py tests/test_catalogue_rebuild.py`

Expected: PASS.

- [ ] **Step 5: Commit only Task 6 files**

```bash
git commit --only scripts/compute_qfrommap_catalogues.py scripts/rebuild_stable_catalogues.py tests/test_qfrommap_catalogues.py tests/test_catalogue_rebuild.py -m "feat: stage aperture q catalogues"
```

### Task 7: Validation and Transactional Archive Publication

**Files:**
- Modify: `src/flamingo/catalogue/rebuild.py`
- Modify: `scripts/rebuild_stable_catalogues.py`
- Modify: `tests/test_catalogue_rebuild.py`

**Interfaces:**
- Produces: `validate_catalogue(path, target, flavour)` and `publish_file(staged, canonical, archive, manifest)`.

- [ ] **Step 1: Write failing rollback and validation tests**

```python
def test_publish_failure_rolls_archived_file_back(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "replace", fail_on_second_replace())
    with pytest.raises(OSError):
        publish_file(staged, canonical, archive, manifest)
    assert canonical.read_text() == "old"
    assert staged.read_text() == "new"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_catalogue_rebuild.py -k 'validate or publish or rollback or archive'`

Expected: FAIL because transactional publication is missing.

- [ ] **Step 3: Implement full validation and rollback-safe publication**

Validate schema, finite values, cuts, base-row identity parity, q parity, and
manifest digests. Refuse an existing archive path. Rename old to archive and
staged to canonical; restore old automatically if the second rename fails.

- [ ] **Step 4: Run all focused tests**

Run: `pytest -q tests/test_catalogue_join.py tests/test_catalogue_rebuild.py tests/test_aperture_snr.py tests/test_qfrommap_catalogues.py`

Expected: PASS.

- [ ] **Step 5: Commit only Task 7 files**

```bash
git commit --only src/flamingo/catalogue/rebuild.py scripts/rebuild_stable_catalogues.py tests/test_catalogue_rebuild.py -m "feat: validate and publish archived catalogues"
```

### Task 8: Pilot, Full Production, and Final Audit

**Files:**
- Runtime outputs only under `/rds/rds-lxu/flamingo/.hbt_join_fix_staging/` and `/rds/rds-lxu/flamingo/archive/hbt_join_fix_20260731/`.
- Modify: global manifest generated by `scripts/rebuild_stable_catalogues.py`.

**Interfaces:**
- Consumes all earlier commands.
- Produces 68 validated canonical CSVs at unchanged names and 68 archived predecessors.

- [ ] **Step 1: Run an L2p8 lightcone0 pilot through staging only**

```bash
python scripts/rebuild_stable_catalogues.py run --only L2p8_m9/lightcone0 --no-publish --workers 4
```

Verify: stable identity is 100%, schema matches, and identity-keyed physical
columns agree with the current L2 catalogue within serialization precision.

- [ ] **Step 2: Run an L1 fiducial pilot through staging only**

```bash
python scripts/rebuild_stable_catalogues.py run --only L1_m9/L1_m9 --no-publish --workers 4
```

Verify: known false massive halos are absent and sampled centres agree with
official particles/map signal.

- [ ] **Step 3: Build all base and q-from-mz catalogues resumably**

```bash
/usr/bin/time -v python scripts/rebuild_stable_catalogues.py run --skip-qmap --workers 8
```

Verify: all 17 targets report complete staged base and both q catalogues.

- [ ] **Step 4: Build all q-from-map catalogues**

```bash
/usr/bin/time -v python scripts/rebuild_stable_catalogues.py derive-qmap --all
```

Verify: all 17 q-from-map catalogues have the same 5e13 row identities and no
`q_from_mz` column.

- [ ] **Step 5: Validate the complete staged matrix**

```bash
python scripts/rebuild_stable_catalogues.py validate --all
```

Expected: 68/68 PASS and no canonical path changed.

- [ ] **Step 6: Publish the complete matrix transactionally**

```bash
python scripts/rebuild_stable_catalogues.py publish --all
```

Expected: each old file is in the mirrored archive and each canonical path
contains the validated replacement with its original filename.

- [ ] **Step 7: Run fresh post-publication verification**

```bash
python scripts/rebuild_stable_catalogues.py validate --all --canonical
pytest -q tests/test_catalogue_join.py tests/test_catalogue_rebuild.py tests/test_aperture_snr.py tests/test_qfrommap_catalogues.py
```

Expected: 68/68 canonical validations PASS and all focused tests PASS.

- [ ] **Step 8: Record counts and resource use without calculating downstream products**

Report archive path, manifest path, row counts per target/flavour, q threshold
counts already emitted by q-from-map generation, wall time, and maximum RSS.
Do not regenerate CNC plots, masked spectra, bandpowers, covariance, or paper
plots in this phase.
