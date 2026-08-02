# L2 All-Lightcone Y_5R500c Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backfill official `Y_5R500c_Mpc2` into canonical L2p8_m9 lightcone2–7 q-from-map catalogues.

**Architecture:** Parameterize the already verified identity-safe backfill utility by lightcone index. Execute missing catalogues in three batches of two processes, with independent atomic publication and backups.

**Tech Stack:** Python, NumPy, pandas, hdfstream, pytest.

## Global Constraints

- Keep lightcone0 and lightcone1 unchanged.
- Accept only lightcone indices 0 through 7.
- Fetch by `(snap, soap_index)` from the matching lightcone SOAP target.
- Preserve all old CSV field text exactly.
- Run no more than two catalogue backfills concurrently.

---

### Task 1: Parameterize the backfill utility

**Files:**
- Modify: `scripts/backfill_l2_y5r500.py`
- Create: `tests/test_backfill_l2_y5r500_lightcones.py`

**Interfaces:**
- Produces: `catalogue_path(lightcone: int) -> Path`, `target_key(lightcone: int) -> str`, and lightcone-aware `fetch_y5r500`/`backfill`.

- [ ] **Step 1: Write failing tests**

Assert literal paths and target keys for lightcones 0, 2, and 7; assert `ValueError` for -1 and 8.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_backfill_l2_y5r500_lightcones.py`; expect missing helper failures.

- [ ] **Step 3: Implement minimal parameterization**

Add the two helpers, pass `lightcone` through lookup and backfill, and expose `--lightcone {0,...,7}`. An explicit `--catalogue` may override the derived path for controlled testing.

- [ ] **Step 4: Verify GREEN and existing behavior**

Run all backfill tests and require zero failures.

### Task 2: Backfill missing canonical catalogues

**Files:**
- Modify externally: canonical q-from-map CSVs for lightcones 2–7.

**Interfaces:**
- Produces: six catalogues with appended `Y_5R500c_Mpc2` and six `.pre_y5r500.bak` files.

- [ ] **Step 1: Record immutable inputs**

Record lightcone0/1 checksums and lightcone2–7 inode, size, header, and absence of backup.

- [ ] **Step 2: Run three two-process batches**

Run lightcones `(2,3)`, `(4,5)`, then `(6,7)`. Require both processes in a batch to exit successfully before starting the next batch.

- [ ] **Step 3: Verify every publication**

For each lightcone2–7 pair its backup with the new canonical file and require equal old-row/prefix digests, equal row counts, stable identities, and finite non-negative new values.

- [ ] **Step 4: Verify all-lightcone inventory**

Require all eight headers to contain `Y_5R500c_Mpc2`, verify lightcone0/1 checksums are unchanged, and run the complete repository test suite.

