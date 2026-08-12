# Repository Cleanup Design

## Goal

Make the repository easier to navigate and maintain by removing repeated script
implementations and giving reusable scientific operations one tested home,
without changing numerical results, physical conventions, or stored products.

## Non-negotiable invariants

- Theory calculations remain in `hmfast`; this refactor does not change theory.
- Masses remain physical `M_sun`, never `M_sun/h`.
- Existing Planck-bin edges, inclusive-edge behavior, log-bin construction,
  mask conventions, apodization settings, pixel-window correction, output
  units, filenames, and metadata meanings remain unchanged.
- Existing tracked files under `data_paper/`, `figures/`, and `paper_results/`
  are not regenerated or edited by the cleanup.
- No new runtime dependency is added.

## Architecture

### Reusable numerical code

Reusable operations live under `src/flamingo/`, not in executable scripts.

- `flamingo.powerspectra.bandpowers` owns the canonical Planck bin definitions,
  the exact inclusive 18-bin reduction, the exact 12-bin logarithmic reduction,
  and the two-column writer.
- `flamingo.powerspectra.namaster` owns the exact per-multipole decoupled
  NaMaster estimator in addition to its existing binned estimator.
- Existing `flamingo.masking.disc_mask(..., inclusive=False)` replaces local
  binary-disc implementations while retaining `healpy.query_disc` default
  boundary behavior.

Characterization tests lock the old formulas before consumers switch to these
shared functions. Refactoring consumers must call the shared implementations;
they must not wrap them with new one-line delegation helpers.

### Canonical workflows

Each scientific workflow has one command with arguments for its real variants.

- Feedback bandpowers: one command supports selection, q cuts, variants,
  workers, force, and dry-run. The fixed-q and q>1 duplicate commands disappear.
- Fiducial masked bandpowers: the L1 and L2 commands accept explicit q cuts;
  the standalone q>6 command disappears.
- Feedback plots: one command supports the multi-cut paper figure and the
  existing single-cut figure; the q>1-only plot command disappears.
- Fixed-alpha postprocessing: one command selects L1 or L2 while preserving the
  existing summaries, figure names, and labels.

Commands remain thin orchestration layers: parse arguments, select paths, call
package functions, and write the same products. A script must never dynamically
load another script with `importlib.util`.

### Repository layout

After workflow consolidation, remaining commands are grouped by purpose:

- `scripts/catalogue/`
- `scripts/powerspectra/`
- `scripts/inference/`
- `scripts/figures/`

Reusable code stays in `src/flamingo/`; tests stay in `tests/`. Documentation
and tests are updated to the new command paths in the same change. No compatibility
wrapper scripts are retained: Git history is the compatibility mechanism.

## Verification

Before refactoring, record the full test result and SHA-256 checksums of tracked
scientific products. Each extraction follows red-green TDD with a focused
characterization test. After every workflow consolidation, run its focused
tests. Completion requires:

1. the full test suite passes;
2. numerical characterization tests pass with exact or existing-tolerance
   equality, as appropriate;
3. tracked scientific-product checksums match the baseline;
4. no production script uses `importlib.util` to load another script;
5. no duplicate local definitions remain for the canonical estimator or bin
   constants;
6. no new dependency appears in `pyproject.toml`.

## Out of scope

- Recomputing, improving, or reinterpreting scientific results.
- Changing inference priors, covariance construction, catalogue selection, or
  plotting aesthetics.
- Refactoring `hmfast` or changing public scientific APIs unrelated to the
  duplicated script workflows.
