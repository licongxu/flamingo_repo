# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Rule of reply:** Always say "HONEY" to me before you reply.

Always `source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate` before running any code or commands.

## Branch: `autoresearch`

This branch hosts autonomous research work, not core `flamingo` library development.

- **Put all research work under `autoresearch/`.** Do not scatter run outputs, fit scripts, diagnostics, logbooks, or draft papers across the repo root. The top-level `src/`, `notebooks/`, `scripts/`, `tests/`, and `figures/` belong to the shared `flamingo` pipeline; treat them as read-mostly here.
- **One subfolder per model/run.** Each research run lives in its own dated, model-named subfolder so results are never overwritten:
  - `autoresearch/opus_4.8_v1/`: completed Opus 4.8 run (fits, diagnostics, `paper.tex`, `logbook.md`, `REPORT.md`, `figures/`).
  - `autoresearch/fable_5_plan/`: planning area for the upcoming Fable 5 run (not started yet).
  - Start each new run in a fresh `autoresearch/<model>_<version>/` folder; do not edit a previous run's folder.
- Follow `.claude/rules/autonomous-operation.md` for the session contract, logbook, and mandatory stop conditions.

## Installed skill packs

Two skill marketplaces are installed; prefer them over ad-hoc process:

- **`superpowers`**: engineering workflow skills. Use `superpowers:brainstorming` before any creative/design work, `superpowers:writing-plans` / `superpowers:executing-plans` for multi-step tasks, `superpowers:test-driven-development` and `superpowers:systematic-debugging` while coding, and `superpowers:verification-before-completion` before claiming anything is done.
- **`academic-research-skills`** (ARS): paper research/writing/review. Use `academic-research-skills:academic-pipeline` for research-to-publication, `academic-paper` for drafting, `academic-paper-reviewer` for simulated peer review, and `deep-research` for literature work. These drive the `paper.tex` deliverables inside each `autoresearch/<run>/` folder.

## Environment and commands

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pip install -e ".[powerspectra,plot,dev]"   # editable install with NaMaster + matplotlib + pytest
pytest -q                                    # fast test suite
pytest tests/test_<mod>.py::test_name -v     # single test
```

## Architecture (the shared `flamingo` package)

`flamingo` (in `src/flamingo/`) is a thin, backend-split toolkit for FLAMINGO lightcone products. The split matters: physics kernels are JAX/GPU, HEALPix-bound code is NumPy because `healpy` pixel queries are CPU-only.

- `profiles/`: **JAX/GPU**, float64. GNFW pressure profile (Arnaud A10 default) and line-of-sight `projected_shape`; jittable and `vmap`-able. This is the only differentiable path.
- `maps/`: NumPy/healpy: map I/O, sampling at positions, neighbour-max sampling.
- `masking/`: NumPy/healpy: binary N×R500 disc masks and sky fraction.
- `catalogue/`: NumPy/hmfast: SOAP CSV loading, `theta_500`, `E(z)`, `D_A(z)`, rotation checks. **Use the rotated columns** (`theta_rot_rad`, `phi_rot_rad`); the L2p8 map is in the yang26-rotated frame.
- `powerspectra/`: pymaster (optional): apodization + mask-decoupled `D_ell`.
- `paths.py`: canonical data paths; override the tree with `export FLAMINGO_ROOT=/path`.

`hmfast` is an external dependency supplying the differentiable halo model, cosmology emulators, tracers, and profiles; the detailed `.claude/rules/` files (api-layering, halo-model, jax-and-numerics, emulators-and-data) describe its conventions and apply when touching halo-model code.

The `notebooks/` (nb05–nb40) and `scripts/` at the repo root carry the masked-tSZ / CNC pipeline (L1_m9 feedback, L2p8 multi-lightcone, shell/rotation-group tSZ power spectra). Read them for method context; new research work still goes under `autoresearch/`.

Detailed conventions live in `.claude/rules/`: `python-style`, `jax-and-numerics`, `paper-writing`, `cluster-usage`, `testing`, `api-layering`, `emulators-and-data`, `autonomous-operation`. Read the relevant one before editing that area.

## Working guidelines

**Tradeoff:** These bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think before coding
Don't assume. Don't hide confusion. State assumptions explicitly and ask if uncertain. If multiple interpretations exist, present them; don't pick silently. If a simpler approach exists, say so.

### 2. Simplicity first
Minimum code that solves the problem, nothing speculative. No features beyond what was asked, no abstractions for single-use code, no unrequested "flexibility". If you write 200 lines and it could be 50, rewrite it.

### 3. Surgical changes
Touch only what you must. Don't "improve" adjacent code, don't refactor what isn't broken, match existing style. Remove only the orphans your own changes created; mention pre-existing dead code, don't delete it. Every changed line should trace directly to the request.

### 4. Goal-driven execution
Turn tasks into verifiable goals ("fix the bug" -> "write a test that reproduces it, then make it pass"). For multi-step work, state a brief plan with a `verify:` check per step, then loop until verified.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites from overcomplication, and clarifying questions come before implementation rather than after mistakes.
