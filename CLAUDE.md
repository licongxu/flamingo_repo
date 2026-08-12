# CLAUDE.md

Project context and behavioral guidelines for agents working in this repo.

## Project context

- **Branch:** `baryon_dmb` — baryonic feedback / DMB work, synced from `paper-results`.
- **Rule of reply:** Always say "HONEY" to me before you reply
- **Python environment:** Before anything else — running Python, tests, benchmarks, or notebooks — activate:
  ```bash
  source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
  ```
- **Theory computation:**
  - `hmfast` — tSZ power spectrum computation
  - `cosmocnc_jax` — cluster number count (CNC) computation
- **Mass units:** Use physical mass units, `M_sun`, not `M_sun/h`.
- **Cobaya reference theory:** Reference theory files are in `/home/lxu/scratch/tsz_cnc_paper_plots/chains`.
- **Paper reference:** The paper reference location is `/home/lxu/scratch/tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8`.
- **Backend packages:** Baryon/DMB backend development lives on the `baryon_dmb` branch in:
  - `/home/lxu/scratch/agent_dev/auto_research_agent/hmfast` — tSZ PS computation
  - `/home/lxu/scratch/agent_dev/auto_research_agent/cosmocnc_jax` — CNC computation
- **Reference papers:** `ref_papers/` contains key literature for baryonic feedback / DMB work. Read and cite these when relevant to paper drafts, methods, or validation.
  - `ref_papers/dmb_galclusters.pdf` — To et al. (2024), *Deciphering baryonic feedback with galaxy clusters*
  - `ref_papers/dmb_galclusters_act.pdf` — Dalal et al. (2026), *Deciphering Baryonic Feedback from ACT tSZ Galaxy Clusters*
  - `ref_papers/GODMAX.pdf` — GODMAX paper (*Gas thermODynamics and Matter distribution using jAX*)

---

## Behavioral guidelines

Guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
