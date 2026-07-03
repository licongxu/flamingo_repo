# Paper plan

**Working title:** *Cosmological Constraints from a Joint Analysis of the tSZ Power Spectrum and Cluster Number Counts with Massive-Cluster Masking*

**Type:** Theory paper — develop and demonstrate a joint analysis method on simulated data.

**Reference:** An existing draft on the same topic (attached PDF) covers related formalism and synthetic tests; this plan follows a simpler two-stage structure and does not mirror that paper section-by-section.

**Code:** `tszsbi` (synthetic mocks) + this repo (`flamingo_repo`, hmfast, Cobaya) for FLAMINGO.

---

## What the paper does

Combine **cluster number counts (CNC)** and the **masked tSZ power spectrum** in one self-consistent analysis:

- Clusters detected above a significance cut are **removed from the map** (masked out of the PS).
- The same detected clusters enter the **CNC** sample.
- The two probes therefore see **different halo populations** — no double-counting.

The paper demonstrates that this joint pipeline works, first on controlled synthetic data, then on FLAMINGO hydrodynamical simulations.

---

## Paper structure (two parts, one paper)

```
Introduction & method
        ↓
Part 1 — Synthetic mocks (painted maps + mock catalogues)
        ↓
Part 2 — FLAMINGO simulations (this repo)
        ↓
Discussion & conclusions
```

---

## Introduction and method (shared setup)

Brief motivation: CNC and tSZ PS are complementary; masking resolved clusters from the PS while counting them in CNC is the natural way to combine them self-consistently.

**Method summary (keep concise in the paper):**

- **CNC:** binned counts N(z, q) with a Poisson likelihood.
- **Masked tSZ PS:** bandpowers D_l after masking detected clusters; Gaussian likelihood on high-ℓ bins.
- **Joint likelihood:** sum of the two (no cross-term if populations are disjoint).
- **Forward model:** halo model with a cluster pressure profile, SNR selection, and (where needed) intrinsic scatter in the Compton-y signal.
- **Masking:** match `q_cat` in the map mask to the catalogue detection threshold.

Key equations can live in a Methods section or an appendix; the main text should focus on the demonstration, not a full formalism review.

---

## Part 1 — Synthetic mocks (painted maps + mock catalogues)

**Goal:** Show the method works when we control the data generation.

### Setup

- Draw mock halo catalogues (mass, redshift, scattered Compton-y).
- Paint cluster profiles onto full-sky Compton-y maps.
- Add survey noise; build a cluster catalogue with significance q.
- Mask detected clusters from the map; measure masked PS bandpowers.
- Run joint inference (CNC + masked PS) with Cobaya / `tszsbi`.

### Case A — No foregrounds

**Expected message:** the pipeline works.

- Map, catalogue, and theory agree for CNC and masked PS.
- Joint fit recovers input cosmology and cluster-scaling parameters better than either probe alone.
- Masked PS is suitable for a Gaussian likelihood (unlike full-sky PS at low ℓ, where massive clusters drive non-Gaussian scatter).

**Figures to include:**

- Schematic of the split: detected clusters → CNC; undetected background → masked PS.
- Example painted map before/after masking.
- N(z, q) data vs theory.
- Masked D_l data vs theory (optionally compare to full-sky).
- MCMC contours: CNC-only vs masked-PS-only vs **CNC + masked PS**.

### Case B — With foregrounds

**Expected message:** foregrounds contaminate the PS; the joint analysis fails unless FG is modelled and constrained.

- Add foreground templates to the painted maps (CIB, IR, radio, etc.).
- **Without FG nuisance parameters:** joint fit is biased / poor fit on the PS side.
- **With FG amplitudes as free parameters (and priors):** constraints recover; CNC anchors cosmology while the PS leg marginalises over FG.

**Figures to include:**

- Same contour layout as Case A, with and without FG marginalisation.
- Residual PS or χ² panel showing FG contamination.
- Optional: degeneracy between tSZ amplitude and FG templates (Fisher or MCMC).

### Part 1 takeaway

The method is validated on synthetic data. No FG → good. With FG → must constrain FG for the joint analysis to be trustworthy.

---

## Part 2 — FLAMINGO simulations (this repo)

**Goal:** Apply the same joint pipeline to realistic hydrodynamical lightcones where the forward model is approximate but the simulation is physically motivated.

### Setup

- **Simulation:** FLAMINGO L2p8_m9 (primary); optionally L1_m9 feedback variants for systematics.
- **Catalogues:** SOAP halos with SNR q from a forward model (GNFW, noise curve).
- **Maps:** Compton-y lightcone maps; mask detected clusters; NaMaster bandpowers.
- **Likelihood:** same CNC + masked PS joint setup as Part 1 (Cobaya configs in `cobaya/configs/cnc_yy_combined_*`).
- **No astrophysical foregrounds** in the sim maps — this section is signal-only unless we optionally inject FG for comparison.

### What to show

1. **Forward-model check**
   - FLAMINGO N(z, q) vs theory (`nb32`, `nb40`).
   - Masked D_l vs hmfast theory with matching completeness / masking (`nb06`, `nb14`, `nb35`).

2. **Joint inference on L2p8_m9**
   - Compare: CNC-only, masked PS-only, full-sky PS-only, CNC + masked PS, CNC + full-sky PS.
   - Main result: does CNC + masked PS tighten constraints relative to CNC alone, as in Part 1?
   - Does CNC + full-sky fail to improve over CNC-only (showing masking matters)?

3. **Physics systematics (optional but valuable)**
   - L1_m9 feedback variants: how do jet / f_gas / M_* prescriptions shift CNC vs masked PS differently? (`nb37`, `nb40`).

4. **Bridge back to Part 1**
   - Same figure layout and probe combinations as synthetic section — qualitative comparison of what carries over and what breaks (e.g. profile mismatch, hydrostatic bias).

### Repo assets

| Item | Location |
|---|---|
| Masked PS pipeline | nb06, nb14; `scripts/compute_masked_tsz_ps_*` |
| CNC binning | `data/cnc/`, nb09, nb32 |
| Bandpowers + cov | `data/bandpowers_arnaudB1_Y500c/`, `data/theory_cov_arnaudB1/` |
| Joint Cobaya config (masked) | `cobaya/configs/cnc_yy_combined_arnaudB1_Y500c.yaml` |
| Existing chains (CNC, full-sky combined) | `chains/cnc_cosmo_*`, nb27, nb28 |
| Feedback variants | nb35, nb37, nb40 |

### Part 2 takeaway

The method developed and validated on synthetic mocks is applied to FLAMINGO. Report agreement with theory, joint constraining power, and where the parametric model disagrees with hydrodynamical reality.

---

## Suggested section outline

| Section | Content |
|---|---|
| 1. Introduction | CNC + masked tSZ motivation; masking avoids double-counting; two-stage validation plan |
| 2. Method | Joint likelihood; CNC and masked PS observables; masking and selection; forward model (brief) |
| 3. Synthetic data | Mock catalogue + painted map pipeline |
| 4. Results: synthetic, no FG | Validation; joint vs single probes |
| 5. Results: synthetic, with FG | Failure without FG model; recovery with FG constraints |
| 6. FLAMINGO data | Simulations, catalogues, maps, bandpowers |
| 7. Results: FLAMINGO | Theory comparison; joint inference; optional feedback systematics |
| 8. Discussion | Synthetic lessons; FLAMINGO limitations; outlook (real survey data) |
| 9. Conclusions | |
| Appendix | Profile details, NaMaster, Cobaya configs |

---

## Figure list (draft)

| # | Part | Figure |
|---|---|---|
| 1 | — | Method cartoon: mask detected clusters → CNC + masked PS |
| 2 | 1 | Painted map and mask |
| 3 | 1 | N(z, q): mock vs theory |
| 4 | 1 | D_l: masked (and optionally full-sky) mock vs theory |
| 5 | 1 | MCMC: no FG — single probes vs joint |
| 6 | 1 | MCMC: with FG — without vs with FG marginalisation |
| 7 | 2 | FLAMINGO N(z, q) vs theory |
| 8 | 2 | FLAMINGO masked D_l vs theory |
| 9 | 2 | MCMC on FLAMINGO: probe comparison |
| 10 | 2 | (Optional) L1_m9 feedback systematics |

---

## Work remaining

### Part 1 (synthetic)
- [ ] Confirm halo-painting + joint chains cover both no-FG and with-FG cases
- [ ] Finalise Part 1 figures (contours, residuals)

### Part 2 (FLAMINGO — this repo)
- [ ] Run / complete CNC + masked PS combined MCMC (`cnc_yy_combined_arnaudB1_Y500c.yaml`)
- [ ] FLAMINGO joint contour figure (same layout as synthetic)
- [ ] Theory-validation plots from existing notebooks
- [ ] (Optional) L1_m9 feedback panel

### Writing
- [ ] Draft Sections 1–2 (intro + method)
- [ ] Draft Sections 3–5 from synthetic results
- [ ] Draft Sections 6–7 from FLAMINGO work
- [ ] Discussion linking FG lesson (Part 1) to FLAMINGO profile/systematic issues (Part 2)

---

## Open choices

1. Detection threshold and hydrostatic bias convention (B = 1 vs B = 1.35) for FLAMINGO section.
2. How much formalism in the main text vs appendix (PDF reference has more; this paper can stay leaner).
3. Whether to include full-sky PS + CNC as a negative control in both parts.
4. Depth of L1_m9 feedback section (one figure vs full subsection).

---

## One-paragraph summary

We propose a theory paper that combines cluster number counts and masked tSZ power spectra by masking detected clusters from the map and counting them in CNC, ensuring self-consistency. We first demonstrate the method on synthetic mock catalogues and painted Compton-y maps: without foregrounds the joint analysis works well; with foregrounds it fails unless foreground amplitudes are constrained. We then apply the same pipeline to FLAMINGO hydrodynamical simulations to test the forward model and joint constraining power in a realistic ICM setting.
