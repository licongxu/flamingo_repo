# Checkpoint — M_500c selection fix and full regeneration (2026-07-30)

Recorded mid-flight so the work can resume on a different cluster. **Nothing
here is running any more**; the interrupted compute has been stopped and its
half-written products quarantined (see "Quarantined" below).

---

## 1. The bug

`flamingo.cnc.SZScaling.calibrated()` built its halo model as
`HaloModel(cosmology=cosmology)`, taking hmfast's **default mass definition,
M_200c**. But the catalogue masses are physical **M_500c**, and
`compute_y0_parametric` / `compute_theta500_arcmin` convert from
`halo_model.mass_definition` to M_500c internally. That conversion must be a
no-op; instead it re-converted M_500c as if it were M_200c:

| quantity | factor |
|---|---|
| mass | x0.67 |
| `y0` (∝ M^1.12) | x0.64 |
| `q = y0 / sigma_y0` | **x0.46** |

So every `q_from_mz` was ~0.46x too low and **every mask was cut at the wrong
threshold**: the mask labelled `q>5` actually contained only true-`q>11`
clusters (732 of them, where a true q>5 cut has 2780).

### Three independent confirmations of the fix

1. The A10-anchored calibration now returns `A_SZ = -4.237656`, matching the
   reference chains' prior centre `-4.237656338145`
   (`/home/lxu/scratch/tsz_cnc_paper_plots/chains/case0{1,2}_*_signal_only_real1`)
   to six decimals. Before the fix it returned `-4.051178`.
2. Regenerated catalogues reproduce `q` from their own `(M_500c, z, soap_index)`
   to `max|dq/q| ~ 1e-14`.
3. The corrected cluster counts predicted from the catalogue matched the counts
   the NaMaster masking step independently reported, exactly, for all five cuts.

### Corrected cluster counts, L1_m9 fiducial

| cut | old N | new N | old f_sky_eff | new f_sky_eff |
|---|---:|---:|---:|---:|
| q>50 | 4 | **14** | 0.9977 | 0.9956 |
| q>20 | 32 | **134** | 0.9930 | 0.9840 |
| q>10 | 158 | **663** | 0.9819 | 0.9550 |
| q>5 | 732 | **2780** | 0.9493 | 0.8869 |
| q>1 | 13475 | **38329** | 0.6893 | **0.4353** |

---

## 2. Code changes (committed, travel with this branch)

* `src/flamingo/cnc.py` — declare `MassDefinition(500, "critical")`,
  `convert_masses=True`, `hm_consistency=False`; `B_HYDROSTATIC` 1.35 -> **1.41**
  so selection and pressure profile share one mass calibration. No `B=1.35`
  remains on this branch.
* `scripts/regenerate_q_catalogues.py` — **new**. Rewrites `q_from_mz` in every
  `*_qfrommz*.csv` in place (atomic, keeps `.pre_massdef_fix.bak`, skips files
  already carrying the fix marker, `--force` to redo). One selection convention
  everywhere: `A_SZ=-4.0953238`, `alpha_SZ=1.12`, `B=1.41`, `sigma_lnY=0.173`.
* `src/flamingo/inference/masked_ps.py` — **new**. Cobaya theory + likelihood for
  masked tSZ bandpowers with **sampled cosmology and both halo terms**. The 1h
  term uses the `n_power=2` conditional moment, the 2h term `n_power=1` (its two
  brackets are linear and sourced by distinct halos). Validated: reproduces the
  stored covariance-script theory to ratio **1.0000** for full-sky and q>5.
  Benchmarks at **22 ms** per evaluation.
* `scripts/run_masked_ps_chains.py` — **new**. Signal-only chains for
  `fullsky, qgt50, qgt20, qgt10, qgt5`, priors copied from the reference
  case01/case02 runs with `A_SZ` recentred on the L1_m9 full-sky best fit.
* `scripts/compute_l2p8_m9_masked_ps_alpha_fixed_1p12.py` — generalised from
  lightcone-0-only to **all 8 lightcones**; outputs are now per lightcone,
  `Dl_yy_L2p8_m9_lc{i}_masked_...`. Paper figures use lc0.
* `scripts/create_l1_m9_bestfit_q_catalogues.py` — **deleted** (superseded; its
  `_qfrommz_bestfit.csv` output was consumed by nothing).
* `paper_results/compute_q.py` — stale `B=1.35` note corrected.

---

## 3. DONE

* **All 34 `q` catalogues regenerated and validated** — 9 L1_m9 feedback variants
  x 2 flavours, plus 8 L2p8_m9 lightcones x 2 flavours, under
  `/rds/rds-lxu/flamingo/{L1_m9,L2p8_m9/lightcone*}/catalogues/`.
  Originals kept as `*.pre_massdef_fix.bak` (one backup, for
  `Jet_..._alpha_fixed_1p12`, was truncated by the crash and deleted; that file
  itself validated clean).
* **9 dead `*_qfrommz_bestfit.csv` deleted** (5.7 GB, consumed by nothing).
* **L1_m9 fiducial masked bandpowers: 4 of 5 cuts** regenerated with corrected q —
  `qgt50, qgt20, qgt10, qgt5`, both the 18-bin and 12-log-bin files.

> **These live on this cluster's `/rds` and are NOT in git.** On a new cluster
> either copy `/rds/rds-lxu/flamingo/*/catalogues/*_qfrommz*.csv` across, or
> re-run `scripts/regenerate_q_catalogues.py` (~45 min for all 34).

---

## 4. Quarantined (stale, renamed `*.STALE_pre_massdef_fix`)

In `data_paper/binned_bandpowers/`, because they sat next to freshly regenerated
siblings and would otherwise read as consistent:

`Dl_yy_L1_m9_masked_qgt{1,3,6}_qfrommz_alpha_fixed_1p12_{binned_18,logbins_dln0p4_lmax10000}.txt`,
`L1_m9_masked_qfrommz_alpha_fixed_1p12_metadata.json`,
`L1_m9_masked_qfrommz_alpha_fixed_1p12.npz`.

Delete them once regenerated.

---

## 5. TODO on the new cluster

### Do this FIRST — the chains are already unblocked (~45 min)

The cobaya goal (total tSZ + q>50, 20, 10, 5 on the L1_m9 fiducial 18-bin
bandpowers) does **not** depend on the rest of the sweep. Every dataset it needs
already carries the corrected selection:

* `Dl_yy_L1_m9_fullsky_binned_18.txt` — full sky, never affected by the bug;
* `Dl_yy_L1_m9_masked_qgt{50,20,10,5}_qfrommz_alpha_fixed_1p12_binned_18.txt` —
  regenerated 2026-07-30 19:53-20:08.

The cut that was interrupted is `q>1`, which the goal does not use. So:

```bash
python scripts/compute_l1_m9_customgnfw_bestfit_covariance.py   # ~1 min, new f_sky
python scripts/run_masked_ps_chains.py                          # 5 chains, ~25 min
# then getdist triangle plots
```

Only copy the `/rds` catalogues (or re-run `regenerate_q_catalogues.py`) if the
masked bandpowers above are not carried over with them. Everything in the table
below can follow afterwards.

### Then the rest of the sweep

Everything below still carries the **old, wrong** selection. Full-sky products
are unaffected and must NOT be regenerated.

| # | job | scale | est. serial |
|---|---|---|---|
| 1 | L1_m9 fiducial `q>1` + metadata/npz (finish the interrupted run) | 1 cut | 10 min |
| 2 | `compute_masked_ps_qgt6` | 1 cut | 8 min |
| 3 | `compute_l1_m9_feedback_bandpowers` (q>5) | 9 variants | 2.2 h |
| 4 | `compute_l1_m9_feedback_bandpowers_qgt1` | 9 variants | 1.5 h |
| 5 | `compute_l1_m9_feedback_ratio_vs_q_bandpowers` (q=50,20,10,5,3,1) | 54 runs | 5.5 h |
| 6 | `compute_l2p8_m9_masked_ps_alpha_fixed_1p12` (**all 8 lightcones**) | 40 runs | 4.1 h |
| 7 | `compute_l1_m9_customgnfw_bestfit_covariance` (needs new `f_sky_eff`) | — | 1 min |
| 8 | all `plot_*.py` + `paper_results/figures.py` | — | 15 min |
| 9 | `run_masked_ps_chains.py` (5 chains) + getdist | — | 25 min |
| 10 | `masking_radius_null_test` + random control + its plot — **run last** | 80 runs | 6.7 h |

~20 h serial; ~6-8 h with 6 concurrent jobs at `OMP_NUM_THREADS=16`.

**Both binnings** (`_binned_18` and `_logbins_dln0p4_lmax10000`) are emitted as a
pair by every writer — no extra step needed.

### Resume commands

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
export FLAMINGO_ROOT=<repo>            # products go to $FLAMINGO_ROOT/data_paper
export PYTHONPATH=<repo>/src
export OMP_NUM_THREADS=16              # x6 concurrent jobs keeps a shared box healthy

python scripts/regenerate_q_catalogues.py          # only if /rds catalogues not copied
python scripts/compute_l1_m9_masked_ps_alpha_fixed_1p12.py
python scripts/compute_l2p8_m9_masked_ps_alpha_fixed_1p12.py --lightcone 0   # ... 1..7
python scripts/compute_l1_m9_customgnfw_bestfit_covariance.py
python scripts/run_masked_ps_chains.py
```

---

## 6. Open question, not yet resolved

Before the fix, the masked theory over-suppressed relative to the masked data
even when the theory's selection was forced to match the (buggy) selection the
masks actually used — theory/data ~0.52 at q>5, ~0.22 in the lowest bin, against
a full-sky theory/data of ~0.84. The selection fix should remove most of this,
since it was largely an artefact of comparing corrected theory to uncorrected
data, **but this has not been re-tested** — the regenerated masked bandpowers
were not yet complete when work stopped.

**First thing to check after job 1 above:** recompute theory vs the regenerated
masked bandpowers per q cut and confirm the ratio is consistent with the full-sky
case. If a residual remains, candidates are the finite mask radius
(`max(4*theta_500, 20')`) versus whole-halo removal in the theory, and NaMaster
MASTER deconvolution on a mask that is correlated with the signal.
