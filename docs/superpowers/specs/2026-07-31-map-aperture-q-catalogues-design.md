# Map-Aperture q Catalogues Design

**Date:** 2026-07-31

## Goal

Build a repository-native pipeline that measures a raw cylindrical Compton-y
aperture signal for every halo in the reduced FLAMINGO catalogues and writes a
new selection catalogue whose signal-to-noise ratio comes from the map rather
than the parametric mass-redshift relation.

The production scope is exactly:

- the nine `L1_m9` lightcone-0 feedback variants under
  `/rds/rds-lxu/flamingo/L1_m9`;
- lightcones 0 through 7 under `/rds/rds-lxu/flamingo/L2p8_m9`;
- the canonical `M_500c > 5e13 M_sun`, `z < 3` input catalogue ending
  `_yang26rot_qfrommz.csv` for each map;
- one sibling output per input ending `_yang26rot_qfrommap.csv`.

The `_qfrommz_alpha_fixed_1p12.csv` derivatives and the much larger
`M_500c > 1e13 M_sun` catalogues are outside this pipeline.

## Observable Definition

For each catalogue row, compute the true angular radius

\[
\theta_{500}=R_{500c}/D_A(z)
\]

with `flamingo.catalogue.theta_500` and the repository's FLAMINGO D3A
cosmology. Catalogue masses and radii use physical `M_sun` and Mpc units.

At the yang26-rotated catalogue coordinates, select every HEALPix RING pixel
whose centre lies within `theta_500`. The raw map aperture signal is

\[
Y_{500}^{\rm cyl}=\Omega_{\rm pix}\sum_{i\in\theta_{500}}y_i,
\]

with no local or global background subtraction. The sum is accumulated in
float64 and stored in arcmin squared. An aperture containing no pixel centre
has `npix_in_aperture = 0` and `Y_500cyl_arcmin2 = 0`.

The aperture calculation uses the map as stored. It does not deconvolve a
beam, compensate the HEALPix pixel window, or assign fractional boundary
pixels.

## Noise and Signal-to-Noise

Use the converted tile curves in
`data/noise/sigma_Y500_dict_szifi.npy` together with
`data/noise/skyfracs_szifi_cosmology.npy` and the `immf6` filter.

Match `hmfast.tracers.tsz_completeness.load_sigma_y0_curve` exactly:

1. reconstruct the 25-point logarithmic theta grid over 0.5 to 32 arcmin;
2. compute the sky-fraction-weighted arithmetic mean of the per-tile standard
   deviations;
3. fit a degree-three polynomial to
   `log(sigma_Y500_arcmin2)` as a function of `log(theta_500_arcmin)`;
4. evaluate the polynomial without clipping theta to the fit interval.

The empirical significance is

\[
q_{\rm from\,aperture}=Y_{500}^{\rm cyl}/
\sigma_{Y_{500}}(\theta_{500}).
\]

Negative raw aperture values are retained and therefore may produce negative
`q_from_aperture`. All computed values must be finite.

## Output Schema and Provenance

Preserve the source row order and every source column except `q_from_mz`.
Append these columns in this order:

1. `theta_500_arcmin`
2. `Y_500cyl_arcmin2`
3. `sigma_Y500_arcmin2`
4. `npix_in_aperture`
5. `q_from_aperture`

The output leading comments record the source catalogue, source map, raw
no-background aperture convention, HEALPix pixel-centre rule, `immf6` noise
files, theta fit range, polynomial degree, and output units.

Outputs are siblings of their inputs. For example:

```text
halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommz.csv
-> halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommap.csv
```

## Architecture

### Reusable library

Add a focused module under `src/flamingo/` containing pure or narrowly scoped
functions for:

- loading and fitting the sky-averaged `sigma_Y500` curve;
- evaluating the fitted noise curve;
- summing raw map pixels in per-object apertures;
- transforming one dataframe chunk into the requested output columns.

The functions accept paths, arrays, or an already-loaded map explicitly so
unit tests can use small synthetic inputs without accessing RDS data.

### Production command

Add one script under `scripts/` that owns dataset discovery, map loading,
streamed CSV I/O, atomic output installation, summaries, and command-line
selection. It supports:

- `--dry-run` to list source/map/output triples without writing;
- repeatable `--only` substring filters;
- `--dataset {all,l1,l2}`;
- `--force` to replace an existing `_qfrommap.csv` atomically;
- a configurable chunk size for testing and operational tuning.

The default invocation processes all 17 canonical catalogues. Each map is
loaded as float32 and each CSV is streamed. A unique temporary sibling is
written first. The final output is installed with `os.replace` only after the
entire input has succeeded and the source file's size and modification time
are unchanged. On failure, the temporary file is removed and any existing
output is preserved.

The command prints row counts, elapsed time, zero-pixel aperture count, and
cumulative counts above `q > 1, 5, 10, 20, 50` for every output.

## Dataset Discovery

For L1, discover only files matching
`halo_catalogue_M500c_5e13_zlt3_*_yang26rot_qfrommz.csv`. Derive the variant
between `zlt3_` and `_yang26rot_qfrommz.csv`, and pair it with
`maps/y_unlensed_<variant>_lc0_nside4096.fits`.

For L2, discover sorted `lightcone*/catalogues` directories and require one
canonical L2 input per directory. Extract the integer lightcone index and pair
it with
`lightconeN/healpix_map/y_unlensed_L2p8_m9_lcN.fits`.

Discovery rejects missing maps, duplicate canonical inputs, malformed
lightcone directory names, or an empty selected dataset before any production
write begins.

## Error Handling

Before processing a catalogue, require the columns `z`, `R_500c_Mpc`,
`theta_rot_rad`, `phi_rot_rad`, and `q_from_mz`. Reject an empty catalogue,
non-finite required inputs, non-positive angular radii, non-finite fitted
noise, non-positive fitted noise, shape mismatches, and non-finite output
values.

Without `--force`, an existing output is an error rather than an implicit
skip. This makes incomplete production runs visible. With `--force`, the old
output remains in place until its complete replacement is ready.

## Testing and Verification

Unit tests use a low-resolution synthetic HEALPix map, a small commented CSV,
and tiny tile-noise dictionaries to verify:

- pixel-centre aperture sums, including a zero-pixel aperture;
- sky-fraction averaging and the cubic log-log fit;
- exact output schema, source-row order, and removal of `q_from_mz`;
- `q_from_aperture = Y_500cyl_arcmin2 / sigma_Y500_arcmin2`;
- canonical L1 and L2 discovery and map pairing;
- refusal to overwrite without `--force`;
- atomic cleanup when chunk processing fails.

An integration regression processes a small prefix of the fiducial L1
catalogue and compares its empirical columns against a direct calculation.
Before the full production run, the fiducial L1 result must reproduce the
previously measured cumulative counts:

```text
q > 50:      3
q > 20:     38
q > 10:    271
q >  5:  1,336
q >  1: 25,402
```

Final verification streams every one of the 17 source/output pairs and proves:

- every requested output exists;
- source and output row counts match;
- output columns match the contract;
- all empirical columns are finite;
- the output identity and row order match the source `snap` and `soap_index`;
- no temporary files remain.

## Non-goals

- No background subtraction or compensated aperture photometry.
- No fractional-pixel integration.
- No beam or pixel-window correction.
- No stochastic measurement noise realization.
- No modification or deletion of any source catalogue.
- No processing of `_alpha_fixed_1p12` or `M_500c > 1e13 M_sun` catalogues.
