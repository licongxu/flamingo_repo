# Stable Halo Join Catalogue Rebuild Design

## Objective

Rebuild every local L1_m9 feedback-variant and L2p8_m9 lightcone catalogue
from the official FLAMINGO halo-lightcone and SOAP-HBT products without using
the positional `InputHalos/SOAPIndex` join. Preserve every canonical catalogue
path and filename. Archive the current CSVs before publishing replacements.

This phase ends when all catalogue CSVs are correct. Masked power spectra,
bandpowers, covariance products, paper plots, and other non-CSV downstream
products are explicitly deferred.

## Evidence and Root Cause

The halo-lightcone `InputHalos/SOAPIndex` is an array row number rather than a
stable halo identity. Across snapshots 50, 69, and 75, deterministic samples
from all nine L1 variants match the current SOAP `HaloCatalogueIndex` at only
3--22 rows out of 101. The same test gives 101/101 for snapshots 50, 70, and
76 in all eight L2p8 lightcones.

An object-level L1 trace established that the lightcone
`InputHalos/HaloCatalogueIndex` matches the halo centre and particle/map
signal, while the released `SOAPIndex` points to a different current SOAP
halo. The repair therefore joins on `InputHalos/HaloCatalogueIndex`. L2p8 is
rebuilt through the same identity-safe path for consistency and to eliminate
future reliance on the currently valid row index.

## Scope

The catalogue matrix contains 68 canonical CSVs:

- Nine L1 lightcone-0 variants: `L1_m9`, `fgas+2sigma`, `fgas-2sigma`,
  `fgas-4sigma`, `fgas-8sigma`, `Mstar-1sigma`,
  `Mstar-1sigma_fgas-4sigma`, `Jet`, and `Jet_fgas-4sigma`.
- Eight L2p8 fiducial lightcones: `lightcone0` through `lightcone7`.
- Four CSV flavours for every variant or lightcone: the
  `M500c_1e13_zlt3` base catalogue, `M500c_5e13_zlt3_qfrommz`,
  `M500c_5e13_zlt3_qfrommz_alpha_fixed_1p12`, and
  `M500c_5e13_zlt3_qfrommap`.

The published Compton-y maps are inputs and are not modified. Existing
non-CSV products in `data_paper/` and `figures/` are not modified in this
phase.

## Canonical Data Model

For each snapshot, variant, and lightcone:

1. Read the official halo-lightcone `InputHalos/HaloCatalogueIndex`,
   `Lightcone/HaloCentre`, and `Lightcone/Redshift`.
2. Read current SOAP `InputHalos/HaloCatalogueIndex` and build a unique
   identity-to-row lookup.
3. Resolve every lightcone entry to its current SOAP row by stable identity.
   Periodic lightcone copies are retained as separate observed entries.
4. Read all current SOAP quantities used by the existing family-specific
   schema: M500c, M200c, M200m, and their radii for every catalogue; also read
   the four integrated Compton-Y fields present in the L1 catalogue family.
5. Select central halos with physical `M_500c >= 1e13 M_sun` and
   `0 <= z < 3`. No `M_sun/h` convention is introduced.
6. Compute natural and yang26-rotated geometry with the official per-shell
   rotations for that box and lightcone. The rotation convention must match
   the corresponding map builder exactly.
7. Store the resolved current SOAP row in `soap_index` for provenance, but
   never use that row number as the join key.

The base output preserves the existing family-specific names and order so
readers do not require migration: 29 columns for L1 (including the four SOAP
integrated-Y fields) and 25 columns for L2p8. Provenance comments state the
stable identity join and official sources.

## Derived Catalogue Flavours

Each 5e13 catalogue is derived from the newly rebuilt base catalogue rather
than from an archived product.

- Apply physical `M_500c > 5e13 M_sun` selection.
- Recompute `q_from_mz` with the existing calibrated convention
  `A_SZ=-4.0953238`, `alpha_SZ=1.12`, `B=1.41`, D3A cosmology, current
  M500c mass definition, and the existing deterministic scatter convention.
- Publish both existing q-from-mz filenames and preserve their schemas (30
  columns for L1 and 26 for L2p8).
- Recompute q-from-map from the corresponding Compton-y map, the current
  cylindrical R500 aperture prescription, and the existing averaged noise
  curve, with no background subtraction. Preserve the current family-specific
  schema (34 columns for L1 and 30 for L2p8) and the
  `q_from_aperture` name.

## Staging, Archive, and Publication

New files are built below a hidden staging tree on `/rds`, which has about
7.1 TiB free. No canonical CSV is moved until its staged replacement passes
validation.

The archive is a timestamped tree below
`/rds/rds-lxu/flamingo/archive/hbt_join_fix_20260731/` that mirrors every
canonical relative path. Publication is per file:

1. Verify the staged file and record its size, row count, schema, and digest in
   a manifest.
2. Rename the canonical old file into its archive path on the same filesystem.
3. Rename the staged file to the exact canonical old path.
4. If step 3 fails, immediately rename the archived file back to the canonical
   path.

No archived file is overwritten. Re-running the migration skips published
files only when the manifest and on-disk validation both confirm completion.

## Validation and Failure Handling

Validation is performed before every publication and again after publication:

- Exact expected column names and order for each flavour.
- Finite values, physical mass/radius units, redshift range, and mass cut.
- Every output `(snapshot, soap_index, HaloCatalogueIndex)` association agrees
  with the current SOAP identity lookup.
- Every natural position and redshift agrees with its official lightcone row.
- Shell index and rotated coordinates reproduce the map-builder convention.
- No duplicate SOAP identity exists in a snapshot lookup; periodic lightcone
  copies are allowed and explicitly counted.
- q-from-map and q-from-mz outputs have exactly the same base row identities as
  their corresponding 5e13 selection.
- For L2p8, the identity-safe rebuild must agree with the archived catalogue
  on identity-keyed physical quantities, apart from serialization precision
  and explicitly recomputed derived columns.
- For L1, known false massive objects must be absent and sampled map cutouts
  must centre on the corrected coordinates.

A failed snapshot, file, or validation leaves its canonical file untouched.
The manifest records failure details, elapsed time, and resource use so work
can resume without guessing.

## Completion Criteria

The catalogue phase is complete only when all 68 canonical paths exist under
their original names, all 68 archived predecessors exist, every validation is
green, no temporary publication gap remains, and the manifest reports a
successful stable-identity join for every run and lightcone.
