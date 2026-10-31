# Enstatite external-data workflow

MgSiO3 enstatite is the first exposed measured development case for the proposed
raw-images-to-PhaseSmith-structure-search workflow. The complete workflow has
not yet passed. See the [promotion audit](single-crystal-promotion.md).

## Storage

Keep all raw/reference files outside the repository. Commit only
`validation/enstatite-mgsio3-local-v1.json`, code, protocol and small numerical
summaries. The manifest pins eight files by exact byte size and SHA-256, totalling
470,052,794 bytes; no local absolute path is part of their identity. Verification
streams files rather than allocating the entire HDF5 file in memory.

`validation/data/` is already ignored. Users can also keep the original directory
anywhere and pass it explicitly. Git LFS is unnecessary for this design.
Redistribution rights and a public download location are not established, so
there is no automatic downloader or upload. A future authorized data deposit can
add a pinned HTTPS record without changing the scientific input identity.

Verify the original folder with the standard library only:

```sh
python tools/verify_single_crystal_data.py --data-root /path/to/enstatite-files
```

Successful verification says only that the registered input bytes match. It
does not run or pass a scientific workflow. Files remain read-only and no
dataset cache, copied raw file or reference model is added to Git.

For routine CI use synthetic count fixtures and, when redistribution is settled,
small deterministic real shoeboxes/compact observation tables with their parent
hashes. A crop exercises a stage; it does not replace the full-scan gate.
Run the full raw-image gate explicitly on a machine with the registered data,
either manually or in a separately configured dataset-enabled CI job.

## Current PhaseSmith search probe

The benchmark `benchmarks/enstatite_structure_search.py` tests PhaseSmith search
on existing intensities, with the frozen corpus search configuration. It does
not reread/reduce the raw frames. It needs the existing feature-enabled native
package, NumPy and Gemmi (the established corpus resource preflight uses Gemmi).
It never invokes SHELXT or SHELXL.

Run the registered source-integration control:

```sh
python benchmarks/enstatite_structure_search.py \
  --data-root /path/to/enstatite-files \
  --output runs/enstatite-source-search
```

Or test the already reduced PhaseSmith intensities from the archived raw-count
rerun, using the exact NPZ registered in the manifest:

```sh
python benchmarks/enstatite_structure_search.py \
  --data-root /path/to/enstatite-files \
  --observations /path/to/ours-own-sigma-observations.npz \
  --output runs/enstatite-phasesmith-search
```

The output directory must be new. The source control splits its appended
SHELX instructions from the HKL reflection block; the original is unchanged.
The PhaseSmith NPZ path uses only `hkl`, `intensity` and `sigma`, retaining row
order, signs and full floating-point text precision. Search receives a separate
input packet containing the supplied cell, group number, element types and
input-derived resource cap; no reference coordinates, ADPs or atom counts.
The worker saves a candidate/result before the parent assesses capacity against
the reference count. A process error or an 18-second outer timeout returns exit
status 2; completed searches return 0 even if no candidate or target is reached.
Inspect the report's separate execution, search and assessment fields.

The current 2,000,000-element dense bound caps both inputs at 36 atoms for
10,694 expanded measured rows. The supplied reference has 80. These searches
cannot establish full structure recovery, irrespective of R1. The benchmark
keeps that limit, four starts, 800 cycles, seed 0, a 13-second native budget and
one worker unchanged. No repeated seed/configuration selection is performed.

On 31 October 2026 one run per input completed without a candidate. Both
reported `search_budget` and missed the numerical target, with no process
timeout. The small recorded comparison is
`benchmarks/enstatite-promotion-search-macos.json`. This is a retained failing
development result, not a passed complete-workflow test.

## Complete runner to implement

1. Verify source hashes and write a run manifest with source/native/configuration
   hashes, explicit fixed priors and a fresh output directory.
2. Decode the original HDF5 array through a small optional example loader.
   Record zero-based source-frame mapping, image orientation, exposure angles,
   gain/noise and masks. The historical use of frames 2--143 and vertical flip
   is an explicit starting convention to review, not an inferred universal rule.
3. Detect positions and discover/refine candidate lattices without the archived
   peak-search table or reference coordinates. Retain competing diamond-anvil
   domains and unassigned positions. If supplied calibration/cell is used, label
   that calibrated/prior-assisted mode; keep it separate from discovery claims.
4. Predict and integrate selected domain crossings from raw counts. Retain signed
   aperture and profile measurements separately, flags, coverage and conditional
   errors. Train profile/sensor/scale models only on declared complete-family
   partitions. Fix weak estimators; never tune against structural R1 or ADPs.
5. Apply only declared detector/polarization/scaling assumptions and bridge the
   resulting corrected signed observations into `HklData`. Preserve provenance,
   equivalence operations and the observation/error policy.
6. Run PhaseSmith structure search with supplied element types/symmetry and an
   input-only resource policy sufficient for its supported problem class. Save
   the selected model and termination before reading reference coordinates.
7. Assess complete element counts and periodic element-preserving coordinate
   recovery up to the stated setting/origin/inversion conventions. Report a
   low-R1 wrong/partial model as such. If no model is returned, preserve the stop
   and diagnostics. Standalone single-crystal structure refinement is outside
   this runner's scope.

Before running, commit a stage-check specification: frame/mask/axis checks,
reserved positional residuals, domain assignment/completeness, integrated-count
controls, held-out equivalent consistency, search budgets and count/coordinate
assessment. Keep the first recorded outcome even when it fails. Enstatite is
previously inspected development evidence; independent transfer validation still
needs additional datasets.
