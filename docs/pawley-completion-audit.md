# Pawley completion audit

Reviewed 2026-09-17 against implementation commit `e39cf72` on
`codex/pawley-implementation-plan`. This is a focused comparison of the code,
tests and recorded evidence with the [original plan](pawley-plan.md), not a new
numerical validation run.

Follow-up status: the matrix-free solver and exact finite-support boundary
treatment now pass both repeated measured CW gates under unchanged scientific
thresholds and budgets. Sucrose converges locally in 137–138 seconds at Rwp
0.066054. Matrix-free products, dependent constraints, allocation limits,
checkpoint mode checks and format-1 migration are tested. The executable
manifest, covariance tests, measured LaB6 cell/profile gate and multiphase
recovery are complete. Sections below retain the original findings as an audit
trail. Mixed native format-6 bundles and cross-language resume are now implemented.
External optimizer comparison, spectra and TOF remain unfinished. The original checkout's numerical work is now committed upstream
and has been merged; the combined Python suite and native checks pass.

The single-histogram, fixed-wavelength CW method is implemented. Its native
objective/solver, NumPy interface, bounded/tied areas, selected cell/profile/
background variables, diagnostics and standalone restart/persistence exist.
“Fully implemented” still has two boundaries: completing the CW release and
completing the broader spectra/TOF roadmap.

## Complete the CW release

### 1. Close the measured sucrose convergence gate

The final measured report records 811 families and 23,003 samples, Rwp
0.0660539713, correlation 0.9967438710 and 16 accepted steps. Both runs stop at
`max_runtime`, around 247 seconds including diagnostics, rather than converging.
Neutron LaB6 passes. The synthetic large case passes, but its fixed geometry and
synthetic observations do not substitute for the measured joint-fit gate.

First add stage timings and counts for profile/Jacobian assembly, backtracking,
QR, active-face solves, multiplier solves and final rank diagnostics. Record
projected-gradient/KKT diagnostics and the convergence criterion that fired.
This distinguishes expensive linear algebra from poor nonlinear progress.

The code currently constructs dense physical and free Jacobians, copies a
weighted Jacobian, performs augmented QR, rebuilds active-face factorizations,
and runs a final dense SVD. Every line-search trial uses the full evaluator.
These are concrete optimization candidates, not proof that one alone explains
all elapsed time. Preserve same-pass value/derivative semantics and the
constraint/support contracts when changing their representation or reuse.

Exit: both measured cases satisfy the existing scientific and convergence
criteria within the declared budget on a recorded machine, with reproducible
accepted states. Do not relabel a budget stop or loosen tolerances to pass.
A longer-budget run can diagnose eventual convergence but does not close the
current performance gate by itself.

### 2. Make the validation contract executable and complete

`python/phasesmith/validation/pawley.py` reads dataset ranges, quality thresholds
and iteration ceilings from the manifest. Other values, including support,
instrument/cell setup, background windows, selected profile parameters,
improvement threshold, weighting tolerance and runtime budget, are hardcoded.
Several corresponding manifest entries are currently descriptive only.

Make a versioned, validated manifest authoritative for the entire scientific
request and acceptance policy; record the resolved settings, reflection/domain
identity and source revision in each result. Preserve existing reports and
explain any future contract change. Add tests proving that changed controls
actually change the request/checks, or reject unsupported settings.

The measured runner compares actual calculated arrays and accepted histories.
`benchmarks/pawley.py`, however, compares only dictionaries containing relative
L2, rank, accepted-step count and termination. Equal summary values do not prove
identical arrays or histories. Add direct comparisons or deterministic array/
history hashes, objective/linear-solve evaluation counts, and isolated repeated
performance measurements. The existing benchmark is still useful profile-error
and timing evidence, with this narrower repeatability scope.

Extend scientific coverage with a measured cell-refinement case and joint
multiphase cell/profile fits. Current measured cases fix cells; current cell
recovery is synthetic. Keep covariance limitations explicit and add focused
checks of uncertainty scaling/constraint propagation as that interface grows.

Exit: an opt-in dataset-backed release check consumes the full contract, fails
on failed gates, and records sufficient provenance and diagnostics to reproduce
its conclusions. Normal unit tests remain independent of external datasets.

### 3. Finish shared project and persistence integration

`phasesmith-pawley` version 1 is a working standalone JSON format. It is not a
Pawley analysis in the existing shared project/bundle model. The shared native
wire format has Rietveld and TOF analysis records but no Pawley analysis field.

Add the shared analysis/state record, Python/native adapters, the appropriate
new bundle version and backward-compatible loading of projects without Pawley
analyses. Test mixed analyses, future-version rejection, corruption, migrations
and explicit Python-save/native-load/resume and native-save/Python-load/resume
fixtures. Existing per-language tests of the common codec provide a foundation;
they do not cover mixed bundles or migration because those features do not exist.
Retain compatibility with the already committed standalone format.

### 4. Finish the independent external comparison

The current pinned GSAS-II profile fixture validates area conventions and
profile extraction. It does not validate a live GSAS-II Pawley optimizer run.
Complete the revision-gated probe described in `oracle/PAWLEY_STUDY.md`, with an
explicit F-squared/multiplicity/LP-to-area conversion. Compare equivalent fixed
states first, then fitted profiles, cells, identifiable isolated areas and
resolvable overlap sums. Record differences in bounds and signed intensities.
Keep all oracle code external to production and do not translate GSAS-II code.

The original plan permits a documented oracle exception backed by independent
evidence. That supports an explicitly limited CW release; it is not equivalent
to completing the full external-comparison roadmap.

## Complete scalability and the broader roadmap

| Work | Required implementation and evidence |
| --- | --- |
| Matrix-free CW | Support-block JVP/VJP, bounded iterative steps, dense/product/adjoint equivalence, memory scaling, scoped rank diagnostics, and restart controls that identify solver mode. Preserve the dense solver as a small-problem reference. |
| Fixed CW spectra (P6) | One intensity per family across weighted wavelength components; component positions/shapes and union-domain coverage; fixed detected-area weight conventions; derivatives, persistence, doublet and measured-data validation. |
| TOF (P7) | Single-bank microsecond/density contracts, then selected calibration/cell refinement and joint multi-bank acceptance. Share cell/symmetry; keep bank-local areas/background/profile by default. Test calibration degeneracies, nonuniform support, atomic resume and measured POWGEN/LANL nickel cases. |

Matrix-free solving planned in P3 is now implemented, with dense/product,
constraint, restart and memory-limit tests. Its measured acceptance gate is
still under investigation. Schur/variable-projection acceleration is optional
and needs independent bound/derivative validation before adoption. P6 and P7
are follow-on scope and do not block a clearly labelled CW release.

## Integration and release housekeeping

The original checkout still has uncommitted changes to CW/FCJ accuracy,
reflection conventions, Python boundaries and persistence. Reconcile these
before merging; the Pawley branch is based on committed `28a10e5` and does not
contain that work. Re-run numerical, oracle and measured checks against the
combined state. This audit did not modify the original checkout.

For release, update the formal versioned public API snapshot without rewriting
historical release snapshots, complete the method-comparison/oracle index
entries, and make the feature matrix agree with passed gates. The supplemental
Pawley API snapshot is already tested by pytest. The existing formatting issue
in `docs/fit-report.md` is unrelated and remains outside this change.

Recommended order: instrument and close the measured CW gate; harden the
validation runner alongside that work; add matrix-free scaling if needed for
the target workload; complete shared bundles and external comparison; reconcile
and release CW; then implement fixed spectra and TOF separately.
