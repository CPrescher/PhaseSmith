# PhaseSmith Project Brief

## Purpose

PhaseSmith is a modern, open-source powder-diffraction computation
library. Its numerical kernels are written in Rust and exposed through a small,
typed Python API. The first product layer is a trustworthy and fast
powder-profile calculator. Refinement orchestration follows only after the
profile kernels are mature; Le Bail extraction is the first supported complete
refinement workflow. Project-file management and graphical user interfaces are
not part of the numerical library.

## Current implementation status

Profile execution optimization is complete through three separately measured stages.
Stage 1 caches exact immutable teacher pixel/voxel intersections in source
order with an explicit additional-memory cap (default 64 MiB) and streaming
fallback; all learning equations, iteration counts and support conventions stay
unchanged. Native entry points retain their convenience signatures and expose
resource-controlled variants. Cache metadata is diagnostic, not a scientific
profile parameter. Stage 2 runs independent bank iteration sequences in a private native pool,
retaining each bank's exact scalar accumulation order; its stable row index is
admitted under the same cache cap, with serial fallback. Stage 3 adds resource-controlled per-reflection GLS: exact first-pass
derivative contributions are cached under a per-pool cap and replayed without
changing arithmetic; independent reflections run in a private pool, and
integer overlap bookkeeping retains only one exposure at a time. Legacy
streaming integration remains available for controls/fallback comparisons.
Every stage has fresh measured runs and numerical-identity reports. Final v16
learning/integration medians are 2.493/2.185 s versus 38.780/17.792 s; the fresh
full numerical worker takes 37.86 s versus 87.77 s and CLI 38.95 s versus 88.80 s.
Every numerical array, candidate and phasing history is bitwise identical to
v11. Rust/Python validation passes 510/3,245 tests, plus strict Clippy and
formatting. The example explicitly admits 256 MiB extra cache per phase;
the generic default remains 64 MiB with streaming fallback. See
`docs/profile-execution-optimization.md` for individual trials and timing limits.

Rust-only production is an explicit user requirement (31 October 2026). The
initial NumPy/SciPy shadow prototype has been replaced by native kernels in
`phasesmith-spots`; Python remains an optional array/loader adapter, and SciPy
is retained in independent numerical references for these single-crystal paths. Production masking
now has native block medians, float32 filters, percentile normalization,
connectivity/morphology and full-exposure DAC visibility. Single-crystal geometry
fitting now also lives entirely in `phasesmith-spots`, including parameter chains,
bounded soft-L1 optimization and conditional/withheld diagnostics. An independent
NumPy/SciPy reference remains separate. Strong-profile and empirical-scaling
experimental optimizers now have native Rust entry points for bounded shape fits,
frozen weak filters, conditional joint sigma and robust within-family scaling;
shape fitting uses a small dense limited-memory BFGS quadratic with generalized
Cauchy/free-subspace steps and safeguarded More–Thuente search. Only its BSD
interpolation helper is vendored, with provenance/license; no new runtime
dependency or Rust minimum change. The first v10 simplified-quasi-Newton run
recovers the same structure but fails frozen reduction equivalence; it is retained
in `benchmarks/enstatite-native-profile-scaling-v10-macos.json`. Frozen-shape
controls isolate that drift to optimization trajectories.
The corrected v11 fresh raw run passes structure recovery and every frozen
reduction-equivalence gate: Mg16 Si16 O48, 4,125 observations/220 negative,
R1 .0408518, maximum/RMS reference errors .0075236/.0045385 A. Intensity/sigma
relative L2 drift is 2.42e-5/2.15e-5, at most .02153 baseline sigma; positions
and export mask match v9. Strict end-to-end parity is still false: R1 delta
.00020038 exceeds .0001 and maximum candidate-coordinate delta .0017993 A
exceeds .001. These limits and the v10 failure are retained. Production workers
reject SciPy/h5py/hdf5plugin imports. Validation passes 509 Rust and 3,235
Python tests, strict Clippy and formatting. See
`benchmarks/enstatite-native-profile-scaling-macos.json`. The isolated
64-profile benchmark is 1.07x faster and the 4,096-row/173-parameter scaling
benchmark 4.32x faster; a NumPy-only wheel passes both with no SciPy present.
See `benchmarks/profile-scaling-fit-rust-macos.json`.
A subsequent common-input audit replays both old/new binaries bitwise: search
code is unchanged. Candidate/path differences account for about 99.45% of the
reported R1 change; the native-run candidate fits both datasets better. Both
polishers hit the 80-iteration cap. Further native iterations stagnate at the
1e-5 A^2 ADP eigenvalue lower bound despite a nonzero feasible coordinate
gradient; a coordinate-only control improves the objective. This existing
boundary-handling limitation needs a constrained-step fix before calling
candidate polishing converged; no production solver change was made in the
audit. See `benchmarks/enstatite-search-convergence-macos.json`.
An isolated, numerically identical raw run takes 88.80 s end to end on an
Apple M4 Pro: profile learning 38.01 s, amplitudes/strong shape fits 24.01 s,
pixel collection/background 8.71 s, detection/linking 6.46 s, compressed
archives 4.81 s, masks 3.18 s, indexing/geometry .23 s and solver .552 s.
The observer partitions wall times without adding nested calls. See
`benchmarks/enstatite-workflow-timing-macos.json`.
Do not claim the complete raw workflow is Rust-only until orchestration and
remaining design/convenience calculations also have Rust entry points. Do not
introduce further production SciPy algorithms. HDF5 I/O now lives in the separate optional `phasesmith-hdf5` adapter.
The facade and Python extension enable it with feature `hdf5`; release wheels
include it. It builds HDF5, Bitshuffle/LZ4, zlib and LZF from Cargo sources,
requiring a C compiler, CMake >=3.26 and Git at build time, with no installed HDF5 or
Python compression plugins at runtime. The minimum Rust patch version is now
1.85.1 for the HDF5 bindings. Default Rust facade builds remain HDF5-free.
The v8 MgSiO3 full rerun passes through this reader plus native masks: Mg16
Si16 O48, R1 .0411159, maximum/RMS coordinate errors .0065804/.0045658 A in
95.85 s. Every reduced array and original-frame hash is bitwise identical to v7.
The clean wheel needs no h5py/hdf5plugin, external HDF5 or dynamic plugins. This
validates I/O, not a completed Rust-only pipeline. The record is
`benchmarks/enstatite-bundled-hdf5-macos.json`.
Python orchestration remains pending. The v9
native-geometry full run passes every frozen v8 equivalence gate: same 4,125
observations/flags/HKL mask and Mg16 Si16 O48, R1 .0410522 versus .0411159,
maximum reference error .0069006 A, maximum aligned coordinate change .0005351 A.
Intensity relative L2 drift is 9.21e-6 (maximum .009495 baseline sigma); indexed
predictions change by at most 2.13e-7 pixels. Native geometry fitting needs no
SciPy, passes independent/reference/derivative/native-map tests, and takes
22.96 ms versus 34.53 ms for the previous fitter on the synthetic 10,888-spot
benchmark. Full validation passes 505 Rust and 3,225 Python tests plus strict
Clippy/formatting. Reports: `benchmarks/enstatite-native-geometry-macos.json` and
`benchmarks/geometry-refinement-rust-macos.json`. Large raw data and full products stay outside Git; only a 10-KiB independent
compressed interoperability fixture is checked in.

The script-first MgSiO3 workflow now passes from original HDF5 frames through
PhaseSmith structure search and post-search labelled-coordinate assessment.
Frozen v5 processes 142 frames, learns 278 profiles and exports 4,310 signed
observations, including 233 negative intensities. It infers Mg16 Si16 O48 at
R1=0.0413175, with maximum/RMS reference errors 0.0078273/0.0050056 A. All raw
sources and large products remain outside Git; references are read only after
the numerical worker saves its candidate and exits. No source HKLs/peak table,
archived reduction, atomic count or coordinate enters that worker.

The retained v1--v4 failures led to two example-level validity repairs: a declared
35-degree DAC cone excludes occulted rays, and profile MISMATCH remains a visible
warning rather than an unconditional export veto. Both are needed in the tested
two-factor diagnostic. Every integrated amplitude, sigma and flag remains
bitwise identical to v4; only explicit search eligibility changes. The full raw
rerun reproduces the corrected diagnostic exactly. This is exposed development
evidence with supplied cell/symmetry/elements and provisional spindle angles,
not independent DAC metrology or an untouched holdout. Centre-ray exclusion does
not model absorption/rocking partiality. Native equations and default builds
are unchanged; standalone structure refinement and public API promotion remain
pending. Small records are `benchmarks/enstatite-raw-workflow-macos.json`,
`benchmarks/enstatite-reduction-diagnosis-macos.json` and the preserved
`benchmarks/enstatite-raw-workflow-v4-macos.json`. See
`docs/enstatite-reduction-diagnosis.md` and `docs/enstatite-raw-workflow.md`.

Shared shadow masking is now implemented in
`python/phasesmith/experimental/frame_validity.py`: immutable coarse/native
validity snapshots with exact source/exposure identity and archives; the
MIT-attributed portable Dioptas illumination-ratio estimator; conservative
full-exposure/four-pixel-corner geometric DAC masks; mask composition; and
flagging of incomplete position supports. Loaders remain application-owned;
there is no Dioptas, Qt or HDF5 runtime dependency. Production estimation and
geometric visibility now execute in Rust with no SciPy dependency; the independent
reference alone uses SciPy. Low global illumination alone is diagnostic; insufficient
reference or wholly dark unclassified exposures return an explicit unavailable
error. Counts are never replaced, normalized or interpolated by a mask.

Synthetic tests check worker equality, independent angular geometry, masks in
foreground/background/teacher/temporal fits, immutable/source identity and
archives. The 40-frame/64-peak benchmark matches the unmodified Dioptas estimator
bitwise for masks and source hashes. MgSiO3 v6 is frozen to use the same estimated
pixel masks before detection and integration, replacing v5's post-hoc centre
cone. Its v6 image-only run recovers the correct structure (maximum error .01223 A)
but misses the numerical intensity target at R1 .13945. That outcome is retained.
The v7 composition with the unchanged illumination estimator and the v5
explicitly declared DAC aperture recovers Mg16 Si16 O48 at R1 .0411159 and
maximum/RMS coordinate errors .0065804/.0045658 A. Full exposure/block masks
replace the centre-only geometric check. The native masking migration is
validated separately against this retained outcome. Default native numerical
kernels/equations, public namespace promotion and Dioptas UI migration remain unchanged. See
`docs/single-crystal-frame-validity.md` for the API, contracts and validation.

Structure search now has an explicit per-dense-reflection-buffer memory policy.
`SolutionOptions.dense_memory_budget_bytes` defaults to the historical
16,000,000 bytes; callers can request up to 1 GiB. Native input-only preflight
derives admissible full-cell atom capacity from expanded measured rows, cell
volume and that budget. P1 preparation/polishing, symmetric Jacobians and their
linear-operator paths carry the same checked policy. FFT/cache bounds and
physical/numerical equations remain unchanged; this is not a total-RSS limit.
Automatic element-search caps now respect both volume and memory; explicit
over-budget caps still fail. See `docs/solution-memory-budget.md` for contracts,
independent integer admission, compatibility and validation.

With a 64 MiB dense budget, native preflight admits 105 atoms for enstatite's
10,694 expanded measured rows. One unchanged-seed/effort PhaseSmith search per
existing intensity source recovers Mg=16, Si=16, O=48. Independent post-search
reference matching gives maximum/RMS errors 0.00920/0.00536 A for the archived
PhaseSmith reduction and 0.01200/0.00682 A for the source integration. Native
amplitude R1 is 0.0428393/0.0481402. Reference coordinates/counts do not enter
search. These two successes use existing reduced intensities; the fresh raw-run
outcome is recorded above. No standalone structural refinement is claimed.
The change passes 489 Rust and 3,174 Python tests plus formatting/strict Clippy;
kernel fingerprints match the preserved baseline exactly. Reports are in
`benchmarks/enstatite-memory-budget-search-macos.json` and
`validation/solution-memory-budget-validation-v1.json`.

The 31 October single-crystal promotion audit proposes a supported
`phasesmith.single_crystal` namespace and an external-data MgSiO3 enstatite
workflow ending in PhaseSmith structure search and candidate assessment.
Standalone single-crystal structure refinement is not implemented; internal
candidate polishing and optional external SHELXL diagnostics remain distinct.
The eight source files total 470,052,794 bytes and stay outside Git behind a
checksum manifest and local verifier. At the audit baseline, search capacity was a concrete
blocker: 10,694 expanded measured rows limit the existing dense search to 36
full-cell atoms, while the reference contains 80. Source HKL instructions must
also be separated explicitly from its reflection block. Public API/build
promotion and independent transfer validation remain pending; the qualified raw
MgSiO3 gate now passes as recorded above. Production equations, defaults, schemas
and feature gates are unchanged. See
`docs/single-crystal-promotion.md`, `docs/single-crystal-api-proposal.md` and
`docs/enstatite-workflow.md`.

An opt-in `experimental-solution` module accepts externally integrated,
corrected HKLF 4 observations, a cell, known symmetry, and full-cell non-H
composition. A separate Rust crate owns multi-start charge flipping, periodic
peak extraction, provisional element assignment and analytical polishing;
independent NumPy references check the numerical equations. Signed source
observations remain authoritative. Symmetry/Friedel merging and derived
nonnegative phasing amplitudes are explicit. After full-cell P1 recovery, the
default experimental path aligns the origin/hand using the candidate alone,
reconstructs the asymmetric unit and constrains coordinates and positive-definite
anisotropic displacement tensors to site symmetry. Isotropic and legacy P1 modes
remain explicit options. Failed alignment retains an honestly labelled P1
hypothesis and diagnostic. Density retains its original frame with an explicit
model transformation; exact supplied operations and both CIF frames are exported.
A fit target is not a correctness certificate. This capability does not require
detector integration, a GUI, or external solvers. Default powder builds and stable
APIs are unchanged. Scope and validation are recorded in
`docs/experimental-structure-solution.md` and `docs/solution-symmetry-adp.md`.

The constrained ADP follow-up improves the thiourea development result from
10.19% to 4.94% amplitude R1. With the same preselected settings, a blind measured
BiSeI transfer case recovers all 12 atoms within 0.003 angstrom at 5.23% R1;
K[BrF4] fails coordinate/symmetry recovery despite 9.12% P1 R1. Both outcomes are
recorded. This is bounded experimental coverage, not general structure-solution
validation. Automatic space-group selection, anomalous scattering, hydrogens,
disorder and twinning remain outside scope.

An additional experimental phasing strategy implements Patterson superposition
minimum-function starts, hybrid difference-Fourier iteration and Gaussian
atomicity masks with deterministic periodic random omission, from the published
Sheldrick (2015) equations. It does not reproduce full SHELXT intrinsic phasing,
space-group selection or resolution extension. `phasing="dual_space"` selects
this alternative; `"portfolio"` alternates it with charge flipping inside the
existing total budgets. Charge flipping remains the default. The input cell,
symmetry and full-cell composition contract is unchanged. Per-trial method,
initialization/fallback, Patterson vector, score and fixed controls are exported
in experimental result schema v3. Independent NumPy complete-cycle and Gaussian
support/moment checks accompany the native implementation. The original uncompleted development
cases recover thiourea and BiSeI, but K[BrF4] fails despite a lower P1 R1;
no increase in demonstrated solve coverage is claimed. Equations, bounded
implementation choices and measured results live in `docs/solution-dual-space.md`.

A bounded partial-framework completion step now follows failed or insufficient
atomic candidates. It removes the weakest species, recovers symmetry from the
stronger atoms, retains up to eight different framework origin/settings and
searches a signed difference-Fourier map for one missing orbit. Generic affine
fixed-set projections generate special-position proposals without reference
coordinates or Wyckoff tables. All complete proposals use fused value/analytic
coordinate derivatives and the existing constrained ADP refiner. A completion
must improve the original amplitude objective; exact counts, non-overlap and
original density-frame transformations remain observable. The explicit
`completion=False` option preserves the earlier path; historical benchmark
commands keep completion off unless `--completion` is supplied. Experimental
JSON schema v4 records typed actual completion effort/outcomes. On reused K[BrF4]
data, the native dual-space search now recovers all 24 atoms to 0.006 angstrom at
R1 0.0222664 without external/reference coordinates. This does not implement
general multi-orbit fragment assembly or automatic space-group selection.
See `docs/solution-completion.md`. External binary inspection is kept outside the
repository and has not been used to implement these numerical equations.

A separate bounded element-only benchmark now enumerates sampled symmetry-orbit
multiplicities and repeated count-constrained solves under a declared volume
bound. Its inputs contain element types, cell and exact symmetry, with no formula
or reference coordinates. It saves every count proposal, failed/capped outcomes
and a descriptive complexity ranking; it does not change the native API to a
general unknown-composition solver. On reused K[BrF4] data, 45 count hypotheses
select Br4 K4 F16 at R1 0.0222664 with all 24 atoms within 0.006 angstrom;
30 hypotheses hit their time caps. Exact I4/mcm and cell remain supplied priors.
See `docs/solution-element-counts.md`.

The experimental `solve_structure_elements` path now uses a shared native search
instead of restarting the fixed-composition solver for every count combination.
One equally weighted element-sharpened charge-flipping map initializes a bounded
beam of symmetry-compatible strong-site settings. Difference-map orbit proposals
share their phase sums and analytical coordinate jets across possible elements;
whole-orbit births, deletions and relabels compete after constrained refinement.
Counts follow from accepted orbit multiplicities. Required-element coverage,
periodic non-overlap, explicit budgets and a descriptive complexity penalty govern
selection. Original density frames and retained partial results remain explicit.
Experimental schema v5 distinguishes inferred counts from input element types;
legacy count-constrained schema v4 and stable/default builds remain unchanged.
The new equations have an independent NumPy reference and derivative tests.
On three reused measured cases it infers the correct non-H counts and matches
reference coordinates: K[BrF4] in median 3.167 s (three repeats), BiSeI in 2.298 s
and thiourea in 27.981 s. K[BrF4] is 36.7 times faster than the recorded 45-count
sweep; this is not a thread-matched comparison with SHELXT. See
`docs/solution-shared-search.md` for controls and qualifications.

Shared search now has a staged refinement schedule: short isotropic fits screen
proposed edits, a conservative per-parent shortlist preserves close scores and
missing-element birth proposals, and only fully refined requested-mode states
enter the retained beam. Per-site isotropic tensors and composed coordinate
frames seed final anisotropic fits exactly. `staged_refinement=False` retains
the previous full-fit path for controlled comparison. Independent selection and
tensor-seed references accompany the unchanged value/derivative kernels. The
paired benchmark pins observation inputs and defers reference-formula/geometry
checks until every solve has been saved. Three serial alternating pairs per case
preserve all counts, reference geometry and exported-CIF fits. Full/staged medians
are 3.250/2.260 s for K[BrF4], 2.314/1.123 s for BiSeI and 28.288/7.572 s
for thiourea (1.44x, 2.06x and 3.74x). These are reused development cases. See
`docs/solution-shared-search.md`.

Shared element search also has an integrated-peak-density guide. One unsharpened
measured-amplitude map per phase start is integrated over Cartesian balls by an
analytic Fourier kernel. Periodic interpolation returns the integral, three center
derivatives and a radius derivative in one pass. Relative mass/Z calibration of
current strong orbits narrows birth/relabel alternatives and the staged shortlist;
nearby elements, missing types and the leading diffraction proposal retain rescue
paths. Exact counts and coordinates are never inputs. The prior staged search is
available with `integrated_peak_density=False`. Independent quadrature, direct
Fourier, derivative and Gaussian-volume checks cover the new kernel.
Three serial paired repeats per measured development case preserve all counts,
reference geometry and R1. Staged/density-guided medians are 2.269/1.696 s for
K[BrF4], 1.150/0.973 s for BiSeI and 7.546/5.658 s for thiourea (1.34x, 1.18x
and 1.33x). See `docs/solution-peak-density.md`. Default powder builds remain
unchanged.

Shared element-only search now also accepts count-free Patterson/dual-space
phasing. A positive map-RMS peak selector replaces the supplied-N atomicity mask;
its explicit peak cap is a resource bound, independent of the downstream atom
bound. Per-cycle requested/effective thresholds and selected/omitted peak counts
are exported separately from inferred composition. The fixed-count path remains
explicit, and portfolio starts share the existing total budgets. Independent
NumPy complete-cycle and projection references, boundary cases and synthetic
count-recovery tests cover the extension. Three serial paired measured repeats
recover K[BrF4] and thiourea with unchanged counts, geometry and R1, but cost
2.547 versus 1.719 s and 6.001 versus 5.583 s. BiSeI fails framework initialization
with the new phasing while charge flipping solves in 0.993 s. No speed or coverage
improvement is claimed; charge flipping stays the default. See
`docs/solution-count-free-dual-space.md` for the frozen comparison and limitations.
Default powder builds remain unchanged.

The count-free dual-space follow-up fixes a missed-origin failure in nearly
rational centrosymmetric frameworks. If the existing coarse origin grid fails,
inversion-related peak pairs provide all half-cell branches analytically before
the unchanged full-group bijection/tolerance check. The saved BiSeI framework
matches within 0.049 angstrom and now completes without input counts or reference
coordinates. Count-free dual-space also hands window-best maps to full shared
search at 4,16,64,... cycles, accepting only complete requested-mode refinements
that meet the original R1 target. Failed handoffs resume the same Fourier arrays
and global cycle budget. Beams and their density frames remain paired across
checkpoints; actual window selections and acceptance are exported. The explicit
`dual_checkpoints=False` ablation keeps full-budget phasing. Independent NumPy
window/seed/cycle references and translated synthetic-orbit regressions cover
these changes; existing profile/structure-factor derivatives are unchanged.
Three serial paired repeats now recover all three measured development structures
with the correct counts, geometry and original-observation fits. Charge-flipping /
dual-space medians are 1.705/0.667 s for K[BrF4], 0.970/0.435 s for BiSeI and
5.486/5.761 s for thiourea: 2.56x and 2.23x speedups for the first two, but a 5%
slowdown for thiourea. The same-build full-budget ablation also recovers all three,
isolating the origin fix from the checkpoint schedule. An experimental shortcut
stopping at the first R1-target fit misassigned C/N and was rejected; the final
path completes the original refinement shortlist and beam round.
See `docs/solution-dual-space-handoffs.md` for diagnosis, validation and timings.
Charge flipping remains the default; default powder builds remain unchanged.

Prepared experimental structure refinements now cache fixed reciprocal/image
geometry in bounded contiguous arrays, rather than recomputing symmetry-rotated
vectors, coordinate phase derivatives and tensor quadratic forms on each fit
evaluation. Isotropic fits also share attenuation and its analytical derivative
across symmetry images of a site/reflection. Values and derivatives remain fused,
with unchanged candidate shortlists, orbit edits, ranking and full anisotropic
final fits. An explicit `cached_refinement=False` shared-search ablation retains
the original direct arithmetic. The 32 MiB per-model cache cap falls back to the
direct kernel without dropping observations; independent direct NumPy values,
centered derivatives, deterministic native comparisons and memory-boundary tests
cover the factorization. Native sampling motivated this change: thiourea has
156 screening and 34 final fits over 4,068 expanded reflections. A targeted
external SHELXT inspection and the published pipeline support preparing reusable
numerical work and keeping cheap screening before expensive final refinement;
no decompiled code was copied or translated. See
`docs/solution-refinement-geometry-cache.md` and
`docs/solution-shelxt-order-review.md` for equations and evidence limits.
Three serial paired dual-space repeats preserve counts, geometry, original-data
R1 and both CIF frames for all three development structures. Direct/cached
medians are 0.732/0.514 s for K[BrF4], 0.487/0.356 s for BiSeI and 6.573/4.451 s
for thiourea (1.42x, 1.37x and 1.48x). Screening/final fit counts are identical
between paired modes; no candidate shortlisting or atom-count prior is changed.

The shared element-only solver now has opt-in supplied-symmetry density recycling
before weak-peak assignment. Each retained framework origin receives ten phase
symmetry/Friedel averaging and positive-density cycles, with measured amplitudes
restored and missing coefficients kept as predictions. Recycled peaks and
integrals seed bounded joint whole-orbit proposals; coordinate-fixed shared
label sweeps and the original constrained refinement/complete shortlist resolve
assignments. A batch may meet the target only after a complete ordinary edit
round finds its preferred complete lineage stable. Counts remain inferred, and
original density frames and both CIF exports retain their provenance. Independent
NumPy full cycles cover translated/inverted origins and screw symmetry; existing
shared structure-factor/derivative references cover the assignment objective.
Three serial paired dual-space repeats preserve all three development structures,
counts, geometry and R1. Previous/recycled medians are 0.521/0.767 s for K[BrF4],
0.356/0.252 s for BiSeI and 4.516/1.499 s for thiourea: 47% slower, 1.41x and 3.01x
faster respectively. The option `symmetry_recycling=True` remains explicit because
of the K[BrF4] regression. There is no compound-specific dispatch or input count
prior. See `docs/solution-symmetry-recycling.md` for equations, safeguards and
frozen comparisons. Default powder builds and default shared search are unchanged.

The shared element-only dual-space solver additionally supports opt-in
`tangent_refinement=True`: empirical count-independent normalized magnitudes,
measured strong-triplet tangent updates, cooling phase perturbations and measured
weak-cross negative-quartet guards. A bounded immutable phase plan is reused
across starts; actual relation counts, updates and accepted/declined quartet
checks are exported. Rust owns the numerical work and fused phase-invariant
values/derivatives; independent NumPy graph enumeration, dense derivatives and
complete-step references validate the equations. This is an independent bounded
variant from published direct-method ideas, not a translation of SHELX code or
an exact SHELXS annealing probability model. Counts and labels remain inferred.
Three paired repeats with and without symmetry recycling preserve correct
counts, coordinates, R1 and both CIF frames for all three development structures.
Without recycling, baseline/tangent medians are 0.5395/0.5490 s for K[BrF4],
0.3722/0.4724 s for BiSeI and 4.6447/3.4749 s for thiourea. With recycling they
are 0.7881/0.8004, 0.2559/0.3212 and 1.5386/6.0270 s. The regression on combined
thiourea search leaves this option off by default. K[BrF4] supplies no eligible
negative quartets, reported explicitly. See `docs/solution-tangent-refinement.md`
for equations, resource bounds, interpretation limits and frozen reports.

A follow-up improves the combined tangent/recycling path by deferring tangent
updates until a strong-framework origin is known. Each existing recycling cycle
then restores supplied symmetry, proposes an annealed tangent step and restores
symmetry again; a second negative-quartet guard checks the actual constrained
phases before positivity. Exact stored Fo values keep the frozen-set ranking
stable across equal symmetry-related amplitudes, and an explicit local cost-tie
convention stabilizes centric cases. Accepted phase-change counts exclude zero
amplitudes. Independent NumPy complete-cycle references and synthetic cases cover
all operations and actual guard rejections without loosening global tolerances.
`tangent_in_recycling=False` retains the original P1 placement for ablation;
without recycling the original tangent path is unchanged. Per-origin typed
audits distinguish constrained work from P1 tangent work. The map diagnostic
traces the thiourea regression to lower symmetry-phase coherence, a one-orbit
joint proposal instead of four, and 258/46 screening/final fits instead of 50/14.
The revised combination restores the four-orbit proposal and one edit round for
seed 0. See `docs/solution-tangent-recycling.md` for equations, timing and an
eight-seed comparison across all three measured development structures.
Three paired legacy/improved medians are 0.7817/0.7896 s for K[BrF4],
0.3083/0.2887 s for BiSeI and 5.7752/1.3347 s for thiourea (4.33x faster).
Eight predeclared seeds recover all three structures and exact non-H counts
(24/24 improved solves); all paired geometry, R1 and both-CIF checks pass.
Thiourea still spans 0.654–7.302 s across seeds: the slow seeds also have long
searches with recycling alone. Both options remain opt-in, and these reused
development cases are not a general solve-coverage claim.
Experimental grain search now additionally offers **Row search (DAFi-based)**
through `find_grains(..., method="rows", row_options=RowSearchOptions(...))`.
The selected independent published-algorithm kernel is imported verbatim into
the separate `phasesmith-row-search` crate; the existing FFT path stays default.
The exact ordered training grouping is retained separately from subsequent
per-group lattice inference, training-only consolidation and unchanged reserved
validation. `group_rows` exposes the exact grouping without indexing. The
selected wrapper enables the sparse naive/smart crossover at 2,048 unique
points, four projection offsets, relative gaps, eight workers and 512 MiB total
cache. Kernel defaults and arithmetic are unchanged. No original DAFi source
or executable, external checkout, SciPy or spglib is used at runtime. See
`docs/experimental-row-search.md` and `docs/row-search-provenance.json` for the
source hashes, equations, integration boundary and frozen parity checks.

The same opt-in pool now parallelizes independent polishing/trimmed fits,
observation/alias/simultaneous-update fits, final training-only forward fits,
reciprocal-branch work and coherent-parent recovery. Each task has one native
worker and single-thread numerical-library limits; dependent competition stays
ordered. Frozen inputs, original RNG order, alias-entry barriers, original
admission and global deduplication preserve every fit and assignment byte in
all four arms, including all 28 original hypotheses, the 66.30562768 Å³ example4
lattice and both example5 orientations. Direct eight-process before/after cold
pairs take 9.9–11.8 s versus 8.9–10.8 s, about 10% less aggregate wall time.
Raw example5 is slightly slower in that pair; no universal speedup is claimed.
Total CPU rises about 4%; speculative unused fits and transfer are included.
Recovery is faster in every arm, but the one-second target remains unmet.
See the local `fft-parallel-fitting-recovery` report and the expanded parallel
control benchmark. No numerical kernel, search budget, or validation change.

Experimental ensemble discovery now has opt-in ordered process scheduling.
Eight spawned processes run independent geometric configurations, Fourier
conditioning preparation, subset searches and paired recovery with one native
worker per search and inherited single-thread numerical-library settings.
Models, cohort budgets and deduplication are consumed in the original order;
validation and numerical kernels are unchanged. Full cold four-arm paired
runs preserve all 28 original hypotheses, the compact example4raw lattice,
both targeted example5 orientations, all model/fit diagnostics and every
assignment byte. Wall times fall from 15.2–17.3 s to 8.8–10.6 s (about 40%
aggregate less wall time), with more total CPU time and worker-local caches.
The one-second target remains unmet. Serial remains the default. The new
standard-library scheduler uses spawn, requires numerical thread limits at
startup, keeps callbacks/competition in the parent, and tears down workers
on success or failure. No dependency, approximation or validation change.
See `benchmarks/grain_parallel.py`, `docs/experimental-grain-search.md`, and
the local `fft-parallel-ensemble` report, including rejected scheduling probes.

The next exact fitting checkpoint owns reciprocal point snapshots once per
proposal cloud and returns only selected rows, preserving the original NumPy
least-squares solver, reciprocal arithmetic and every fit/validation gate.
Native Fourier ascent also replaces per-vector hash tables with fixed exact
memo storage derived from its existing 81-query bound; scalar fused moments
and query sequences are unchanged. Final four-arm paired inverses preserve all
28 original hypotheses, both targeted example5 orientations, the compact
example4raw lattice, all model/fit diagnostics and every assignment byte.
Full wall times fall from 17.0–17.4 s to 15.9–16.6 s (about 5% aggregate),
while isolated complete 80-proposal fits improve about 23%. Ascent-only gains
are around one percent and close to timing noise. The one-second target
remains unmet. No production dependency, approximation, search budget or
validation threshold changes. See `benchmarks/grain_fitting.py`,
`docs/experimental-grain-search.md`, and the local `fft-ascent-fitting-speed`
report, including negative prototypes and exact-parity provenance.

The next exact Fourier checkpoint reuses invariant native moment factors and
FFT plans, deposits only occupied XY columns in one contiguous float32 block,
skips X lines outside the spherical window, and precomputes maxima adjacency.
Fresh four-arm before/after inverses preserve all 28 original hypotheses, the
compact example4raw target, both targeted example5 orientations, every model
and fit diagnostic, and all assignment archive bytes. Final paired wall times
are 17.0–18.9 s versus 17.9–24.3 s at task start, about 10% aggregate less wall
time under variable host load; corrected example4 is essentially flat. One
second remains unmet. Isolated realistic conditional FFTs improve by about
32%; native ascent improves modestly. No search/validation setting, public
Python API, default native-ascent option or dependency changes. A paired-ascent
prototype provided no benefit and was removed. See `benchmarks/grain_fourier.py`,
`docs/experimental-grain-search.md`, and the local `fft-fourier-kernel-speed`
report for exact before/after sources, controls and limitations.

The next experimental FFT speed checkpoint preserves all 28 original retained
hypotheses and the requested example4/5 targets in fresh paired before/after
runs. Native exact triplet enumeration/ranking, batched first-cycle support,
scalar-volume reuse and bounded within-inverse Fourier reuse reduce work
without changing validation. An explicit `native_fourier_refinement=True`
option additionally moves the unchanged bounded ascent to a per-vector Rust
loop with fused moments and exact-trial reuse. Its small-matrix arithmetic is
numerically compared to the independent NumPy reference; it remains off by
default. Original 28 basis matrices and all accepted owners are preserved;
some supplementary example4 corrected candidates change slightly. Paired full
times on the final build fall from 28.2–30.0 s to 17.7–19.5 s, still above one
second. Both scalar
and native paths pass the strict two-component control. See
`benchmarks/grain_remaining.py`, `docs/experimental-grain-search.md`, and the
local `fft-remaining-bottlenecks` report. No new dependency or profile term is
introduced; geometry, masks, saturation policy and physical interpretation
remain unchanged.

Broad experimental FFT discovery now computes only the spectrum needed by the
requested direct-radius window, reusing the actual transform of empty deposited
lines. Retained FFT lines keep their original length, axis order, float32
deposition, maxima boundary semantics and stable tie order. An owned Fourier
cloud reuses invariant derivative coefficients and small moment batches bypass
thread dispatch, preserving ordered fused values/gradients/Hessians. Fresh
example4/5 replays retain all 28 prior hypotheses, the compact example4raw
target and both targeted example5 orientations with unchanged validation.
The one-second complete-inverse target remains unmet; full timings and CPU-load
caveats are in the local `fft-broad-discovery-speed` report. Aggressive Newton
termination lost hypotheses and was removed from production, as was a
trigonometric approximation without a measured speed benefit. See
`benchmarks/grain_broad_discovery.py` and `docs/experimental-grain-search.md`.

The latest opt-in FFT retention experiment restores all 28 prior accepted
example4/5 hypotheses across matched raw and arPLS arms, while retaining the
compact example4raw target and both targeted example5 orientations. A fresh
conservative training-only branch retains every fitted incumbent (including
later rejected ones); a separately fitted repair branch can add models only
through training-null, rank and protected-ownership checks. Final validation
and completed fits are reused unchanged. Cohort-aware fitting budgets distinguish
nearby matrices which actually fit different observations or HKLs. Native fused
physical fields accelerate one ranking stage; the complete cold inverse still
takes roughly 50–58 seconds, not one second. This examined-data recovery is not
a general recall or physical-grain guarantee and does not change defaults.
See `docs/experimental-grain-search.md`, `benchmarks/grain_retention.py`, and
the local `fft-hypothesis-retention` development report for the exact audit.

Earlier experimental conditional-search options rank seed directions by sharp
integer-phase population, reuse within-call directions, and fit deterministic
trimmed forward proposals. Already eligible input proposals can be preserved
before geometry fitting mutates them. Lazy valid-triplet prefixes and small
native point-score batches reduce cost while preserving their numerical
contracts. All options default off; final tight-core, ownership, rank/count,
null and reserved-validation criteria are unchanged. A common examined-data
setting recovers the compact example4 target and the two targeted example5 DAFi orientation
matches, but loses other prior hypotheses and remains above the one-second
budget. This is a partial development result, not a regression-free default;
see `docs/experimental-grain-search.md` and `benchmarks/grain_conditional.py`.

The experimental FFT direction kernel now shares fused Fourier moments during
bounded Newton ascent and parallelizes directions while preserving ordered
point reductions. Exact packed phase intersections accelerate triplet ranking;
a separate bounded proposal scorer shares three explicit reciprocal cutoffs.
These numerical accelerations preserve the measured full-run candidate bases
and ownership in the example4raw/5raw replays. Analysis-only fast portfolios
have not preserved all previous lattices or established a common one-second
validated search. Defaults and final acceptance gates remain unchanged; see
`docs/experimental-grain-search.md` and `benchmarks/grain_fft_speed.py`.

A further experimental recovery path adds distinct fitting budgets, competitive
training evidence, optional paired FFT directions and bounded coherent local
fits. Final forward fitting has an explicitly unvalidated training-only API;
training-ineligible competitors can be removed before the fixed global
validation pool is frozen. The forward fit is reused once, not run twice.
The4raw66.4cell and the missing raw5DAFi orientation can be recovered in different
explicit settings, but the added search can lose other prior matches. No common
regression-free setting was established; existing defaults remain unchanged.
See `docs/experimental-grain-search.md` for budgets, phase boundaries and limits.
A conservative training-only subcell-alternative check also prevents a sampled
index5subset from being counted as another synthetic component. Both bases and
exact integer maps remain audited. The original strict two-component control
passes without changing its assertion; physical grain counts in measured data
remain unresolved.

An additional opt-in observation-aware FFT ownership path uses native reciprocal
Jacobians, physical eligibility before competition, and shared training-only
responsibilities while retaining hard final ownership. Independent reciprocal-
and observation-fitting branches preserve useful compact-cell trajectories;
integer parent/child training evidence and all alternatives remain audited.
The count/rank/null/retention thresholds are unchanged, and nulls rotate both
coordinates and local error charts. Discovery never consumes DAFi cells or
labels. The extra fitting budget and changed ownership decisions require matched
comparators. Defaults and UI remain unchanged; see
`docs/experimental-grain-search.md` for equations, budgets and validation limits.
The complete examined-data pipeline with the existing final forward fit now
recovers both prior DAFi orientations in corrected example5, but still misses
the second raw-arm orientation and loses a prior example4raw hypothesis. This
mixed outcome remains opt-in. Ablations also prevent attributing recovery solely
to soft responsibilities or preserved branches.

Further opt-in FFT proposal experiments retain multiple conditional alternatives
and enumerate exact homogeneous HKL congruences on detector-consistent training
positions to propose smaller cells. Parent hypotheses are retained; congruence
support does not establish crystallographic primitivity. Existing final
tight-core, competition, positional and rotated-control criteria are unchanged.
No comparator cells or labels enter the inverse. Bounds and independent residue
tests are documented in `docs/experimental-grain-search.md`.
The earlier expanded/congruence development experiment remains off by default after
mixed example4/5 results, including example4 regressions. A second example5
orientation can be reconstructed in late training-only diagnostics but is not
admitted by that checkpoint's criteria. The newer ownership pipeline and its
identical final forward fit are a separate opt-in investigation, documented
below; do not generalize an examined-data recovery to blind accuracy.

The experimental FFT ensemble additionally exposes opt-in diverse training
resamples and early training-only positional proposal refinement. Original
proposals remain in the pool; final tight-core, ownership, rank/support and
rotated-control gates are unchanged. Detector observations and fixed geometry
are required only for positional polishing. Defaults and the Dioptas UI remain
unchanged. See `docs/experimental-grain-search.md` for bounds and development
validation on the frozen, mask-matched example4/5 populations.

The opt-in FFT ensemble additionally supports training-only Fourier-plane
conditioning for sparse lattice proposals. A promising single translation
selects an integer-phase subset before another bounded FFT/triplet search;
smaller exact integer cores can propose removal of noise-contaminated supercell
aliases. These proposals enter the existing consolidated ownership and unchanged
null/rank/support validation. DAFi is only an external diagnostic comparator;
no comparator cells/groups, chemistry or held-out coordinates enter discovery.
The original search and ensemble defaults remain unchanged. Examples4/5 are
development evidence, with provisional geometry and recorded mask departures;
conditioning alone left the earlier example4 background arm negative. The newer
mask-matched stability/polishing campaign improves positional support but does
not establish all DAFi primitive cells or physical grain counts. See
`docs/experimental-grain-search.md` for bounds, equations and limitations.

The opt-in rotation package now has a portable intensity-reduction coordinator.
Rust owns exact profile masses with fused centre/width derivatives, conditional
GLS/fixed weak filters, slab response, and empirical value/Jacobian passes.
Typed Python orchestration separates explicit detector/polarization assumptions,
strong shape choices, declared sensor centroid maps, equivalent-family scaling
and optional observation-error floors. Canonical snapshots and immutable
provenance-keyed caches replace data-specific imports for the reusable path.
Dioptas owns image orientation, masks, recipe persistence, UI and distinct
aperture/profile exports; external SHELXL is an optional local validator.
Strong covariance near sharp finite-grid boundaries remains conditional and
can differ from historical finite-difference estimates. Models and correction
choices are experimental; no atomic/U target selects them. See
`docs/experimental-intensity-reduction.md` for equations, contracts and replay.


An opt-in FFT ensemble now pools bounded geometric searches before training-only
candidate consolidation and joint grain reassignment. Short-vector and smaller-
volume searches can recover proposals suppressed by a single long-vector search.
A positional evidence rank accounts for reciprocal-lattice density; symmetric
core overlap avoids deleting a compact candidate merely because a larger cell
contains its points. The original single-search API and native ownership kernels
are retained. Validation still uses a frozen pool and the same support, null-
rotation and positional gates. No DAFi labels, supplied material/cell or intensity
reference enters discovery. See `docs/experimental-grain-search.md` for the fixed
parameter grid, equations, budgets and measured limitations.

The experimental panel adapter now accepts an optional application-owned batch
pixel coordinate map. Measured coordinates are mapped before native reciprocal
conversion; predicted plane coordinates and both fused analytical Jacobians are
chained back to image coordinates. The affine path and Rust kernels are unchanged.
Detector-pose refinement retains the fixed map. Dioptas owns the explicit CrysAlis
response, serialization and nonunique boundary inverse policy; no camera model
enters phasesmith-core. Equations, derivatives and validation are documented in
`docs/experimental-pixel-coordinate-map.md`.

The opt-in rotation workflow now supports indexed-position detector calibration.
A rigid detector pose and crystal orientation are refined with a declared fixed
reference cell or candidate cell lengths (angles fixed). Rust evaluates Ewald
positions and both analytical Jacobians in one pass; optional SciPy orchestration
applies exact physical parameter chains. Independent NumPy equations, finite
differences, known-pose recovery and complete-family holdouts validate the path.
Dioptas owns reference-cell import/axis mapping, review, original-calibration
preservation, PONI export and downstream lattice rechecks. Intensity references
and atomic models do not enter geometry fitting. Detector roll remains relative
to the fixed spindle gauge. See `docs/experimental-crystal-geometry.md`.

The opt-in rotation workflow now includes a structure-independent relative scan
scale fitted from explicit equivalent-family labels. A deterministic Rust Huber
log-linear fit eliminates family intensities; a fused signed application returns
analytical scale/angle partials and separate correlated model uncertainty. Python
orchestrates symmetry labels and held-out masks; an independent NumPy reference,
known-scale recovery, finite differences and a many-family benchmark validate the
numerics. No atomic model or external intensity reference enters scale fitting.
The enstatite study improves R1 with local profiles and scan scaling but still
shows a non-positive displacement tensor in a conditional training-only structural
check; this is experimental and is not an automatic validated intensity correction.
See `docs/experimental-scan-scaling.md` for the gauge, angular support, covariance
limits and the distinction from a complete absorption/scaling pipeline.

The opt-in profile follow-up adds local scattering-plane pixel Jacobians with
exact parallelogram/voxel intersections and fused analytical centre partials.
Strong references can teach nonnegative mixtures of finite, normalised spatial
and rocking-width variants. Target shapes are then frozen before fitting signed
intensities; fitting shape parameters separately to weak observations is guarded
as a training diagnostic because known-count trials demonstrate positive bias.
The default fixed-profile API remains available alongside signed summation.
Independent NumPy equations and known-count/null controls pass. A new
reflection-family holdout across two TIFF scans gives median differences from
DIALS of -5.9% and +5.5%, but P1 still fails the declared per-reflection agreement
gate (90th-percentile absolute difference 39.0%); P2 passes at 15.0%. This changed
holdout protocol is not a direct before/after comparison with the original 40
pairs. DIALS is an external comparator, not intensity truth. A limited archived
structure check is encouraging but reaches a displacement-parameter bound and
does not establish structure solution or reliable absolute intensities. All
methods remain experimental. See `docs/fe7c3-profile-bias-investigation.md` for
controls, the corrected CCD adapter, performance and the remaining work.

The preceding opt-in increment implemented local empirical 3D reflection
profiles in detector x/y and a signed geometry-scaled rocking coordinate. Strong
training observations learn
finite, normalised voxel masses; exact box overlaps conserve pixel/exposure
mass and supply analytical position derivatives in the same pass. Profile and
signed summed counts remain distinct, with coverage/model-quality flags and
conditional shared-background covariance. Independent NumPy equations and
known-count fixtures precede a frozen-geometry, held-out Fe7C3 TIFF comparison
against isolated DIALS. The measured comparison currently fails the intensity
agreement gate (40 reserved strong pairs, median relative difference -43.8%);
that result is preserved as historical evidence. This detector-coordinate model is not labelled equivalent
to DIALS's reciprocal-space model; no DIALS runtime dependency is introduced.

Opt-in reflection summation additionally offers a guarded, robust local background
plane. Rust owns clipping, least squares, conditional covariance propagation and
signal-boundary diagnostics; an independent NumPy reference checks known-count
injections and pixel derivatives. Signal supports remain fixed for weak/negative
measurements: no per-pixel significance threshold defines their integrated counts.
Background coefficients, rejection and boundary diagnostics remain inspectable.
The original flat-background estimator and default library results are preserved.
This is detector-count extraction, not crystallographic intensity correction.
The TIFF support study does not justify a universal aperture radius. A source
comparison with DIALS, XDS, MOSFLM and public CrysAlisPro documentation identifies
locally learned 3D reflection profiles as the next experimental milestone; it is
implemented experimentally in the separate profile increment described above. A separate external DIALS probe
now produces actual summed/profile intensities on the TIFF data without entering
the production dependency graph. See `docs/single-crystal-integration-methods.md`.

The experimental integration-quality report now adds image-by-image measurement
counts, mask/background/overlap diagnostics, positional offsets and explicit
angular-endpoint/alternate-window sensitivity. It preserves raw measurements and
provides advisory actions without fitting scales or assuming intensity symmetry.
The Dioptas consumer links flagged images to their individual measurements. See
`docs/experimental-integration-quality.md`; no powder numerical behavior changes.

The opt-in single-crystal workflow now adds prediction-driven rotation-reflection
integration through a reusable PhaseSmith API. Rust aperture evaluation and
ordered frame reduction return signed finite-support counts, conditional errors,
per-frame rocking records and explicit coverage/overlap flags. Dioptas supplies
loading, masks, selected cells and inspection. No profile extrapolation, partiality
or crystallographic intensity correction is implied. See
`docs/experimental-reflection-integration.md`; powder/release scope is unchanged.

The opt-in experimental omega background engine now lives in PhaseSmith Rust,
with an independent SciPy equation reference and a pixel-major NumPy API under
`phasesmith.experimental.omega_background`. A reusable private pool fits all
contiguous valid intervals in one tile call, preserving per-pixel reduction
order across worker counts. Dioptas owns angular interpretation, seat-mask
preparation/cache, fixed masks, temporary staging, progress/cancellation and signed
exports. Its normal calculation retains a session correction with the original
files/settings; TIFF export is a separate explicit action that reuses the cache.
Memory-limited sessions may retain a temporary work file rather than image exports.
This replaces its former local C kernel; default powder APIs and
standard native builds remain unchanged. No peak-model derivative API is added
for this preprocessing operation. See `docs/experimental-omega-background.md`
for equations, validation, migration differences and measured performance.
The same arPLS equations additionally support a reciprocal banded LDLᵀ solve,
removing factorization square roots and most divisions. The library retains
Cholesky as its default and exposes LDLᵀ explicitly; Dioptas selects LDLᵀ in
its experimental background workflow. Cross-solver rounding differences and
performance are validated and reported rather than claimed bit-identical.

The opt-in rotation adapter accepts an optional per-frame variance loader for
background-subtracted detector arrays. It forwards supplied variances to the
existing native spot detector, snapshots reused loader buffers for parallel
work, and includes variances in frame hashes. Native equations, raw-count
defaults, linking and centroid accumulation are unchanged. Correlated uncertainty
from a fitted background remains outside this independent-pixel model.

The opt-in rotation position hunt now supports bounded parallel frame detection
while retaining serial loaders, ordered callbacks/budgets and unchanged linking
and centroid arithmetic. Default library execution remains serial; Dioptas uses
an adaptive number of native detector workers with frame-indexed previews. A preceding
single-thread threshold-loop optimization hoists clipped supports and avoids
unused variance reads without changing equations or floating-point operation
order. Complete per-frame and linked records are compared byte-for-byte with an
explicitly preserved baseline on synthetic and three measured Fe7C3 scans.
See `docs/rotation-processing-performance.md`; default/powder behavior and release
boundaries are unchanged.

An opt-in primitive/Bravais comparison now follows position-only lattice discovery.
Native Selling-Delaunay reduction retains exact integer basis transformations;
complete low-index reciprocal HNF hypotheses expose excluded reflections rather
than silently discarding them. All 14 Bravais types are searched in bounded
reduced axes, retaining distinct metric constraints and conventional centring.
Training-only constrained positional fits use the existing fused native prediction
partials, with fixed HKLs/populations and descriptive held-out residuals. The
recommendation ranks highest lattice symmetry first within a declared window of
comparable penalised training scores. Stable fit IDs, explicit ranks, score deltas
and ranking reasons are exported; poorer/invalid fits remain visible. Scores stay
within each fixed primitive population and are not symmetry probabilities.
P1 favours an orthorhombic primitive metric but retains its larger inferred cell;
no material/cell prior, proven space group or smaller physical cell is supplied.
See `docs/experimental-cell-comparison.md`. Default releases remain unchanged.

An opt-in **FFT grain search** now centralises position-only lattice discovery
and competing grain ownership in PhaseSmith. Unindexed reciprocal positions
produce inferred HKLs and grain orientations; several orientations may share
one cell. RustFFT and a bounded private Rayon pool live only in the experimental
spots crate. Training-only discovery/refinement freezes the model pool before
reserved rotation-block validation; all candidates, rejected models and proposal
audits remain accessible. Dioptas supplies geometry, tables and selection, with
no runtime dependency on a research checkout. The compact search remains a
comparison method and has an explicit candidate budget. Neither method proves
primitivity, Bravais class or space group. See `docs/experimental-grain-search.md`.
Default builds/releases and the powder numerical core are unchanged.

The experimental rotation workflow can now search a bounded set of integer
lattice axes for an orthorhombic metric setting (P/A/B/C/I/F), retaining the exact
change of basis and any conventional-cell lattice index. An optional six-parameter
orthorhombic fit constrains angles to 90° while retaining training-only selection,
fixed HKLs, fixed instrument and independent held-out diagnostics. Centred settings
restrict predictions to the original reciprocal lattice. This is a metric candidate,
not a proof of primitivity, intensity symmetry, extinction rules or space group.
See `docs/experimental-orthorhombic.md`. Default builds and releases are unchanged.

The experimental detector branch now prioritizes rotation-series **positions**
before integrated intensities: streamed per-frame detection, explicit omega
exposure bounds, deterministic adjacent-frame linking, weighted position
summaries and retained frame observations. An optional immutable per-frame
detection observer supports live inspection without changing the numerical
results or source hashes. Ambiguity/scan-edge flags remain
visible. An optional explicit planar detector/sample geometry now transforms
each observation to reciprocal space before combining tracks. A bounded blind
Fourier search proposes unconstrained cells from positions alone, with separate
training/validation data and supercell checks. It supplies no material prior,
symmetry assignment, proven primitive cell or integrated rocking intensities.
See `docs/experimental-reciprocal-indexing.md` for equations and limitations.
A further optional positional step predicts Ewald crossings in detector x/y
and omega, with fused native analytical derivatives. Robust training-only
refinement updates the unconstrained cell and orientation while holding the
instrument fixed. Initial/refined predictions and independent validation
residuals remain available even when validation does not improve; there is no
intensity, extinction or space-group model. See
`docs/experimental-spot-prediction.md` for equations, boundaries and validation.
File/angle import and inspection stay in opt-in Dioptas, outside this
library and the standard release path. See `docs/experimental-rotation-positions.md`.

An isolated experimental branch adds structure-independent 2D detector spot
detection and signed finite-aperture intensity extraction. The separate
`phasesmith-spots` numerical crate is exposed through the
`experimental-spots` feature only; default facade/extension builds omit it.
The Python API is explicitly under `phasesmith.experimental`, and Dioptas is
an external opt-in consumer. No standard release or powder API changes are
authorized. See `docs/experimental-spots.md` for support, noise, flags,
independent-reference validation and the distinction from corrected I(hkl).
This user-requested detector experiment is separate from the deferred public
one-dimensional powder peak picker and introduces no GUI into the library.

The final 0.7 release gate exposed NumPy 1.26 AVX-512 trigonometric rounding
in the independent FCJ/component reference. Four controlled hosted probes
reproduce both failures with vector trig and pass with scalar libm or with
AVX-512 disabled. The reference now evaluates its small angular arrays through
scalar libm; Rust kernels, equations, support, and assertion tolerances are
unchanged. CI retains default SIMD dispatch and adds a subprocess comparison
of default versus disabled AVX-512 reference geometry. See
`docs/fcj-reference-cpu-dispatch.md` for evidence and the numerical boundary.

A fresh post-merge review repairs the complete native Rietveld result contract:
selected/trial calculations retain fixed axial derivatives in their existing
fused pass. Returning the accepted state requires no extra profile evaluation,
including at a runtime limit. Optimization and accepted histories retain their
existing arithmetic. Refinement options and checkpoints now reject invalid
profile-accuracy policies at validation/load time. CW and TOF Pawley constructors
also reject unsigned HKLs that overflow signed storage instead of wrapping their
values. No equation, support boundary, Pawley solver, or persistence format
changes in this follow-up.
Validation integrals and tests select the supported NumPy trapezoidal-integration
name on both 1.26 and 2.x; CI now exercises the declared NumPy minimum explicitly.

The 0.7.0 release reconciles main's 0.6 API cleanup and bounded
validation-download retries with the validated develop/Pawley implementation.
Module-owned imports and warning-backed 0.5 compatibility aliases are retained;
Pawley modules are explicit refinement exports. Existing snapshots remain
immutable, with new 0.7 snapshots covering the combined surface. This
reconciliation changes no numerical equations or persistence formats.
See `docs/release-candidate-0.7.md` for the final release review, validation
evidence and authoritative publication links. Native implementation remains
unchanged by final reference/documentation preparation. The package passes 982 Python
tests on NumPy 1.26 and 2.x, 368 Rust tests/doctests and the cross-platform
distribution dry run; the frozen QARR 1h failure remains.

Pawley integration is validated after the 2026-09-21 repair. Scale-relative
active-set feasibility fixes the Intel dense-solver boundary failure without
relaxing convergence or physics; cached norms preserve performance and exact
scientific records. Installed validation assets resolve from explicit paths or
the working directory. Both architectures pass 970 Python tests; full Rust,
measured Pawley gates and the cross-platform distribution dry run pass.
See `docs/pawley-boundary-feasibility.md` and `docs/pawley-repair-20260921.md`.
The shared FCJ correction and Pawley APIs accompany the consolidated develop
baseline. Feasible-width research remains outside this integration; the API
cleanup is now reconciled in 0.7. The 970-test counts above describe the
earlier Pawley integration checkpoint, superseded by the fresh review.

Repository consolidation (2026-09-21) preserves the original working state,
Pawley history and feasible-width research on separate remote branches. The
agent skill now ships as version-matched package resources with CLI discovery,
generated documentation and CI drift checks. No model-provider dependency is
introduced.

The complete CW solver and independent Python fallback now verify candidate
stops with undamped local information. Coordinate recovery shares one ordinary
line-search trial allowance across ranked directions, rather than multiplying
it by the parameter count. Exhausting this bounded search reports stagnation;
actual caller runtime guards retain their own stop reasons. Scientific gates,
physical models and budgets are unchanged. The canonical QARR 1g entry point
has a dedicated regression test, and the fixed eight-case assessment remains
the promotion boundary. See `docs/refinement-bounded-recovery.md` and the
`docs/repository-consolidation.md`. The feasible-width option remains solely on
its preserved experimental branch and is not added to the committed public API.

The live pinned GSAS-II Pawley optimizer comparison now covers fixed-cell and
six-cell-parameter fits with 53 families, isolated areas and overlap sums.
The revision-gated initializer, immutable fixture, compiled-binary provenance
and executable comparison are committed separately from production code.
Declared engineering agreement checks pass; a stricter 1e-5 profile-equivalence
check remains false. A subsequent black-box diagnosis isolates the mismatch
to the pinned FCJ numerical profile evaluator and finite cutoff policy: native
quadrature matches a converged independent reference, oracle profiles with their
cutoffs reconstruct the histogram, and an independent linear solve attributes
the overlapping-area difference to the profile basis rather than the optimizer.
A subsequent direct, high-precision angular-integral audit exposed a shared
native/NumPy change-of-variable error: the geometric denominator must contain
`1+z²`, not `sqrt(1+z²)`. Values and analytical height derivatives are corrected
together. Its effect in the Pawley fixture is about 1e-9 relative at sampled
points and does not account for the much larger pinned-oracle mismatch.
The new angular reference uses adaptive tanh-sinh integration without the
production substitution; mpmath is a test-only dependency.
See `docs/pawley-profile-diagnosis.md` for the before/after equation audit.
Direct inspection of the pinned GSAS-II call path subsequently confirmed
single-node FCJ quadrature for every reflection in this fixture. Rebuilding
the unmodified external routine in double precision preserves the discrepancy
and reproduces the predicted single shifted intrinsic peak. This identifies
under-integration as the dominant oracle cause, with secondary roundoff and
support differences; no GSAS-II implementation was copied into PhaseSmith.

Native project format 7 now adds typed CW and joint TOF Pawley analysis ownership and a lossless
mixed-method `ProjectBundle` API. Shared histogram arrays are stored once;
cell-only Pawley domains require no atom models. All existing native analysis
families and checkpoints survive mixed load/save, formats 1–6 migrate, and
Python/native save/load/resume tests verify exact accepted profiles and history.
Standalone Pawley formats remain compatible. The latest validated numerical
work is integrated into develop and retained in the release candidate.

Pawley now provides bounded CW family-area refinement in dense and matrix-free
modes, analytical cell/profile/background derivatives, exact ties, independent
NumPy validation, accepted-state restart and standalone format-2 persistence
with format-1 migration. The matrix-free measured release gate passes unchanged
scientific thresholds and budgets: sucrose converges in 137–138 seconds with
Rwp 0.066054, and LaB6 converges with Rwp 0.284869. Repeated arrays and histories
are exact. Sucrose convergence is explicitly local to the hard-support objective;
positive finite cutoff jumps define one-sided feasible directions, with
conservative exclusions for interacting or cancelling events. Support values
and endpoints are unchanged. Matrix-free mode omits global rank/covariance;
dense mode remains the small-problem reference. The separate measured LaB6
cell/profile gate passes at Rwp 0.2184. Fixed CW spectra now use one area per family, normalized fixed detected weights,
component-union domains and native analytical chains. Standalone and shared
bundles retain spectra; independent derivatives, pinned doublet profiles and
measured ceria comparisons are covered. TOF now reuses the same bounded native
solver with microsecond density observations, bank-local areas/calibration/profile
and backgrounds, and symmetry-constrained shared cells. Analytical chains,
nonuniform support, uncertainty conventions, a DIFC/cell gauge guard, atomic
restart, native format-7 bundles and independent NumPy/pinned-profile checks
are implemented. The measured POWGEN gate converges at Rwp 0.21680; the joint
three-bank nickel gate converges at Rwp 0.02099 with all banks below 0.03.
Upstream committed numerical work through `9afd808c` is reconciled. See `docs/pawley.md`,
`docs/tof-pawley.md`, `docs/pawley-validation.md` and the completion audit.

Refinement development priority: validate practical bounded workflows across
the fixed assessment panel. Preserve fit/composition gates, truthful stops and
acceptable cost together. Further QARR-specific tuning and feasible-width
development remain paused. Historical timings and convergence claims below
refer to their stated implementation checkpoints.

The pre-consolidation original-recipe convergence audit identified a termination
weakness: the general solver can report convergence when repeated damping
increases make its proposed step tiny, even though valid descent directions
remain. Large-budget original-width fits report Rwp 19.87759% on 1g and
20.10151% on 1h, with repeatable full-workflow timings of 1.720/2.413 seconds
at eight workers. These are times to reported convergence, not stationarity
certificates. Tighter 1h fits can stall at worse results; feasible coordinate
probes demonstrate the issue. See `docs/qarr-convergence-audit.md`. That audit did not itself change production
behavior; the bounded stopping repair described above now addresses the
reported termination weakness, with its measured runtime cost retained.

The subsequent QARR holdout investigation separates budget exhaustion from
width decomposition. The common diagnostic's original-width recipe converges
on 1h at Rwp 20.10151% with a 500-iteration allowance; the already-established
empirical frozen-background recipe reaches 19.44367%, with all stages converged
and maximum composition error 1.55045 percentage points. Tight-tolerance and
one/eight-worker controls agree. Joint background lowers Rwp but fails the
unchanged 2-percentage-point composition gate. These exploratory results do
not replace the frozen native holdout validator or make 1h an untouched test
for future tuning. See `docs/qarr-holdout-investigation.md`.

Rietveld rejected-step recovery now retains the unchanged accepted-state
Jacobian and restores damping to at least the caller's initial value after a
failed backtracking sequence. The independent Python and native structural
solver share the damping rule; forward equations, bounds and convergence
criteria are unchanged. A native-only benchmark trace measures time to Rwp
targets while keeping final convergence, QPA and profile checks separate.
On the QARR empirical frozen-background recipe, total evaluations fall from
376 to 251 with identical final profiles, parameters and covariance at Rwp
19.21994%. The joint-background diagnostic also retains its solution but its
larger composition error; it remains a separate recipe. Unchanged PbSO4 and
QARR reference checks pass, while the existing QARR 1h holdout failure remains.
See `docs/refinement-quality-recovery.md` for the contract and validation.

Opt-in `EmpiricalGaussianConvention` now anchors a caller-selected phase's RMS
strain and transfers shared Gaussian variance into U while preserving the
starting profile. It removes the isotropic U/strain redundancy using existing
Rust kernels and fixed constraints. Python input/checkpoint, project metadata,
readiness and fit reports preserve the explicitly empirical interpretation.
Defaults and physical width domains remain unchanged. On QARR the bounded
fast-profile recipe improves Rwp from 19.9888% to 19.6776%; a longer default-
profile fit reaches 19.2199% but is substantially slower. The longer fast case
fails the termination gate. See `docs/empirical-gaussian.md` for equations,
constraints, reproducible measurements and interpretation limits.

The rietx solver audit identifies width parameterization, coupled instrument/
sample variance, early background freezing and rejected-step recovery as the
next fit-quality priorities. A PhaseSmith-only diagnostic reaches 18.9115%
QARR Rwp with convergence at all stages, but takes many more evaluations and
worsens maximum phase-fraction error to 1.863 percentage points. It is not a
new default or an equal-quality speed win. See `docs/rietx-solver-audit.md`;
the diagnostic did not change production solver behavior or validation recipes.
The later recovery change above is evaluated separately.

A subsequent same-objective comparison feeds PhaseSmith's residuals and
analytical Jacobian to SciPy TRF with identical starting values, scaled
physical parameters and bounds. A direct solver replacement does not recover
rietx's advantage: SciPy repeatedly hits PhaseSmith's coupled instrument-width
domain and stops early. Fixing instrument parameters gives close solver
agreement. Finite-difference and objective parity checks pass; details and
controls are in `docs/solver-isolation.md`. Prioritize identifiable, feasible
width coordinates before deciding on an optimizer replacement.

Structural powder intensities now average Friedel mates before profile
accumulation, correcting merged-family intensities for non-centrosymmetric
structures with anomalous scattering. Individual complex structure-factor
APIs retain their representative-reflection meaning. Dense, selected and
matrix-free intensity derivatives follow the same average, including custom
provider derivatives. This is an unconditional powder-physics correction,
separate from optional numerical accuracy controls. See `docs/powder-friedel.md`.

Opt-in CW `ProfileAccuracy` now separates conservative tail-area support from
lower-order small-span FCJ quadrature. The Rust kernel retains the physical
axial correction and fused analytical derivatives; established support and
quadrature remain the default. The policy travels through structural engines,
native/Python refinement, project persistence and checkpoints. Resume rejects
a changed policy. Rejected native trials omit unused fixed-axial rows, with
complete final diagnostics retained. Equations, boundaries, supported paths
and validation are in `docs/profile-accuracy.md`.

The post-optimization rietx workload audit identifies a major numerical-policy
difference: rietx 1.4.0 skips FCJ convolution for every reflection in the QARR
case and uses smaller area-based windows. Controlled interventions and native
sampling are recorded in `docs/rietx-workload-audit.md`. They motivate validated
small-span acceleration and cheaper rejected trials; default numerical
policies and the existing equivalence gates remain unchanged.

The second refinement performance pass batches four FCJ sample evaluations
without reordering any sample's quadrature sum, assembles selected structural
Jacobian rows contiguously, and reuses a unit-scale profile basis for native
scale-only fits under the existing dense-memory ceiling. Bounds, constraints,
step limits and checkpoint rules remain in the general solver. Python receives
native phase diagnostics; missing fixed-axial rows are completed once on the
native side, while scale-only bases already retain those rows. Numerical
contracts and measurements are in `docs/refinement-performance-round2.md`.

The QARR performance follow-up adds selected native structural derivatives,
fixed-axial derivative elision, a bounded per-refinement geometry/scattering
cache, value/local-derivative backtracking without dense structural assembly,
native fixed-spectrum Python dispatch, and residual-checked small Cholesky
solves with the established CG fallback. Support and physical equations are
unchanged. Algorithm contracts and validation are in
`docs/refinement-performance.md`; fit trajectories may change with the solver's
floating-point summation, so real-data gates and determinism remain mandatory.

The rietx capability comparison now has a first implemented reporting slice:
native, mask-aware residual localization with a typed Python `FitReport` and
`RietveldProject.fit_report()`. Reports preserve the fit's actual weights,
rank intervals by chi-square contribution, disclose missing covariance/rank,
and retain advisory-only review actions. Residual location is not physical
attribution; parameter-gain ranking remains future work. The equations and
boundaries are in `docs/fit-report.md`. The existing measured IUCr QARR 1g
workflow is the primary rietx 1.4.0 performance comparison, with retained
quality gates and explicitly failed cross-implementation equivalence gates.
Controlled kernel and synthetic-fit measurements are supporting diagnostics.
Results and limitations are in `docs/rietx-comparison.md`; neither library
has a blanket speed advantage.

PhaseSmith 0.5.0 was the published baseline before the 0.7.0 release. The
completed library includes Python-free CW and TOF Le Bail/Rietveld workflows, public Python
facades, native project persistence, and facility-neutral single-/multi-bank
TOF composition with explicit calibration, correction, background-domain, and
provenance contracts. The TOF roadmap through implementation unit 38d is
complete; additional beamline adapters or absorption, extinction, and texture
models are new evidence-driven scope rather than unfinished POWGEN work.
Version 0.4.0 additionally includes orthorhombic Stephens microstrain with
analytical derivatives, broader Hall-expression import, bounded parallel
value-only structure factors, an XRD-Rust comparison, and the reviewed citrate
conversion-fidelity campaign.

The 0.5.0 development line adds an application-neutral automation boundary on
top of persisted typed Rietveld projects. Inspection and planning are
non-numerical; plans are bound to exact project bytes and finite execution
budgets. A human or external AI may propose an explicit cumulative staged
recipe, but PhaseSmith admits only caller-authorized parameters, validates the
complete constraint graph, requires exact plan-ID approval, rejects stale
inputs and unintended overwrites, and retains complete audit outputs. Model
providers are optional external advisors and never become numerical-core or
runtime dependencies. Raw powder/CIF inspection deliberately reports missing
physical metadata instead of guessing a runnable project.

Plans now carry a sanitized scientific advisor context containing model,
parameter, bound, constraint, and experiment summaries without paths or raw
pattern/CIF contents. External proposals can be deterministically linted for
their strict contract and risky scientific ordering. Completed workflow
directories can produce digest-bound stage/parameter/identifiability/residual
reviews, then seed a new non-executing plan from the saved accepted project
state. The child plan records parent/result/review lineage, uses a new output
directory outside the parent audit directory and a new plan ID, and requires
fresh approval; no AI callback runs
inside a numerical solve and no deterministic review auto-accepts a fit.

The 0.6.0 line is the deliberate pre-1.0 Python API cleanup. It aligns the
top-level, I/O, and refinement aggregate namespaces with their documented
module ownership, keeps warning-backed compatibility aliases until 1.0 for moved 0.5
imports, and records the resulting exported names and signatures in a new
immutable snapshot. It changes no numerical equation, tolerance, support
boundary, default representation, or persistence format.

External handoff is now a first-class path-free packet rather than a manual
selection of plan fields. A second packet composes the digest-bound prior
review with the next child plan's sanitized advisor context. Proposals retain
structured human/model/software provenance and are bound to the advisor-packet
digest; stored plans are rebuilt and compared before CLI execution so lineage
cannot be dropped. All automation records have published CLI/static JSON
Schemas. The repository also carries a compact agent skill, a runnable offline
two-cycle example, and synthetic high-risk recipe cases; these remain external
orchestration and add no model-provider dependency.

The remainder of this section is a chronological implementation record. Terms
such as “next slice”, “pending”, and “follow-on” describe the checkpoint at
which their paragraph was written; the final current-state summary and next
milestone are maintained in `IMPLEMENTATION_PLAN.md` under “Recent completed
milestones and next release gates”.

Implementation units 0 through 10 in `IMPLEMENTATION_PLAN.md` are complete as
of 2026-08-05. The repository now includes validated symmetric TCH, CW
U/V/W/X/Y, FCJ, wavelength-component, sample-physics, multi-phase, neutron CW,
and neutron TOF calculation paths; shared refinement infrastructure; first-class
Le Bail extraction; versioned JSON+NPZ persistence; and an application-neutral
NumPy interoperability boundary. GSAS-II fixtures remain external-oracle
products, and neither GSAS-II nor SciPy is required for normal installation.
Third-party reflection physics is supported through an explicit versioned batch
provider and persistence-codec contract; arbitrary intrinsic line-shape plugins
remain a future compiled-extension boundary rather than a promised Rust ABI.
Implementation units 11 and 12 add Rust-owned general unit-cell geometry, P1
complex structure factors with dense/JVP/VJP derivatives, exact symmetry,
special-position expansion, systematic absences, metric constraints, and
bounded d/Q/CW/TOF reflection generation. Typed NumPy APIs and independent
NumPy references cover both slices. Unit 13 adds optional parser-independent
CIF models/import plus fixed-cell CIF-to-Le Bail. Unit 14 adds independently
sourced native non-resonant X-ray and coherent-neutron scattering tables,
prepared value/derivative kernels, and a versioned vectorized provider API.
Unit 15 adds general-symmetry integrated intensities, explicit neutral and
Bragg--Brentano LP corrections, a fused structural CW pattern kernel with
analytical JVP/VJP products, scriptable `RietveldPhase` models, version-2
persistence, independent NumPy comparisons, and combined Rust/Python
benchmarks. At that checkpoint, full structural refinement remained a
follow-on unit and the live pinned GSAS-II P1 and reflection-behavior fixtures
had not yet been run in the working environment.

A separate-environment benchmark now compares the same symmetric CW profile,
finite support, and analytical derivative outputs against the pinned GSAS-II
profile interface. It validates numerical agreement before reporting timings
and does not treat the result as a complete-refinement speed comparison. A
second numerically gated benchmark starts with an equivalent typed P1 neutron
crystal and compares structure-factor values, integrated intensities, and the
controlled structure-to-symmetric-CW calculation. Project construction,
reflection generation, file I/O, and process startup remain outside its timed
regions; its scope likewise is not a complete-refinement comparison.

Implementation unit 16 is complete. It adds setting-aware physical lattice
parameters for every crystal system, conservative bounded monochromatic
reflection domains with stable Miller-family intensity transfer, analytical
CW/TOF lattice geometry chains, accepted-step Le Bail topology regeneration,
and a one-call CIF-backed request constructor. Persistence format 3 retains the
dynamic phase/domain restart contract while loading formats 1 and 2. The full
gate passes 367 Python tests (one unavailable external-oracle case deselected),
55 Rust tests, strict linting, and a realistic release benchmark. At that
checkpoint, the live pinned GSAS-II lattice-perturbation fixture had not yet
been run in the working environment; equations and normal operation contain no
GSAS-II dependency.

PhaseSmith 0.4.1 was released on 2026-08-14 and is licensed under the MIT
License. The
architecture and delivery gates for CIF import, the remaining native
crystallographic calculations, and structure-factor-based Rietveld
orchestration are specified in `docs/crystallography-plan.md`.

Unit 17 begins with a mandatory refinement-runtime safety shell. All iterative
methods expose structured events, thread-safe cooperative cancellation,
explicit execution budgets and termination reasons, and checkpoints of the
last accepted state. Terminal key handling is an optional adapter; numerical
code never reads standard input or configures global logging.

That safety shell is now also owned by the Python-free
`phasesmith-workflows` crate: native solvers and future application consumers
share the same event, budget, typed-checkpoint, resume-counter, and cross-thread
cancellation contracts without embedding CPython. The Python runtime remains
the scripting implementation and differential oracle until solver delegation.

The first native solver built on that shell is the fixed-reflection Le Bail
workflow. It now performs complete non-negative intensity extraction,
mask/uncertainty handling, residual histories, unresolved-column diagnostics,
checkpoint/restart, cancellation, and display-series calculation without
Python. Its iteration history and final intensities are differentially checked
against the scripting workflow. Parameter motion is layered onto this fixed
extraction core in independently reviewed migration substeps.

Fixed-topology native Le Bail now also refines analytical CW profile, phase
scale, and independent reflection-position parameters through the native
constraint graph. Its bounded regularized solve, backtracking, covariance,
parameter audit trail, evaluation accounting, and profile-aware restart state
are Python-free; accepted position-refinement histories agree with the Python
oracle. The workflow crate now additionally owns the setting-aware lattice
parameterization, finite bound box, analytical CW geometry chain, conservative
guarded reflection generator, and stable-ID intensity-transfer contract needed
for dynamic topology. Native Le Bail now uses that foundation directly:
lattice variables join the constraint graph and analytical pattern Jacobian,
accepted cells regenerate topology with stable-ID intensity transfer, changed
domains are reported, and checkpoints may resume changed reflection lists only
under an identical guard contract. Synthetic recovery, topology-changing
restart, and complete iteration histories agree with Python. The full built-in
Le Bail scientific workflow is therefore Python-free; delegating the Python
scripting façade to it is the next adapter step.

Native Rietveld migration has begun with an owned Rust calculation boundary in
`phasesmith-workflows`. A Rust or future Tauri adapter can now submit validated
observations, monochromatic experiment state, ordered structural phases, and
sample-physics contributions and receive display-ready phase/profile/background
arrays, crystallographic intermediates, and residual metrics without importing
Python. The operation revalidates adapter-visible records immediately before
calculation, remains bitwise deterministic across configured worker counts,
and is differentially checked against the existing Python structural workflow.
This boundary does not yet replace the Python Rietveld optimizer; native
parameter motion, matrix-free solving, dynamic reflection topology,
checkpoints, recipes, and scripting-facade delegation follow as independently
reviewed steps.

The native Rietveld boundary now also owns stable structural parameter
identities and analytical derivative transforms. Explicit phase/site IDs feed
the shared `ParameterSet`; setting-aware lattice variables and deterministic
special-position coordinate bases map exactly into the engine's full cell/site
layout, and reverse products map back to the same physical order. Native tests
cover identity validation, general/partially constrained/fixed sites, complete
key and bound packing, invalid shapes, and the transformed JVP/VJP adjoint
identity. Instrument, background, position, and sample-physics parameter
families remain part of the subsequent solver integration step.

A reusable prepared Rietveld objective now composes that structural layout with
the native multiphase engine. It provides JVP, VJP, uncertainty/mask-weighted
gradient, and damped matrix-free normal products while keeping fixed background
out of derivative columns. Layout identity checks reject stale phase, site,
symmetry, or anisotropic-site state before evaluation. Analytical-column and
adjoint tests lock down the operator that the bounded conjugate-gradient solver
will consume next.

The inverse structural transform is now implemented too: bounded physical
parameter vectors rebuild validated phase definitions through stable identities
without exposing native derivative-array offsets. It covers lattice, general
and special-position coordinates, occupancy, isotropic displacement, and phase
scale, and rejects invalid values before structural preparation.

Structural Rietveld refinement is now Python-free for the selected structural
families. The native workflow uses scaled matrix-free conjugate gradients,
bounds-aware backtracking and damping, the shared runtime
budgets/cancellation/events, accepted-only history, and typed restart
checkpoints. Rust integration tests recover synthetic phase
scale and a cubic cell, reproduce uninterrupted history after restart, and
exercise cancellation, evaluation exhaustion, empty masks, corrupt restart
state, and host event/checkpoint delivery. Guarded dynamic phases now regenerate
HKLs and multiplicities after bounded cell motion, transfer every sample-physics
array by stable reflection ID, record accepted topology changes, and allow
changed-topology restart only for the identical domain contract. The remaining
non-structural parameter families are the next slice.

Native Rietveld requests can now own an optional analytical background in
addition to the pattern's fixed supplied background. Calculation validates the
model on the observation grid, adds it exactly once, and exposes the combined
background separately from the structural profile. This keeps background state
inside the same Rust request used by refinement and external applications.

The complete native Rietveld parameter/objective layer now composes the
matrix-free structural tangent with selected CW U/V/W/X/Y, wavelength,
zero/sample-position, and analytical-background columns. Stable physical keys,
bounds and scales use instrument/background/structural order; accepted value
installation updates the full owned request, including wavelength-coupled
correction/domain state. Rust tests check centered differences and the complete
JVP/VJP adjoint identity. Built-in phase-owned sample-physics records now add
isotropic size, isotropic microstrain, March--Dollase, and ordered composition
without Python. They use exact corrected reflection geometry, re-evaluate after
dynamic topology changes, and contribute analytical scalar, position, and all
six reciprocal-metric cell chains to the general objective. Native finite
differences, adjoint products, and a configured Python differential cover the
new values and derivative arrays. The complete constraint-aware solver now
maps that full physical layout through fixed, affine, and multi-source graphs
into frozen scaled-free coordinates and the matrix-free normal operator. Its
restart state owns the accepted request and exact selection/bounds/constraint
contract. Final bounded diagnostics report weighted rank and correlations and
produce full physical covariance only for a full-rank free normal matrix.
Recovery, cancellation, restart, singular-diagnostic, covariance-chain, and
configured Python-history tests cover the boundary. Native explicit recipes
now execute cumulative complete-solver stages while promoting only
policy-accepted physical states. Their deterministic intelligent planner stays
within the caller-authorized maximum, discloses rationale, validates the full
constraint contract before work, disables intermediate covariance, and stops
safely on cancellation or a rejected termination. Native execution and
planner-oracle tests cover these contracts. Delegating the built-in Python
facade to the shared native refinement boundary remains next.

Unit 17 now includes the first full monochromatic CIF-backed Rietveld vertical
slice. Typed profile/background/phase/lattice/site parameter families map
through fixed/affine/multi-source linear
constraints into the Rust structural pattern JVP/VJP, and damped Gauss--Newton
is solved without a dense sample Jacobian. The accepted-state runtime covers
structured logs, cancellation, budgets, checkpoint callbacks, restart, and
rank/correlation diagnostics. Multiple phases and built-in X-ray and neutron
scattering use the same interface. Persistence format 4 retains the request,
guarded structural domains, options, and restart checkpoint while loading
formats 1--3. The internal gate currently passes 410 Python tests (one optional
external-oracle case deselected), 55 Rust tests, strict Ruff, formatting, and
Clippy.

Unit 18 starts with practical real-pattern usability while preserving the
numerical/refinement boundaries. Its first slice is a native, deterministic
Smooth Bruckner background estimator compatible with the pinned MIT-licensed
xypattern implementation used by Dioptas-style workflows. It is exposed as
plain-array preprocessing under `phasesmith.background`; it never imports
Dioptas or xypattern and is not inserted into the differentiable refinement
model. The raw smoothed envelope and the optional Chebyshev-compressed
background are both scriptable, and the result can be passed directly into a
`PowderPattern`.

The background slice is implemented and independently reviewed. The pinned
xypattern 1.2.3 behavior agrees to `8.89e-16` for the raw smoother and
`8.31e-14` for the complete default Chebyshev pipeline in an isolated
environment. A 20,001-sample, 50-iteration release benchmark measures 3.61 ms
median for native smoothing and 13.15 ms for the complete public subtraction
pipeline.

Unit 18 is complete. Structural CW calculations now refine wavelength, zero
shift, and typed Bragg--Brentano specimen displacement in the fused Rust path;
the wavelength derivative includes both position motion and
Lorentz--polarization intensity dependence. Built-in size, microstrain, and
March--Dollase parameters are phase-owned refinable scalars. The
March--Dollase lattice chain includes the analytical reciprocal-metric
derivative and remains isolated between phases.

Power, Chebyshev, point-interpolated, broad normalized-Gaussian amorphous, and
ordered composite backgrounds implement one analytical protocol. Their values,
checked row-major derivative bases, validation, and coefficient replacement are
now also owned by the Python-free `phasesmith-workflows` crate for shared
scripting/desktop orchestration. The
application-neutral `RietveldProject` facade provides calculate/refine/stop,
checkpoint continuation, format-5 persistence, and finite JSON/plain CSV
reports without replacing the typed request API.

On the development host, realistic release calculations with 20,001 samples,
eight sites, and built-in sample physics take 13.19 ms median for a
423-reflection laboratory X-ray pattern and 8.79 ms for a 257-reflection
monochromatic-neutron pattern. The existing 423-reflection, 29-parameter,
three-iteration matrix-free refinement benchmark takes 145.56 ms median. The
pinned GSAS-II comparison was not rerun because no external checkout was
supplied; its optional harness remains separate from installation.

The complete source and fresh-wheel gates pass 442 Python tests (one optional
external-oracle case deselected), 60 Rust tests, strict Ruff, formatting, and
Clippy.

A cross-cutting real-data validation checkpoint now adds immutable plain-column
and unpacked FXYE input, checksum-pinned external dataset provenance, explicit
offline verification, and Hill--Howard phase-scale-to-weight-fraction
conversion. The official monochromatic APS 11-BM sucrose pattern exercises the
complete background/Le Bail/profile-refinement script and has a committed
measured baseline. The IUCr QARR 1g input is verified but explicitly reports a
blocked full structural Cu K-alpha doublet refinement; the lower-level profile
component kernel is not misrepresented as complete multi-wavelength Rietveld
support. The implementation sequence is frozen in
`docs/real-data-validation-plan.md`.

The pinned sucrose run uses 23,003 samples and 811 generated reflection
families. Both PhaseSmith and the pinned GSAS-II comparison receive the same
fixed Smooth Bruckner array and refine one constant Chebyshev residual from
zero. Both start from the explicit symmetric instrument state U=1.163,
V=-0.126, W=0.063, X=0.173, Y=0 in GSAS units with SH/L=0; the GSAS-II worker
overrides the legacy importer before refinement. PhaseSmith's first-cycle
`Rwp` of 27.80% falls to 14.18%, the background-subtracted
observed/calculated correlation is 0.99070, and all extracted intensities
remain finite and non-negative. The matched seeded GSAS-II run reaches Rwp
14.53% and correlation 0.97300, placing the Rwp delta at 0.00350. QARR verifies its 7,251
samples and Cu K-alpha doublet metadata before returning the expected blocked
status. The complete gate passes 469 Python tests (one optional external-oracle
case deselected), 60 Rust tests, Ruff, Rust formatting, strict Clippy, and a
fresh release-wheel smoke test in an isolated environment.

The fixed-spectrum structural checkpoint removes the QARR capability block at
the calculation and fixed-cell refinement layers. `ComponentRadiation` now
drives component-specific Bragg positions, built-in Lorentz--polarization
values, native fused structural values, and analytical JVP/VJP products. Each
discrete wavelength is one Rust structural batch; Python performs only the
small spectrum-level sum and never orchestrates atoms, reflections, support,
or samples. CIF-backed Rietveld requests generate the exact union of visible
fixed-cell families, refine shared phase/site/profile/background/sample terms,
and report component-indexed reflection diagnostics. Component wavelength and
lattice refinement fail explicitly until a multi-wavelength topology guard is
implemented. Persistence format 6 stores the typed radiation union and loads
formats 1--5. The next real-data checkpoint is the three-phase IUCr QARR fit
and quantitative acceptance assessment.

The following structural-intensity checkpoint adds native fixed X-ray
dispersion offsets and polarized symmetric Bragg--Brentano LP. Fixed complex
`f' + i f''` values are explicit caller-owned wavelength data layered on the
independently sourced Waasmaier--Kirfel baseline; no absorption-edge database or
GSAS-II code enters the runtime. Polarization uses
`[P + (1-P) cos²(2theta)]/[sin²(theta) cos(theta)]`, with `P=0.5` exactly equal
to the existing unpolarized model. Values and cell/wavelength JVP/VJP chains
remain in Rust for monochromatic and fixed-component radiation. Persistence
format 7 stores both typed models while loading formats 1--6.
On the 20,001-sample, 256-reflection, 32-site release benchmark, the fused
baseline takes 770.77 microseconds median and fixed dispersion takes 777.43
microseconds, an observed 0.86% increment. A clean snapshot of the preceding
commit measured 783.78 microseconds for the baseline on the same host, so this
checkpoint introduces no measured baseline regression.

A repository-hardening checkpoint follows format 7 without changing the
scientific model. TOF batches share one immutable quadrature table, symmetric
FCJ profiles allocate only their active node, and structure-factor VJPs reuse
site trigonometric, scattering, and displacement terms. Value-only reflection
paths no longer construct cell derivatives, while exact systematic-absence
checks memoize their bounded cyclotomic polynomials. Native reflection
generation and fused structural calculate/JVP/VJP bindings release the Python
GIL during Rust work. New native finite-difference tests cover the full CW
component/FCJ and sample-contribution chains, and the wavelength correction
derivative no longer divides by a possibly zero correction value.

Rietveld defaults now budget all requested iterations, accepted checkpoints
carry their guarded lattice domains and wavelength state through persistence,
and backtracking reuses the accepted-state special-position coordinate models
instead of repeating their SVDs. Le Bail's quadratic rank-deficiency analysis
is explicit opt-in diagnostics. Persistence format 7 now has a matching schema
and documentation, and generator-source hashes remain recorded provenance
rather than formatting-sensitive test assertions.

The next performance checkpoint adds a bounded reusable structural
linearization. Built-in Rust paths calculate one parameter-major pattern
Jacobian and values per accepted or trial state, then reuse it for every
optimizer JVP/VJP; plugins and requests above the configurable memory ceiling
retain the matrix-free path. The 20,001-sample structural benchmark falls from
148.7 to 38.3 milliseconds median while preserving the final Rwp to 2.6e-11
absolute. QARR stage 2 temporarily requests the matrix-free fallback because
its then-approximate model had a rounding-sensitive flat basin even though
cached and matrix-free products agreed to about 2e-15 relative. After fixed
anisotropic displacement and adaptive FCJ removed those dominant model and
execution changes, three repeated release runs gave identical scientific
records with the cached path. The temporary override is removed.

The structural FCJ checkpoint composes the independently implemented
Finger--Cox--Jephcoat convolution with built-in and third-party vectorized
sample physics inside the Rust support-limited accumulator. Optional
`ConstantWavelengthExperiment.axial_geometry` propagates through monochromatic
and fixed-component structural values, JVPs, VJPs, and reusable
linearizations. Global derivatives are ordered U/V/W/X/Y, axial sample and
detector ratios, then provider parameters; zero geometry exactly preserves the
symmetric values and non-axial derivative rows. Persistence format 8 owns this
optional geometry on the experiment while loading formats 1--7.

The QARR checkpoint now applies the supplied `SH/L=0.002` through the published
equal-height mapping `sample_over_radius = detector_over_radius = SH/L / 2`.
One reviewed release run passes all scientific gates at 19.672% Poisson Rwp,
13.273% unit-weight Rwp, 0.99070 profile correlation, and 1.883 percentage
points maximum QPA error. Exact continuous FCJ currently raises PhaseSmith's
workflow time from about 7.6 seconds to 158.8 seconds on the development host;
FCJ convolution reuse/vectorization is therefore a measured performance
priority before treating that workflow as production-speed.

The fixed-anisotropic displacement checkpoint evaluates preserved CIF U
tensors directly in the Rust general-symmetry value, dense, JVP, VJP, and fused
structural-pattern paths. Tensor components use CIF order
`11,22,33,23,13,12`; each symmetry mate applies `p=R^T h` before reciprocal-axis
scaling and the standard `exp(-2 pi^2 v^T U v)` factor. Analytical direct-cell
derivatives include reciprocal-axis motion. Isotropic sites preserve their
existing calculation, and fixed tensor sites expose a zero compatibility Uiso
row while refinement selection omits that scalar.

The reviewed 256-reflection, 32-site release benchmark measures 801.9
microseconds for the isotropic fused path and 826.6 microseconds when all sites
use fixed tensors, a 3.1% increment on the development host. The QARR workflow
now uses the Al2O3 CIF tensors rather than trace-mean Uiso. Its reviewed run
passes at Al2O3 30.778%, ZnO 34.205%, CaF2 35.018%, 0.598 percentage points
maximum QPA error, 19.888% Poisson Rwp, 13.256% unit-weight Rwp, and 0.99061
correlation. It ended safely at the stage-2 evaluation budget after 354.1
seconds; FCJ optimization remains the dominant performance task.

The adaptive-FCJ checkpoint keeps the independently derived regular-height
integral and its fused analytical derivatives, but selects quadrature from the
physical axial-span/TCH-FWHM ratio. Ratios at or below 0.2 use an independently
validated 8-point Gauss--Legendre rule per non-empty smooth interval; all
larger ratios retain the conservative 48-point rule. Equal sample and detector
heights omit the zero-measure flat overlap interval, including its cancelling
symmetry-averaged derivative terms. A deterministic near-boundary study against
a 256-point NumPy integral bounds every value and direct derivative below
3.4e-11 scaled error. The realistic 200-reflection FCJ benchmark improves from
6.76 to 2.14 milliseconds, and the doublet case from 14.16 to 4.36
milliseconds, before full-workflow optimizer changes.

The reviewed cached-optimizer QARR checkpoint reduces stage 2 from the full
1,500 matrix-free evaluation budget to 52 cached-state evaluations across 35
accepted iterations. Three repeated release runs are scientifically identical.
The final PhaseSmith result is Al2O3 30.754%, ZnO 34.229%, CaF2 35.018%,
0.616 percentage points maximum QPA error, 19.826% Poisson Rwp, 13.174%
unit-weight Rwp, and 0.99062 profile correlation. In a warmed same-run native
workflow comparison, PhaseSmith takes 1.673 seconds and pinned GSAS-II takes
2.718 seconds, a 1.624x PhaseSmith speed advantage for this explicitly
non-matched parameterization benchmark.

The deterministic multicore checkpoint adds a public, persistence-safe
`ExecutionPolicy` with a bounded two-thread default and explicit fixed, serial,
or automatic worker budgets. Multiphase values, dense linearizations,
JVPs, and VJPs execute concurrently while results are always combined in phase
input order. The scheduler now exposes each phase/wavelength component as one
flat leaf, so a single-phase doublet and an unbalanced multiphase spectrum can
use the whole fixed worker budget without nested pools. Component results are
reassembled first in component order and then in phase order. Python provider
fallbacks remain serial unless every provider involved explicitly declares its
thread-safety capability; built-in immutable/native providers declare that
capability. Guarded special-position coordinate models, affine constraint
derivatives, native tangent maps, March--Dollase row maps, sample weights, and
parameter-invariant background bases are reused across trial states. Nonlinear
amorphous-background derivatives remain state-local. After flattening and
invariant caching, three fresh three-run QARR release medians are 1.232 seconds
on one thread, 0.829 seconds on two, and 0.710 seconds on three, with identical
status and scientific checks.
The third worker now remains useful because the dominant Al2O3 doublet is two
independent leaves. Against the preceding same-host pinned GSAS-II median of
2.718 seconds, the three-thread result is 3.83x faster, while remaining an
explicitly non-matched native-workflow comparison. Persistence format 10 stores
the same execution policy for generic calculation, Le Bail, and Rietveld while
loading formats 1--9 with an explicit one-thread migration default. The
complete gate passes 535 Python tests
(three optional cases deselected), 70 Rust tests, Ruff, Rust formatting, and
strict Clippy.

The bounded-native-context checkpoint adds a private Rayon pool owned and
reused by each native structural phase; PhaseSmith never mutates Rayon's global
pool. Fixed reflection partitions (at most 64) parallelize structure-factor
values, dense derivatives, and JVPs with bounded scratch, ordered merges, and
bitwise-identical one-, two-, and three-thread results. Structure-factor VJPs
retain canonical serial reflection reduction until a row-owned reverse kernel
can preserve that same bit pattern. Dense pattern
chaining is partitioned by parameter row so workers own disjoint outputs while
each row preserves reflection order. The scriptable `PreparedStructuralPattern`
and `calculate_structural_pattern` APIs accept `ExecutionPolicy` directly. A
single native leaf receives the full budget, while the multiphase/component
Python scheduler assigns one native thread per concurrent leaf to avoid nested
oversubscription. The complete gate passes 536 Python tests (three optional
cases deselected), 73 Rust tests, Ruff, Rust formatting, and strict Clippy.

The internal-profile checkpoint evaluates symmetric CW, FCJ, and TOF
reflection supports in parallel-owned blocks and merges them in original
reflection order. One-thread and small batches keep their original direct loops
without scratch allocation. A 254-reflection single-phase dense structural
linearization improves from a 1.242 ms median on one thread to 0.926 ms on two
and 0.814 ms on three. An 80-reflection TOF batch improves from 103.60 ms to
53.50 ms and 40.71 ms respectively. The public `accumulate_tof` API now accepts
`ExecutionPolicy`; all values and local/global analytical derivatives remain
bitwise identical across worker counts. The complete gate passes 537 Python
tests (three optional cases deselected), 75 Rust tests, Ruff, Rust formatting,
and strict Clippy.

The final multicore audit ran the complete paired QARR gate against a temporary
detached worktree of the exact pinned GSAS-II revision
`c0bc79b259cdf0065480b5fbd57674ddf12c4a23`. Across three measured complete
runs, PhaseSmith with the public two-thread default took a median 863.266 ms;
GSAS-II took 2887.243 ms, a 3.345x PhaseSmith advantage for the explicitly
documented non-matched native-workflow comparison. The scientific gate passed:
PhaseSmith returned Al2O3/ZnO/CaF2 30.754/34.229/35.018 wt% and 19.826% Poisson
Rwp, while GSAS-II returned 31.479/33.652/34.869 wt% and 18.389% Rwp. The
temporary oracle worktree was removed after the run; GSAS-II remains absent
from normal installation and runtime.

Laboratory fundamental-parameters validation now also pins the CC BY 4.0
Rowles/Curtin TOPAS v6 archive and converts mixtures `1a` and `1e` into a
neutral XY/CIF/instrument/JSON bundle. The GSAS-II-parity slice deliberately
omits TOPAS fundamental-parameters optics. It adds an independently documented
Lorentzian microstrain convention, corrects the GSAS size-factor mapping, and
alternates exact weighted scale/background solves with nonlinear instrument
and specimen blocks. Against exact-revision GSAS-II, PhaseSmith now returns
8.782%/8.264% Poisson Rwp versus 9.085%/8.195%; maximum cross-program phase
fraction deltas are below 0.25 percentage points. The parity gate requires
Rwp deltas below 0.5 percentage points and phase-fraction deltas below 0.5
percentage points. Deposited TOPAS source terms remain provenance metadata only
and no TOPAS or GSAS-II code enters the runtime.

The same independently implemented size/microstrain profile and exact-linear
scale/background alternation now transfer to both IUCr QARR mixtures. A
separate matched parity workflow uses ten Chebyshev terms, fixed Cu doublet and
dispersion values, refined U/V/W/zero, isotropic size and Lorentzian
microstrain, and the explicit trace-mean isotropic representation of deposited
anisotropic displacement tensors in both programs. With 29 free parameters,
PhaseSmith and pinned GSAS-II respectively return 18.226%/18.395% Poisson Rwp
for 1g and 18.358%/18.636% for the untouched 1h holdout. Maximum phase-fraction
deltas are 0.00355 and 0.00415 absolute. Both cases pass limits of 0.005 for
phase fractions and both Rwp conventions and 0.002 for profile correlation.
The prior QARR acceptance workflow remains available as a distinct model and
is not silently redefined by this oracle comparison.

The next independent laboratory checkpoint uses specimen 100a from NIST SRM
660c. It preserves NIST's released fundamental-parameters curve as the primary
reference, while a separate 17-parameter physical common subset compares
PhaseSmith with pinned GSAS-II: fixed positive-variance U/V/W and zero
microstrain, refined zero, isotropic size, two Uiso values, scale, and twelve
Chebyshev coefficients. The oracle boundary explicitly converts NIST's
millimetre specimen displacement to GSAS-II's micrometre `Shift` convention.
At the matched small-asymmetry setting SH/L=0.002, PhaseSmith and GSAS-II return
20.495% and 20.864% Poisson Rwp, 28.057% and 27.803% unit-weight Rwp, and
0.95959 and 0.95982 profile correlation. Every cross-program gate passes.

The original SH/L=0.02 setting is retained separately as a large-FCJ stress
holdout. Correct displacement units give 18.912% PhaseSmith Rwp versus 17.024%
GSAS-II Rwp, so the expected profile gate remains failed and isolates the
continuous-versus-discretized FCJ difference. The NIST reference remains much
better at 6.055% Rwp and 0.99948 correlation because its full Cu spectrum and
fundamental-parameters optics are deliberately outside this empirical parity
subset. Unconstrained GSAS-II runs remain rejected as targets because they
drove U and microstrain negative, producing nonphysical high-angle Gaussian
variance.

The subsequent offline calibration increment adds independently derived full
source/sample/receiver axial ray geometry and triangular incident/diffracted
Soller transmissions. It remains outside the refinement runtime and compresses
only into the existing analytical Rust `U/V/W/X/Y + SH/L` candidate. An
exploratory NIST 100a run using the documented 12 mm / 15 mm / 5 mm lengths,
6.776 degree Soller widths, and a provisional narrow Cu doublet lowers the
PhaseSmith Poisson Rwp from 20.495% to 12.525% and raises profile correlation
from 0.95959 to 0.99482. The compression is correctly rejected by normal shape
gates (global relative L2 0.1909), and it remains well behind the NIST 6.055%
reference. The graphite analyser/spectral passband is therefore a separate
required validation increment; the exploratory spectrum is not a golden
instrument calibration.

The next offline increment adds an explicit unit-height Gaussian wavelength
passband with caller-supplied center and FWHM. It multiplies each continuous
TCH emission line before the full axial/equatorial ray convolution, and the
integrated transmitted line areas become the compressed model's fixed
component weights and their first moments set its effective wavelengths. The
26.6 degree NIST graphite value is the analyzer diffraction angle, not a
recoverable bandwidth, so no guessed NIST default is provided. A full-order
balanced-band probe gives 15.6665% Rwp and 0.99319 correlation versus 12.5254%
and 0.99482 without the band after the transmitted line centroids are carried
into the compressed spectrum. The negative result shows that the provisional
narrow doublet already acts as an effective post-analyzer spectrum; it does not
justify tuning an unreported bandwidth to the validation specimen.
Compression of that passband target still fails the standard gates with global
relative L2 error 0.1590 and minimum per-peak correlation 0.97614.
Incident-monochromator dispersion, flat-specimen effects, tube tails, and a
provenance-complete effective Cu spectrum remain separate work.

## Design commitments

- Use GSAS-II only as a pinned validation oracle, never as the architecture.
- Preserve validated profile equations and observable numerical behavior while
  independently designing data structures, APIs, memory layout, and execution.
- Accumulate peaks directly into their finite support on a sorted grid. Values
  and analytical derivatives are calculated in the same Rust pass.
- Keep the public Python API small, array-oriented, deterministic, and suitable
  for notebooks as well as automated pipelines.
- Keep instrument, phase, pattern calculation, and refinement concepts in
  separate typed modules. Refinement methods are explicit submodules rather
  than mode flags in a monolithic workflow.
- Make all workflows script-first. Domain objects use NumPy arrays and
  serialization-friendly plain records so GUI applications can integrate
  without adopting internal core types.
- Keep integration boundaries application-neutral. Package-specific GUI
  adapters are compatibility conveniences, not core architecture or roadmap
  milestones.
- Keep future desktop runtimes Rust-only. Application-neutral workflows move
  behind shared native APIs used by both PyO3 and presentation adapters; Tauri,
  GUI state, CPython embedding, and Python sidecars are not library architecture.
- Maintain an independent, readable Python reference implementation for every
  numerical kernel before optimizing it.
- Treat GSAS-II licensing conservatively: implement from published equations
  and public behavior, record sources, and compare numerical output. Do not
  copy source code into this repository.
- License the independently implemented project under MIT. GSAS-II remains a
  separately licensed external validation oracle and is not redistributed.
- Keep crystallographic calculation in Rust: cell/reciprocal mathematics,
  symmetry application, reflection generation, scattering/structure factors,
  integrated intensities, and analytical derivative products. CIF adapters
  translate files into typed plain structures but do not calculate diffraction.

## First vertical slice

The first release-quality slice provides a symmetric pseudo-Voigt peak:

\[
p(\Delta; H, \eta) = \eta L(\Delta; H) + (1-\eta)G(\Delta; H),
\]

where `H` is full width at half maximum and `eta` is the Lorentzian fraction.
Both components have unit area on the real line. A peak contributes
`intensity * p(x - position)` only inside an explicitly configured number of
FWHM values. The finite support is a computational contract, not an accidental
floating-point cutoff.

The slice includes:

- a pure Rust kernel and fused multi-peak accumulator;
- analytical derivatives with respect to intensity, position, FWHM, and `eta`;
- Python bindings plus an independent NumPy reference implementation;
- Rust unit tests and Python differential/finite-difference tests;
- a GSAS-II scripting adapter boundary for extracting `X`, `Ycalc`, background,
  and reflection lists, with a narrowly scoped internal probe fallback;
- microbenchmarks for the scalar profile and fused accumulator.

The derivative at a moving support boundary is intentionally undefined. The
reported Jacobian is the derivative of the in-support profile with the active
sample set held fixed. Tests avoid samples exactly on that boundary.

## Validation strategy

Validation is layered so failures are localizable:

1. Compare Rust values and derivatives against closed-form expectations.
2. Differentially compare the Rust extension with the independent Python
   reference across regular and randomized inputs.
3. Check every analytical derivative by centered finite differences away from
   support boundaries.
4. Verify continuous and sampled integrated intensity, centroid, variance-like
   moments, FWHM behavior, and reflection parameters.
5. Run pinned GSAS-II oracle cases and compare extracted `X`, `Ycalc`,
   background, reflection lists, and any explicitly probed intermediates.
6. Track performance with repeatable Rust and Python benchmarks. On the
   controlled oracle host, also record the numerically gated GSAS-II comparison
   with exact scope, software provenance, raw timings, median, and p95.

Oracle fixtures must record the GSAS-II commit or distribution version, Python
version, input project/checksum, adapter version, and platform. Generated
fixtures may be committed when redistribution is lawful and their provenance is
documented; GSAS-II itself is never vendored.

## Architecture

- `crates/phasesmith-solution`: opt-in, file-independent single-crystal phasing
  and candidate-polishing kernels; RustFFT stays outside `phasesmith-core`.
- `crates/phasesmith-core`: dependency-light numerical types and kernels.
- Numerical primitives such as CW width laws and FCJ geometry remain separate;
  explicit composition modules fuse them for production accumulation.
- `crates/phasesmith-execution`: bounded reusable native execution contexts;
  it never mutates a process-global worker pool.
- `crates/phasesmith-py`: PyO3 extension exposing array-oriented functions.
- `crates/phasesmith-crystallography`: file-independent cell, exact symmetry,
  reflection-generation, and P1 structure-factor kernels; scattering follows.
- `crates/phasesmith-engine`: native composition facade; structural-intensity
  composition with the support-limited profile kernel and structural JVP/VJP.
- `crates/phasesmith-workflows`: application-neutral parameter, constraint,
  residual, background, runtime, and refinement orchestration shared by PyO3
  and future native applications; it contains no GUI or interpreter dependency.
- `python/phasesmith`: public Python package, separated instrument/phase/pattern/
  calculation/refinement modules, reference implementation, optional
  integrations, and validation tooling.
- `python/phasesmith/background`: model-independent background estimation and
  subtraction preprocessing. Refinable additive background models remain in
  `python/phasesmith/refinement/background`.
- `python/phasesmith/io`: optional format adapters; CIF uses a lazy Gemmi backend
  and returns only parser-independent structures and diagnostics; powder text
  readers return immutable arrays and source metadata.
- `python/phasesmith/refinement/readiness`: non-mutating pre-refinement review
  that carries import diagnostics forward, exposes retained provenance and
  active physical models, and reports contradictory optics or risky selected
  parameter combinations with stable machine-readable codes.
- `python/phasesmith/quantitative`: phase-scale interpretation and quantitative
  results, separate from iterative refinement.
- `python/phasesmith/validation`: opt-in external dataset provenance and
  machine-readable real-data workflows; no import-time network access.
- `tests`: Python differential and contract tests.
- `benchmarks`: end-to-end Python benchmarks and stored methodology.
- `oracle`: pinned GSAS-II environment metadata, adapters, and fixture schema.

The native application boundary keeps GUI frameworks outside the scientific
workspace, provides application-neutral owned model/I/O/workflow layers, and
preserves the Python scripting surface without requiring Python in native
consumers.

The 1.0 direction is fixed in `docs/api-stability.md`: typed physical-unit
domain APIs are the normal surface, direct H/eta remains a public mathematical
primitive, support-block Jacobians remain the default, and named adapters own
external conventions. Oracle fixtures and generated scientific tables require
explicit provenance and redistribution review.

The core accepts plain numeric slices and explicit peak/instrument batches. It
does not know about files, refinement iterations, Python phase objects, GUI
objects, or GSAS-II dictionaries. Higher layers translate domain models into
flat, validated kernel inputs. The durable public-module contract is documented
in `docs/public-api.md`.

The Python boundary validates and borrows NumPy inputs while it owns the Python
interpreter lock, releases that lock for every potentially long-running
pure-Rust batch kernel, and reacquires it only to construct Python-owned output
arrays or exceptions. This rule applies equally to profiles, backgrounds,
reflection/symmetry work, scattering, structure factors, intensity corrections,
and fused pattern accumulation. It keeps embedding applications responsive and
allows the bounded execution scheduler to overlap independent native batches
without weakening numerical determinism.

Physics extensions use explicit, versioned provider objects rather than global
core registration. A vectorized Python provider may calculate reflection-batch
width/intensity contributions and derivative chains before one native
accumulation call. New intrinsic line-shape kernels require a compiled native
provider for production speed; Python implementations remain valid reference
and prototyping paths and are never called inside the peak/sample hot loop.
Built-in and third-party models must satisfy the same calculation contracts so
Le Bail, Rietveld, persistence, and GUI adapters do not branch on model origin.

Monochromatic constant-wavelength X-ray and neutron radiation are supported
baseline models. Discrete
wavelength components are optional composition: one component is numerically
identical to the monochromatic path, while K-alpha doublets or other spectra are
supplied explicitly. FCJ geometry is independently optional. Neutron CW does
not inherit X-ray doublet or polarization assumptions.

## Milestones

1. Symmetric pseudo-Voigt profile, fused accumulation, derivatives, Python API,
   reference tests, GSAS-II adapter boundary, and benchmarks.
2. Constant-wavelength U/V/W/X/Y broadening.
3. Finger-Cox-Jephcoat asymmetry and K-alpha doublets.
4. Crystallite size, microstrain, and preferred orientation.
5. Multiple phases and scale/intensity composition.
6. Neutron constant-wavelength profiles.
7. Time-of-flight profiles.
8. Only after the numerical layers are mature: shared refinement
   infrastructure, followed by a first-class Le Bail workflow.
9. Plain-data persistence and an application-neutral NumPy interoperability
   boundary.
10. Rust crystallographic domain types and a P1 structure-factor vertical slice.
11. Symmetry, systematic absences, multiplicity, and reflection generation.
12. Optional CIF import into typed structures, followed by CIF-to-Le Bail.
13. Native X-ray and neutron scattering models with reviewed data provenance.
14. Fused structural-intensity/profile calculation with analytical JVP/VJP.
15. First full CIF-backed Rietveld refinement.

## Deferred post-1.0 roadmap

Public automatic peak picking is a future usability target only after the 0.6
cleanup and 1.0 API stabilization. Its claim is deterministic detection of
candidate peaks in a measured one-dimensional powder pattern, with explicit
preprocessing, threshold, boundary, uncertainty, and repeatability contracts.
It is not known-structure reflection generation, assignment of Miller indices
(HKL indexing), or database-backed phase identification. Those are separate
problems with separate inputs and validation claims.

No peak picker is implemented or exported in the 0.6 cleanup. In particular,
the private opXRD validation helper is not promoted as the public design; a
future implementation needs an independently reviewed API, algorithm source,
real-pattern validation set, and deterministic edge-case contract.

## Non-goals

- Porting nested GSAS-II dictionaries or global mutable state.
- Reproducing GSAS-II's file-driven refinement workflow.
- Calling Python once per reflection in production profile calculation.
- Calling Python once per atom or reflection in structure-factor calculation.
- Treating CIF parser objects as crystallographic domain or refinement state.
- Building a GUI in the numerical library.
- Requiring any GUI toolkit or package-specific adapter to use the package.
- Claiming numerical equivalence from pointwise values alone.

## Current real-data checkpoint

The pinned IUCr QARR 1g workflow is now an accepted Python-free Rust three-phase
fixed-spectrum structural validation, not a readiness placeholder. Its staged
refinement ends with a scale-only polish and converts scales using reviewed
phase Z, formula mass, and cell volume metadata through the native Hill--Howard
API. The reviewed Rust baseline gives Al2O3 30.460%, ZnO 34.113%, and CaF2
35.427%, with a maximum absolute error of 1.007 percentage points from the
independently weighed fractions. Poisson-weighted and unit-weight Rwp are
reported separately (0.19828 and 0.13179), alongside profile correlation
0.99062. Release tests repeat the native run exactly and compare the stable
measurements with the independent Python runner under explicit tolerances.

Phase-scale initialization is now an application-neutral
`phasesmith-workflows` operation rather than validation-harness logic. It
evaluates unit-scale native phase profiles on the exact finite-support grid and
solves a deterministic weighted non-negative least-squares problem using only
mask-included observations after subtracting fixed and analytical background.
The result owns a validated restart-ready `RietveldInput`, ordered scales, and
solve diagnostics. QARR consumes this public boundary; desktop adapters may use
the same operation without depending on `phasesmith-validation`.

This checkpoint uses explicit remaining approximations: fixed Cu K-alpha1
dispersion offsets for both doublet components and no absorption. SH/L=0.002
FCJ asymmetry and fixed CIF anisotropic displacement are active. It also
established two
refinement safety rules: nonphysical bounded trials are logged and backtracked
without losing the last accepted state, and phase-scale conditioning follows
the current nonzero scale magnitude instead of assuming scales are order one;
an exact zero retains an order-one escape scale.

All refinement families now use one explicit background-composition rule: the
pattern's supplied Smooth Bruckner estimate remains a fixed broad baseline,
and an optional differentiable polynomial or Chebyshev model is added as a
refinable residual correction. Structural Rietveld already followed this rule;
CW Le Bail now owns the same state in results and checkpoints, and TOF Le Bail
no longer substitutes its Chebyshev model for the fixed array.
Coefficient-invariant CW Le Bail backgrounds are solved by weighted linear
least squares after each intensity redistribution and are held out of the
nonlinear profile step. Synthetic Rust/Python tests recover the residual
coefficients and verify the additive calculation exactly. The APS sucrose
validation uses this convention directly: the fixed Smooth Bruckner array is
shared verbatim with the GSAS-II oracle and a one-coefficient residual is
refined by both workflows. Echidna retains its separate anchor-initialized
polynomial convention. QARR retains its separately staged residual-background
behavior and phase-fraction result.

The PbSO4 real-data checkpoint adds a separate optional staged-workflow layer
above the unchanged general Rietveld solver. Explicit `RietveldRecipe` objects
own parameter activation and stage acceptance. The deterministic
`intelligent_rietveld_recipe` planner is advisory: it may select only families
already authorized by the caller, records human-readable reasons and active
parameter identities, and never runs implicitly. The accompanying physics
checkpoint adds the constant-wavelength neutron powder Lorentz factor
`1/(sin(theta) sin(2 theta))` with native values and analytical derivatives.
Persistence format 13 retains that typed correction and the Debye--Scherrer
goniometer radius/X/Y geometry while loading formats 1--11.
The PbSO4 workflow composes background layers explicitly: a Smooth Bruckner
estimate is fixed preprocessing, while a three-term differentiable Chebyshev
correction is initialized by weighted linear least squares and then remains
active throughout the cumulative recipe. This preserves the distinction
between broad baseline estimation and refinable residual background structure.
The neutron position model is no longer approximated by one constant shift:
typed Debye--Scherrer X/Y displacements use the documented 650 mm radius,
native analytical derivatives, refinement, persistence, and finite-difference
tests. The pinned PbSO4 check now returns 4.217% neutron Rwp versus 4.535% from
GSAS-II and a maximum relative cell difference of 0.000424.
The independent X-ray workflow is accepted at 10.346% Poisson Rwp, 8.735%
unit-weight Rwp, 0.99555 profile correlation, and the fixed reference cell.
Its final polish accepts 48 improving steps before exhausting the consecutive
rejection budget on the flat tail. `repeated_rejections` remains a distinct
non-convergence reason and is not accepted by general recipe defaults. The
PbSO4 validation treats it as safe final-stage stagnation only when every
planned stage was attempted, at least one step was accepted, the returned
state is finite, and that stage improved Rwp by at least 0.0001. Numerical
failure, divergence, missing observations, and iteration exhaustion remain
unsafe; rejected trials never replace the last accepted state or checkpoint.
The Python staged-workflow adapter now also preserves the complete initial
constraint graph and maximum authorization across every temporary stage view.
All stages are validated before numerical work, later-stage constraints cannot
be erased by an earlier filtered stage, and a stopped workflow retains its
accepted physical state without narrowing the next planner proposal.

The built-in monochromatic Python Rietveld facade now delegates complete
refinement and checkpoint continuation to the Rust solver while retaining the
existing scripting types. Python-defined callbacks and provider extensions use
the explicit scripting fallback. The Python-free application model binds at
most one validated native analysis to each project histogram through
`RietveldProjectState`; native project format 2 introduced persistence for its built-in sample
physics, guarded lattice domains, analytical backgrounds, constraints, options,
covariance controls, and exact accepted restart checkpoint. Format 1 remains
readable as project-only state, so a future native application can load and
resume projects directly without a Python sidecar.
The PyO3 adapter exposes this codec and independently loadable native
checkpoint handles while releasing the GIL for project I/O; this is the bridge
used by the remaining public Python-facade migration, not a dependency of the
Rust workflows.
The public cancellation token shares the native solver's thread-safe state, so
the stateful Python project facade keeps cooperative `stop()` while using the
detached Rust refinement path.
The same facade now delegates representable built-in monochromatic project
persistence to the native format and reconstructs its scripting dataclasses and
restart handle from Rust-validated state. Python-only providers, component
radiation, non-native checkpoints, scripting optimizer controls, and rich
parser provenance continue to use the compatible format-13 scripting codec.
This completes the public Python refinement/persistence migration without
making Python part of native consumers.
Real-data validation follows the same boundary. `phasesmith-validation` owns
checksum verification, stable reports, native sucrose, QARR, PbSO4 neutron,
and exact fixed-doublet PbSO4 X-ray runners plus a standalone Rust CLI. Python
reconstructs its existing immutable report types from native JSON for ordinary
calls; callback/custom-execution and explicit reference runs stay in Python.
The validation matrix now also includes an accepted ANSTO Echidna LaB6 neutron
smoke gate and an official NIST SRM 660c archive-integrity gate. QARR 1h is kept
as a failing unchanged-setup holdout: its phase fractions pass while its
profile gates demonstrate that the 1g parameterization is not transferable.
The official POWGEN LaB6 bank and calibration are pinned and parsed in native
microseconds. `TofPatternRecord` and the native SLOG FXYE reader provide a
unit-explicit boundary that cannot be confused with `PatternRecord.x_deg`.
Adjacent SLOG boundary rows and their bin-width-multiplied Y/sigma values are
converted to the same 6,824 bin-center density samples exposed by GSAS-II; the
production position, derivative, and calibration-range checks pass. The native
fixed-instrument TOF Le Bail workflow now carries d-spacing reflections through
the fused value/local/global derivative kernel and nonnegative intensity
redistribution on nonuniform grids. The POWGEN run generates 330 LaB6 families,
returns all 15 instrument rows, and jointly refines an optional 16-term
microsecond-domain Chebyshev background through analytical basis columns and a
weighted linear solve after each redistribution. The reviewed uncertainty-weighted
run reaches Rwp 0.26461 and profile correlation 0.96729 and is an accepted
real-data workflow.
At this historical Le Bail milestone, TOF structural Rietveld parameter motion
remained a separate capability and was not implied by that result; Unit 38 now
provides it under an explicit structural observation/correction contract.
The legacy bank-2 `ICONS` adapter is black-box checked against pinned GSAS-II:
the record maps to Zero=4.41 µs, DIFC=22581.63 µs/Å, DIFA=0, and DIFB=0. The
live POWGEN oracle comparison matches all 329 GSAS-II calculation reflections'
positions exactly and their variance/alpha/beta terms at floating-point scale.
Using GSAS-II's extracted intensities and reflection list, the PhaseSmith kernel
reconstructs the peak-only oracle pattern with 0.999994 correlation and 0.00512
relative L2 error. The separately configured native workflows are reported for
context but their Rwp values are not treated as like-for-like because their
optimizers and endpoint conventions differ. GSAS-II uses a 16-term Chebyshev
background while PhaseSmith uses the same-order residual above fixed Smooth
Bruckner; the live workflow comparison now gates absolute Rwp delta below
0.03 and profile-correlation delta below 0.02.
The controlled pinned-oracle job now enforces live cross-implementation gates
for the sucrose and Echidna Le Bail cases, both QARR mixtures, and the existing
PbSO4 paired workflow. Workers run only in GSAS-II's isolated interpreter and
return plain finite JSON; normal installation remains independent. NIST 660c
continues to use its certified values and released profiles as the primary
oracle rather than treating a GSAS-II refit as replacement truth. POWGEN now
has the same live pinned-oracle boundary as the other accepted
real-data workflows, while its synthetic derivative fixture remains unchanged.
POWGEN bank 2 now also has a native one-bank structural acceptance path. The
checksum-pinned header records the vanadium run, bin-width scaling, and
proton-charge normalization; the type-0 calibration supplies no incident
spectrum, so the request explicitly declares `already_normalized` and applies
the named 90-degree TOF-neutron Lorentz correction. A staged structural solve
reaches Rwp 0.14415481, profile correlation 0.97638031, a=4.15792396 A, and
x(B)=0.19954072. The latter values agree with the published POWGEN SRM-660b
result within 0.000424 A and 0.000060. The existing pinned GSAS-II Le Bail
oracle continues to gate profile/calibration translation, while the LANL
nickel oracle remains the stricter structural cross-implementation gate.
The first public POWGEN application slice promotes those validation entry
points into bounded production adapters. Legacy GSAS profile-function-3 banks
are translated by `phasesmith-io` into the typed 15-coefficient
`TofInstrument`; the Python boundary exposes distinct `TofPowderData` and
`TofPowderPattern` types, so microseconds cannot silently enter the CW angle
workflow. `TofLeBailInput.from_files` composes one SLOG FXYE/GSA bank, matching
PRM bank, and CIF into generated fixed-cell reflection families, and
`refine_tof_lebail` delegates the complete extraction to the Rust workflow.
This remains a fixed-instrument, fixed-cell Le Bail path. Multi-bank
coordination, structural parameter motion, and full TOF Rietveld refinement
were explicit follow-on work at this milestone and are completed in Units
33--38 rather than being attributed to this API.
The TOF boundary is no longer POWGEN-specific. Plain bin-center/density columns,
GSAS logarithmic FXYE boundaries, and GSAS packed constant-step STD counts all
resolve into the same `TofPatternRecord` convention; packed counts are shifted
from lower boundaries to centers and divided by bin width. Legacy GSAS profile
functions 1 and 3 both translate into the typed coefficient record, while any
facility can bypass GSAS syntax and provide public microsecond arrays and a
`TofInstrument` directly. A revision- and checksum-pinned LANL nickel bank is
the independent transferability gate: 4,431 fitted samples, 102 Fm-3m
reflection families, Rwp 0.02485869, and background-subtracted profile
correlation 0.99891242. This establishes facility-neutral operation for the
supported back-to-back-exponential/TCH law; it is not a claim that every
beamline-specific tail or resolution law is already implemented.
The TOF Le Bail application boundary now also uses the shared native refinement
runtime. Cancellation is checked before each accepted cycle and model
evaluation; partial candidate work is discarded, every accepted state can be
delivered as a typed checkpoint, and continuation reproduces uninterrupted
history and final arrays exactly. Structured start/iteration/termination events
are available to Rust hosts and as plain progress dictionaries in Python. This
closes the cooperative-control part of application hardening. Native project
format 3 now persists this state without reusing the constant-wavelength
record: `ProjectRecord.tof_histograms` owns an explicit `tof_us` pattern and
validated 15-coefficient TOF experiment, while `TofLeBailProjectState` binds at
most one runnable analysis to each TOF histogram. Reflection topology and
intensities, options, refinable Chebyshev background, masks, residual arrays,
complete accepted history, and the last restart checkpoint round-trip through
deterministic JSON plus typed NPZ arrays. Histogram IDs are unique across the
angle and TOF collections, summary reports name the coordinate convention, and
format-1/2 projects remain readable with empty TOF state. This closes durable
native TOF persistence; it does not imply structural TOF refinement or
multi-bank coupling.
The first multi-bank layer is now complete too. `TofMultiBankInput` requires at
least two stable bank IDs and an exact shared fixed-cell contract: phase order,
phase/reflection IDs, labels, HKLs, and d-spacings are identical across banks.
The grids, 15-coefficient instruments, observations, uncertainties, masks,
fixed/refinable backgrounds, phase scales, and extracted intensities remain
bank-local. `refine_tof_multibank_with_runtime` evaluates the three stages of
each redistribution cycle across every bank and accepts them atomically under
one runtime. Aggregate Rp, Rwp, chi-square, and reduced chi-square are computed
from the concatenated included observations, while each bank retains its full
residual arrays. Cancellation cannot expose a partially advanced bank set, and
a cancelled checkpoint resumed under the same contract reproduces the
uninterrupted per-bank arrays and joint history exactly. This is a true shared
fixed-cell application contract, not alternating single-bank calls.
The shared-lattice increment is now native too. `tof_lattice_geometry`
evaluates d-spacings, calibrated microsecond positions, and their analytical
derivatives with respect to the symmetry-independent cell variables. The
reciprocal-metric d-spacing chain is bank-independent; each bank's Zero/DIFC/
DIFA/DIFB coefficients enter only through the final `d(tof)/d(d)` chain. Rust
finite differences cover every supported crystal system and the rhombohedral
setting, alongside the independent NumPy reference. `TofMultiBankLatticeInput`
adds bounded symmetry-independent cells to the exact shared topology. Each
cycle performs the atomic local extraction first, then assembles one weighted
cell Jacobian from every bank's fused `dY/dd` rows and the analytical `dd/dp`
chain. A scaled damped step is clipped to declared bounds and backtracked on
the concatenated chi-square. Accepted checkpoints contain every bank-local
state and shared cell, and cancelled continuation is exact. Synthetic banks
with different grids, calibrations, masks, scales, backgrounds, and
intensities recover the common cell; full pattern columns pass centered cell
differences.
That review exposed a finite-support derivative omission in the original TOF
kernel: internal quadrature limits and `R = support_fwhm H` move with profile
parameters. The Rust kernel and independent NumPy reference now include the
analytical Leibniz boundary terms, define the derivative as zero at an exact
clamp, and retain the existing calculated-value operation order. The realistic
200-reflection/5,001-sample benchmark changed by about 1.4% at `tail_log=8`.
Selected bank-local instrument motion is now native for the fixed-cell
multi-bank contract. `TofInstrumentParameter` binds every selection to the 15
fused global rows; finite bounds contain the initial physical coefficients and
unselected values remain bitwise fixed. Each atomic cycle appends the selected
bank-local columns to the summed weighted objective, solves a scaled damped
step, rejects nonphysical profiles by backtracking, and checkpoints instruments
with intensities/backgrounds. Final numerical rank and weighted-column
correlations remain explicit result data. Synthetic banks independently recover
Zero and DIFC, same-bank correlations are surfaced, and cancelled continuation
is exact. The realistic two-bank, 80-reflection-per-bank, 4,001-sample-per-bank
one-cycle benchmark measures 426.04 ms on the review machine.
Joint shared-cell plus local-instrument motion is now complete too.
`TofMultiBankGeometryInput` concatenates the analytical shared `dY/dd * dd/dp`
block and selected bank-local global rows against one bitwise-identical weighted
residual vector. One bounded damped/SVD step and one backtracking decision move
both families atomically. Typed diagnostics name lattice/instrument correlation
pairs, report numerical rank, and never fabricate covariance for a deficient
selection. Synthetic distinct-bank data recover the common cubic cell and both
local Zero terms while surfacing their cross-family correlations; cancelled
continuation is exact. The two-bank, 80-reflection-per-bank, 4,001-sample-per-
bank one-cycle benchmark measures 214.18 ms. Application facade, persistence,
and real-data oracle coverage remain separate gates before any structural TOF
Rietveld claim.
The first Unit 37 application slice now exposes the same joint solver through
typed Python bank, shared-cell, instrument-selection, option, result,
diagnostic, and checkpoint records. The exact native `SpaceGroup` topology is
reused when constructing setting-aware lattice bounds, the GIL is released for
the complete solve, and progress/cancellation plus opaque restart checkpoints
preserve the Rust runtime contract. Python integration tests recover the shared
cell and local Zero terms and prove exact cancelled continuation. Durable
multi-bank project persistence and real-data/oracle acceptance are separate
Unit 37 slices.
Unit 37b now adds native project format 4. A
`TofMultiBankGeometryProjectState` assigns disjoint TOF histogram sets to
stable analysis IDs and cross-checks patterns, initial instruments, phase
order/labels, shared initial cells, and exact symmetry settings. The explicit
JSON/NPZ codec round-trips bank inputs, cell and instrument bounds, solver
controls, and the full accepted checkpoint: cells, instruments, intensities,
backgrounds, per-bank residual arrays, aggregate metrics, and physical change
history. Versions 1--3 remain readable with empty joint state.
The native part of Unit 37c now uses the revision- and checksum-pinned LANL
nickel example as a genuine three-bank joint gate. Banks 2--4 retain distinct
grids, function-1 instruments, backgrounds, and intensities while sharing the
complete 102-family Fm-3m topology. Twenty accepted cycles jointly refine one
cubic cell plus three local Zero terms over 13,293 observations. The reviewed
result has joint Rwp 0.02273306, per-bank Rwp 0.02206--0.02336,
a=3.52361196 A against 3.5234 A, and rank 4/4. The complete shared topology is
intentional: finite bank grids, not an intersection-only reflection list,
determine local visibility. Unit 37c's pinned GSAS-II half is complete too. Its
isolated worker creates one three-histogram project and exports only arrays and
JSON. PhaseSmith matches reflection positions exactly, variance and tail-rate
chains at floating-point scale, and the same-extracted-intensity bank patterns
with minimum correlation 0.99999898 and maximum relative L2 0.00181021. The
independent shared-cell results differ by 0.00025472 A. The isolated worker
first writes checksum-derived one-bank views because the pinned scripting
loader otherwise selects the first legacy RAW dataset when repeatedly called
with `databank`. The explicit GSAS-II and PhaseSmith endpoint conventions
select 13,290 and 13,293 observations. Rwp is
context only because the Le Bail intensity and background procedures are not
the same optimization contract.
Unit 38's structural capability assessment is complete. Reciprocal-metric
geometry, symmetry, neutron structure factors, the fused TOF profile, and the
atomic multi-bank objective are reusable, but the current structural pattern
engine remains explicitly constant wavelength. Structural TOF additionally
requires a typed bank scattering angle and an explicit observation-correction
contract. PhaseSmith will support neutral/precorrected observations and a named
legacy one-dimensional neutron TOF Lorentz factor as distinct choices; it will
not infer incident-spectrum, detector-efficiency, absorption, or extinction
corrections from a facility name. `TofInstrument` continues to represent the
empirical profile/calibration law rather than pretending that fitted `DIFC`
uniquely supplies physical bank geometry. The complete equations, ORNL/POWGEN
request checklist, and staged implementation gates are recorded in
`docs/tof-structural-readiness.md`.
Unit 38a's first numerical slice adds the explicitly selected conventional
one-dimensional neutron TOF Lorentz factor `d^4 sin(theta_bank)`. Rust evaluates
its value and analytical reciprocal-metric derivative together, and the public
Python model agrees with an independent NumPy equation and centered finite
differences. Invalid bank angles fail at the boundary, wavelength derivatives
are exactly zero, and the existing constant-wavelength structural engine
rejects this TOF-only correction rather than silently applying it on an angular
pattern. Typed bank geometry and bounded legacy import were the next Unit 38a
slice. That slice is now complete: `TofBankGeometry` owns a strictly validated
facility-neutral `two_theta_deg`, while the bounded GSAS adapter optionally
maps the independently documented second `BNKPAR` field. Missing `BNKPAR`
remains compatible with profile-only imports but cannot satisfy a structural
request. Checksum-pinned POWGEN bank 2 and LANL bank 2 acceptance runs verify
the distinct 90.000 and 88.05 degree records. Unit 38a is complete, and Unit
38b's fused structural TOF calculation is now complete. The Rust
engine composes reciprocal geometry, constant coherent neutron structure
factors, explicit neutral or matching-angle TOF Lorentz correction, and the
finite-support asymmetric TOF profile. Values, dense structural Jacobians,
JVPs, and VJPs share the same profile-local `dY/dI` and `dY/dd` blocks; cell
motion includes intensity and d-spacing chains, and the 15 bank-instrument rows
remain available for later joint composition. Centered finite differences,
dense/product/adjoint agreement, explicit invalid-boundary tests, an independent
NumPy composition, and a realistic 128-reflection benchmark form the review
gate. The primitive remains Rust-only at this stage. Unit 38c's joint objective
is now complete: it atomically sums masked/uncertainty-weighted banks, shares
symmetry-aware lattice and site parameters, and namespaces scale, selected
instrument coefficients, and optional Chebyshev backgrounds per bank. Its
matrix-free JVP, VJP, gradient, and normal products pass joint centered
differences and the adjoint identity. The request freezes observation,
geometry, correction, bound, support, and identity contracts and rejects CW
sample physics or dynamic topology. The bounded checkpointable solver completes
Unit 38c before the separately reviewed Python/persistence/oracle exposure. It
assembles the exact analytical scaled normal matrix from the
joint objective's normal products, applies positive Levenberg damping, enforces
declared physical bounds and a scaled step cap, and accepts a trial only when
the summed masked/weighted objective decreases. Accepted input, parameters,
history, objective, and next damping are checkpointed atomically. Exact resume,
checkpoint corruption, cancellation after acceptance, evaluation exhaustion
during normal assembly, and simultaneous recovery of two bank scales and Zero
terms are tested. Unit 38c is complete; Unit 38d's Python facade, native project
persistence, and checksum-pinned structural real-data/oracle gates are complete
in the following reviewed slices.
Unit 38d's public Python facade is now complete as its first separate slice.
`phasesmith.refinement.tof_structural` constructs the same guarded native
shared-phase/bank-local request and returns immutable bank arrays, the updated
structural and instrument state, stable physical parameters, accepted history,
bounded termination, and an opaque exact-resume checkpoint. Python recovery
tests cover two local scales and Zero terms, cancellation/resume equivalence,
and one genuinely shared cubic lattice parameter.
The application-neutral project facade is now complete as a separate guard
before serialization. `StructuralTofMultiBankProjectState` binds every analysis
to unique project TOF histograms, requires exact pattern/instrument equality,
requires every member histogram to reference exactly the shared structural
phase, rejects external provider requirements, and revalidates an optional
solver checkpoint. Native project format 5 now serializes that validated state.
It stores stable site identity, structural and bank-local selections/bounds,
explicit bank geometry and correction, bounded numerical controls, and a
complete last-accepted checkpoint while referencing project-owned observation
arrays, initial instruments, and structural topology. Loads reconstruct typed
domain objects, revalidate exact request/checkpoint identity, and retain
formats 1--4 as structural-analysis-free migrations. The checksum-pinned
structural real-data/oracle acceptance gate described below completes Unit 38d.
The native half of that acceptance gate now passes on the checksum-pinned LANL
nickel banks 2--4. A facility-neutral `TofIncidentSpectrum` implements the
published Maxwellian-plus-Chebyshev law, its analytical TOF derivative, and an
inclusive calibration interval. The bounded GSAS adapter translates only
explicit type-4 records and rejects unsupported incident-spectrum functions.
Raw packed counts and uncertainties are divided by the calibrated incident
intensity before the structural fit; no facility name triggers correction.
The joint Fm-3m neutron refinement shares the cubic cell and Ni isotropic
displacement while refining three bank scales and Zero terms. Across 13,293
observations it reaches Rwp 0.03278208, minimum bank profile correlation
0.99749010, a=3.52373113 A against 3.5234 A, and Uiso=0.00396507 A^2. The
isolated pinned-oracle comparison is complete. Its public-API-only worker
refines the same shared cubic cell and Ni Uiso plus three bank-local HAP scales,
Zero terms, and twelve-term backgrounds. GSAS-II reaches Rwp 0.03367362,
minimum correlation 0.99730908, a=3.52368699 A, and Uiso=0.00401813 A^2 over
13,290 centers. PhaseSmith and GSAS-II differ by 0.00004414 A in cell and
0.00005306 A^2 in Uiso; both clear their independent Rwp/correlation gates.
Rwp equality remains context rather than a false cross-gate because background
and optimizer contracts differ. Unit 38d is complete.
The structural request now accepts one or more banks. A single detector bank is
a complete structural TOF experiment and uses the same objective, bounded
solver, checkpoint, Python facade, project validation, and format-5 persistence
as a many-bank request. The established `MultiBank` type names remain for API
stability; they no longer impose an artificial two-bank minimum.
The public one-bank file composition path is complete.
`StructuralTofMultiBankInput.from_files(...)` reads one bounded reduced-data
bank, its selected GSAS calibration with required detector geometry, and a CIF,
then generates fixed structural reflection topology. It requires the caller to
choose `already_normalized` or `calibration_type4` and `neutral` or
`tof_lorentz`; `already_applied` is also explicit when either the
integrated-intensity correction or sample corrections were performed upstream.
Missing requested calibration data or a named but unsupported sample-correction
model fails explicitly. The composed request retains byte sizes and SHA-256
identities for the pattern, calibration, CIF, and embedded or separately
supplied reduction record, plus the fixed-background array hash and every
explicit physics choice.
Hashing obeys the readers' bounded byte limits, and background provenance
distinguishes absent input from a supplied all-zero array. This audit metadata
stays in the Python application boundary rather than the numerical Rust kernel;
Python checkpoint continuation requires the provenance record to match as well
as the native numerical contract. The helper is a convenience adapter, not a
facility policy: other beamlines and calibration
formats construct the same typed request directly, and names never trigger
normalization or intensity corrections.
The same public composer now has a checksum-pinned POWGEN integration gate. It
loads the real 6,824-sample bank and 90-degree calibration with a deterministic
citation-annotated inline LaB6 structure, generates 330 families, and verifies
the registered data/calibration hashes plus structure and embedded-reduction
digests before the separate native structural refinement gate.
The composer now also supports an explicit inclusive `tof_range_us`, applied
before incident normalization and reflection generation and retained in
provenance. This closes a non-POWGEN portability issue for raw banks that extend
beyond an incident-spectrum calibration domain. On pinned LANL nickel bank 2,
1101.6--8189.6 us selects 4,431 samples, applies type-4 normalization, imports
88.05-degree geometry, and produces 100 center-visible reflection families.
`combine_file_banks(...)` joins explicit bank 2--4 requests into the declared
shared 0.2--3.0 A topology of 186 families while retaining ordered per-bank
provenance.
Per-bank provenance now binds the source bank number to the exact public
`bank_id`, and request validation requires provenance order to align with bank
order. Length-only provenance attachment is rejected.
Fixed-background normalization is now explicit too: `input` backgrounds follow
raw observations through type-4 division, while `normalized` backgrounds are
installed afterward and cannot be divided twice. A real public LANL bank 2--4
scale-only solve with independently estimated normalized Smooth Brückner
backgrounds converges in four accepted steps, with maximum bank Rwp 0.12434.
The structural TOF accepted-state solver now also preserves the established
special-position coordinate convention. Symmetry-reduced `q` values are local
tangent changes applied once to the current phase; after acceptance the moved
physical coordinates are retained and the next solver/checkpoint `q` state is
zero. A dedicated symmetry-site regression verifies accepted motion, valid
checkpoint publication, and exact continuation.
The public Python normalization slice is complete too.
`TofIncidentSpectrum` independently evaluates the documented NumPy equation
and analytical TOF derivative, returns immutable arrays, and transforms an
observed `TofPowderPattern` by dividing observations, uncertainties, and fixed
background consistently. `GsasTofInstrumentData` now carries the parsed
spectrum without preventing non-GSAS callers from constructing the same typed
record directly. The isolated structural oracle gate now passes as recorded
above.
The historical native-workflow comparison passes both Le Bail parity contracts
and QARR 1g. Its unchanged QARR 1h acceptance recipe retains the reviewed
profile-quality failure, rather than having thresholds relaxed. The newer
matched-parameterization QARR workflow is a separate parity contract and passes
both 1g and the untouched 1h holdout.
Release packaging preserves the same separation. The public crates.io
`phasesmith` facade re-exports the application-neutral native component crates;
PyO3 and validation tooling remain unpublished workspace packages; GUI
applications are separate consumers. A
`v<version>` Git tag drives tested ABI3
Python wheels, an sdist, provenance, PyPI trusted publishing, the public Rust
crate graph, and a tagged GitHub Release. Version checks bind Cargo,
Python, the changelog, and all internal registry requirements before any
publishing job receives credentials. GUI-specific state, job ownership, IPC,
and presentation behavior belong to a separate application repository.

The final planned parity boundary, joint refinement, is now native.
`JointRietveldLayout` shares structural phase/site parameters while preserving
local scale, background, profile, radiation, correction, geometry, masks, and
uncertainties. The constraint-aware solver evaluates the summed objective for
every trial and retains cancellation/checkpoint state and aggregate metrics.
The pinned Rust-only PbSO4 X-ray/neutron workload exercises the complete joint
path without Python, including the exact 1.5405/1.5443 Å X-ray doublet and the
monochromatic neutron histogram; alternating independent fits are not used.

The native-objective performance checkpoint closes the large regression that
appeared when those real-data workflows moved below Python. At each accepted or
trial state, `PreparedGeneralRietveldObjective` now requests the fused native
profile/analytical-derivative pass and reuses the projected structural Jacobian
for gradients, normal products, and covariance. The default ceiling is
10,000,000 `f64` elements (about 80 MB for structural rows); larger problems or
an explicit zero ceiling remain matrix-free. Joint objectives aggregate this
choice and its actual expensive-product accounting across histograms.

A realistic fixed-doublet Rust benchmark with 256 reflections, 10,001 samples,
eight sites, and FCJ measures 17.987 ms for bounded dense preparation, 0.276 ms
for a cached normal product, and 19.660 ms for the equivalent matrix-free
product on the development host. The repeated product is therefore 71.3x
faster. Fresh release application-boundary runs take 0.645 s for QARR, 2.337 s
for PbSO4 neutron, and 4.223 s for PbSO4 X-ray, versus the recorded pre-fix
13.02 s, 17.56 s, and 157.44 s. All scientific gates pass under the explicit
accepted-state termination policy above. The pure-Rust X-ray
Poisson Rwp is 10.34601%, while a fresh 3.560 s Python scripting run gives
10.34604%; the native path is scientifically coincident and about 19% slower.
It now follows the scripting optimizer's weighted free-coordinate Jacobian,
phase-scale conditioning, accepted-trial carry-forward, bounded backtracking,
and physical Bruckner-width conversion. The final stage's repeated rejections
occur only after material improvement on the flat tail of the minimum.
Historical pinned
GSAS-II timings remain separate because its PbSO4 number covers a joint
two-histogram recipe.

The 2026-08-09 hardening pass closes the remaining authoring and robustness
gaps without changing the scientific layering. All three LM solvers retry
rejected line searches with increased damping under the
existing attempt/evaluation/rejection budgets, and the joint solver reuses the
same finite-safe conjugate-gradient core as the single-histogram solvers. Joint
checkpoint continuation is pinned against uninterrupted history. Rust and the
independent Python references now share an explicit covariance convention:
known active uncertainties use the unscaled inverse weighted normal matrix;
unit-weight fits scale by reduced chi-square. FXYE zero-ESD rows become masked
exclusions and UTF-8 BOM input is accepted. Native persistence probes future
versions before strict decoding, uses application-sized default allocation
limits, and recovers the previous manifest/archive pair after an interrupted
overwrite.
The Python Rietveld result exposes `backend` as `native` or `python`, making an
eligibility fallback visible to callers.

The first post-unit-24 experimental workflow estimates an effective
constant-wavelength starting profile from one predominantly single-phase
pattern. The Python-free workflow reuses Le Bail independent intensities and
analytical profile derivatives, holds the pre-calibrated Dioptas/pyFAI
wavelength bitwise fixed, optionally aligns a bounded lattice as a nuisance,
and stages W, UVW, then UVWXY. Automatic promotion requires both absolute and
relative Rwp improvement, identifiable covariance, and bounded correlation.
The Python facade delegates width fitting to the same native implementation.
This is intentionally not an instrument-only calibration: sample broadening is
disclosed, known contaminant regions must be masked, and the output is a
starting profile. Calibrant registries, CeO2/LaB6 detector-geometry workflows,
in-situ Si separation, persistence, and GUI presentation remain deferred until
representative real data can validate their behavior.

The next experimental calibration slice follows the conventional GSAS-II
fundamental-profile workflow without adding full fundamental-parameter physics
to the refinement runtime. An independent NumPy generator builds isolated
peaks from discrete emission wavelengths and intrinsic wavelength widths,
ideal uniform equatorial source/receiving apertures, and published FCJ axial
geometry. A bounded offline fit then uses the existing Rust analytical
derivatives to compress those targets into production `U/V/W/X/Y` and a single
equal-height `SH/L`. Global and per-peak L2/correlation diagnostics reject
non-representable physical targets rather than presenting them as successful
calibrations. Its second reviewed increment adds finite axial
source/sample/receiver lengths and triangular incident/diffracted Soller
transmissions through an independently derived deterministic ray integral.
The third increment adds a fixed Gaussian wavelength passband before those ray
convolutions and propagates transmitted line areas into the compressed fixed
spectrum. Transparency, equatorial divergence beyond the ideal apertures, tube
tails, coupled incident-monochromator dispersion, and PSD defocusing remain
separate reviewed increments. No GSAS-II or NIST implementation code is copied
or required at runtime.

The deposited Rowles geometry has now been tested through that same strategy
with the exact pinned GSAS-II FPA workflow. A version-gated external worker
parses the neutral manifest and includes radii, axial lengths, incident and
diffracted Soller angles, LPSD aperture/divergence, tube tails, and the five
edge-transmitted Cu K-alpha lines. Its empirical compression reaches 6.952%
Rwp on 13 synthetic peaks, but fixing that profile worsens real-pattern GSAS-II
Rwp from 9.085% to 13.928% for 1a and from 8.195% to 11.710% for 1e. Specialized
LPSD, tube-tail, and continuum runtime physics are therefore deferred; the next
milestone is broader independent experimental validation using in-house
benchmark examples. That evidence must establish transferable profile behavior
before any specialized LPSD, tube-tail, continuum, or coupled-dispersion term
is promoted into the production runtime.

The first Unit-29 infrastructure slice is complete. A reviewed JSON manifest
now describes the existing real-data GSAS-II comparison drivers, their dataset
identities, oracle-revision locations, scientific assertions, and diagnostic
classification. One non-destructive command validates the exact pinned
checkout and every selected dataset checksum before executing the drivers, then
embeds the finite result records and canonical hashes in a single campaign
report. This is orchestration only: numerical workflows remain independently
reviewed, and no golden result is regenerated implicitly. The next Unit-29
scientific slice is the sodium-dihydrogen-citrate plus Si 640b holdout. Its
official IUCr supplement is checksum-registered and the reviewed converter
extracts the 4,452-point deposited range, both structures, raw counts, and
legacy calculation. The accepted common subset fixes the deposited zero and
complete silicon-derived U/V/W/X/Y profile, compresses the unequal source axial
ratios to matched equal-height SH/L=0.0187, calibrates specimen displacement
from fixed-cell Si, and refines only linear scales/background plus one
March--Dollase (001) ratio. PhaseSmith and pinned GSAS-II respectively reach
18.183%/18.945% Poisson Rwp, 0.95589/0.94997 profile correlation,
22.173%/21.651% Si, and March ratios 0.63851/0.63237. All seven predeclared
cross-implementation gates pass. This is preferred-orientation common-subset
parity, not equivalence to the deposited 8.433% Rwp and 18.74 wt% Si: the
source's spherical-harmonic orientation, Stephens anisotropy, and Suortti
surface roughness remain explicitly outside the transferable model.

The independent anhydrous tripotassium-citrate holdout now separates numerical
parity from physical-model sufficiency. On the same fixed-geometry common
subset, PhaseSmith/pinned-GSAS-II reach 23.965%/23.933% Rwp, correlations
0.61846/0.62079, and 4.450%/4.343% Si, passing all full-pattern parity gates.
The deposited Si fraction is only 1.44 wt%, and the two independent Si-window
fits infer -0.02492/+0.00180 mm displacement. Their 0.02672 mm disagreement
fails the declared anchor-identifiability gate, so neither diagnostic value is
applied. The PhaseSmith millimetre value is conditional on the disclosed
141.5 mm radius assumption because the source does not deposit that radius.
The much better deposited 4.853% Rwp and 1.44 wt% Si remain a
phase-specific richer-model reference; they are not a target for undocumented
parameter tuning or GSAS-II implementation copying.

The next source audit qualifies both official trirubidium-citrate/Si pdCIFs,
but orders them by information value. The anhydrous deposit is next because it
contains a 141.5 mm radius, 2.15 wt% Si, common base profile coefficients, no
surface-roughness correction, and a contiguous 4,106-point refined range. The
monohydrate is checksum-registered but follows later: its 1.30 wt% Si, absent
radius, phase-specific profiles, Stephens broadening, and Suortti roughness
largely repeat the limitations already isolated by the potassium holdout.
The anhydrous converter now reproduces deposited Rwp 2.45834% and Rp 1.95046%
on the exact contiguous source-index interval 594–4699, emits both structures,
and explicitly translates its legacy profile contract. Profile-function-4
`LX=3.634` maps to the current Lorentzian `X/cos(theta)` size axis, while
`shft=-8.7503` centidegrees maps at the deposited radius to -0.1080505 mm
in the current PhaseSmith/GSAS-II displacement convention. The legacy manual's
physical shift variable uses the opposite sign (+0.1080505 mm). The corrected
common-model comparison gives
PhaseSmith/pinned-GSAS-II Rwp 9.579%/9.735%, correlations 0.94732/0.94248, and
2.575%/2.548% Si, passing all full-pattern parity gates. The 2.15 wt% Si-only
windows still infer incompatible -0.07519/-0.11031 mm values; they remain
diagnostic and do not replace the directly translated source displacement.
Separate recomputation under Poisson weights gives 4.852883% for the deposited
potassium curve and 2.458326% for the deposited rubidium curve, effectively the
same as their deposited-weight 4.852875% and 2.458338% values. The large
common-subset gap is therefore not an Rwp-weight-definition artifact.

A phase-local isotropic-width ablation now closes the next diagnostic question.
With all instrument, geometry, structure, and silicon-width terms fixed,
potassium finite coherent-domain size remains the only accepted three-start
isotropic model, refining to 90.19 nm and 20.737% Rwp. After the rubidium source
translation is corrected, its base is already 9.579% Rwp and no size/strain
variant passes repeatability. The former 56.23 nm/14.823% rubidium conclusion is
invalidated as compensation for omitted position/profile semantics. The
current isotropic models therefore cannot explain the targets.

The first Stephens production slice is now implemented for orthorhombic cells.
It uses six physical ångström⁻⁴ coefficients for the fourth-order variance of
`1/d_hkl²`, plus an explicit PhaseSmith Gaussian/Lorentzian mixing fraction.
Python and Rust independently evaluate values and analytical coefficient,
mixing, position, and cell derivatives; native and scripting persistence and
Rietveld parameter rows are wired, and centered-difference plus realistic
multi-peak benchmark coverage is present. The provider rejects nonorthorhombic
cells and any materially negative active-reflection variance, so this slice
does not overclaim other Laue classes. The pinned GSAS-II oracle and real-data
sufficiency checks are now complete. All 395 reflection widths agree at roughly
machine precision and three complete profile probes agree within 2.40e-6 of
peak height. On the citrate holdouts, fixed deposited Stephens widths improve
Poisson Rwp by only 0.040 percentage points for potassium and 0.008 for
corrected rubidium. A free common rubidium amplitude still requires 382 times
the deposited value to reach only 9.416% Rwp, while size/Stephens combinations
are not repeatable. Stephens is therefore implemented and convention-verified
but falsified as the sole explanation for the deposited 4.853%/2.458% targets
under the fixed nuisance model.

A 16-case pinned-GSAS-II factorial forensic closes the legacy translation
question. Source displacement supplies 87.3% of the Shapley weighted-SSE gain,
corrected `LX` supplies 18.0%, Stephens supplies 0.07%, and translated
transparency is detrimental. The legacy manual fixes this sign convention and
also shows that deposited positive `trns=1.30` implies formally negative
effective absorption; a reversed-sign sensitivity improves the fit but does
not represent the deposited model. The best fixed subset reaches 9.726% Rwp and
closes 73.19% of the former base-to-deposited SSE gap. Of the remaining SSE,
65.8% is below 30 degrees and its width-mode correlation exceeds its position
mode.

The subsequent 17.004916–30 degree component audit is complete on 643 samples.
After local phase-scale and constant-background adjustment, the translated
source profile gives 9.8275% Poisson Rwp versus 2.7107% for the deposited curve.
No isolated background, peak-group intensity, position, `W`, `X`, or FCJ probe
closes half of that weighted-SSE gap. Their best physical profile combination
plus a rank-four, well-conditioned residual-background projection reaches
4.9772% and closes 80.47%, but still does not reconstruct the source. The FCJ
scan reaches pinned GSAS-II's
effective `SH/L=0.002` floor; because the source records separate
`S/L=H/L=0.0097`, this is an axial-compression fidelity warning rather than
evidence for zero divergence. No new production term is justified. The next
checkpoint first compares source-native reflection intensities and then legacy
profile-function-4 axial shapes before changing PhaseSmith physics.

The source-reflection checkpoint is complete. The converter now preserves all
1,197 deposited reflection rows, which reduce to 600 unique phase/HKL records.
For the 20 low-angle rubidium reflections, converted structure plus deposited
Cromer–Mann factors reproduces source calculated F² with 0.0289% intensity-
weighted L1 and 0.0301% normalized RMS error; d-spacings agree within
1.54e-5 angstrom. Production Waasmaier–Kirfel scattering gives 0.301% weighted
L1 and 0.296% normalized RMS error. The table choice is measurable but far too
small to explain the profile gap. Structure, symmetry, ADPs, and source
reflection intensities are cleared; legacy profile-function-4 axial-shape
fidelity was therefore the next checkpoint.

That axial-shape checkpoint is now complete. Legacy-GSAS full-height/diameter
ratios `S/L=H/L=0.0097` are numerically the FCJ half-height/radius ratios, so
PhaseSmith's native `FcjGeometry(0.0097, 0.0097)` preserves the source geometry.
It is bit-for-bit identical to PhaseSmith's documented equal-height mapping of
the GSAS-II formal sum `SH/L=0.0194`. For three low-angle source reflections,
however, pinned GSAS-II at that formal sum differs from the source-native
continuous profile by as much as 17.53% normalized L1, 15.38% of peak height,
and 0.00847 degrees in centroid. A tested `SH/L=0.0097` is empirically closer
but is not the documented sum and must not replace the physical conversion.
This clears the PhaseSmith axial conversion while identifying the reduced,
discretized GSAS-II oracle as non-shape-faithful for the legacy two-parameter
case. Exact recovery now requires a source-native legacy profile oracle rather
than another PhaseSmith broadening term. Other Stephens Laue classes remain
optional completeness work.

The first independent Unit-29 constant-wavelength transfer test now uses the
external IUCr ceria size/strain round robin without relabelling it as in-house
evidence. Exact-size/SHA-256 registration retains three contiguous Birmingham
ranges for the annealed calibration specimen and three for the untouched
broadened specimen. Three calibration starts select the same W-only empirical
profile on 8,726 samples (Rwp 0.192590, correlation 0.953870). Freezing that
profile before the 4,126-sample holdout and adding only the existing isotropic
size/Gaussian-microstrain model lowers Rwp from 0.553990 to 0.058097, reaches
0.985574 correlation, and reduces weighted SSE by 98.9002%. Three dispersed
starts and twelve deterministic Poisson resamples pass the stability gates;
peak-wise centroid, FWHM, area, L1, and moment diagnostics remain visible.

This is an empirical transfer result, not fundamental-parameters calibration.
The source omits radius, aperture/detector geometry, axial lengths, Soller
angles, monochromator passband, and specimen mounting metrology, and the current
Le Bail estimator discloses a dominant-K-alpha1 approximation for the 1.6%
K-alpha2 component. Those missing physical inputs block a fair specialized-term
test. Unit 29 therefore closes with the evidence-backed decision not to add
LPSD, tube-tail, continuum, or coupled-dispersion production terms. The frozen
promotion protocol in `docs/profile-transferability-gate.md` still applies to
any future in-house or provenance-complete external pair.

## Experimental structure-solution transfer benchmark

The structure-solution numerical implementation remains frozen at `8bc4efc` for
the first broader transfer screen. A checksum-pinned corpus now registers 100
measured COD reflection/reference pairs from 100 distinct DOI/formula families,
covering all seven crystal systems and 27 space-group types. A deterministic
family-separated split reserves 65 cases as unopened solver holdout outcomes;
35 development cases were compared serially against user-provided SHELXT 2018/2,
one thread, element types only, supplied cell/symmetry and a 15-second per-job
wall ceiling. Both consume identical rounded HKLF4 observations, without reference
coordinates or composition quantities in numerical workers. Exclusions and failed
searches remain visible. No production numerical code or dependencies change.

Exact full-cell non-H element counts plus coordinate recovery (maximum 0.05 Å,
RMS 0.025 Å) succeed in 5/35 PhaseSmith cases and 13/35 SHELXT cases. Total atom
counts alone agree in 9/35 and 29/35. All-outcome median wall times are 13.220 s
and 0.889 s on native arm64 PhaseSmith versus x86_64-compatible SHELXT; these are
not matched successful-solve times. Seven references exceed PhaseSmith's current
input-derived dense-model capacity and remain in the denominator. This evidence
shows a substantial generalization gap, despite prior success on three reused
structures. Count/orbit selection, capacity and element assignment should be
improved on development cases before opening the holdout.

Registry v5 corrects rounded special-position reference-image multiplicities for
nine cases, corroborated against registered cell-formula counts across all 100.
It preserves the v4 collection registry, source bytes, solver inputs, predictions
and timings. Output syntax/origin/precision corrections are re-audits of each
program's saved chosen model, never reference-selected alternatives or search
reruns. Protocol, thresholds, limitations and results are documented in
`docs/solution-corpus-benchmark.md`; the committed full report is
`benchmarks/structure-solution-corpus-development-macos.json`.

A subsequent development-only count policy preserves the initial density batch,
then competes one-/two-orbit residual additions using common isotropic screens
before a requested-mode fit. Weak-density candidates can use diffraction-based
labels, poor early maps have bounded handoff budgets, and promising models retain
the global budget. `adaptive_counts=False` preserves the earlier policy. No
expected atom counts enter either path. The first complete follow-up screen
improves exact total counts from 9/35 to 13/35 and full element-labelled coordinate
recovery from 5/35 to 7/35, preserving all five former successes. Runtime has not
improved overall. The contemporary same-native ablation recovers 4/35 versus
7/35 structures and 7/35 versus 13/35 total counts, with median all-outcome times
13.357 versus 13.390 s; these are serial collections, not paired repeated timings.
Independent NumPy score checks, synthetic symmetry/count ablations and a realistic
full-search microbenchmark accompany the existing unchanged analytical kernels.
See `docs/solution-count-selection.md`. The 65 holdout outcomes remain unopened;
this solver is still serial, and SHELXT remains pinned to one thread for comparison.

A further opt-in shared-search path fits an isotropic Wilson/shell envelope from
signed independent-family intensities with symmetry epsilon factors and an
explicit unit-mean E-squared convention. Mixed E/F amplitudes feed dual-space
phasing; measured-family CC minus weak calculated E-squared selects complete
atomic maps before count refinement. Rust values and analytical phase-quality
derivatives have an independent NumPy reference. This is a published-equation
implementation with explicit empirical shell/scheduling choices, not a recovered
SHELXT implementation. Both new options remain off by default: the complete
35-case development campaign preserves the same seven recovered structures but
recovers only 12 total counts versus the preceding campaign's 13, with no overall
speed improvement and slower successful solves. The same-build control recovers
8/35 structures and 13/35 total counts versus 7/35 and 12/35 with the new options;
the seven common recoveries slow from median 1.776 to 10.320 seconds. This is a
regression, with no gained structures and one lost recovery (COD 2243618).
All original scoring gates and input-derived caps remain fixed; the 65 holdout
outcomes remain unopened.
See `docs/solution-phase-quality.md` for the same-build control, timing limitations,
normalization assumptions and retained complete results.

A subsequent frozen-parent ablation separates observed normalization (8/35
structures, 14/35 total counts) from the original quality/scheduling option
(7/35 structures, 12/35 total counts). A renewed external SHELXT data-flow review
finds an important difference: calculated map amplitudes are independently
normalized before its weak-reflection statistic. PhaseSmith had reused the
observed envelope instead. An independent conditional-second-moment shell
normalization now supplies calculated E-squared and fused analytical derivatives;
the prior statistic remains an explicit reference mode. A separate checkpoint
portfolio preserves early amplitude-map fits before offering distinct quality
alternatives; the previous phase-first schedule remains explicit. These changes
are tested separately on all 35 development cases, retaining the original input,
count/geometry and budget controls. No holdout outcomes are opened. Equations,
source-confidence limits, comparisons and qualifications are recorded in
`docs/solution-phase-ablation.md`.

The corrected calculated normalization with phase-first scheduling recovers
9/35 structures versus a same-build old-statistic control's 7/35, gaining COD
2010705 and 2243618 with no lost recoveries. Both get 11/35 total counts correct.
Seven common successes take median 8.803 versus 10.932 seconds; all-outcome
medians remain near the 13-second budget. The checkpoint alternative recovers
only 7/35, so quality selection keeps phase-first ordering by default. All feature
switches remain experimental and opt-in. Count accuracy remains limited and
SHELXT still recovers 13/35 labelled structures and 29/35 total counts. No holdout
outcomes have been opened.

An additional opt-in `phase_origin_selection` path changes the order of the
element-only search. A measured phase-equivalence origin/hand search aligns each
P1 density before atom labels are assigned. Supplied-symmetry density recycling
then provides complete orbit proposals. Cached orbit Fourier columns and fused
coordinate jets support cheap whole-orbit count/element edits, followed by short
isotropic screens; only a small complete-model shortlist receives requested-mode
refinement on all original observations. Independent NumPy equations check both
new kernels. This is an implementation from published equations and first
principles, with explicit bounded heuristics, not translated decompiled SHELX
code. It searches the supplied group only. The previous solver remains the
default reference path. Development results, stage costs and limitations are in
`docs/solution-phase-origin.md`; the 65 holdout outcomes remain unopened.
The same-build 35-case comparison improves correct labelled structures from
8 to 12, exact total counts from 14 to 26 and exact per-element counts from
8 to 13; median all-outcome wall time falls from 13.217 to 4.376 seconds.
Six recoveries are gained and two lost, so the new path is not a universal
improvement. Requested-mode fit attempts fall from 712 to 100. SHELXT still
recovers 13 structures and 29 totals at a 0.877-second median. These are fixed
single-seed development collections, not repeated timing or holdout evidence.

A further opt-in pair adds motif-based density scale proposals and coupled
post-isotropic-fit label changes (`chemical_assignment`) and a diagonally scaled
conjugate-gradient normal solve for large refinements (`iterative_refinement`).
The normal solver retains the original analytical Jacobian and all resource
caps, but avoids building its dense normal matrix. The chemistry search keeps
required types and total topology, tests single changes and paired swaps after
coordinate/displacement relaxation, and accepts only improved refined screens.
Both are independently validated and leave prior defaults available. Equations,
controls and development-only comparisons are in `docs/solution-chemical-refinement.md`.
In the same-build four-way 35-case development ablation, chemistry improves
labelled recovery from 12 to 14 and exact element-count sets from 13 to 15,
with no lost recoveries and unchanged 26 exact total counts. Iterative refinement
alone keeps the same 12 structures and lowers median all-outcome time from
4.308 to 3.535 seconds. Combined reaches 14 structures at 3.348 seconds and
11 strict 0.02-angstrom recoveries versus control's 9. Requested-mode fitting
time drops from 63.895 to 34.825 seconds across the panel. These are single-seed
serial development results with unchanged caps; both controls remain opt-in and
the 65 holdout outcomes remain unopened.

A further opt-in prediction experiment distinguishes the existing free missing
coefficients inside the measured sphere from bounded, observed-envelope-normalized
prediction (`missing_reflections`) and added reciprocal support beyond measured
resolution (`resolution_extension`). New coefficients remain theoretical predictions:
original observations, origin/weak-reflection tests, peak-mass integrals, final
fits and atom caps remain measured-only. Both P1 atomicity and supplied-symmetry
positivity recycling can use the bounded predictions. Local complex values and
analytical coefficient Jacobians have independent NumPy and finite-difference
checks, and complete Fourier-cycle parity is tested. The previous combined path
remains the control. Equations, bounds and 35-case development-only comparisons
are in `docs/solution-reflection-prediction.md`; the holdout stays closed.
The same-build four-way development comparison gives 14/35 structures and 26
correct total counts for the current control, 14/27 for missing-only, 15/26
for extension-only, and 14/27 for both. Exact per-element counts stay at 15/35
in all modes. Medians rise from 3.388 seconds to 4.455, 4.244 and 4.268 seconds.
Extension's one gained structure is also recovered by a larger-grid-only probe
without extrapolated coefficients; the gain cannot be assigned to prediction.
The combined path gains that structure but loses another coordinate recovery.
Both controls therefore remain off, with all outcomes retained and no change to
correctness gates or input-derived caps. The 65 holdout outcomes remain unopened.

A further opt-in experiment adds adaptive phase-first handoff checkpoints
(`adaptive_checkpoints`) and measured-harmonic origin FFT seed sampling
(`independent_origin_sampling`). Stable-map decisions use only P1 phase quality,
amplitude fit and density agreement; downstream complete-model validation remains
mandatory. Origin sampling is independent of the density grid, bounded to 128
per axis, with exact measured-phase loss/derivatives retained for polishing.
Neither control changes measured constraints, refinement kernels or atom caps.
Independent NumPy references and synthetic full-model checks accompany the
separate four-way 35-case development comparison. Equations, scheduling limits
and results are in `docs/solution-adaptive-sampling.md`. Both controls remain
experimental and false by default; the 65 holdout outcomes stay closed.
The initial apparent speed gain used a flawed comparison of stored champion maps
and traded one recovered structure for another. The corrected policy checks live
maps and has no stable-map early stops on the 35 development cases. Its fresh
four-way control/adaptive/sampling/combined comparison recovers 14/14/14/13
structures, with 26 total counts and 15 element-count sets in every mode. Wall
medians are 3.393/3.819/3.973/4.558 seconds, but substantial host contention also
slows unchanged SHELXT and changes the cycles completed before deadlines, so
these are not isolated algorithmic slowdown estimates. A selected fixed-400-cycle
probe recovers the lost structure in all four modes, supporting a deadline-sensitive
failure. No useful end-to-end improvement is established; both controls stay off.

An additional execution-only experiment adds `score_only_evaluation` and
`reusable_fourier_workspaces`. The scalar scorer caches fixed moments/shell
populations and reuses scratch while preserving the full fused value/gradient
control. P1 phasing reuses map/mask/complex storage, prepared Gaussian geometry
and input vectors; contiguous FFT lines can run in place using the same f64 plan.
Supplied-symmetry recycling shares that contiguous-axis shortcut. Equations,
precision, support, ranking, refinement and input-derived caps remain unchanged.
Bitwise fixed-work trajectories, independent NumPy values, invalid boundaries
and full synthetic structures/counts accompany a balanced, serial interleaved
four-way comparison on all 35 development cases. All modes' chosen predictions
are saved before reference auditing. Both flags remain independently opt-in;
the 65 holdout outcomes stay unopened. Details and results are in
`docs/solution-execution-workspaces.md`.
The completed interleaved four-way development comparison preserves all 14/35
correct structures, 26 total counts and 15 per-element count sets in every mode.
All 31 returned models are exactly identical; four cases return no model as
before. Scalar scoring is 2.51x faster and combined 64-cycle kernel time is 8.5%
lower, but typical paired whole-solve time is essentially unchanged (1.0012x).
Overall wall medians are 3.405/3.334/3.336/3.291 seconds. Fixed wall-time slices
mostly spend the saved work on more phasing cycles (+2.36% combined), with the
same 95 starts and 104 full refinements. No end-to-end speed claim is made from
the lower unpaired median alone. The full schedule/audits are retained, and no
holdout outcome has been opened.

A further opt-in `work_sliced_phasing` policy limits the first half of phase-first
starts to half the requested cycles (floor 32), retaining full-quota fallback
starts and the original wall/global safety ceilings. With four starts and 800
requested cycles the work quotas are 400/400/800/800, independent of atom counts
or element identities. Existing symmetry/count/chemical/refinement validation
still decides acceptance; no plateau or fit target becomes a correctness
certificate. Per-start quota/cycle/time/reason audits and independent integer
references accompany unchanged phase/refinement equations. A paired serial
interleaved comparison on all 35 development cases keeps both fast execution
controls on, with all chosen predictions saved before auditing references.
Details and results are in `docs/solution-time-slice-scheduling.md`; the 65
holdout outcomes remain unopened.
The completed paired comparison recovers 15/35 structures versus 14/35, with no
lost success and unchanged 26 total-count/15 per-element-count recoveries. Phase
cycles fall 14.4%, phase time 8.0% and summed solve wall time 3.7%; the typical
paired wall ratio is only 1.00345x, so a broad latency gain is not established.
The raw wall medians are 3.297/3.398 seconds. Twenty-nine early starts finish their
short work quota, while 50 starts still reach the wall backstop. Extra fallback
starts/refinements partly offset savings; the option stays experimental and
disabled by default. All 140 predictions precede the two reference barriers,
and source/native hashes match across panels. The 65 holdout outcomes stay closed.

## Quality bar

An experimental `cached_assignment_refinement` option activates the existing
checked normal-matrix cache in short isotropic assignment fits. It requires
iterative and isotropic assignment refinement, stays false by default, and keeps
all competing labels, iteration limits, conservative guards and final anisotropic
fits. Original-Jacobian checks and original-solver fallbacks remain; screening
cache counters are separate from final-fit counters. New dense/iterative short-fit
checks reuse independent equations and finite differences, with full search
recovery compared on all 35 development cases. Diagnosis and results are in
`docs/solution-assignment-cache.md`; the 65 holdout outcomes remain unopened.
The completed matched comparison retains the same 19 structures, 26 total counts,
19 per-element counts and 15 strict structures, with unchanged recovery case sets
and inferred count vectors. Screening time falls 14.087 to 12.646 seconds (10.23%),
and summed solve wall time falls 70.908 to 69.393 seconds (2.14%). The median paired
wall ratio is only 1.00712x (0.71% less time); a broad typical-latency gain is not
established. All 675 screening fits and 125 final fits remain. There are 4,640
accepted cached screening steps and zero fallbacks. Full validation passes 2,209
Python and 414 Rust tests; all 140 predictions precede both reference barriers,
with matching native/source hashes. This is single-seed development evidence;
the option stays opt-in and the holdout remains closed.

An experimental `ambiguous_assignments` option adds displacement-compensated
single/coupled near-element relabels, bounded count-diverse isotropic competition
and retention of close requested-mode alternatives. It requires the existing
isotropic assignment path and is false by default. It keeps original parents,
observations, resource caps, scoring/correctness gates and final anisotropic
refinement. The original fixed-U proposal path is the explicit control. A close
final count competition can earn one corroborating start within the same budget.
The guarded follow-up fits the original assignment first, requiring agreement
of final rank and R1 before a challenger displaces it; metric tradeoffs remain
reported alternatives. Poor fits near the input-only atom cap defer extra label
exploration. The initial 18-versus-16 experiment with one lost control success
is preserved separately, rather than replaced by only the improved result.
Independent NumPy enumeration and complete synthetic checks accompany the
existing value/derivative equations. Equations and the 35-case comparison live in
`docs/solution-ambiguous-assignments.md`; the 65 holdout outcomes remain closed.
The fresh guarded 35-case comparison recovers 19 labelled structures versus 16,
with three gains and no lost success. Correct per-element counts rise 16 to 19,
total counts stay 26, and strict structures rise 13 to 15. The typical paired
wall ratio is 0.97029x (3.06% more time); summed solve wall time rises from
64.586 to 73.759 seconds. Raw wall medians are 1.349/1.854 seconds. Added bounded
competition performs 249 extra short fits, with final fits rising 96 to 125.
Fourteen metric disagreements are recorded; no corroborating starts are needed.
Full validation passes 2,199 Python and 414 Rust tests. All 140 predictions precede
two audit barriers with matching source/native hashes. This is single-seed
development evidence; the flag stays opt-in and holdout outcomes remain closed.

An experimental `multiplicity_weighted_refinement` option evaluates final
constrained fits on one representative per supplied symmetry/Friedel family,
using actual row multiplicities to retain the expanded-data objective, scale,
R1 and profiled derivatives. The original row/Jacobian/atom admissibility limits
apply before reduction; unequal amplitudes within a family are rejected. P1
phasing, independent-family isotropic screens and final ADP mode are unchanged.
The option is false by default and changes floating-point reduction order.
Independent weighted NumPy values, finite differences, gradient/normal comparisons
and complete synthetic model/count checks accompany the implementation. Equations,
benchmarks and the matched 35-case comparison are documented in
`docs/solution-multiplicity-refinement.md`; the 65 holdout outcomes remain closed.
The completed matched 35-case experiment retains the same 16 correct labelled
structures, 26 total counts, 16 per-element counts and 13 strict structures;
all inferred count vectors are unchanged. The median paired wall ratio is
1.15807x (13.65% less time; 30 faster cases), with summed solve wall time
81.998/63.808 seconds and final-refinement time 26.386/8.333 seconds. Both modes
perform 96 final fits, with treatment row counts reduced from 627,106 supplied
rows to 182,331 evaluated family rows. Numerical models differ in round-off.
Full validation passes 2,168 Python and 413 Rust tests. All 140 predictions precede
two audit barriers, with matching native/source hashes. This is single-seed
development evidence; the option remains opt-in and holdout outcomes stay closed.

An experimental `fused_phase_workspace` option caches exact Gaussian axis
geometry and fuses real-density transfers at packed-Fourier boundaries. It
requires reusable workspaces; packed real FFTs enable the transfer part. The
same Gaussian support/exponential/accumulation and FFT normalization order are
retained. A private geometry workspace adds at most 4.5 KiB, without a per-voxel
cache or additional full-grid buffers. Existing candidates, observations,
refinement, caps, seeds and scoring gates remain. Bitwise projection, complete
trajectory, direct-transfer and synthetic model/count checks accompany the
independent references. The option is false by default. The profile, initial
geometry-only probe and published SHELXT lessons are recorded separately in
`docs/solution-phase-workspace.md`; the 65 holdout outcomes remain unopened.
The completed matched 35-case experiment retains the same 16 labelled
structures, 26 total atom counts, 16 per-element counts and 13 strict structures.
All 31 nonempty selected models match. The median paired wall ratio is 1.06094x
(5.74% less time; 31 faster cases), with wall medians 1.874/1.803 seconds and
summed wall time 90.347/85.260 seconds. Phase time falls 13.87% while final
refinement remains around 28 seconds. Fixed-work phasing/small-search kernel
medians improve 10.0%/2.5%. Full validation passes 2,137 Python and 411 Rust tests;
all 140 predictions precede two reference barriers, with matching source/native
hashes. This is one seed and one timing collection; the option stays opt-in.

An experimental `real_fft` option targets the remaining dominant P1 Fourier
cycles. It packs even/odd real samples into half-length z transforms, runs x/y
FFTs on a half spectrum, restores the full reciprocal array before constraints,
and computes the real return map through an explicit Hermitian projection. The
original normalization, support classes, observations, f64 precision and all
search/refinement gates remain. It requires batched FFTs, is false by default,
and leaves initialization/Patterson/symmetry-recycling transforms unchanged.
The additional array workspace stays below 18 MiB at the existing maximum grid;
x/y plans are shared. Operation/axis order changes round-off, with independent
NumPy, short-trajectory, deterministic repeated-run and complete-model checks.
Fixed-work phasing and small-search benchmarks improve 28.2% and 8.5%; these are
separate from the matched 35-case experiment. Fresh paired panels retain the
same 16/35 labelled structures, 26 total counts, 16 per-element counts and 13
strict structures, including identical recovery case sets. The typical paired
wall ratio is 1.13421x (30 faster cases; 11.83% less time), and 1.16179x on the
same 16 successful cases. Summed wall time falls 96.077 to 86.990 seconds, and
P1 phase time 49.084 to 36.964 seconds (24.69%). Nonlinear trajectories change:
starts fall 98 to 96, screening fits rise 425 to 433, final fits fall 98 to 96,
and final-fit time rises 23.955 to 26.636 seconds. All original validation gates
remain. Actual packed-real transform count is 99,596. The preceding profile and
small copy-only prototype are retained separately. Details and results are in
`docs/solution-real-fft.md`; the 65 holdout outcomes remain unopened.

An experimental `cached_normal_refinement` option targets the final-fit linear
solver after the batched-FFT work. It caches a bounded (at most 8 MiB), f64,
diagonally preconditioned normal matrix per current Jacobian and checks each
returned CG step with the original Jacobian operator. It requires iterative
refinement, affects final requested-mode fits only, and preserves the original
matrix-free fallback, damping, iteration/residual criteria, observations,
analytic derivatives, physical constraints, line searches and resource caps.
Fixed eight-lane reductions change round-off order, with independent normal-
equation/residual and complete-fit checks. Screening remains unchanged. The
flag is opt-in. A follow-up extends blocked normal formation to 12–31-parameter
final models while retaining Cholesky, original damping norms and a checked
original-calculation fallback. Smaller models retain the old path. The complete
CG-only experiment/source patch is preserved separately. Fresh final paired
35-case panels retain the same 16 labelled structures, 26 total counts, 16
per-element counts and 13 strict structures. Final fitting falls 27.556 to 25.092
seconds (8.94%), while summed solve wall time falls 101.487 to 99.297 seconds.
The typical paired wall ratio is only 1.01876x (24 faster cases); among successful
cases it is 1.00421x. Thus the whole-search gain is modest despite 20.5%/44.5%
fixed-work small/large fit gains. All 425 screening and 98 final fits remain;
1,743 checked cache steps return without a successful fallback. Independent
numerical checks, 2,016 Python and 408 Rust tests pass. Protocol, validation and
results are in `docs/solution-normal-cache.md`. The 65 holdout outcomes remain
unopened.

An experimental `batched_fft` execution option applies planned contiguous FFT
batches after blocked small-slab line packing in P1 phasing and supplied-
symmetry recycling. It restores each group to XYZ layout and preserves original
f64 plans, z/y/x order, normalization, support and equations. A reusable eight-
line buffer adds at most 16 KiB per FFT plan under the unchanged grid limits. The
flag requires reusable workspaces and defaults to false. Direct FFT/NumPy,
bitwise phase trajectories and synthetic complete models/counts accompany
realistic Fourier/full-search benchmarks and a complete paired 35-case
development experiment. The fresh paired panels retain the same 16/35 labelled
structures, 26/35 total counts and 16/35 per-element counts (13 strict structures).
Batching is faster on 30/35 cases: the typical paired wall ratio is 1.07339x,
with wall medians 2.582/2.362 seconds and summed wall time 106.100/98.776 seconds.
P1 phase time falls 12.73% while scored cycles rise 7.75%; fixed-work phasing and
whole-search microbenchmarks fall 26.2% and 11.9%. Direct transforms and fixed-work
trajectories match bitwise; deadline-limited selected models agree in 28/31
nonempty cases, with no recovery-set changes. The initial slower whole-grid
layout is preserved with its benchmark and reproducible patch. The flag stays
opt-in. Details are in `docs/solution-batched-fft.md`; the 65 holdout outcomes
remain unopened.

An experimental `filtered_peak_search` path targets the dominant remaining P1
phase work. It computes the unchanged RMS threshold before local-maxima tests,
skips voxels that cannot survive any raised threshold, and caches periodic axis
offsets in O(sum(axis lengths)) memory. All 26 original neighbor values, index-
stable plateau ties, threshold equality, sub-grid interpolation, Gaussian support
and fixed-cycle trajectories are preserved. The original scan remains the
execution control; both paths have independent NumPy and bitwise native checks.
The flag requires reusable workspaces and defaults to false. A source-hashed
profile, realistic scan/Fourier/full-search benchmarks and a complete paired
35-case development comparison are documented in
`docs/solution-filtered-peak-search.md`. The 65 holdout outcomes stay unopened.
The completed same-build interleaved comparison preserves the same 16/35 correct
structures, 26 correct total counts, 16 correct element-count sets and 13 strict
recoveries. Typical paired wall time now improves 1.18703x (about 15.8% less
elapsed time), with 34/35 cases faster and all 35 native solver times lower.
Raw wall medians are 3.418/2.543 seconds; cumulative time falls 15.2% and phase
time 23.7%, despite 17.4% more scored cycles. Short quota completions rise 26-to-43
under unchanged limits. The scan is 6.76x faster and fixed 64-cycle phase time is
41.0% lower in the realistic benchmark. All 38 source hashes/native builds match,
and all 140 predictions precede reference auditing. Fixed-work trajectories are
bitwise equal; additional deadline-limited cycles can change chosen maps. The
flag remains opt-in and the 65 holdout outcomes remain unopened.

An additional experimental `isotropic_assignment_refinement` path gives the best
two complete label models longer isotropic relaxation on the existing bounded
independent-family observations,
then tests two bounded single/swap alternatives, including near ties that can
improve after geometry/U relaxation. At most four comparable isotropic candidates
compete before one requested-mode final fit; a close runner-up remains available
when that fit misses the existing target. Original count proposals, final ADP mode,
ranking/gates and resource caps remain explicit. The flag requires phase-origin
selection and chemical assignment and defaults to false. Independent enumeration,
synthetic counts/geometry/final-mode checks and a complete paired 35-case
development comparison are specified in `docs/solution-isotropic-assignments.md`.
All predictions precede reference auditing; the 65 holdout outcomes remain closed.
The first all-row prototype gains one structure/count set but is typically 11.6%
slower, so its reports and reproducible patch are retained. The revised common-
family comparison recovers 16/35 structures versus 15/35, retains every control
success, and improves per-element counts 15-to-16 with total counts unchanged at
26. Anisotropic fits fall 106-to-100 and their stage time 29.3%; added isotropic
work leaves a 5.0% cumulative wall saving. Typical paired whole-solve time remains
essentially unchanged (0.99591x control/new; raw medians 3.362/3.360 seconds).
Both panels preserve final anisotropic fits on all original observations, and
all 140 predictions precede reference auditing. The flag stays opt-in; the 65
holdout outcomes remain unopened.

An opt-in `specialized_isotropic_refinement` execution path hoists canonical
isotropic tensor/cache dispatch outside the structure-factor loop and evaluates
values and derivatives together with unchanged per-reflection reduction order.
It retains the generic evaluator for anisotropic models, noncanonical tensor
bases and unavailable or mutated geometry caches. The independent NumPy equations
and centered derivatives remain authoritative cross-checks. Actual eligible-fit
counts and requested flags are distinct. Profile evidence, rejected tiled/Gram
prototypes and a paired complete 35-case development protocol are recorded in
`docs/solution-isotropic-evaluation.md`. The matched run preserves all 19
successful cases, all 35 model records and count vectors (26 correct total counts,
19 per-element counts, 15 strict structures). Screening time falls 6.58%; summed
wall time changes only 70.044-to-69.438 seconds (0.87% less), with a paired median
ratio of 1.00961. The small overall gain remains host-sensitive despite 16.5%
lower isolated evaluation cost. The flag stays opt-in; the holdout remains closed.

An opt-in `streamed_peak_search` path targets P1 adaptive-map statistics with
fixed eight-lane RMS accumulation and a conservative threshold interval.
Ambiguous or non-normal thresholds recompute the original scalar statistic;
streamed local maxima retain all possible scalar survivors and all 26-neighbor
tie rules. Exact peak/phase decisions, current refinement, grid caps and search
budgets remain explicit. The option requires filtered peak search and reports
actual selections and scalar fallbacks. Equations, error bounds, independent
checks, a set-aside FFT prototype and the complete development protocol are in
`docs/solution-peak-statistics.md`. The matched run preserves all 19 structures,
all 35 inferred count vectors/model records, 26 correct total counts, 19 correct
per-element counts and 15 strict structures. P1 phase time falls 30.978-to-29.634
seconds (4.34%) and summed wall time 69.511-to-67.610 seconds (2.74%), with a paired
median ratio of 1.01977. The gain remains modest and host-sensitive. The execution
flag stays false by default; the holdout remains closed.

An opt-in `reuse_normal_matrices` path retains one checked CG normal linearization
while rejected/backtracked steps change only damping. Accepting any model clears
it before changing the evaluation; separate fits and candidates never share it.
The original matrix-free residual guard, fallback, scaling, line search, input
caps and stopping rules remain intact. It requires iterative refinement and at
least one normal-cache control, and reports actual preparations/reuses separately
from requested execution. The retained bounded matrix can add up to 8 MiB plus
small vectors during backtracking. Equations, exactness/lifetime tests and the
complete development protocol are in `docs/solution-normal-reuse.md`. The matched
run retains all 19 structures, all 35 model records/count vectors, 26 correct
total counts, 19 per-element counts and 15 strict structures. Reuse avoids 882
of 6,215 eligible preparations. Final refinement time falls 14.00%, screening
3.61%, and summed wall time 69.783-to-67.437 seconds (3.36% less). The paired
median ratio is only 1.00539; typical-case time remains almost unchanged. The
option stays false by default and the holdout remains closed.

An opt-in `short_phase_starts` experiment targets typical-case P1 work after
matrix reuse showed no opportunity in 27/35 cases. It requires work-sliced phasing
and allocates quarter work with a 32-cycle floor to the first half of starts,
retaining full-length later starts, original wall/resource limits and every
model-validation gate. Its equations are unchanged; selected maps can change.
Independent integer references, exact phase-prefix checks, a set-aside FFT
prototype and the complete development protocol are recorded in
`docs/solution-short-phase-starts.md`. The paired development comparison retains
the same 19 successful structures, 26 correct total counts, 19 correct element
counts and 15 strict structures. Typical paired time falls 9.39%, summed wall
time 64.248-to-58.278 seconds (9.29%), and P1 time 19.82%, with 32/35 cases faster.
One failed case changes its incorrect C/N count assignment; no successful count
or structure case is lost. The option remains false by default and the holdout
remains closed.

An opt-in `blocked_normal_matrices` experiment forms eligible reused normal
matrices in 32-by-32 SIMD tiles through pinned, single-threaded ndarray. It keeps
the original damped equations and original-Jacobian residual guard, retries
legacy Gram formation on failure, and leaves the powder core unchanged. Safe
borrowed views avoid a second full matrix; backend packing storage is explicit.
Independent NumPy equations, partial-tile/guard tests, full physical fits and the
balanced 35-development-case protocol are documented in
`docs/solution-blocked-normal-matrices.md`. The complete paired comparison retains
all 19 structures, every one of the 35 count vectors, 26 correct total counts,
19 correct element counts and 15 strict structures. Summed wall time falls
61.649-to-58.625 seconds (4.90% less), typical paired time 2.49%, final refinement
20.77% and screening 6.43%. No blocked-system fallback is needed in completed
fits; 29/35 cases are faster. The option remains false by default and the holdout
remains closed.

Two opt-in stopping controls test high-quality phase plateaus on early starts
while retaining full-length later starts, and repeated negligible accepted
improvements in short isotropic assignment screens. Final refinement convergence
and physical equations remain unchanged. Independent references, exact work
provenance, solver-only SHELXT listing comparisons and a balanced four-mode
35-development-case protocol are documented in `docs/solution-search-stopping.md`.
The complete four-mode comparison retains every previous successful case set
and all 35 atom-count vectors. Combined stopping reduces summed library solve
time 52.315-to-49.477 seconds (5.42%), summed wall time 4.76%, and typical paired
solver time 4.94%. It records 22 plateau handoffs and 367 converged assignment
screens. All 19 structures, 26 correct total counts, 19 element counts and 15
strict structures remain. The 34 completed SHELXT pairs list 108 trials and
12,106 iterations versus 35,621 completed PhaseSmith cycles after this change;
iteration definitions and refinement endpoints differ. Both controls remain
false by default and the holdout remains closed. The collector's final round-
count assertion was corrected after collection with original snapshots retained;
all 280 predictions and four audit barriers are verified.

Experimental element-only solving now accepts an explicit `workers` budget
(default one, 1..64 capped to available logical CPUs). One solve-owned reusable
pool partitions fused reflection value/derivative evaluation; serial reductions,
seeded phase starts, candidate ordering and stopping stay ordered. Numerical
state and pools do not escape the call. This reuses `phasesmith-execution` without
installing a global pool or changing the powder core. Extra tile storage is one
bounded Jacobian rather than one full model per worker. Independent equations,
bitwise worker-count checks, lifetime tests and the 1/2/4/8-worker latency and
cold independent-process throughput protocol are in
`docs/solution-bounded-workers.md`. Concurrent callers must divide an application
budget across solves; throughput experiments use one inner worker per outer job.
The complete 35-development-case 1/2/4/8 comparison retains the exact prior 19
structures, all 35 inferred count vectors, 26 correct total counts, 19 element
counts and 15 strict structures in every latency and batch panel. Four internal
workers reduce summed library solve time 38.269-to-34.412 seconds (10.08% less).
Eight independent single-worker jobs reduce cold batch elapsed time from 46.546
to 9.967 seconds (4.670x throughput); SHELXT's corresponding batch takes 17.396
seconds, including one timeout. SHELXT's multi-threaded single solves scale more
strongly but recover 13/14/13/10 structures at 1/2/4/8 threads, so timing and
recovery must be read together. All 560 predictions precede reference grading.
The worker default remains one and the holdout remains closed.

A further opt-in cycle-effort experiment separates long intermediate-start
plateaus from agreement between independent complete models. The latter checks
labelled full-cell geometry, fit/ambiguity gates and actual random-start identity
instead of confusing origin lineages with independent solutions. It reports a
separate consensus termination without changing whether the requested R1 target
was met. The final full-length fallback remains available when acceptance fails.
Independent geometric references, start-provenance tests and a four-mode
35-development-case protocol are recorded in `docs/solution-cycle-effort.md`.
All four development variants retain every prior success set and all 35 atom-
count vectors. Combined policies reduce completed cycles from 38,120 to 23,242 (39.03%),
summed solve time from 46.317 to 35.598 seconds (23.14%), and summed wall time by 20.36%.
The policies fire in 14 cases; typical paired solver time improves only 1.00%.
Nine consensus stops include three already incorrect interpretations, so the
new termination is not a correctness certificate and does not mark an unmet R1
target as met. All 19 structures, 26 correct total counts, 19 element counts and 15
strict structures remain. Both controls remain false by default and the holdout
remains closed.

Public behavior is typed and documented. Invalid shapes, non-finite values,
unsorted grids, non-positive widths, and fractions outside `[0, 1]` fail with a
clear error. Tests are deterministic. Numerical tolerances are justified near
the assertion. Formatting, linting, unit tests, Python tests, and benchmarks are
available as ordinary project commands.

The dependency-ordered delivery plan and milestone exit gates are maintained in
`IMPLEMENTATION_PLAN.md`.

The opXRD corpus is now onboarded as an opt-in robustness/capability campaign,
not a bulk Rietveld oracle. The exact non-time-series archive and 14 fixed
members are checksum-pinned across all nine archive directories. Source axes
and intensities are never sorted, resampled, shifted, or clipped: five
nonuniform-grid background rejections and two nonmonotonic-grid rejections are
reported as observable boundaries, while negative observations disable only
the inapplicable Poisson metric. Three fully labelled single-phase cases also
run a disclosed expanded-P1 common model in PhaseSmith and exact-revision
GSAS-II. A bounded global-zero scan and zero-only polish now precede width
refinement. This preserves close Li2TeC2 agreement and closes the former ZrC
optimizer gap. NbS2 instead requires the same approximately +0.521 degree
offset in both programs and remains an explicit calibration warning; GSAS-II's
lower residual is accompanied by a 1.65 degree maximum instrument Gaussian
FWHM and is not treated as physical superiority. Residual and parameter-
plausibility evidence remain separate. No production numerical behavior or
GSAS-II runtime boundary changes.

An opt-in `joint_assignment_refinement` experiment adds bounded post-search
categorical repair using the fitted Cartesian tensors, bounded scalar-shift
optimization and neighbor-derived displacement seeds, whole-orbit vacancies and
up to two difference-map orbit locations. A width-four discrete beam and at most
three two-fit rounds share a two-second safeguard and the original deadline.
Physical tensors seed the existing invariant log-SPD refiner; only candidates
with non-worse final rank/R1 and at least one-percent lower amplitude objective
can replace the retained original. Counts remain inferred and original atom/data
caps stay unchanged. Equations, independent references and the development-only
protocol are documented in `docs/solution-joint-assignment.md`; holdout remains
closed and this feature remains off by default.
The complete matched all-35 comparison retains the exact previous 19 structures,
26 formerly correct total-count cases, 19 per-element-count cases and 15 strict
structures. Correct total counts increase to 27 through a missing four-atom orbit;
three incorrect element-count vectors move closer to their references. Exact
per-element and full labelled-structure recovery remain 19. Across 31 returned
models, full-cell count-vector L1 error falls 210-to-182. The new stage adds 5.193
seconds, 18 full fits and five accepted rounds; summed library solve time rises
41.741-to-46.620 seconds (11.69%), typical paired time 4.25%. All 140 predictions
precede development grading; seven above-cap references remain in the denominator.
The option stays experimental/off by default, and the 65 holdout cases stay closed.

An execution-only `cached_repair_evaluation` option reuses bounded immutable
categorical columns, two scalar reflection fields, line-search buffers and
coherence-neighbor preparation inside joint assignment repair. Original equations,
reduction order, proposals, work budgets, physical fits and acceptance remain.
The column cache caps payload at 4 MiB and falls back to direct multiplication;
reflection buffers add at most 208 KiB. Caches expire before full physical fits,
with no global numerical state or additional threads. Independent equations,
finite differences, bitwise original/fast comparisons and a complete matched
35-development-case protocol are described in `docs/solution-repair-evaluation.md`.
The option requires joint assignment repair, remains off by default, and the
65 holdout cases remain closed.
The matched all-35 three-mode collection retains the exact 19 structures,
27 correct total counts, 19 correct per-element vectors and 15 strict structures;
all 35 count vectors and all logical repair work match the preceding repair.
Repair stage time falls 4.477-to-3.466 seconds (22.59%); summed library time falls
39.605-to-38.566 seconds (2.62%) and process time 46.880-to-45.639 seconds (2.65%).
Median paired solver time improves only 0.26%. Relative to the same-collection
35.118-second no-repair reference, added solver overhead falls 12.78%-to-9.82%.
The isolated categorical kernel improves 35.125-to-17.479 ms (2.01 times).
All 210 predictions precede grading; seven above-cap references and four no-model
cases stay in the denominator. Experimental defaults and the closed holdout stay
unchanged. Provenance is in the `structure-solution-repair-evaluation-*` reports.

The user has now authorized a single frozen evaluation on the 65 holdout cases.
The numerical implementation and release extension remain fixed at `870d61a`;
the precommitted protocol pins the accepted cached-repair development controls,
one worker per program, original inputs/caps/thresholds and 13/15-second budgets.
All 130 predictions precede grading; source/binary hashes are checked throughout.
Algorithm attribution cites the published Sheldrick (2015, 1990) methods and
documents independent PhaseSmith choices. SHELXT is solely an external test
comparator. See `docs/solution-holdout-freeze.md` and the immutable freeze record.
Once evaluated, these 65 cases are exposed validation evidence and must not be
described as an untouched test of future tuning.

The frozen holdout evaluation is complete: all 130 predictions preceded grading,
with zero numerical tuning or search reruns. All 402 pinned source files and the
release binary remain unchanged. On all 65 cases, PhaseSmith recovers 35 full
structures, 40 exact total counts, 36 per-element vectors and 34 strict structures;
the external test comparator recovers 18, 51, 22 and 11 respectively. Median wall
times are 0.683/0.782 seconds and summed wall times 83.717/93.209 seconds; the
median paired PhaseSmith/comparator ratio is 0.9721. There are zero/two hard
timeouts. All 24 above-cap references remain in the primary denominator; among
the 41 within-cap cases, PhaseSmith recovers 35 structures and 40 total counts.
There are no audit errors or post-reference model changes. Source/binary/input
hashes, fixed worker controls and comparator thread logs are verified. Results,
limitations, attribution to the published papers and immutable provenance are in
`docs/solution-holdout-results.md`. The 65-case holdout is now exposed validation
evidence, with no claim that it remains untouched for future algorithm changes.

Integration validation now checks the historical holdout's 402 source hashes
against Git blobs from its recorded protocol commit and keeps its native-binary
hash tied to the recorded result. Later merged code need not impersonate the
frozen executable. The actual holdout runner retains its strict live-source and
binary checks and will reject changed code. No historical freeze, predictions,
counts or grading thresholds are rewritten by integration.
