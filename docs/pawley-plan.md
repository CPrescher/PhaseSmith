# Pawley refinement implementation plan

Status: first CW implementation delivered in this worktree; the original plan
below remains the broader roadmap. See [the user guide](pawley.md) and
[validation evidence](pawley-validation.md) for the actual supported boundary.

Implementation deviations: objective and contracts share `pawley.rs`; a bounded
dense QR/SVD solver is delivered before matrix-free products. Native/Python
persistence uses a distinct standalone JSON format, without migrating existing
mixed-method bundles. A supplemental unreleased Python API snapshot preserves
the released 0.5.0 snapshot. Existing GSAS-II CW fixtures validate profile/area
conventions; live Pawley optimizer parity remains an explicit deferred gate.
P6 spectra and P7 TOF are not implemented.
Prepared 2026-09-17 against commit `28a10e5` on the separate
`codex/pawley-implementation-plan` worktree. The original checkout contains
uncommitted numerical/performance changes that are deliberately not included.
Reconcile those changes before merging, particularly profile accuracy,
solver controls, reflection-family conventions, and persistence versions.

## Delivered scope versus the original roadmap

| Slice | Current state |
| --- | --- |
| P1 | Equations, independent NumPy reference and explicit area/constraint contracts implemented. |
| P2 | Dense bounded extraction, overlap/rank diagnostics and exhaustive small-problem validation implemented. Matrix-free solving remains deferred. |
| P3 | Analytical joint cell/profile/background refinement and accepted-state runtime implemented. |
| P4 | Native/Python API, examples, standalone version-1 persistence and supplemental API snapshot implemented. Mixed-method bundle integration remains deferred. |
| P5 | Regression tests, pinned profile fixture, measured gates and dense benchmark runner implemented. See the validation report for passing and failed measured gates; full live optimizer parity remains deferred. |
| P6–P7 | Fixed wavelength spectra and TOF remain future work. |

## Goal and first release boundary

Add a first-class, Python-free Pawley workflow with a thin NumPy facade.
Users supply an indexed cell and symmetry or explicit reflection families;
atomic coordinates and structure factors are unnecessary. Pawley fits
reflection intensities as least-squares parameters together with selected
lattice, profile, and background parameters. Le Bail uses intensity
redistribution; Rietveld derives intensities from a structural model.

The methodological source is G. S. Pawley, “Unit-cell refinement from powder
diffraction scans”, *Journal of Applied Crystallography* **14**, 357–361 (1981),
[doi:10.1107/S0021889881009618](https://doi.org/10.1107/S0021889881009618).
Its abstract identifies overlap ill-conditioning and constraints as central
issues. The contracts and implementation choices below are PhaseSmith design
proposals, not claims to reproduce the original algorithm in every detail.

The first complete milestone covers one CW histogram, one or more phases,
monochromatic X-ray/neutron data, symmetric TCH or existing FCJ profiles,
fixed wavelength, selected U/V/W/X/Y and symmetry-independent cell parameters,
fixed plus linear refinable background, reflection constraints, runtime
recovery, native persistence, and Python/Rust examples. Fix axial geometry in
this first milestone. Fixed wavelength-component spectra follow in a separate
slice, then single-bank and multi-bank TOF.

Indexing, space-group search, atom refinement, structural restraints,
quantitative phase fractions, automatic model selection, and automation recipe
integration are outside this milestone. Pawley residuals and extracted areas
do not establish a unique structure or yield Rietveld phase mass fractions.
Independent peak positions are also excluded from the initial public workflow:
positions follow the supplied cell and instrument model.

## Scientific contract

For included samples, define

\[
\begin{aligned}
y_i^{\mathrm{calc}} &= b_i^{\mathrm{fixed}} + b_i(\beta)
  + \sum_k I_k p_{ki}(q), \\
r_i &= \sqrt{w_i}(y_i^{\mathrm{calc}}-y_i^{\mathrm{obs}}), \\
\Phi &= \tfrac12\sum_i r_i^2, \qquad
w_i=\sigma_i^{-2}\ \text{or}\ 1.
\end{aligned}
\]

Here `k` includes phase and reflection-family identity, `q` contains nonlinear
cell/profile parameters, and `beta` contains background coefficients. Use the
existing residual convention and masks. Bin widths enter area diagnostics,
not an additional least-squares weight. Do not inherit Le Bail's redistribution
weights or clip negative observations/background-subtracted data.

`I_k` is the integrated area of the complete powder family before finite
support truncation, in observed-y times axis units (degrees for CW, microseconds
for TOF density data). It is not F-squared. Multiplicity, LP, preferred
orientation, and other fixed amplitude effects are absorbed into that area;
do not apply them a second time. Reflection generation still uses symmetry and
systematic absences. Freeze the family/Friedel convention in the request and
oracle translation, especially when reconciling ongoing reflection work.

Fix phase scales at one: freely fitted phase scale times freely fitted
intensities is an exact gauge ambiguity. Reject a request that selects both.
Do not include structure factors, occupancy, ADPs, or amplitude-only sample
corrections as Pawley parameters. Preserve phase IDs for reporting without
interpreting phase sums as mass fractions.

Default to non-negative intensities using explicit bounds. Also provide an
explicit signed-intensity option for unconstrained extraction/oracle studies;
never implement positivity by clipping a fitted result or squaring an
unconstrained parameter. Record the selected policy in results and checkpoints.
A zero initial intensity must be able to leave its lower bound.

### Derivatives and support

Use existing fused profile values and analytical derivatives:

\[
\frac{\partial y_i}{\partial I_k}=p_{ki},\qquad
\frac{\partial y_i}{\partial q_j}
=\sum_k I_k\frac{\partial p_{ki}}{\partial q_j},\qquad
\frac{\partial y_i}{\partial\beta_j}
=\frac{\partial b_i}{\partial\beta_j}.
\]

Apply the existing exact parameter-constraint transform to the complete
Jacobian. Reuse the native cell-to-d-spacing-to-position/width chains.
The intensity column comes from the unit-area profile, including at `I_k=0`;
it must not be reconstructed by dividing a calculated contribution by `I_k`.

Inherit each kernel's finite-support convention without renormalizing to the
observed grid. Symmetric support includes both physical endpoints; FCJ and
later TOF retain their documented asymmetric limits. Jacobians hold support
membership fixed; derivatives at moving cutoffs are undefined. Trial values
use the trial support. Test exact endpoints separately from centered
differences away from support changes. Area/moment tests must account for
finite tails and clipped observation windows; Lorentzian infinite-domain
variance is not a finite normalization target.

### Overlap, constraints, and uncertainty

Stable intensity keys include phase and family identity, never array offsets.
Reuse exact fixed/affine/linear constraints for explicit intensity ties or
fixed ratios, validate feasibility with bounds, and expose all applied ties.
Do not silently infer equal intensities for nearby reflections.

Analyze the undamped weighted Jacobian for identifiability. Damping may make a
step computable, but cannot make unresolved intensities identifiable. Exact
coincidences can have an identifiable group sum while their individual areas
remain undetermined. Report that distinction, correlations and rank thresholds;
use deterministic ordering/tie breaking for any chosen representative solution.
Near-overlap remains separate from exact equivalence. Initial ratios are not
evidence of resolved intensities.

A reflection with no included support is inactive: retain its stored value,
exclude it from the solve and fitted degrees of freedom, and report it as
unobserved. Empty masks or an entirely uninformative problem fail clearly.
Bound-active, rank-deficient, or budget-limited uncertainty calculations must
return explicit limitations rather than plausible-looking zero errors.

For interior full-rank fits use the complete intensity/background/nonlinear
Jacobian for covariance, including cross terms. Known sigmas use unscaled
inverse information; unit weights use the existing reduced-chi-square scale.
Report selected/free/interior parameter counts and numerical rank separately.
Do not silently redefine shared goodness-of-fit metrics for active bounds:
state when reduced chi-square and Gaussian errors are only approximations.
Large problems may request selected covariance blocks; never allocate a full
reflection-squared matrix without a configured budget.

## Architecture and reuse

| Owner | Planned work |
| --- | --- |
| `crates/phasesmith-core` | Reuse CW/FCJ support blocks, fused accumulation, intensity/local derivatives and analytical products. Add only missing generic primitives, with independent tests. |
| `crates/phasesmith-crystallography` | Reuse cells, settings, absences, conservative reflection domains and stable family identities. No scattering calculation is required. |
| `crates/phasesmith-workflows` | Add `pawley.rs`, `pawley_objective.rs`, `pawley_solver.rs`; compose `parameters.rs`, `constraints.rs`, `lattice.rs`, `backgrounds.rs`, `residuals.rs`, and `runtime.rs`. |
| `crates/phasesmith-model`, `crates/phasesmith-persistence` | Typed Pawley analysis/project state and versioned wire records, distinct from Le Bail and Rietveld analyses. |
| `crates/phasesmith-py`, `crates/phasesmith-rs` | Thin native bindings and public facade exports, with native examples and no Python requirement for Rust consumers. |
| `python/phasesmith/refinement/pawley.py` | Proposed `PawleyInput`, `PawleyOptions`, `PawleyResult`, `calculate`, `refine`, and parameter construction helpers. Exact signatures are settled with the contract tests. |
| `python/phasesmith/pawley_reference.py` | Readable, independent NumPy objective, Jacobian, and small dense reference solve. It must not call Rust or wrap Le Bail iteration. |
| `python/phasesmith/oracle`, `oracle`, `tests`, `benchmarks` | Version-gated black-box probes, provenance, differential tests, and realistic fit benchmarks. |

Le Bail's phase/domain code is a reuse candidate, not a required public input
type. Extract small neutral helpers only where actually shared; preserve the
existing Le Bail API and numerical behavior. Likewise, reuse solver primitives
without forcing Pawley into a structural `RietveldInput` or copying the complete
Rietveld solver. Keep workflow state out of `phasesmith-core`.

## Dependency-ordered implementation slices

### P1 — Contract and independent reference

Write the canonical equations, conventions, typed request/result proposal and
small deterministic fixtures before production changes. Implement a dense
NumPy reference with intensity/background columns and analytical profile/cell
chains. A test-only enumerated active-set solve can validate small non-negative
linear cases without introducing SciPy as a runtime dependency.

Exit: isolated, overlapping, exactly coincident, masked, negative-observation,
zero-intensity and invalid-input cases have reviewed expected behavior. Fix
numerical rank and convergence tolerances locally in these tests before running
real-data acceptance. Pin the oracle revision and audit accessible Pawley
scripting operations now; document unsupported probes rather than postponing
the question until release.

### P2 — Fixed-geometry native extraction

Build a native fixed-cell/profile objective and solve intensities and linear
background jointly by bounded weighted least squares. Use a deterministic
active-set method with rank-revealing QR/SVD for the bounded dense baseline;
reuse existing linear algebra where its rank and bound contracts suffice.
Check KKT/projected-gradient residuals, including release from zero bounds.
For signed intensities use the corresponding unconstrained least-squares path.
Expose calculation, intensity results, fit metrics and overlap diagnostics.

Exit: Rust matches the independent NumPy objective, solution on identifiable
cases, and KKT residuals; singular cases compare calculated patterns/group
sums instead of arbitrary intensity splits. Native code runs without Python.

### P3 — Joint CW parameter refinement and runtime

Add selected U/V/W/X/Y and symmetry-independent bounded cell parameters to the
same least-squares vector as intensities/background. Use bounded damped
Gauss–Newton with the full Jacobian and complete-objective backtracking.
P2 supplies initialization and the fixed-geometry special case. All active
families move under one acceptance decision; do not substitute Le Bail updates.

Start with a documented dense memory limit and explicit rejection above it.
Then add support-block JVP/VJP and bounded iterative steps for larger workloads,
validated against dense steps. A Schur-complement or variable-projection
optimization is deferred until an independently tested derivative and bound-
active-set contract exists; it is not necessary to ship the first correct solve.
Large-mode rank diagnostics must disclose their scope and budget limitations.

Use one conservative generated family superset over the allowed cell bounds
for the first release. Preserve IDs/order throughout trials and checkpoints;
activate columns by current observed support without deleting guard families.
Reject out-of-domain trials. Require an explicit restart for bounds/domain
changes. Accepted-state dynamic regeneration is a later optimization only if
benchmarks demonstrate the need and state-transfer/objective semantics are tested.

Wire budgets, events, cancellation, rejection reasons and last-accepted
checkpoints from `runtime.rs`. Checkpoints bind data/weights/masks, support,
identity, bounds, ties, domain, solver mode, active set and next damping state.
Discard incomplete trials; count expensive work during linear solves too.

Exit: synthetic lattice/profile/intensity recovery, derivative/product/adjoint
tests, convergence and failure distinctions, exact cancellation/resume, and
repeatability across supported worker counts. Compare existing Le Bail and
Rietveld regression cases when shared helpers change.

### P4 — Public APIs, projects and persistence

Add the Rust and NumPy facade, GIL release for long native operations, immutable
result arrays, and a constructor from cell plus symmetry. An optional CIF
constructor extracts only crystallographic metadata; it must not require a
structural intensity model. Validate dtype, shape, sorted grids, finiteness,
IDs, bounds, symmetry and profile validity at the boundary.

Add a distinct native Pawley analysis record with calculation/refinement,
checkpoint continuation and plain-data reports. Choose the next available
persistence version at implementation time; preserve old readers' explicit
future-version rejection and migrate old files with no Pawley analysis.
Round-trip constraints, intensity policy, uncertainties, inactive families,
algorithm controls and accepted state. Cross-language load/resume is a gate.
Do not serialize an implementation-specific matrix factorization.

Exit: complete Python and Rust examples, API contract tests, native/Python
round trips, corrupt/stale checkpoint rejection, and compatibility fixtures.

### P5 — Scientific acceptance, performance and release documentation

Use the checksum-pinned APS sucrose CW dataset first, followed by an independent
CW-neutron case such as the existing Echidna LaB6 dataset. Reuse raw data
provenance, not historical Le Bail output as Pawley truth. Fix data range,
weights, background, support, radiation, reflection list, bounds and constraint
policy before evaluating fits; predeclare per-dataset acceptance thresholds in
a reviewed manifest. Add a synthetic multiphase severe-overlap case.

Compare plain arrays against a pinned GSAS-II Pawley run, documenting any
F-squared/multiplicity/LP-to-area conversion. First compare fixed-state patterns
and derivatives where exposed, then fitted patterns, cells, isolated intensities
and resolvable group sums. Record negative-intensity/constraint differences.
A lower Rwp alone is not parity. If the oracle cannot cover a feature, keep the
limitation explicit and provide independent numerical evidence; do not claim a
passed oracle gate or silently regenerate golden data.

Benchmark fixed-geometry and complete joint fits with at least 256 reflections
and 10,001 samples, plus the sucrose-scale approximately 811-family workload.
Include heavy overlap, FCJ, multiphase and dense/matrix-free paths. Report
median/p95, peak memory, evaluations, termination, residuals and rank scope;
record hardware, thread counts, build and raw results. No speedup is assumed.
Run existing hot-loop benchmarks if shared profile code changes and report
regressions without weakening scientific tolerances.

Exit: all numerical, runtime, persistence, documentation and realistic
benchmark gates pass. Only then describe CW Pawley as supported and add its
public API snapshot under the repository's version policy.

### P6 — Fixed CW spectra

Compose component-specific positions and shapes into each family's unit-area
basis with explicit fixed weights summing to one. Tie components through one
family intensity; specify that weights describe effective detected areas and
are not inferred from structural LP/scattering. Validate that contract against
any oracle conversion. Do not duplicate independent intensity parameters for
K-alpha components. Keep component wavelengths/weights fixed initially.

Exit: one-component equivalence, doublet overlap and derivative tests, union
domain coverage, persistence, and a dedicated measured-data/oracle comparison.

### P7 — TOF extension

First add a single bank with explicit microsecond/density semantics, fixed
calibration and the existing asymmetric kernel. Then selected calibration/cell
motion and a genuine joint multi-bank objective: share cell/symmetry, keep
intensities/background/profile terms bank-local by default. Sharing areas across
banks requires a separate calibrated correction/normalization contract.
Surface cell/DIFC and other calibration degeneracies; require identifiable
selections. Preserve the existing incident-spectrum provenance rules.

Exit: nonuniform-grid derivative/support checks, single-bank reduction,
atomic joint acceptance/resume, native persistence, and independently reviewed
POWGEN and LANL nickel gates. P7 does not block a clearly labelled CW release.

## Documentation work by delivery stage

| Stage | Documents and required changes |
| --- | --- |
| Planning (this change) | Add this plan; update `PROJECT_BRIEF.md`, `IMPLEMENTATION_PLAN.md`, the documentation index and MkDocs navigation to mark it planned. |
| P1–P3 | Add Pawley equations to canonical `crates/phasesmith-rs/src/guide/mathematics/refinement.md`; update `scripts/sync_math_docs.py` mappings and regenerate `docs/mathematics/refinement.md`. Explain area convention, signed/bounded fits, overlap, covariance, masks, support and degrees of freedom. |
| P4 | Add `docs/pawley.md` with runnable Python/Rust examples, cell/symmetry inputs, parameter selection, stop/resume and diagnostic interpretation. Update `docs/refinement.md`, `docs/refinement-runtime.md`, `docs/public-api.md`, `docs/api-reference.md`, `docs/rust-api.md` and the Rust refinement operations guide. |
| P4 | Update `docs/persistence.md`, `docs/native-persistence.md`, applicable schemas and API snapshot tooling for distinct Pawley state and migrations. |
| P5 | Update `README.md`, `docs/index.md`, `mkdocs.yml`, `docs/lebail.md`, `docs/rietveld.md`, `docs/rietx-comparison.md`, validation reports, `oracle/README.md` and `CHANGELOG.md`. Explain method choice and measured limits; publish the complete supported-feature matrix. |
| P6–P7 | Extend guides/examples and feature matrix only after spectrum/TOF gates pass. Keep each geometry's observation, correction and intensity conventions explicit. |

Every implementation slice updates its status in this plan and the main brief.
Do not hand-edit generated mathematical pages or advertise proposed API names
as importable before P4. Keep historical validation results unchanged.

## Required checks

For numerical slices run `cargo fmt`,
`cargo clippy --workspace --all-targets --all-features`,
`cargo test --workspace --all-features`, and `pytest` in the correctly built
worktree environment. Include randomized deterministic Rust/NumPy comparisons,
centered differences away from boundaries, finite-support area/moment checks,
invalid inputs, rank/constraint/bound tests and realistic benchmarks.

For documentation run `python scripts/sync_math_docs.py --check` and
`mkdocs build --strict`. For public Rust/API changes also run warning-free
`cargo doc --workspace --all-features --no-deps`, `cargo test --doc --workspace`,
and the versioned API snapshot check. A docs-only plan does not need numerical
tests and does not constitute implementation acceptance.

The critical path is P1 → P2 → P3 → P4 → P5. P6 and P7 are follow-on scope.
The main implementation risks are overlap/rank handling, bounded-solver
conditioning, reflection-domain size and reliable covariance—not new peak
physics. Resolve them with the small dense reference before optimizing.
