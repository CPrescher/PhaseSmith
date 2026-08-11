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
benchmarks. Full structural refinement remains a follow-on unit. The live
pinned GSAS-II P1 and reflection-behavior fixtures are pending because the
external checkout is not available in the current environment.

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
55 Rust tests, strict linting, and a realistic release benchmark. The live
pinned GSAS-II lattice perturbation fixture remains pending because the
external checkout is unavailable; equations and normal operation contain no
GSAS-II dependency.

The project remains pre-release and is licensed under the MIT License. The
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

The built-in monochromatic Python Rietveld facade now delegates complete
refinement and checkpoint continuation to the Rust solver while retaining the
existing scripting types. Python-defined callbacks and provider extensions use
the explicit scripting fallback. The Python-free application model binds at
most one validated native analysis to each project histogram through
`RietveldProjectState`; native project format 2 persists its built-in sample
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
persistence to native format 2 and reconstructs its scripting dataclasses and
restart handle from Rust-validated state. Python-only providers, component
radiation, non-native checkpoints, scripting optimizer controls, and rich
parser provenance continue to use the compatible format-12 scripting codec.
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
TOF structural Rietveld parameter motion remains a separate future capability
and is not implied by this Le Bail result.
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
13.02 s, 17.56 s, and 157.44 s. All scientific gates pass. The pure-Rust X-ray
Poisson Rwp is 10.34601%, while a fresh 3.560 s Python scripting run gives
10.34604%; the native path is scientifically coincident and about 19% slower.
It now follows the scripting optimizer's weighted free-coordinate Jacobian,
phase-scale conditioning, accepted-trial carry-forward, bounded backtracking,
and physical Bruckner-width conversion. The remaining final-stage iteration
difference occurs only on the flat tail of the minimum. Historical pinned
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

## Quality bar

Public behavior is typed and documented. Invalid shapes, non-finite values,
unsorted grids, non-positive widths, and fractions outside `[0, 1]` fail with a
clear error. Tests are deterministic. Numerical tolerances are justified near
the assertion. Formatting, linting, unit tests, Python tests, and benchmarks are
available as ordinary project commands.

The dependency-ordered delivery plan and milestone exit gates are maintained in
`IMPLEMENTATION_PLAN.md`.
