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
ordered composite backgrounds implement one analytical protocol. The
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

The first pinned sucrose run uses 23,003 samples and 811 generated reflection
families. Its first-cycle `Rwp` of 27.80% falls to 18.79%, the
background-subtracted observed/calculated correlation is 0.99159, and all
extracted intensities remain finite and non-negative. QARR verifies its 7,251
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

The planned native application boundary is specified in
`docs/native-application-plan.md`. It keeps Tauri outside the scientific
workspace, introduces application-neutral owned model/I/O/workflow layers, and
preserves the Python scripting surface without requiring Python in a desktop
distribution.

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

The pinned IUCr QARR 1g workflow is now an accepted native three-phase
fixed-spectrum structural validation, not a readiness placeholder. Its staged
refinement ends with a scale-only polish and converts scales using reviewed
phase Z, formula mass, and cell volume metadata. The current 2026-08-07
fixed-anisotropic baseline gives Al2O3 30.778%, ZnO 34.205%, and CaF2 35.018%,
with a maximum absolute error of 0.598 weight-percentage points from the
independently weighed fractions. Poisson-weighted and unit-weight Rwp are
reported separately (0.19888 and 0.13256), alongside profile correlation
0.99061.

This checkpoint uses explicit remaining approximations: fixed Cu K-alpha1
dispersion offsets for both doublet components and no absorption. SH/L=0.002
FCJ asymmetry and fixed CIF anisotropic displacement are active. It also
established two
refinement safety rules: nonphysical bounded trials are logged and backtracked
without losing the last accepted state, and phase-scale conditioning follows
the current nonzero scale magnitude instead of assuming scales are order one;
an exact zero retains an order-one escape scale.

The PbSO4 real-data checkpoint adds a separate optional staged-workflow layer
above the unchanged general Rietveld solver. Explicit `RietveldRecipe` objects
own parameter activation and stage acceptance. The deterministic
`intelligent_rietveld_recipe` planner is advisory: it may select only families
already authorized by the caller, records human-readable reasons and active
parameter identities, and never runs implicitly. The accompanying physics
checkpoint adds the constant-wavelength neutron powder Lorentz factor
`1/(sin(theta) sin(2 theta))` with native values and analytical derivatives.
Persistence format 12 stores that typed correction and the Debye--Scherrer
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

The remaining parity boundary is joint refinement. PhaseSmith currently owns
one pattern/experiment per `RietveldInput`; GSAS-II shares one PbSO4 structure
across both histograms. The next slice must introduce a first-class
multi-histogram objective with explicit shared structural parameters and local
scale/background/profile/radiation/geometry parameters. Alternating independent
fits is not accepted as an equivalent implementation.

## Quality bar

Public behavior is typed and documented. Invalid shapes, non-finite values,
unsorted grids, non-positive widths, and fractions outside `[0, 1]` fail with a
clear error. Tests are deterministic. Numerical tolerances are justified near
the assertion. Formatting, linting, unit tests, Python tests, and benchmarks are
available as ordinary project commands.

The dependency-ordered delivery plan and milestone exit gates are maintained in
`IMPLEMENTATION_PLAN.md`.
