# Rietveld Engine Project Brief

## Purpose

Rietveld Engine is a modern, open-source powder-diffraction computation
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
formats 1--3. The internal gate currently passes 399 Python tests (one optional
external-oracle case deselected), 55 Rust tests, strict Ruff, formatting, and
Clippy.

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

- `crates/rietveld-core`: dependency-light numerical types and kernels.
- Numerical primitives such as CW width laws and FCJ geometry remain separate;
  explicit composition modules fuse them for production accumulation.
- `crates/rietveld-py`: PyO3 extension exposing array-oriented functions.
- `crates/rietveld-crystallography`: file-independent cell, exact symmetry,
  reflection-generation, and P1 structure-factor kernels; scattering follows.
- `crates/rietveld-engine`: native composition facade; structural-intensity
  composition with the support-limited profile kernel and structural JVP/VJP.
- `python/rietveld`: public Python package, separated instrument/phase/pattern/
  calculation/refinement modules, reference implementation, optional
  integrations, and validation tooling.
- `python/rietveld/io`: optional format adapters; CIF uses a lazy Gemmi backend
  and returns only parser-independent structures and diagnostics.
- `tests`: Python differential and contract tests.
- `benchmarks`: end-to-end Python benchmarks and stored methodology.
- `oracle`: pinned GSAS-II environment metadata, adapters, and fixture schema.

The core accepts plain numeric slices and explicit peak/instrument batches. It
does not know about files, refinement iterations, Python phase objects, GUI
objects, or GSAS-II dictionaries. Higher layers translate domain models into
flat, validated kernel inputs. The durable public-module contract is documented
in `docs/public-api.md`.

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

## Quality bar

Public behavior is typed and documented. Invalid shapes, non-finite values,
unsorted grids, non-positive widths, and fractions outside `[0, 1]` fail with a
clear error. Tests are deterministic. Numerical tolerances are justified near
the assertion. Formatting, linting, unit tests, Python tests, and benchmarks are
available as ordinary project commands.

The dependency-ordered delivery plan and milestone exit gates are maintained in
`IMPLEMENTATION_PLAN.md`.
