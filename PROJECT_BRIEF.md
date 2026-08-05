# Rietveld Engine Project Brief

## Purpose

Rietveld Engine is a modern, open-source powder-diffraction computation
library. Its numerical kernels are written in Rust and exposed through a small,
typed Python API. The first product layer is a trustworthy and fast
powder-profile calculator. Refinement orchestration follows only after the
profile kernels are mature; Le Bail extraction is the first supported complete
refinement workflow. Project-file management and graphical user interfaces are
not part of the numerical library.

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
- Maintain a thin optional Dioptas integration boundary; Dioptas and other GUI
  packages are never runtime dependencies of the core package.
- Maintain an independent, readable Python reference implementation for every
  numerical kernel before optimizing it.
- Treat GSAS-II licensing conservatively: implement from published equations
  and public behavior, record sources, and compare numerical output. Do not
  copy source code into this repository.

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
6. Track performance with repeatable Rust and Python benchmarks.

Oracle fixtures must record the GSAS-II commit or distribution version, Python
version, input project/checksum, adapter version, and platform. Generated
fixtures may be committed when redistribution is lawful and their provenance is
documented; GSAS-II itself is never vendored.

## Architecture

- `crates/rietveld-core`: dependency-light numerical types and kernels.
- Numerical primitives such as CW width laws and FCJ geometry remain separate;
  explicit composition modules fuse them for production accumulation.
- `crates/rietveld-py`: PyO3 extension exposing array-oriented functions.
- `python/rietveld`: public Python package, separated instrument/phase/pattern/
  calculation/refinement modules, reference implementation, optional
  integrations, and validation tooling.
- `tests`: Python differential and contract tests.
- `benchmarks`: end-to-end Python benchmarks and stored methodology.
- `oracle`: pinned GSAS-II environment metadata, adapters, and fixture schema.

The core accepts plain numeric slices and explicit peak/instrument batches. It
does not know about files, refinement iterations, Python phase objects, GUI
objects, or GSAS-II dictionaries. Higher layers translate domain models into
flat, validated kernel inputs. The durable public-module contract is documented
in `docs/public-api.md`.

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
9. Plain-data persistence and optional external integration adapters, starting
   with a Dioptas-oriented NumPy boundary.

## Non-goals

- Porting nested GSAS-II dictionaries or global mutable state.
- Reproducing GSAS-II's file-driven refinement workflow.
- Calling Python once per reflection in production profile calculation.
- Building a GUI in the numerical library.
- Requiring Dioptas or any GUI toolkit to use the package.
- Claiming numerical equivalence from pointwise values alone.

## Quality bar

Public behavior is typed and documented. Invalid shapes, non-finite values,
unsorted grids, non-positive widths, and fractions outside `[0, 1]` fail with a
clear error. Tests are deterministic. Numerical tolerances are justified near
the assertion. Formatting, linting, unit tests, Python tests, and benchmarks are
available as ordinary project commands.

The dependency-ordered delivery plan and milestone exit gates are maintained in
`IMPLEMENTATION_PLAN.md`.
