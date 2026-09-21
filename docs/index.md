# PhaseSmith documentation

PhaseSmith provides validated powder-diffraction calculations and refinement
through a native Rust library and a typed Python/NumPy scripting interface.
Python users install the same Rust numerical core that native applications can
consume directly—no Python process or sidecar is required for a Rust GUI.

## Start here

1. [Getting started](getting-started.md) — install PhaseSmith and run a first
   calculation.
2. [Powder-file and CIF import](cif-import.md) — turn measured arrays and a CIF
   into crystallographic inputs.
3. [Background subtraction](background-subtraction.md) — estimate or supply the
   background explicitly.
4. [Le Bail extraction](lebail.md) — extract non-negative reflection
   intensities and optionally refine profile or lattice parameters.
5. [Rietveld refinement](rietveld.md) — run the matrix-free structural workflow
   with checkpoints and reports.
6. [AI-guided automation](ai-automation.md) — let an external AI propose
   auditable staged recipes without giving it numerical or execution authority.
   The [agent skill](agent-skill/index.md) packages the operating protocol and
   scientific decision rules for use in an agent's own environment.
7. [Persistence](persistence.md) — save and restore Python workflow state.
8. [Native project persistence](native-persistence.md) — use the Rust-only,
   multi-histogram JSON+NPZ project and reporting boundary.
9. [Python API map](api-reference.md) — find public types and functions by
   task.

For shared refinement controls, cancellation, logs, and safe checkpoints, see
[refinement](refinement.md) and [refinement runtime](refinement-runtime.md).
For post-fit evidence, see [fit reports](fit-report.md). The
[rietx comparison](rietx-comparison.md) records controlled speed measurements
and the proposed capability development sequence.

## Numerical conventions

- [Complete mathematical reference for Rust and Python](mathematics/index.md)
- [Equations and units](equations.md)
- [TCH profile](tch-profile.md), [CW broadening](cw-profile.md), and
  [FCJ asymmetry](fcj-profile.md)
- [Symmetry and reflection generation](symmetry-reflections.md)
- [Structural intensities](structural-intensities.md) and
  [scattering models](scattering-models.md)
- [Lattice refinement](lattice-refinement.md)

## Additional models and workflows

- [Multi-phase calculations](multiphase.md) and
  [quantitative phase analysis](quantitative-phase-analysis.md)
- [Sample physics](sample-physics.md) and
  [wavelength components](wavelength-components.md)
- [Offline fundamental-profile calibration](fundamental-profile-calibration.md)
- [Neutron CW](neutron-cw.md), [neutron TOF](tof-profile.md),
  [multi-bank TOF Le Bail](tof-multibank.md), and
  [shared TOF lattice](tof-lattice-refinement.md) and
  [bank-local instrument refinement](tof-instrument-refinement.md), and
  [joint TOF geometry refinement](tof-geometry-refinement.md)
- [Structural TOF readiness and correction contract](tof-structural-readiness.md)
- [Public Python architecture](public-api.md)
- [Native Rust API and GUI integration](rust-api.md)

- [Empirical Gaussian widths](empirical-gaussian.md) — explicit reference-phase
  convention when independent instrument calibration is unavailable.

## Validation and performance

- [Original-recipe convergence audit](qarr-convergence-audit.md) — repeated
  timings and feasible-descent checks expose a stopping-condition weakness.

- [QARR holdout investigation](qarr-holdout-investigation.md) — distinguish
  convergence, width assumptions and phase-fraction accuracy on a second sample.

- [Quality-first recovery](refinement-quality-recovery.md) — fixed-quality
  convergence benchmarks and retained-state damping recovery.

- [Powder Friedel averaging](powder-friedel.md) — corrected anomalous powder
  intensities and analytical derivatives; individual-reflection API semantics.
- [Choosing profile accuracy](profile-accuracy.md) — explicit FCJ quadrature
  and tail-area controls with persistence and validation contracts.
- [rietx workload audit](rietx-workload-audit.md) — controlled real-data
  experiments isolate axial approximations, support windows and solver work.
- [Refinement performance improvements](refinement-performance-round2.md) —
  batched FCJ evaluation, reusable scale profiles, native diagnostics and real
  QARR measurements, with links to the earlier optimization work.
- [PhaseSmith versus XRD-Rust](xrd-rust-performance.md) — reproducible public
  stick-pattern timings, numerical checks, and retained raw results.
- [PhaseSmith versus GSAS-II](gsasii-performance.md) — pinned-oracle kernel and
  complete-workflow comparisons.
- [Real-data benchmark report](gsasii-benchmark-report.md) — reviewed
  scientific results and supported boundaries.

## Packages and source

- [Python package on PyPI](https://pypi.org/project/phasesmith/)
- [Rust facade on crates.io](https://crates.io/crates/phasesmith)
- [Rust API documentation on docs.rs](https://docs.rs/phasesmith/)
- [Source and releases on GitHub](https://github.com/CPrescher/PhaseSmith)

Planning and historical implementation notes describe design provenance, not
the current user contract. The documents above and `PROJECT_BRIEF.md` are the
maintained references.
