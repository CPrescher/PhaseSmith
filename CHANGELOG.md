# Changelog

All notable changes to PhaseSmith will be documented here. The project is
pre-release; compatibility may still change deliberately between minor
development versions.

## Unreleased

- Add the checksum-pinned anhydrous tripotassium-citrate/Si holdout, including
  exact pdCIF conversion, an independent native workflow, and a pinned GSAS-II
  comparison. The shared fixed-geometry model passes parity, while the 1.44 wt%
  Si displacement anchor is explicitly rejected as non-transferable and the
  richer deposited refinement remains a qualified reference.

## 0.3.0

- Add facility-neutral TOF powder boundaries for center/density columns, packed
  constant-step data, GSAS SLOG FXYE, and bounded legacy profile-function 1/3
  calibration adapters. POWGEN LaB6 and LANL nickel provide checksum-pinned
  acceptance cases without facility-name-triggered numerical behavior.
- Add atomic multi-bank TOF Le Bail refinement with shared analytical lattice
  motion, selected bank-local analytical instrument motion, joint
  lattice/instrument systems, rank and correlation diagnostics, cooperative
  cancellation, and exact accepted-state checkpoint continuation.
- Add structural TOF Rietveld calculation and refinement with typed bank
  geometry, the neutron TOF Lorentz correction, fused dense/JVP/VJP products,
  shared structure and cell parameters, bank-local scale/instrument/background
  parameters, bounded solving, and exact special-position restart semantics.
- Expose fixed-instrument, joint-geometry, and structural TOF workflows through
  immutable Python facades and versioned native project persistence. Add
  single- and multi-bank file composition with explicit selected ranges,
  incident normalization, upstream/sample corrections, background domains,
  ordered bank identities, and checksum-bound provenance.
- Validate structural TOF on real POWGEN LaB6 and three-bank LANL nickel data.
  The isolated pinned GSAS-II comparisons reproduce peak positions and profile
  chains at floating-point scale, exceed 0.99999 same-intensity pattern
  correlation, and agree on the nickel structural cell and Uiso within
  0.00004414 A and 0.00005306 A^2.
- Add offline fundamental-profile, Soller axial-divergence, and Gaussian
  spectral-passband calibration utilities together with independent laboratory
  XRD benchmarks and expanded Rowles, QARR, NIST, PbSO4, sucrose, and Echidna
  validation/oracle coverage.
- Treat PbSO4 X-ray final-polish `repeated_rejections` as safe bounded
  stagnation only after an accepted, materially improved state, while
  preserving the termination reason, accepted-state checkpoint semantics, and
  all scientific tolerances.
- Document the complete TOF equations, supported contracts, ORNL/POWGEN request
  boundary, cross-implementation refinement results, and explicit unsupported
  profile/calibration/correction scope.

The TOF implementation is facility-neutral within its declared profile,
calibration, reduction, and correction contracts. This release does not claim
support for unimplemented beamline-specific profiles or implicit absorption,
extinction, or texture models.

## 0.2.0

- Standardize refinement background composition as a fixed supplied baseline
  plus an optional analytical residual. Add CW Le Bail background state,
  checkpoint persistence, analytical columns, and deterministic weighted
  linear elimination; make TOF Chebyshev explicitly additive rather than a
  replacement.
- Rebuild the sucrose GSAS-II oracle comparison around one matched recipe:
  identical bin centers, fixed Smooth Bruckner array, constant residual,
  reflection set, cycle count, and explicit symmetric instrument parameters.
- Add native and Python staged effective-profile estimation from a dominant
  single phase, with fixed wavelength, optional bounded lattice alignment,
  conservative W/UVW/UVWXY selection, and explicit sample-broadening warnings.
- Add a typed native fixed-instrument TOF Le Bail workflow and promote the
  checksum-pinned POWGEN LaB6 pattern to a passing acceptance validation.
- Propagate quantitative phase-fraction covariance, persist the new Le Bail
  background state in format 13, and retain readers for earlier formats.
- Expand checksum-pinned validation with Echidna, NIST SRM 660c, QARR 1h, and
  POWGEN cases plus deterministic suite comparison and scheduled CI coverage.
- Add a docs.rs-native Rust guide hierarchy with compiled examples and a
  complete mathematical reference for profile calculation, crystallography,
  scattering, pattern/sample composition, backgrounds, refinement, and
  application-host integration.
- Expand all published component-crate landing pages and validate rustdoc with
  warnings denied in CI.
- Publish the same complete mathematical reference in the Python-facing Read
  the Docs site, link public Python modules to it, and enforce automatic
  rustdoc/MkDocs synchronization in CI.

## 0.1.0

- Initial typed Python API, public Rust facade, and native numerical kernels.
- Harden CI, oracle configuration, persistence v7 documentation, and local
  development validation.
- Add derivative guardrails and remove correction, TOF, FCJ, reflection, and
  structure-factor performance cliffs.
- Preserve guarded lattice-domain state in Rietveld checkpoints and make the
  quadratic Le Bail rank diagnostic opt-in.
