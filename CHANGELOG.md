# Changelog

All notable changes to PhaseSmith will be documented here. The project is
pre-release; compatibility may still change deliberately between minor
development versions.

## Unreleased

- Treat PbSO4 X-ray final-polish `repeated_rejections` as safe bounded
  stagnation only after an accepted, materially improved state, while
  preserving the termination reason, accepted-state checkpoint semantics, and
  all scientific tolerances.

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
