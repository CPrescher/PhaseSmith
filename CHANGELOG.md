# Changelog

All notable changes to PhaseSmith will be documented here. The project is
pre-release; compatibility may still change deliberately between minor
development versions.

## Unreleased

- Add a docs.rs-native Rust guide hierarchy with compiled examples and a
  complete mathematical reference for profile calculation, crystallography,
  scattering, pattern/sample composition, backgrounds, refinement, and
  application-host integration.
- Expand all published component-crate landing pages and validate rustdoc with
  warnings denied in CI.

## 0.1.0

- Initial typed Python API, public Rust facade, and native numerical kernels.
- Harden CI, oracle configuration, persistence v7 documentation, and local
  development validation.
- Add derivative guardrails and remove correction, TOF, FCJ, reflection, and
  structure-factor performance cliffs.
- Preserve guarded lattice-domain state in Rietveld checkpoints and make the
  quadratic Le Bail rank diagnostic opt-in.
