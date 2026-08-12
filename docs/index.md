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
6. [Persistence](persistence.md) — save and restore Python workflow state.
7. [Native project persistence](native-persistence.md) — use the Rust-only,
   multi-histogram JSON+NPZ project and reporting boundary.
8. [Python API map](api-reference.md) — find public types and functions by
   task.

For shared refinement controls, cancellation, logs, and safe checkpoints, see
[refinement](refinement.md) and [refinement runtime](refinement-runtime.md).

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
- [Neutron CW](neutron-cw.md), [neutron TOF](tof-profile.md), and
  [multi-bank TOF Le Bail](tof-multibank.md)
- [Public Python architecture](public-api.md)
- [Native Rust API and GUI integration](rust-api.md)

## Packages and source

- [Python package on PyPI](https://pypi.org/project/phasesmith/)
- [Rust facade on crates.io](https://crates.io/crates/phasesmith)
- [Rust API documentation on docs.rs](https://docs.rs/phasesmith/)
- [Source and releases on GitHub](https://github.com/CPrescher/PhaseSmith)

Planning and historical implementation notes describe design provenance, not
the current user contract. The documents above and `PROJECT_BRIEF.md` are the
maintained references.
