# PhaseSmith documentation

## Start here

1. [README examples](../README.md) — construct typed patterns,
   instruments, phases, and calculations.
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
8. [Releasing](releasing.md) — build and publish synchronized crates.io, PyPI,
   and GitHub releases.

For shared refinement controls, cancellation, logs, and safe checkpoints, see
[refinement](refinement.md) and [refinement runtime](refinement-runtime.md).

## Numerical conventions

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
- [Neutron CW](neutron-cw.md) and [neutron TOF](tof-profile.md)
- [Public Python architecture](public-api.md)

Planning and historical implementation notes describe design provenance, not
the current user contract. The documents above and `PROJECT_BRIEF.md` are the
maintained references.
