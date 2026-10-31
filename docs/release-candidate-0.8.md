# 0.8 single-crystal release candidate

This development candidate promotes the validated single-crystal workflow into
the main PhaseSmith package. It is not a publication receipt. No tag, upload or
deployment is performed by the promotion task.

The canonical API is [phasesmith.single_crystal](single-crystal.md). Normal Python
builds include the native capabilities and bundled HDF5. The Rust facade defaults
to named single-crystal capabilities, retains a minimal opt-out and keeps HDF5
optional. Implementation crates have synchronized 0.8 metadata and notices for
workspace publication. Old module and Cargo-feature aliases remain available.

The v24 raw protocol is a namespace/configuration promotion control, with the v23
scientific and execution settings unchanged. The new explicit named search
configuration records its complete expansion; historical protocols and API
snapshots remain frozen. New summary schema migration changes metadata only.

Required local checks are default/minimal/legacy-feature builds, strict Clippy,
all-feature Rust tests, complete ordinary-build Python tests, API/metadata checks,
installed NumPy-only wheel smoke, source reconstruction, crate packaging and
strict documentation. Fresh original-frame MgSiO3 comparison must retain all
v11 array bytes, full candidate records and phasing histories, with source files
unchanged. Actual results are recorded after execution in the promotion receipt.

CI now makes capability checks mandatory rather than silently skipping absent
experimental kernels. Ordinary installed-wheel smoke covers native detection,
row capability, structure search and independent Bitshuffle/LZ4 count decoding,
with optional Python numerical packages blocked. Cross-platform CI results and
the manual unpublished release workflow remain remote gates until they run.

Scientific limitations remain explicit: supplied symmetry/element priors,
bounded structure-recovery coverage, provisional MgSiO3 acquisition geometry,
conditional uncertainty models and ADP-bound candidate-polishing stagnation.
Standalone single-crystal structure refinement is outside this release.


Local source validation passes 515 Rust tests and 3,277 Python tests (11
external-data skips, 34 slow tests deselected by the default marker policy),
strict all-feature Clippy, default/minimal/legacy-feature checks, strict Rust
and MkDocs documentation, API/version/skill/math-documentation checks. All 13
public crate archives package with `cargo package --workspace --no-verify`;
this checks packaging/resolution, not a registry publication or independent
build of every packaged crate. Wheel/sdist reconstruction and v24 results are
recorded in the final receipt after execution.

A whole-repository Ruff audit still reports 191 pre-existing diagnostics in
unchanged research tools/references/tests. None is in a changed Python file;
changed files pass targeted lint. This inherited release-CI gate remains to be
resolved before tagging. It is not suppressed or presented as passing.
