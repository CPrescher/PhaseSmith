# Rust-only original-frame workflow

`phasesmith::workflows::single_crystal::run` coordinates original detector counts,
shared masks, detection/linking, grain discovery, geometry calibration,
prediction, profile learning/fitting, physical and empirical scaling, uncertainty
estimation and structure search. The coordinator does not import, embed or start
Python. HDF5 input and numeric/CIF output also run natively.

The first versioned recipe uses a supplied primitive-orthorhombic metric and
element types. Exact symmetry can be supplied or withheld for
[space-group assessment and search](space-group-search.md). Its fixed scientific controls preserve the reviewed
MgSiO3 workflow: original counts, explicit exposure intervals, whole-family
holdouts, measured competing crossings, scattering-coordinate profiles and the
literal `validated-v1` search strategy. Other single-crystal kernel APIs remain
available separately. This recipe supports 5–225 exposures; its explicit free
frame scale design must fit the existing 256-parameter scaling limit.

## Build and run

```sh
cargo build --release -p phasesmith --features hdf5 --example single_crystal_workflow

target/release/examples/single_crystal_workflow --mgsio3 \
  '/path/to/MgSiO3 Enstatite Calibration Crystal' \
  validation/enstatite-raw-workflow-v40.json \
  /path/outside/the/repository/new-native-run
```

Protocol v40 selects automatic symmetry assessment and `auto` search effort.
It tries a short coarse-grid search first and uses finer phasing/more candidate
settings only if that attempt fails the explicit acceptance checks. It removes
the exact space group from the input and supplies no Laue group. Predictions
retain all primitive reflections. The worker assesses measured intensity
symmetry before empirical scaling, screens conventional settings using a shared
P1 phase map and retains candidate models, alternatives and all budget omissions.
New products include `preliminary-symmetry.json`, `space-group-evidence.json`,
`space-group-phase-map.npz` and `space-group-search.json`. Their selected setting
is a ranked hypothesis; the report does not assert uniqueness.
Use `--effort fast`, `--effort auto` or `--effort thorough` after the standalone
command's output argument. The optional benchmark observer accepts the same flag.
`fast` permits no deeper attempt; `auto` escalates after numerical rejection;
`thorough` starts with the full-resolution bounded profile. The original signed
measurements always enter final model fitting and assessment. Effort, acceptance,
escalation reason and both actual pass records are saved in `space-group-search.json`.
The [speed and adaptive-effort report](space-group-search-speed.md) gives the
limits, measured comparison and forced-fallback check. Protocols before v40 keep
their frozen execution and do not accept this flag without explicit effort controls.

The selected conventional setting and asymmetric sites are saved separately as
`candidate/space-group.cif`; `candidate/best.cif` preserves the full P1 model in
the density frame. Measured inputs to both symmetry assessments are archived
for independent replay. To reproduce the supplied-Pbca control, explicitly use
`validation/enstatite-raw-workflow-v29.json`.

The example checks the registered raw-file and PONI hashes before and after the
run. It never opens the source HKL, atomic reference CIF or archived reductions.
It reads detector pixel sizes from the original HDF5 metadata and preserves the
existing explicit vertical reversal and provisional spindle-angle convention.
All products use exclusive creation in a new external directory.

The `hdf5` feature bundles the same native filters as the
[native input adapter](native-hdf5.md). A source build needs the documented
Rust/C/CMake toolchain; running the resulting executable needs no Python or
installed HDF5/filter package. The small controls, source-identity and null-rotation
assets are bundled with the Cargo example and checked against their frozen
repository counterparts by packaging tests. Build this example from the Git
checkout/source archive or the Rust facade's crate source. The ordinary Python source distribution now includes the facade/file-host
dependency used by the [Dioptas application boundary](dioptas-native-workflow.md).

An optional benchmark observer verifies all registered source files and grades
saved candidates only after the executable exits:

```sh
python benchmarks/enstatite_native_workflow.py \
  --data-root '/path/to/MgSiO3 Enstatite Calibration Crystal' \
  --protocol validation/enstatite-raw-workflow-v40.json \
  --executable target/release/examples/single_crystal_workflow \
  --output /path/outside/the/repository/new-observed-run
```

This observer is optional. It launches the native worker with a PATH containing
only system utilities and an empty HDF5 plugin path. NumPy/SciPy/Gemmi used for
independent post-run assessment do not enter the numerical worker.

## Typed library boundary

The coordinator belongs to `phasesmith-workflows`, enabled by that crate's
`single-crystal` feature and the facade's default single-crystal capability.
Minimal powder-only builds retain their existing dependency boundary. Rust hosts
supply these typed records:

| Record | Contract |
| --- | --- |
| `FrameSource` | Original u32 counts and source dimensions; read-only file adapters detect changed sources |
| `Acquisition` | Inclusive original frame indices, explicit start/step/exposure, orientation and reserved-value mask |
| `Priors` | Cell, optional exact symmetry and element types, with nonempty provenance; no atomic counts or coordinates |
| `DiscoveryOptions` | Explicit catalogue, evidence model, P1 phase screen and candidate/time/ambiguity budgets; required when exact symmetry is absent |
| `ShadowOptions` | Original-count illumination controls and an explicitly supplied full-exposure DAC cone |
| `Options` | Scientific input records, cancellation and independent worker/payload/cache/search budgets |
| `Observer` | Synchronous immutable stage products; errors stop before the next stage and join bounded work |
| `Output` | Original positions, discovery/calibration evidence, observations/bank, signed reduction and complete native search result |

Source loading stays on the caller's thread. Original frame preparation and
independent numerical tasks overlap in bounded source order. Load errors are
queued after earlier frames; later frames are not loaded after a source failure.
Observer errors and cancellation stop further loading and join outstanding work.
A Rust host can use `NoObserver` to retain results entirely in memory.
The library's search time limit is cooperative. The example host additionally
enforces the frozen hard search-process deadline (75 seconds for v29, 210 seconds
for unknown-group protocols including v40); its waiting guard is
cancelled as soon as search/output completes. The optional benchmark parent also
limits the complete worker process to 1,800 seconds.

Source-independent shape, finite-value, frame, worker, mask and payload limits
are checked before reading image counts. Source hashes are rechecked against
the mask snapshots during later passes. The decoded-count cache validates file
identity before each request; cached counts are copied into owned buffers.

## Equations, support and numerical ownership

The coordinator constructs the same existing models; it adds no new peak/profile
terms or derivative conventions. In particular, for laboratory ray
`r = origin + x fast + y slow`, let `u = r / |r|`, incident direction `b` and
sample axis `a`. The local transverse axes are
`e1 = unit(u × b)` and `e2 = u × e1`. The angular pixel Jacobian, in degrees per
pixel, is

```text
J = (180 / (pi |r|)) [[e1·fast, e1·slow], [e2·fast, e2·slow]]
zeta = e1·a
```

Values and these analytical spatial derivatives are constructed together and
checked with centered ray-angle differences. Existing native geometry and
profile fits still evaluate their values/partials in the same sample pass.
See [geometry fitting](experimental-crystal-geometry.md),
[profiles](experimental-3d-profiles.md) and [intensity corrections](experimental-intensity-reduction.md).

Predicted crossings and collection overlaps retain their half-open conventions.
DAC masks test complete closed exposure and block footprints. The cached native
DAC blocks equal direct kernel evaluation; cache admission does not relax support.
Angular-truncation flags propagate from the parent reflection into native profile
groups. The learned foreground still determines when that flag affects a profile.
Signed values, flags, empty/failed rows and the original aperture baseline remain.

The ray-sector regularization uses `Q = S Vᵀ / 0.25`, from the SVD of `H / sqrt(N)`.
The new construction preserves `QᵀQ = HᵀH / (N·0.25²)` and is independently tested.
The empirical kernel still owns log-scale values and analytical derivatives,
family elimination, bounds and robust iterations. The equivalent-error port
retains the existing family-balanced weighted median, 60 bisection iterations,
nonnegative family median, pair budget and signed observations.

`nalgebra` now performs the small least-squares/SVD constructions previously
owned by NumPy. IEEE rounding can differ. The original Python equations remain
independent and readable, and the existing numerical tolerances are retained.
GSAS-II is not an oracle for these single-crystal orchestration operations;
original-frame replay and the independent reference paths provide the comparisons.

## Outputs and scientific qualification

The executable writes lossless numeric NPZ arrays, original source hashes,
geometry/discovery records, per-stage timings, complete candidate coordinates and
ADPs, phasing histories, a P1 CIF and native search diagnostics. Its native search
JSON uses a distinct versioned schema; it does not pretend to be a complete
Python `SolutionResult` serialization. Large arrays and all trial outputs remain
outside the repository.

Labelled structure recovery and numerical parity are separate checks. The
supplied-group comparison below uses v29. The current unknown-group v40 replay,
its alternatives and measured speed improvements are recorded in the
[space-group search validation](space-group-search-validation.md).
The initial missing angular-flag handoff was corrected and the failed trial retained. The
corrected native path exports the same 4,125 observations, including 220 negative
values, with the same quality flags and teacher membership. Geometry roundoff is
about 1e-12 pixels and intensity differences are about 1e-6 in relative L2 norm.

The final search can retain a slightly different candidate-polishing trajectory.
The native executable reproduces the Python baseline R1 exactly when given its
original reduced intensities. Common-objective checks put the R1 response to
swapping only the intensities at about 4e-8, while the two retained candidates
differ by about 2.8e-4. Both stop at the iteration limit near the existing ADP
lower bound. The frozen R1-difference gate remains failed; no tolerance, phasing
seed, scientific control or input was tuned to remove that failure.

The measured timing, strict gate results, stage breakdown and source/artifact
provenance are recorded in the [native coordinator validation report](single-crystal-native-validation.md). A numerical
R1 target does not establish uniqueness, absolute intensity accuracy or a
converged structural refinement. The endpoint remains structure search and
candidate assessment.
