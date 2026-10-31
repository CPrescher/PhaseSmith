# 0.8 single-crystal release candidate

This development candidate promotes the validated single-crystal workflow into
the main PhaseSmith package. It is not a publication receipt. No tag, upload or
deployment is performed by the promotion task.

The canonical API is [phasesmith.single_crystal](single-crystal.md). Normal Python
builds include the native capabilities and bundled HDF5. The Rust facade defaults
to named single-crystal capabilities, retains a minimal opt-out and keeps HDF5
optional. Implementation crates have synchronized 0.8 metadata and notices for
workspace publication. Pre-release experimental imports/features are removed after updating Dioptas;
independent references remain under `single_crystal.reference`.

The v24 raw protocol is a namespace/configuration promotion control, with the v23
scientific and execution settings unchanged. The new explicit named search
configuration records its complete expansion; historical protocols and API
snapshots remain frozen. New summary schema migration changes metadata only.

Required local checks are default/minimal/focused-feature builds, strict Clippy,
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

The original candidate recorded 191 inherited Ruff diagnostics. Subsequent
maintenance clears repository-wide lint and formatting; see the
[package review](package-review-0.8.md) for the current checks and simplifications.


## Completed distribution and raw-workflow checks

The ordinary ABI3 macOS arm64 wheel passes all **3,277 Python tests outside the
checkout**. A separate source archive builds from scratch into another empty
environment and passes the same native-capability smoke with only NumPy installed.
Optional numerical packages are blocked by that smoke. Native linkage contains
system libraries only, with no external HDF5/filter dependency. All 13 public
crate payloads and the applicable notices are retained outside Git.

The installed package runs two fresh original-frame v24 workflows in
**19.06 / 19.39 s** (workers **17.95 / 18.29 s**). Every retained numerical array,
complete candidate record and phasing history matches v11 bitwise; R1 stays
**.04085182070733226**. New summary schema/configuration metadata are intentional.
The prior v23 pair was 18.65/18.56 s: the new pair is 3.3% longer on this shared
desktop. These are not controlled interleaved A/B measurements; no promotion
speedup or causal performance-regression claim follows. The full times are
retained. Peak worker RSS remains about 4.03 GiB with the explicit 1 GiB cache.

The final 19.39 s run partitions as follows; I/O and parallel work are included
in their owning stage and nested timings are not added again.

| Area | Seconds |
| --- | ---: |
| Shadow estimation | 2.783 |
| Detection/linking | 3.404 |
| Pixel collection/background | 2.906 |
| Profile learning | 2.532 |
| Profile fitting | 2.852 |
| Archive output | 1.033 |
| Corrections/scaling/errors | 0.286 |
| Indexing/geometry | 0.253 |
| Structure-search process | 0.796 |
| Other worker assembly/bookkeeping | 1.446 |
| Launch/parent checks/initialization | 1.099 |
| **Total** | **19.390** |

The small machine-readable receipt is
[single-crystal promotion record](https://github.com/CPrescher/PhaseSmith/blob/main/benchmarks/single-crystal-promotion-0.8-macos.json).
Artifacts, complete equivalence reports and logs are external under
`/Users/clemens/ChatGPT/PhaseSmith/single-crystal-release-0.8-20261031`.

## Storage cleanup

After validation, generated build/doc targets and 12 superseded environments
were removed. Every environment's complete PhaseSmith payload was checked
against a retained wheel, and its Python/dependency versions were journaled for
recreation. The directories accounted for 15,660,044,288 allocated bytes;
filesystem free space increased **13,808,058,368 bytes (13.8 GB / 12.86 GiB)**
during cleanup. These measures differ because of shared files/filesystem activity.
All eight original files were rechecked afterward. Raw data, unique results,
failed/slower trials, source/Git, active development environment and retained
release artifacts remain. Fifty-eight Criterion files were archived and verified
before clearing the target. Other worktrees and shared dependency caches were
not cleaned. Future Rust builds regenerate the removed compiler artifacts.
