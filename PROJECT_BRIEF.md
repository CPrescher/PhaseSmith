# PhaseSmith Project Brief

## Purpose and current release

The 0.8 public Rust package boundary groups detector/integration, structure-search
and row-grouping kernels into `phasesmith-single-crystal`, with independently
selectable `spots`, `solution` and `row-search` features. These were three
unpublished development crates, not separate product APIs. Existing facade paths
(`phasesmith::spots`, `phasesmith::solution`, `phasesmith::row_search`) and Python
imports remain unchanged. HDF5 stays an independent file-adapter crate; the
complete coordinator stays in `phasesmith-workflows` to avoid a dependency cycle.
This is a module/package move, with no intentional numerical or protocol changes.
The public release now contains eleven crates, including two new registry names.

Legacy NumPy cell-metric coordinates use explicit matrix-axis Frobenius
reductions in both their derivative chain and metric retraction. Intel macOS
Accelerate produced alignment-dependent last bits through the axis-free BLAS
vector-norm fast path even with one thread. The equation is unchanged; exact
training/held-out independence remains required, and floating-point changes from
the former reduction are checked locally, with finite differences and a full
multi-reflection fit benchmark. Native Rust kernels/coordinator are unchanged.

The 0.8 release-preparation pass completes the first-use/native-workflow input
guides and shared Rust/Python single-crystal mathematical reference. Runnable
synthetic examples are installation/API controls, separate from measured
validation. Release validation includes the declared NumPy minimum and an
unpublished full platform/source-distribution workflow. Publication and the
release tag remain separate from this preparation; retain exact reproducibility
gates and historical scientific receipts.

`develop` is the consolidated starting point for continuing development. It
includes the 0.7 mainline, the recent HKL/single-crystal/native-workflow work and
the reviewed unfinished detector/research changes. `main` retains the 0.7 release
baseline until release review. Start new task branches from `develop`; see the
[3 November branch audit](docs/integration/development-baseline-20261103.md) for
inclusion, superseded work and intentionally separate historical experiments.

PhaseSmith is a powder and single-crystal diffraction library with native Rust
numerical kernels and an optional typed Python/NumPy frontend. GUI applications
own presentation, acquisition configuration and application state. The current
source version is the **0.8 development candidate**, not a publication receipt.

Powder workflows include profile/background evaluation, crystallography,
CW/TOF structural calculation, Le Bail, Pawley, Rietveld and quantitative analysis.
Single-crystal workflows cover masks/shadows, spots and rotation positions,
reciprocal indexing, lattice/grain discovery, geometry fitting, prediction,
integration, learned profiles, explicit corrections/scaling and structure search.

The single-crystal endpoint is **structure search and candidate assessment**.
Positional detector/crystal fitting and internal coordinate/ADP polishing are
separate operations; no standalone single-crystal structure-refinement workflow
is promised. The interpreter-free coordinator now lives in
`phasesmith-workflows::single_crystal`, behind its optional `single-crystal`
feature and included by the facade single-crystal capability. It owns stage
coordination, independent family splits and native design construction; file I/O
stays in the HDF5/persistence adapters. The first native recipe uses a supplied
primitive-orthorhombic metric and explicit source/DAC assumptions. Existing Python
orchestration remains an independent regression path. See
[the native workflow](docs/single-crystal-native-workflow.md) for scope, execution
budgets and measured numerical differences; strict final-search parity is not
implied by successful labelled recovery.

Unknown symmetry is a native workflow mode. It must not use an exact
space group to filter predictions or define scaling families before discovery.
Retain all measured signed reflections; masks/unmeasured crossings never supply
absence evidence. Assess intensity symmetry before empirical scaling, enumerate
metric-compatible conventional Hall settings, and report ranked alternatives,
origin/model evidence and bounded-search omissions. Exact-symmetry input remains
an explicit supported mode. Automatic discovery is not proof of uniqueness or a
complete SHELXT implementation. Freeze data-run controls before frame reads.
The measured v33 MgSiO3 run withholds exact symmetry and Laue group, assesses
345 metric-compatible settings, phase-screens 64 and searches 12 models. It
ranks Pbca first and recovers Mg16 Si16 O48. Shared-map assignment and four
ordered independent P1 workers reduce observed full time from about 47.7 to
36.2–36.5 s. Its bounded coverage does not prove uniqueness; the supplied-group
control remains bitwise unchanged. See [the assessment guide](docs/space-group-search.md).
Further speed experiments preserve full-resolution final measurements and freeze
cache reuse, a published optional light-element centrosymmetric exit, and coarse
quality-controlled P1 phasing separately. See the
[speed experiment plan](docs/space-group-speed-plan.md); success and generality
must be measured rather than inferred from the reduced workload.
The current v40 example uses explicit `fast`/`auto`/`thorough` effort. `auto`
first attempts the short coarse path and only starts full-resolution phasing and
broader model search after numerical acceptance fails; both share the original
remaining deadline and retain complete pass evidence. The exposed MgSiO3 search
is about 0.5 s on the measured M4 Pro, versus about 0.65 s for the measured
10-worker SHELXT control. A forced fast-timeout test recovers the correct Pbca
structure through the full-resolution fallback. These timings and candidate
acceptance thresholds do not establish general speed/reliability or uniqueness.
See the [adaptive-effort report](docs/space-group-search-speed.md).
The subsequent fixed 14-case exposed COD comparison verifies nine natural
fallbacks, but finds premature acceptance and unresolved subgroup/chemistry
cases. Fast/auto recover eight labelled atomic structures each; both structure
and published group are recovered in two/four cases, respectively. Median
times are 1.02/9.58 s versus 0.56 s for ten-worker SHELXT, which receives
published Laue/centring priors. This is a documented limitation of general
unknown-group recovery, not a new solver guarantee. See the
[cross-structure report](docs/unknown-group-corpus-validation.md).
The follow-up adds explicit `SymmetryPrior` Laue rotations and exact centring
translations, used for phase-map processing and catalogue eligibility while
retaining measured contradictions/insufficient evidence. Acceptance also guards
unresolved higher Laue symmetry, atomic inversion and BIC-close group alternatives.
An unaccepted fallback cannot replace an existing approximation solely on R1/BIC.
Both passes and selection provenance survive. The fixed comparison now supplies
the same symmetry priors to both programs; see the
[matched-prior specification and validation](docs/matched-prior-search-validation.md).
On the same exposed 14 cases, matched-prior fast/auto recover 11 labelled
structures and 10 with the published group. Earlier auto recovered 8 labelled
structures and 4 with the published group. Four
accepted cases all pass both checks; seven other correct atomic models remain
conservatively unaccepted. Median times are 1.10/8.08 s, versus 0.49 s for
ten-worker SHELXT (8 complete labelled structures with their groups). Ten natural
escalations retain both passes; separate forced-budget MgSiO3 and acentric COD
controls verify accepted full-resolution upgrades. This is regression evidence,
not a general recovery guarantee; phasing/assignment and canonical group handling
still have documented failures.
The next performance change reuses the existing initial fitting evaluation,
identical-bit isotropic reciprocal weights, fixed-site phases and unchanged
orbit topology. It keeps scalar fallbacks, value/derivative ordering and all
acceptance/quality gates. A paired fresh baseline, fitting-only stage and final
matched-prior comparison measure performance and result identity; see the
[candidate fitting report](docs/candidate-fitting-speed.md).
Automatic effort may defer deeper work only when space-group ambiguity is its
sole failure and all close alternatives have numerically equivalent complete
labelled models and common residuals with the same parameter count. Such a
candidate stays unaccepted and ambiguous, with explicit equivalence evidence.
Other failures still escalate; explicit thorough effort always runs directly.
The measured ten-atom/4,168-point benchmark reduces preparation plus eight iterations
by 5.5% (value/Jacobian evaluation by 10.1%). On the same 14 exposed cases, one
redundant ambiguity fallback is deferred; every retained model, R1, density, map
and acceptance flag is bitwise unchanged. Overall auto time remains essentially
unchanged (118.86 to 118.31 s summed). The isolated fitting gain is not a complete
search speedup or SHELXT speed parity. Positive forced fallback controls still
recover accepted MgSiO3 and acentric COD structures with identical retained outputs.
A subsequent six-run, extra-effort diagnostic uses the same element-only chemistry
and symmetry priors. It recovers the lithium case's framework and group but still
mislabels C/N; nickel retains a correct framework with wrong labels, while chromium
also misses coordinates. None passes labelled grading or acceptance. Next work
should investigate joint element/ADP correction, the centred-prior/translation-free
phasing contract mismatch, common-score origin recovery and target-aware consensus
stopping. These diagnostics preserve the frozen 14-case results; see the
[remaining-case evidence](docs/candidate-fitting-speed.md#remaining-case-extra-effort-diagnostic).
The recovery revision adds an explicit native `ElementSearchPolicy` while
preserving the legacy entry points and named profiles. Bounded final anisotropic
polishing/reassignment retains its isotropic parent and uses actual tensor
parameter counts on the unchanged common data. Subsequent stages repair centred
intensity-group validation, verify saved candidate origins before blind recovery,
and require the requested R1 target for native consensus stopping. Inversion
evidence can prefer a compatible centric candidate only within the existing
promising-model R1 work ceiling; both selection regressions are retained.
The same 14 exposed cases now recover 12 labelled structures and 11 with the
published group, versus 11/10 before, preserving every previously correct model.
Five are accepted, all correct; nickel/lithium labels and the Hg/I group remain
unresolved. Acceptance criteria are byte-identical. Fast median time falls
1.083 to 0.944 s, but auto total stays essentially flat (117.019 to 116.802 s).
SHELXT is a reference for reliable recovery, not an outperformance target.
See the [staged recovery report](docs/structure-search-recovery.md).

The Dioptas integration exposes this existing coordinator through the optional
`single-crystal-files` Rust facade adapter and the normal Python
`single_crystal.workflow` frontend. File protocol decoding and lossless output
move out of the example into that adapter; the numerical coordinator is unchanged.
The CLI retains its hard process guard; in-process hosts use cooperative native
cancellation and stage callbacks and never terminate the GUI process. Large
products remain in a new external directory. The first application recipe retains
the HDF5/primitive-orthorhombic supplied-cell and explicit acquisition/DAC limits;
it is not a general TIFF/multigrain-to-structure or structure-refinement promise.
The complete Dioptas MgSiO3 replay accepts Pbca, Mg16 Si16 O48 at R1
0.0514043045582282. Both GUI runs preserve all 68 arrays in ten NPZ archives,
complete candidates/histories and both CIFs bitwise against a fresh standalone
v40 control. The final native coordinator takes 15.14 s; the GUI job takes
15.74 s and 15.85 s through its verified raw-frame/spot preview. See the
[application validation](docs/dioptas-native-validation.md).

## Reconciled detector work and retained research

The 3 November 2026 reconciliation brings the unfinished annular-threshold
optimization and its clipping/adjacent-float tests into the consolidated
single-crystal implementation. Interior columns use row-sized statistics and
independent threshold lanes; clipped columns retain scalar execution. Equations,
strict thresholds, two-pass exclusions, extraction and ordered reductions remain
unchanged. The independent direct-sum Python reference remains unchanged.
See the [reconciliation record](docs/integration/unfinished-work-reconciliation.md)
for fresh checks, exact-record comparisons and local benchmark results.

The recovered Fe7C3/DIALS scripts, figures and portable reports are research
evidence. In particular, the recorded Fe7C3 intensity comparison fails its
agreement gate; successful synthetic controls and later MgSiO3 validation do not
remove that qualification. Reproduction helpers use `phasesmith.single_crystal`;
old binary baselines use an explicit benchmark-only namespace selector. Original
JSON receipts and their hashes are preserved, not regenerated.

The [single-frame multigrain plan](docs/single-frame-multigrain-plan.md) retains
the unfinished roadmap proposal. It is deferred work, including weak-grain
recovery and signed prediction-led extraction, not a current capability claim.

## Ownership and dependency direction

| Layer | Owner |
| --- | --- |
| Profile/background kernels | `phasesmith-core` |
| Cells, symmetry, scattering, structure factors | `phasesmith-crystallography` |
| Bounded worker/execution policy | `phasesmith-execution` |
| Structural pattern composition | `phasesmith-engine` |
| Parser-independent application records | `phasesmith-model` |
| CIF, powder, space-group and instrument adapters | `phasesmith-io` |
| Powder and single-crystal workflows | `phasesmith-workflows` |
| Versioned projects and lossless numeric archives | `phasesmith-persistence` |
| Detector, geometry, profile and scaling kernels | `phasesmith-single-crystal::spots` |
| Independent DAFi-based row grouping | `phasesmith-single-crystal::row_search` |
| Structure-search kernels | `phasesmith-single-crystal::solution` |
| Optional bundled detector-file input | `phasesmith-hdf5` |
| External comparisons and validation orchestration | `phasesmith-validation` |
| Native application facade | `phasesmith` (`crates/phasesmith-rs`) |
| Array translation and Python bindings | `phasesmith-py` and `python/phasesmith` |

Keep serialization, file adapters, FFT/search dependencies and application state
outside `phasesmith-core`. Do not collapse these dependency boundaries merely
to reduce the number of directories. Large binding/coordinator modules can be
split internally while preserving public signatures and operation order.

## Numerical and provenance invariants

- GSAS-II is a pinned external oracle only. No runtime import, implementation
  copying/mechanical translation or GSAS-II object model enters the kernel.
- Derive equations independently from publications/first principles. Keep the
  readable Python references independent of Rust production evaluation.
- Compute values and analytical derivatives in the same peak/sample pass.
  Finite support, clipping and endpoint semantics are observable contracts.
- Preserve deterministic scalar/reduction order unless a measured, documented
  change establishes explicit tolerances. Never loosen global tolerances.
- Preserve signed observations, uncertainties, flags, explicit equivalences,
  supplied priors and truthful termination. A low R1 is not structure proof.
- Keep source/teacher/geometry identity, immutable array ownership, bounded
  allocations and cancellation/result-order semantics explicit.
- Do not silently regenerate golden data, rewrite historical receipts or select
  scientific settings using source HKLs, reference coordinates/counts or R1.
- Scattering tables and vendored helpers retain exact provenance and notices.
  Generated table size alone is not a reason to hand-rewrite scientific data.

## Public APIs and compatibility

The Python owner is `phasesmith.single_crystal`, with explicit exports and lazy
submodule ownership. The maintainer chose to remove pre-release
`phasesmith.experimental` imports after migrating their Dioptas consumer.
Independent references live under `phasesmith.single_crystal.reference` as
validation support, outside the supported production exports.

The Rust facade defaults to `single-crystal` (`spots`, `solution`, `row-search`).
`--no-default-features` keeps a smaller facade; `hdf5` remains optional for Rust.
Experimental Cargo feature aliases are removed. Normal Python builds include
the native capabilities and bundled HDF5. By maintainer decision, 0.8 removes
the internal legacy Python facades from the 0.5 namespace transition: top-level
adapter/orchestration aliases, aggregate method-owned refinement names and
aggregate dataset-converter names. Their owning modules remain the supported
imports. This supersedes the earlier plan to retain these aliases until 1.0.
Retain the current Python convenience exports and Rust application facade; these
are supported entry points, not retired compatibility layers. No numerical or
persisted-data behavior changes.

`ElementSearchConfig.validated_v1` contains a literal, complete, typed expansion
of the reviewed controls. Changing a low-level default must not change that
named profile. Explicit low-level algorithm defaults are preserved.

New solution summaries use `phasesmith.single-crystal-solution.v1`; explicit
v4/v5 metadata migration and legacy summary output remain available. Historical
profile-case/reduction/cache schema identifiers remain readable. Saved portable
formats are pickle-free; retired Python module paths in arbitrary old pickles
are not automatically migrated. Implementation hashes intentionally invalidate
numerical caches after source changes, without rewriting their provenance.

## Dependencies and builds

The workspace requires **Rust 1.89.0**, matching the locked numerical dependency
floor; Python requires 3.11+ and NumPy 1.26+. The only mandatory Python runtime
dependency is NumPy. SciPy/Gemmi are explicit reference/interop/example extras,
not implicit production numerical fallbacks. Missing native omega-background
support raises an actionable error; `backend="scipy"` is an explicit reference.

HDF5/Bitshuffle/LZ4/zlib/LZF build from Cargo sources. Source builds need a C
compiler, CMake >=3.26 and Git; prebuilt wheels need no installed HDF5, h5py or
filter plugin. Keep the adapter optional for native Rust consumers. The direct
`libz-sys` static feature is intentional link configuration, not an unused import.

Keep Cargo/Python versions, the changelog, API snapshots and crate metadata
synchronized. Do not modify released snapshots to accept surface drift.
Document default, minimal and optional-feature builds truthfully.

## Evidence and remaining scientific limits

The MgSiO3 example is exposed development/regression data with supplied
cell/element priors and optional exact symmetry, provisional spindle angles and explicit aperture
assumptions. Its eight source files total 470,052,794 bytes and stay external.
No reference atomic count/coordinate, source HKL or archived observation enters
the numerical worker. Reference grading follows candidate save and worker exit.

Retained v11–v24 execution checks reproduce masks, signed intensities, candidate
records and phasing histories bitwise: R1 .04085182070733226; Mg16 Si16 O48;
4,125 observations including 220 negative values; maximum/RMS reference
errors .0075236/.0045385 Å. The full measured workflow is about 19 s on the
recorded M4 Pro, not a platform-independent guarantee. The example uses at most
ten workers per concurrent stage and explicit independent memory budgets.

The [fully Rust coordinator replay](docs/single-crystal-native-validation.md)
uses the same priors and scientific controls, with all I/O/output native. It
recovers Mg16 Si16 O48 and passes reduction tolerances in 16.24–16.57 s observed
full-process time. Small native matrix roundoff changes the retained internal
polish trajectory: R1 .041128354912261904; the original R1-difference and
maximum-candidate-coordinate parity gates fail. Keep that qualification separate
from labelled structure recovery; the existing ADP-bound limitation is not fixed
by this orchestration change.

Candidate polishing has a documented ADP-bound stagnation limitation; do not
call every polish converged. The 65-case corpus is now exposed regression data,
not an untouched holdout. New generalization claims need new untouched evidence.
Powder limitations include the frozen QARR 1h bounded-workflow failure and
qualified GSAS-II FCJ/finite-support differences; preserve their reports.

## Validation and documentation

For numerical changes, follow `AGENTS.md`: equations/conventions, independent
references, values/derivatives, boundary/invalid-input tests, deterministic
comparisons, finite differences, normalization/moments, oracle coverage or its
reasoned exclusion, and realistic benchmark evidence.

Required checks include:

```sh
ruff check .
ruff format --check .
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
pytest
python scripts/public_api_snapshot.py --check
python scripts/release_version.py
python scripts/sync_math_docs.py --check
python scripts/sync_skill.py --check
mkdocs build --strict
```

Also check the minimum Rust toolchain, default/minimal/focused features, normal
installed wheels and source reconstruction. Cross-platform CI/manual release
execution remain separate from local results. Never tag/publish as a side effect
of maintenance or benchmarking.

User guides are in `docs/`, implementation evidence is indexed by
[development.md](docs/development.md), and the current assessment is
[package-review-0.8.md](docs/package-review-0.8.md). Raw data, full arrays, trial
outputs and distribution artifacts stay outside Git. Remove regenerable build
caches only after preserving unique evidence and verifying recovery artifacts.

The previous 3,400-line brief is preserved exactly in
[the repository archive](docs/archive/project-brief-before-package-review-2026-10-31.md).
It is excluded from documentation search so historical status cannot be mistaken
for current guidance. Frozen protocols and historical numerical receipts remain
unchanged; use their recorded Git/source snapshots for historical replay.
