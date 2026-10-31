# Single-crystal promotion audit

Audit date: 31 October 2026. Reviewed baseline: `54784c5` on
`codex/hkl-structure-solution`, incorporating integration `63bb354` and the
committed spot/intensity, FFT and Row search work through `297d157`.

Follow-up: the [explicit memory budget](solution-memory-budget.md), implemented
in `5e6bfa8`, resolves the search-capacity blocker for this example. Both existing
enstatite intensity sources recover the complete 80-atom labelled structure with
post-search maximum coordinate errors below 0.02 A. The original failing probes
below remain historical evidence. The [fresh raw-frame runner](enstatite-raw-workflow.md)
now executes all stages but produces no candidate; its structure-recovery gate,
public API promotion, packaging and the other release gates remain pending.

## Decision

Single-crystal analysis should become a normal PhaseSmith capability. The next
release needs a reviewed public surface, ordinary wheel coverage and an honest
measured workflow boundary before changing its supported status. This audit
changes no production equations, defaults, feature gates or saved formats.

The proposed public namespace is `phasesmith.single_crystal`; see the
[API proposal](single-crystal-api-proposal.md). MgSiO3 enstatite is the first
complete-workflow development case. Its data remain external to Git.

The intended endpoint is **PhaseSmith structure search and candidate assessment**.
There is no standalone single-crystal structure-refinement workflow. Positional
crystal refinement and candidate polishing inside a search are distinct existing
operations. External SHELXL refinement may provide an explicitly optional
diagnostic, but cannot satisfy the PhaseSmith structure-search gate.

## Source and evidence

The combined branch is clean at the reviewed baseline. The integration record
reports 3,158 Python tests and 485 Rust tests passing, together with formatting
and strict all-target/all-feature Clippy. These are historical baseline checks,
not a new test run by this audit. See
[integration validation](integration/structure-search-merge.md).

The active `/Users/clemens/Programming/PhaseSmith` checkout still contains
uncommitted detector/prediction changes and research reports. These remain
outside the reviewed baseline. Do not silently include them in a promotion or
replace that checkout. The older Documents checkout has an invalid Git HEAD;
the combined Programming checkout is the source of commits.

The [frozen structure-search comparison](solution-holdout-results.md) recovered
35 complete labelled structures among 65 measured-HKL cases. It establishes
bounded coverage with supplied symmetry and element types, starting from
externally integrated observations. The 65 cases are now exposed data. They
remain useful regression evidence; future claims of unseen-data improvement
need a new untouched set.

## Promotion work by area

| Area | Current evidence or implementation | Required action |
| --- | --- | --- |
| Native ownership | Spot/intensity kernels, row grouping and structure search have separate Rust crates | Retain separation; keep FFT and orchestration dependencies outside `phasesmith-core` |
| Installed capabilities | Python and Rust facade feature gates are `experimental-spots` and `experimental-solution` | Include supported capabilities in ordinary Python builds; define normal Rust defaults and preserve named opt-outs |
| Public Python surface | Modules live under `experimental`; many expose implementation helpers | Select explicit `__all__` exports and options; add compatibility aliases without duplicate type identities |
| Search defaults | Frozen benchmark uses many explicit flags beyond public defaults | Publish a named, fully expanded search configuration; verify it reproduces its tested baseline before changing defaults |
| Measured pipeline | Script-first raw-frame-to-PhaseSmith-search runner implemented; frozen fresh run produces no candidate | Diagnose reduction/phasing failure and pass complete coordinate recovery; retain earlier outcomes |
| Search capacity | Default remains 2,000,000 elements; explicit budget and input-only preflight implemented | Complete: 64 MiB admits 105 atoms; both existing intensity sources recover the correct 80-atom candidate; see follow-up |
| Scientific scope | Conditional profile errors, finite supports and empirical corrections are documented | Keep signed observations, quality flags, support conventions, explicit correction provenance and truthful termination |
| Optional packages | Geometry/strong-shape/scaling orchestration can require SciPy | Declare the installation extra and test missing-dependency errors; avoid import-time optimizer requirements |
| File loading | Image orientation, HDF5 decoding and masks have been application-owned | Provide a small optional example loader or adapter; keep Dioptas/Qt outside runtime requirements |
| Persistence | Profiles, recipes and search result schemas exist; some name `experimental` | Version migrations deliberately; preserve old readers and exact recipe meanings; do not replace schema IDs by text substitution |
| Rust distribution | Spots, row-search and solution crates have `publish = false` | Make the crates publishable with metadata/notices, or explicitly bound release scope; test the facade from packaged registry-resolvable dependencies |
| CI | Standard Python CI asserts spots are unavailable; special CI covers only an earlier subset | Require supported capabilities in default-wheel tests and run all relevant suites across supported platforms |
| Releases | Release workflow builds without experimental flags; docs/API description remain powder-focused | Add installed-wheel smoke workflows, sdist reconstruction and crate dry runs; update description, docs, snapshots, changelog and synchronized version |

Merely enabling both Cargo features does not complete these items. Conversely,
supporting bounded methods does not require universal structure recovery or
every research option becoming a stable public export.

## MgSiO3 finding

The supplied eight-file dataset totals **470,052,794 bytes**. The HDF5 file is
450,571,787 bytes, with a `uint32` array of shape `(144, 2162, 2068)` at
`/entry_0000/ESRF-ID11/eiger/data`. The historical reduction used zero-based
frames 2 through 143 and reversed image rows. Acquisition notes explain that the
first two frames were on the DAC seat/body. These are source-specific conventions.

The source CIF supplies an orthorhombic Pbca cell, Mg/Si/O element types and
wavelength. They are declared priors. The sample was already used for geometry,
profile and correction development, so this example is exposed regression data.
It cannot become a new blind holdout by changing its label.

Two concrete blockers were found:

1. The source `.hkl` contains 5,389 reflections followed by a `000` terminator and
   appended SHELX instructions/atomic information. The public strict HKL reader
   rejects that trailer. The benchmark explicitly separates the reflection
   block, retains the original and archives the trailer outside solver inputs.
   No reader behavior is relaxed.
2. Both that reflection block and the archived PhaseSmith-reduced observations
   expand to 10,694 allowed measured reciprocal rows. The established
   input-only resource policy limits the search to **36 full-cell atoms**:
   `min(128, ceil(V/8), floor((floor(2000000/M)-7)/5))`. The reference has
   **80 non-H atoms**. Thus complete recovery is impossible under this cap.
   The public volume-only default instead requests more than the dense guard
   allows and needs a clearer preflight resource diagnostic.

Do not set the search atom limit from the reference count, trim reflections to
make the answer fit, or silently change the existing guard. The next numerical
unit should review bounded workspace allocation/caching or a matrix-free path,
with an explicit caller memory budget, independent checks, deterministic
comparisons and realistic benchmarks. Preserve frozen comparison evidence.

The archived 29 October raw-count rerun recalculated counts, masks, backgrounds,
profiles, sensor coefficients and scaling. Its protocol retained global geometry,
HKL identities, measurement windows, teacher/family memberships and a structural
starting template. Its external SHELXL R1(all) was 4.222218% with matched source
sigmas or 4.156486% with its own error model on 5,389 observations/1,479 unique
reflections. These are existing refinement diagnostics, not PhaseSmith structure
recovery. The manifest pins the original protocol, preparation and comparison
receipts by hash rather than importing their research scripts.

## Implementation order and exit gates

1. **Source/API preparation.** Resolve inclusion of remaining uncommitted work;
   retain this baseline otherwise. Adopt the module/export/configuration plan.
   Inventory mutable arrays, exceptions, units and serialized contracts. Freeze
   a migration fixture before moving implementation modules.
2. **Search resource capacity (completed in follow-up).** Implement a general bounded resource solution
   and input preflight. Run the repository's full numerical gates and a realistic
   many-reflection benchmark. No enstatite-specific atom-cap default.
3. **Enstatite workflow.** Implement the external-data runner described in the
   [dataset protocol](enstatite-workflow.md). Freeze stage-specific checks before
   execution. Run PhaseSmith search with no reference coordinates or counts.
   Save a result even if it fails; investigate without rewriting the baseline.
4. **Public modules and packaging.** Move the reviewed implementations, preserve
   aliases, include ordinary-build native features, define optional extras and
   persistence migrations, and update package resources/docs together.
5. **Distribution checks.** Build fresh wheels and sdists; install outside the
   checkout; require native capability presence and working pipeline/search
   smoke cases. Run complete default-build Python tests, full Rust checks,
   cross-platform jobs, strict documentation, API/skill checks, package dry runs
   and representative performance comparisons.
6. **Release candidate.** Review the supported scope and remaining limitations,
   update the synchronized release version and changelog, then run the manual
   unpublished release workflow before merging/releasing.

For the enstatite gate, successful process exit, low amplitude R1, and correct
total atom count are separate from a correct labelled structure. Assess exact
per-element full-cell counts and periodic coordinate recovery after saving the
chosen model. The existing 0.05 A maximum / 0.025 A RMS count-and-coordinate
comparison can be proposed as a common reference gate, subject to the source
model's suitability and uncertainty. Record supplied symmetry, cell setting and
any origin/inversion transform. No standalone refinement or experimental
uncertainty claim follows from internal search polishing.

The full promotion remains pending these exit gates. Default package behavior
is deliberately unchanged by this preparation task.

## Preparation checks

The new external-data integrity/conversion tests and existing corpus-adapter
checks pass: 24 tests. Ruff and formatting pass for the new Python tooling.
All eight source files pass streamed byte-size/SHA-256 verification.

One bounded search was executed for each existing intensity source, sequentially.
Both returned no candidate with `target_met=false`, `termination=search_budget`,
and no process timeout. Complete structure recovery and a fresh raw-frame
workflow remain unperformed. The small summaries, exact input/native hashes and
explicit search packets are retained in
`benchmarks/enstatite-promotion-search-macos.json`; large inputs, prepared HKLs
and diagnostic search artifacts remain external/ignored.

The full strict MkDocs build currently fails on 19 inherited links from
`solution-corpus-benchmark.md`, `solution-count-selection.md`,
`solution-phase-ablation.md` and `solution-phase-quality.md` to repository
`../benchmarks/*.json` files outside the documentation tree. These source pages
are unchanged by this task, and no warning names a newly added page. Resolve
the documentation artifact-link strategy before the release-candidate docs gate;
do not suppress warnings or claim that strict documentation passed. No Rust or
full-package numerical checks were rerun for these documentation/benchmark-tool
changes; the earlier integration results remain separately attributed.
