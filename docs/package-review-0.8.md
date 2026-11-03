# Package review for 0.8

The counts and receipts below describe earlier 0.8 preparation stages. The
[final release dry-run record](release-dry-run-0.8.md) records the consolidated
release candidate, current platform gates and remaining publication steps.


Review baseline: `fbfc0d0`, followed by the independently committed Ruff cleanup
`0291c41`. This review covers package ownership, imports/features, build/runtime
dependencies, public configuration, documentation navigation/links, distribution
contracts and the Dioptas consumer. It does not retune numerical algorithms or
reopen the exposed structure-search corpus as a new holdout.

The main architecture is sound: keep the numerical core, domain records,
composition, workflows, adapters and persistence boundaries. Concentrate further
simplification on orchestration modules, configuration and documentation. The
most consequential findings are corrected below; larger refactors remain
explicit follow-up work.

The subsequent [native coordinator implementation](single-crystal-native-workflow.md)
addresses the Rust orchestration follow-up below. Its validation and numerical
limits are recorded separately; this review receipt remains unchanged.

## Findings and actions

| Priority | Finding | Action/state |
| --- | --- | --- |
| Release | 191 Ruff diagnostics and 41 formatting failures across scripts, references, tests and examples | Fixed repository-wide; no global rule suppression or numerical tolerance changes. |
| Release | Advertised Rust 1.85.1 was below locked `nalgebra`/`wide`/`safe_arch` requirements | Workspace/docs now declare 1.89.0; the actual 1.89.0 compiler passes the full workspace tests with all features; added minimum-toolchain CI. Updated version-gated Clippy idioms while retaining floating-point reduction order. |
| High | `validated-v1` inherited unspecified values from the current solver signature | All 51 controls are now literal and immutable. Tests simulate a changed future default and reject wrong scalar types. Existing valid expansion is unchanged. |
| High | Experimental compatibility imports added a second production namespace | Removed 35 wrappers and the experimental Cargo aliases after updating Dioptas. Ordinary `single_crystal` APIs remain the owner. |
| High | Independent numerical references were mixed with retired production imports | Retained all 35 references under `single_crystal.reference`; updated tests/tools, without wrapping Rust in reference equations. |
| High | Default omega background could silently select SciPy when native capability was missing | Default/native modes now report missing native capability. `backend="scipy"` remains an explicit independent reference path. |
| Medium | Several method guides contradicted promotion by claiming special builds or SciPy production fitting were required | Corrected current status, import locations and build guidance; historical result records remain unchanged. |
| Medium | Single-crystal method pages were difficult to discover | Added a dedicated navigation section, current release/API links and a development evidence catalogue. Existing page URLs are preserved. |
| Medium | The project brief mixed current instructions with 3,419 lines of development history | Current brief reduced to 167 lines. The exact earlier file is archived in Git and excluded from documentation search. |
| Medium | Release instructions still counted nine public crates and described the six-module 0.7 API snapshot as current | Updated to thirteen public crates and the 0.8 single-crystal surface. |
| Follow-up | Large coordinators and a broad low-level search signature | Prioritized below; no unvalidated numerical refactor is bundled into this review. |
| Follow-up | Complete original-frame orchestration still needs Python/NumPy | A typed native end-to-end coordinator remains the main step toward fully Rust-only workflows. |

The old single-crystal import paths are intentionally removed before the 0.8
release, as requested by the maintainer. Published powder aliases have a separate
1.0 deprecation policy. Legacy portable data readers and explicit summary migration
remain; arbitrary Python pickles containing retired module names are not migrated.

## Dependency assessment

Python has one mandatory runtime dependency: **NumPy**. SciPy supports independent
reference calculations and an explicitly selected linear solver adapter; Gemmi is
optional CIF interoperability/reference grading. Pytest, Ruff, JSON Schema and
mpmath belong to testing/development. MkDocs and its theme belong to documentation.
No Dioptas, Qt, GSAS-II, h5py or Python filter plugin is a PhaseSmith runtime
requirement. The empty `masking` extra is harmless installation compatibility;
removing it can be a documented future cleanup.

The locked Cargo graph has 15 workspace packages and 164 external identities.
That total includes development, optional and target-specific dependencies.
The more useful normal/build dependency counts are:

| Build selection | External normal/build packages |
| --- | ---: |
| `phasesmith-core` | 6 |
| Minimal facade (`--no-default-features`) | 61 |
| Default facade (single-crystal enabled) | 70 |
| Default facade plus bundled HDF5 | 101 |
| Python extension plus bundled HDF5 | 112 |

These are transitive Cargo package identities, not Python runtime requirements
or dynamic-library counts. The measured Apple Silicon wheel links system
libraries and bundles its HDF5/filter implementation.

| Direct dependency family | Owning use | Recommendation |
| --- | --- | --- |
| Rayon | Bounded execution, detector/profile/search work | Retain; use explicit per-call budgets and preserve ordered reductions. |
| RustFFT | Reciprocal grain discovery and structure phasing | Retain; removing it removes supported FFT algorithms. |
| ndarray | Blocked normal-matrix multiplication in `solution/linear_step.rs`, and NumPy/HDF5 adapters | Retain; it is used even though only a small kernel imports it directly. |
| nalgebra | Powder least-squares, eigen/SVD/Cholesky work and validation | Retain; replacement would require substantial numerical validation. Its locked version sets the actual Rust minimum. |
| moyo | Space-group database, settings, Hall parsing and operation construction | Retain. A future smaller database adapter can be evaluated independently, including general Hall expressions and provenance. |
| serde/serde_json | Portable records, row-search exchange and persistence | Retain; keep serialization outside the numerical core. |
| ZIP/flate2 | Lossless numeric/project archives and validation input | Retain. Existing feature selections already avoid broad default ZIP codecs. |
| SHA-2 | Source/count identity and cache/persistence fingerprints | Retain; source verification is observable scientific provenance. |
| hdf5-metno/Bitshuffle/libz | Optional native detector input and bundled filters | Retain in the adapter. `libz-sys` explicitly selects static linkage; absence of a direct Rust import is not evidence it is unused. |
| PyO3/rust-numpy | Optional Python binding layer | Retain outside Rust-only consumers. |
| Criterion | Development benchmarks | Retain as a development dependency; it is absent from normal/build counts. |

No unjustified direct production dependency was found in the inspected owning
code and feature graph. This is a usage/build-scope review, not an exhaustive
vulnerability or upstream-maintenance audit. Active normal builds use two `syn`
versions required by their respective macro dependencies; unused lockfile variants
of other packages do not establish duplicate runtime cost. Do not force version
unification or disable static-link features merely to shorten a dependency list.

Potential dependency reductions need their own measurements. In particular,
separating space-group lookup does not automatically eliminate nalgebra, because
powder workflows also use it. Keep HDF5 optional for Rust consumers; the extra
build/toolchain cost in normal Python source builds is deliberate and documented.

## Organization and complexity

Large files are useful pointers for inspection, not automatic defects. The
4,800-line `scattering_data.rs` is generated scientific data with provenance;
retain it as generated data. The more useful decomposition candidates are:

| Area | Observed size | Next step |
| --- | ---: | --- |
| `crates/phasesmith-py/src/lib.rs` | 4,418 lines | Extract profile/structure bindings and repeated array conversion into private modules, preserving public registration and ownership. |
| `python/phasesmith/refinement/rietveld.py` | 3,043 lines | Separate options/results, phase conversion and workflow coordination. |
| `python/phasesmith/automation.py` | 2,992 lines | Separate immutable schemas/planning from execution/review; preserve authorization and lineage contracts. |
| `crates/phasesmith-solution/src/shared_search.rs` | 2,992 lines | Split proposal generation, ranking and termination bookkeeping after exact replay coverage. |
| `python/phasesmith/single_crystal/grains.py` | 1,946 lines | Separate discovery, proposal refinement and evidence/selection policies. |
| `solve_structure_elements` | 58 arguments, 459 lines | Keep the existing explicit interface; introduce typed scientific/resource/strategy options incrementally rather than another parallel flag dictionary. |

The main future simplification is a typed Rust workflow coordinator spanning
original-count input, masks, indexing, geometry, profiles, scaling and search.
Python can remain an optional adapter. This should preserve explicit priors,
source identity, memory admission, callbacks/cancellation and stage evidence.
Moving orchestration must not introduce hidden default changes or a new
implementation of already validated numerical equations.

Other focused follow-ups are synchronizing copied third-party notices from one
canonical source and making the boundary between packaged validation bindings
and ordinary native bindings clearer. Existing validation entry points are
supported contracts; feature-gating them out by default requires an explicit
API/distribution decision, not an unused-import cleanup.

## Documentation and interlinking

The documentation now separates task guides from implementation evidence:

- [Single-crystal guide](single-crystal.md) and its method navigation cover current
  production APIs; historical filenames remain valid URLs.
- [Development index](development.md) makes retained scientific/research reports
  discoverable without presenting every experiment as the normal workflow.
- [API stability](api-stability.md), [API map](api-reference.md) and
  [release process](releasing.md) describe the current version and ownership.
- The old brief remains an exact repository snapshot and is excluded from
  documentation search to avoid stale status appearing as current instructions.

A rendered-site crawl checks local targets and fragments and reachability from
its index, in addition to strict MkDocs. The initial reviewed graph contained
181 HTML pages, 22,339 local link instances, no broken local targets/fragments
and no unreachable documentation pages. The final graph is recorded in the
machine-readable review receipt. Its external URLs are inventoried but not all
live-tested; no external-availability claim follows from the local link check.

## Scientific and release limits

This maintenance does not establish broader structure-recovery coverage,
absolute intensity accuracy or converged candidate polishing. Supplied priors,
provisional MgSiO3 geometry, conditional uncertainty models, the ADP-bound
stagnation limitation and the exposed status of previous validation sets remain.
See the [single-crystal guide](single-crystal.md) and
[release candidate](release-candidate-0.8.md).

Local results and the consumer migration are recorded in the review receipt.
Remote platform CI and the manual unpublished release workflow remain required
before a public tag. No publication, tag or push is performed by this review.


## Completed local validation

The final native source is `c7f48d0`; `92d69e3` additionally corrects current
examples and three unavailable-backend messages. All earlier scientific JSON
results remain unchanged. The small frozen v25 protocol records the new namespace
and package controls without changing the v24 scientific or execution settings.

| Check | Result |
| --- | --- |
| Repository Ruff lint and formatting | Zero diagnostics; 794 files formatted |
| Full Rust workspace, all features, Rust 1.89.0 | 515 passed, zero failed; 34 ignored |
| Strict Clippy, all targets/features | Passed |
| Strict Rust documentation | Passed |
| Minimal facade and minimal Python binding feature builds on Rust 1.89.0 | Passed |
| Full installed-wheel Python suite | 3,281 passed; 11 skipped, 34 deselected |
| Final guide/message follow-up and API tests from rebuilt wheel | 54 passed |
| Dioptas consumer checks | 335 passed across single-crystal, omega, reciprocal, CrysAlis and integration adapters |
| Dioptas offline lock check and pre-existing tracked edits | Lock valid; original file hashes unchanged |
| NumPy-only installed wheel | Detection, search and filtered HDF5 smoke passed with SciPy/Gemmi/h5py/filter-plugin imports blocked |
| Strict MkDocs and rendered local-link/fragment crawl | Passed; final graph counts are in the receipt |
| Fresh original-frame MgSiO3 v25, normal installed wheel | 19.06 s process wall time; recovery passed |

The raw replay retains the v11 arrays, complete candidates and phasing histories
bit for bit. Its best amplitude R1 is 0.04085182070733226; labelled coordinate RMS
error is 0.00454 angstrom after origin/hand alignment. The measured runtime is
consistent with the preceding 19.06–19.39 s runs; this maintenance makes no new
speedup claim. Input and all large products remain outside the repository.

The machine-readable receipt is `validation/results/package-review-0.8.json`.
Full logs, link graphs, tested distributions and the strict replay comparison
are retained outside the repository under
`/Users/clemens/ChatGPT/PhaseSmith/package-review-0.8-20261031`.
Dioptas migration commit: `31d79408`. No remote CI run, publication, tag or push
is implied by these local checks.
