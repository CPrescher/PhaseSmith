# 0.8 release dry-run record

This records the consolidated 0.8 preparation, separate from the earlier
[promotion receipt](release-candidate-0.8.md) and [package audit](package-review-0.8.md).
Version 0.8.0 remains unpublished until the reviewed release commit is tagged and
registry publication is explicitly performed.

## Completed facade-cleanup dry run

The maintainer subsequently requested removal of the internal legacy Python
facades before publication. The final tested cleanup commit is
`57f8da90732e4b1edb49563231523551839c242a`; this receipt is a documentation-only
follow-up. All 168 retired aliases are removed (67 top-level, 85 refinement,
16 I/O); the owning modules remain importable without deprecation warnings.
Current `__all__` exports, numerical code and portable data readers are unchanged.
Three benchmark/tool callers, embedded Python in three Rust test files and the
API documentation now use the owning modules. A scan of 1,162 tracked files,
including 704 parsed Python snippets, found no active retired-alias callers.
Both Dioptas checkouts also have no callers of these retired aliases.

| Gate | Result |
| --- | --- |
| [Release dry run](https://github.com/CPrescher/PhaseSmith/actions/runs/41720236629) | All five wheel targets, source build/install/tests, eleven verified Rust archives and strict docs pass |
| [General CI](https://github.com/CPrescher/PhaseSmith/actions/runs/41720237004) | All nine jobs pass, including Python 3.11–3.14, NumPy 1.26.4 and the migrated Rust/Python differential contracts |
| [Installed single-crystal wheels](https://github.com/CPrescher/PhaseSmith/actions/runs/41720236769) | Linux, macOS and Windows native/Python/CLI checks pass |
| Local suites | 567 Rust tests; 3,384 Python tests with NumPy 2.5.4 and all 3,384 again from an installed wheel with NumPy 1.26.4; all eleven optional Rust/Python workflow comparisons pass |
| Downloaded Apple Silicon wheel | NumPy-only capability/filter smoke, synthetic tutorial and fresh original-frame MgSiO3 control pass |
| Artifact review | Five wheels and the sdist contain the cleaned namespaces; eleven Rust archives identify the clean final source commit; metadata, licenses and all seventeen SHA-256 hashes reviewed |
| Measured regression parity | All 68 arrays in ten NPZ files, complete candidate record and both CIFs remain bitwise identical to the validated baseline; accepted Pbca/Mg16 Si16 O48 at R1 0.0514043045582282 |

The [cleanup receipt and artifact hashes](https://github.com/CPrescher/PhaseSmith/blob/develop/validation/results/release-dry-run-0.8-facade-cleanup-20261103.json)
record the exact source, checks and publication boundary. Local suites retain
27 explicit skips, 34 external-data/oracle marker exclusions and 34 ignored Rust
tests; optional cross-language comparisons run separately. The measured control
remains exposed regression evidence, not independent scientific validation.

An earlier cleanup attempt exposed old imports embedded in Rust differential
tests; those imports were migrated before the final green runs above. The
pre-cleanup numerical receipt below remains evidence for its own exact commit.
No tag or package publication was performed.

## Completed pre-cleanup dry run

The tested source commit is `f41f6cae2e6c07cc181a5e3ec12ce23fcd9d6495`.
Its receipt was a documentation-only follow-up to that tested source; the later
facade cleanup is validated separately above.

| Gate | Result |
| --- | --- |
| [Release dry run](https://github.com/CPrescher/PhaseSmith/actions/runs/41713728212) | All five wheel targets, source build/install/tests, eleven verified Rust archives and strict docs pass |
| [General CI](https://github.com/CPrescher/PhaseSmith/actions/runs/41713728515) | Rust/Clippy/docs, Rust 1.89, Windows persistence, Python 3.11–3.14 and NumPy 1.26.4 pass |
| [Installed single-crystal wheels](https://github.com/CPrescher/PhaseSmith/actions/runs/41713728368) | Linux, macOS and Windows native/Python/CLI capability and comparison checks pass |
| Local full suites after metric fix | 567 Rust tests and 3,384 Python tests; all 3,384 Python tests also pass with an installed NumPy 1.26 wheel |
| Final downloaded Apple Silicon wheel | NumPy-only capability/filter smoke, synthetic tutorial and fresh original-count MgSiO3 control pass |
| Measured numerical parity | All 68 arrays in ten NPZ files, complete candidate record and both CIFs are bitwise identical to the validated baseline; Pbca/Mg16 Si16 O48 remains accepted |
| Artifact review | Five wheels, one sdist and eleven Rust archives have reviewed metadata, versions, licenses, source identity and consolidated module layout; SHA-256 hashes retained |

The [machine-readable receipt and all artifact hashes](https://github.com/CPrescher/PhaseSmith/blob/develop/validation/results/release-dry-run-0.8-20261103.json)
record the exact commit and job results. Local Python suites retain 27 explicit
external-input/native-cross-comparison skips and 34 external-data/oracle marker
exclusions; Rust retains 34 explicitly ignored tests. Required remote differential
and native cross-comparison checks run in their designated jobs. Earlier checks
and subsequent fixes are retained below rather than rewritten as new science.

Registry publishing, attestations and GitHub Release creation are intentionally
skipped on a manual dispatch. No release tag is created. The new-crate bootstrap
and final release-status wording/main/tag steps remain publication operations,
as described in [the release process](releasing.md).

## Validated scope

- Add a runnable NumPy-only synthetic tutorial, full native file-workflow input,
  configuration/result/error guide, migration instructions and shared Rust/Python
  geometry, integration/scaling and structure-search mathematics.
- Consolidate three unpublished numerical component crates into
  `phasesmith-single-crystal`, with selectable `spots`, `solution` and `row-search`
  features. Keep facade/Python APIs, equations, protocols and reference data.
  HDF5 remains separate. The public workspace now has eleven crates.
- Correct minimum-NumPy positional-test startup: both parent and spawned workers
  inherit single-thread BLAS settings before NumPy import. Existing exact model
  bytes and audit comparisons remain the gates, with no relaxed tolerances.
- Add verified Rust archives, strict rendered docs and native Linux ARM wheel
  tests to the nonpublishing release workflow. Gate registry jobs on all checks.

## NumPy 1.26 investigation

The earlier CI run on Linux x86-64 failed one legacy positional-proposal test:
serial and spawned fits differed by roughly 2e-13 in the audit histories. Its
helper changed environment variables after NumPy was imported, while new workers
loaded BLAS with one thread. The documented process-search contract requires
limits before starting Python. On the same runner, the diagnostic with those
startup limits produced exact audit equality and exact model bytes, using NumPy
1.26.4/OpenBLAS 0.3.23.dev. Arm64 macOS also passed that configuration.

The diagnostic `benchmarks/positional_reproducibility.py` retains first-step
inputs/normal matrices, runtime thread information and complete comparisons.
Minimum-NumPy CI runs a checked proper-startup case and an unchecked late-setting
control. A late control may happen to match on another processor; that does not
make changing environment strings a supported way to reconfigure loaded BLAS.

## Execution receipt

The first consolidated source commit is `22f7369b62a5ae9e213c0ced801ccf64a79da96c`.
Its local checks passed: 567 Rust tests (34 explicitly ignored), 3,381 Python tests
(27 skipped and 34 external-data/oracle tests deselected), strict Clippy/rustdoc,
minimal and individual-feature builds, Ruff, strict MkDocs and API/version/skill/
mathematics checks. All 70 relocated numerical source files match their originals
after namespace substitution and Rust formatting.

A fresh installed macOS arm64 wheel with only NumPy passes native-capability and
HDF5-filter smoke, the synthetic tutorial and offline advisor-cycle example.
The wheel links system libraries only. Its source archive contains all three
consolidated modules and optimizer notices, without retired crate source paths.

A fresh original-count MgSiO3 run from that installed wheel accepts Pbca with
Mg16 Si16 O48. All 68 arrays in ten NPZ files, the complete candidate/result
record (including phasing histories), P1 CIF and selected-setting CIF are bitwise
identical to the earlier validated native control. The recorded elapsed time is
15.49 s; this is a regression control, not a new performance comparison or
independent scientific validation.

The first [remote dry run](https://github.com/CPrescher/PhaseSmith/actions/runs/41698688678)
verified Rust package compilation but exposed two harness defects: the Rust
archive upload pointed above Cargo's `tmp-crate` directory, and external-working-
directory pytest could not import repository benchmark helpers. The corrected
workflow retains the verified archives and adds only the repository root to the
test helper search path; `python/` is not added, preserving installed-package
coverage. The installed wheel also passes all 3,381 tests from outside the
checkout, and all 21 explicit native/Python cross-comparison tests pass.
Package-content review adds the canonical PhaseSmith MIT license to each public
crate alongside applicable third-party notices; release metadata checks enforce parity.

The [second run](https://github.com/CPrescher/PhaseSmith/actions/runs/41699952961)
passes every target except Windows, where 3,377 tests pass and four benchmark-
harness checks fail. Fixes preserve the benchmark's owned process-tree timeout
on Windows, report unavailable load-average metadata as null, and force LF for
committed corpus/protocol JSON so Windows checkout conversion cannot invalidate
recorded byte hashes. Frozen JSON records and numerical thresholds are unchanged.
The 66 focused portability/provenance tests pass locally, including emulated
Windows checkout hashes. The [next run](https://github.com/CPrescher/PhaseSmith/actions/runs/41703054562)
clears those four Windows failures and exposes one mismatched bundled/reference
JSON copy. The final policy fixes LF for all tracked JSON, after checking all
695 originals are UTF-8/LF, and adds an early Windows native-asset check before
compilation. No fixture or frozen JSON content is regenerated. A full-platform
run verifies the final policy.

The [following run](https://github.com/CPrescher/PhaseSmith/actions/runs/41705768773)
clears Windows and every other target except one Intel macOS exact training/
held-out fit check (maximum basis difference about 1e-22). A focused trace finds
identical starting bases, native Jacobians, residuals and Gram matrices, with the
first differing value in the Gram scale. NumPy's axis-free norm uses a BLAS
vector-dot fast path; Accelerate changes its last bit with buffer alignment.
Explicit matrix-axis Frobenius reductions in both metric-chain and retraction
preserve the equation and exactness gate. The regression fails before this fix
and passes afterward. All 73 cell-comparison tests pass locally; selected NumPy
1.26 tests pass. A [focused Intel run](https://github.com/CPrescher/PhaseSmith/actions/runs/41712658106)
passes 100 paired fits, including every captured intermediate, plus the full
cell-comparison module. The temporary wheel-reuse diagnostic workflow is then
removed; its reusable benchmark and captured evidence remain.

A 1,000-index benchmark (ten paired-fit repetitions, Apple Silicon) records median
0.2345 s before and 0.2314 s after, within timing noise. The maximum fitted-basis
change is 1.85e-22, with unchanged recommended oP setting and history lengths.
No tolerance, acceptance criterion, native kernel or frozen observation is changed.
The completed full-platform results at the top include this correction.

## Publication boundary

The new names `phasesmith-single-crystal` and `phasesmith-hdf5` need first uploads
with an owner credential before crates.io trusted publishing can cover them.
Follow the [reviewed bootstrap sequence](releasing.md#registry-configuration-and-new-crate-names).
No dry-run command creates the release tag or publishes packages. Public Read the
Docs `stable` remains on the published release until its release build completes.
The scientific scope remains bounded candidate search, with the exposed corpus
and provisional acquisition geometry documented in their original receipts.
