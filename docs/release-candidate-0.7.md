# PhaseSmith 0.7.0 release review

## Version and scope

**0.7.0** is the next release after 0.5.0. It includes the intentional 0.6
namespace cleanup and its immutable API snapshot; 0.6 was never tagged or
published. The combined release adds the validated develop/Pawley
implementation and the final fresh-review repairs. This remains a pre-1.0
release, with no 1.0 stability claim or separate published `rc1` artifact.

[PR #7](https://github.com/CPrescher/PhaseSmith/pull/7) reconciled the histories.
[PR #8](https://github.com/CPrescher/PhaseSmith/pull/8) repaired the four
fresh-review defects below. Reviewed implementation commit
`4f4fe93941fe199cde90be3d7d1037e5ea346365` and its main merge
`002b76a7ba25dcbd5fcccb84172bf1a8e887a57d` have the same tree. Final release
preparation updates documentation only; numerical gates and oracle fixtures
remain unchanged.

The reconciliation starts from develop `2e5edb7` and merges main `9207016`.
Both histories and all preservation branches remain intact. Main contributes
module-owned Python exports, warning-backed compatibility aliases, updated
examples and bounded validation-download retries. Develop contributes the
qualified Pawley/FCJ integration, truthful Rietveld stopping, performance work,
and the distributed agent instructions. No solver or profile equation is
changed by this reconciliation.

## Release highlights

- Add native CW, fixed-spectrum and single-/multi-bank TOF Pawley fitting with
  bounded/tied family areas, analytical selected cell/profile/background
  derivatives, dense and matrix-free solvers, diagnostics and accepted-state
  restart. Pawley fits require indexed cells/reflections, not atom models.
- Add standalone Pawley codecs and lossless mixed-method native format-7
  bundles. Native formats 1–6 migrate; existing standalone formats remain
  readable. Persistence versions are independent of the package version.
- Correct the FCJ geometric Jacobian and powder Friedel-family averaging,
  and repair tiny-step Pawley boundary feasibility on Intel. Keep scientific
  acceptance thresholds explicit. Numerical results can change from 0.5;
  the namespace compatibility layer itself performs no numerical conversion.
- Verify Rietveld stopping independently of damping, with bounded rejected-step
  recovery and truthful stagnation. Include the validated native performance
  improvements and optional profile-accuracy/empirical-width controls.
- Clarify Python ownership: import method records/functions from their named
  refinement modules, converters from their named I/O adapters, and automation
  from `phasesmith.automation`. Explicit moved 0.5 imports keep the same objects
  with `DeprecationWarning` until 1.0. See [migration](migration-0.6.md).
- Include installed agent instructions with `phasesmith skill --path` and
  `--print`, clear external validation asset paths, and checksum-preserving
  download retries. Normal package use does not require GSAS-II.
- Retain complete fixed FCJ axial derivatives in native Rietveld results within
  the existing fused selected/trial passes. Returning the accepted state uses
  no extra profile evaluation, including after a runtime limit.
- Validate persisted profile-accuracy policies in refinement options and
  checkpoints, reject oversized unsigned CW/TOF Pawley HKLs before signed
  conversion, and correct stale native-persistence/schema documentation.
- Use supported integration names on both NumPy 1.26 and 2.x and exercise the
  declared NumPy minimum in CI.

## API and persistence review

The 0.4.1 and 0.6.0 aggregate API snapshots are retained byte-for-byte.
Develop had updated the file labelled 0.5.0 after its release; the canonical
0.5.0 snapshot is restored exactly from tag `v0.5.0` (matching main), and the
develop variant is preserved as `api/python-public-api-develop-2e5edb7.json`.
New 0.7.0 aggregate and Pawley snapshots record the combined names, targets and
signatures. The original unreleased Pawley snapshot is retained as evidence.
`phasesmith.refinement.pawley` and `tof_pawley` follow the named-module layout;
their workflow signatures and the `project_bundle` API retain the repaired
develop contract. Newly integrated physical-domain and fit-diagnostic records
retain their develop exports. Compared with 0.6, the aggregate surface adds
eight physical/diagnostic names and two Pawley modules, removes no names, and
retains six already-reviewed develop signature additions for profile accuracy
and powder-family averaging. The exact diff is recorded in
`validation/results/release-0.7-api-diff.json`.

Review gates cover all removed 0.5 aliases, installed imports, native/Python
bundle exchange, older-format migration, accepted-state resume and examples
outside the checkout. The combined change adds native format 7 and optional
profile-accuracy fields to Rietveld records while retaining older-format
compatibility. The reconciliation itself adds no further schema changes beyond
those already present in the repaired develop implementation.

## Validation

The final reviewed installed wheel passes **981 Python tests** on each of
NumPy 1.26.4/Python 3.12 and NumPy 2.5.3/Python 3.13 (11 unavailable external-data
skips and 34 opt-in deselections each), **368 Rust tests/doctests**, **12
Rust/Python differential checks**, both frozen QARR 1g entry-point tests and
three pinned-spectrum engineering comparisons on NumPy 1.26. Strict Clippy,
Rust/Python formatting, generated-document checks and MkDocs pass.

All eight fixed real-data assessment cases have exactly identical scientific
records across the pre-repair and final builds, including evaluation counts,
and exact deterministic repeats within each build. Seven pass; the frozen
QARR 1h bounded-workflow failure remains. No scientific gate or tolerance was
relaxed. The tested native binary has SHA-256
`807e98b026b45102c84a556ed4b5e381b147b5e13867c99b81df489fbe367156`.

The [final distribution dry run](https://github.com/CPrescher/PhaseSmith/actions/runs/35619489937)
passes on implementation commit `4f4fe93`: Linux x86-64, Windows, macOS Intel
and Apple Silicon installed-wheel tests, Linux AArch64 construction, and an
independently installed source distribution. AArch64 Linux is build-tested,
not runtime-tested by this workflow. Publication jobs were skipped for this
manual dispatch. [Post-merge CI](https://github.com/CPrescher/PhaseSmith/actions/runs/35621053727)
also passes on `002b76a`. The [PR #8 report](https://github.com/CPrescher/PhaseSmith/pull/8)
records the final checks and timings. Its two review findings (an evaluation
budget overrun and stale persistence documentation) were fixed in `4f4fe93`;
their unresolved review-thread status does not indicate unfixed code.

Earlier reconciliation evidence remains in
`validation/results/release-0.7-*.json`: the focused API/TOF suite passed 30
tests with deprecation warnings treated as errors, all 168 moved aliases
resolved with their expected warnings, and installed CW/TOF examples, skill
discovery and the offline advisor example passed outside the checkout.
CW, cell/profile, TOF, fixed-spectrum and pinned-oracle engineering gates
passed with scientific records identical to the repaired develop reports
apart from timings, source/binary provenance and checkout paths. Sucrose took
133.36 and 137.11 seconds per repeat against its unchanged 240-second limit.
Those records describe the earlier reconciliation checkpoint; the final
fresh-review counts above supersede its 975 Python/365 Rust counts.

### Measured cost of restoring complete FCJ derivatives

The realistic 256-reflection/eight-site benchmark measured **6.60 ms** with
axial derivatives disabled and **7.59 ms** with them retained (about 15% for
that pass). Complete FCJ refinements cost approximately **6–10% more**, up to
10.3%, while other measured cases vary about -1% to +2%.

| Complete refinement | Before (s) | After (s) | Change |
| --- | ---: | ---: | ---: |
| QARR 1g | 0.7581 | 0.8023 | +5.8% |
| QARR 1h | 0.6013 | 0.6462 | +7.5% |
| PbSO4 CW X-ray | 4.5024 | 4.8208 | +7.1% |
| PbSO4 CW neutron | 0.5681 | 0.5630 | -0.9% |
| Rowles 1a | 3.7001 | 4.0616 | +9.8% |
| Rowles 1e | 3.2773 | 3.6146 | +10.3% |
| Echidna LaB6 | 0.0093 | 0.0095 | +2.0% |
| LANL nickel TOF | 52.4041 | 52.3543 | -0.1% |

These medians use two timed repetitions per build/case on a shared desktop;
they are not universal performance guarantees. Restoring the result contract
does not change optimization histories or evaluation counts.

## Known scientific limits

- The frozen QARR 1h bounded-workflow assessment still fails its existing gate.
  Exploratory larger-budget successes do not replace that result.
- Strict equivalence to the pinned GSAS-II Pawley profile remains false;
  diagnosed FCJ quadrature and finite-cutoff differences remain visible.
- Matrix-free Pawley omits global rank/covariance. Hard-support convergence
  claims remain local to the documented support objective.
- Pawley does not establish a unique structure or Rietveld mass fractions.
  Automation remains Rietveld-only, and feasible-width research stays separate.

## Publication workflow

The final documentation is reviewed through a pull request and merged after
its checks pass. The annotated `v0.7.0` tag must point to that reviewed main
commit. The tag-triggered workflow builds and tests distributions, publishes
PyPI and creates the GitHub Release with checksums and provenance. Publishing
the nine public Rust crates additionally requires the repository variable
`CRATES_IO_TRUSTED_PUBLISHING=true` and configured trusted publishers; the
crates job is skipped otherwise, and that skip does not block a GitHub Release.
The variable was verified enabled during 0.7.0 preparation. Verify all nine
registry versions after the run. Manual dispatch only validates distributions.

The [GitHub release](https://github.com/CPrescher/PhaseSmith/releases/tag/v0.7.0),
[PyPI version](https://pypi.org/project/phasesmith/0.7.0/),
[crates.io version](https://crates.io/crates/phasesmith/0.7.0) and
[Release workflow runs](https://github.com/CPrescher/PhaseSmith/actions/workflows/release.yml)
are the authoritative publication records; the preparation checks above do
not themselves establish successful publication. Follow the
[release checklist](releasing.md) to verify registry artifacts and installation.
Never replace an existing release tag.
