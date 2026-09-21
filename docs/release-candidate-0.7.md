# PhaseSmith 0.7.0 release candidate review

## Version proposal and scope

Propose **0.7.0** as the next published release after 0.5.0. Main already
records the intentional 0.6 namespace cleanup and its immutable API snapshot,
but 0.6 has not been tagged or published on GitHub. The combined candidate adds
the validated develop/Pawley implementation without replacing that baseline.
The package metadata uses 0.7.0 for candidate builds; this document does not
authorize a tag or publication. This is a pre-1.0 release, not a 1.0 stability
claim or a published `rc1` artifact.

The reconciliation starts from develop `2e5edb7` and merges main `9207016`.
Both histories and all preservation branches remain intact. Main contributes
module-owned Python exports, warning-backed compatibility aliases, updated
examples and bounded validation-download retries. Develop contributes the
qualified Pawley/FCJ integration, truthful Rietveld stopping, performance work,
and the distributed agent instructions. No solver or profile equation is
changed by this reconciliation.

## Proposed release notes

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
outside the checkout. No saved-project schema is changed by the merge.

## Validation

Candidate validation is in progress. The prior Pawley scientific, differential
and benchmark evidence is in the [repair report](pawley-repair-20260921.md).
The combined candidate must pass fresh installed-wheel tests, Rust checks,
API snapshots, strict documentation, installed examples, and the complete
distribution dry run before being recommended for publication.

## Known scientific limits

- The frozen QARR 1h bounded-workflow assessment still fails its existing gate.
  Exploratory larger-budget successes do not replace that result.
- Strict equivalence to the pinned GSAS-II Pawley profile remains false;
  diagnosed FCJ quadrature and finite-cutoff differences remain visible.
- Matrix-free Pawley omits global rank/covariance. Hard-support convergence
  claims remain local to the documented support objective.
- Pawley does not establish a unique structure or Rietveld mass fractions.
  Automation remains Rietveld-only, and feasible-width research stays separate.

## Publication decision

Review the reconciled change and this version proposal after validation.
Publishing would be a separate action: approve the final version/release notes,
merge the reviewed candidate to main, and create its release tag. The release
workflow publishes only on tag pushes; a manual dispatch tests distributions
without publishing them.
