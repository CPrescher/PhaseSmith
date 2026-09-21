# Pawley consolidation review, 2026-09-21

## Follow-up outcome: promotion gates resolved

The [Pawley repair](pawley-repair-20260921.md) resolves the Intel convergence
failure and installed CLI paths without relaxing any scientific or convergence
gates. The final distribution matrix and measured validations pass. The initial
review below is retained to explain the earlier decision and its evidence.

## Initial decision: keep the integration branch separate

`codex/pawley-integration` is preserved for continued review. Develop receives
only the validated Rietveld and skill-packaging consolidation. This branch
combines the existing Pawley history with that baseline and the shared FCJ
geometric-Jacobian correction. Its local success does not make it portable
across all supported wheel targets yet.

## Gates found by the initial review

The [release dry run](https://github.com/CPrescher/PhaseSmith/actions/runs/35579612465)
fails on macOS Intel in
`tests/test_pawley.py::test_joint_composed_lorentzian_boundary[dense]`.
The fit meets `rwp < 1e-6`, but returns `stagnated` where the unchanged test
requires `converged`. The matrix-free counterpart passes. Reproduce using the
release-built Intel wheel and:

```sh
python -m pytest -q tests/test_pawley.py -k joint_composed_lorentzian_boundary
```

Investigate the dense bounded solver and the actual convergence certificate;
do not relabel stagnation or relax the existing assertion. Windows, Linux,
macOS ARM64 and sdist jobs pass in the same run. The earlier Windows failure
was byte-hash mismatch from checkout CRLF conversion; `.gitattributes` now
keeps oracle JSON as LF, and the subsequent Windows run passes. Golden fixture
contents were not regenerated.

Two installed validation CLIs also have repository-path assumptions:

```sh
python -m phasesmith.validation.tof_pawley --data-root validation/data
python -m phasesmith.validation.pawley_spectrum
```

They look for manifests/fixtures under the installed Python library directory.
The original tracebacks are retained in dated result text files. TOF numerical
validation passes with an explicit `--manifest
validation/pawley-tof-acceptance-v1.json`. Spectrum numerical diagnostics pass
through `oracle_comparison(Path("oracle/fixtures/wavelength_components_v1"))`
and `measured_comparison(Path("validation/data/iucr-ceria-size-strain-round-robin"))`
in the installed `phasesmith.validation.pawley_spectrum` module. These alternate
calls establish numerical results, not a fix for the default CLI behavior.
Future validation needs explicit external asset paths with clear errors.

## Completed evidence

The reviewed native binary SHA-256 is
`47d21c4130a2596ed6ca7491201b777ce222cd7123b10e743b3b1046b972be68`.
Evidence records installed runner, source and input hashes where supported.
Measurements ran sequentially with one BLAS worker on macOS ARM64, after the
builds and ordinary tests. The local suite reports 964 passed, 11 skipped and
34 marker deselections; [normal CI](https://github.com/CPrescher/PhaseSmith/actions/runs/35579826329)
also passes. Public API, strict documentation and formatting checks pass.

The unchanged eight-case Rietveld panel retains seven passes and the existing
QARR 1h failure, with exact repeated scientific records. Shared FCJ changes
therefore do not introduce a new panel failure. PbSO4 X-ray median time moves
from about 4.42 to 4.70 seconds; other large timings remain close. Three-repeat
200-peak/5,001-sample profile smoke measurements are retained; they are not a
strong claim of performance equivalence.

| Measured Pawley case | Rwp | Termination | Frozen gates |
| --- | ---: | --- | --- |
| APS sucrose, 811 reflections | 0.066054 | converged | passed |
| Echidna fixed cell | 0.284869 | converged | passed |
| Echidna cell/profile | 0.218412 | converged | passed |
| POWGEN TOF | 0.216797 | converged | passed |
| Nickel joint TOF | 0.020989 | converged | passed |

Sucrose repeats take about 134.8 seconds each, within the original 240-second
allowance. Matrix-free covariance/rank limitations remain explicit. The fixed
spectrum ceria diagnostic repeats exactly and changes Rwp from 0.210111 to
0.209520 for secondary/reference area 0.016. This small improvement is a
model diagnostic, not a newly established release gate. Pinned spectrum oracle
normalized maximum errors remain 0.00453–0.02302; no exact equivalence is claimed.
The separate Pawley oracle report passes its declared engineering gates and
retains `strict_fixed_profile_equivalence: false` for both cases.

The realistic CW, fixed-spectrum and TOF benchmarks all finish with their
repeatability checks intact. No goldens, fit budgets, scientific tolerances or
convergence assertions were changed to obtain these results.

Raw artifacts live under `validation/results/consolidation-20260921-*`:
combined assessment, CW/cell/TOF results, spectrum explicit-path diagnostics,
oracle comparisons, three Pawley benchmark panels, profile smoke timings and
the exact sequential command ledger. The ledger correctly records the two
original CLI failures. The explicit-path diagnostic JSONs are additional
results and do not replace those failures.

The follow-up repair resolves these gates and repeats the full distribution
matrix before promotion. Original separate Pawley evidence snapshots remain
available for provenance.
