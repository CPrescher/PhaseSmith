# CW Pawley validation and performance

This implementation targets one fixed-wavelength CW histogram, independent
family areas and selected joint cell/profile/background variables. It uses the
existing native CW/FCJ kernels; no GSAS-II runtime is imported. Full live GSAS-II
Pawley optimizer equivalence, matrix-free solving, wavelength spectra and TOF
are not claimed.

## Reproducible checks

The independent NumPy reference checks randomized CW and fixed-axial FCJ values
and analytical derivatives. Centered finite differences exercise selected
profile and triclinic cell chains away from support boundaries. Small bounded
linear problems are checked against exhaustive active-face enumeration.
Native and Python tests cover overlapping families, exact coincidences, rank,
signed areas, unobserved/masked data, dependent bounds and ties, zero-area starts,
zero Lorentzian-width solutions, finite-support endpoints and area/centroid
behavior. Cancellation preserves accepted states; continuation is deterministic.
Persistence tests reject stale data, malformed checkpoints and byte-limit abuse.

The existing pinned `cw_instrument_profile_v1` oracle fixture validates fitted
area conventions: area relative tolerance 8e-6 and calculated-profile relative
L2 below 6e-6. See `oracle/PAWLEY_STUDY.md` in the repository for exact revision,
provenance and the distinction between profile comparison and optimizer parity.
No GSAS-II implementation code was copied or translated.

```sh
cargo fmt --check
cargo clippy --workspace --all-targets --all-features
cargo test --workspace --all-features
pytest
python scripts/public_api_snapshot.py --check
python scripts/public_api_snapshot.py --check --module phasesmith.refinement.pawley \
  --snapshot api/python-pawley-api-unreleased.json
python scripts/sync_math_docs.py --check
mkdocs build --strict
```

## Measured gates

`validation/pawley-acceptance-v1.json` freezes ranges, quality thresholds and
40-iteration ceilings before the measured runs. The runner also enforces a
240-second per-fit cooperative budget; final diagnostics can extend elapsed
time beyond that target. Each run verifies dataset
checksums, uses supplied one-sigma weights and repeats twice. Both runs must
converge, reduce chi-square by more than 5%, preserve nonnegative areas, reproduce
manual Rwp within 1e-12, and return exactly equal calculated arrays and histories.
The runner does not download or alter input data.

The model fixes cell and wavelength, refines all five CW coefficients plus areas
and one residual-background constant, and fixes a Smooth Bruckner baseline.
Echidna's nonuniform grid uses an explicit 20-sample background window; sucrose
uses a 0.1-degree window. These backgrounds are disclosed preprocessing choices,
not GSAS-II background parity. The manifest thresholds are Rwp < 0.30 and profile
correlation > 0.90 for neutron LaB6, and Rwp < 0.25 and correlation > 0.95 for
synchrotron sucrose. Correlation is computed after subtracting the fitted total
background.

```sh
python -m phasesmith.validation.pawley --data-root /path/to/validation/data \
  --manifest validation/pawley-acceptance-v1.json --output /new/path/results.json
```

Results are retained under `validation/results/pawley-20260917-*.json`.
The final `pawley-20260917-final.json` record passes every neutron gate: Rwp
0.2848692925, correlation 0.9575662657, 10 accepted steps, 13 families and 2,111
samples. The weighted
Jacobian has rank 18; a good profile fit does not establish individually
identifiable width coefficients. Earlier failed runs remain visible and were
not promoted to passes by changing acceptance thresholds. They exposed missing
composed-width constraints and the need for relative-objective convergence
with local-model agreement.

The same final record **fails the sucrose convergence gate**. It reaches Rwp
0.0660539713 (initially 0.6276574517), correlation 0.9967438710 and 16 accepted
steps over 811 families/23,003 samples. Both runs return exactly equal profiles
and histories, and every quality, nonnegativity and weighting gate passes.
However, both stop with `max_runtime` at the 240-second cooperative budget
(about 247 seconds including final diagnostics). This is a stopped accepted
state, not a converged result. The complete large-data release gate remains open.
Further dense-solver optimization or the planned matrix-free solver is needed
before claiming the full P5 milestone. Raising budgets is available to callers,
but no longer-budget convergence claim has been validated here.

The final binary SHA-256 in both the measured and synthetic reports is
`1d457593de812c9118e1a6bdc474149835b7b956e1760c5dd9b1da6010a96825`.
Historical failed development reports remain in the results directory. The
`preoptimization` sucrose record explicitly flags that its original hashes were
captured during a rebuild and cannot identify the loaded code.

The full Python suite passes 816 tests (11 external-data skips and 33
configuration deselections); the native workspace suite passes 345 tests,
including doctests. Clippy, warning-denying rustdoc, strict MkDocs, both API
snapshots, mathematical-document synchronization, guide examples and JSON Schema
validation pass. Repository-wide Ruff lint passes. Ruff formatting is clean for
all changed files; the existing `docs/fit-report.md` snippet needs a blank line
and remains untouched.

## Timing contract and limitations

`benchmarks/pawley.py --large --output /new/path/timing.json` measures three
serial release-build fits: 256 overlapping families on 10,001 samples, the same
size with fixed FCJ geometry and joint W refinement, and 811 families on 23,003
samples. Every repetition must converge, reproduce results exactly and recover
the synthetic profile with relative L2 below 1e-6 before any timing is reported.
The report records median/p95 wall time, process peak RSS, platform, Python and
NumPy versions, and binary/runner SHA-256 provenance.

The solver is deliberately dense. Reported peak RSS includes Python,
serialization and returned dense Jacobians, and may exceed `max_elements * 8`.
The explicit element limit bounds the estimated native workspace, not the whole
process. Timings characterize this implementation; they are not a speed claim
against Le Bail, Rietveld or another program with a different objective.

### Recorded dense baseline

The final release-build synthetic report is
`validation/results/pawley-20260917-synthetic-final.json` (three repetitions).
All cases converged with exactly repeatable accepted histories and profiles.

| Families / samples | Model | Median / p95 seconds | Relative profile L2 |
| --- | --- | --- | --- |
| 256 / 10,001 | symmetric, fixed geometry | 0.861 / 0.861 | 3.82e-14 |
| 256 / 10,001 | fixed FCJ, joint W | 1.357 / 1.370 | 1.89e-16 |
| 811 / 23,003 | symmetric, fixed geometry | 19.494 / 20.584 | 3.67e-14 |

Peak process RSS was 1,254,735,872 bytes (about 1.17 GiB). These measurements
were collected while separate validation jobs were running on other cores;
they are a reproducibility baseline, not isolated comparative performance.
The earlier `synthetic` development report precedes solver initialization and
constraint changes and is retained as historical evidence. No performance
claim is made for unchanged Le Bail/Rietveld workflows.
