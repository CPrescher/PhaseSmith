# Pawley validation and performance

Follow-up: [the fixed-profile diagnosis](pawley-profile-diagnosis.md) isolates the discrepancy
to the pinned FCJ numerical evaluator and finite cutoff policy. Production
quadrature, width/area conversion and the Pawley optimizer pass the isolation
controls; “different physical models” is not established.

## Completed scope and final checks

The final combined implementation covers bounded/tied CW and TOF areas,
symmetry-constrained cells, selected profile/calibration parameters, linear
backgrounds, fixed CW spectra, dense and matrix-free solving, cancellation,
accepted-state restart and lossless native format-7 bundles. The original
checkout's committed changes through `9afd808c` are included; later concurrent
work in that checkout is outside this validated worktree. The historical reports below remain
unchanged so that failed experiments and intermediate limitations stay visible.

| Measured regression | Final scientific outcome |
| --- | --- |
| APS sucrose, 811 CW families | Converged, Rwp 0.06605375; hard-support local convergence is explicitly reported |
| Echidna LaB6, fixed cell | Converged, Rwp 0.28486929 |
| Echidna LaB6, selected cell/profile | Converged, Rwp approximately 0.2184 |
| Birmingham ceria, fixed detected spectrum | Converged, Rwp 0.20951987; exploratory comparison, not an independent holdout |
| POWGEN LaB6, 330 TOF families | Converged, Rwp 0.21679733, correlation 0.96900 |
| LANL nickel, three TOF banks and shared cell | Converged, joint Rwp 0.02098942; each bank below 0.03; a = 3.523669 Å |

Final release-build reports are
`validation/results/pawley-20260917-release-{cw,cell,tof,spectrum,oracle}.json`.
They record dataset/fixture, runner and native-binary provenance. Measured
repetitions compare complete arrays and histories exactly. Run the contracts
without changing the predeclared scientific thresholds:

```sh
python -m phasesmith.validation.pawley --data-root PATH \
  --manifest validation/pawley-matrix-free-acceptance-v4.json --output NEW_CW_REPORT
python -m phasesmith.validation.pawley --data-root PATH \
  --manifest validation/pawley-cell-acceptance-v3.json --output NEW_CELL_REPORT
python -m phasesmith.validation.tof_pawley --data-root PATH --output NEW_TOF_REPORT
python -m phasesmith.validation.pawley_spectrum \
  --data-root PATH/iucr-ceria-size-strain-round-robin --output NEW_SPECTRUM_REPORT
python -m phasesmith.validation.pawley_oracle --fixture oracle/fixtures/pawley_optimizer_v1 \
  --output NEW_ORACLE_REPORT
```

The scheduled real-data CI job fetches the pinned sources, runs these measured
contracts and uploads their reports. Normal pytest uses synthetic and committed
oracle fixtures; it does not require external measurement downloads or GSAS-II.
The fixed-spectrum and CW live-optimizer model exceptions remain explicit.
TOF is checked against pinned profile fixtures; live TOF optimizer parity is
not an acceptance claim.

The independent TOF reference and tests cover nonuniform grids, all fifteen
profile/calibration coefficients, six triclinic cell chains, dense/product
and adjoint agreement, area/centroid/support semantics, bank-local zero-area
release, single-bank reduction, DIFC/cell anchoring, complete joint cancellation
and restart, schema validation and Python/native bundle exchange. Observational
uncertainty conventions and matrix-free rank/covariance limits are documented
in [the TOF guide](tof-pawley.md).

## Final benchmarks and verification

`benchmarks/pawley.py` and `benchmarks/tof_pawley.py` ran sequentially after
the test workloads using the same release binary as all five final reports.
Each case repeats twice, requires convergence and profile error below `1e-6`,
and compares complete calculated arrays and accepted histories exactly.

| Workload | Median seconds |
| --- | ---: |
| 256 CW families / 10,001 samples, fixed symmetric | 0.0376 |
| 256 CW FCJ families / 10,001 samples, joint W | 0.0927 |
| 811 CW families / 23,003 samples, fixed symmetric | 0.2069 |
| Same CW workloads with a fixed two-component spectrum | 0.0472 / 0.1338 / 0.2997 |
| 256 TOF families / 10,001 nonuniform samples, dense | 1.9699 |
| Same single-bank TOF workload, matrix-free | 1.3662 |
| Three TOF banks, each 256 families / 10,001 samples, matrix-free | 4.1799 |

CW medians are comparable to the earlier 0.0394/0.0957/0.2145-second matrix-free
baseline; this run shows no meaningful regression from sharing the solver.
TOF maximum relative profile error is below `9e-12`. TOF process peak RSS is
262,471,680 bytes and includes the preceding dense case, Python and returned
arrays; it is not a per-fit or matrix-free-only memory measurement. The CW and
spectrum processes peak at 175,931,392 and 183,484,416 bytes respectively.
The `release-benchmark-*.json` reports retain diagnostics and hashes.

The final measured sucrose repetitions take 152.61/152.14 seconds and remain
within the unchanged 240-second budget. Some verification work ran concurrently
with this acceptance run, so these timings are not isolated speed comparisons.
Final nickel repetitions take 11.75/11.93 seconds with unchanged exact profiles.

Verification: 362 Rust tests/doctests pass (34 external/opt-in tests ignored),
and pytest reports 934 passed, 11 skipped, 33 deselected. External measured
checks above run separately from default pytest. Formatting, strict Clippy,
strict Rustdoc/MkDocs, synchronized mathematics, both API snapshots and actual
standalone/bundle schema validation pass. The Python-free TOF example and
Python guide example also execute successfully. CI definitions are updated;
remote CI has not been run from this worktree.

## Matrix-free follow-up

The optional matrix-free solver has dense/product/adjoint comparisons, bounded
and tied-step tests, multiphase cell/profile recovery, explicit memory limits,
and exact accepted-checkpoint continuation. Its standalone format-2 controls
are checked during restart; format-1 dense checkpoints remain readable.

The release-build two-repeat benchmark in
`validation/results/pawley-20260917-matrix-free-benchmark.json` records median
0.0394 s for 256 fixed symmetric families, 0.0957 s for 256 FCJ families with
joint W, and 0.2145 s for 811 fixed symmetric families on 23,003 samples.
Peak process RSS is 181,174,272 bytes, including Python and returned arrays.
Both repetitions have exactly equal profiles and accepted histories; relative
profile errors are below 4e-14. These are synthetic fixed-domain comparisons,
not replacements for measured convergence. Matrix-free mode omits global rank
and covariance explicitly. The matching previous dense benchmark required
14.66 s for the 811-family case and 1.21 GB peak process RSS.

The version-4 acceptance manifest selects this solver while retaining every
version-2 scientific input, quality threshold and runtime/iteration budget.
`validation/results/pawley-20260917-matrix-free-acceptance.json` passes both
measured cases. Sucrose takes 137.27/138.08 seconds, accepts 20 steps, and reaches
Rwp 0.0660537487 with `support_relative_objective` convergence. This is local
convergence of the exact hard-truncated objective, not of an infinite-support
profile. LaB6 takes about 0.015 seconds and reaches Rwp 0.284869293. Both runs
have identical profiles and accepted histories. Earlier dense failures below
are historical evidence; they are preserved rather than silently overwritten.
The combined state after upstream reconciliation also passes both gates; see
the final evidence above. Historical timings are not isolated comparisons to
the combined-state runs.


Current coverage includes monochromatic/fixed-spectrum CW and single-/multi-bank
TOF Pawley. Production uses the existing fused native profile kernels, without
GSAS-II runtime dependencies. The live pinned CW optimizer comparison passes
its declared engineering agreement gates; strict profile-model equivalence is
not claimed. TOF has independent NumPy/derivative and pinned-profile checks,
not a live external optimizer comparison.

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
  --module phasesmith.project_bundle --module phasesmith.refinement.tof_pawley \
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
  --manifest validation/pawley-acceptance-v2.json --output /new/path/results.json
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
samples. Every repetition must converge, reproduce the recorded summary metrics exactly and recover
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
All cases converged with exactly repeatable recorded summary metrics. The
benchmark currently compares relative L2, rank, accepted-step count and
termination; it does not directly compare arrays or histories. See the
[completion audit](pawley-completion-audit.md) for the follow-up validation work.

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

## Executable acceptance controls

Version 2 makes the resolved data range, cell/symmetry, instrument, support,
background window/iterations, profile selection, weights, intensity policy,
solver controls, runtime budgets and acceptance thresholds explicit. Both
resolved requests were compared exactly against the original version-1 runner;
the scientific inputs and thresholds are unchanged. The runner rejects obsolete
or unknown manifest fields instead of ignoring them. Version-1 reports remain
historical evidence; use version 2 for new runs.

New reports include checksum-verified input hashes, the complete request digest,
accepted history and native stage diagnostics. `--checkpoint-directory` can save
accepted states for follow-up diagnostics without overwriting files. The updated
synthetic benchmark compares actual arrays/histories and records their hashes,
plus evaluation and linear-iteration counts.


## Completion follow-up: cell fitting and support products

`validation/pawley-cell-acceptance-v3.json` adds an independent measured LaB6
cell/profile gate; it does not replace the unchanged version-2 sucrose gate.
Version 3 explicitly specifies selected lattice parameters, the conservative
cell domain and acceptance intervals. Its initial cubic cell is 4.16 Å and
initial U is 0.04 deg², keeping widths positive over the larger family envelope.
The fixed wavelength remains 2.047 Å. Run it with the same command above,
substituting that manifest and a new output filename.

`pawley-20260917-measured-cell.json` records convergence in both repetitions,
Rwp 0.2184121 and cubic a = 4.1593457 Å. Actual profiles and accepted histories
are identical. The declared cell interval is a sanity gate conditional on the
fixed calibration, not an independent certification of absolute cell accuracy.
Synthetic tests also recover two simultaneous cells and a shared profile width,
and check covariance scaling with supplied uncertainty and propagation through
a tied area ratio.

The native coupled-face regression fixes a numerical failure exposed by this
measured case: a nearly singular constraint projector admitted spurious tangent
directions. Explicit orthonormal null-space bases preserve the face equations.
Support-block JVP/VJP, weighted norms and adjoint identities now agree with the
dense Jacobian. The production solver is still dense.

`pawley-20260917-support-products-benchmark.json` records three repeated runs
with identical arrays and accepted histories: medians 0.680 s (256 fixed),
1.056 s (256 FCJ/joint W) and 14.657 s (811 fixed); relative profile errors are
unchanged. Peak process RSS is 1,210,236,928 bytes. Earlier timings were collected
under a different concurrent workload, so these numbers are not a controlled
speedup ratio.


The final follow-up report, `pawley-20260917-diagnostics-final.json`, preserves
both original measured requests and their thresholds. Neutron LaB6 passes.
Sucrose gives Rwp 0.0660539713 in 84.54/85.07 seconds but stops as `stagnated`,
with projected-gradient norm about 0.08467. It **fails the convergence gate**.
Damping reacts to severe backtracking and avoids spending the entire budget
repeating large proposals, but does not manufacture convergence. Diagnostic
coordinate polling improved the objective slightly but did not close the gate;
that experimental fallback is not included in the implementation. Hard-support
transitions remain a solver issue to resolve without silently changing the
finite-support convention or loosening acceptance thresholds.
