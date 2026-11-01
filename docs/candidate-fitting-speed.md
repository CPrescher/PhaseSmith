# Candidate fitting and unnecessary fallback work

This follows the [matched-prior comparison](matched-prior-search-validation.md).
The same 14 exposed cases, numerical HKL values, Laue/centring priors, element
types, resource limits and grading criteria are retained. No numerical acceptance
criterion or scientific tolerance is relaxed.

## Fitting reuse

The production change preserves the original structure-factor and derivative
arithmetic, reflection/site/image order and scalar fallback:

- An initial value/Jacobian evaluation already made for the search audit is moved
  into a crate-private refinement entry point. It is not evaluated a second time.
  The public refinement API still evaluates and validates its own initial vector;
  failed initial evaluations keep their previous audit semantics.
- For `q = B h`, compute `w = -2 pi² (q.q)` in the same arithmetic order. Canonical
  isotropic displacement is `D = exp(w U)` and its parameter derivative remains
  `w (dU/dp) D`. A preparation plan groups **identical f64 bit patterns** of w,
  without resolution rounding. It is used only when at least half repeat; each
  parameter evaluation computes one D per distinct weight/site. Values and
  derivatives still accumulate together in the original image pass.
- Sites with no free coordinates reuse their constant phase sine/cosine. Two
  otherwise unused coordinate-cache lanes store those values. The image entry
  remains 80 bytes and its 32 MiB admission limit is unchanged.
- Orbit multiplicity is reused only at the exact prepared site position, under
  the same group operations, coordinate basis and image topology. Changed trials
  and mutated public geometry use the original expansion check. An invalid
  prepared multiplicity still fails; validation is not removed.

The displacement scratch table is no larger than the admitted dense Jacobian:
distinct weights <= reflections and canonical isotropic site count <= parameter
count. The immutable plan adds bounded per-reflection metadata; it is shared by
the ordered evaluation tiles rather than rebuilt in every worker.

These are exact reuse changes, not new equations or profile terms. Independent
Python reference equations and finite-difference tests stay unchanged. New Rust
tests compare residual, Jacobian, scale, R1, objective and fit outcomes bitwise,
check adjacent representable reciprocal weights are not merged, and retain
special-position/group mutation failures. GSAS-II's powder oracle does not cover
this single-crystal optimizer.

The paired kernel benchmark uses a ten-atom field with fixed and free sites and
4,168 reciprocal points. It measures both complete value/derivative evaluation
and preparation plus eight fitting iterations. The same benchmark driver is
compiled against the frozen previous source and the updated implementation.

## Deferring redundant ambiguity work

`auto` retains its existing acceptance checks and first tests all of them. It can
defer the thorough pass only when the sole rejection is space-group ambiguity
and **every** BIC-close alternative has the same fitted parameter count, complete
element-labelled atomic model (maximum separation <=1e-8 Å under one origin/hand)
and full normalized common-data residual vector (maximum difference
<=128 machine epsilons, approximately 2.84e-14). This tests numerical equivalence,
not a relaxed fit or position-accuracy criterion. One equivalent pair cannot hide
a distinct competing model. The comparison respects cancellation and deadlines.

The returned candidate remains **unaccepted**, with all alternatives and the
`ambiguous_space_group` rejection preserved. `EffortTrace.deferred_ambiguity`
records the compared settings and maximum discrepancies; the native example
also archives their residual arrays. `selection_reason` is
`deferred_equivalent_group_models`. This is a bounded scheduling decision, not
proof of a unique model or of the correct canonical group. Explicit `thorough`
still performs the full search. Poor fits, missing models, distinct alternatives
and every other failed quality check retain automatic escalation.

## Measurement discipline

A fresh run of the preserved previous executable supplies the paired baseline.
Fitting-only and final stages use newly built, fingerprinted executables. Normal
collection is sequential and does not overlap builds, tests, profiling or kernel
benchmarks. Each stage is frozen before its first model is run, and all chosen
models are saved before reference grading. Every failure remains in the report.

Raw inputs, source snapshots, binaries, complete predictions, profiles and logs
stay outside Git under
`/Users/clemens/ChatGPT/PhaseSmith/candidate-fitting-speed-0.8-20261101/`.
## Measured results

**Candidate fitting improves, while complete search time is effectively unchanged.**
All retained numerical models remain bitwise identical. One redundant automatic
fallback is deferred; the other nine still run. This is a measured, limited gain,
not general SHELXT speed parity.

The paired kernel benchmark gives:

| Workload | Previous | Updated | Observed time reduction |
| --- | ---: | ---: | ---: |
| Full value/Jacobian evaluation | 546.87 µs | 491.68 µs | 10.1% |
| Preparation plus eight fitting iterations | 7.859 ms | 7.428 ms | 5.5% |

Criterion slope intervals (95%, within each 20-sample run) are 545.44–549.05
versus 490.52–493.39 µs, and 7.850–7.873 versus 7.419–7.436 ms. These intervals
characterize these process runs, not hardware/platform or repeated-run uncertainty.
The fitting benchmark includes preparation; the corpus also includes phasing,
atomic-mask/candidate construction, symmetry recovery, assessment and output.

The fresh previous binary, fitting-only revision and final revision give:

| Stage | Fast median s | Auto median s | Sum of 14 auto runs s | Natural escalations |
| --- | ---: | ---: | ---: | ---: |
| Previous | 1.089 | 8.142 | 118.857 | 10 |
| Fitting reuse only | 1.081 | 8.043 | 119.268 | 10 |
| Fitting plus deferral | 1.092 | 8.044 | 118.307 | 9 |

The approximately 0.5% change in total auto time, and 1.2% median change, are too
small to claim an overall speedup from one auto execution per case. Some cases
are slower; the table below retains them. The fitting-only total is slightly
higher despite its faster controlled fitting benchmark. Consequently the kernel
gain must not be presented as a corresponding complete-workflow gain.

There are 112 previous, 56 fitting-only and 112 final sequential corpus executions,
plus eight separate controls. Fast uses three repeats per case, auto one,
SHELXT-four one and SHELXT-ten three. Reported repeated timings first take the
within-case median, then the median over all 14 cases, including the no-model
failure. Every timing includes process startup, assessment and output. The generic
native diagnostic host also opens unused MgSiO3 HDF5 metadata; it does not reduce
frames during this intensity-file comparison. First-launch overhead is retained
in raw receipts. Collection does not overlap builds, tests or profiling.

### Same comparison with SHELXT

| Final execution | Median s | Labelled structure | Published group | Both | Strict labelled positions |
| --- | ---: | ---: | ---: | ---: | ---: |
| PhaseSmith fast, four P1 workers | 1.092 | 11/14 | 11/14 | 10/14 | 10/14 |
| PhaseSmith auto, four P1 workers | 8.044 | 11/14 | 11/14 | 10/14 | 10/14 |
| SHELXT, four workers | 0.458 | 9/14 | 12/14 | 8/14 | 6/14 |
| SHELXT, ten workers | 0.563 | 8/14 | 12/14 | 8/14 | 6/14 |

All recovery counts match the fresh baseline. Primary grading requires exact
non-H element counts, RMS <=0.025 Å and maximum <=0.05 Å under one common
origin/inversion. Strict grading additionally requires maximum <=0.02 Å. The
predeclared position-only secondary count remains 12/14 for native fast/auto,
11/14 for SHELXT-four and 10/14 for SHELXT-ten. All three native fast repeats
are byte-identical where a model exists; the no-model case repeats too.
SHELXT-ten's group/recovery outcomes also repeat unchanged on this run.

SHELXT remains faster overall. Its fresh baseline medians were 0.484/0.488 s
for four/ten workers, versus 0.458/0.563 s in the final replay. This variation is
another reason not to interpret small complete-run timing changes as firm gains.
Both receive the same numerical HKL values, cell, element types, Laue rotations
and centring; exact group, counts and coordinates remain withheld. The 14 cases
are exposed regression data, not a new holdout or a general solver ranking.
Native remains arm64 on an M4 Pro (14 logical CPUs, 48 GiB RAM, macOS 26.6.2);
the unchanged SHELXT 2018/2 executable runs x86_64 under Rosetta. Rust is 1.99.1.

### Retained results and fallback handling

Across all 56 native executions in each new stage, selected setting, full
candidate record, full-cell CIF, common R1 and acceptance flags match the fresh
baseline exactly. Candidate density and shared phase-map arrays also match
bitwise, including shape/dtype. Four accepted cases remain correct in both
labelled structure and published group. Seven other correct atomic models remain
conservatively unaccepted. No acceptance threshold or grading tolerance changed.

COD 2215687 is the one deferred case: #221 and #207 have the same fitted parameter
count, exactly identical complete labelled positions and exactly identical common
normalized residual vectors (both measured maximum differences are zero). Auto
falls from 1.349 s before the fitting change, or 1.386 s at the fitting-only stage,
to 0.242 s final. Its result remains unaccepted and `ambiguous_space_group`.
Other ambiguous/poor-fit/no-model cases still escalate. This does not implement
canonical highest-symmetry determination or establish uniqueness.

| COD | Previous auto s | Fitting-only auto s | Final auto s | Final fast s | Final SHELXT-ten s | Native retained result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2200381 | 15.622 | 16.277 | 16.248 | 1.864 | 1.484 | labelled pass, #221 |
| 2215687 | 1.349 | 1.386 | 0.242 | 0.235 | 0.454 | labelled pass, #221 |
| 2243618 | 18.161 | 18.464 | 18.373 | 1.117 | 0.889 | labelled pass, #194 |
| 2017324 | 0.421 | 0.460 | 0.460 | 0.458 | 0.452 | labelled pass, #194 |
| 1549038 | 12.933 | 12.816 | 12.752 | 0.841 | 0.462 | labelled pass, #4 |
| 2010875 | 20.863 | 20.888 | 20.962 | 2.342 | 0.669 | labelled fail, #4 |
| 2241300 | 0.641 | 0.676 | 0.680 | 0.673 | 0.665 | labelled pass, #56 |
| 2021040 | 3.272 | 3.281 | 3.258 | 3.254 | 0.877 | labelled pass, #33 |
| 2207083 | 2.310 | 2.248 | 2.250 | 0.237 | 0.242 | labelled pass, #139 |
| 2233659 | 8.391 | 8.391 | 8.297 | 1.697 | 0.454 | labelled pass, #121 |
| 2011241 | 7.893 | 7.695 | 7.791 | 1.491 | 0.451 | labelled fail, #1 |
| 2010993 | 0.371 | 0.351 | 0.340 | 0.342 | 0.237 | labelled pass, #2 |
| 2203173 | 17.338 | 17.113 | 17.273 | 5.230 | 0.664 | labelled pass, #167 |
| 2201295 | 9.292 | 9.221 | 9.385 | 1.068 | 0.889 | no model |

### MgSiO3 and positive fallback controls

Six normal MgSiO3 auto controls, with/without the optional mmm/P prior, all accept
fast Pbca with Mg16 Si16 O48, R1 0.0514043045582282. CIF, candidate-density and
shared-map arrays are bitwise identical to the previous controls. Warm process
times are 0.503–0.569 s; the new binary path's first launch is 0.843 s. This does
not measure a new raw-frame reduction or establish a MgSiO3 speed gain.

A 1-microsecond fast-pass budget still produces `no_model`, triggers the full
four-start, 800-cycle resolution search, and returns accepted, correctly labelled
structures for MgSiO3/Pbca in 19.778 s and acentric COD 2021040/Pna21 in 26.798 s.
Both forced controls' retained CIF, density and phase map are bitwise identical to
the previous accepted fallback controls. These observed times are slower than the
previous 18.803/25.505 s; the change does not claim a deep-search speedup.

### Remaining time and optimization targets

Sum over all 14 auto executions:

| Area | Previous s | Final s |
| --- | ---: | ---: |
| Shared initial phase maps | 3.864 | 3.843 |
| Origin/phase screening | 6.944 | 6.962 |
| Model searches, including independent rephasing, construction and fitting | 107.179 | 106.520 |
| Search coordinator after initial assessment | 118.025 | 117.361 |
| Full process wall time | 118.857 | 118.307 |

The coordinator includes the first three areas plus overhead; it is not another
additive stage. Model-search time remains about 90% of wall time, but includes
independent rephasing and atomic-mask/model construction. It is not an isolated
fitting timer. Diagnostic samples identify those areas as substantial work; they
were collected separately from normal timings. The next useful optimization is
to separate and reduce that work, especially repeated independent phasing/mask
construction after unsuccessful shared-map candidates. More fitting microkernel
work alone is unlikely to move complete time much. Further fallback reductions
need evidence that the remaining work is redundant, preserving all rejection
checks and successful forced-fallback controls.

## Validation and receipts

Strict Clippy (workspace/all targets/all features), formatting and the full Rust
workspace pass: 552 tests, plus four native-host tests. The installed new Rust
kernel passes 3,318 Python tests (11 external-data skips, 34 default deselections),
including independent references and finite differences. A release-mode exactness
test passes too. The final wheel's runtime files are identical to the tested
wheel; only generated SBOM/RECORD metadata differ. Rust 1.89 all-feature facade
and minimal-facade checks pass. Ruff remains clean; no runtime dependency is added.

The existing 32 MiB/80-byte-image geometry admission remains unchanged. Common
residual vectors are retained as bounded report metadata, outside the dense
Jacobian budget: at most two rows per source observation for each bounded model
attempt. The deferral example saves only the compared vectors in its NPZ receipt.

The versioned [policy](https://github.com/CPrescher/PhaseSmith/blob/main/validation/candidate-fitting-speed-policy-v1.json) was
frozen before final collection. The compact
[receipt](https://github.com/CPrescher/PhaseSmith/blob/main/validation/results/candidate-fitting-speed-v1-macos.json) records
stage timings, grading, bitwise checks, source/executable hashes and controls.
The original matched-prior receipts remain intact. Source snapshots, benchmark
outputs/samples, executable copies, complete predictions, barriers, wheels,
profiles, reference assessments and logs stay at the external artifact root above.
Temporary task-owned compilation caches and the extracted baseline source copy
are removed after preserving the benchmark driver, frozen source archives and
executables. No source datasets or prior receipts are removed.

## Remaining-case extra-effort diagnostic

After the implementation was committed as `797ecfc`, a separately frozen
[diagnostic plan](https://github.com/CPrescher/PhaseSmith/blob/main/validation/hard-case-diagnostic-policy-v1.json) tested the
three remaining labelled-structure failures. The same executable, numerical
observations, cell, element list and Laue/centring priors were retained. Exact
space group, element counts and reference coordinates were withheld. Direct
thorough search used seeds 1 and 2, eight starts capped at 1,600 cycles, eight P1
workers and 30 seconds per candidate, within the existing 180-second total and
210-second process caps. The acceptance and grading thresholds stayed unchanged.
Every planned output was saved before reference grading; all six runs are
reported, with no reference-based selection between seeds.

| COD | Seed | Process s | Chosen group | Own R1 | Position-only criterion | Labelled criterion | Accepted |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2010875, Ni/O/N/C | 1 | 37.236 | 4 | 0.09292 | pass | fail | no |
| 2010875, Ni/O/N/C | 2 | 38.957 | 4 | 0.09328 | pass | fail | no |
| 2011241, Cr/O/N/C | 1 | 10.111 | 1 | 0.07812 | fail | fail | no |
| 2011241, Cr/O/N/C | 2 | 9.151 | 1 | 0.07835 | fail | fail | no |
| 2201295, Li/O/N/C | 1 | 16.802 | 152 | 0.08720 | pass | fail | no |
| 2201295, Li/O/N/C | 2 | 17.188 | 152 | 0.08720 | pass | fail | no |

The six executions total 129.44 s with no hard timeout. The lithium case improves
from no retained model to the correct group and complete 66-site atomic framework
in both runs (RMS 0.01454 Å, maximum 0.02216 Å after common origin/inversion).
Its assignment is Li6 O30 N12 C18, versus the reference Li6 O30 N6 C24: six C
sites are labelled N. This is partial recovery, not a fully labelled or accepted
solution. The nickel framework also passes position-only grading, while C/N/O
labels fail. Chromium still misses coordinates and labels. All three have the
correct total non-H atom count in these diagnostic runs; simply raising the atom
cap is therefore not the indicated next fix. These additional-effort results do
not replace the frozen 14-case comparison or establish new holdout performance.

The saved diagnostics and owning code identify concrete follow-up work:

1. **Joint element/ADP correction for promising frameworks.** The unknown-group
   coordinator currently forces isotropic fitting; known-group search already
   permits anisotropic polishing. Test a bounded final anisotropic polish and
   repeated coupled light-element assignment before spending more time on
   rephasing. Recompute the common-data score and actual parameter count, preserve
   independent-observation/overfitting guards, and keep every acceptance threshold.
   This is a proposed experiment, not proof that anisotropy alone fixes the labels.
2. **Centred-lattice phasing compatibility.** `SymmetryPrior::processing_group`
   includes the supplied centring translations, while `validate_phase_input`
   rejects every nonzero translation. The saved I/R-centred cases explicitly record
   that error and use independent per-group searches. This integration mismatch
   needs a tested centring-aware contract; screw/glide assumptions must remain
   withheld and missing intensities must never become fabricated zeros.
3. **Recovery and stopping diagnostics.** Some lithium candidates are still
   rejected by the second origin/hand recovery during common scoring. Preserve
   those rejected models and their transformations to distinguish genuinely bad
   candidates from a recovery-path problem. Separately, model-consensus stopping
   permits R1 <=0.1 despite the requested own target of 0.06; nickel seed 1 stops
   after five starts while its target remains unmet. Test target-aware stopping
   for explicit thorough effort. Chromium also needs better coordinate recovery;
   relabelling alone is insufficient for its present model.

An optional independently supplied composition/formula could constrain light-element
counts in real experiments. It would be an explicit additional prior and a
separate benchmark protocol; these reference counts were never solver inputs.
Any implementation should replay this exposed regression set and then undergo
fresh validation after development is frozen.

The compact [diagnostic receipt](https://github.com/CPrescher/PhaseSmith/blob/main/validation/results/hard-case-diagnostic-v1-macos.json)
retains every run, count comparison, geometric error, failed attempt and source
identity. Full inputs, executable, predictions, barrier, runner and grader stay
outside Git at
`/Users/clemens/ChatGPT/PhaseSmith/hard-case-diagnostics-0.8-20261101/`.
