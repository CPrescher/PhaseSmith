# Frozen 65-case structure-solution comparison

All 65 previously held-out measured HKL cases were evaluated once using the
unchanged numerical solver at `870d61aaf29ec4220bb176a63f2d7085b361a50e`.
The protocol and attribution were committed as
`bc4c639fe2dfb2448a6141c26f00c9777b19377b` before the holdout was opened.
No solver source, binary, controls, budgets, inputs, atom caps or correctness
thresholds changed during the comparison. No numerical searches were rerun.

## Algorithm attribution and test comparator

Published algorithm sources include G. M. Sheldrick (2015), *Acta Cryst.* A71,
3–8, [doi:10.1107/S2053273314026370](https://doi.org/10.1107/S2053273314026370),
and G. M. Sheldrick (1990), *Acta Cryst.* A46, 467–473,
[doi:10.1107/S0108767390000277](https://doi.org/10.1107/S0108767390000277).
Individual method notes distinguish these published mechanisms from independent
PhaseSmith implementation choices and numerical execution optimizations.
SHELXT 2018/2 is used solely as the separately supplied external test comparator.

## Primary results: all 65 cases

| Outcome | PhaseSmith | External test comparator: SHELXT 2018/2 |
| --- | ---: | ---: |
| Correct full non-H labelled structures | 35/65 (53.85%) | 18/65 (27.69%) |
| Correct total non-H atom counts | 40/65 (61.54%) | 51/65 (78.46%) |
| Correct complete per-element count vectors | 36/65 (55.38%) | 22/65 (33.85%) |
| Strict structures, maximum coordinate error <=0.02 A | 34/65 | 11/65 |
| Returned models | 53/65 | 61/65 |
| No model | 12/65 | 4/65 |
| Hard process timeouts | 0 | 2 |
| Median process wall time, all outcomes | 0.683 s | 0.782 s |
| p95 process wall time, all outcomes | 3.562 s | 2.882 s |
| Summed process wall time, all outcomes | 83.717 s | 93.209 s |

Full correctness retains the predeclared exact element-count and coordinate
requirements: maximum error 0.05 A and RMS error 0.025 A, allowing a common origin
translation and inversion. Hydrogen is excluded. A correct total count alone
does not establish a correct labelled structure. There are no post-solve audit
errors or unassessed cell-setting changes in this collection.

Both programs recover 11 structures, PhaseSmith alone recovers 24, the comparator
alone recovers seven, and neither recovers 23. PhaseSmith's 35/65 recovery rate is
similar to its repeatedly used development result of 19/35; this is new transfer
evidence under the frozen input contract, not proof of general solver parity.

## Speed and interpretation

The median paired PhaseSmith/comparator wall-time ratio over all 65 cases is
0.9721, or about 2.79% lower paired time. Summed process time is 10.18% lower.
The two comparator hard timeouts remain in these primary numbers; the medians
of each program are not themselves a paired speedup. PhaseSmith has the larger
p95 despite the lower median and total time.

On the secondary subset of 11 cases recovered correctly by both programs, the
median paired ratio is 0.8756 (12.44% lower PhaseSmith wall time). This subset
does not replace the all-outcome comparison. Summed PhaseSmith library solve
time is 70.191 s, recorded separately from its process time; it is not divided
by the comparator's unmatched internal timer.

Both programs use one worker/thread, identical input packets and HKLF4 text,
supplied symmetry and element types without atom counts. They retain the original
15-second process ceiling and PhaseSmith's 13-second native safeguard. The jobs
run serially, alternating program order. PhaseSmith is native arm64 and the
external comparator is x86_64 through the host compatibility runtime. These are
single-collection, final-model-at-stop observations rather than time-to-correct-
structure or instruction-set-matched speed claims.

## Resource limits and failures

Twenty-four references exceed PhaseSmith's unchanged input-derived atom cap;
all 24 stay in the primary 65-case denominator. PhaseSmith returns a partial or
incorrect model in 12 of them and no model in the other 12, recovering no exact
total count or full structure among those above-cap cases. The comparator has
21 correct total counts and five correct structures in this subset.

For the secondary, predeclared 41-case within-cap subset, PhaseSmith recovers
35 full structures and 40 correct total counts; the comparator recovers 13 full
structures and 30 correct total counts. Thus the poor aggregate count recovery
is substantially associated with the fixed capacity limit. These limits were
not enlarged after seeing the holdout.

Across all 65 cases, PhaseSmith's remaining model failures comprise 17 wrong
element-count/label vectors and one coordinate mismatch, in addition to the
12 no-model outcomes. The comparator has 39 wrong count/label vectors, four
coordinate mismatches and four no-model outcomes (including its two timeouts).
Low residuals or a successful process exit do not replace reference correctness.

## Freeze evidence and future use

All 130 chosen predictions were saved to an immutable barrier snapshot before
any reference grading. Every audited model matches its pre-reference model
exactly. All 260 copied input-file hashes, worker controls, comparator thread
logs, 402 frozen source-file hashes, the native binary and input registry were
verified. The wrapper performed 133 source/binary checks across the run. The
existing output/reference auditor was used unchanged, including its subsequent
normalization pass; that pass executes no numerical searches and changes no
chosen model.

The immutable freeze is `validation/solution-holdout-freeze-v1.json`. Full
outcomes, summary and verification provenance are committed under
`benchmarks/structure-solution-holdout-frozen-*-macos.json`. Raw inputs, logs,
predictions, source snapshots, schedule and the pre-reference barrier snapshot
remain in `runs/holdout-frozen-v1/`. Reproduction is documented in
`solution-holdout-freeze.md`; a repeat is a repeat on exposed data.

The earlier solver checks passed 429 Rust tests and 2,743 Python tests. The new
freeze/protocol, corpus and prediction-barrier checks passed 33 tests before the
holdout run. No task-owned tests, builds, profiles or kernel benchmarks overlapped
timed collection. Numerical code and the tested release extension are unchanged.

The 65 cases are now **exposed validation data**. Subsequent changes may use them
for diagnosis, but must be validated on a new untouched set before claiming
unseen-data improvement.
