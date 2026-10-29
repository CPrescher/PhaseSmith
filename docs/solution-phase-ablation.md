# Normalization and phase-selection ablation

This development follow-up isolates the two experimental options introduced in
[the phase-quality experiment](solution-phase-quality.md). The previous complete
results and their source/native hashes are retained. The 65 holdout outcomes
remain unopened. No reference formula or coordinates enter a numerical worker.

## Diagnosis and method

The return to the decompilation found a substantive mismatch before changing
scheduling. In the existing external SHELXT P1 export, calculated complex-map
coefficients produce an amplitude workspace (around lines 1408–1424). That
workspace is passed to the normalization routine, whose separate output is
squared in the weak-observation statistic (around lines 2012–2038). The
normalization routine itself estimates resolution-bin second moments. This is
consistent with the published statement that Gc is normalized to Ec; it is
**not** merely conversion using the expectation fitted to observed intensities.
The static review and source hashes are retained outside the repository in
`/Users/clemens/Codex/shelxt-binary-audit-2026-10-28/followup-2026-10-29/normalization-dataflow.json`.
The global MAIN__ remains unresolved; this local data flow does not establish
every input convention or the entire SHELXT pipeline.

The preceding PhaseSmith statistic used c^2/sqrt(observed expected intensity)
for Wilson coefficients, followed by one global rescale. A calculated map with
a different radial envelope therefore received a different weak score, even
if its within-shell weak/strong contrast was unchanged. Atomic masking and map
width can change that envelope. The new independent shell normalization below
removes this confounding factor. It is derived from conditional second moments,
not a translation of the inspected implementation; its disjoint shell scheme
does not claim to reproduce SHELXT's interpolation or smooth limiting.

The previous quality option also changed the order of work: it collected phase
starts before any supplied-symmetry/count search and used an absolute Q>=0.90
threshold for early stopping. These are PhaseSmith scheduling choices, not a
consequence of the published CC/Rweak equations. A correct easy structure in the
development panel can have maximum P1 Q only about 0.55, so this threshold forces
unnecessary start exploration. Conversely a high P1 score does not establish
correct element assignments after supplied-symmetry recycling.

The original combined run sends COD 2243618's selected late phase map to count
search and returns 38 instead of 30 atoms. Its control's early amplitude map
recovers the correct structure. Across the seven common recoveries the original
phase-first schedule increases median runtime from 1.776 to 10.320 seconds.

Normalization-only changes neither scheduling nor the amplitude-SSE map ranking.
Quality-only retains the original phase-first schedule in the frozen parent
build. These full-panel experiments separate the original feature options;
the latter still combines a quality statistic with a work schedule. It is not
an isolated test of the statistic at fixed work. The published procedure's P1
score and subsequent symmetry/density/assignment stages are distinct
([Sheldrick 2015, sections 2–4](https://journals.iucr.org/a/issues/2015/01/00/sc5086/index.html)).

An additional tested option uses a checkpoint portfolio. In each completed Fourier
window, let S_i be the existing profiled amplitude SSE and Q_i=CC_i-Rweak_i.
Exclude every third omitted-mask map from both candidate sets, as before. Keep

\[
 p=\arg\min_i S_i,\qquad q=\arg\max_i Q_i,
\]

with the first cycle winning exact ties. At the original 4, 16, 64, ... cycle
checkpoints, assess p first with the existing supplied-symmetry/count search.
If it is not accepted, q differs from p, and time remains, assess q as an
alternative. Earlier model winners and their density frames remain candidates.
All acceptance and final model comparisons use the original observations,
existing complexity ranking and requested refinement. There is no new absolute
Q gate and no requirement to finish all phase starts before the first fit.
Quality can add a proposal but cannot discard the amplitude winner on its own.

`phase_quality_selection=True` retains the phase-first order by default, with
corrected calculated normalization. `phase_first_selection=False` selects the
checkpoint portfolio; it is ignored when quality selection is off. With
checkpoints disabled, quality selection still uses phase-first mode. Both
normalization and quality feature options themselves remain opt-in.
`alternative_handoffs` records actual extra quality proposals; existing
checkpoint records retain each map's completed/best cycles and amplitude score.
The old `deferred_handoffs` field remains for loading earlier records.

## Independent normalization of calculated amplitudes

Observed Wilson normalization, signed-source treatment, CC and weak-set
membership remain unchanged. For the calculated P1 map, sort the comparison
families by the same stable resolution/HKL convention and partition into
floor(M/80) equal-population shells, bounded to 1–20. The population includes
observed absence families. No composition or supplied-symmetry epsilon factor
is imposed on these unsymmetrized P1 calculated amplitudes.

For shell s with n_s rows and raw calculated magnitudes c_i, use

\[
 D_s={1\over n_s}\sum_{i\in s}c_i^2,\quad
 A_s=\sum_{i\in s,weak}c_i^2,\quad
 R_{weak}={1\over W}\sum_s {A_s\over D_s}.
\]

Each positive calculated shell has mean Ec-squared equal to one. A completely
zero shell contributes zero with zero derivative by an explicit convention;
the all-zero map is rejected. This boundary is not differentiated through.
The weak score is invariant to an independent positive amplitude multiplier
in each shell. CC retains its original global amplitude comparison.

Values and analytical derivatives are evaluated together. For i in shell s,

\[
 {\partial R_{weak}\over\partial c_i}=
 {2c_i\over W D_s}\left(1_{i\in weak}-{A_s\over n_sD_s}\right).
\]

`calculated_shell_normalization=False` retains the preceding observed-envelope
conversion and global normalization. It is an explicit numerical reference mode.
The new choice is recorded in the search audit and the number of calculated
shells is recorded per quality trace. The same native build tests the corrected
statistic with both the old phase-first schedule and the checkpoint portfolio,
so a score correction is not inferred from a scheduling change alone.

## What the inspection does and does not reproduce

Having routines with the same method names is insufficient: their input scales,
normalization populations and handoff rules must agree as well. The previous
implementation did not establish this correspondence. This review checks a
specific data-flow mismatch instead of treating the decompilation as a recovered
source-level specification.

The installed executable's saved help and the inspected exports also identify
remaining concrete differences. They are not silently changed in this experiment:

| Operation | External evidence | PhaseSmith in this comparison |
| --- | --- | --- |
| Calculated amplitudes for Rweak | Independently normalized in the P1 routine | Corrected here; earlier observed-envelope mode retained |
| Empirical normalization | Interpolated second-moment bins and a smooth limiting operation | Independent Wilson/shell fit for observations; disjoint second-moment shells for calculated maps |
| Phase-set effort/acceptance | Help specifies an initial 100 cycles and a try-dependent CFOM acceptance threshold | Original phase-first mode has a 32-cycle minimum and fixed 0.90 gate; checkpoint mode uses downstream fits |
| Random omission | Installed default fraction 0.3 every third cycle | Fixed fraction 0.1 every third cycle |
| P1 peak search | Installed defaults include sigma 2.5 and an atomic-volume control | Sigma 2.0 and a bounded adaptive peak cap |
| Resolution extension | Predicted coefficients extend the input sphere | No extension beyond the input sphere |
| Element-density scale | Published C/bond, B-cage and oxyanion tests precede the heavy-peak fallback | Density calibration and bounded orbit/label fits; no equivalent chemistry-scale hierarchy |

The executable help is recorded at
`/Users/clemens/.local/opt/shelx/versions/2026-10-28/checks/shelxt.txt`.
The first correction is supported by local call/data flow and the publication;
the others are explicit follow-up hypotheses, not claims of recovered equations
or measured benefits. Address-ordered calls in the unresolved MAIN__ region are
not interpreted as execution order. A complete semantic reconstruction and
stage-level numerical equivalence with SHELXT have not been established.

## Protocol and results

The exact 35 development cases, quantized observations, input-derived atom caps,
known symmetry/element-only inputs, seed 0, four starts, 800-cycle ceiling,
13-second internal/15-second outer budgets, one-thread execution and original
atom-count/geometry correctness gates are preserved. All predictions in a panel
are saved before any reference audit. Frozen-parent normalization-only and
quality-only campaigns run serially; implementation changes in the active
checkout cannot change their copied native library or Python sources.

The complete frozen-parent ablation is:

| Mode | Correct structures | Correct total counts | Exact element counts | Median all-outcome time (s) |
| --- | ---: | ---: | ---: | ---: |
| Neither option (prior control) | 8 | 13 | 8 | 13.331 |
| Observed normalization only | 8 | 14 | 8 | 13.217 |
| Original quality/schedule only | 7 | 12 | 7 | 13.272 |
| Original combination | 7 | 12 | 7 | 13.374 |

The single-option runs use the same copied native/Python parent as the original
control/combination. Normalization alone gains COD 2011241 but loses COD 2011451.
Original quality alone loses COD 4326074; the combination instead loses COD
2243618. Equal aggregate numbers therefore do not establish identical behavior.
This isolates the main aggregate regression to the original quality/scheduling
option rather than observed normalization alone. It does not prove the observed
normalization is uniformly beneficial.

Here the normalization flag controls the amplitudes fed to phasing. Quality-only
still prepares observed E values to define its weak-reflection set, as required
by that statistic; it leaves the phasing coefficients on the legacy transform.
The report therefore does not describe quality-only as using no normalization
anywhere in the algorithm.

The complete corrected-score run with the **old phase-first schedule unchanged**
recovers 9/35 structures, all at the strict 0.02 angstrom gate. It retains all
eight original control successes and gains COD 2010705. Exact total counts are
correct in 11/35, so the improved structure recovery is not a uniform improvement
in count prediction. The original seven common successes have median runtime
8.803 seconds rather than 10.320 seconds; the all-outcome median is 13.321 seconds.

The same-build corrected-score/checkpoint run recovers only 7/35 structures and
12/35 total counts (median 13.306 seconds). It loses COD 2010705, 2243618 and
2011451 relative to the corrected phase-first run and gains COD 2011241. It
executes 39 alternative-map handoffs and reduces completed P1 cycles from
43,351 to 6,626. Less Fourier work does not compensate for the changed candidate
search and assignments. This proposed checkpoint schedule is therefore **not
selected as the default**. The retained original phase-first order with the
calculated-normalization correction is the preferred experimental quality mode.

The final reference collection uses **the same native executable**, phase-first
schedule and normalized phasing coefficients, restoring only the earlier weak
statistic. It recovers 7/35 structures and 11/35 total counts, median 13.399
seconds. Corrected calculated normalization recovers 9/35 structures and 11/35
total counts, median 13.321 seconds. It gains COD 2010705 and 2243618 with no lost
recoveries. Across seven common successes the median decreases from 10.932 to
8.803 seconds (1.24 times faster); this is still one serial collection per mode,
not a repeated timing estimate. The earlier 12-versus-current-11 total-count
reference result illustrates deadline sensitivity and is retained, not replaced.

The new score improves complete structure recovery, while correct total counts
remain substantially below SHELXT's 29/35 and even the observed-normalization-only
result of 14/35. SHELXT recovers 13/35 fully labelled structures under the same
primary gate. There is no claim of parity or a universal improvement. Seven
references remain larger than the unchanged native input-derived model cap.

Retained full reports and the source/hash-checked comparison:

- [Observed normalization only](../benchmarks/structure-solution-phase-normalization-only-macos.json)
- [Original quality only](../benchmarks/structure-solution-phase-quality-only-macos.json)
- [Same-build old score](../benchmarks/structure-solution-phase-legacy-score-control-macos.json)
- [Corrected score, phase-first](../benchmarks/structure-solution-phase-repaired-score-macos.json)
- [Corrected score, checkpoint alternative](../benchmarks/structure-solution-phase-repaired-checkpoint-macos.json)
- [All comparisons and per-case results](../benchmarks/structure-solution-phase-ablation-macos.json)

After these complete explicit-mode comparisons, the public default is set to
retain phase-first ordering when quality is requested. The checkpoint alternative
stays available rather than being silently substituted for the stronger result.
Timed source snapshots/native hashes remain unchanged; the post-collection API
default choice does not change either explicitly tested numerical mode.

## Validation and reproduction

An independent NumPy window-selector reference checks candidate ordering,
omission and ties. Native tests compare the first checkpoint maps and complete
Fourier histories with and without quality, including a low-quality candidate
that must not be gated. Synthetic count/geometry cases cover both schedule modes.
The independent quality-gradient reference now includes calculated-shell
coupling. Native/NumPy values and centred finite differences check both the
corrected score and its earlier observed-envelope reference mode. Per-shell
amplitude-scale invariance is an additional targeted regression test.
The GSAS-II powder oracle does not cover this structure-search orchestration.

Validation passes: 397 Rust tests (34 ignored), 1,532 Python tests (11 skipped,
34 deselected under the existing external-data policy), Clippy with warnings
denied, formatting, Ruff and whitespace checks. The 31 focused numerical/API
cases cover both score modes, both schedules, independent values and finite
differences, per-shell amplitude-scale invariance, zero-shell boundaries,
unchanged observed normalization and synthetic count recovery. Four report
guard tests reject incomplete, opened-holdout and changed-budget collections
and exercise explicitly paired report labels.

The realistic eight-atom/2,108-reflection Criterion fixture measures corrected
quality values plus gradients at 17.33 microseconds and observed normalization
at 1.028 milliseconds (ten samples). The extra calculated-shell work is more
expensive than the earlier 7.46-microsecond statistic. Unchanged normalization
code also measures slower than the previous session, so cross-session ratios
include host/build variation. There is no claim of a kernel speedup.

With the same 16-cycle and one-round limits, the full schedule benchmark takes
451.5 milliseconds for phase-first and 880.0 milliseconds for checkpoints
(95% estimate intervals 438.8–464.7 and 828.4–956.9 milliseconds). Different
handoff/model work is intentionally included; these are not isolated FFT times.
[Raw samples and exact fixture/source provenance](../benchmarks/structure-solution-phase-ablation-kernels-macos.json)
are retained. The full-panel coverage/count outcomes take precedence over this
small fixture's timing result.

The corpus adapter lock is advanced separately from dataset curation:

```sh
.venv/bin/python benchmarks/solution_corpus_benchmark.py seal-adapters \
  --manifest validation/solution-corpus-v5.json \
  --parent-lock validation/solution-corpus-v5.benchmark8.lock.json \
  --lock-revision 9
```

This explicit adapter-only operation requires the identical registry hash,
records the immutable parent lock, and does not reopen raw/reference data.
Omit either feature flag for separate ablations. Add `--quality-checkpoints`
for the non-default schedule and `--legacy-quality-normalization` for the earlier
weak statistic. `--phase-first-selection` explicitly records the default schedule.
