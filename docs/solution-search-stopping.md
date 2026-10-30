# Phasing and assignment stopping

The preceding best is `4c47491` (blocked normal matrices). This experiment changes
work allocation while retaining equations, source observations, precision,
physical support, derivative evaluation and correctness gates. Both new controls
are false by default. Neither uses reference counts, coordinates or case IDs.

## What the SHELXT listings establish

`benchmarks/solution_shelxt_effort.py` reads only the 35 development-case listings
from the completed comparison. It records every available trial row, declared
attempt count, N(iter), Fourier grid and printed internal stage/total time, with
raw listing SHA-256. The remaining 65 holdout outcomes are not opened. A missing
or truncated total remains missing; a timeout is never assigned zero time.
N(iter) is listed iteration work, not an independently observed executed-cycle
counter. We do not infer exact iterations from time or silently impute missing
trials. The parser is tested on complete and truncated listings.

On the 34 cases with finished internal timers, the preceding run has median
PhaseSmith solver time 1.11595 s and SHELXT internal time 0.88750 s, and median
paired solver ratio 1.18378. The complete 35-case listing collection contains
132 printed trial rows and 21,013 listed iterations, versus 95 PhaseSmith starts
and 39,432 completed/scored phase cycles. Within the same 34 completed pairs,
SHELXT lists 108 trials and 12,106 iterations versus 37,432 PhaseSmith cycles.
The timeout listing contains 24 trial rows but no final attempts/total summary;
its 8,907 listed iterations are partial work, not a completed solve.
The comparison is qualified: library
solve time and SHELXT's millisecond internal elapsed total have different timing
boundaries, reflection limits and grids; PhaseSmith finishes anisotropically,
SHELXT isotropically. All 35 outcomes remain in recovery/wall-time denominators.
`benchmarks/structure-solution-stopping-diagnosis-macos.json` records every case.

The executable's own option text lists initial phasing work of 100 iterations,
increasing trial work and a trial-dependent quality acceptance rule. The observed
trial budgets and outcomes do not reveal its entire internal stopping logic.
The published density/phase-quality/isotropic-assignment sequence is described
by Sheldrick (2015), doi:10.1107/S2053273314026370. The rules below are independent
bounded policies, not a translation of SHELXT implementation code.

## Early phase plateau

`phase_plateau_stopping=True` requires work-sliced phase-first execution and
excludes adaptive checkpoints. It applies only to the first floor(T/2) starts;
remaining starts retain their original full quota. The existing short-start
quotas are unchanged (200,200,800,800 for C=800,T=4). Single-start searches retain
full work. Existing target, cancellation, deadline and model checks remain.

Let q_j be the existing measured-only CFOM at zero-based cycle j, and B(n) its
maximum among the first n eligible, non-omit cycles. With random omission active,
every third cycle is excluded exactly as in best-map selection. A phase start
hands off its existing best map when all of the following hold:

    n >= 128, n modulo 32 = 0
    all q_j are finite
    B(n) >= 0.8
    B(n) - B(n-32) <= 0.002.

The last cycle's map need not be good: the handoff uses the existing best map,
never the latest deteriorated/omitted iterate. No phase score is a correctness
certificate. The complete origin/symmetry/count/assignment/final-fit checks still
run, and independent later starts remain available on failed acceptance. This
policy does not resume the stopped early trajectory at full length. Stopped
maps or endpoints can differ, so all development counts/recovery must be checked.
Actual `phase_slices.stop_reason` is `phase_plateau`; quotas and completed cycles
are exported separately. The unchanged high-CFOM target takes precedence.

## Converged assignment screen

`assignment_convergence_stopping=True` requires isotropic assignment refinement.
Only short Screen/ScreenSeed fits use it. Requested final isotropic/anisotropic
fits retain their established convergence criteria and budgets. Label proposal,
count selection, ambiguity exploration, promotion ranking and geometric checks
are unchanged.

For two consecutive accepted objective-decreasing steps, require

    objective_old - objective_new <= 1e-6 * max(objective_old, 1e-30)
    abs(R1_new - R1_old) <= 1e-6.

The fit may stop only after at least four attempted iterations and if unspent
iterations remain. A rejected or materially improving step resets the streak.
Damping, step residual guards, value/derivative evaluations and backtracking are
unchanged. Termination is `assignment_converged`, and the aggregate
`assignment_convergence_stops` counter records actual short-fit exits. Floating
point model values/rankings may change; bitwise equality is not an assumption.

## Validation and development protocol

The pure Python plateau reference has no native calls. Randomized deterministic
streams, omitted spikes, thresholds, warm-up, partial intervals and nonfinite
inputs are compared with Rust. Full three-symmetry synthetic element-only
searches check exact counts, geometry, full fallback quotas and identical phase
prefixes. Multiatom screen fits check independent NumPy physical values and
centered derivatives and repeat deterministically. Existing support/moment and
analytical-derivative tests remain applicable; no physical term changes.
The GSAS-II powder oracle does not cover this scheduling policy.

`solution_stopping_interleaved.py` collects four modes: preceding best control,
phase stopping alone, assignment stopping alone, and both. Mode position rotates
by case (9/9/9/8 leading); all 280 predictions precede the four reference-audit
barriers. Each mode includes all 35 development cases and seven references above
the native resource cap, seed zero, four starts, 800 cycles, native13/outer15-second
limits, supplied symmetry and requested anisotropic refinement. Adapter-only
revision32 inherits31 without changing the registry/split or reopening source
structures. No task-owned tests, builds or kernels overlap timed collection.

`solution_stopping_verify.py` verifies source/native hashes, all predictions and
quotas, compares the exact prior 19-success and count-success sets, and records
every inferred count vector for every mode. No failed experiment is hidden.
Criterion `stopping_search_` covers full eight-atom Fe/C/N/O searches on a
32-cubed grid with 2,108 rows, not a scalar stopping-predicate call.

## Complete development results

All four modes retain the exact previous 19/35 successful structures, 26/35
correct total non-H counts, 19/35 correct per-element counts and 15/35 strict
0.02-Angstrom structures. Every inferred count vector agrees across all 35 cases
and with the previous best, including incorrect/no-model outcomes. There are
four no-model outcomes and no PhaseSmith hard timeout. The holdout remains closed.

| Mode | Summed library solve | Summed wall time | Solver saving | Wall saving |
| --- | ---: | ---: | ---: | ---: |
| Previous stopping rules | 52.31485 s | 60.41926 s | — | — |
| Phase plateau only | 51.65967 s | 59.69712 s | 1.25% | 1.20% |
| Assignment convergence only | 50.73319 s | 58.83596 s | 3.02% | 2.62% |
| Both | 49.47749 s | 57.54339 s | 5.42% | 4.76% |

Both controls together give 4.94% less typical paired library solve time and
4.41% less typical paired wall time, with 29/35 wall-time cases faster. These
are paired medians and cumulative times from one balanced collection, not a
repeated uncertainty estimate. Gains are not additive: early maps can change
subsequent work, and host timing varies. Separate aggregate wall medians are
not used to define typical speed.

The combined mode records 22 plateau handoffs and 367 converged assignment
screens. Completed phase cycles fall 39,246-to-37,621 (4.14% less); the Fourier
workspace attempt counters are each one larger because a deadline can interrupt
a cycle before it is scored. Screening time falls 10.90896-to-9.35338 s (14.26%
less), and P1 time 25.08399-to-23.98111 s (4.40% less). Starts remain 95; screens
are 667 versus 671 and final fits 125 versus 124. Ambiguity relaxations are 250
versus 248. Assignment cached linear steps fall 4,591-to-3,835. Every stopping
count, quota, remaining full start and inferred count vector is in the reports.
Final refinement criteria are unchanged; differing selected maps can change the
number and endpoint of final fits.

For the same 34 completed SHELXT timing pairs in this new campaign, its listings
consistently report 108 trials and 12,106 listed iterations. PhaseSmith performs
91 starts and 37,246 completed cycles in control, versus 35,621 with both rules.
The median paired PhaseSmith library-time/SHELXT internal-time ratio changes
1.22390-to-1.16497. The one SHELXT timeout remains in all 35-case wall/recovery
results and is explicitly unavailable for completed internal timing; its partial
trial listing is retained separately. The difference in remaining cycle work
and isotropic/anisotropic endpoints still prevents a claim of equal workloads
or general solver parity.

The final isolated Criterion medians show no useful gain on the small synthetic
fixtures: the warm-started 12-step fit is 8.130-to-8.157 ms (+0.33%), with both
finishing their budgets, and the full synthetic search is 1.10404-to-1.11035 s
(+0.57%), with unchanged printed work. Phase-only and assignment-only searches
are also about 0.6% slower there. The real-development gain comes from work that
actually meets the stopping conditions. Earlier kernel measurements overlapping
smoke work are not used. Raw samples, exact benchmark executable/source hashes
and logs are retained in `structure-solution-stopping-kernels-macos.json`.

A collector bookkeeping error was found after all predictions and audits had
finished: `collect_balanced` returns case/method rounds (70), whereas the new
four-mode final assertion incorrectly expected 140. The check is corrected to
70 and has a synthetic complete-run regression test covering 280 predictions
and four barriers. The original timed source is retained in every source
snapshot. The retention verifier checks the sole textual difference exactly,
records both hashes and verifies all other numerical/binding/build-input source
hashes and the actual native library. No numerical code or timed prediction was
changed to repair this reporting check. All 280 predictions precede the four
reference barriers, verified directly from the 284-event schedule.

Evidence is recorded under
`benchmarks/structure-solution-stopping-{control,phase,assignment,cache,comparison,retention,kernels,diagnosis}-macos.json`
and `structure-solution-stopping-shelxt-{control,phase,assignment,cache}-macos.json`.
The retention verifier chooses the lowest cumulative solver time only among
variants retaining every previous structure/count success set; both controls
are the selected development variant. Both remain opt-in.

Validation: workspace formatting and Clippy with all targets/features and
warnings denied; 421 Rust tests passed, 34 ignored; 2,589 Python tests passed,
11 skipped, 34 deselected. Both experimental solution and spot modules were
built for the Python run. The post-collection reporting fix and parser are also
tested. These are development results; no held-out generalization claim is made.
