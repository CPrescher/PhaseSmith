# Isotropic competition before selective final refinement

The experimental `isotropic_assignment_refinement=False` default retains the
existing path. The true option requires phase-origin selection and chemical
assignment. It changes bounded candidate relaxation and promotion decisions;
the requested final ADP mode, observation arrays, count/geometry scoring gates,
atom/Jacobian limits and global/local time backstops remain unchanged.

## Independent algorithm and conventions

The published SHELXT workflow assigns elements from integrated density and uses
isotropic conjugate-gradient refinement (Sheldrick 2015, sections 4–5,
https://journals.iucr.org/a/issues/2015/01/00/sc5086/). PhaseSmith already implements
that density guidance, bounded coupled label moves and iterative normal equations.
The following competition/promotion policy is a new independent design, not a
recovered SHELXT implementation.

After the original short isotropic screens and improving chemical moves, take
the best two candidates with R1 <= 0.3. Relax coordinates, individual isotropic
displacements and the profiled amplitude scale for up to 12 further iterations
using the existing deterministic subset of at most 2,048 independent observed
families. Keep both successful parent fits. The objective and analytical
derivatives are the existing fused equations:

    F_h = sum_j f(label_j,s_h) exp(-8*pi^2*U_j*s_h^2) P_j(h),
    k = sum_h Fo_h*|F_h| / sum_h |F_h|^2,
    J = sum_h (k*|F_h|-Fo_h)^2 / sum_h Fo_h^2.

The orbit phase P and all three coordinate jets are calculated together. The
existing constrained isotropic refiner also fits U with its analytical derivative.
Scale is profiled anew at every evaluation. Only the original measured amplitude
model is used for these fits; predicted missing coefficients do not become data.
The same family representatives are used throughout candidate competition. Final
requested-mode fits still use all original prepared observations, including all
rows used by the control.

For each relaxed parent, rebuild the existing bounded independent-family cache
at its updated coordinates/U. Enumerate single-element changes and paired swaps
among at most 24 sites prioritized by their best single-change objective. No
required input type may disappear. Counts/topology are fixed within each such
proposal, while the parent's original count alternatives still compete. Retain
the two lowest-ranked distinct alternatives satisfying

    J_proposal <= 1.25*J_parent + 1e-4.

This is an explicit empirical proposal band, not a statistical confidence bound.
It admits a modest fixed-coordinate worsening that isotropic relaxation could
reverse. Homometric alternatives can remain tied; they are not claimed identified.
Candidates sort by descriptive rank then label vector for deterministic ties.

Select at most two neighbors across the parents using
`(J_proposal+1e-4)/(J_parent+1e-4)` on that same bounded cache. Already-retained
identical labels/coordinates are skipped. Give those neighbors up to eight
isotropic iterations on that same independent-family subset, retaining parents
as well as neighbors. This leaves at most four comparable candidates. Their rank
is the existing `N_family*log(max(J,1e-300))+(p+1)*log(N_family)`, with actual
symmetry-constrained isotropic parameter count p. Every parent and neighbor rank
uses the same families and isotropic parameter convention; requested-mode final
ranks remain separate. If no parent fit succeeds, preserve the original screen
path and original promotion policy.

Promote the best eligible candidate to the originally requested final ADP mode.
If it meets the existing complete-model/R1 target, no runner-up fit is needed.
Otherwise, a runner-up within six rank units of the best retained isotropic
candidate is also promoted; the gap uses the retained ranks before the R1
eligibility filter. If the first fit fails the runner-up remains available
regardless of the gap. A six-unit gap is a bounded selection heuristic, not calibrated model
odds. Larger-gap alternatives can still arise from independent phase starts.
Only fits in the requested final mode enter the final winner beam and ranking.
This can miss a candidate rescued only by anisotropic motion; the full paired
development experiment must measure that risk.

All iteration equations, f64 precision, support conventions, symmetry, scattering,
required-element coverage, original seed schedules and native resource ceilings
remain unchanged. The option does not introduce formulas or expected counts.
Per-map audits record parent/neighbor fits, cheap neighbor scores, retained ranks
and deferred promotion slots. These are attempted work, not counts of correct or
accepted models. Deferred slots refer to candidates eligible under the new path,
not a direct subtraction from the control's final-fit count. Timers include
rejected work.

## Validation and declared comparison

An independent NumPy enumeration checks fixed-site attenuation, profiled amplitude
scores, the proposal band and ordered alternatives on deterministic P1/P-1/Pm
examples with unequal U. Homometric swaps, required-type preservation, malformed
labels and bounded neighborhood controls are tested. Full synthetic searches
require correct counts, geometry and the original final anisotropic mode.
Existing independent structure-factor values/derivatives, finite-difference,
symmetry/support and iterative-step checks remain applicable because those
equations do not change. The GSAS-II powder oracle does not cover this stage.

The realistic whole-search benchmark uses eight atoms, a 32-cubed grid and 2,108
measured rows. Both panels enable work-sliced phasing and both fast kernel
controls; only isotropic assignment competition differs. Kernel results include
the entire search and candidate fits rather than a scalar label decision.

The complete experiment uses all 35 development cases, seed zero, four starts,
800 requested cycles, 13-second native/15-second outer budgets and one numerical
thread. Final refinement remains anisotropic in both panels. Input-only caps and
the original exact non-H counts/maximum 0.05-angstrom and RMS 0.025-angstrom gates
are preserved, including seven above-cap cases in the denominator. A balanced
serial two-panel scheduler alternates leading mode by case. All 140 predictions
are saved before either reference audit is released. The adapter-only revision-16
seal retains the registry, sources and splits. No task-owned builds, tests or
microbenchmarks overlap timed collection. The 65 holdout outcomes remain unopened.

## Preserved first experiment: excessive isotropic work

The initial implementation used all expanded original observations for every
parent/neighbor isotropic fit. On a complete paired interleaved 35-case campaign,
it gained COD 2011241 without losing a contemporaneous control success:

| Measure | Control | Initial all-row isotropic competition |
| --- | ---: | ---: |
| Correct structures | 14/35 | 15/35 |
| Correct total non-H counts | 26/35 | 26/35 |
| Correct per-element counts | 15/35 | 16/35 |
| Strict 0.02-angstrom structures | 12/35 | 13/35 |
| Median wall time | 4.302 s | 4.462 s |
| Sum of solve wall times | 159.813 s | 163.008 s |
| Isotropic / anisotropic fits | 189 / 103 | 428 / 97 |
| Isotropic stage time | 5.310 s | 30.910 s |
| Anisotropic stage time | 53.475 s | 33.168 s |

The median paired control/new wall ratio is 0.89576x: typical paired solves are
about 11.6% slower despite 5.8% fewer anisotropic fits. The extra isotropic cost
outweighs the final-fit savings. COD 2011241 changes O6/N6 to the correct O8/N4,
with Cr2/C10 unchanged, and RMS/max coordinate errors 0.00710/0.01232 angstrom.
The policy is therefore not enabled by default. Its eight-atom benchmark has
medians 116.293/119.082 ms, also showing overhead on that case.

All 140 predictions precede both reference barriers and all 37 source hashes
match the two initial snapshots. Native SHA-256 is
`f5c18e90dce66e3e7a1000de0676fa0262cf351707010255230a5afcafc30212`.
Reports/comparison/schedule and raw microbenchmark samples are retained under
`benchmarks/structure-solution-isotropic-*-macos.json`. The initial numerical
implementation is preserved in `benchmarks/structure-solution-isotropic-v1.patch`,
relative to starting commit `9ff8c15`. The patch includes its API, diagnostic
bindings, focused tests, independent reference, benchmark adapters and lock;
applicability was verified against that commit using an isolated temporary Git
index. Exact collection source snapshots remain under `runs/isotropic-assignment/`.

This diagnosis motivates the current independent-family version: longer
isotropic fitting should not repeatedly pay for expanded equivalent rows when
all candidates can compete on the same bounded observed-family subset. The
final requested-mode fitting data remain unchanged. A second complete paired
experiment, rather than reused initial timings, evaluates that change.

Other CPU-intensive user calculations were active during collection; they were
left untouched. Both panels retain per-job load samples. The initial control
also misses historical COD 2108342 under its phase wall slice, and the new path
does too. Therefore the 14-to-15 comparison above is between contemporaneous
panels, not a claim to retain the previous campaign's full 15-case success set.
External SHELXT controls retain the same 13 structures in both panels. Host load
and deadline-dependent trajectories limit small performance/reliability claims.

## Completed independent-family comparison

Both fresh panels use native SHA-256
`7121f9815456edd7dd031917bd1da857c95e74cfd80079c0c42f38ab9b45f832`.
All 37 recorded implementation files match the current sources and both saved
snapshots. Every mode/method contains exactly the 35 development cases; seven
above-cap references remain in each denominator. All 140 prediction events precede
the two reference barriers, with 18 control-first and 17 assignment-first cases.
Every actual assignment fit uses the recorded common screening-family count;
the two-parent/two-neighbor/four-retained bounds hold throughout the full panel.

| Measure, all 35 development cases | Current control | Independent-family competition |
| --- | ---: | ---: |
| Correct structures with exact per-element counts | 15/35 | 16/35 |
| Correct total non-H atom count | 26/35 | 26/35 |
| Correct per-element non-H counts | 15/35 | 16/35 |
| Strict 0.02-angstrom structures | 12/35 | 13/35 |
| Wall median, including startup | 3.362 s | 3.360 s |
| Wall p95 | 7.789 s | 6.953 s |
| Sum of all solve wall times | 127.951 s | 121.523 s |
| Phase starts / scored cycles | 98 / 39,400 | 98 / 39,396 |
| Isotropic / anisotropic fits | 185 / 106 | 426 / 100 |
| Isotropic stage time | 3.254 s | 7.874 s |
| Anisotropic stage time | 37.469 s | 26.487 s |

The new path performs 5.7% fewer anisotropic fits and spends 29.3% less time in
that stage. Added isotropic work partly offsets this: combined isotropic/final
fit time falls 15.6%, while cumulative solve wall time falls 5.0% (6.429 seconds).
Phase-search time is effectively unchanged, 72.081/72.161 seconds. There is still
no broad typical-latency improvement: the median paired control/new wall ratio is
0.99591x, a 0.4% slowdown, with 17 cases faster and 18 slower. The median paired
difference is +0.00755 s and native-time ratio 1.00055x. Across all 15 common
successes the paired ratio is 1.01638x, about 1.6% faster, while the unpaired
successful-only medians are 1.422/1.334 s. The paired measures, rather than the
unpaired medians alone, describe the typical matched effect.

No current control success is lost, and all 15 successes of the preceding
work-slice campaign are retained. COD 2011241 is gained again with correct
Cr2/O8/N4/C10 and the same 0.00710/0.01232-angstrom RMS/max coordinate errors as
the initial implementation. Its R1 is 0.06759, above the solver's 0.06 early-stop
target; reference correctness and target attainment remain distinct. COD 2108342
is correct in both fresh panels, which reach 398/397 cycles before their phase
backstop, versus 137/169 during the heavily loaded initial campaign. These are
separate experiments, not outcomes selected by combining panels.

Some improvements are material: the correctly recovered COD 2242657 falls from
2.166 to 0.683 seconds. Several still-incorrect searches also become cheaper,
including COD 2010543 (10.049 to 5.856 s). Regressions remain, including still-
incorrect COD 2020988 (6.942 to 8.286 s) and correctly recovered COD 2021199
(5.587 to 6.240 s). Only one returned model is exactly identical to control;
the four no-model outcomes remain unchanged. Ten cases still have the right
total count but wrong per-element counts. Improved labels do not solve the full
remaining count/structure problem.

The fresh collections ran at much lower and steadier host load: median one-minute
load samples are 3.53/3.51, versus 28.30/27.59 in the initial experiment. No
task-owned tests/builds/microbenchmarks overlap either timed collection, and other
user jobs were never stopped. External SHELXT controls give 13/35 structures and
29/35 total counts in both fresh panels, with wall medians 0.861/0.862 s and a
paired ratio of 0.99215x. Small timing differences remain load-sensitive. PhaseSmith
still receives the exact space group, so these bounded scores do not establish
SHELXT capability or general reliability parity.

The revised eight-atom whole-search benchmark gives medians 114.008/115.089 ms,
about 0.9% overhead on that particular case. Raw samples and hashes are retained
in `benchmarks/structure-solution-isotropic-family-kernels-macos.json`. Complete
normalized reports, the guarded paired comparison and verified collection
schedule are `benchmarks/structure-solution-isotropic-family-{control,assignment,
comparison,schedule}-macos.json` (each name is a separate file).

Final validation passes 1,778 Python tests (11 external-data skips, 34 marker
deselections), 403 Rust tests (34 ignored), strict all-feature/all-target Clippy,
Rust formatting, repository Ruff and whitespace checks. No correctness threshold
or numerical tolerance was loosened. This is single-seed development evidence;
the option remains off by default, and all 65 holdout outcomes remain unopened.

To reproduce collection, use a fresh directory with the release experimental
extension installed:

```sh
.venv/bin/python benchmarks/solution_assignment_interleaved.py \
  --output runs/isotropic-assignment-new --seconds 15
```

This scheduler selects development only and enables the recorded matched
controls. After collection, normalize each panel with
`benchmarks/solution_corpus_reaudit.py`, then compare the two complete reports
using `benchmarks/solution_phase_ablation.py`. The committed family comparison
contains the exact input report hashes and all case-level results.
