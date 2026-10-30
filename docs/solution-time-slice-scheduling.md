# Work-sliced phase-first scheduling

`work_sliced_phasing=False` preserves the existing phase-first schedule. The
experimental true option changes phase work allocation, not numerical equations
or acceptance criteria. It requires phase-origin, phase-quality and phase-first
selection, and is mutually exclusive with the earlier adaptive plateau rule.
Existing score-only and reusable-workspace optimizations remain independent.

## Policy and boundaries

The preceding solver allocates each P1 start a phase-search wall slice of
`max(0.01,0.30*max_seconds/trials)`, bounded by the overall deadline. A fast kernel
can therefore perform extra cycles without returning earlier to model completion.
The new policy keeps this same safety ceiling and the overall job deadline, but
adds a deterministic work ceiling to early starts.

For C requested cycles, T starts and zero-based start j, define

    early_starts = floor(T/2),
    early_quota = min(C, max(32, ceil(C/2))),
    quota(j) = early_quota if j < early_starts else C.

The standard C=800,T=4 campaign therefore uses quotas 400,400,800,800. With one
start the full quota is retained; odd start counts retain at least half their
starts at full work. The 32-cycle floor preserves the existing high-quality
warm-up; smaller requested budgets are never increased. Input ranges remain
C in [1,20000], T in [1,128], and j in [0,T). The quota depends on these controls
only, with no cell, element, atom-count, formula or reference-dependent branch.
If quality preparation falls back without an actual phase-first plan, the
original full quota is used rather than applying this policy to another path.

Reaching an early quota hands the selected best map to the ordinary supplied-
symmetry search, density recycling, integrated peak assignment, chemical screens
and requested-mode refinement. The existing complete-model target/shortlist and
winner ranking decide acceptance. If the target is not met, independent later
starts remain available, including full-quota fallback starts. The policy does
not assert that a plateau, correct count or low R1 proves structure correctness.
It does not resume the same early start at full work after a rejected handoff;
that is a bounded allocation choice assessed by the full development comparison.

The high-quality CFOM stop, seeds, iteration equations, random-omit sequence,
Fourier/ADP precision, Gaussian and reciprocal support, reflection sets, atom/
Jacobian limits, model budgets and correctness gates are unchanged. No unused
wall allowance is filled merely because it exists. The old wall slice remains a
backstop, so expensive starts can still end before their requested cycle quota.
This does not guarantee a speed gain for every input or every machine load.

Each actual P1 start records requested quota, completed/scored cycles, elapsed
phase-call time and stop reason (`cycle_limit`, `phase_time_limit`,
`quality_target`, `cancelled`, or `degenerate`). A phase-call time can include
legacy in-function model callbacks outside the phase-first path; the comparison
here uses phase-first calls only. These diagnostics distinguish deliberate short
work from slower kernels hitting the wall ceiling.

## Validation and experiment

The independent Python integer reference checks all quota boundaries and odd/
single-start behavior. Native tests exercise invalid requests and bounded full
fallback across the established cycle/start ranges. Synthetic P1, P-1 and Pm
searches require original atom counts and reference geometry, preserved input
arrays, and bitwise identical phase-history prefixes at equal seeds/cycles under
a generous wall ceiling. No profile/scoring/refinement equation or analytical
derivative changes, so existing independent value/gradient and support checks
remain applicable. The GSAS-II powder oracle does not cover scheduling.
A realistic whole-search microbenchmark uses eight atoms, a 32-cubed grid and
2108 measured rows. It measures the scheduling policy including model completion,
not a scalar quota call or a claim of equal work.

The predeclared complete comparison uses all 35 development cases, with optimized
score-only evaluation and Fourier workspaces enabled in both panels. Only work-
sliced scheduling changes. Seed 0, four starts, requested 800-cycle ceiling,
13-second native/15-second outer budgets, one numerical thread, original input-
derived atom caps and reference count/geometry gates remain fixed. Exact non-H
per-element full-cell counts plus maximum 0.05-angstrom/RMS 0.025-angstrom geometry
remain the primary structural criterion; strict 0.02-angstrom recovery is secondary.
Seven references above native capacity remain in every denominator.

`solution_timeslice_interleaved.py` runs both modes serially for each case/method,
rotating which mode goes first by case (18/17 leading positions). Its balanced
scheduler and source hashes are retained. All 140 predictions are saved before
both reference barriers are released. No test/build/microbenchmark from this task
overlaps timed collection; other users' calculations are left untouched. Per-job
load samples, external SHELXT controls and matched case timing ratios remain
available to assess residual load variation.

Adapter-only seal revision 15 retains the registry, split and source bytes.
The 65 holdout solver outcomes remain unopened. This is a development comparison,
not a holdout or general reliability claim; the scheduling control remains opt-in.

## Completed development comparison

Both panels used native SHA-256
`54e8743315d3891a95f8d888b4ee5ffbff09462b9f7e83d5d17298961e3987d4`.
All 36 recorded implementation files match the two retained source snapshots and
current sources. The schedule contains 140 prediction events followed by two
reference barriers, with 18 control-first and 17 work-first cases. Every case
remains in the denominator, including seven references beyond the native cap.

| Measure, all 35 development cases | Original schedule | Work-sliced schedule |
| --- | ---: | ---: |
| Correct structures with exact per-element counts | 14/35 | 15/35 |
| Correct total non-H atom count | 26/35 | 26/35 |
| Correct per-element non-H counts | 15/35 | 15/35 |
| Strict 0.02-angstrom structures | 11/35 | 12/35 |
| Wall median, including startup | 3.297 s | 3.398 s |
| Wall p95 | 7.700 s | 7.710 s |
| Sum of all solve wall times | 131.645 s | 126.796 s |
| Sum of phase-search times | 78.230 s | 71.954 s |
| Quality-scored phase cycles | 46,134 | 39,481 |
| Starts / final refinements | 95 / 104 | 98 / 106 |

The new schedule saves 14.4% of scored phase cycles, 8.0% of measured phase time,
and 3.7% of cumulative solve wall time in this campaign. It genuinely ends 29 of
57 short-quota starts at their work limit; 50 starts still hit the unchanged wall
backstop. Faster kernels can now return to model completion rather than always
spending the entire slice, but this is not a broad typical-latency improvement:
the median paired control/work wall ratio is only 1.00345x (median paired
difference -0.0272 s), with 19 cases faster and 16 slower. The native-time paired
ratio is 1.00723x. Across the 14 common successes, the paired wall ratio is
1.00080x, effectively unchanged. The higher unpaired work-panel median is also
reported above; neither unpaired median alone establishes the scheduling effect.

Some cases benefit materially: COD 2010901 falls from 1.492 to 1.044 s, COD
2010705 from 4.934 to 3.963 s, and COD 2011590 from 3.135 to 2.645 s. The most
substantial regression is COD 1550276, from 1.659 to 3.867 s: the early map does
not meet the existing model target, so all four starts run instead of one. Both
predictions have incorrect element counts. This extra fallback/refinement work
partly offsets the phase savings; target attainment still does not certify count
or geometry correctness. The policy retains this fallback rather than adding a
reference-informed acceptance shortcut.

No control success is lost. COD 4326074 is newly correct: its unchanged exact
counts previously accompanied RMS/max coordinate errors 1.401/1.882 angstrom;
the shorter map gives 0.000254/0.000424 angstrom. COD 2108342 remains correct.
There are 21 exactly identical nonempty selected models and four unchanged
no-model outcomes; other selected-model differences are retained in the full
reports. Count recovery remains the larger unresolved limitation.

External SHELXT controls recover 13/35 under this campaign's element-only input,
resource/timeout and exact-count/coordinate gates in each panel. Their paired
median timing ratio is 1.00013x; their wall medians are 0.886/0.845 s. These
bounded development results do not establish general superiority over SHELXT.

The eight-atom whole-search microbenchmark has medians 115.003/112.685 ms;
its overlapping mean confidence intervals do not establish an isolated speedup.
Raw samples and hashes are retained in
`benchmarks/structure-solution-timeslice-kernels-macos.json`. The complete
normalized reports, paired comparison and verified prediction schedule are
`benchmarks/structure-solution-timeslice-{control,work,comparison,schedule}-macos.json`.

Validation passed: 1,754 Python tests (11 external-data skips, 34 deselections),
403 Rust tests (34 ignored), all-target/all-feature Clippy with warnings denied,
Ruff, formatting and whitespace checks. Fixed-work phase prefixes are bitwise
identical in synthetic P1/P-1/Pm count-and-geometry recovery tests. The 65 holdout
outcomes remain unopened. Both the scheduling and execution flags remain opt-in;
this experiment supports modest cumulative savings and one extra development
success, not a reliable speedup for every structure.
