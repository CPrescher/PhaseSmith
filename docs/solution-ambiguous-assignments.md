# Bounded ambiguity-aware element assignments

This experiment starts from `412c050`, with multiplicity-weighted final fits and
all earlier best execution controls enabled in both panels. The new experimental
`ambiguous_assignments` option is false by default and requires
`isotropic_assignment_refinement`. It adds bounded categorical exploration,
retains close final competitors and can request one corroborating phase start.
It does not supply a formula or enumerate all atom-count combinations.

## Proposal equations and bounds

For orbit i, element e and isotropic displacement u, reuse the prepared complex
orbit column and neutral scattering factors:

    C_ie(h,u) = P_i(h) f_e(s_h) exp[-8*pi^2*u*s_h^2]
    F_h = sum_i C_i,label_i(h,u_i)
    k = sum_h |F_h| Fo_h / sum_h |F_h|^2
    J = sum_h (k|F_h|-Fo_h)^2 / sum_h Fo_h^2.

Each complete parent has per-orbit isotropic displacements from its relaxed fit.
For allowed element changes with |Z_new-Z_old| <=2, score both the original U
and a displacement-compensated proposal. Let s_star^2 be the observed-amplitude-
squared-weighted mean s_h^2; use the measured row closest to that value for the
scattering ratio. The second seed is

    delta_u = clip(log[f_new(s_anchor)/f_old(s_anchor)] /
                   (8*pi^2*s_star^2), -0.02, 0.02)
    u_new = clip(u_old + delta_u, 1e-5, 0.2).

This matches scattering near one resolution as a proposal heuristic, not an
optimized physical identification. Nonpositive/nonfinite ratios retain the
original seed. Explicit interval boundaries and first-row tie ordering are
deterministic. Both seeds remain available for coupled changes; discarding one
before coupling could lose an alternative rescued only by a coordinated edit.

Keep the twelve best single-edit columns, including their displacement variants,
then evaluate pairs affecting different orbits. These include arbitrary two-label
changes, not only swaps preserving element counts. Each candidate's full-cell
counts follow from its original orbit multiplicities. Required elements remain
represented; atom topology/count ceilings are unchanged. Up to 32 distinct label
vectors inside J <=1.25*J_parent+1e-4 compete. Keep at most four proposals, preferring
distinct inferred count vectors before filling remaining slots with positional
label alternatives. All candidates are still ordered by their calculated J.
This bounded diversity and proposal band are heuristics, not confidence regions.
Cache storage keeps only twelve single-edit columns, with existing family/orbit
limits retained. Cancellation/deadline checks bound enumeration.

The plan must have bare orbit columns; passing already displaced columns is
rejected to prevent double attenuation. Inputs are not modified. Empty/incomplete
labels, invalid U and invalid proposal limits fail explicitly. Types separated
by more than two atomic numbers do not trigger this extra near-element search;
the original broader assignment path remains available.

## Relaxation and final competition

The original isotropic parents and neighbors remain in the retained pool. From
its current best parent, the new path gives up to four proposals eight isotropic
iterations each on the same bounded independent-family screening data. If a
proposal improves the best comparable relaxed rank, a second round can fit two
more proposals. Thus at most six extra isotropic fits occur per map. It does not
restart from every possible stoichiometry. Identical retained assignments at
near-identical coordinates are skipped.

The guarded policy fits the best original isotropic assignment first, retaining
it as the reference for new competitors. At most one challenger is then promoted;
reaching the R1 target on the reference does not suppress a challenger within six
isotropic rank units or a challenger with a better isotropic rank. A larger-gap,
worse-ranked challenger remains deferred after a successful reference fit.
Final fits retain the original anisotropic mode, measured observations,
multiplicity weighting and parameter penalties. A changed assignment must have
both non-worse final rank and non-worse R1 to displace its reference (local
arithmetic tie margins 1e-7 rank units and 1e-10 R1). A challenger that trades one
metric against the other is retained after supported candidates, with the
disagreement counted explicitly. The original winner-beam width still bounds
retained alternatives. The usual rank still orders supported candidates.
This is a conservative selection heuristic, not a correctness certificate.
No extra physical scoring term, valence rule, hidden hydrogen model or organic-only prior
is introduced. Close assignments may remain unresolved by these observations.

If two retained final states from the first start have different element-count
vectors and a final rank gap <=6, permit one additional independent phase start
before the usual R1 early stop. This does not change the meaning of `target_met`,
which remains the existing fit target rather than a correctness certificate.
All starts still share the original global trials/cycles/time caps. The six-unit
gap is an empirical search-allocation rule, not calibrated model odds.

Per-map audits record cheap scores, extra attempted/successful relaxation fits,
rounds and unresolved count competition. The summary separately records actual
corroborating-start requests that override an otherwise eligible R1 stop. Actual
work includes unsuccessful proposals and fits.

## Numerical validation and experiment protocol

The independent NumPy reference directly enumerates columns, elemental factors,
attenuation seeds, paired changes, objective values and count-vector diversity.
Random deterministic P1/P-1/Pm cases include special sites and unequal U. Proposed
objectives are also checked against the existing displaced amplitude/analytical-
gradient evaluator. Invalid inputs and complete synthetic structure/count
recovery are tested. A nitrogen row from the existing pinned scattering constants
is transcribed into the independent compact Python table; production factors
and tables are unchanged. Existing fused structure-factor values/derivatives,
finite differences, finite support and integrated intensity tests remain
applicable: this introduces no new differentiable physical term. Proposal clipping
is categorical initialization and is not differentiated. The GSAS-II powder
oracle does not cover this single-crystal search stage.

The matched experiments use all 35 development cases, seed zero, four starts,
800 requested cycles, native 13-second/outer 15-second caps, original input-only
admissibility bounds and final anisotropic fits. Both panels enable weighted
refinement and all earlier best controls; only assignment ambiguity differs.
Serial case-interleaved collection alternates the leading mode. All 140
predictions precede both reference barriers. Seven above-cap references stay in
every denominator. Adapter-only seal revisions 23/24 retain the unchanged registry
and 35/65 split. No task-owned builds, tests or microbenchmarks overlap collection.
The 65 holdout outcomes remain unopened.

```sh
.venv/bin/python benchmarks/solution_ambiguity_interleaved.py --output runs/assignment-ambiguity-new --seconds 15
```

Normalize both completed panels with `benchmarks/solution_corpus_reaudit.py` and
compare them using `benchmarks/solution_phase_ablation.py`. Criterion benchmarks
selected by `ambiguity_` cover cached eight-orbit multi-reflection proposals and
a complete count-free search with adjacent element types.

## Validation and kernel measurements

The feature-enabled full suite passes 2,199 Python tests (11 external-data skips,
34 marker deselections) and 413 Rust tests (34 ignored), with strict workspace
all-target/all-feature Clippy, formatting and changed-file Ruff. Thirty new
focused tests compare direct independent enumeration, coupled count-changing
proposals, invalid/double-applied U controls and complete synthetic searches.
The feature-mismatch report guard covers the new flag. Existing local tolerances
are unchanged.

The eight-orbit C/N/O proposal benchmark on 1,054 measured rows takes a median
0.631 ms. A separate eight-atom Fe/C/N/O complete 32-cubed search on 2,108 expanded
rows takes 240.334 ms in control and 228.862 ms with ambiguity handling (4.8% less
time on this example). Both execute two starts and 192 phase cycles; the new path
records eight extra short fits and 172 cheap ambiguity scores across its two
maps. Added search work can change which final fits are needed, so these timings
do not isolate the cost of only adding fits or establish a corpus-wide gain.

An earlier unanchored C/N/O whole-search benchmark never reached assignment;
it was replaced by the heavy-anchor example before corpus collection. Its
whole-search timing is not used for the paired feature comparison. Criterion's
automatic previous-build change estimates refer to that changed example and
must not be interpreted as the control/new effect. The direct proposal case
remains unchanged. Raw current samples, source/executable hashes and actual
work metadata are preserved in
`benchmarks/structure-solution-assignment-ambiguity-kernels-macos.json`.

## Preserved initial experiment and guarded follow-up

The initial complete 35-case comparison gains COD 1549173, 1550276 and 2243197
but loses COD 2011241: 18 versus 16 correct structures, 18 versus 16 per-element
counts, unchanged 26 total counts, and 14 versus 13 strict structures. Its
median paired wall ratio is 0.91654x, or 9.11% more time, with ten cases faster
and 25 slower. Summed wall time rises from 79.878 to 93.450 seconds. Screening
time rises 10.840 to 20.287 seconds; final fits increase from 97 to 108. There
are 291 extra short fits and 11,766 cheap scores, with zero corroborating starts.
Initial reports, normalized audits, schedule and kernel results are preserved
under `benchmarks/structure-solution-assignment-ambiguity-*-macos.json`. The full
initial implementation patch is `structure-solution-assignment-ambiguity-v1.patch`,
relative to `412c050`; clean applicability was verified in an isolated Git index.
Raw initial workers and source snapshots remain in `runs/assignment-ambiguity/`.

COD 2011241's initial winning competitor changes correct Cr2/O8/N4/C10 to
Cr2/O8/N6/C8. Squared amplitude error falls from 0.00590093 to 0.00588646, but R1
worsens from 0.0675929 to 0.0700928. Thus the original candidate was not absent;
ranking the extra competitor on squared error alone could displace it despite
the worsening absolute residual. This motivates the general reference/dual-metric
guard above, without any case identifier or reference counts in solver decisions.

The follow-up also defers extra relabel exploration when the current isotropic
parent has R1 >0.15 and its atom count is at least 90% of the input-only cap.
Poor fit near the count ceiling can signal an incomplete framework; this is an
explicit empirical allocation rule, not proof of missing atoms. It retains
exploration for high-R1 isotropic models away from the cap, where anisotropic
motion or wrong labels can still explain the error. The independent original
search remains available in every case. Focused development probes restore the
lost count vector while retaining the three gained count vectors; a fresh,
complete matched comparison is required before claiming that result generally.

The final guarded source passes the full 2,199-test Python suite (11 skips,
34 deselections), 414 Rust tests (34 ignored), strict all-feature/all-target
Clippy and formatting/lint checks. The additional native test constructs an
explicit squared-error/absolute-residual tradeoff and requires preserving the
reference; it also checks jointly improving and nonfinite competitors.

Current guarded kernel medians are 0.576 ms for cached proposals and
213.096/219.751 ms for control/new whole searches (3.1% more time on this example).
Both whole searches execute two starts and 192 cycles; treatment adds eight short
fits and 172 cheap scores. Source/executable hashes and raw current samples are
`benchmarks/structure-solution-assignment-guarded-kernels-macos.json`. These are
separate measurements from the initial source, not combined timing samples.

## Completed guarded comparison

| Measure, all 35 development cases | Original assignments | Guarded ambiguity search |
| --- | ---: | ---: |
| Correct labelled structures | 16/35 | 19/35 |
| Correct total non-H counts | 26/35 | 26/35 |
| Correct per-element non-H counts | 16/35 | 19/35 |
| Strict 0.02-angstrom structures | 13/35 | 15/35 |
| Raw median wall time, all outcomes | 1.349 s | 1.854 s |
| Summed solve wall time | 64.586 s | 73.759 s |
| Summed phase-search time | 32.430 s | 32.058 s |
| Summed isotropic-screening time | 8.613 s | 14.686 s |
| Summed final-refinement time | 8.458 s | 12.132 s |
| Starts / screening fits / final fits | 96 / 433 / 96 | 95 / 675 / 125 |

The median paired control/new wall ratio is 0.97029x, or 3.06% more time. Eight
cases are faster and 27 slower; the median paired difference is +0.017 seconds.
Summed wall time increases 14.20% (9.174 seconds). The larger ratio of raw medians
compares different positions in the two outcome distributions, not the typical
matched per-case effect. On the 16 common successes, the median paired ratio is
0.97389x (2.68% more time); raw wall medians are 0.831/0.894 seconds. These are
single-seed, single balanced collections on an uncontrolled host, not repeated
confidence intervals. Initial and guarded absolute timings must not be mixed.

The three structure/per-element-count gains are COD 1549173, 1550276 and 2243197;
no correct control structure or per-element-count recovery is lost. The total-
count recovery case set is unchanged. COD 2011241 retains correct Cr2/O8/N4/C10,
restoring the initial experiment's lost success. The new successes are:

| Case | Inferred non-H full-cell counts | RMS / maximum coordinate error | R1 |
| --- | --- | ---: | ---: |
| COD 1549173 | Se4 Cl4 C32 N8 | 0.01353 / 0.02449 angstrom | 0.07925 |
| COD 1550276 | S8 O16 N8 C4 | 0.00728 / 0.01721 angstrom | 0.04893 |
| COD 2243197 | P2 O12 N8 C24 | 0.00927 / 0.01873 angstrom | 0.04431 |

Reference counts/coordinates above are retrospective audits of saved predictions,
never solver inputs. Seven models still have correct totals but incorrect element
counts; four cases still return no model. There are no hard PhaseSmith timeouts.
The treatment records 10,305 cheap ambiguity scores, 249 extra short fits,
14 metric disagreements and three unresolved map-level count competitions.
It requests zero corroborating starts; the gains do not come from extra phase
starts. Guarded selection deliberately pays for the reference final fit, which
accounts for part of the increased final-refinement work. This improves recovery
at a measurable runtime cost and remains opt-in.

Both panels have the same native binary and 43 matching runtime-source snapshots.
All 140 prediction events precede both reference audit barriers, with 18 control-
leading and 17 ambiguity-leading cases. The original seven above-cap references
stay in the denominator; results within the 28 admissible references are 16/28
and 19/28. External SHELXT controls retain 13/35 labelled structures and 29/35
total counts in both panels. These PhaseSmith results still assume supplied
space-group symmetry and do not establish general SHELXT parity.

Reports, comparison, current kernel samples and verified schedule are preserved as
`benchmarks/structure-solution-assignment-guarded-{control,ambiguity,comparison,kernels,schedule}-macos.json`
(each name is a separate file). Raw workers, normalized audits and sealed source
snapshots remain in `runs/assignment-guarded/`. Adapter revision 24 preserves the
same registry and 35/65 split. The flag remains false by default; all 65 holdout
outcomes remain unopened.
