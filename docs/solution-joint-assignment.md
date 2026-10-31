# Tensor-informed joint assignment and count repair

The opt-in `joint_assignment_refinement` option acts once the ordinary element-only
search has finished, with the exact original winner retained as a fallback. It
requires phase-origin selection. It does not supply a formula, counts, reference
coordinates, hydrogen model, or compound-specific dispatch. Default behavior is
unchanged. The GSAS-II powder oracle does not cover this single-crystal stage.

## Diagnosis and bounds

On the preceding 35 development cases, seven misses have correct total counts
and wrong element counts. Post-hoc reference relabelling of the saved geometry
is used only as an explicitly oracle-assisted diagnostic. Some correct labels
improve the full anisotropic fit but require several coordinated edits; others
have worse squared-amplitude fit despite a smaller R1. The latter remain a
physical/ranking limitation, rather than a guarantee that broader search helps.
The diagnostic never enters the blind solver or selects benchmark predictions.

Repair applies to a complete, non-disfavoured refined model with R1 in
[0.04,0.18], and only when two allowed elements differ by at most two atomic
numbers. These are explicit work-allocation heuristics, not correctness gates.
There are at most three repair rounds and two full fits per round, inside a
single two-second repair safeguard and the unchanged overall deadline. No extra
phase starts run, and no existing phasing/refinement allowance is removed.

Up to two new separated orbit locations come from the existing common-U
partial-model difference Fourier map, with special-position projections and
original separation/atom bounds. That map proposes locations only. Existing
sites retain their individually refined Cartesian tensors; new sites use an
isotropic seed. Labels for new sites start vacant. Required elements must remain
represented throughout. All inputs remain the supplied HKL observations, cell,
symmetry and element types.

## Cached equations and discrete search

With q the Cartesian reciprocal vector without 2*pi, R_g a fractional symmetry
rotation and Q_g its Cartesian rotation, cache for orbit i

    P_i(h) = sum_g exp[-2*pi^2 (Q_g^T q)^T U_i (Q_g^T q)]
                  exp[2*pi*i h.(R_g x_i+t_g)].

Coordinate jets accumulate the same terms multiplied by
2*pi*i*(R_g^T h)_axis in the same image/reflection pass. Tensors are fixed in
this categorical cache; coordinate and displacement refinement retains the
existing fused analytical kernel and independent reference equations.

For categorical label e_i and diagonal tensor shift d_i,

    F(h) = sum_i P_i(h) f_e_i(s_h) exp[-8*pi^2 d_i s_h^2].

A vacant orbit contributes zero. Profile the positive amplitude scale and use
normalized squared-amplitude residual J. Cache ranking is

    n*log(max(J,1e-300)) + (4*active_orbits+2)*log(n).

It is a proposal ranking, not a statistical probability or the final score.
The search expands a width-four beam for at most eight depths. Moves relabel,
delete or add one orbit. Relabels are limited to atomic-number difference <=2;
count changes remain whole-orbit changes. Each move tests the current diagonal
shift and a compensated seed

    delta = clip(log[f_new(s_anchor)/f_old(s_anchor)] /
                 (8*pi^2*mean(s^2,Fo^2)), -0.02,0.02).

The anchor is the first measured radius nearest that weighted mean. Cumulative
shifts are bounded by [1e-5-lambda_min(U_i), 0.2-lambda_max(U_i)], preserving
positive-definite tensors and the original displacement interval. Seeds requiring
|8*pi^2*d_i*s_h^2|>600 are omitted explicitly. No differentiable clipping term is
added to physical refinement.

A third seed locally minimizes the same profiled objective in the diagonal
shift. For amplitude a, its shift derivative da, scale k and normalized
residual r=(k*a-Fo)/sqrt(sum(Fo^2)),

    dk = sum_h da_h (Fo_h - 2*k*a_h) / sum_h a_h^2
    dr_h = (k*da_h + a_h*dk) / sqrt(sum_h Fo_h^2)
    dJ = 2*sum_h r_h*dr_h,  H_GN = 2*sum_h dr_h^2.

Values and derivatives are computed in the same reflection pass. Up to three
projected Gauss-Newton steps are clipped to +/-0.01 A^2 and backtracked at
1,1/2,1/4; only a lower objective accepts a scalar step. Independent NumPy
formulas and centered finite differences check this derivative.

Beam candidates remain within J <=1.25*J_parent+1e-4. Deterministic ordering,
visited categorical states and count-vector diversity bound duplicate work;
at most 32 states within six proposal-rank units of the parent are retained.
Allowing close worse fixed-geometry hypotheses lets a later full tensor fit
resolve changes that the categorical cache alone cannot accept.

A second proposal route uses local displacement coherence without imposing a
restraint in the final model. For active near-element sites, neighboring symmetry
images at 0.9--2.3 A and atomic-number separation <=3 provide a median U-equivalent.
Try diagonal shifts halfway and fully toward that median, clipped to +/-0.02 A^2
and the physical eigenvalue interval. Up to four coordinate-fixed categorical
sweeps relabel near elements under each displacement seed. These are heuristic
seed locations, not chemical bond tests or calibrated displacement priors.
One final-fit slot is reserved for a distinct coherence proposal when available;
the other remains available to the tensor-aware beam. Coherence proposals may
have worse cheap ranks and still require the full-data acceptance guard.
Only two full fits are attempted per round.
Peak/orbit caches and original model capacity limits remain unchanged.

## Physical fits and acceptance

Full Cartesian tensor seeds are embedded through H=log(U) in the existing
orthonormal site-invariant log-tensor basis. Seed reconstruction checks site
symmetry and displacement mode. Tensors are transported through exact symmetry
images when representative sites change; they are not silently replaced by
isotropic averages. Existing complete-observation, requested-mode refinement
then uses the unchanged values, derivatives, weighting, scale and rank.

A repaired state must have non-worse final rank and R1, and at least a one-percent
relative reduction in the physical amplitude objective. This material-gain rule
is heuristic and must be checked empirically for loss of prior successes.
The original candidate remains available. Accepted repairs use a separate
`joint_assignment_repair` termination; any earlier consensus stop remains recorded
as search history rather than evidence for the new labels. Final `target_met`
still means only the original requested R1 target.

Actual scores, beam depths, attempted fits, accepted rounds and total repair
seconds include rejected attempts in the shared-search audit. Tensor caches,
beam ordering, invalid inputs, whole-orbit count moves, finite differences and
complete synthetic searches are checked against independent Python equations.
The 65 holdout cases remain closed. Performance and recovery require a complete
matched development campaign before promoting this experimental option.

## Complete matched development comparison

The baseline is `fd17515`, with all preceding best phasing, assignment, stopping
and numerical controls enabled in both panels. The only requested algorithm
change is `joint_assignment_refinement`. Both run one worker, seed zero, four
starts, 800 requested cycles, original input-only caps, 13-second native and
15-second process safeguards, and final anisotropic fits. There is no task-owned
build, test or kernel run during timed corpus collection. Mode order alternates
by case. All 140 predictions are saved before either reference audit. Adapter
seal 35 inherits the unchanged registry/split from seal 34 without opening raw
holdout data. The holdout stays closed.

| All 35 development cases | Control | Repair |
| --- | ---: | ---: |
| Correct labelled structures | 19 | 19 |
| Exact total non-H counts | 26 | 27 |
| Exact per-element full-cell counts | 19 | 19 |
| Strict 0.02-Angstrom structures | 15 | 15 |
| No model | 4 | 4 |
| Hard timeouts | 0 | 0 |
| Summed library solve seconds | 41.741 | 46.620 |
| Summed process wall seconds | 50.947 | 55.820 |

Every prior successful structure/count/strict-coordinate case set is retained.
The fresh control also reproduces every one of the previous 35 count vectors.
The repair changes three count vectors, all toward their deposited reference
counts; the other 32 stay unchanged. All seven above-cap references stay in the
35-case denominator. This is a total-count gain and partial label progress,
with no additional complete element-labelled structure recovery.

| Case | Original inferred counts | Repaired inferred counts | Reference counts | Count-vector L1 error |
| --- | --- | --- | --- | ---: |
| COD 2011426 | Br16 Ti4 N12 C16 | Br16 Ti4 N8 C24 | Br16 Ti4 N4 C28 | 20 to 8 |
| COD 2021292 | Cl4 O10 N14 C10 | Cl4 O10 N10 C14 | Cl4 O8 N4 C22 | 24 to 16 |
| COD 2242589 | S2 O4 N10 C17 | S2 O5 N6 C20 | S2 O6 N3 C22 | 14 to 6 |

COD 2011426 gains its missing four-atom orbit: total count 48 to the correct 52.
Its remaining C/N assignments are still wrong. Across the 31 paired returned
models, the sum of absolute full-cell element-count errors decreases from 210
to 182; four no-model cases remain explicit failures and are excluded only from
this auxiliary vector-error metric. Count-vector error does not assess whether
atoms with matching total counts occupy the right positions.

Repair increases summed solve time by 11.69% and summed process time by 9.56%.
The median paired library time increases by 4.25%. These use matched case ratios,
not the ratio of separately computed medians. Actual work is 184318 proposal
scores (including scalar shift optimization), 188 beam/sweep depths, 18 full
fits and five accepted rounds across three cases. The repair stage itself costs
5.193 seconds in aggregate. The two-second safeguard is cooperative rather
than an exact per-call interruption guarantee. The reported timings come from
one shared-host collection and must not be mixed with earlier absolute timings.

The eight-orbit, 1054-reflection Criterion proposal workload takes approximately
35.0 ms; the separate coherent reassignment workload takes 0.276 ms. Both retain
raw samples, executable and source hashes. These measure proposal kernels,
not the cost of a complete solve or a claim that the new route is faster.
The feature remains off by default: the measured count gain costs runtime,
and several reference assignments remain disfavoured by the current physical
objective even after further search.

Full local validation passes **428 Rust tests** (34 ignored), **2720 Python
tests** (11 external-data skips, 34 deselections), strict all-target/all-feature
Clippy and formatting. Forty-three focused tests cover direct tensor Fourier
values/jets, tensor seeding and invalid symmetry, scalar-shift finite differences,
independent beam and coherent sweeps, whole-orbit births/deletions, and complete
synthetic searches. New report tests reject feature mismatches and verify that
all 140 predictions precede reference grading.

Artifacts are `benchmarks/structure-solution-joint-assignment-*-macos.json`.
The oracle-assisted `diagnosis` artifact is explicitly separate from the blind
`control`/`repair` predictions and their `retention` audit.

```sh
.venv/bin/python benchmarks/solution_joint_interleaved.py --output runs/joint-assignment-new --seconds 15
# Normalize both completed panels with solution_corpus_reaudit.py, using new output paths.
.venv/bin/python benchmarks/solution_joint_verify.py --base runs/joint-assignment-new
```
