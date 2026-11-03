# FFT grain search

**Supported namespace:** this method is included in the 0.8 single-crystal
capability. Use `phasesmith.single_crystal`; ordinary builds include the native
kernels. The historical filename is retained for existing links. See the
[supported guide](single-crystal.md) for scope and migration. Numerical equations,
finite support and the specific limitations below remain unchanged.


An opt-in alternative is now available as **Row search (DAFi-based)**:
`find_grains(q, training=mask, method="rows", row_options=RowSearchOptions(...))`.
It groups training reflections before per-group lattice inference and the same
frozen-pool validation. `result.row_grouping` retains its exact raw output;
`group_rows` provides grouping alone. The existing default and all ensemble
searches remain FFT. See [Row search](experimental-row-search.md) for settings,
attribution and the distinction between grouping and final indexed ownership.

`phasesmith.single_crystal.grains.find_grains(q, training=mask, options=...)`
uses **unindexed reciprocal peak positions** (NumPy `(N,3)`, inverse Å, no
2π factor). It infers integer HKLs, direct lattice bases and competing grain
orientations. It accepts no input cell, HKLs, material, composition or symmetry.
Several orientations can describe grains of the same cell. This is not a space
group solver, a chemical identification or an exact DAFi implementation.
For primitive hypotheses and all 14 Bravais metric types after discovery, see
[experimental cell comparison](experimental-cell-comparison.md).
The native `spots` capability is included in ordinary 0.8 builds.

## Ownership and provenance

The FFT deposition/transform, maxima and nearest-lattice ownership kernels are
in `phasesmith-single-crystal::spots`, exposed through private native bindings. The array-only
Python API coordinates the search; Dioptas owns files, instrument translation,
controls, candidate tables and overlays. No runtime dependency on another
checkout, SciPy, spglib, Qt, GSAS-II or a process-global thread configuration is
introduced. RustFFT is isolated to the experimental crate, using at most four
private Rayon workers. The powder numerical core is unchanged.

The original workspace FFT/ownership implementation was reused at the user's
request. Source hashes are in `sfscxrd-fft-provenance.json`; that historical
project name is not the public algorithm/API name. The orchestration is adapted:
bounded native Newton vector ascent replaces L-BFGS-B; deterministic integer
size reduction replaces Niggli reduction; a caller-provided rotation-block split
replaces random splitting; and there is **no post-validation all-data polish**.
Do not describe this as bitwise equivalence of the entire old workflow.

## Equations and boundaries

For deposition spacing Δq, each q contributes unit weight to its eight
neighbouring nodes using trilinear cloud-in-cell weights. Accumulation is
float32, serial offset-then-point order, with the physical origin retained.
The unnormalised forward FFT's real part is the cosine sum. The rFFT half
spectrum has shape `(n,n,n/2+1)`. Each axis transforms in a fixed order; workers
do not change reduction order. DC equals the number of input positions, within
float32 roundoff. This is positional counting, not an intensity transform.

`Δq = min(2 max|q|/(n-4), 1/(2.1 max_length))`.
Every point must satisfy `|q_i|/Δq < n/2-1`; the search raises if the bounded
16..256 even grid cannot hold the full reciprocal extent. No point is silently
cropped. Fourier maxima lie in the inclusive direct radius interval; each is
at least as large as all 26 neighbours, with periodic x/y and clamped z edges.
Equal-score sites survive before stable flattened-index ranking and truncation.
These boundary conventions retain the original native kernel behaviour.

The existing fused float64 cosine/gradient/Hessian kernel refines translations:

`C(r) = mean cos(2π q·r)`

`∇C = -2π mean[q sin(2π q·r)]`,
`H = -(2π)² mean[q qᵀ cos(2π q·r)]`.

Twenty damped Newton passes, declared four-step backtracking, a 0.3 Å step
limit and ±1.5 Fourier-grid-step coordinate bounds limit each fit. Only training
positions enter this pass. Collinear vectors (`|cosθ| > .9995`) are suppressed;
arbitrary linear combinations are retained. Top pair supports screen triplets
with integer-phase residual strictly below `fractional_tolerance`. Volume,
conditioning, minimum and maximum edge limits are explicit.

With direct basis rows A, reciprocal rows B = A⁻ᵀ:

`h = rint(q Aᵀ)` (ties to even), `d = ||q - h B||₂`.

Each candidate undergoes at most eight alternating integer assignment / linear
least-squares reciprocal-basis fits. The first two use 2× the reciprocal
threshold. Rank-three support is required. Exact Euclidean row operations on
inferred HKL cores of sizes 24/48/96 propose integer sublattice corrections;
a smaller cell must retain ≥90% of the current support. Size reduction is an
integer basis change, not a Niggli or conventional-cell guarantee. Missing
reflections, noise and bounded search can leave a supercell unresolved.

Competing models use the nearest reciprocal residual, excluding 000, with
strict `d < tolerance`. Near ties with `best > ambiguity_ratio*second`, and
exact coincident zero residuals, remain unassigned. Zero label means residual
or ambiguous; positive labels are one-based frozen model IDs. Joint training
reassignment can change previous ownership. Discovery excludes points already
compatible with any existing model, preventing repeated rediscovery of one
lattice. Selection maximises support minus the approximate uniform-background
expectation `N V 4π tolerance³/3`, then minimises RMS and volume. Search stops
at the declared grain budget or lack of a new training model. An exhausted
budget is not evidence that all grains have been found.

## Reserved-data validation

The caller supplies a boolean partition; default is every fifth point held out.
For rotation data, Dioptas supplies its existing whole-track/five-frame-block
partition. All discovery, fitting and model competition finish before held-out
coordinates enter evaluation. Pool order and bases remain frozen thereafter.
Rejected models remain in the pool and exports; their removal must not silently
change reserved-data competition.

The fixed 64-control default uses uniformly distributed rotations from normalised
Gaussian quaternions (seed 842025). Each control rotates held-out q and reassigns
to the same frozen pool. Acceptance requires:

- At least `min_indexed` unique, rank-three training HKLs and six held-out matches.
- Held-out matches greater than every rotated control and at least twice
  `max(control mean,1)`.
- At least `mean + 4 sqrt(max(sample variance, mean,1))` held-out matches.
- At least half the training support extrapolated to the held-out population.

These are heuristic geometric checks, not calibrated false-discovery rates.
No held-out-driven replacement, ranking or final fitting is performed. The result
retains all frozen grain models, labels and per-round proposal/reassignment/gate
diagnostics. Compact FFT cell search remains available for comparison and now
returns up to a declared `max_candidates` (default 40), rather than silently
showing only five. This list is bounded search output, not all possible cells.

## Opt-in parameter ensemble and consolidated grain assignment

`find_grains_ensemble(q, training=mask, options=GrainSearchOptions(...))`
adds a multi-start path without changing `find_grains`. Each search sees only
the training coordinates. The default grid has 28 configurations: maximum edge
fractions `(1/3, 1/2, 2/3, 1)` of the caller's bound, reciprocal tolerances
`(1/5, 8/15, 1)` of its tolerance, and fractional proposal gates `(.06, .15)`;
four additional searches combine reciprocal fractions `(1/5, 1)` with volume
fractions `(.05, .15)`. Empty edge/volume intervals are skipped. Default searches
retain a caller's smaller grain allowance and cap it at eight models per search;
consolidation uses the caller's final grain allowance.
The example defaults of 30 Å and 20,000 Å³ therefore span 10–30 Å and include
1,000/3,000 Å³ volume bounds. These are generic search windows, not material
cells or a crystal-system prior. Caller-provided `searches=(...)` can replace
this grid; all configurations and failures remain explicit.

Only discovered bases enter consolidation; per-search held-out acceptance does
not select the pool. Reserved coordinates are first used after the consolidated
model pool and training ownership have been frozen. There is no DAFi executable,
DAFi assignment, supplied cell or atomic/intensity reference in this algorithm.

For each candidate direct basis A, the native score kernel supplies rounded HKLs
and reciprocal distance d, with 000 excluded as before. Use
`sigma = final_reciprocal_tolerance/4`, `V=abs(det A)` and positional evidence

```
E(q,A) = log(1 + exp(-d²/(2 sigma²)) / (V (2 pi)^(3/2) sigma³)).
```

The Gaussian is normalized by the reciprocal-lattice point density V. Thus a
large supercell cannot receive the same evidence as a compact cell merely by
placing many possible reciprocal sites near observations. This dimensionless
score is invariant under consistent reciprocal/direct unit changes. It is a
bounded proposal-ranking heuristic, not a calibrated likelihood, phase prior,
grain-count probability or intensity/partiality model.

Greedy selection maximizes marginal training evidence
`sum(max(E_candidate - E_best_selected, 0))`, with smaller V then stable pool ID
breaking ties. Every added model must have at least `min_indexed` training
positions in its strict core `d < sigma`, including at least `min_indexed` core
positions outside the already covered cores. Candidates with shared core count
`>= .8 * max(core_count_a, core_count_b)` are duplicates. The inclusive symmetric
overlap prevents a broad superset from deleting a compact model using only a
one-way containment test. These core rules govern proposals; the final native
distance and ambiguity boundaries and validation thresholds are unchanged.

Selected models undergo the existing joint training reassignment/refinement;
points can move between grains. This is consolidation across searches, not
concatenation of separately accepted groups. The final model pool is validated
once using the same rank/support/retention and 64-rotation checks as the single
search. Rejected final models remain visible and continue to participate in the
frozen ownership model, as before. Neither symmetric overlap nor a passing
candidate proves a primitive cell or a distinct physical grain.

Bounds: 1–32 searches, at most 256 total proposed models, at most 40 final models,
and an 8-million training-position/model-pair consolidation budget. Exceeding a
budget raises rather than truncating the inputs. Searches run sequentially with
their declared native worker allowance; concurrency is not multiplied. Evidence
and deterministic selection are array orchestration over existing native scores;
FFT, score and ownership kernel equations and smooth derivatives are unchanged.
The independent NumPy `grains_reference.ensemble_evidence` checks the equations.
No GSAS-II oracle covers positional multigrain discovery.

```python
from phasesmith.single_crystal.grains import GrainSearchOptions, find_grains_ensemble

result = find_grains_ensemble(
    q,
    training=training,
    options=GrainSearchOptions(grid_size=256, max_grains=8),
)
```

The ensemble costs the sum of its bounded searches, plus O(NM) evidence storage
and O(NMG) bounded consolidation for M pooled and G selected candidates. Use
`benchmarks/grain_search.py --mode ensemble --output ...` alongside `--mode single`
for a complete multi-peak workload and exact-repeat check. Synthetic tests check
compact-cell preservation against a dense superset, independent evidence and
unit scaling, reserved-coordinate independence, deterministic ownership,
invalid budgets and cancellation. The original single-search tests remain exact.

### Dataset 1 continuation, 29 October 2026

On the fixed 1,963 eligible tracks, the original three-way partition is retained
(1,082 discovery, 423 selection, 458 test). The single-search workflow gives two
accepted position-refined candidates, 535 strict matches overall and 121 in test.
The default ensemble gives three accepted candidates, 569 overall and 128 in
test, identically in two complete repeats. Its added candidate has 29 strict
training and 11 selection matches at admission; the final assignment contains
49 tracks, including 37 previously unmatched, and eight test tracks. Only two of
its final 49 tracks meet the detector gates of either other consolidated model.

The separately labelled normal 80/20 workflow gives two candidates / 542 overall /
123 test matches for the single search, versus three / 561 / 130 for the ensemble.
This uses 1,505 training positions and is not the same discovery population as
the primary three-way comparison. Both ensemble partitions repeat exactly. The
positions, previously selected image mapping, fixed detector plane and provisional
0.5-degree exposures are unchanged. These examined data are development evidence,
not a new unseen benchmark. No physical grain count, primitive cell or chemical
identity is established, and most eligible tracks still remain unexplained.

The 28-search ensemble costs about 27 seconds versus 1 second on this dataset;
it is a thorough-search option, not a speed improvement. The synthetic
2,674-position benchmark gives 0.988–0.999 s single versus 27.893–27.964 s ensemble,
with exact results on three repeats. A 350-point uniform-cloud control yields no
ensemble candidates. Full validation gives 1,450 Python passes, 11 skips and
34 deselections; Rust formatting, workspace Clippy and all-feature Rust tests pass.
No golden data or production numerical defaults are replaced.

A subsequent post-selection HKL/cell audit finds that candidates 1 and 3 are
nearly co-oriented (about 0.32 degrees after an integer basis change), with a
reciprocal scale ratio 1.4946, near 3:2. An explicit shared larger-cell model,
fitted on the same frozen training positions, indexes all eight of candidate 3's
test tracks: XY RMS 0.978 pixels, versus 1.217 pixels for its separate-cell refit.
The common cell has edges about 30.3 Å, just above the original 30 Å bound.
Thus three accepted indexing candidates do **not** establish three physical
grains. Candidate 3 retains a common-cell alias alternative; the original
cross-check against only the other two fixed bases was insufficient to exclude
it. The larger cell is also an indexing hypothesis, not an established physical
cell. These are conditional diagnostics on examined data, not new blind evidence.

## Cost and validation

### Optional Fourier-plane conditioning (30 October 2026)

`find_grains_ensemble(..., conditioning=GrainConditioningOptions())` adds
bounded training-only proposals before the existing consolidated ownership and
validation. `conditioning=None` retains the previous numerical path. This is
an opt-in experimental search, not a replacement for the default single search
or an automatic Dioptas UI change. No DAFi executable, groups, cells, source
chemistry, supplied HKLs or intensity information enters the search.

The development failure audit found that several sparse, independently
DAFi-supported lattices supplied only one or two independent vectors in the
global Fourier shortlist. A triplet search cannot propose their cells from that
shortlist. Conditioning addresses the concentration of signal before attempting
another triplet search. For a training-only Fourier translation `r`, retain

```
S(r) = { q : |q.r - rint(q.r)| < phase_tolerance }.
```

The strict default phase gate is 0.08. An integer projection from a lattice
compatible with `r` remains near an integer; an unrelated uniform phase survives
with probability approximately `2*phase_tolerance`. This is a proposal heuristic,
not a membership probability or a physical grain assignment. Nonuniform measured
clouds need not satisfy that approximation.

Defaults search the half-length window of the final options, retain at most 80
noncollinear selectors from 400 initial maxima, and use the caller's worker
allowance. Each subset uses at most 96 maxima, 40 vectors, 64 pairs and 16 cell
refinements, with a reciprocal proposal tolerance one fifth of the caller's
final tolerance. The subset grid is at most128 initially; if the full subset
extent cannot fit, it expands to256 without cropping positions or changing
physical bounds. `seed_search` and `subset_search` accept explicit
`GrainSearchOptions` overrides; all actual settings and expansions are recorded.
Final physical basis bounds also apply to every proposed cell.

Sparse subsets additionally try exact integer generated-lattice cores of
8/12/16/20 positions, followed by the original24/48/96 cores, sorted by reciprocal
residual. This lets a small clean core propose removal of a supercell index when
a few incorrect HKLs would make a larger core generate the full integer lattice.
Rank three is required by the exact integer construction. A smaller cell must
retain at least90% of the current subset's support at the same proposal tolerance;
its smaller volume is not assumed to establish the physical primitive cell.
The original `_refine` path keeps its original24/48/96 cores.

Each selector contributes at most one candidate, ranked on the full training
population by the existing support-minus-uniform-background expectation, then
RMS, volume and stable proposal order. These at most80 additions enter the same
density-normalized consolidation, frozen competition, training refinement and
64 rotated-control validation as the global ensemble. Final reciprocal tolerance,
ambiguity, rank, support, fold, retention and null-deviation checks are unchanged.
Rejected candidates stay visible. Selection/test positions never define the
selectors, subsets, primitive corrections or proposed pool.

There are at most256 global plus80 conditioned proposals, with the existing
8-million training-position/model-pair consolidation limit still enforced.
Conditioning adds up to81 Fourier searches, each bounded by the grid/vector/fit
budgets above. This is a thorough-search option, not a speed optimization.
Progress remains monotone in ensemble-stage units, with repeated notifications
during the conditioning stage for cancellation. Per-seed audits identify the
actual work. Native FFT, scoring and ownership kernels, smooth Fourier
derivatives, and their independent NumPy references are unchanged; no new
profile value/Jacobian or GSAS-II oracle boundary is involved.

```python
from phasesmith.single_crystal.grains import (
    GrainConditioningOptions,
    GrainSearchOptions,
    find_grains_ensemble,
)

result = find_grains_ensemble(
    q,
    training=training,
    options=GrainSearchOptions(grid_size=256, max_grains=40, workers=2),
    conditioning=GrainConditioningOptions(),
)
```

Known-lattice tests check sparse recovery among unrelated positions, compact-cell
volume, contamination, frozen reserved-coordinate independence, exact repeat,
null clouds, option bounds and cancellation. `benchmarks/grain_search.py --mode
conditioned` measures a full multi-grain cloud and repeated deterministic output.
Examples4/5 are examined development data, not fresh blind validation. Their
original frozen peak populations and geometry are retained, including the
documented preprocessing departures. The method does not resolve example4's
background-subtracted negative result or establish physical grain counts.

On the frozen example5 development peaks, the new path retains1/2/0 candidates
in raw/diamond/arPLS arms, predicting20/37/0 unique reserved-test positions.
The paired diamond-arm pipeline changes from0 candidates/0 matches without
conditioning to2/37 with conditioning. An ablation using conditioning but only
the original larger integer cores still retains none. The independent DAFi
training fits agree with the new A/B candidates through approximate unimodular
orientation maps (0.26–0.75degrees residual rotation,0.57–1.62% stretch).
Example4's paired diamond result remains2 candidates/32 reserved matches;
conditioning provides no material improvement there. The example5 arPLS arm
now proposes compact cells but still fails native null/fold validation.

A15381-position empirical-marginal null (coordinates independently permuted
within training and selection) gives16 proposals and0 accepted candidates.
This supplements the synthetic uniform-cloud control, not a calibrated
false-discovery-rate claim. All36 grain tests pass. The full Python campaign
has1463 passes plus one CLI-path environment failure that passes when rerun with
an absolute PYTHONPATH;11 tests skip and34 are deselected. Rust formatting,
workspace all-feature Clippy/tests and focused Ruff checks pass.

Serialized three-repeat2674-position benchmarks give median27.361seconds for
the existing ensemble versus32.900seconds with conditioning, about20.2% extra
work at four native workers. Both retain exactly two synthetic grains and
repeat bases/labels exactly. Full real-data times are separately descriptive
because verification jobs overlapped. No performance improvement is claimed.

For N training positions, n³ grid nodes, V retained vectors, P screened pairs,
G discovered models and fixed fit budgets, deposition costs O(N), FFT
O(n³ log n), maxima O(n³ + K log K), vector fitting O(N V), screening roughly
O(N V² + P V N) in the readable orchestration, and ownership O(N G) per pass.
FFT memory is O(n³), ownership output O(N); no N×G native distance matrix is
allocated. Multiple discovery rounds repeat these bounded costs. There is no
promise of exhaustive or polynomial-cost general lattice enumeration.

Independent NumPy deposition/rFFT and assignment references live in
`grains_reference.py`. Tests cover randomized fields, DC mass, exact worker
repeat, maxima ties/boundaries, strict distance cuts, origin exclusion, same-cell
orientations, two different cells plus outliers, random-cloud rejection, inferred
integer sublattices, frozen held-out independence, and validation budgets. Existing
Fourier derivative tests cover the fused value/gradient/Hessian. The new discrete
FFT/ownership kernels have no smooth intensity/profile derivative or integrated
intensity convention. GSAS-II's powder oracle does not cover this positional
single-crystal search. `benchmarks/grain_search.py` measures a realistic mixed
multi-grain cloud and 3D grid, not a scalar profile call.

## Local first validation (2026-10-25)

A deterministic 2,674-position mixed two-grain benchmark (192³, four workers)
recovers both synthetic lattices in 0.94–0.97 s on the recorded Apple Silicon
host, with exact bases/labels on three repeats. This is a new search API, with
no prior PhaseSmith full-search timing baseline. A separate 3,000-position,
128³, ten-model black-box comparison reproduces the original native field,
maxima, scores and assignments exactly; millisecond FFT timings fluctuate and
are not a claim of a speed improvement. The adapted full workflow is not
numerically identical to its predecessor.

The frozen P1 comparison uses the existing 1,295 reciprocal tracks and
1,052/243 training/held-out split, with no input HKLs or known material/cell.
The negative vertical-axis sign yields one passing model, approximately
4.3494 × 11.6430 × 13.4169 Å, angles 89.9387°, 89.9714°, 90.0056°,
volume 679.43 Å³; 550 training and 127 held-out matches. The worst of 64 rotated
controls has six held-out matches. Eight positive-axis model candidates all
fail reserved-data gates. The model is consistent with the previous lattice;
this has not established a smaller primitive cell or the correct symmetry.
Input hashes, fixed options, labels and audits are retained in the separate
ignored run `runs/fe7c3-fft-grains-20261025/`, without changing earlier reports.

### Optional stability and early positional proposal refinement

Two independent opt-in proposal stages address dependence on the highest global
Fourier amplitudes and premature rejection of plausible, imperfect bases.
They leave `_consolidate`, joint ownership and `_finish` unchanged. Use
`stability=GrainStabilityOptions()` and/or
`proposal_refinement=GrainProposalRefinementOptions()`. Positional refinement
also requires `observed` detector `(x_pixel,y_pixel,omega_degree)` rows and a
fixed `PanelGeometry`. All new arguments default to `None`; existing calls and
Dioptas defaults are unchanged. These are experimental thorough-search options.

Stability adds the full training cloud and two no-replacement 80% training
subsets, with fixed seed `options.seed+2` and row indices sorted back to their
original deposition order. These supplement existing global and conditioned
proposals. Each uses the existing bounded Fourier-plane conditioning with
proposal-only minimum support12 (final minimum remains20). Fourier vectors keep
half the amplitude-ranked shortlist, then cycle through36 strata: three equally
spaced length bins across the search window and12 angular sectors. Canonicalize
`r` and `-r` by making the largest absolute Cartesian component positive. Its
axis and the signs of the other two components define the angular sector.
Within each stratum amplitude rank is stable; sorted stratum order resolves
ties. Existing collinearity and physical basis bounds still apply. Resampling
supports0–4 subsets; fraction is bounded to[0.5,1]. Each view adds at most80
proposals and records its training row IDs, vectors and chosen triplets.

Early polishing ranks proposed bases using only global training observations
and the existing density-normalized reciprocal evidence. A proposal needs at
least18 nonzero training HKLs with reciprocal error<0.015 (the caller's final
tolerance), finite unflagged predictions, initial XY distance<25pixels and
absolute omega residual<2degrees; rank3 is required. These initial gates are the
existing `RefinementOptions` gates. At most64 candidates are fitted. For each,
choose the narrowest of tolerance/4, tolerance/2 and tolerance with at least18
eligible observations, sort by reciprocal residual and cap at256 positions.
At least12 rows fit the nine reciprocal-basis entries; at least5 other rows are
withheld internally. **Both internal sets consist entirely of global training
rows.** Internal withholding records optimizer diagnostics; it does not select
final acceptance. It never exposes global selection/test coordinates to fitting.

The existing robust positional objective, fixed geometry and fused native
prediction/Jacobian implementation are reused:

```
r_i = ((x_calc-x_obs)/2, (y_calc-y_obs)/2,
       (omega_calc-omega_obs)/0.3)
L(B) = sum_i rho_Huber(||r_i||; transition=2.5).
```

The optimizer's40-iteration bound is unchanged. Original proposals are retained
and finite, physically bounded changed bases are appended before the original
tight-core cutoff. Convergence, objective histories, internal training IDs,
initial and refined core support, fit failures and budget exclusions are
recorded. A successful optimizer run is not an acceptance certificate.

With default stability and early polishing, at most256 global+80 original
conditioned+240 stability+64 polished bases can enter consolidation. The
8-million training-position/model-pair bound remains enforced. Stage progress
supports cancellation. No FFT kernel, profile value/derivative, support
convention, intensity extraction, detector calibration or oracle interface has
changed. Existing independent NumPy Fourier/score references and finite-
difference prediction tests still apply; there is no new GSAS-II oracle case.
Tests additionally check recovery of a distorted synthetic lattice with fewer
than20 initial tight-core positions, option/budget limits, null clouds,
cancellation, exact repeat/worker invariance and independence from both reserved
reciprocal coordinates and reserved detector observations.

`benchmarks/grain_proposals.py` measures the extra stages on two oriented
synthetic lattices with detector-space centroid noise and600 unrelated positions.
It compares conditioned versus combined discovery serially, checking recovery
of both lattices and exact repeated bases/labels. The standalone development
report freezes the four real-data populations and includes one-stage ablations;
its outputs live outside the repository and do not become golden fixtures.

On the subsequently frozen mask-matched development populations, baseline
conditioned versus both new stages gives accepted hypotheses/reserved-test
matches of3/40→4/51 (example4raw),0/0→2/20 (4arPLS),0/0→2/31 (5raw) and
1/13→3/32 (5arPLS). The example5raw ≈282.5Å³ candidate matches one accepted
DAFi-assisted orientation (0.146° residual rotation,0.302% maximum stretch).
The second DAFi orientation remains missed. In5arPLS a ≈280Å³ rejected proposal
matches one DAFi cell within the existing2% relation bounds but fails native
rotated-maximum,fold and null-deviation checks. The accepted larger-cell
hypotheses are not established as the DAFi primitive cells. Counts therefore
measure conditional lattice hypotheses, not established physical grains.
On5raw, stability alone gives1/19, early polishing alone0/0, and both2/31.
The two additions have complementary effects; more proposals do not guarantee
better final primitive-cell identification. Full scientific provenance,
settings parity and every rejected model are retained in the external
`fft-stability-refinement` development report.

In the paired2254-position forward-geometry synthetic benchmark, both modes
recover exactly two known oriented lattices and repeat bases/labels exactly
three times. Median wall cost is6.319s conditioned versus20.145s combined
(3.19×), with two global searches and two native workers. The four eight-search
real-data cases take120–133s, about1.47–1.72× the saved conditioning-only times.
These are additional search costs, not speed gains; the host was not reserved
exclusively. One detector-marginal empirical null, independently permuting
x/y/omega within each global partition then recomputing q, produces three
consolidated hypotheses and zero admitted candidates. This is a diagnostic
negative control, not a calibrated false-discovery rate or proof of cell truth.

### Sparse conditional alternatives and positional congruence proposals

The next optional experiment separates two failure modes: missing independent
Fourier translations in a conditional shortlist, and larger-cell proposals
whose otherwise strong primitive correction is vetoed by unrelated broad-q
matches. `sparse_conditioning=GrainConditioningOptions(...)` adds independent
training-only conditional views while retaining the original global/conditioned
pool. It uses the fixed `GrainStabilityOptions` views (full training plus two80%
subsets by default). `candidates_per_seed` defaults to1 and may be1–3. Alternative
bases are deduplicated only for an approximate unimodular row change
(coefficient departure<0.005, relative Cartesian departure<0.002); integer
supercells remain alternatives. Each view still contributes at most80 models.
`selected_all` records every submitted candidate; `selected` retains the legacy
best-per-selector diagnostic. The development experiment uses grid256,
400maxima,80vectors,128pairs and80refinements in these extra subset searches.
No known cell or comparator coordinate is passed to the inverse.

`aliases=GrainAliasOptions()` additionally requires original detector
observations and fixed geometry. For each training-only proposed parent basis
A (row vectors), B=A^(-T), and inferred integer HKLs h, enumerate canonical
projective normals n modulo prime p. The first nonzero coefficient is one.
For its position j, define an integer row basis H with rows

```
H_i = e_i - n_i e_j  (i != j),  H_j = p e_j.
det H = p;    h.n == 0 (mod p) <=> h = k H.
k_i = h_i (i != j);    k_j = (h.n)/p.
B_child = H B;    A_child = H^(-T) A.
```

These are exact integer-kernel identities, derived directly from the congruence.
They preserve the Cartesian prediction `k B_child = h B`. Floating-point
positional fitting follows that exact assignment conversion; volume shrinks by
p before fitting. No DAFi/GSAS-II implementation, space-group/chemistry hint or
centering table enters this construction. Exhaustive small-prime residue
references independently test the kernels and exact round trips.

A parent needs≥18 rank3 global training observations with reciprocal error
below the caller's unchanged final tolerance, unflagged finite predictions,
XY residual≤3pixels and absolute omega residual≤0.5degrees. A proposed kernel
must retain≥80% of those strict training observations and≥18 points. These are
proposal conditions, not tests of crystallographic primitivity. In particular,
weak superlattice peaks, systematic absences or incomplete sampling can make
smaller cells appear plausible. Original parent models are retained.

Defaults enumerate primes2/3, at most64 eligible root parents,96children,
three reduction levels and cumulative index≤12. Roots rank by maximum retained
strict count, then smaller volume and stable pool order. Internal positional
fitting uses the existing40-iteration Huber optimizer,≥12fitted and≥5internally
withheld global training rows, with deterministic seed derived from the root,
depth and kernel. Equivalent children are deduplicated with the same strict
unimodular criterion. Child history, kernel, cumulative index and training IDs
are exported. The8-million position/model pair guard applies to the augmented
pool; progress supports cancellation. Larger permitted limits remain explicit
in the typed options; no unlimited integer search is performed.

The original tight-core selection, joint fitting, frozen candidate competition,
nearest-lattice ownership, rank/support,64rotated controls, fold/null-deviation
and retention gates are unchanged. The downstream experiment retains the same
training-only positional refinement and≥20training/≥6selection strict admission.
Reserved test coordinates never choose kernels, proposals or fitted cells.
A candidate passing a proposal condition is still rejected if it fails any final
gate. Defaults for existing API calls and the Dioptas UI are unchanged. Native
FFT, scoring, values/Jacobians and support conventions are unchanged; no new
profile derivative or GSAS-II oracle case is introduced.

The subsequent examined-data campaign has mixed results. Relative to the
previous stability/polishing run, the expanded+congruence path changes
accepted hypotheses/test matches from4/51→4/43 (example4raw),2/20→0/0
(4arPLS),2/31→4/49 (5raw) and3/32→4/35 (5arPLS). It does not establish a
validated recovery of the second DAFi lattice. More hypotheses are not a
validated improvement, and the example4 regressions argue against promoting
these options to defaults.

In5arPLS, the expanded pool contains a rejected ≈1128Å³ candidate compatible
with an index4 alias of DAFi's second orientation. Training-only subsequent
congruence diagnostics can reconstruct an approximate ≈281Å³ primitive-sized
cell from it. Broad-q ranking, refitting and competition can still suppress or
distort such hypotheses. Exploratory detector-ranked/fixed-basis refinements
recover that oriented relationship but still fail the unchanged rotated-null
admission in the tested frozen pools. These diagnostics are retained outside
the repository; they are not additional default algorithms or a claim that
primitive-cell recovery is solved. Default behavior and final validation gates
remain unchanged. These data have been repeatedly examined for development;
reserved positions here are no longer a fresh independent algorithm benchmark.

The alias-only ablation provides a narrower positive result on5arPLS: it
retains an approximately283.27Å³ child matching the first DAFi oriented lattice
(0.227°residual rotation,0.437%maximum stretch). Its lineage is an approximately
3379Å³ parent and three congruence reductions with cumulative index12; the
candidate has45tight training points and25novel points at consolidation.
The complete alias-only run changes3/32→4/38hypotheses/test matches,while raw5
remains2/31. The second DAFi lattice remains unvalidated. This is conditional
alias recovery on examined data, not proof of a physical primitive cell or a
reason to enable the wider experimental search generally.

The frozen2254-position two-lattice synthetic benchmark recovers both known
oriented lattices with exact bases/labels on all three repeats in both modes.
Median cost is21.016s for the preceding stability/polishing path versus94.474s
for expanded+alias proposals (4.50×),using two native workers and BLAS1. Wider
real-data runs take approximately310–333s. One empirical detector-marginal
null produces three consolidated candidates and zero admitted hypotheses.
All1487Python tests pass (11skip,34deselected),along with the required Rust
workspace formatting,Clippy and all-feature tests. The wider path stays off
by default; these controls do not turn the mixed development result into a
validated general improvement.

## Opt-in observation-aware ownership and preserved fitting branches

`find_grains_ensemble(..., ownership=GrainOwnershipOptions())` requires fixed
`PanelGeometry` and observed `(x_pixel,y_pixel,omega_degree)` coordinates.
Existing calls default to `ownership=None` and keep their original numerical
path. No DAFi source code, supplied cell, group label or chemistry is used.

Let `J=dq/d(x,y,omega)` at each observed centroid and `K=J^-1`. Rust evaluates
`q` and `J` in the same pass from the planar ray and Rodrigues rotation:
`dq/dx=R(-omega)(I-u u^T)fast/(lambda |ray|)` and similarly for y;
`dq/domega=-(pi/180) axis cross q`. An optional pixel map chains its Jacobian.
The SVD rank condition is `s_min > 1e-12 s_max`; singular observations stay in
the original population but cannot claim a reflection. Source geometry is fixed.

For nearest rounded nonzero HKL `h=rint(q A^T)`, with reciprocal basis
`B=A^-T`, the observation chart residual is `r=K(q-hB)`. It is a local linear
approximation at the detector centroid. Mean reciprocal track positions can
differ from converting separately averaged centroids, and are deliberately not
silently replaced. Eligible matches require the original strict `|q-hB|<0.015`
and the existing inclusive physical bounds `hypot(r_x,r_y)<=3` pixels and
`|r_omega|<=0.5` degrees. Ownership ranks
`sqrt((r_x/2)^2+(r_y/2)^2+(r_omega/0.3)^2)` and retains the original best/second
ambiguity ratio0.8, stable ties and origin exclusion. Unlike legacy q-distance
ownership, a candidate outside the physical bounds cannot block an eligible
one. Rust streams observations without an N-by-M allocation. In every rotated
control, transform coherently as `q'=qR`, `K'=KR`; rotating q alone changes the
error model and is invalid.

Training selection ranks density-adjusted Gaussian evidence
`log(1+exp(-d^2/2)/(V_A (2pi)^1.5 sigma_xy^2 sigma_omega/|det K|))`.
It requires at least20 tight q-core matches,20 new tight-core matches and20
new physically eligible training observations. A larger candidate is deferred
only when a smaller candidate has an exact integer HKL map of index2..12 on
at least20 rank-three shared physical training observations, fewer than20
additional physical observations support the parent, and the child has higher
common evidence. Additional coefficient/Cartesian/rotation/strain bounds are
0.025/0.02/2degrees/0.02. Every deferred parent and map remains in diagnostics.
This is training model selection, not proof of crystallographic primitivity.

At most64 candidates per branch receive the existing forward positional fit.
Early and late integer children are refitted on their own training membership
for at most8 outer cycles. This prevents reusing a parent's fit after integer
reduction changes membership. Inner fixed-HKL fitting uses a fused native
Huber2.5 cost, analytical gradient and Gauss-Newton normal, checked against an
independent dense Python reference and centered finite differences. The normal
is an approximation to the Hessian, not a full covariance estimate.

The main branch also preserves an independent training-only reciprocal-space
consolidation/refinement trajectory before observation fitting changes any
parent. That bounded branch receives the same observation fits and alias
proposals; its selected alternatives are appended to the main pool before final
training selection. It is constructed once, with no recursive branch spawning.
This deliberately increases proposal/fitting work. No rejected global-selection
candidate is removed to make another model pass.

Joint training fitting uses two simultaneous updates with frozen competitors.
For each model, set `L_i=exp(-d_i^2/2)` inside its training eligibility region and
zero elsewhere. Uniform noise is normalized over the existing eligibility
cylinder: `eta=(2pi)^1.5 sigma_xy^2 sigma_omega/(2pi r_xy^2 r_omega)`.
With `b_i=eta+sum_{j!=i}L_j`, minimize
`sum log((b_i+1)/(b_i+exp(-d_i^2/2)))`. Responsibility
`w_i=exp(-d_i^2/2)/(b_i+exp(-d_i^2/2))` multiplies the fused gradient and
positive Gauss-Newton normal. Only training fitting uses shared fractional
responsibilities. Final ownership remains hard, including ambiguous zeros.

Final count/rank,64-rotation maximum, twofold null excess, four-null-SD and
train-to-selection retention expressions and thresholds remain unchanged.
The development campaign additionally retains the unchanged exact forward
3pixel/0.5degree gates and minimum20training/6selection. Test observations are
reserved until cells and the complete competing pool are frozen. These are
changed ownership decisions under unchanged validation thresholds, not identical
labels to the previous metric. New and comparator models must use the same
ownership rule for a matched comparison. More accepted hypotheses or assigned
positions alone do not establish accuracy or physical grain count.

No GSAS-II oracle fixture covers multigrain discovery or these observation
coordinates; geometry derivatives, independent normal-equation references,
rotation covariance, worker repeats, synthetic lattices and negative controls
provide the relevant checks. Source profiles and their finite support are
unchanged. The optional path remains experimental and off by default.

The frozen example4/5 ownership campaign, including the identical downstream
final forward fit and explicit optimizer-completion gate, retains
hypotheses/unique strict test observations of4/55 (4raw),7/65 (4arPLS),7/57
(5raw),10/110 (5arPLS). The previous stability/polishing path gave4/51,2/20,
2/31,3/32. Both previously accepted DAFi orientations are matched in5arPLS;
the second approximately280.60Å³ candidate owns41training,9selection and16test
observations, versus a maximum rotated-null selection count2. It differs from
the reference orientation by approximately0.107degrees, with maximum relative
stretch0.00511 within the pre-existing equivalence bounds. This is positional
lattice evidence, not a solved structure or proof of primitivity.

The result is mixed:5raw still matches only one reference orientation, and4raw
loses the previously accepted approximately66.4Å³ hypothesis within the fixed
oriented/integer-alias bounds. Its test assignments change by17lost/21gained.
The4arPLS path preserves oriented/integer-related alternatives for both previous
accepted hypotheses. Exact model counts must not be read as physical grains;
5raw includes two accepted fitted variants compatible with the same first DAFi
reference. Further duplicate/family competition remains unresolved.

Initial pre-final diagnostics omitted the common last forward fit and are not
the primary comparison. Once that fit is applied consistently, a hard-joint
ablation also recovers both corrected5 orientations. Soft responsibilities and
branch preservation therefore cannot individually be credited as necessary for
recovery. These examined-data results do not justify changing defaults. The
standalone report, input/source hashes, all hypotheses, assignment overlap and
negative ablations are retained under the local2026-10-30
`fft-ownership-investigation` analysis directory.

## Experimental recovery budgets and training-only feasibility

Three opt-in controls accompany the follow-up recovery experiment:
`GrainProposalRefinementOptions(distinct_budget=True)` uses the existing64fit
budget on distinct starting bases and ranks marginal evidence on physically
consistent training predictions. Exact/strict unimodular repeats do not consume
another fit. `GrainOwnershipOptions(competitive_selection=True)` permits a
candidate to improve evidence on at least20already-covered physical training
observations; it retains20total tight-core and20physical proposal support.
This replaces a *training-selection* union-coverage veto. Final validation
expressions and thresholds do not change.

`GrainRecoveryOptions(paired_search=True)` adds bounded paired Fourier
conditioning and coherent local fitting. Take the existing diverse measured
Fourier translations `r_i`, define
`M_i={q:|q.r_i-rint(q.r_i)|<0.08}`, and rank noncollinear pairs by
`(n_ij-n_i n_j/N)/sqrt(max(n_i n_j/N,1))`. The sine of their angle must be
at least0.2. Examine at most64pairs, each with at least20training positions.
A subset FFT proposes the third vector while the first two remain locked.
Use grid128, expanded to256only for reciprocal extent;400maxima,80vectors and
at most80third-vector fits. Keep at most3alternatives/pair and80total.
Proposal support12 matches the existing sparse-stability convention; it is not
an acceptance threshold. All positions used in these operations are training.

For up to8paired proposals, deterministic seed `options.seed+6` draws at most
4096rank-three training-HKL triples per parent. The first-principles equation
is `B=H^-1 Q`. These local proposals use a wider *initial search* window:
`q` distance below4times the ordinary tolerance, predictedXY below50pixels and
absolute omega below5degrees; fitted deformations are bounded by5degrees and
5percent relative stretch from the FFT parent. These bounds do not replace the
final3pixel/0.5degree bounds. Require12narrow reciprocal matches before robust
q refinement and17physical training matches before ranking; retain at most
8distinct alternatives per parent. No DAFi cell or group enters this inverse.

The final forward fit now has a separate, explicitly unvalidated entry point,
`fit_crystal_proposal`. It shares the existing optimizer with `refine_crystal`
and returns immutable coefficients, history and training mask without validation
metrics. Tests establish bit-identical coefficients/history for the same
training rows. The validated API still requires independent held-out rows.
Selected models receive this fit **once** before final validation. Remove the
weakest model failing owned-training count20,20unique HKLs or rank3; reassign and
repeat until the training pool is feasible. Audit every removed model and fit.
This selection never sees global selection/test coordinates or their outcomes.
Rejected candidates after global validation still remain in the frozen pool.

Downstream positional validation must reuse the completed forward fit, retaining
the original completion gate and at least5initially eligible selection
centroids under the original25pixel/2degree initial fit window. Refitting a
second time changes HKL/eligibility membership and is a different experiment.
The unchanged final count/rank/null/retention checks and exact3pixel/0.5degree
gates then apply to the frozen pool. A training-only fit is not itself validation.

The result remains mixed across settings. With the paired search disabled, the
budget/competition repair recovers the original approximately66.4Å³ oriented
candidate in example4raw. With paired search enabled, the additional FFT and
coherent fits recover raw example5's second DAFi orientation. Enabling the
extra search can suppress the example4 candidate again and can lose a prior
corrected-example5 orientation. No common regression-free setting or automatic
selection policy was established. All controls remain off unless requested;
these development results must not be advertised as universal grain recovery.
Negative plane-exclusion and local-fit experiments remain in the analysis audit;
the unused native plane-exclusion API was removed. No native profile value,
derivative, finite support or GSAS-II oracle behavior changed in this follow-up.

The strict known-two-lattice control initially exposed an additional failure:
an index5subset of a true lattice was counted as another component. After
training feasibility, defer a smaller cell as an alternative representation
when it has fewer than20exclusive physical training observations, a larger cell
has at least twice its physical support, and their shared>=20rank-three HKLs
obey an exact integer map of index2..12. The existing0.025coefficient,
0.02Cartesian,2degree and0.02stretch bounds apply. Preserve both bases, all fit
records and the integer map in `subcell_alternatives`. Independent orientations
and cells with>=20exclusive training observations are protected. This does not
establish crystallographic primitivity or a space group.
The unchanged exactly-two benchmark and known-orientation checks now pass twice
with exact repeated bases/labels; the initial failure artifact is preserved.
The two raw-target recoveries remain unchanged by this correction. The
cross-example search-setting regressions described above are still unresolved.

## Exact search-kernel acceleration and the one-second experiment

The experimental native engine now evaluates independent Fourier directions on
its bounded worker pool. For positions q_i and direct vector r, the objective is
F(r) = mean_i cos(2 pi q_i dot r), with gradient
-mean_i 2 pi q_i sin(2 pi q_i dot r) and Hessian
-mean_i (2 pi)^2 q_i q_i^T cos(2 pi q_i dot r). Value, gradient and Hessian are
still accumulated together in one ordered pass over observations for each
vector. Only vectors run in parallel; there is no reordered point reduction.
The serial API remains available and independent NumPy equations remain in
`reciprocal_reference.py`.

Bounded Newton ascent retains the original 20 iterations, minimum curvature
0.01, 0.3 angstrom step cap, grid-derived box, and trial scales 1, 1/2, 1/4,
1/8. Accepted fused moments are cached. Backtracking visits only directions
which have not yet accepted a step. A direction whose four trial scales all
fail is retired: its position, moments, Newton step and all future trials would
otherwise repeat exactly, independently of the other directions. This is exact
memoization rather than a new stopping tolerance. No cache survives a call.

Phase membership S_ji = [abs(q_i dot r_j - rint(q_i dot r_j)) < tau_phase]
is packed into zero-padded 64-bit words. Pair and triple support are exact
population counts of bitwise intersections. The final candidate ordering still
uses descending support, ascending determinant magnitude and lexicographic
vector indices. Batched NumPy determinant/SVD checks retain the original edge,
volume and condition bounds. Packing snapshots caller arrays, ignores padding,
and preserves stable results for one through four workers.

A separate native proposal scorer shares nearest-HKL rounding and residuals
across three explicit ascending reciprocal cutoffs. Its output triples are
(nonzero-HKL count, sum of squared reciprocal errors, physically eligible
count). It excludes 000, uses strict reciprocal radii and inclusive physical
bounds in the caller's fixed observation chart. Each model's sum remains in
point order. This low-level scorer does not select or admit a lattice and is
not enabled in the standard ensemble. The readable array reference is
`grains_reference.candidate_counts`. Native snapshots are bounded to 100000
points, 50000 models and 500 million point-model pairs; packed phase matrices
are bounded to 400 vectors and 20 million memberships. Numerical errors and
invalid dimensions/cutoffs are rejected at the Python boundary.

The local example4/5 development report `fft-common-configuration/report.html`
under the 2026-10-30 analysis records exact before/after direction and full-run
replays, and experiments sharing one direction field across three reciprocal
cutoffs. The fast portfolios change proposal budgets, remain analysis-only,
and do not preserve all previously retained lattices. They are not a validated
replacement or a new default. Kernel times below one second must not be
reported as the time for a complete validated grain search. Full timing must
include every configuration, competition, final fitting and the unchanged
selection/held-out controls, starting from the existing peak/track table.
The complete conservative pipeline remains substantially above the requested
approximately one-second budget even after these exact accelerations.

`benchmarks/grain_fft_speed.py` measures 12000 mixed lattice/noise tracks,
400 Fourier directions, the original versus packed triplet ranking, and
512 candidates scored at three cutoffs. It asserts exact Fourier outputs and
candidate order. It is explicitly a kernel benchmark. The objective is a
position-only Fourier mean, not a normalized intensity profile; integrated
intensity/moment tests and GSAS-II profile-oracle fixtures do not apply. Fused
Fourier gradient/Hessian finite differences, independent array references,
worker-repeat tests, exact support boundaries/padding and full-search replays
provide the applicable numerical controls. Final acceptance thresholds,
chemistry boundaries, geometry assumptions and image preprocessing are unchanged.

## Conditional-search priorities and preserved proposal fits

Further opt-in changes address conditional-search cost and the fragile compact
example4 hypothesis. Defaults and the final validation function are unchanged.

* `GrainConditioningOptions(rank_by_population=True)` visits a bounded number
  of seed directions by descending count of training observations satisfying
  `abs(q dot r - rint(q dot r)) < phase_tolerance`. Ties retain the original
  direction order. Counts use the actual sharp phase window; they are not a
  Fourier-amplitude rank or an inference about chemistry. A private within-call
  `prepared` argument permits reuse of directions for the exact same training
  points/settings. It is not a persisted result or a supplied-cell shortcut.
* Triplets can request a limited prefix. Exact phase support and determinant
  order are established first; conditioning/SVD is evaluated only until that
  many valid triplets are found. The returned valid prefix matches the complete
  original ranking. Diagnostics now call this `returned_triplets` and record
  `triplet_limit`, rather than misreport it as the number of all feasible cells.
  Batched direction-collinearity comparisons fall back to the original scalar
  dot/norm arithmetic within 32 machine epsilons of the strict 0.9995 boundary.
* Small native point-score batches (at most 2048 points) avoid worker scheduling;
  point arithmetic and output order are unchanged. Fourier-field maxima use
  partial selection before sorting the retained prefix under the same total
  score/index ordering. These are exact accelerations, not altered cutoffs.
* `GrainProposalRefinementOptions(trim_fractions=(0.9, 0.8, 0.7))` adds deterministic
  training-only proposal fits. First fit every eligible training observation
  with the existing robust forward model. Rank its squared residuals
  `d_i^2=(dx_i^2+dy_i^2)/xy_scale^2+(domega_i/omega_scale)^2` in stable order.
  Independently fit the closest `max(12, ceil(f*N))` rows for each declared
  retained fraction, starting each fit from the same all-row anchor. Up to
  three distinct fractions in [0.5,1) are allowed. The full-row proposal is
  retained, and a failed trim does not delete successful alternatives. By default the
  original internal-split fit is also retained in its original output prefix;
  trimmed alternatives are appended. Explicit `retain_split_fit=False` isolates
  a deterministic trim-only policy and requires nonempty `trim_fractions`.
  The reported partial common-setting recovery uses that explicit policy;
  the additive policy is separately audited and does not preserve every target. This is
  a bounded robust proposal heuristic, not a claim of a global trimmed-loss
  optimum, validation, or calibrated measurement uncertainty. It uses the
  unvalidated `fit_crystal_proposal` API, so removing an internal random split
  never fabricates held-out metrics. Native fused positional derivatives and
  the final reserved selection/test coordinates are unchanged.
* `GrainOwnershipOptions(preserve_input_models=True)` preserves copies of
  already tight-core/physical-eligible input proposals before geometry fitting
  can mutate them. A geometry fit is an alternative rather than replacement.
  Inputs which cannot enter the existing selector/family comparison need not
  be duplicated. Preserved inputs and source indices are audited, and all
  alternatives subsequently compete in the same frozen ownership pool. This
  option does not keep a candidate because of a selection/test outcome.

A compact candidate can otherwise be lost before global validation: a proposal
fit's pool-index-dependent internal training split or a later geometry update
can move it from 20 to 19 narrow-core training observations. Deterministic trim
alternatives and preservation of an already eligible proposal can repair that
trajectory without reducing the original 20-point tight-core proposal gate.
The final minimum of 20 owned training observations/unique rank-three HKLs,
six selection observations, 64 coherent rotated controls, fold retention,
reciprocal/physical gates and exact forward assessment remain unchanged.

The local `fft-fast-conditioning/report.html` development report records cold
raw/arPLS example4/5 tests, actual changes in prior-hypothesis retention, the
partial common-setting recovery, complete wall time, and rejected alternatives.
A faster conditional stage is not a one-second complete inverse. These options
are off by default and do not establish regression-free recovery, primitive
cells, physical grain counts, chemistry or solved structures. Raw tables,
mask-matched preprocessing, provisional geometry and EIGER saturation None are
inherited without modification.

`benchmarks/grain_conditional.py` benchmarks a realistic many-track conditional
search and optionally loads explicitly saved first-party before-source/native
snapshots for an exact replay. It separates the equal-output 80-seed run from
changed-coverage population-ranked shortlists. No comparator model is an input.
The scalar boundary, valid-prefix, worker-dispatch, deterministic robust-fit,
partial-failure and option-default tests complement the independent Fourier/
position reference and derivative checks. No profile equation is added, so
intensity normalization/moment or GSAS-II profile-oracle tests are inapplicable.

## Retaining fitted training branches

An opt-in `phasesmith.single_crystal.retention.merge_fitted_training_pools`
combines **already qualified and fitted global-training pools**. It accepts
NumPy q positions, fitted direct bases and observation charts, with no chemistry,
reference cells, acceptance flags or reserved coordinates. The caller must first
perform the ordinary upstream proposal qualification and forward fitting.
The private ensemble `_training_only=True` staging hook returns immutable bases,
labels and diagnostics before `_finish`; it carries no accepted-candidate flags
and permits a training-only input array. Default ensemble behavior is unchanged.

Every incumbent stays in the pool unchanged, including one which may fail final
validation. The merge ranks additions by marginal physical evidence, tests each
against eight independent training rotations (seed+881), requires at least 20
physically supported and uniquely owned training observations, 20 distinct
rank-three HKLs, and positive evidence gain. After each proposed addition, every
incumbent must retain at least `min(original_owned, min_indexed)` training owners.
Integer oriented alternatives (index at most 12 by default) are not added twice.
All choices, failures, source indices and training counts are audited. Budgets
remain at most 40 models/additions and eight million point/model pairs per pool.
This training guard is not a general guarantee of reserved-fold retention.

The narrow-core gate (`q_error < tolerance/4`, at least 20 points) belongs to
the **pre-fit selector**. Reapplying it to a different final fitted matrix changes
the stage contract: the recovered example4 compact hypothesis has 19 narrow-core
points after fitting but passes the original physical/owned/rank and validation
gates. The merge reuses its upstream qualification and does not refit or apply
this additional post-fit rejection. Final assessment reuses exactly one completed
forward fit per model, freezes the entire pool before using reserved observations,
and retains failed competitors. There is no pruning or tuning on test outcomes.

`GrainProposalRefinementOptions(cohort_aware_budget=True)` augments the opt-in
distinct fitting budget. Nearby oriented matrices count as the same start only
when the actual bounded observation cohorts and their integer HKL maps match.
When the legacy internal split is enabled, its actual training-row membership
must also match. Pool-index-dependent splits can otherwise turn similar matrices
into different fit problems. The default matrix-only policy is preserved.

`GrainOwnershipOptions(native_fields=True)` fuses the existing physical evidence,
narrow reciprocal core and physical eligibility into a bounded Rust model/point
pass. With `h=rint(q A^T)`, `delta=q-h A^-T`, `r=K delta`, and
`d2=(r_x/s_xy)^2+(r_y/s_xy)^2+(r_omega/s_omega)^2`, the nonzero-HKL eligible
evidence is

`E = log(1 + exp(-d2/2) / (abs(det(A))*mu))`,
`mu = (2*pi)^(3/2)*s_xy^2*s_omega/abs(det(K))`.

Reciprocal cutoffs are strict; the detector XY-radius and omega bounds are
inclusive. Disabled charts have zero evidence/physical eligibility but retain
their defined reciprocal-core flag. Nonfinite or nonpositive active density
normalizations fail explicitly. Each model keeps ordered per-point arithmetic;
workers distribute independent models. The independent NumPy reference uses
array equations, not Rust, and local tolerances cover the expected elementary
floating-point differences. This is a positional ranking heuristic with no new
profile term or optimizer derivative. Existing forward values and fused
derivatives remain unchanged. Default fields still use the prior orchestration.

The local `fft-hypothesis-retention/report.html` records a fresh four-arm
conservative discovery plus training-only repair. The previous fast policy lost
seven hypotheses before the fitted pool and five during fitting/competition;
reducing 80 directions and changing the sparse 256-cubed grid to 128-cubed also
changed proposal coverage. The repaired cold pipeline recovers 28/28 previous
accepted hypotheses, the compact example4raw target and both targeted example5
orientations under unchanged counts, rank, nulls, fold retention and exact
detector gates. It remains approximately 50–58 seconds overall, including final
assessment, despite a 2–4 second repair branch. Cached-pool diagnostics and the
millisecond merge are not complete inverse timings. The one-second goal is
unmet. These development datasets have been inspected repeatedly; they are not
a new blinded accuracy benchmark. Primitive cells, physical grain counts,
space groups, geometry/omega calibration and chemistry remain unresolved.

`benchmarks/grain_retention.py` measures only the fields stage on 12,000 tracks
and 64 candidates. Tests cover independent equations, boundaries, coherent null
rotations, worker repeatability, invalid inputs, fit-cohort identity, negative
noise controls, preserved incumbent provenance and the stage-specific core gate.
An external diffraction/profile oracle does not cover this training-pool policy.

## Broad discovery without shortening its coverage

The native `windowed_maxima(q, n, step, minimum, maximum, limit)` path computes
the same maxima as `build_field(...).maxima(...)`, without materializing the
unused spectrum. The mathematical transform, deposition weights and search
grid are unchanged. Each retained line executes the same length-n RustFFT in
the original Z, Y, X axis order:

1. Retain only Z frequencies with `(z/(n*step))² <= maximum²` after the Z FFT.
2. Execute the full Y FFT for those frequencies, retaining Y frequencies whose
   squared signed frequency is at most `maximum²` before the X FFT.
3. Execute the full X FFT and retain the corresponding X frequencies.
4. Apply the original inclusive radial annulus and 26-neighbour maxima test.

Discarded components cannot lie in the requested sphere because squared
frequency components are nonnegative. The original maximum finder masks
neighbours outside that annulus to negative infinity too, so no extra halo
is needed. Periodic X/Y and clamped Z neighbours retain their exact semantics,
including Nyquist edges. Compact array indices are monotonic in the original
flattened indices, preserving score/index ties and the pre-truncation total.
Empty deposited Z lines reuse one actual zero-line FFT rather than assuming
the sign bits of its floating-point zeros. Deposition still accumulates in
the same serial offset/point float32 order. Invalid derived nonfinite direct
grid spacing is rejected explicitly. The old dense-field API remains available
for array access and replay comparisons.

`prepare_fourier(q)` returns an immutable owned point snapshot. Its
`evaluate(vectors)` computes all 13 original Fourier values/gradient/Hessian
entries in one ordered pass. The coefficients `2π q_j` and
`(2π)(2π) q_j q_k` are invariant across trial vectors and are reused without
moving their multiplication by sine/cosine or the final 1/N scale. Native
arithmetic order and the platform trigonometric functions are unchanged.
Inputs are copied before releasing the GIL. At most 100,000 points and
200 million point/vector pairs are allowed by the binding and work boundary.
Trial-vector validation and overflow errors remain explicit. Batches with
at most 32,768 pairs execute serially; larger batches distribute independent
vectors across the caller's bounded pool. Each vector retains the same
ordered point reductions, so worker choice does not change its result.

The Fourier evaluator prepares a cloud lazily for batches with at most 32
vectors and retains the original compact point-only evaluator for larger
batches. The coefficient cache has not shown a reliable large-batch benefit;
this dispatch avoids imposing its extra memory traffic on those calls.

The ordinary `_fourier_peaks` path uses these native capabilities when built;
the dense/moment fallback remains available for older native builds. It still
runs the original bounded 20-pass Newton ascent, four backtracking scales,
step/radius bounds, seed directions, fixed views, reciprocal grids and fitting
budgets. No new convergence cutoff or approximate trigonometry is enabled.
Exploratory early termination at 1e-4 Å lost three prior hypotheses and the
compact example4 target; that policy was removed. A polynomial-trigonometry
prototype had no convincing speed benefit and was also removed. Their local
diagnostics remain distinct from the final exact path.

`tests/test_experimental_windowed_fft.py` checks dense/windowed retained scores,
vectors and total counts exactly across grid sizes 16–256, workers, sparse
empty lines, tied amplitudes, narrow shells and radius/Nyquist/wrap boundaries.
Prepared moments match the independent NumPy equations and the original native
path; finite differences verify the gradient and Hessian on both sides of the
dispatch boundary. Caller mutation, empty vectors, invalid shapes/nonfiniteness
and bounded work are tested. The independent deposition/FFT reference remains
`grains_reference.fourier_field`; no new profile equation or external profile
oracle fixture is required.

`benchmarks/grain_broad_discovery.py` compares dense and bounded field/maxima
stages and ordinary/prepared fused moments on 12,000 synthetic tracks and
integer-phase conditional subsets. It asserts exact arrays and reports the
stage scope, including one-time preparation. The local
`fft-broad-discovery-speed/report.html` preserves complete fresh four-arm
inverse outputs, matched before-kernel replay, all rejected experiments,
source/input/validation parity and actual wall times. Initial bounded-spectrum
replays approximately halved the prior complete runtime; overlapping unrelated
CPU jobs make later wall times variable. Neither a stage speedup nor a cached
candidate merge establishes a one-second complete inverse. Physical grain
counts, cell primitivity, geometry/omega calibration and chemistry remain open.

## Remaining discovery work and explicit native ascent

The subsequent `fft-remaining-bottlenecks` experiment keeps all 28 original
accepted bases bit-for-bit, the compact example4raw orientation and both
targeted example5 orientations under the same validation. Four paired fresh
before/after inverses with four discovery workers take 27.5–32.8 seconds before
and 17.5–21.6 seconds after. Across the four arms total wall time falls from
118.97 to 78.36 seconds (34.1%); process CPU time falls from 272.11 to 192.86
seconds. These are single paired observations per arm on a shared host, not
an idle-host latency guarantee. A second paired check of the final guarded
build gives 28.2–30.0 s before versus 17.7–19.5 s after: total wall time
117.42→74.39 s (36.6% less), process CPU time 261.12→179.18 s (31.4% less).
The one-second objective remains unmet.

The exact-path changes are:

* Native packed-support candidate keys reproduce the original strongest pair
  prefix, pair/index tie order, unique increasing triples and lexicographic
  candidate set. Native ranking uses the same integer intersection count,
  caller-supplied NumPy determinant volume and key tie order. NumPy still owns
  the existing length, determinant and SVD condition gates; no candidate budget
  or finite geometry boundary is changed. Only the required valid prefix is
  converted to Python records when a prefix is requested.
* `_refine_many` batches the original **first-cycle** strict residual and
  nonzero-HKL support test. Scalar NumPy inverses and the existing native
  per-point `indexed` arithmetic are retained. A candidate with fewer than six
  observations is already rejected at the first line of the original `_refine`;
  the batched test skips only those doomed calls. Six is not a new grain gate:
  the final minimum of 20 owned training observations, 20 unique rank-three
  HKLs and six selection observations is unchanged. Every survivor still runs
  the original rank check, alternating fits and integer alternatives.
* Oriented competition computes each scalar determinant once and reuses it in
  the model-pair and greedy loops. Its floating convention and choices are
  unchanged.
* `discovery_cache` stores pure Fourier proposals only during a single ensemble
  call. Keys contain the engine type, exact ordered coordinate bytes, shape,
  dtype, grid, radial bounds, peak budget and native-refinement setting. The
  tracked storage budget is 64 MiB with at most 512 entries; native FFT scratch
  and general Python allocator overhead are outside that bookkeeping metric.
  Returns are copied, contexts reset even on exceptions, and no file, prior
  inverse model or validation outcome is read. The four primary runs reuse
  66–70 identical queries, preserving logical proposal order and fitting cohorts.
* Two independent Fourier lanes share point loads for large batches. The
  platform sine/cosine and each lane's ordered value/gradient/Hessian arithmetic
  are unchanged. Bit-pattern comparisons cover odd tails and worker dispatch.

`GrainSearchOptions(native_fourier_refinement=True)` is an explicit alternative
for the proposal ascent. Explicit search objects should inherit that option
from the base options if they are intended to use it too. The ordinary default
continues to use the NumPy ascent. The native alternative uses the same fused
moment equations, twenty-pass cap, four line scales `(1,.5,.25,.125)`, strict
objective improvement, 0.3 Å step cap and inclusive initial-coordinate box
`initial ± 1.5/(n*Delta_q)`. Independent vectors run in the bounded pool. Exact
repeated trial coordinates reuse their fused moments; keys distinguish signed
zeros. No early convergence cutoff, subsampling or approximate trigonometry is
introduced.

For the symmetric lower triangle of `H=-Hessian(C)`, write
`mu=trace(H)/3`, `p=sqrt((sum(diag(H-mu*I)^2)+2*sum(offdiag(H)^2))/6)` and
`r=det((H-mu*I)/p)/2`. The characteristic polynomial gives
`lambda_min=mu+2*p*cos(acos(clip(r,-1,1))/3+2*pi/3)`; diagonal/repeated cases are
handled directly and positive scaling avoids overflow. The step solves
`(H + max(0,.01-lambda_min)*I) delta = gradient(C)` by pivoted three-by-three
elimination, then applies the original norm cap and line search. This differs
from LAPACK in roundoff, so native proposal vectors need not be bit-identical
to the NumPy trajectory. The independent array reference is
`grains_reference.fourier_newton`; fused moments at returned native vectors are
checked against the original evaluator. The kernel bounds at most 100,000
points, 400 initial vectors, coordinates/radius at most 1,000 and 81 logical
moment queries per vector. Invalid/nonfinite arithmetic fails explicitly.

The recorded 12,000-track kernel fixture (2,286 conditional observations) shows
median paired gains of about 3.5x for triplets, 30x for the first support test
and 1.65x for maxima plus refinement. That first-support fixture crosses the
old point-dispatch threshold and is not an end-to-end gain claim. Its maximum
native/NumPy proposal-coordinate difference is about 1.4e-8 Å. The strict
synthetic detector-space control still returns exactly its two known components
in both modes without passing cells or HKLs into discovery or relaxing its
assertions.

The original 28 accepted matrices, all acceptance flags and all accepted track
owners match the previous complete pipeline. In corrected example4, one
supplementary accepted matrix changes by about 4.5e-9 Å and two rejected matrices
move slightly; one all-pool ambiguity changes. Full rejected-pool archive
equality is therefore not claimed for the native option. These differences do
not establish better accuracy. The extra raw example5 DAFi reference remains
unmatched and the second targeted raw orientation still owns only two test
reflections. Geometry/omega, masks, EIGER saturation None, primitive-cell and
space-group interpretation, chemistry and physical grain counts remain open.

`benchmarks/grain_remaining.py` records isolated stages and the strict synthetic
control. Tests cover exact discrete rankings/ties, original geometry prefixes,
strict first-cycle boundaries, native-reference ascent, objective monotonicity,
box bounds, exact fused moments, signed-zero cache keys, worker repeatability,
scope/input/budget isolation and invalid arrays. No new profile term or physical
derivative is added; existing gradient/Hessian finite-difference tests and the
exact moment comparisons remain applicable. A powder-profile oracle does not
cover this positional proposal optimizer. The full source, hashes, saved
pre-change binary, paired runner, before/after outputs, rejected candidates and
profiling diagnostics are preserved in the local report directory.

### Occupied-column FFT and invariant Fourier factors (2026-10-30)

The next speed checkpoint changes storage and invariant arithmetic only. The
common experimental search, all eight configurations, conditional subsets,
256-cubed grids, maxima budgets, Newton passes, fitting and final validation
remain fixed. The native ascent remains explicitly opt-in; no approximate
trigonometry, lower precision, convergence relaxation or proposal filtering is
introduced. No external implementation code or new dependency is used.

For ordered reciprocal points `q_i` and a direct vector `v`, the unchanged
objective and analytical derivatives are

```
C(v)    = (1/N) sum_i cos(2*pi*q_i.v)
g_j(v)  = -(1/N) sum_i (2*pi*q_ij) sin(2*pi*q_i.v)
H_jk(v) = -(1/N) sum_i ((2*pi)^2*q_ij*q_ik) cos(2*pi*q_i.v).
```

The bounded native ascent now prepares the 12 invariant factors once per point
cloud. Factors use the original multiplication order, including separate `jk`
and `kj` entries; per-point multiplications, normalization and ordered fused
accumulations remain unchanged. Values, all three derivatives and all nine
Hessian entries are still computed in the same point pass. The independent
NumPy equations and ascent are unchanged. Tests compare every moment bit with
the original scalar kernel, preserve signed zeros, and retain the existing
finite-difference and independent-reference checks. A paired-ascent prototype
did not improve measured speed and was removed from production.

For cloud-in-cell deposition, `low=floor(q/step)` and fractional coordinates are
prepared once. Only touched XY columns receive storage, in one contiguous
float32 allocation. Contributions still execute in the original
`dx,dy,dz,point` order and use the original weight multiplication/cast/addition;
untouched columns are positive zero. A native test compares every dense grid
voxel bit, not only selected maxima. Actual zero-line transforms supply empty
Z columns and Y slabs, retaining their signed-zero conventions. All retained
lines use the original length and Z-to-Y-to-X transform order. X transforms
whose entire YZ line is strictly outside the spherical maximum are omitted.
Their values never enter the annular maximum extraction.

The score domain remains inclusive `min^2 <= x^2+y^2+z^2 <= max^2` in the
half-spectrum. Neighbors outside that annulus remain negative infinity; XY
neighbors wrap on the original full grid, and Z neighbors clamp to the original
half-spectrum bounds. A small precomputed adjacency table replaces repeated
modulo/clamp arithmetic without changing neighbor visit order, strict `>`
comparison, plateau handling, score/index ordering or total maxima count. Tests
include empty slabs, repeated points, exact planes, narrow windows, even
non-power-of-two grids, ties and bit-pattern comparisons with the original
complete field.

Each engine lazily retains plans for its validated even grid lengths in
`16..=256`; scratch arrays remain call-local and work uses the same bounded
private pool. No point data, Fourier field, model, validation labels or results
are cached across inverses by this plan cache. Public Python boundaries and
work limits are unchanged. The powder-profile oracle does not cover positional
Fourier discovery; the independent first-principles reference and saved
first-party binary replay provide the relevant checks.

`benchmarks/grain_fourier.py` times realistic full and phase-conditioned clouds
and verifies byte-identical maxima, all refined vectors/moments and exact
query/cache counts against an explicitly saved task-start binary. It does not
read recovered cells, comparator groups or acceptance labels. Full four-arm
before/after inverses, immutable validation audits, rejected pools and timing
provenance are recorded separately in the local `fft-fourier-kernel-speed`
report. Kernel timing excludes imports/input loading and is not a complete
inverse timing claim. The one-second complete-inverse target remains a target;
physical grain counts, cell aliases, provisional geometry/omega and chemistry
are unresolved by these computational changes.

### Exact fitting snapshots and bounded ascent memo (2026-10-31)

The next speed checkpoint targets repeated reciprocal-space proposal fitting.
`Engine.prepare_fitting` owns a finite, ordered snapshot of at most 100,000
reciprocal points and shares the engine's bounded worker pool. Its selected-row
method evaluates the original scalar equations

```
h_j = round_ties_even(q.a_j)
e   = sqrt(sum_k (q_k - sum_j h_j*b_jk)^2)
keep = any(h != 0) and e < tolerance.
```

It returns only the selected `h`, original `q`, original row indices and errors,
in input order. All dot products, rounding, residual arithmetic and the strict
cutoff are the original `indexed` operations. Larger clouds use ordered chunks;
no floating reduction is parallelized. Python inputs are copied before releasing
the GIL, and the snapshot is immutable and remains valid after input mutation
or engine disposal. Empty snapshots return correctly shaped empty arrays.
Nonfinite/oversized coordinates, malformed/nonfinite bases and nonpositive or
nonfinite tolerances fail at the boundary.

A snapshot is created once per `_refine_many` cloud. Each surviving proposal
uses selected arrays directly instead of repeatedly copying the full cloud
through the extension boundary and allocating full HKL/error arrays followed
by Python masks. The least-squares solver is still the original NumPy
`lstsq(..., rcond=None)`. The six-row/rank-three checks, eight-cycle cap, first
two doubled reciprocal cutoffs, iteration-dependent `1e-8` convergence test,
inversions, basis reduction, integer-core sizes, stable error ordering,
90-percent overlap and volume/conditioning rules are unchanged. Final ownership,
selection, held-out and null-control validation are unchanged. The old full-score
path remains a readable fallback when a fitting snapshot is unavailable.
`grains_reference.fitting_selection` independently implements the component
arithmetic in NumPy without calling a production/native selector. No new physical
profile or derivative term is introduced; the powder-profile oracle does not
cover this positional storage/selection optimization.

The existing opt-in native Fourier ascent uses the existing mathematical query
bound, `1 + 20*4 = 81`, to store exact trial keys and fused moments in separate
fixed arrays. This eliminates per-vector hash-table allocation and hashing;
lookup visits only populated keys. The bound is derived from the actual
iteration/line-scale constants and does not stop or prune queries. It uses
roughly 10 KiB of stack storage per active worker. Every original query, moment,
improvement test, damping and line-search iteration is retained; signed zeros
remain distinct. Scalar trigonometry and all fused moment arithmetic are
unchanged. Unit tests cover all 81 entries and exact coordinate/moment bits;
saved-binary benchmarks also compare refined vectors and query/cache counts.
Ascent-only timings improve by about one percent in the recorded fixtures,
which is small relative to host-load variation. Large ascent speed gains are
not established by this storage change.

Measured duplicate whole fits were only 136 of 6,764 calls in a diagnostic raw
example4 run, accounting for about 0.06 seconds, so no whole-fit cache was
introduced. A SIMD trigonometry prototype using `wide 0.7.33`, six-factor Hessian
storage and combined point/factor layouts did not provide convincing speed gains.
They are preserved as analysis-only diagnostics and were removed from production;
there is no new production dependency or approximation.

`benchmarks/grain_fitting.py` compares selected-row traffic and complete
80-proposal fitting groups on a deterministic 1,353-point signal/background
cloud, including snapshot setup. It replays the explicitly archived task-start
first-party function/binary, rather than wrapping the new selector as its
reference. Tests cover original-score and independent-equation bit parity,
strict boundaries, ties, noncontiguous row order, empty inputs, ownership,
invalid arrays and complete proposal results including aliases and failures.
Full four-arm inverses and all original hypotheses are checked separately in
`fft-ascent-fitting-speed`; kernel timings are not complete-inverse timings.
Masks, original counts, EIGER saturation None, background, geometry/omega,
chemistry and physical interpretation remain unchanged. The one-second complete
inverse target remains unmet at this checkpoint.

### Ordered parallel ensemble searches (2026-10-31)

`find_grains_ensemble(..., ensemble_workers=8)` opts into bounded spawned
processes. Serial execution remains the default. `ensemble_workers` accepts
integers 1–16; booleans and other values fail. Parallel mode forces the native
search option `workers=1`, including explicit geometric search configurations,
conditioning and subset options. The actual measured configuration uses eight
processes and one native worker per member on the 14-core host. These are worker
budgets, not CPU affinity pinning or a promise of full CPU utilization.

Start Python with `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`VECLIB_MAXIMUM_THREADS=1`. Parallel startup validates these inherited settings
instead of changing the parent environment. Changing these variables after NumPy
has loaded does not change its active BLAS thread pool; parent/worker reductions
can then differ in their last bits (including on NumPy 1.26). See the
[0.8 reproducibility guidance](migration-0.8.md#exact-positional-search-reproducibility).
Use a normal multiprocessing main
guard. `spawn` starts fresh interpreters without forking existing Rayon/native
worker pools. The standard library is the only scheduling dependency.

```python
from phasesmith.single_crystal.grains import find_grains_ensemble

if __name__ == "__main__":
    result = find_grains_ensemble(points, ensemble_workers=8, **fixed_search_inputs)
```

For a complete inverse with an additional recovery stage, the experimental
`search_execution.parallel_searches(8)` context can cover discovery and recovery
once, sharing the same worker pool. Place this context inside the complete
inverse clock. The recorded four-arm runs include cold worker startup/imports,
serialization, shutdown, both training branches, fitting/competition, unchanged
validation and local exports; only parent imports and initial data/geometry
loading remain excluded as in the previous benchmarks. The context is intended
for one inverse; reusing it across inverses would amortize startup and must not
be reported as a cold inverse. The final assessment continues with its original
two-worker budget after the search pool is closed.

Independent geometric searches are submitted in original configuration order.
Full-training and fixed-resample conditioning directions can prepare alongside
them. Prepared views then queue independent subset calculations. This avoids
leaving one process with a long sparse view while other processes sit idle.
Paired recovery calculations also execute independently. These tasks receive
only global-training points and existing geometric options, never reserved
selection/test observations, comparator groups, chemistry or acceptance flags.

Completion order never selects candidates. The parent reads results in the
original configuration/view/seed/pair order and performs the original sorting,
per-seed and global budgets, deduplication and ownership competition. Fixed
resampling uses the original seed and sorted row indices. Numerical subset and
paired calculation blocks, candidate-selection conditions and immutable final
validation functions are audited against the task-start source. No within-dot,
FFT-line, Fourier-moment, least-squares or other floating reduction is reordered.
The native binary is unchanged. Every original 28 matrix and the complete
accepted/rejected assignment archives match bit for bit in all four measured
arms; model records, fit histories, null counts and acceptance flags also match.

The parent owns progress callbacks. Parallel progress is delivered at ordered
collection/stage boundaries; callbacks are not serialized or called from child
processes. Serial conditioning still allows its callback to cancel before the
first numerical subset calculation. Worker failures propagate through futures;
the session resets parent context, cancels queued work and joins running workers
in a finally block. Nested contexts share the budget and cannot silently expand
it. Worker-side calls have no active parent executor, preventing nested pools.

Each process owns a bounded 64 MiB Fourier cache with at most 512 entries. It
contains only pure exact-ordered-input Fourier results and is discarded with
the pool. Across eight workers that is a tracked cache budget of 512 MiB;
interpreter/allocator overhead, native scratch and transferred arrays are
additional. Distribution and cache hits can vary with scheduling, so the
parallel implementation is not claimed to perform less total computation.
Parent and child CPU time are recorded separately, using waited-for child
resource usage so startup and shutdown CPU are included. The measured total
CPU time increases while wall time decreases.

The primary four-arm before/after pairs fall from 15.2–17.3 seconds for the
serial ensemble with four native workers to 8.8–10.6 seconds with eight parallel
single-worker searches, about 40% less aggregate wall time. The one-second
complete-inverse target is not achieved. Initial fine-field scheduling took
about 11.1 seconds on raw example4; coarse whole-view scheduling took about
14.2 seconds. Queuing preparation and subset tasks reduced that diagnostic to
about 9.2 seconds. Ten- and fourteen-process raw4 probes gave little additional
wall-time benefit and used more CPU, so all primary arms use the common eight
process setting. Diagnostic probes are not pooled with the alternating primary
pairs, and no statistical confidence or universal scaling claim is made.

`benchmarks/grain_parallel.py` runs the unchanged strict detector-space
synthetic control in ordinary and native ascent modes with process scheduling.
It must recover exactly two known components without inverse cells/HKLs.
Focused tests compare serial/process results byte for byte for geometric,
conditional, stability and paired searches; check one-worker settings and
inherited limits; preserve callback cancellation; and check exception cleanup
and nested budgets. The full local `fft-parallel-ensemble` report preserves
rejected scheduling experiments, immutable validation audits, serial and
parallel runners, output sources and provenance. EIGER saturation None, masks,
background, original-count noise, geometry/omega assumptions and interpretation
of cells/physical grains are unchanged.

### Parallel fitting and coherent recovery (2026-10-31)

The opt-in process context now also schedules independent proposal polishing,
trimmed-fit variants grouped by parent, initial observation fits, alias-cell
refinement, simultaneous ownership-update fits, final training-only forward
fits, and coherent-parent recovery. The independent reciprocal branch runs
alongside observation fitting. These stages share the existing bounded pool;
children use one native worker and inherit the same single-thread library
limits. No child creates another executor. Serial defaults are unchanged.
`parallel_searches(8, fit_tasks=False)` retains parallel discovery while using
the original serial fitting/recovery trajectory for a direct counterfactual.

Parallel fitting freezes the inputs for each independence barrier. Fits from
one simultaneous ownership cycle all receive the original frozen likelihoods;
the parent joins the cycle before recomputing assignments. Alias entries join
in original order, so overlapping entries see prior updates. Each alias cell's
inner convergence iterations remain sequential. Trimmed variants retain their
original per-parent fit order, followed by the original parent-side global
deduplication; even locally redundant variants must reach that coordinator
because the equivalence threshold is not transitive. Final fits run once and
are reused by unchanged validation, rather than fitting the pool again.

Initial ownership fitting has dynamic admission: previous fitted bases affect
whether a later proposal is skipped. Therefore candidate fits may be computed
speculatively on frozen initial bases, but only the original ordered coordinator
admits results. Unused fit exceptions are deferred and do not affect decisions;
a consumed exception propagates. Speculative results never supply new proposals
or change the fit budget. Batch payloads share the training arrays within each
worker batch rather than serializing them separately for every tiny fit. Some
extra computation and transfer are exchanged for shorter wall time. Execution
statistics join and include unused speculative batches; total CPU timing also
includes all child startup, work and shutdown.

Coherent recovery keeps every trial within a parent in its original order,
including within-parent deduplication. The parent draws samples using the
original single seeded RNG and original eligible-parent order, then dispatches
one independent task per eligible parent. Skipped parents consume no random
draws. Completed parent results are sorted/capped/merged in the original order,
never by completion time. These tasks receive global-training observations
only; they do not receive reference cells, DAFi assignments, chemistry,
selection/test folds, or acceptance flags.

Feedback stages remain ordered: ownership competition, family decisions,
assignment recomputation, training-only removal, and final validation. Parallel
work does not reorder floating reductions within a fit or trial. The native
binary, independent NumPy reference, numerical gates and frozen assessment are
unchanged. Byte-level four-arm regression and the strict two-component control
are reported separately from timing; a process budget is not CPU affinity
pinning. `benchmarks/grain_parallel.py --serial-fits` provides the previous
parallel-search-only counterfactual, with otherwise identical control settings.
The local `fft-parallel-fitting-recovery` report records complete cold inverses,
CPU costs, every retained/rejected model, original 28-hypothesis parity, and the
remaining gap to one second. These development datasets are regression cases,
not an untouched generalization test.
