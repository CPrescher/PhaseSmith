# Experimental FFT grain search

`phasesmith.experimental.grains.find_grains(q, training=mask, options=...)`
uses **unindexed reciprocal peak positions** (NumPy `(N,3)`, inverse Å, no
2π factor). It infers integer HKLs, direct lattice bases and competing grain
orientations. It accepts no input cell, HKLs, material, composition or symmetry.
Several orientations can describe grains of the same cell. This is not a space
group solver, a chemical identification or an exact DAFi implementation.
For primitive hypotheses and all 14 Bravais metric types after discovery, see
[experimental cell comparison](experimental-cell-comparison.md).
The feature remains opt-in (`experimental-spots`), outside default releases.

## Ownership and provenance

The FFT deposition/transform, maxima and nearest-lattice ownership kernels are
in `phasesmith-spots`, exposed through private native bindings. The array-only
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

## Cost and validation

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
