# Count-free Patterson/dual-space phasing

The [29 October follow-up](solution-dual-space-handoffs.md) fixes the BiSeI origin
search and adds early full-structure handoffs. Results below retain the original
28 October implementation as a recorded baseline.

The shared element-only solver can phase without a supplied composition or
atomicity-mask atom count. A data-adaptive selection replaces the fixed-N mask.
The existing fixed-count dual-space path remains separate and reproducible.

For density rho on N grid nodes, compute RMS with scaled squares to avoid
overflow. Keep positive periodic local maxima with height >= tau * RMS. Local
maxima use the existing 26-neighbor test and deterministic index ties. If more
than `dual_peak_cap` maxima qualify, multiply tau by 1.5 until the pool fits.
The cap is a resource bound, not an input or inferred full-cell composition.
No peak is invented if the threshold leaves none. Refined peak positions use
the existing three-axis parabolic interpolation.

Construct the existing Cartesian unit-volume Gaussian atomicity mask from the
selected peaks. Its finite support is squared distance <= (3 sigma)^2 times
(1+1e-12), including the boundary; sigma defaults to 0.35*d_min. Deterministic
10% omission applies every third cycle. The hybrid Fourier coefficients are
`m Go - (m-1) Gc`, m=3, with `Go = sqrt(E Fo)` and equal-element sharpening
`E = Fo/sqrt(sum_e f_e(s)^2)`. Thus no atom counts enter normalization or masking.
Patterson minimum-superposition initialization uses the mixed-amplitude squared
map and excludes centring vectors, retaining the existing explicit random
fallback if no usable vector exists. Different starts use ranked Patterson
vectors with a deterministic random start every fourth attempt.

Count-free and fixed-count projections use the same Gaussian value kernel.
Adaptive maxima selection is a discontinuous search projection, not a fitted
profile term; no derivative of the selection is claimed. Its Gaussian support,
normalization/moments, complete Fourier cycles and deterministic omission are
checked against an independent dense NumPy implementation. No structure-factor
or refinement Jacobian changes. GSAS-II exposes no equivalent phasing projection
and supplies no oracle fixture. Published dual-space mechanisms and qualifications
are documented in `docs/solution-dual-space.md`.

`SolutionOptions(phasing="dual_space")` selects the new path in element-only
search; `"charge_flip"` keeps the existing phasing and `"portfolio"` alternates
within the same total start/time budgets. Charge flipping stays the default.
Shared controls include `dual_peak_sigma` and `dual_peak_cap`. Every adaptive
trial records requested/effective thresholds, actual selected/omitted counts,
Patterson vector or fallback, cycle history and selected map cycle. Counts are
inferred later by the unchanged shared orbit search and assignment/refinement.

## Validation and interpretation

The RMS selector and its overflow policy are explicit PhaseSmith heuristics,
not a reconstruction of SHELXT's atom-count inference. The full-cell atom bound
only enters downstream orbit search. A regression changes that bound from 2 to
32 and requires identical phasing histories and adaptive-selection records.
Zero/all-negative maps produce no mask peaks; a tied plateau at exactly the
threshold is included, and tied peaks can disappear together as the cap raises
the threshold. There is deliberately no hidden fallback to a supplied count or
to charge flipping.

The independent reference uses vectorized periodic comparisons, dense Cartesian
Gaussian masks and NumPy FFTs. Tests check the inclusive threshold, cap raising,
scale invariance, invalid/nonfinite inputs, zero maps, omission and complete
cycles. Long random-start comparisons include the cycle-50 selection boundary;
Patterson starts also have complete-cycle comparisons. Symmetric Patterson maps
can contain equal-height peaks whose ordering separates at FFT round-off;
omission then makes long trajectories diverge. Cross-backend comparisons avoid
that discontinuity rather than relaxing tolerances. Repeated native runs must
remain bitwise deterministic. Gaussian finite-support, volume and moment tests
from the fixed-count projection still apply to the shared value kernel.

Synthetic P1 and P-1 Fe/O structures solve on the first Patterson start. The Pm
Fe/O case requires the four-start schedule (three ranked Patterson starts, then
one random start) to recover Fe2 O3. This is bounded coverage, not a general
count-inference validation. Different phasing histories use different score
functions and amplitude conventions; compare final fits against original
observations, not charge-flipping and dual-space history values directly.

## Measured comparison, 28 October 2026

`benchmarks/solution_count_free_comparison.py` pins the same three observation
files, cells, exact space groups and element types as the previous shared-search
comparison. Both strategies use seed 0, 800 cycles, one total start, a 60 s total
budget, R1 target 0.06, staged refinement and integrated-density assignment. Three
serial paired repetitions alternate method order. All solves and their reports
are saved before opening reference formulas or coordinates. No supplied atom
counts enter either strategy. These are reused development cases, not a held-out
coverage survey. The failed case is retained in the report.

| Case | Charge-flipping median | Count-free dual-space median | Correct structure with dual-space? |
|---|---:|---:|---|
| K[BrF4] | 1.719 s | 2.547 s | Yes, all three repetitions |
| BiSeI | 0.993 s | 0.484 s, failed attempt | No, all three attempts rejected |
| Thiourea, non-H atoms | 5.583 s | 6.001 s | Yes, all three repetitions |

K[BrF4] infers Br4 F16 K4, R1 0.0222664 and maximum element-matched coordinate
error 0.005909 angstrom. Thiourea infers C4 N8 S4, R1 0.0494090 and maximum error
0.010372 angstrom. Their final R1 and geometry agree with charge flipping within
paired gates (R1 increase <=1e-7, maximum-error increase <=1e-4 angstrom).
All 30 CIFs from the 15 successful solves reproduce their candidate R1 through
the general structure-factor API within 2e-10, in both exported coordinate frames.
The BiSeI charge-flipping result is Bi4 I4 Se4, R1 0.0522851, error 0.002516
angstrom. The dual-space map has no initial density prefix matching Pnma within
the declared symmetry tolerance; no complete or partial structure is claimed.
Its shorter failure time is explicitly excluded from solve-speedup metrics.

For the successful cases dual-space costs 1.48 and 1.07 times as much wall time,
respectively. This version therefore does not improve demonstrated speed or
coverage, and charge flipping remains the default. A separate four-start check
also records failures instead of silently changing phasing method. Diagnostic
threshold explorations at 0.5, 1, 1.5, 3 and 4 RMS also failed to initialize BiSeI;
no case-specific threshold was selected for the frozen comparison.

The frozen primary report is
`benchmarks/structure-solution-count-free-phasing-macos.json`; saved full outputs
are under `runs/count-free-phasing-comparison-final/`. The report includes native
and implementation hashes, options by method, actual peak-count/threshold traces,
all rejected starts, count/geometry audits and both CIF checks. The selected mask
peak count is observable and distinct from the final inferred composition: e.g.
K[BrF4]'s adaptive mask spans 8–108 peaks while the final structure has 24 atoms.

The saved four-start comparison
`benchmarks/structure-solution-count-free-four-starts-macos.json` gives the same
coverage: BiSeI rejects all three Patterson vectors and its random start, taking
1.881 s total. Both strategies receive the same four-start ceiling and stop
when their target is met. K[BrF4] and thiourea each still stop after one start.

A separate realistic multi-peak kernel benchmark uses a 32^3 grid, 2,108
reflections and 50 cycles. Criterion estimates 59.32 ms for adaptive dual-space
versus 27.68 ms for charge flipping with no cleanup (2.14x kernel cost), recorded
in `benchmarks/structure-solution-count-free-kernel-macos.json`. This benchmark
measures computation only; it does not establish structure recovery.

Enable the strategy through the existing experimental API:

```python
result = solve_structure_elements(
    observations, cell=cell, space_group=group, elements=elements,
    options=SolutionOptions(phasing="dual_space", trials=4),
    dual_peak_sigma=2.0, dual_peak_cap=128,
)
```

Validation for this change: 1,361 Python tests passed (11 external-data skips,
34 deselections), 388 Rust tests passed (34 ignored), strict workspace Clippy,
Cargo formatting and Python Ruff checks passed. A separately built default wheel
imports without SciPy and exposes none of the experimental solver symbols. No
release, push, external solver runtime or GSAS-II dependency was introduced.
