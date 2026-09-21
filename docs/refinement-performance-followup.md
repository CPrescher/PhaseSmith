# Investigation after the first refinement optimizations

The subsequent [implementation](refinement-performance-round2.md) records which
of these candidates were implemented and their measured results.

Investigation date: 2026-09-16. No production numerical behavior or defaults were
changed in this follow-up. The subject is the optimized release build described
in [the implementation report](refinement-performance.md), using the same real
QARR 1g data and existing scientific gates.

## Findings and priorities

### 1. Optimize FCJ evaluation and finish derivative selection

Three warmed single-worker workflows under cProfile took 2.949 seconds;
2.511 seconds (85%) were inside the native `refine` calls. Moving more Python
orchestration into Rust alone cannot remove the main bottleneck.

A separate native sampling run recorded 3,651 main-thread samples and 70
short-lived worker-thread samples. The two FCJ evaluator variants account for
1,585 exclusive top-of-stack samples, with another 452 in `exp` and 100 in its
call stub. CW accumulation adds 438 samples. Not all exponential calls are
necessarily FCJ calls, but the call tree confirms FCJ convolution dominates.
The small direct linear solver accounts for only 81 samples. Further replacing
the linear solver is therefore a lower priority than improving profile work.

Concrete opportunities in the current code:

- `fcj.rs::evaluate_with_radius` evaluates one sample through all quadrature
  nodes. A batch over adjacent samples could expose vectorization while keeping
  each sample's node accumulation order and closed support checks.
- `tch.rs::tch_pseudo_voigt_from_shape` converts primitive width/mixing
  derivatives to Gaussian/Lorentzian derivatives at every node. Moving constant
  chain-rule factors outside the node sum could reduce arithmetic, but changes
  floating-point ordering and needs explicit derivative/tolerance validation.
- The selected structural path skips fixed axial derivatives, but backtracking
  still calls the full `calculate_rietveld_pattern` path, including those axial
  derivatives. Carrying derivative needs through trial evaluation would remove
  that remaining work while preserving fused values and required derivatives.
- The profile accumulator still creates all instrument/sample-physics rows.
  Propagating selection into those rows, and iterating compact active structural
  rows rather than checking a mask inside each sample loop, would reduce work
  and memory traffic. The payoff is not yet measured independently.

These are implementation candidates, not measured speedup promises. Keep the
existing 8/48-node quadrature rules, support convention and physical model while
testing them. Changing quadrature order or tail support requires a separate
scientific convergence study.

### 2. Cache a profile basis for linear-only stages

The final QARR stage frees only three phase scales. With other quantities fixed,
`y = A s + b`, where columns of A are unit-scale phase profiles. Its weighted
least-squares solution needs one profile basis instead of repeated FCJ profile
evaluations. Bounds and constraints determine whether a simple direct solve is
admissible; a general implementation must retain a safe fallback.

An independent NumPy prototype on the actual stage input computed the basis,
solved the weighted system, and formed scale covariance in **12.71 ms median**.
The current complete stage took **62.30 ms median**. All three solution scales
are positive; their largest relative difference from the native result is
**8.88e-15**, and full calculated-pattern relative L2 difference is **7.11e-15**.
Rwp is 0.19816460010715778, matching the accepted fit to numerical precision.

This is a feasibility result, not a fivefold complete-workflow speedup. The
prototype omits result/checkpoint packaging, cancellation and QPA propagation.
Even removing the entire current final stage would save only about 6% of this
single-worker workflow. Eliminating linear scales/background coefficients from
the nonlinear iterations is a larger possible extension, but needs careful
handling of constraints, bounds, derivatives and covariance.

### 3. Return native phase diagnostics without recalculating the pattern

`python/phasesmith/refinement/rietveld.py::_refine_native` calls `calculate`
after the Rust solve to reconstruct per-phase diagnostics. The native
`RietveldCalculation` already owns per-phase results, but the Python adapter
does not expose all the arrays needed for those diagnostics.

Timing instrumentation recorded approximately 12–17 ms for this calculation
per stage in the single-worker warmup, about 41 ms across the three stages.
Returning the native arrays through PyO3 would avoid duplicate profile work.
It should preserve the complete public diagnostic contract, including wavelength
components. Savings overlap with the linear-stage proposal and must not be added
as independent guaranteed gains.

### 4. Select a useful worker budget at the application boundary

The thread sweep used one warmup and three measured repetitions per setting,
reversing configuration order on alternating rounds. All thread settings return
exactly identical scientific records and pass all gates.

| Requested workers | Median complete workflow |
|---:|---:|
| 1 | 982.49 ms |
| 2 (current default) | 604.88 ms |
| 4 | 433.55 ms |
| 6 | 380.32 ms |
| 8 | 352.72 ms |
| 12 | 339.44 ms |
| 14 | 339.61 ms |

Eight workers reduce time by about 42% relative to the two-worker default on
this host. Moving from eight to twelve buys only another 3.8%. An application
can already request `ExecutionPolicy(threads=8)` for an interactive fit. This
does not establish a universal default: machine topology and concurrent fits
change the appropriate budget. The differing one-worker time from the previous
report reflects a separate timing run, not an additional code optimization.

## Solver experiments that did not provide a safe shortcut

Stage 1 has 31 model evaluations for seven accepted iterations and 11 summed
backtracks on accepted iterations; that sum excludes trials from unaccepted
iterations. Stage 2 has no accepted-step backtracking, but all 28 accepted steps
hit the configured 0.15 scaled step limit. This motivates studying trust-region
scaling, but simply increasing the limit did not pass validation.

| Experimental change | Median workflow | Result |
|---|---:|---|
| Stage 1 initial damping 0.001 | 986.67 ms | Passes, no timing benefit; Rwp worsens to 19.9347% |
| Stage 1 initial damping 1 | 992.76 ms | Passes, no timing benefit |
| Stage 1 initial damping 1000 | 930.83 ms | Fails Rwp gate: 20.6480% > 20% |
| Stage 2 step limit 0.3 | 957.70 ms | Fails termination gate |
| Stage 2 step limit 0.6 | 1160.58 ms | Fails termination gate and takes longer |

The larger-step runs reach an acceptable Rwp but stop through the existing
rejection guard. Their gates were not loosened. None of these settings was
adopted. Better parameter scaling or a trust-region acceptance model remains a
research candidate requiring broader real-data and difficult-start validation.

## Evidence and reproduction

- `benchmarks/investigate_qarr_performance.py`: thread sweep, solver experiments,
  stage histories, diagnostic timing and linear-scale proof.
- `validation/results/rietx-20260916-optimization-investigation.json`: raw timing
  samples, dataset hashes, complete quality checks (including failed variants),
  stage histories and linear-probe agreement.
- `validation/results/rietx-20260916-optimized-profile.txt`: Python profile and
  native top-of-stack summary. Profiling counts guide attribution, not speed
  claims; use uninstrumented workflow measurements for comparisons.
- `validation/results/rietx-20260916-optimized-profile-stages.json`: stage timings
  and evaluation counts from the three cProfile runs.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python benchmarks/investigate_qarr_performance.py --repetitions 3 \
  --json-output investigation.json
```

The script requires a release PhaseSmith build and checksum-pinned local QARR
data. Every configuration must reproduce its scientific record exactly; the
thread variants must also match each other exactly. Experimental solver failures
are retained as evidence, not treated as benchmark successes. The linear probe
requires positive scales and a residual no worse than the native final fit.
