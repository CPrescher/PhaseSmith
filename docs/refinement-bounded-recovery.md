# Bounded convergence verification and recovery

The September 2026 consolidation retains the independent Rust and Python
stopping checks from the [recovery investigation](refinement-convergence-recovery.md)
and bounds their search cost. No profile equation, support convention,
physical bound, scientific tolerance, or caller runtime limit changes.

For weighted residuals `r`, scaled Jacobian `J`, gradient `g = J.T r` and
normal matrix `H = J.T J`, a small damped step is only a candidate stop.
A residual-checked effectively undamped solve must give a small step, or
the sum of box-projected diagonal model gains must meet the existing
objective tolerance, before convergence is reported. The independent Python
implementation uses its own NumPy operations and conjugate-gradient solve.

When the check instead identifies material predicted improvement, rank the
box-projected coordinate steps by descending predicted gain, preserving
parameter order for ties. Cap each step with the caller's existing step cap.
Try each ranked direction at factor 1 before moving to factor 1/2, then 1/4,
and so on. The **entire recovery search shares at most `max_backtracks + 1`
trials**, the same allowance as one ordinary line search. The former
experiment spent that allowance separately on every coordinate.

Only an actual decrease greater than
`objective_tolerance * max(current_objective, 1)` accepts a recovery trial.
If that bounded search finishes without an accepted step, report `stagnated`.
This means the limited search found no material improvement, not that every
direction was exhausted or constrained stationarity was certified. Caller
evaluation, rejection and cancellation guards remain authoritative: if one
fires first, retain its actual termination reason. Recovery does not reset
the rejection counter or reserve a larger budget.

The canonical QARR validator accepts stagnation under its existing bounded-
workflow contract; it does not require every stage to converge. The repair
therefore preserves the distinction between scientific acceptance and
optimizer convergence. No validator acceptance rule is changed.

Tests cover excessive damping, tiny user caps, a bound optimum, misleading
small objective changes, exact checkpoint continuation, multiple coordinate
directions with a shared trial allowance, and a smaller rejection ceiling that
must still report `repeated_rejections`. The real-data integration suite now
also calls QARR 1g without an execution argument, exercising the previously
missed canonical native recipe.

The fixed eight-case driver in `benchmarks/refinement_assessment.py` measures
whole multi-peak workflows, stop reasons, fit/composition gates and evaluation
counts. Retained before/after results and the final validation record belong
to the repository consolidation report. Historical long-budget results in the
earlier recovery document describe the old per-coordinate search, not this
bounded algorithm.

The feasible-width active-set experiment is preserved on
`codex/feasible-width-experiment-20260917`; it is not part of this integrated
solver API. Its option never appeared in the committed develop public API.
Existing committed project and checkpoint formats are unchanged. Experimental
artifacts using that option remain usable with their preserved experimental
branch, rather than silently acquiring different solver semantics here.

No new GSAS-II fixture is needed for this orchestration change: the underlying
profile values, fused derivatives and finite support are unchanged. Existing
kernel/reference/oracle fixtures remain regression controls.
