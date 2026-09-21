# Pawley feasibility at small steps

The 2026-09-21 Intel wheel failure was reproduced with an x86-64 release build
running under Rosetta. The dense fit at a coupled Lorentzian width boundary
returned `stagnated`: 99 evaluations, 85 rejected backtracks and 74 physically
infeasible trials. Its projected-gradient norm remained above the existing
convergence threshold. ARM64 rounding happened to avoid the same trajectory.

## Equation and numerical convention

The bounded linear subproblem minimizes the augmented least-squares norm in
column-normalized step coordinates `d`, subject to `A d >= b`. Composed widths
remain physical inequalities; signs of individual X/Y coefficients do not
substitute for them. In a row `a`, numerical feasibility now allows at most

```text
roundoff(a, b, d) = 1e-13 * (abs(b) + norm(a) * norm(d))
```

The old allowance included an absolute unit floor. For a step of norm 1e-8,
it could admit outward width motion of order 1e-14 even when the current width
was smaller. Trial evaluation correctly rejected the resulting negative width,
but repeated rejections increased damping and caused stagnation. Removing the
unit floor keeps the same relative coefficient and scales the allowance with
the subproblem. The full norm accounts for null-space factorization roundoff;
using only a row's scalar products was too strict for coupled solves.
Constraint-row norms are cached for each active-set solve, and step norms are
evaluated once per unchanged vector. This avoids repeated norm calculations
inside each constraint-row loop without changing the arithmetic convention.

Any negative motion toward a violated constraint can now identify a blocking
face; the former absolute `-1e-14` motion cutoff is removed. This does not change
physical validation, convergence thresholds, model equations, derivatives,
finite support, oracle fixtures or caller runtime budgets. Stagnation is never
relabelled convergence. The accepted solution must still pass the existing
undamped projected-gradient or other documented convergence certificate.

## Regression and independent checks

An analytic projection test covers `min ||d-(s,-2s,z)||` subject to `d0+d1>=0`.
Its exact solution is `(1.5s,-1.5s,z)`. Scales from 1 to 1e-20 expose the old
absolute floor. The existing independent NumPy objective, derivative and
exhaustive nonnegative-face comparisons continue to check the solver against
a separate implementation. No new profile term needs a reference equation or
oracle golden; only subproblem feasibility bookkeeping changes.

The Intel reproduction now converges in five evaluations, with zero infeasible
trials, Rwp about 7.15e-17 and projected-gradient norm about 7.95e-15. The original
boundary test retains its thresholds and additionally requires the actual
projected-gradient certificate. The benchmark supports `--boundary`, adding a
256-reflection, 10,001-sample joint X/Y boundary case to its existing panel.
Full suite, measured data, benchmark and distribution evidence is recorded in
the consolidation review after verification.
