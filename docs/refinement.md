# Refinement infrastructure

Refinement is a Python orchestration layer over immutable domain models and the
native calculation API. No optimizer state, constraint graph, observed pattern,
or iteration history enters `phasesmith-core`.

## Parameters and constraints

`ParameterKey(module, owner_id, name)` is the durable identity of a scalar.
`ParameterSpec` adds its physical unit, current value, closed bounds, numerical
scale, and refine flag. `ParameterSet` preserves explicit order, so packing is
deterministic and never depends on dictionary order or a generated variable
name.

`ConstraintTransform` maps scaled free values to physical values. It supports:

- `FixedConstraint(target, value)`;
- `AffineConstraint(target, source, multiplier, offset)`, representing
  `target = multiplier * source + offset`;
- `LinearConstraint(target, terms, offset)`, representing a multi-source sum
  suitable for composition and shared-parameter relationships.

Dependent constraints are ordered and acyclic. Their Jacobian is exact; bounded
finite perturbations are not used to discover the chain rule.

## Residual convention

`evaluate_residuals` defines residual as `calculated - observed`. A pattern mask
uses `True` for included samples. When uncertainty is present, it is one
standard deviation and the weight is `1 / uncertainty**2`.

For included samples `i`, the reported quantities are

```text
Rp   = sum(abs(ycalc_i - yobs_i)) / sum(abs(yobs_i))
Rwp  = sqrt(sum(w_i (ycalc_i-yobs_i)^2) / sum(w_i yobs_i^2))
chi2 = sum(w_i (ycalc_i-yobs_i)^2)
```

Reduced chi-square uses `included_sample_count - free_parameter_count` degrees
of freedom. Values are fractions, not percentages.

## Hybrid Jacobian products

`jacobian_vector_product` and `transpose_jacobian_vector_product` operate
directly on the support-block local derivatives plus dense shared rows. They do
not materialize the full reflection-by-parameter-by-sample tensor. Tests compare
both products with a dense reconstruction and verify the adjoint identity.

## Optimizers

`LeastSquaresOptimizer` is the small protocol for a linearized bounded step.
The built-in Le Bail profile update is dependency-free and uses damped normal
equations with a least-squares fallback and backtracking. The optional
`ScipyLeastSquaresAdapter` lazily imports `scipy.optimize.lsq_linear`; install it
with `pip install 'phasesmith[refinement]'`. Objective calculation remains
independently callable. Pass an adapter as `lebail.refine(..., optimizer=adapter)`;
the solver receives the weighted linearized residual, analytical Jacobian, and
physical-bound-aware scaled step limits.

Iteration and termination records are immutable. Every accepted profile step
records typed parameter keys plus before, after, and scaled-change values.
Checkpoints are sufficient for deterministic continuation; their non-pickle
representation is documented with the persistence schema.
