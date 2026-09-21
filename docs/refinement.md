# Refinement infrastructure

Refinement orchestration is migrating behind an application-neutral Rust
workflow layer over immutable domain models and the native calculation API.
The Python scripting API remains the public interface during that migration.
No optimizer state, constraint graph, observed pattern, or iteration history
enters `phasesmith-core`.

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

The same contract now exists in the Python-free `phasesmith-workflows` crate.
Its parameter and constraint fields are constructor-validated and privately
owned, transforms return structured errors for unknown/duplicate/cyclic
dependencies and invalid vectors, and derivative matrices use checked
row-major allocation. A configured differential test compares native packing,
expansion, and exact chain rows with the existing Python implementation. Python
delegation waits until the adjacent residual/background/runtime records are
native, so the current scripting interface and reference remain unchanged.

Nonzero phase-scale parameters use their current magnitude as their numerical
scale. Consequently, refinements remain conditioned when absolute phase scales
are much smaller than one, as is common for measured count data. An exact zero
has no relative magnitude and therefore uses an order-one fallback so it can
move away from the bound.

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

This evaluation is now also owned by `phasesmith-workflows` over the native
`PatternRecord`. It preserves full sample-aligned residual/weighted arrays and
uses structured errors for missing observations, mismatched calculated arrays,
and non-finite calculated values. A configured differential gate compares the
native arrays and metrics with the independent Python implementation.

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

CW Le Bail accepts an optional `DifferentiableBackground` through
`LeBailInput.with_refinable_background`. Its calculated background is always
the supplied fixed array plus the analytical residual. Polynomial, Chebyshev,
and fixed-knot models have coefficient-invariant bases, so their coefficients
are eliminated by a deterministic weighted linear least-squares update after
each intensity redistribution instead of competing in the nonlinear profile
step. Their analytical basis remains available for covariance and
finite-difference checks. Rietveld and TOF use the same additive fixed-plus-
residual convention.

Iteration and termination records are immutable. Every accepted profile step
records typed parameter keys plus before, after, and scaled-change values.
Checkpoints are sufficient for deterministic continuation; their non-pickle
representation is documented with the persistence schema.

## Pawley workflows

[CW Pawley refinement](pawley.md) supports monochromatic and fixed-spectrum
family areas; [TOF Pawley](tof-pawley.md) supports single banks and joint banks
with shared cells. Both provide bounded/tied parameters, analytical derivatives,
dense or matrix-free solving, cancellation and accepted-state restart.
Use `phasesmith.refinement.pawley` or `phasesmith.refinement.tof_pawley` in Python,
and `phasesmith::workflows` in Rust. Standalone CW version 2 and TOF version 1
codecs also integrate into lossless [native format-7 bundles](native-persistence.md).
See [validation](pawley-validation.md) for measured gates and model limitations.
