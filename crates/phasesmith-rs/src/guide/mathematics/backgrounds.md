# Background models

`PhaseSmith` distinguishes non-differentiable preprocessing from backgrounds
that participate in refinement.

## Smooth Bruckner preprocessing

[`crate::core::smooth_bruckner`] operates on an intensity vector `y` with
half-window `N` and a fixed iteration count. It first creates a work vector
padded by `N` copies of each endpoint. Each iteration forms a moving mean over
`2N+1` samples and applies the clipping update

```text
y_i <- min(y_i, moving_mean_i).
```

The implementation preserves the pinned compatibility scan range exactly; the
last `2N+2` returned samples are not clipped by that scan. Because the update
branches at `y_i = moving_mean_i`, this estimator is non-linear and
non-differentiable at clipping decisions. It is preprocessing, not a
refinement parameter family.

## Normalized coordinate

Refinable polynomial and Chebyshev models map a closed input domain
`[x_min,x_max]` to

```text
t(x) = 2(x-x_min)/(x_max-x_min) - 1,     -1 <= t <= 1.
```

[`crate::workflows::PolynomialBackground`] uses

```text
b(x) = Σ_(k=0..n) c_k t(x)^k
∂b/∂c_k = t(x)^k.
```

[`crate::workflows::ChebyshevBackground`] uses

```text
T_0(t)=1,   T_1(t)=t,   T_(k+1)(t)=2tT_k(t)-T_(k-1)(t)
b(x) = Σ_(k=0..n) c_k T_k[t(x)]
∂b/∂c_k = T_k[t(x)].
```

Their basis matrices are invariant while the grid/domain stays fixed and may
be cached across optimizer trials.

## Point background

[`crate::workflows::PointBackground`] linearly interpolates fixed ordered
knots `(x_k,c_k)`. Between adjacent knots,

```text
u = (x-x_k)/(x_(k+1)-x_k)
b(x) = (1-u)c_k + u c_(k+1).
```

Values outside the knot domain are constant at the nearest endpoint. The
coefficient basis contains the corresponding interpolation weights.

## Area-normalized amorphous component

For area `A`, center `μ`, FWHM `H > 0`, `Δ=x-μ`, and `q=4ln(2)`,
[`crate::workflows::AmorphousPeak`] uses

```text
g(x) = sqrt(q/π)/H * exp[-q(Δ/H)²]
b(x) = A g(x)

∂b/∂A = g(x)
∂b/∂μ = b(x) 2qΔ/H²
∂b/∂H = b(x)[-1/H + 2qΔ²/H³].
```

The infinite-domain area is `A`. Its derivative basis depends on the current
center and width and must be rebuilt when either changes.

## Composition

[`crate::workflows::CompositeBackground`] is an ordered additive model:

```text
b_total(x) = Σ_j b_j(x)
∂b_total/∂p_j = ∂b_j/∂p_j.
```

Component IDs and parameter keys keep derivative columns stable. The fixed
background already stored in [`crate::model::PatternRecord`] is added once and
has no refinement column.
