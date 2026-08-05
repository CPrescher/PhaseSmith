# Symmetric pseudo-Voigt equations

Let `d = x - position`, `H > 0` be the full width at half maximum, and
`0 <= eta <= 1`. The first vertical slice uses the normalized components

```text
G(d, H) = sqrt(4 ln(2) / pi) / H * exp(-4 ln(2) d^2 / H^2)
L(d, H) = 2 / (pi H) / (1 + 4 d^2 / H^2)
p(d, H, eta) = eta L + (1 - eta) G.
```

Both `G` and `L` have unit integral over the real line and reach half their
central height at `abs(d) = H / 2`. Their mixture therefore has FWHM `H`.

With `a = 4 ln(2)`, `z = d/H`, and `q = 1 + 4 z^2`, the derivatives are

```text
dG/dd = G * (-2 a d / H^2)
dL/dd = L * (-8 d / (H^2 q))
dG/dH = G/H * (-1 + 2 a z^2)
dL/dH = L/H * (-1 + 8 z^2/q)
dp/deta = L - G.
```

For a peak contribution `y = intensity * p(x - position)`, the parameter
derivatives are

```text
dy/dintensity = p
dy/dposition  = -intensity * dp/dd
dy/dH         =  intensity * dp/dH
dy/deta       =  intensity * (L - G).
```

Evaluation is restricted to `abs(d) <= support_fwhm * H`. This deliberately
truncates the normalized infinite-domain function without renormalizing it.
Consequently, `intensity` denotes the infinite-support integrated intensity,
while the sampled pattern contains the analytically predictable in-window
fraction. The derivative treats the selected samples as fixed; distributional
derivatives at the moving cutoff are outside the API contract.

## Derivative storage

The calculated pattern remains dense. Per-peak derivatives are stored only for
the inclusive active interval found by two binary searches. For peak `p`,
`starts[p]` is its first active sample and `offsets[p]:offsets[p + 1]` selects
sample-major rows in the order `(intensity, position, H, eta)`. Empty and
entirely out-of-grid supports have zero-length blocks. Dense
`(peak, parameter, sample)` storage is an explicit compatibility conversion.
