# Finger-Cox-Jephcoat axial-divergence profile

This unit implements the axial-divergence geometry of Finger, Cox and
Jephcoat (FCJ), *J. Appl. Cryst.* **27** (1994), 892–900,
[doi:10.1107/S0021889894004218](https://doi.org/10.1107/S0021889894004218).
The normalized convolution and derivative treatment follow the published
formulation and the corrected derivative analysis of Hester, *J. Appl. Cryst.*
**46** (2013), 1219–1220,
[doi:10.1107/S0021889813016233](https://doi.org/10.1107/S0021889813016233).
The implementation is independently derived below; GSAS-II is only a numerical
oracle.

## Geometry and public parameters

Let `b` be the ideal Bragg position in radians `2theta`, `a` an apparent
detector angle, and

```text
s = S / L
h = H / L,
```

where `S` and `H` are the sample and receiving-slit axial half-heights and `L`
is the sample-to-detector radius. Public Python names are
`sample_over_radius` and `detector_over_radius`; both are finite,
dimensionless, and non-negative.

For a normalized absolute axial separation `z`, cone/cylinder geometry gives

```text
a(z, b) = acos(cos(b) sqrt(1 + z^2)).
```

The maximum separation is `s + h`. Consequently the aberration lies between
`b` and

```text
a_limit = acos(cos(b) sqrt(1 + (s + h)^2)).
```

It is on the low-angle side below 90 degrees `2theta` and the high-angle side
above 90 degrees. Inputs for which the `acos` argument leaves `[-1, 1]` are
domain errors.

FCJ equation 6 expresses the angular density through the overlap weight and an
integrable `1/z` singularity at `a=b`. Hester equations 1–5 make the
normalization explicit. We remove the numerical singularity without changing
the model by integrating over `z` rather than `a`.

## Regular height integral

The sample/detector overlap weight is trapezoidal in absolute separation. Set

```text
M = max(s, h)
m = min(s, h)
A = M - m
B = M + m.
```

Changing variables in Hester equation 3 gives the regular geometry factor

```text
g(z,b) = 1 / [sqrt(1+z^2) sin(a(z,b))].
```

For any intrinsic normalized profile `R(x-a(z,b))`, define

```text
N(x) = A integral_0^1 g(A t,b) R(x-a(A t,b)) dt
     + 2m integral_0^1 (1-t) g(A+2m t,b)
                              R(x-a(A+2m t,b)) dt

Z    = A integral_0^1 g(A t,b) dt
     + 2m integral_0^1 (1-t) g(A+2m t,b) dt

y(x) = N(x) / Z.                                           (1)
```

Equation 1 is algebraically the angular FCJ density and its explicit
normalization, but both integration intervals are fixed and all integrands are
regular. `s=h=0` is defined as the symmetric-profile limit.

Deterministic Gauss-Legendre quadrature is applied separately to the flat and
sloping pieces. Splitting at `A` avoids integrating across the weight kink. If
`A=0`, the flat interval has zero measure and is omitted; its one-sided
major/minor derivative terms cancel in the symmetry-averaged equal-height
derivative.

## Analytical derivatives

With `c=cos(b)`, `r=sqrt(1+z^2)`, and `q=sqrt(1-c^2 r^2)`:

```text
da/dz = -c z / (r q)
da/db =  sin(b) r / q.
```

For `f(z,b)=R(x-a(z,b))`:

```text
df/dz = -R_delta da/dz
df/db = -R_delta da/db.
```

Gaussian and Lorentzian component-width derivatives pass through the same
quadrature. The derivatives with respect to `M` and `m` are obtained by
differentiating both `N` and `Z` in equation 1, including their coefficients,
geometry factors, and transformed coordinates. For example, for the flat piece `z=A t`,
`dz/dM=t` and `dz/dm=-t`; for the sloping piece `z=A+2mt`,
`dz/dM=1` and `dz/dm=-1+2t`. These analytical terms are evaluated beside the
profile values and combined as `(dN-y*dZ)/Z`. Detector/sample derivatives are mapped back from major/minor
parameters. At `s=h`, symmetry gives equal partial derivatives; the two
one-sided transformed expressions are averaged to suppress quadrature noise.

When FCJ is composed with CW broadening, the reflection-position derivative
also includes the previously documented angular dependence of Gaussian and
Lorentzian widths.

## Support convention

For intrinsic symmetric support radius `r_profile`, the exact FCJ union support
is

```text
[min(b, a_limit) - r_profile,
 max(b, a_limit) + r_profile].
```

Both endpoints are included. Inside this union, an individual quadrature
component contributes only when its own intrinsic support includes the sample.
Derivatives hold all selected sample and component support sets fixed.

## Quadrature acceptance rule

The committed quadrature study compares candidate orders with a 256-point
independent reference across low/middle/high angle, equal and unequal axial
ratios, narrow/broad TCH widths, values, and every direct derivative. Define
the resolution ratio

```text
rho = abs(a_limit - b) / H,
```

where `H` is the transformed TCH FWHM. Production uses 8 points per non-empty
smooth piece for `rho <= 0.2`, and the conservative 48-point rule otherwise.
The deterministic near-boundary randomized study reaches `rho=0.195` and has
maximum scaled value/derivative error below `3.4e-11`; the recorded fixed
small-span matrix is below `1.5e-8`. The demanding low-angle, equal-height,
narrow-width case remains on the 48-point path: 32 points has maximum scaled
error `2.62e-5`, 48 points has `4.99e-8`, and 64 points has `8.31e-11`.

This selection follows the physically relevant aberration-span/peak-width
scale rather than a fixed order. It is independently derived and more
conservative than the pinned GSAS-II kernel's adaptive span/width heuristic.
Changing the threshold or either order is observable numerical behavior and
requires the same convergence study, oracle comparison, and benchmark review.

## Pinned-oracle interpretation

The pinned GSAS-II #5838 probe offers only the combined `SH/L` parameter. For
comparison it is mapped to equal half-heights, `s=h=(SH/L)/2`, which gives the
same triangular separation density as the published one-parameter form. The
compiled oracle uses a discretized convolution: compared with the converged
integral its sampled area agrees within `4e-6`, while peak-normalized maximum
value differences reach `2.31e-2` and centroid differences reach `1.55e-3`
degrees in the committed low/middle/high matrix. Those fixture-local bounds are
recorded explicitly. Production keeps the converged published model rather
than reproducing oracle discretization artifacts; all direct derivatives are
also checked against centered finite differences of that model.
