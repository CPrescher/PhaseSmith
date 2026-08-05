# Neutron time-of-flight profile

The TOF layer maps reflection d-spacing to a bin-center time coordinate and to
four peak-shape quantities, then convolves the normalized TCH pseudo-Voigt with
normalized back-to-back exponentials. Public coordinates and widths are
microseconds; d-spacing is ångströms; exponential rates are inverse
microseconds.

The physical profile family follows R. B. Von Dreele, J. D. Jorgensen and
C. G. Windsor, *J. Appl. Cryst.* **15** (1982), 581–589,
[doi:10.1107/S0021889882012722](https://doi.org/10.1107/S0021889882012722).
The modern calibration form and its parameter roles are independently stated
in equations 5–9 of A. Huq *et al.*, “POWGEN: rebuild of a third-generation
powder diffractometer at the Spallation Neutron Source,” *J. Appl. Cryst.*
**52** (2019), 1189–1201,
[doi:10.1107/S160057671901121X](https://doi.org/10.1107/S160057671901121X).
The TCH component-width transform is documented separately in
[`tch-profile.md`](tch-profile.md).

GSAS-II is not an equation source. Its pinned executable behavior validates the
coefficient translation, profile values, derivatives, and bin-coordinate
convention.

## Calibration and broadening equations

For d-spacing `d > 0`, the compatibility model is

```text
position = zero + difC d + difA d^2 + difB / d
alpha    = A / d
beta     = beta0 + beta1 / d^4 + betaq / d^2
sigma^2  = sigma0 + sigma1 d^2 + sigma2 d^4 + sigmaq d
g        = sqrt(8 ln 2) sqrt(sigma^2)
l        = Z + X d + Y d^2.
```

`g` and `l` are Gaussian and Lorentzian component FWHMs passed to the TCH
transform. The model requires positive `difC`, `alpha`, `beta`, and `sigma^2`,
plus non-negative `l`; invalid derived values are errors rather than clamps.

The `sigmaq d` term is an explicitly named GSAS-II compatibility convention.
Some published instrument models instead use a reciprocal-d term (for example
equation 7 of the POWGEN paper). A future published-law variant must therefore
be a distinct typed instrument model; `sigmaq_us2_per_angstrom` will never be
silently reinterpreted.

All derivatives of these equations with respect to d-spacing and all 15
instrument coefficients are evaluated analytically. Their stable global order
is

```text
zero, difc, difa, difb, alpha, beta0, beta1, betaq,
sigma0, sigma1, sigma2, sigmaq, x, y, z.
```

## Normalized asymmetric primitive

Let `q(u; g, l)` be the normalized TCH profile and let `s = x - position`.
The normalized back-to-back exponential density is

```text
b(v) = alpha beta / (alpha + beta) * exp(alpha v),  v < 0
     = alpha beta / (alpha + beta) * exp(-beta v), v >= 0.
```

The profile is the convolution

```text
p(s) = integral b(v) q(s - v; g, l) dv.
```

For deterministic finite tails, the implementation substitutes a dimensionless
exponential coordinate `t`, truncates it at `0 <= t <= L`, and renormalizes by
`1 - exp(-L)`. The defaults use `L = 20`. This gives left and right mixture
weights `beta/(alpha+beta)` and `alpha/(alpha+beta)` respectively.

The standalone primitive uses eight fixed panels of 48-point Gauss–Legendre
quadrature. In the fused finite-support kernel, four 48-point panels are mapped
onto each exact active integration interval rather than masking fixed nodes.
The independent Python reference uses NumPy-generated high-order Gauss–Legendre
nodes and does not call Rust.

Profile value plus derivatives with respect to position, alpha, beta, Gaussian
FWHM, and Lorentzian FWHM are accumulated from the same quadrature evaluations.
The fused reflection kernel chains those direct derivatives to d-spacing and
all instrument coefficients in the same sample pass.

## Finite support and bins

For transformed TCH FWHM `H`, `R = support_fwhm H`, and exponential cutoff
`L`, a reflection has observable support

```text
left  = position - R - L / alpha
right = position + R + L / beta.
```

Grid samples exactly on either boundary belong to the reflection's support
block. The TCH factor itself is limited to `abs(delta) <= R`. Analytical
derivatives hold the active support set fixed; finite-difference tests avoid
moving boundaries.

The public `x_us` array always contains bin centers. GSAS-II can ingest TOF bin
minima, but its scripting `getdata("X")` result contains the centers used for
calculation; the `PNT` fixture records and tests that translation explicitly.

## Local derivatives and oracle scope

Local support-block columns are integrated intensity and d-spacing. Shared
instrument rows are dense. This is the same `AccumulationResult` contract as
constant-wavelength calculations, so refinement code does not need a TOF-only
Jacobian representation.

The pinned `tof_v1` fixture contains public `X`, `Ycalc`, background, and an
18-column TOF reflection list. Private probes are limited to selected
`getEpsVoigt` values and derivatives. The production code is independently
implemented from the equations above and compared numerically; GSAS-II is not a
runtime dependency and no GSAS-II implementation code is copied.
