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
boundaries with bin-width-multiplied Y and sigma. The typed SLOG FXYE reader
converts adjacent boundaries to arithmetic centers and divides Y and sigma by
the bin width, matching the density convention used by the profile kernel and
GSAS-II's scripting `getdata` arrays. The final boundary is not an output
sample; zero-intensity or zero-sigma bins are explicitly excluded. The `PNT`
fixture records and tests this translation explicitly.

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

## Native pattern and Le Bail workflow

`phasesmith-workflows` owns a separate fixed-instrument TOF path built around
`TofPatternRecord`, `TofLeBailPhase`, and `TofLeBailInput`. It does not cast a
microsecond grid into `PatternRecord.x_deg`. Reflections retain d-spacing as
their durable coordinate, and each calculation returns the core accumulation
containing intensity/d-spacing support columns and all 15 dense instrument
rows.

`refine_tof_lebail` performs nonnegative multiplicative redistribution on the
actual, potentially nonuniform, bin-center grid. Trapezoidal cell widths enter
the discrete extraction sums; masks and optional one-sigma uncertainties are
applied locally. The instrument and reflection topology remain fixed during
this workflow.

An optional `TofChebyshevBackground` is a refinable residual added on top of
the fixed pattern background. A Smooth Bruckner estimate can therefore remain
the broad, non-differentiable baseline while the Chebyshev terms absorb smooth
low-order mismatch. On the explicit microsecond domain `[lower, upper]`, define
`u = 2 (tof - lower) / (upper - lower) - 1`; the background is
`sum_k c_k T_k(u)`. Its exact coefficient derivatives are `T_k(u)`. After each
nonnegative intensity redistribution, the coefficients are updated by a
deterministic weighted linear least-squares solve using the same mask and
uncertainty convention as the residual. The final calculation returns the
sample-major basis and combined fixed-plus-residual background. Omitting the model preserves the
previous fixed-background behavior. This is a complete Le Bail
intensity/background extraction capability, not a claim of TOF structural
Rietveld refinement.

Legacy GSAS type-3 instrument files are translated only in the validation
adapter. Its `ICONS` fields map to `difC`, `difA`, `zero`, and an unused value;
legacy records do not supply `difB`. This potentially ambiguous mapping is
verified through the pinned GSAS-II bank import. For profile function 3, the
first three `PRCF11` values map to
`alpha`, `beta0`, and `beta1`; the first two `PRCF12` values map to `sigma1`
and `sigma2`. Unsupported coefficients are explicit zeros. This adapter detail
does not enter the core instrument model.
