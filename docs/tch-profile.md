# Thompson-Cox-Hastings width transform

This module implements the pseudo-Voigt width approximation introduced by
P. Thompson, D. E. Cox and J. B. Hastings, *J. Appl. Cryst.* **20** (1987),
79–83, [doi:10.1107/S0021889887087090](https://doi.org/10.1107/S0021889887087090).
The coefficients and FWHM convention are also tabulated in the International
Tables for Crystallography, section 3.3.3, equations 3.3.10–3.3.11.

The implementation was derived from these equations. GSAS-II is used only for
black-box numerical validation and is not a source of implementation code.

## Width and mixing equations

Let `g >= 0` and `l >= 0` be the Gaussian and Lorentzian component full widths
at half maximum in the same coordinate unit, with at least one width positive.
Define

```text
P = g^5
  + a1 g^4 l
  + a2 g^3 l^2
  + a3 g^2 l^3
  + a4 g l^4
  + l^5

(a1, a2, a3, a4) = (2.69269, 2.42843, 4.47163, 0.07842)
H = P^(1/5)
r = l / H
eta = c1 r - c2 r^2 + c3 r^3
(c1, c2, c3) = (1.36603, 0.47719, 0.11116).
```

`H` and `eta` parameterize the normalized symmetric primitive documented in
[`equations.md`](equations.md). The canonical internal component-width unit is
FWHM. A Gaussian standard deviation is converted explicitly as
`g = sqrt(8 ln 2) sigma`; no API accepts an ambiguously named Gaussian width.
For floating-point evaluation, both component widths are divided by their
maximum before the fifth-order polynomial is evaluated and the resulting `H`
is rescaled. Homogeneity makes this algebraically identical while preventing
intermediate overflow and underflow.

## Analytical transform derivatives

The polynomial partial derivatives are

```text
dP/dg = 5 g^4 + 4 a1 g^3 l + 3 a2 g^2 l^2 + 2 a3 g l^3 + a4 l^4
dP/dl = a1 g^4 + 2 a2 g^3 l + 3 a3 g^2 l^2 + 4 a4 g l^3 + 5 l^4

dH/dg = (dP/dg) / (5 H^4)
dH/dl = (dP/dl) / (5 H^4).
```

For `r = l/H`,

```text
dr/dg = -r/H * dH/dg
dr/dl = (1 - r dH/dl) / H
deta/dr = c1 - 2 c2 r + 3 c3 r^2
deta/dg = deta/dr * dr/dg
deta/dl = deta/dr * dr/dl.
```

If `p(delta; H, eta)` is the primitive profile, the component-width
derivatives evaluated in the same sample pass are

```text
dp/dg = dp/dH * dH/dg + dp/deta * deta/dg
dp/dl = dp/dH * dH/dl + dp/deta * deta/dl.
```

The finite support is selected from the transformed `H`, includes samples on
both boundaries, and is held fixed while differentiating.

## GSAS-II oracle conversion

The pinned `getdPsVoigt` probe accepts Gaussian variance in centidegrees
squared and Lorentzian FWHM in centidegrees, while returning density per
centidegree. For a public Gaussian FWHM `g` and Lorentzian FWHM `l` in degrees,

```text
sig_gsas = (100 g / sqrt(8 ln 2))^2
gam_gsas = 100 l

d(profile_per_degree)/dg = 100 dF/dsig_gsas * 20000 g / (8 ln 2)
d(profile_per_degree)/dl = 100 dF/dgam_gsas * 100.
```

The pinned probe's reported position derivative has the opposite sign from a
derivative with respect to peak position, so the fixture stores `-100 dF/dpos`.
These conversions live only in the oracle generator and tests.
