# Peak profiles and instrumental broadening

The equations on this page are implemented by [`crate::core`]. Fused batch
functions evaluate values and analytical derivatives in the same sample pass.

## Normalized symmetric pseudo-Voigt

Let `Δ = x - μ`, where `μ` is the peak position, `H > 0` the full width at half
maximum, and `0 <= η <= 1` the Lorentzian fraction. The normalized Gaussian,
Lorentzian, and mixture are

```text
G(Δ,H) = sqrt(4 ln(2)/π) / H * exp[-4 ln(2) Δ²/H²]
L(Δ,H) = 2 / [π H (1 + 4 Δ²/H²)]
p(Δ,H,η) = (1-η) G(Δ,H) + η L(Δ,H).
```

Both components integrate to one on the real line and have FWHM `H`. With
`a = 4 ln(2)`, `z = Δ/H`, and `Q = 1 + 4z²`, the direct derivatives are

```text
∂G/∂Δ = G (-2a Δ/H²)          ∂L/∂Δ = L [-8Δ/(H²Q)]
∂G/∂H = G/H (-1 + 2az²)       ∂L/∂H = L/H (-1 + 8z²/Q)
∂p/∂η = L - G.
```

For integrated intensity `I`, `y = I p(x-μ,H,η)` and

```text
∂y/∂I = p
∂y/∂μ = -I ∂p/∂Δ
∂y/∂H =  I ∂p/∂H
∂y/∂η =  I (L-G).
```

See [`crate::core::symmetric_pseudo_voigt`], [`crate::core::accumulate_batch`],
and [`crate::core::Peak`].

## Thompson–Cox–Hastings transform

[`crate::core::TchShape`] converts Gaussian and Lorentzian component FWHMs
`g,l >= 0` to the pseudo-Voigt `H,η`:

```text
P = g⁵ + a1 g⁴l + a2 g³l² + a3 g²l³ + a4 gl⁴ + l⁵
(a1,a2,a3,a4) = (2.69269, 2.42843, 4.47163, 0.07842)
H = P^(1/5)
r = l/H
η = c1 r - c2 r² + c3 r³
(c1,c2,c3) = (1.36603, 0.47719, 0.11116).
```

A Gaussian standard deviation is converted explicitly as
`g = sqrt(8 ln(2)) σ`. The transform derivatives are

```text
∂P/∂g = 5g⁴ + 4a1g³l + 3a2g²l² + 2a3gl³ + a4l⁴
∂P/∂l = a1g⁴ + 2a2g³l + 3a3g²l² + 4a4gl³ + 5l⁴
∂H/∂g = (∂P/∂g)/(5H⁴)        ∂H/∂l = (∂P/∂l)/(5H⁴)
∂r/∂g = -(r/H) ∂H/∂g         ∂r/∂l = [1-r(∂H/∂l)]/H
∂η/∂r = c1 - 2c2r + 3c3r².
```

The component-width profile derivatives follow the chain rule

```text
∂p/∂g = (∂p/∂H)(∂H/∂g) + (∂p/∂η)(∂η/∂g)
∂p/∂l = (∂p/∂H)(∂H/∂l) + (∂p/∂η)(∂η/∂l).
```

## Constant-wavelength U/V/W/X/Y broadening

For `φ = 2θ` in degrees, define `θ_rad = φ π/360`, `t = tan(θ_rad)`, and
`sθ = sec(θ_rad)`. [`crate::core::ConstantWavelengthInstrument`] evaluates

```text
q_G = U t² + V t + W
g = sqrt(8 ln(2)) sqrt(q_G)
l = X sθ + Y t.
```

`U,V,W` are Gaussian variances in degree²; `X,Y,g,l` are degrees `2θ`.
The domain requires `0 < φ < 180`, `q_G > 0`, and `l >= 0`. With
`k = π/360` and `c = sqrt(8 ln(2))`, analytical chains are

```text
∂g/∂q_G = c/[2 sqrt(q_G)]
∂g/∂U = (∂g/∂q_G)t²       ∂g/∂V = (∂g/∂q_G)t
∂g/∂W =  ∂g/∂q_G          ∂l/∂X = sθ       ∂l/∂Y = t

dt/dφ = k sθ²             dsθ/dφ = k sθ t
dq_G/dφ = (2Ut+V) dt/dφ
dg/dφ = (∂g/∂q_G) dq_G/dφ
dl/dφ = X dsθ/dφ + Y dt/dφ.
```

The position derivative includes both profile translation and width motion:

```text
d[I p(x-φ,g(φ),l(φ))]/dφ
  = I[-∂p/∂Δ + (∂p/∂g)(dg/dφ) + (∂p/∂l)(dl/dφ)].
```

See [`crate::core::CwProfileParameters`] and
[`crate::core::accumulate_cw_batch`].

## Discrete wavelength components

For positive component wavelengths `λ_j`, non-negative relative intensities
`q_j`, and reference component zero,

```text
w_j = q_j / Σ_k q_k                 r_j = λ_j/λ_0
φ_j = 2 asin[r_j sin(φ_0/2)].
```

The angle expression is in a consistent angular unit internally; public
positions are degrees. For secondary intensity ratio `R_j = q_j/q_0`,

```text
∂[Σ_k w_k P_k]/∂R_j = w_0 [P_j - Σ_k w_k P_k].
```

Position chains are

```text
dφ_j/dφ_0 = r_j cos(φ_0/2) / cos(φ_j/2)
dφ_j/dr_j = (360/π) sin(φ_0/2) / cos(φ_j/2)    [degrees].
```

Component widths are evaluated at each `φ_j`. One crystallographic reflection
still owns one total integrated intensity. See
[`crate::core::WavelengthComponentsView`] and the `cw_components` module.

## Finger–Cox–Jephcoat axial asymmetry

[`crate::core::FcjGeometry`] stores dimensionless half-height ratios
`s = S/L` and `h = H/L`. Let `b` be the ideal Bragg position in radians and
`z` a normalized axial separation:

```text
a(z,b) = acos[cos(b) sqrt(1+z²)]
M = max(s,h)       m = min(s,h)
A = M-m            B = M+m
g(z,b) = 1/[sqrt(1+z²) sin(a(z,b))].
```

For intrinsic normalized profile `R`, the regularized FCJ integral is

```text
N(x) = A ∫₀¹ g(At,b) R[x-a(At,b)] dt
     + 2m ∫₀¹ (1-t) g(A+2mt,b) R[x-a(A+2mt,b)] dt

Z    = A ∫₀¹ g(At,b) dt
     + 2m ∫₀¹ (1-t) g(A+2mt,b) dt

y(x) = N(x)/Z.
```

The transformed intervals are regular and fixed for deterministic
Gauss–Legendre quadrature. With `c = cos(b)`, `r = sqrt(1+z²)`, and
`Q = sqrt(1-c²r²)`, geometry derivatives are

```text
∂a/∂z = -c z/(rQ)                ∂a/∂b = sin(b) r/Q
∂[R(x-a)]/∂z = -R_Δ ∂a/∂z       ∂[R(x-a)]/∂b = -R_Δ ∂a/∂b
∂y/∂p = [∂N/∂p - y ∂Z/∂p]/Z.
```

The exact union support combines the intrinsic radius with the aberration
limits. `s=h=0` is the symmetric-profile limit. See
[`crate::core::FcjProfile`] and [`crate::core::accumulate_cw_fcj_batch`].

## Neutron time-of-flight

For d-spacing `d > 0`, [`crate::core::TofInstrument`] uses

```text
position = zero + difC d + difA d² + difB/d
alpha    = A/d
beta     = beta0 + beta1/d⁴ + betaq/d²
σ²       = sigma0 + sigma1 d² + sigma2 d⁴ + sigmaq d
g        = sqrt(8 ln(2)) sqrt(σ²)
l        = Z + Xd + Yd².
```

Coordinates and widths are microseconds; exponential rates are inverse
microseconds. The normalized back-to-back exponential is

```text
b(v) = alpha beta/(alpha+beta) exp(alpha v),   v < 0
b(v) = alpha beta/(alpha+beta) exp(-beta v),  v >= 0
p(s) = ∫ b(v) q_TCH(s-v;g,l) dv.
```

The implementation truncates a dimensionless exponential coordinate at
`0 <= t <= L` and renormalizes by `1-exp(-L)`; the default is `L=20`. For TCH
support radius `R`, the observable interval is

```text
left  = position - R - L/alpha
right = position + R + L/beta.
```

All d-spacing and 15 instrument-coefficient chains are analytical. See
[`crate::core::TofProfileParameters`] and
[`crate::core::accumulate_tof_batch_with_context`].

## Finite support convention

Symmetric profiles include samples satisfying `|x-μ| <= mH`, where `m` is the
[`crate::core::SupportPolicy::FwhmMultiple`] value. Truncation is not
renormalized, so intensity denotes infinite-support area. Derivatives hold the
selected discrete support fixed; distributional derivatives of a moving cutoff
are outside the API.
