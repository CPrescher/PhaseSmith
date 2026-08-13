# Crystallography, scattering, and structural intensity

The equations on this page are implemented by [`crate::crystallography`] and
composed into powder patterns by [`crate::engine`].

## Direct and reciprocal cells

For cell `(a,b,c,α,β,γ)`, with lengths in ångströms and angles in radians while
evaluating trigonometric functions, the direct metric is

```text
G = [ a²          ab cosγ     ac cosβ ]
    [ ab cosγ     b²          bc cosα ]
    [ ac cosβ     bc cosα     c²      ].
```

[`crate::crystallography::UnitCell::geometry`] derives

```text
G* = G^-1
q² = h^T G* h
d  = 1/sqrt(q²)
V  = sqrt(det G).
```

Reciprocal vectors omit the `2π` factor. For any direct-cell parameter `p`,

```text
∂G*/∂p = -G* (∂G/∂p) G*
∂q²/∂p = h^T (∂G*/∂p) h
∂d/∂p  = -(1/2) d³ ∂q²/∂p
∂V/∂p  = (V/2) trace[G* (∂G/∂p)].
```

The derivative order is `a,b,c,alpha,beta,gamma`; angular rows include the
radians-per-degree chain.

## Exact symmetry and reflection conditions

[`crate::crystallography::SymmetryOperation`] acts on fractional coordinates
and Miller indices as

```text
x' = R x + t  (mod 1)          h' = R^T h
F(h) = exp(2πi h·t) F(R^T h).
```

`R` is integer unimodular and `t` is exact rational. For operations that share
the same `R^T h`, a general-position systematic absence occurs when every
translation-character group is exactly zero:

```text
Σ_g exp(2πi h·t_g) = 0.
```

[`crate::crystallography::SpaceGroup`] evaluates these sums with rational
phases and cyclotomic reduction, not a floating extinction tolerance. Metric
compatibility is the exact constraint

```text
R^T G R = G.
```

Reflection generation enumerates an inclusive physical range and merges a
reciprocal orbit into one family. For `d_min`, a safe integer search box uses

```text
q_max = 1/d_min
λ_min(G*) >= 1/trace(G)
|h_i| <= q_max sqrt(G_ii).
```

See [`crate::crystallography::PreparedReflectionGenerator`].

## Scattering factors

For `s = sinθ/λ = 1/(2d)`, non-resonant X-ray form factors use the
Waasmaier–Kirfel five-Gaussian representation:

```text
f0(s) = c + Σ_(n=1..5) a_n exp(-b_n s²)
df0/ds = -2s Σ_(n=1..5) a_n b_n exp(-b_n s²).
```

[`crate::crystallography::PreparedXrayScattering`] evaluates the tabulated fit
only within its declared `s` range. Fixed dispersion offsets use

```text
f(s,λ_fixed) = f0(s) + f'(λ_fixed) + i f''(λ_fixed)
df/ds = df0/ds.
```

Coherent nuclear-neutron scattering uses a species/isotope-specific real bound
length `b_c`:

```text
f(s) = b_c                 df/ds = 0.
```

See [`crate::crystallography::PreparedNeutronScattering`].

## Symmetry-expanded structure factors

For asymmetric site `j`, unique symmetry mate `r`, occupancy `o_j`, isotropic
displacement `U_j`, and scattering amplitude `f_j(s)`,

```text
x_jr = R_jr x_j + t_jr                 (mod lattice translations)
T_j(h) = exp(-2π² U_j q²) = exp(-8π² U_j s²)
S_j(h) = Σ_r exp(2πi h·x_jr)
F_h = Σ_j o_j f_j(s) T_j(h) S_j(h).
```

`B = 8π²U`. Special-position duplicate mates are included once. For a fixed
anisotropic CIF tensor with component order `(U11,U22,U33,U23,U13,U12)`,

```text
p_jr = R_jr^T h
v_jr,i = p_jr,i a*_i
T_jr(h) = exp[-2π² v_jr^T U_j v_jr]
F_h,j = o_j f_j(s) Σ_r T_jr(h) exp(2πi h·x_jr).
```

The mate rotation precedes reciprocal-axis scaling, including in non-orthogonal
settings. These equations are owned by the `structure_factor` module and used
by [`crate::engine::PreparedStructuralPhase`].

## Structural derivatives

For any parameter `p`,

```text
∂|F|²/∂p = 2 Re[conj(F) ∂F/∂p]
∂S_j/∂x_jk = Σ_r 2πi [h·column_k(R_jr)] exp(2πi h·x_jr)
∂F/∂o_j = f_j T_j S_j
∂F/∂U_j = -2π²q² o_j f_j T_j S_j.
```

Cell motion includes scattering and displacement chains:

```text
∂s/∂p = (∂q²/∂p)/[4 sqrt(q²)]
∂(f_jT_j)/∂p = T_j[(df_j/ds)(∂s/∂p) - 2π²U_j f_j (∂q²/∂p)].
```

For anisotropic displacement,

```text
∂(v^T U v)/∂p = 2 (Uv)^T ∂v/∂p
∂T/∂p = -2π² T ∂(v^T U v)/∂p.
```

Dense, JVP, and VJP APIs evaluate the same derivative layout. See
[`crate::crystallography::calculate_structure_factor_jvp`] and
[`crate::crystallography::calculate_structure_factor_intensity_vjp`].

## Integrated reflection intensity

For multiplicity `m_h`, phase scale `S`, and correction `C_h`,

```text
I_h = S m_h C_h |F_h|²
∂I_h/∂p = m_h[(∂S/∂p)C_h|F_h|²
             + S(∂C_h/∂p)|F_h|²
             + S C_h ∂|F_h|²/∂p].
```

[`crate::crystallography::IntegratedIntensityCorrectionModel`] implements:

```text
Neutral:       C_h = 1

Unpolarized X-ray Bragg–Brentano:
C_h = [1 + cos²(2θ)]/[2 sin²θ cosθ]

Polarized X-ray Bragg–Brentano:
C_h = [P + (1-P)cos²(2θ)]/[sin²θ cosθ],  0 <= P <= 1

Constant-wavelength neutron powder:
C_h = 1/[sinθ sin(2θ)]

Conventional one-dimensional neutron TOF powder at fixed bank angle:
C_h = d_h^4 sinθ_bank = sinθ_bank/(q_h²)^2
dC_h/d(q_h²) = -2 sinθ_bank/(q_h²)^3.
```

The constant-wavelength neutron expression is exactly one half of the
polarized implementation's `P=1` algebraic form. Correction values include
analytical `q²` and wavelength derivatives. The TOF derivative with respect to
wavelength is exactly zero because its independent geometry input is the fixed
bank `2theta`; spectrum and detector-efficiency terms are separate correction
models rather than an implicit part of this Lorentz factor.

## Bragg positions and specimen displacement

For monochromatic wavelength `λ`,

```text
θ = asin[λ sqrt(q²)/2]             φ = 2θ
∂φ/∂q² = λ/[2 sqrt(q²) cosθ]       (radians).
```

[`crate::engine::MonochromaticPositionCorrection`] then applies a constant
zero shift and at most one specimen-geometry correction. For flat-plate
Bragg–Brentano displacement `D` and radius `R`, both in millimetres,

```text
δφ_deg = -(180/π) (2D/R) cosθ.
```

For Debye–Scherrer displacements `X,Y` in micrometres and radius `R` in
millimetres,

```text
δφ_deg = -0.18/(πR) [X cosφ + Y sinφ].
```

Corrections are evaluated at the unshifted Bragg position; the constant zero
shift is added independently. Cell and wavelength derivatives include the
correction chain.
