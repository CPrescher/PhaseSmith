# Structural intensities and fused-pattern conventions

This document freezes the equations, units, parameter ordering, correction
ownership, and derivative contract for implementation unit 15. GSAS-II is not
an equation or data source. Its pinned external process may later validate the
observable reflection fields and perturbation scales described in
`oracle/STRUCTURAL_PARAMETER_STUDY.md`.

## Reciprocal and symmetry conventions

For canonical Miller index `h`, let `q² = hᵀG*h = 1/d²` in inverse square
ångströms and `s = sqrt(q²)/2 = sin(theta)/lambda` in inverse ångströms. For
asymmetric site `j`, the exact space-group operations generate a set of unique
positions

```text
x_jr = R_jr x_j + t_jr  (modulo lattice translations).
```

The structure factor is

```text
T_j(h) = exp(-2 pi² U_j q²) = exp(-8 pi² U_j s²)
S_j(h) = sum_r exp(2 pi i h dot x_jr)
F_h = sum_j occupancy_j f_j(s) T_j(h) S_j(h).
```

This is the standard unit-cell Fourier sum documented by the IUCr
[structure-factor definition](https://dictionary.iucr.org/Structure_factor)
and its
[symmetry-operation derivation](https://www.iucr.org/education/pamphlets/9/full-text).
The isotropic displacement convention follows `B = 8 pi² U` and the
`exp[-B (sin(theta)/lambda)²]` form summarized in the IUCr article
[A new theory for X-ray diffraction](https://journals.iucr.org/a/issues/2014/03/00/sc5066/index.html),
equation 2.

Special-position duplicates are included once. The native expansion retains
the representative exact rotation for each unique position so coordinate
derivatives use the same fixed orbit topology. At a special position, a free
Cartesian coordinate perturbation can change orbit multiplicity and is not a
valid derivative. Refinement therefore applies only site-stabilizer-compatible
tangents; unit 17 will construct those constraints. Values are valid regardless
of whether structural derivatives are requested.

Non-resonant X-ray and constant real nuclear-neutron scattering obey Friedel's
law. A generated powder family merges Friedel mates exactly once and supplies
`multiplicity_h`; `F_h` is evaluated only for its canonical representative.
The symmetry-generated atom sum and reciprocal-family multiplicity are
different operations and must never be multiplied into each other twice.

## Integrated intensity and correction ownership

The profile receives the integrated reflection area

```text
I_h = phase_scale * multiplicity_h * C_h * |F_h|².
```

Two initial correction choices are explicit:

- `NeutralIntegratedIntensityCorrection` sets `C_h = 1`. It is required when
  supplied data are already corrected or when the caller wants raw
  multiplicity-weighted structure intensities.
- `BraggBrentanoUnpolarizedLp` applies a reflection-constant, integrated-area
  Lorentz--polarization factor

  ```text
  C_h = (1 + cos²(2 theta)) / (2 sin²(theta) cos(theta)).
  ```

  The constant `1/2` is retained from the unpolarized polarization factor;
  removing it would only redefine phase scale. The Lorentz term is the
  integrated peak-area convention (equation 29) and polarization is equation
  32 in Dinnebier and Scardi,
  [X-ray powder diffraction in education. Part II](https://journals.iucr.org/j/issues/2023/03/00/dv5004/).

The initial LP model is valid only for monochromatic unpolarized X-rays in
symmetric angular-dispersive reflection geometry and `0 < 2theta < 180°`.
Synchrotron polarization, transmission/capillary geometry, neutron, TOF,
absorption, extinction, and pointwise broad-peak corrections require distinct
typed models. They are never selected from an ambiguous boolean.

Preferred orientation is owned by the existing reflection-physics provider.
The structure-factor correction batch has no preferred-orientation field; the
profile composition multiplies the existing provider result exactly once.
Likewise, `RietveldPhase.scale` is the only structural phase scale and the
downstream profile phase multiplier is neutral.

## Analytical derivatives

For every parameter `p`,

```text
d|F|²/dp = 2 Re(conjugate(F) dF/dp)
dI/dp = multiplicity * [
    d(scale)/dp * C * |F|²
  + scale * dC/dp * |F|²
  + scale * C * d|F|²/dp
].
```

The fixed-topology site terms are

```text
dS_j/dx_jk = sum_r 2 pi i (h dot column_k(R_jr))
                       exp(2 pi i h dot x_jr)
dF/d(occupancy_j) = f_j T_j S_j
dF/dU_j = -2 pi² q² occupancy_j f_j T_j S_j.
```

Cell derivatives include both displacement and scattering-vector chains:

```text
ds/dp = [d(q²)/dp] / [4 sqrt(q²)]
d(f_j T_j)/dp = T_j [df_j/ds * ds/dp
                       - 2 pi² U_j f_j d(q²)/dp].
```

For monochromatic CW position `phi = 2theta` in radians,

```text
phi = 2 asin(lambda sqrt(q²) / 2)
dphi/dp = lambda * d(q²)/dp / [2 sqrt(q²) cos(theta)].
```

The LP cell chain uses

```text
d log(C)/dphi = -2 sin(phi) cos(phi)/(1 + cos²(phi))
                - cot(phi/2) + 0.5 tan(phi/2).
```

Public angle derivatives are per degree, so native radian derivatives carry
the explicit `pi/180` conversion where appropriate.

## Stable parameter and array layouts

Structural parameters retain the established order:

1. `cell.a`, `cell.b`, `cell.c`, `cell.alpha`, `cell.beta`, `cell.gamma`;
2. `site.<id>.x`, `.y`, `.z`, site-major;
3. `site.<id>.occupancy`, site-major;
4. `site.<id>.u_iso`, site-major;
5. `phase.scale`.

Reflection arrays are reflection-major. Scattering arrays have shape
`(reflection_count, asymmetric_site_count)`; structural dense diagnostics are
parameter-major. The production engine exposes JVP/VJP products and bounded
dense diagnostics. The combined engine call owns symmetry expansion,
structure factors, integrated intensities, CW positions, and support-limited
profile accumulation; no Python loop over sites, reflections, or samples is a
production orchestration step.

## Unit 15 implementation sequence

1. Extend exact symmetry expansion with representative rotations and tests.
2. Add a general-symmetry Rust values/dense/JVP/VJP structure-factor kernel
   consuming scattering values plus `df/ds`.
3. Add typed neutral and Bragg--Brentano LP correction kernels with derivatives.
4. Bind the structural result and provider-facing Python API; retain the P1
   compatibility API unchanged.
5. Compose reflection generation, built-in scattering, structural intensity,
   CW position, existing sample physics, and profile accumulation in one
   `rietveld-engine` call.
6. Add `RietveldPhase`, stable diagnostics, persistence migration, independent
   NumPy references, invariance/finite-difference/adjoint tests, and realistic
   separate/combined benchmarks.
7. Run the full quality gate, review the complete unit, and commit. External
   GSAS-II fixtures remain a separately generated validation artifact.
