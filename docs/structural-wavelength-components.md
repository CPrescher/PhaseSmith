# Structural wavelength components

## Scope and ownership

This slice extends constant-wavelength structural calculation and Rietveld
refinement from one wavelength to a fixed, discrete spectrum. It targets
calibrated laboratory doublets such as Cu K-alpha1/K-alpha2. Component
wavelengths and relative intensities are immutable configuration in this slice;
their refinement is deliberately deferred until fixed-component values and
shared structural derivatives pass real-data validation.

`ComponentRadiation` owns the probe plus `WavelengthComponents`. The profile
instrument retains the reference (component-zero) wavelength and the shared
physical U/V/W/X/Y coefficients. A one-component spectrum is numerically
identical to the monochromatic calculation. Every component used as structural
radiation has positive intensity; a zero-weight entry is absent physics and
must be removed from the spectrum.

## Equations

For reciprocal squared length `q_h`, component wavelength `lambda_c`, and
normalized source weight `w_c`,

```text
s_h = sqrt(q_h) / 2
theta_hc = asin(lambda_c sqrt(q_h) / 2)
p_hc = 2 theta_hc + zero/sample-displacement corrections
I_hc = w_c S m_h C_hc |F_h|^2 P_hc
y(x) = background(x) + sum_h sum_c I_hc profile(x - p_hc).
```

`C_hc` is evaluated independently for each wavelength. For the built-in
Bragg--Brentano model this includes the component-specific Lorentz--polarization
factor. Non-resonant X-ray and coherent-neutron structure factors depend on
`s_h`, not on the component wavelength, but they remain inside each native
structural batch so the existing value/JVP/VJP contract is unchanged. `P_hc`
contains optional size, microstrain, preferred-orientation, or provider terms
evaluated with the component geometry.

Shared structural JVPs sum component products. If the physical phase scale is
`S`, the native component phase stores `w_c S`, so the scale tangent and reverse
gradient are multiplied by `w_c`. All other structural derivatives already
carry `w_c` through the component scale. Shared instrument and sample-physics
global Jacobian rows are summed by stable name. Fixed component-wavelength
rows are not exposed as refinable instrument derivatives.

## Execution and diagnostics

Each component is one prepared fused native structural call. Python performs
only the small component-level composition; it never loops over atoms,
reflections, active support, or pattern samples. Finite support is applied to
each physical component peak with the same inclusive boundary convention as a
monochromatic calculation.

Component reflection diagnostics are flattened component-major. Every row
records a component index and base-reflection index, so reports can retain the
original hkl while exposing component-specific position and integrated
intensity.

Fixed-lattice reflection generation first uses a conservative d-spacing range
that is physically valid for the longest wavelength, then retains the exact
union of families visible for any component in the requested two-theta
interval. This prevents invalid inverse-sine evaluations while retaining
component-only reflections. Guarded lattice refinement with multiple
components is not part of this slice and must fail explicitly; it requires a
domain guard over both lattice motion and wavelength extrema.
