# Monochromatic-neutron constant-wavelength profiles

Neutron CW is a typed configuration of the shared constant-wavelength profile
pipeline. It does not copy the TCH, U/V/W/X/Y, FCJ, support, accumulation, or
derivative implementations.

## Type contract

`MonochromaticRadiation` records a `RadiationProbe` and one wavelength.
`ConstantWavelengthExperiment` pairs it with the profile instrument and rejects
any wavelength mismatch. The convenience constructor
`ConstantWavelengthExperiment.neutron(instrument)` therefore creates an
explicitly neutron, explicitly monochromatic experiment without repeating a
wavelength in user code.

The neutron entry points do not accept `WavelengthComponents`. Consequently a
K-alpha doublet cannot be attached accidentally to neutron calculations.
`calculate_monochromatic_pattern` is probe-generic;
`calculate_neutron_pattern` additionally enforces the neutron probe. The
low-level `calculate_neutron_fcj_pattern` composes the same published FCJ
geometry with a typed neutron experiment.

## Shared profile equations

When a neutron instrument supplies the same physical U/V/W/X/Y coefficients as
an X-ray instrument, both probes produce the same normalized peak shape and
analytical profile derivatives:

```text
q(2theta) = U tan(theta)^2 + V tan(theta) + W
l(2theta) = X sec(theta) + Y tan(theta).
```

The probe type changes the interpretation and future intensity physics, not
these profile equations. Wavelength still enters Bragg geometry and compatible
sample providers such as Scherrer size. Nuclear scattering lengths, magnetic
structure factors, absorption, and other neutron intensity terms remain a
separate future physics layer; this milestone accepts base integrated
intensities.

## Validation boundary

The pinned neutron fixture uses GSAS-II histogram type `PNC`. Public scripting
arrays include `X`, total `Ycalc`, background, and the complete reflection list.
Selected low-, middle-, and high-angle reflections are probed in symmetric and
FCJ forms. The fixture validates component widths, values, area, centroid, and
asymmetric moments, while ordinary tests finite-difference the shared
analytical derivatives.
