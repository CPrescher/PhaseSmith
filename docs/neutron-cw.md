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

## Debye--Scherrer specimen displacement

Capillary/transmission experiments use `DebyeScherrerGeometry`, which is
deliberately distinct from the flat-plate `BraggBrentanoGeometry`. Its fixed
goniometer radius is in millimetres; `displace_x_micrometre` and
`displace_y_micrometre` are in micrometres. X is perpendicular and Y is
parallel to the incident beam. For an uncorrected reflection position
`phi = 2 theta`, PhaseSmith applies

```text
delta(phi)_deg = -0.18 / (pi R_mm)
                 * [X_um cos(phi) + Y_um sin(phi)].
```

The correction is evaluated at the unshifted Bragg position and the separate
constant `zero_shift_deg` is then added. The native structural kernel exposes
analytical rows for X and Y and includes the correction's chain derivative in
cell and wavelength derivatives. A typical refinable neutron experiment is:

```python
experiment = ConstantWavelengthExperiment(
    MonochromaticRadiation.neutron(1.909),
    instrument,
    zero_shift_deg=-0.1,
    geometry=DebyeScherrerGeometry(goniometer_radius_mm=650.0),
)
selection = RietveldParameterSelection(
    instrument_parameters=(
        "u_deg2", "v_deg2", "w_deg2",
        "displace_x_micrometre", "displace_y_micrometre",
    ),
)
```

The radius remains caller-owned fixed geometry. The two displacement values
can be selected explicitly or proposed by the intelligent staged recipe's
position step; the general solver does not impose that sequence.

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
sample providers such as Scherrer size. Nuclear scattering lengths are built
in. Magnetic structure factors, absorption, and other neutron intensity terms
remain separate future physics layers.

## Powder Lorentz correction

Constant-wavelength neutron powder intensities require the angular Lorentz
factor

```text
L(theta) = 1 / [sin(theta) sin(2 theta)].
```

`ConstantWavelengthNeutronLorentz(wavelength_angstrom)` evaluates this factor
and its analytical reciprocal-metric and wavelength derivatives in the native
structural kernel. It is explicit rather than silently attached to every
neutron calculation, so controlled neutral-intensity studies remain possible.
The model is neutron-only and rejects a wavelength mismatch with the experiment.

## Validation boundary

The pinned neutron fixture uses GSAS-II histogram type `PNC`. Public scripting
arrays include `X`, total `Ycalc`, background, and the complete reflection list.
Selected low-, middle-, and high-angle reflections are probed in symmetric and
FCJ forms. The fixture validates component widths, values, area, centroid, and
asymmetric moments, while ordinary tests finite-difference the shared
analytical derivatives.
