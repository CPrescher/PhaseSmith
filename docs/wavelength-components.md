# Discrete constant-wavelength components

`WavelengthComponents` represents an optional discrete spectrum on top of a
constant-wavelength instrument. Component zero is the reference and must match
`ConstantWavelengthInstrument.wavelength_angstrom`. A one-component model is
exactly monochromatic; the ordinary `accumulate_cw` and `accumulate_cw_fcj`
functions remain the simplest monochromatic APIs. Two components can represent
K-alpha1/K-alpha2, and the kernel supports any positive component count.

## Intensity convention

For finite non-negative relative integrated intensities `q[j]`, with `q[0]>0`,
the component weights are

```text
w[j] = q[j] / sum_k q[k].
```

Their common scale is irrelevant. The reported secondary intensity parameters
are ratios `rI[j]=q[j]/q[0]`, for `j>=1`. Holding every other ratio fixed,

```text
d sum_k(w[k] P[k]) / d rI[j]
  = w[0] (P[j] - sum_k w[k] P[k]).
```

Thus total integrated intensity belongs to the crystallographic reflection,
not to expanded component peaks. This is essential for Le Bail: one reflection
continues to own one intensity and one position derivative block.

## Bragg-law positions

Let `p` be the base reflection position in degrees `2theta`, `lambda0` the
instrument reference wavelength, and

```text
r[j] = lambda[j] / lambda0.
```

For `theta=p/2`, the component position is

```text
p[j] = 2 asin(r[j] sin(theta)).
```

The implementation converts the result back to degrees and rejects components
for which `r[j] sin(theta)` is outside `(0,1)`. Component zero uses the supplied
base position exactly. The analytical chains are

```text
d p[j] / d p = r[j] cos(theta) / cos(p[j]/2)

d p[j] / d r[j]
  = (360/pi) sin(theta) / cos(p[j]/2)       [degrees].
```

The second derivative is reported for every secondary wavelength ratio. It
includes translation, the angular U/V/W/X/Y width change, and FCJ geometry in
the same component/sample pass.

## Widths, FCJ, and support

U/V/W/X/Y component widths are evaluated at each component angle. If FCJ is
requested, each component has its own convolved geometry and intrinsic support.
The logical reflection support is the inclusive union of all component FCJ
supports. Within that union a component contributes only inside its own exact
intrinsic support; fixed-active-set derivative semantics are unchanged.

The symmetric shared-Jacobian order is

```text
U, V, W, X, Y,
wavelength_ratio[1..N-1],
intensity_ratio[1..N-1].
```

The FCJ version inserts `sample_over_radius` and `detector_over_radius` after Y.
Local columns remain `intensity, position` for each crystallographic
reflection.

## Validation and performance rule

Tests cover the exact one-component/monochromatic identity, Bragg positions,
resolved and unresolved doublets, area and ratio conservation, overlap, all
analytical derivatives, and the low/middle/high pinned GSAS-II FCJ matrix. The
Python reference expands components for clarity; production expands them only
inside the Rust reflection loop and never creates Python pseudo-reflections.
