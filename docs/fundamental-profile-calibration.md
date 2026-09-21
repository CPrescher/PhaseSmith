# Offline fundamental-profile calibration

PhaseSmith can generate isolated laboratory constant-wavelength peaks from a
small, explicit physical model and compress them into the production
`U/V/W/X/Y + SH/L` profile. This is an **offline calibration workflow**. The
ordinary Le Bail and Rietveld hot loops continue to evaluate the fast Rust
profile; they do not evaluate a fundamental-parameters model for every
reflection and trial.

This matches the useful part of the GSAS-II workflow: make physical reference
peaks, fit the conventional profile over the intended angular range, inspect
the residuals, and use the fitted coefficients for subsequent refinement. It
does not reproduce GSAS-II source code or project objects.

## First-slice physical target

The reviewed target includes:

- any fixed set of discrete emission wavelengths and integrated intensities;
- an intrinsic Gaussian and Lorentzian wavelength FWHM for each line;
- ideal uniform source and receiving-slit equatorial apertures;
- physical sample and receiving-slit axial half-lengths through the published
  Finger--Cox--Jephcoat convolution; or
- full source, illuminated-sample, and receiving-slit axial lengths with
  independent triangular incident and diffracted Soller transmissions; and
- an optional unit-height Gaussian wavelength passband applied to the source
  spectrum before the geometrical convolutions.

Let line `j` have wavelength `lambda_j`, and let `d` be defined by the first
line's requested peak position. Its Bragg position is

```text
2 theta_j = 2 asin(lambda_j / (2 d)).
```

The local wavelength-to-angle derivative converts intrinsic wavelength FWHMs
to degrees `2theta`:

```text
d(2 theta_j)/d(lambda_j)
    = (180/pi) 2 tan(theta_j) / lambda_j.
```

For source coordinate `s`, detector coordinate `r`, and goniometer radius `R`,
the first-order tangent-circle equatorial shift is

```text
delta(2 theta) = (180/pi) (s + r) / R.
```

Both coordinates are uniform across their supplied full aperture widths and
are integrated with deterministic Gauss--Legendre quadrature. Axial
half-lengths are divided by `R` and passed to the independent NumPy FCJ
reference described in [FCJ asymmetry](fcj-profile.md). The FCJ equations are
from Finger, Cox and Jephcoat, *J. Appl. Cryst.* **27** (1994), 892--900,
[doi:10.1107/S0021889894004218](https://doi.org/10.1107/S0021889894004218).
The general convolution strategy follows the fundamental-parameters treatment
of Mendenhall, Mullen and Cline, *J. Res. NIST* **120** (2015), 223--251,
[doi:10.6028/jres.120.014](https://doi.org/10.6028/jres.120.014), but the code
here is independently derived from the stated equations.

### Full axial and Soller target

`SollerAxialGeometry` selects the full axial ray target. Let `z_s`, `z`, and
`z_r` be coordinates across the source, illuminated sample, and receiving
slit, respectively. The incident and diffracted axial angles are

```text
beta  = atan((z - z_s) / R),
gamma = atan((z_r - z) / R).
```

For the nominal equatorial scattering angle `a0`, the apparent angle `a` is
obtained from the ray-vector dot product:

```text
cos(a) = (cos(a0) - sin(beta) sin(gamma))
         / (cos(beta) cos(gamma)).
```

A Soller value is the full base width of its triangular transmission. With
`B_i` and `B_d` in the same angular units as `beta` and `gamma`,

```text
S_i(beta)  = max(0, 1 - 2 abs(beta) / B_i),
S_d(gamma) = max(0, 1 - 2 abs(gamma) / B_d).
```

`None` means unit transmission. The target integrates the finite coordinate
volume with the ray-density factor

```text
cos(beta) cos(gamma) / sin(a)
```

and normalizes the positive ray weights for every Bragg peak. The generator uses
a deterministic two-dimensional Gauss--Legendre integral in `(beta, gamma)`;
the analytically computed overlap of the three finite axial intervals removes
the sample coordinate. Constant Jacobian factors cancel during normalization.
The default order is 255. For the NIST 12 mm / 15 mm / 5 mm geometry with
6.776 degree Soller widths, order 191 differs from order 255 by less than
`2e-3` in relative L2 norm on the test grid.

The triangular transmissions and finite source/sample/receiver geometry
follow Cheary and Coelho, *J. Appl. Cryst.* **31** (1998), 851--861,
[doi:10.1107/S0021889898006876](https://doi.org/10.1107/S0021889898006876),
and the explicit NIST conventions in Mendenhall, Mullen and Cline. The
integration and coordinate transform are independent PhaseSmith derivations.
The separately evaluated point-source/point-sample case uses the exact FCJ
coordinate density `cos(gamma)^2 / sin(a)` and is tested against the independent
FCJ reference. This branch was corrected by the direct angular-equation audit;
the finite-source ray-density approximation above is a separate model and is
not validated by that point-source equality test.

This increment still excludes flat-plate transparency, equatorial divergence
beyond the ideal apertures, tube tails, passband-induced angular dispersion,
and PSD defocusing. These require separately reviewed equations and validation
data.

### Gaussian spectral passband

`GaussianSpectralPassband` represents an explicitly supplied analyzer or
monochromator transmission with center `lambda_c` and wavelength FWHM `B`:

```text
T(lambda) = exp[-4 ln(2) ((lambda - lambda_c) / B)^2].
```

It has unit peak transmission and infinite mathematical support. For emission
line `j`, the target wavelength density is multiplied before axial and
equatorial convolution,

```text
p_filtered,j(lambda) = p_TCH,j(lambda) T(lambda).
```

The transmitted line area is

```text
A_effective,j = A_j integral p_TCH,j(lambda) T(lambda) d(lambda).
```

These effective areas become the fixed component weights returned by a
calibration. The first wavelength moment of each transmitted line likewise
becomes its effective component wavelength, so attenuation and passband-induced
centroid motion are not lost when the target is compressed. The Gaussian parts
of both moments are evaluated analytically. The Lorentzian parts use the
substitution `lambda = lambda_j + (H/2) tan(t)`, which maps its infinite support
to a finite Gauss--Legendre integral and makes the Lorentzian measure uniform
in `t`.

The passband is intentionally parameterized by wavelength center and FWHM.
NIST identifies the 26.6 degree value for its graphite post-analyzer as the
analyzer diffraction angle, not a passband width. The SRM 660c pdCIF says that
a Gaussian approximation and offset were used, but does not contain enough
information to reconstruct a unique bandwidth. PhaseSmith therefore does not
invent a NIST default. The current post-analyzer slice also adds no dispersion
term, consistent with the pdCIF statement that the analyzer is adjacent to the
detector. Incident focusing monochromators need the coupled dispersion model of
Mendenhall, Black and Cline, *J. Appl. Cryst.* **52** (2019), 1087--1094,
[doi:10.1107/S1600576719010951](https://doi.org/10.1107/S1600576719010951),
and remain a separate increment.

Applying a spectral passband currently requires `SollerAxialGeometry`, even
when all three axial lengths are zero. This makes the operation order explicit:
wavelength filtering precedes the independent ray convolution. It avoids
silently multiplying an already FCJ-convolved angular profile by a wavelength
window, which would be a different model.

## Validation and oracle boundary

The production side of the compression is already covered by the pinned
GSAS-II `U/V/W/X/Y + SH/L` profile fixtures. The new target side is checked
independently: wavelength conversion and equatorial-aperture moments have
closed-form tests, FCJ has its separate high-order reference matrix, the full
axial target recovers the FCJ limiting case, Soller filters are checked through
peak moments, and the transformed quadrature has an explicit convergence test.
The variable-projection Jacobian is checked with centered finite differences
away from support boundaries.

There is intentionally no golden claim that this reduced target equals the
GSAS-II/NIST FPA generator. The pinned GSAS-II boundary does not expose that
GUI workflow as a stable plain-array scripting API, and the first slice omits
several physical contributions. A future black-box FPA fixture must pin
all input conventions and provenance before it can become an oracle gate.

The target ray calculation is an offline, fixed-data generator and therefore
does not add target-geometry derivative columns to refinement. Values and all
six derivatives of the compressed candidate remain evaluated together by the
production Rust pass. The synthetic target is evaluated only on the requested
isolated windows; those window edges are its explicit finite comparison
support.

Passband validation includes the exact half-height convention, immutable
finite transmission arrays, an independent dense wavelength integral, peak
moment narrowing, transmitted multi-line weights, and invalid-domain tests.
The passband adds no refinement columns: its parameters are fixed physical
inputs to the offline target, while the compressed candidate continues to use
the existing fused Rust values and analytical derivatives.

An exploratory SRM 660c specimen 100a probe used the documented 12 mm source,
15 mm illuminated sample, 5 mm receiving slit, and 6.776 degree incident and
diffracted Soller widths with a provisional narrow Cu doublet. The resulting
compressed profile reduced PhaseSmith's weighted Rwp from 20.495% to 12.525%
and increased profile correlation from 0.95959 to 0.99482. NIST's released
fundamental-parameters curve remains at 6.055% Rwp and 0.99948 correlation.
More importantly, the compression has global relative L2 error 0.1909 and
therefore fails the normal 0.03 acceptance limit. This is evidence that the
axial/Soller physics matters, not an accepted calibration. At that stage the
graphite analyzer passband had not yet been tested, and the provisional line
widths could not be promoted to a golden instrument model.

After adding the explicit Gaussian passband, a full-order balanced probe with
center 1.5425 Å and FWHM 0.004 Å gives 15.6665% weighted Rwp and 0.99319
correlation after its transmitted line centroids are propagated into the
compressed spectrum. The first effective component is reanchored to NIST's
certified Kα1 wavelength for this diagnostic while preserving its filtered
component separation. The same Soller model without the passband gives
12.5254% and 0.99482. A coarse sweep using the published multi-Lorentzian Cu
spectrum was also substantially worse after compression. The passband
implementation is therefore retained as a supported physical input, but it is
not promoted as the solution to the NIST residual. The existing provisional
narrow doublet already behaves like an effective post-analyzer spectrum, and
the released pdCIF does not justify fitting a hidden bandwidth to specimen
100a. The passband compression itself also remains rejected by the standard
gates: global relative L2 error 0.1590, maximum per-peak relative L2 error
0.2006, and minimum correlation 0.97614.

## Compression model and acceptance

The fitted production model is the documented PhaseSmith convention:

```text
q = U tan(theta)^2 + V tan(theta) + W
g = sqrt(8 ln 2) sqrt(q)
l = X sec(theta) + Y tan(theta)
sample_over_radius = detector_over_radius = (SH/L) / 2.
```

Every isolated peak receives its own non-negative integrated scale. The six
shared shape parameters are fitted with the analytical derivatives returned by
the production Rust peak/sample pass. The target generator remains independent
NumPy code. Default diagnostics require all of the following:

- global relative L2 error at most 0.03;
- every peak's relative L2 error at most 0.05;
- every peak's profile correlation at least 0.995.

`accepted=False` is a useful result: it means the requested physical target is
not safely representable by this empirical profile over the selected angular
range. For example, emission components with substantially different intrinsic
relative widths can fail even though the discrete wavelengths themselves are
represented exactly. Do not loosen acceptance limits merely to turn such a
model into a nominal calibration.

## Python workflow

```python
import phasesmith

model = phasesmith.BraggBrentanoFundamentalProfile(
    radius_mm=217.5,
    source_width_mm=0.02,
    receiving_slit_width_mm=0.04,
    sample_half_length_mm=7.5,
    detector_half_length_mm=2.5,
    emission_lines=(
        phasesmith.FundamentalEmissionLine(
            wavelength_angstrom=1.5405929,
            relative_intensity=2.0,
            gaussian_fwhm_angstrom=0.0002600,
            lorentzian_fwhm_angstrom=0.0001200,
        ),
        phasesmith.FundamentalEmissionLine(
            wavelength_angstrom=1.5444274,
            relative_intensity=1.0,
            gaussian_fwhm_angstrom=0.00026065,
            lorentzian_fwhm_angstrom=0.00012030,
        ),
    ),
    soller_axial_geometry=phasesmith.SollerAxialGeometry(
        source_full_length_mm=12.0,
        sample_full_length_mm=15.0,
        receiving_slit_full_length_mm=5.0,
        incident_soller_full_width_deg=6.776,
        diffracted_soller_full_width_deg=6.776,
    ),
    spectral_passband=phasesmith.GaussianSpectralPassband(
        center_wavelength_angstrom=1.5425,
        gaussian_fwhm_angstrom=0.008,
    ),
)

physical = phasesmith.simulate_fundamental_peaks(model)
calibration = phasesmith.calibrate_fundamental_profile(model)

if calibration.accepted:
    instrument = calibration.instrument
    components = calibration.components
    axial_geometry = calibration.axial_geometry
else:
    print(calibration.warnings)
```

`physical.grid_deg`, `physical.intensity`, `calibration.target_y`, and
`calibration.calculated_y` are immutable arrays suitable for plots and audit
records. `peak_diagnostics` reports the scale, relative L2 error, and
correlation for every requested angle. The fitted `axial_geometry` is the
single GSAS-II-compatible equal-height `SH/L` representation; the original
physical geometry remains available on the input model.
