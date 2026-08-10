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
  Finger--Cox--Jephcoat convolution.

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

The first slice deliberately excludes flat-plate transparency, full divergence
and Soller-slit optics, tube tails, monochromator/analyser passbands, and PSD
defocusing. These require separately reviewed equations and validation data.

## Validation and oracle boundary

The production side of the compression is already covered by the pinned
GSAS-II `U/V/W/X/Y + SH/L` profile fixtures. The new target side is checked
independently: wavelength conversion and equatorial-aperture moments have
closed-form tests, FCJ has its separate high-order reference matrix, and the
variable-projection Jacobian is checked with centered finite differences away
from support boundaries.

There is intentionally no golden claim that this reduced target equals the
GSAS-II/NIST FPA generator. The pinned GSAS-II boundary does not expose that
GUI workflow as a stable plain-array scripting API, and the first slice omits
several of its physical contributions. A future black-box FPA fixture must pin
all input conventions and provenance before it can become an oracle gate.

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
    sample_half_length_mm=1.0875,
    detector_half_length_mm=1.0875,
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
)

physical = phasesmith.simulate_fundamental_peaks(model)
calibration = phasesmith.calibrate_fundamental_profile(model)

if not calibration.accepted:
    raise RuntimeError(calibration.warnings)

instrument = calibration.instrument
components = calibration.components
axial_geometry = calibration.axial_geometry
```

`physical.grid_deg`, `physical.intensity`, `calibration.target_y`, and
`calibration.calculated_y` are immutable arrays suitable for plots and audit
records. `peak_diagnostics` reports the scale, relative L2 error, and
correlation for every requested angle. The fitted `axial_geometry` is the
single GSAS-II-compatible equal-height `SH/L` representation; the original
physical geometry remains available on the input model.
