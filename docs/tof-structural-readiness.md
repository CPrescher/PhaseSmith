# Structural TOF readiness and correction contract

PhaseSmith's completed TOF workflow is a facility-neutral profile and Le Bail
implementation. It can refine shared lattice geometry and bank-local profile
calibrations, but those capabilities alone do not make a structural Rietveld
model. Structural TOF intensities need an explicit observation convention in
addition to a peak shape.

This document records the Unit 38 capability review and the contract that must
be implemented before PhaseSmith claims structural TOF refinement.

## What is already reusable

The following production components do not depend on constant wavelength and
can be reused directly:

- reciprocal-metric d-spacings and analytical cell derivatives;
- symmetry expansion, multiplicity, nuclear structure factors, coordinate,
  occupancy, and displacement-parameter derivatives;
- constant bound coherent neutron scattering lengths for species that are not
  marked energy dependent by the pinned table;
- the 15-coefficient TOF calibration/profile model and its fused value,
  intensity, position, width, and coefficient derivatives;
- deterministic finite support, background composition, masks, uncertainties,
  runtime cancellation, checkpoints, and atomic multi-bank objectives; and
- stable shared reflection and parameter identities.

The current structural pattern engine itself is not reusable unchanged. Its
public input, position correction, sample-width contributions, wavelength
installation, and peak accumulation are explicitly in degrees `2theta` and use
a `ConstantWavelengthInstrument`.

## The missing physical contract

For a conventional focused one-dimensional TOF bank, each reflection has

```text
lambda_h = 2 d_h sin(theta_bank)
tof_h = Zero + DIFC d_h + DIFA d_h^2 + DIFB / d_h
```

where `2 theta_bank` is fixed bank geometry. `BNKPAR` is therefore not optional
metadata for structural work even though profile-only Le Bail extraction does
not need it.

The calculated integrated intensity must separate three choices:

```text
I_h = scale * multiplicity_h * |F_h|^2
      * C_geometry(d_h, theta_bank)
      * C_spectrum(lambda_h)
      * C_detector(lambda_h)
      * C_sample(h, lambda_h)
```

The factors after `|F|^2` depend on what upstream reduction already applied.
The POWGEN example discussed by Jacobs et al. was background corrected and
normalized by a vanadium measurement to account for detector efficiency and
remove the wavelength-dependent incident distribution. A raw or differently
reduced beamline dataset cannot safely inherit the same assumptions.

PhaseSmith will therefore keep two explicit baseline modes:

1. `Neutral` means the supplied observations have already been reduced to a
   convention in which no additional integrated-intensity correction is
   requested.
2. A named TOF neutron powder Lorentz model implements the conventional
   one-dimensional Rietveld factor

   ```text
   C_TOF = d_h^4 sin(theta_bank)
   d C_TOF / d(q_h^2) = -2 sin(theta_bank) / (q_h^2)^3
   ```

   with `q_h^2 = 1 / d_h^2`. It is selected explicitly and never inferred from
   a facility name or file extension.

Incident-spectrum, detector-efficiency, absorption, extinction, and other
sample corrections remain separate typed factors. A tabulated wavelength
factor must state its interpolation and extrapolation rules and provide the
derivative needed by the fused structural pass. PhaseSmith must not hide such a
factor in a phase scale or a profile coefficient.

Recent work on TOF Lorentz factors also shows why the observable must be named:
the appropriate factor changes with the reduced quantity and its binning. The
legacy one-dimensional Bragg-Rietveld convention is not a universal correction
for event data, total scattering, or a two-dimensional angle/wavelength
observable.

## Facility-neutral bank metadata

The application boundary needs a typed bank-geometry record independent of any
legacy file format. Its first required field is finite
`0 < two_theta_deg < 180`. A GSAS `BNKPAR` adapter may populate it, but the core
model must not consume legacy records or dictionaries. Flight paths and other
source metadata can be retained when their definitions are independently
documented; they are not required by the calibrated `tof(d)` equation.

The existing `TofInstrument` remains the profile/calibration record. Keeping
geometry separate avoids pretending that fitted `DIFC` uniquely determines a
physical flight path or bank angle.

## Unit 38 delivery sequence

The structural extension is split into reviewable numerical increments:

1. Add the explicit TOF neutron Lorentz correction with an independent NumPy
   equation, analytical reciprocal-metric derivative, invalid-angle tests, and
   centered finite differences.
2. Add typed TOF bank geometry and parse the independently documented
   scattering angle from bounded legacy instrument input.
3. Add a Rust structural-TOF calculation primitive that evaluates values and
   structural/profile derivatives in the same finite-support pass. Cover dense,
   JVP, and VJP products and a realistic multi-reflection benchmark.
4. Compose the primitive into a guarded multi-bank structural objective and
   solver with shared structure/cell and bank-local scale, background,
   instrument, geometry, masks, and uncertainties.
5. Expose the same native contract through Python and project persistence, then
   validate a checksum-pinned real structural dataset against an isolated
   pinned oracle.

No stage may claim “all TOF beamlines” merely because another profile function
fits. Acceptance requires a declared reduction/correction convention and bank
metadata for every dataset.

## ORNL/POWGEN acceptance boundary

For an ORNL/POWGEN structural request to be scientifically complete, it must
contain or reference:

- the measured TOF grid, mask, uncertainty, and background convention;
- the bank's calibrated profile and `tof(d)` coefficients;
- the bank scattering angle;
- a crystallographic phase and neutron scattering identities;
- whether vanadium normalization removed detector efficiency and the incident
  wavelength distribution;
- the selected Lorentz convention;
- any absorption, extinction, texture, or other sample corrections; and
- provenance and checksums for the data, calibration, structure, and reduction.

The existing POWGEN profile-only example establishes the first two items and
the general TOF kernel. It does not establish the remaining structural
intensity contract.

## References

- P. Jacobs et al., “A Rietveld refinement method for angular- and
  wavelength-dispersive neutron time-of-flight powder diffraction data,”
  *Journal of Applied Crystallography* **48** (2015), 1627–1636,
  <https://doi.org/10.1107/S1600576715016520>.
- J. Liu et al., “Lorentz factor for time-of-flight neutron Bragg and total
  scattering,” *Journal of Applied Crystallography* **56** (2023),
  <https://doi.org/10.1107/S1600576723002127>.
