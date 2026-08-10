# Effective starting-profile estimation

## Purpose and scientific scope

High-pressure synchrotron experiments often have a calibrated detector geometry
and wavelength from Dioptas/pyFAI, but no separate instrumental-resolution
standard measured under the same configuration. PhaseSmith therefore supports
a deliberately narrower operation: estimate an **effective starting profile**
from a predominantly single-phase integrated pattern and a CIF for its dominant
phase.

The result is suitable for obtaining stable first Le Bail/Rietveld fits when
precise peak-width interpretation is not the objective. It is not labeled an
instrument-resolution calibration. Crystallite size, microstrain, pressure
gradients, non-hydrostatic stress, and other sample contributions may be folded
into the fitted U/V/W/X/Y values.

This first slice does not include a bundled calibrant registry, automatic CeO2
or LaB6 geometry calibration, or in-situ Si separation. Those workflows need
real representative measurements and their own validation before being added.

## Inputs and ownership

The workflow consumes:

- an integrated one-dimensional `2theta` pattern with observed intensity;
- an optional fixed background, uncertainty, and inclusion mask;
- a constant-wavelength U/V/W/X/Y starting model;
- exactly one dominant CIF-backed Le Bail phase; and
- an optional bounded lattice-reflection domain for nuisance cell alignment.

The wavelength normally comes from the `.poni`/Dioptas calibration. It is never
inserted into the estimator parameter set. Native and Python boundaries verify
that its floating-point bit pattern is unchanged.

Known diamond, gasket, impurity, detector-artifact, and other unidentified peak
regions should be excluded with `PowderPattern.mask`. Automatic unexplained-peak
detection is not part of this first slice.

## Profile convention

The constant-wavelength Gaussian variance and Lorentzian component FWHM are

\[
\sigma_G^2(\theta) = U\tan^2\theta + V\tan\theta + W
\]

and

\[
H_L(\theta) = X\sec\theta + Y\tan\theta.
\]

They feed the existing Thompson-Cox-Hastings pseudo-Voigt transform. U, V, and
W have units degree squared; X and Y have units degree. Values and analytical
derivatives are evaluated by the existing fused Rust profile kernel in the same
sample pass. Finite support retains the Le Bail convention selected by
`LeBailOptions.support_fwhm`.

An optimizer seed can be made from a rough Gaussian FWHM `H`:

\[
W = \left(\frac{H}{2.3548200450309493}\right)^2,
\qquad U=V=X=Y=0.
\]

This is only a starting value.

## Staged workflow

1. If requested, align the symmetry-independent bounded lattice variables with
   all profile coefficients fixed. This is a nuisance alignment, not a
   wavelength refinement.
2. Fit `W` while Le Bail redistribution supplies independent non-negative
   reflection intensities.
3. In automatic mode, try `U/V/W`.
4. Only if that model is accepted, try `U/V/W/X/Y`.
5. Return the last accepted model, fitted independent intensities, calculated
   pattern, covariance diagnostics, warnings, and a complete accepted/rejected
   stage audit.

Automatic complexity increases require all of the following defaults:

- relative Rwp decrease of at least `0.002` (0.2%);
- absolute Rwp decrease of at least `1e-5`;
- an identifiable full-rank candidate covariance; and
- maximum absolute pairwise covariance correlation no larger than `0.98`.

The absolute threshold prevents machine-precision residual changes from looking
important merely because the baseline Rwp is already nearly zero. Explicit
`UVW` and `UVWXY` modes are available as caller overrides and record that the
model was forced rather than selected automatically.

## Python workflow

```python
import phasesmith
from phasesmith.refinement import lebail

pattern = phasesmith.PowderPattern(
    two_theta_deg,
    observed_y=intensity,
    background=background,
    uncertainty=sigma,
    mask=included,
)

# wavelength_angstrom is taken from the prior Dioptas/pyFAI calibration
start = phasesmith.starting_profile_from_fwhm(wavelength_angstrom, fwhm_deg=0.04)
request = lebail.LeBailInput.from_cif(
    pattern,
    start,
    "dominant-phase.cif",
    phase_id="dominant",
    refine_lattice=True,
)
result = phasesmith.estimate_effective_profile(
    request,
    phasesmith.ProfileEstimationOptions(align_lattice=True),
)

print(result.instrument)
print(result.active_parameters)
for stage in result.stages:
    print(stage.kind, stage.accepted, stage.decision)
```

The Python call delegates profile fitting to the Python-free native workflow.
Optional lattice alignment currently reuses the independent scripting Le Bail
implementation, then passes the aligned fixed reflection geometry to Rust. A
native application can call `estimate_effective_profile` in
`phasesmith-workflows` directly with a bounded dynamic `LeBailPhase`, so it does
not require Python.

## Validation boundary

Synthetic deterministic tests cover W recovery, fixed-wavelength preservation,
masking of an unmodelled contaminant peak, automatic rejection of unsupported
extra terms, and invalid-input behavior in both Rust and Python. No real sample
has been supplied for this feature yet. Real-pattern acceptance, default tuning,
and any future Si/CeO2/LaB6 workflow remain explicitly pending a representative
dataset rather than being inferred from synthetic data.
