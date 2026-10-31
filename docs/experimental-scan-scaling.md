# symmetry-based scan scaling

**Supported namespace:** this method is included in the 0.8 single-crystal
capability. Use `phasesmith.single_crystal`; ordinary builds include the native
kernels. The historical filename is retained for existing links. See the
[supported guide](single-crystal.md) for scope and migration. Numerical equations,
finite support and the specific limitations below remain unchanged.


The ordinary 0.8 `spots` capability provides a deliberately small relative
scale model. It operates on intensities already corrected for the single-crystal
rotation Lorentz and polarization factors. Supply confirmed equivalent-family
labels within the same crystal; the API does not infer symmetry from a cell metric,
merge different grains, read atomic coordinates, or accept a reference intensity
table. Detector geometry and crystal predictions remain unchanged.

The decomposition into scan-dependent illumination and other correction terms
is described by [Beilsten-Edmands et al. (2020), section 3.1](https://journals.iucr.org/d/issues/2020/04/00/di5035/).
This implementation covers one smooth scale term only. It is neither a complete
DIALS scaling model nor an ABSPACK replacement. Absorption harmonics, detector-area
terms, radiation damage and outlier classification remain separate problems.

## Equations and conventions

For declared closed angular support `[lo,hi]`, define
`half=(hi-lo)/2`, `mid=lo+half`, `x_i=(omega_i-mid)/half`. The log model for strong
training observations is

```
y_i = log(I_i) = a_family(i) + beta*x_i + error_i
v_i = (sigma_i/I_i)^2 + relative_error_floor^2
base_weight_i = 1/v_i
```

The unknown family intercepts are eliminated by weighted within-family centering:
`dx_i=x_i-weighted_mean_family(x)` and similarly `dy_i`. Each IRLS step evaluates
`beta=sum(w*dx*dy)/sum(w*dx^2)`, then residuals `r_i=dy_i-beta*dx_i`, with Huber
weights `w_i=base_weight_i*min(1,huber*sqrt(v_i)/abs(r_i))`. A constant log term is
omitted to fix the arbitrary overall scale at the scan midpoint. References must
be eligible training observations with `I/sigma > minimum_snr`, at least two per
family and nonzero within-family angular spread. Singleton/angle-degenerate
families do not supply references. Missing counts are allowed only in rows excluded
from training. Insufficient angular information produces an error, not a zero slope.

The covariance is conditional on reference selection and final IRLS weights:
`Var(beta)=max(1,sum(w*r^2)/(n_reference-n_families-1))/sum(w*dx^2)`, with a
one-degree-of-freedom floor. It is not a robust sandwich estimate or the full
uncertainty of all processing/model choices. Log fitting is a strong-reference
approximation; it must not be used to fit weak or negative measurements.

All signed target measurements use the linear application

```
C = exp(-beta*x)
I_corrected = C*I
sigma_counting = C*sigma
dI/dbeta = -x*I_corrected
dI/domega = -beta*I_corrected/half
sigma_model = abs(dI/dbeta)*sqrt(Var(beta))
```

Rust evaluates the factor, value, conditional counting error and both analytical
partials in the same observation pass. Model error is returned separately because
it correlates different measurements; adding it independently to every pixel or
reflection would lose that covariance. Application accepts negative intensities
and exact zero. Both support endpoints are included, and extrapolation raises an
error. Counts are never thresholded during application. No missing flux or peak
tails are reconstructed by this scale term.

## Usage

```python
import numpy as np
from phasesmith.single_crystal.scan_scaling import fit_scan_scale

# family_ids: explicit confirmed equivalent-family integer labels, same crystal.
# Fold complete families together; do not randomly split individual equivalents.
training = family_ids % 5 != 0
model = fit_scan_scale(
    lp_corrected_intensity,
    lp_corrected_sigma,
    omega,
    family_ids,
    omega_range=(-35.0, 36.0),
    training=training,
    eligible=quality_flags == 0,
)
if not model.converged:
    raise RuntimeError("Relative scan scale did not converge")
valid = np.isfinite(lp_corrected_intensity) & np.isfinite(lp_corrected_sigma)
corrected = model.apply(
    lp_corrected_intensity[valid],
    lp_corrected_sigma[valid],
    omega[valid],
)
# Columns: factor, signed I, counting sigma, dI/dbeta, dI/domega, model sigma.
```

Family-label hashing and symmetry canonicalization are caller responsibilities.
Modulo labels above are an illustration; labels assigned in source order can
correlate with acquisition conditions, so real validation should hash canonical
HKLs and retain complete families. The enstatite experiment uses a fixed SHA256
family partition. Neither the scale fitter nor its native kernel loads external
intensities or structure factors.

## Validation and limitations

The independent NumPy implementation in `scan_scaling_reference.py` uses explicit
family regressions. Tests check known-scale recovery, deterministic randomized
native/reference parity, outliers, singleton/angular-degenerate groups, missing
excluded data, unchanged fits after perturbing held-out counts, sparse integer
labels, signed corrections, endpoint/extrapolation behavior, and centered finite
differences of both analytical partials. `benchmarks/scan_scaling.py` covers a
many-reflection workload with family repeats; it is not a scalar-only benchmark.
GSAS-II's powder oracle does not cover rotation-data scan scaling. Known-scale
fixtures and external crystallographic refinement comparisons are used instead;
no external implementation code was translated into the kernel.

The current enstatite study combines this term with detector/omega-local empirical
profiles. Its controlled R1 decreases from 7.18% to about 5.75% on the same 1,280
unique reflections (CrysAlis comparator 4.05%). The training-only atomic refinement
still gives a non-positive displacement tensor. This is an experimental progress
result, not complete intensity validation or a recommended automatic correction.
The data were previously inspected and the starting structure came from CrysAlis;
reserved-family scores are conditional checks, not an untouched new experiment.

An HKL-specific follow-up identifies a separate integration-centre error. Keeping
the CeO2 calibration, crystal model, background and local profile bank fixed,
using independently detected strong-spot x/y centroids changes two example
intensities from 18.09 to 35.77 and 18.96 to 36.39 (external comparator 37.34 and
34.61). Their profile-foreground reduced chi-squares decrease from 3.46 to 1.19
and 3.63 to 1.07. The foreground support can change, so these are diagnostic
residual scores, not a constant-ROI likelihood comparison. The same predeclared
strong-target rule across the dataset changes common-set R1 from 5.749% to 5.623%
and the conditional family-holdout score from 6.592% to 6.390%. The resulting
full atomic refinement has a non-positive O008 tensor; the training-only fit
has non-positive O007/O009 tensors. This centroid experiment is retained as an
offline diagnostic, not a production default. Weak targets retain predicted
centres; an intensity-dependent free position search is not applied to them.
It explains some individual errors, not the entire remaining discrepancy.

The next offline control transfers a smooth x/y residual field from 1,121 clean
strong training spots to fixed target centres. Complete HKL families are kept
together: folds 2--4 fit linear/quadratic candidates, fold 1 selects quadratic
only if its positional RMS improves by at least 5%, and folds 1--4 then fit the
final field. Fold 0 never teaches it. Transfer is limited to the training
teachers' convex hull in detector x/y and omega, with a three-pixel offset guard;
outside it the original regional profile centre remains. This is an explicit
empirical centre correction, not a revised physical detector calibration.
The original PONI, crystal basis, omega mapping, background and profile bank stay
fixed. No target-specific position optimization is applied to weak reflections.

On the same 1,280-reflection structural comparison, the transferred field changes
strong-reference R1 from 5.749% to 5.188%, with all full-refinement displacement
tensors positive. The conditional family-holdout score changes from 6.592% to
5.753%; its training-only refinement still has non-positive O007/O009 tensors.
Keeping the original intensity scale coefficients gives essentially the same
result (5.190% / 5.754%), isolating the position correction from scale refitting.
The effective-profile-centre RMS on 270 withheld strong spots changes from
0.447 to 0.317 pixels. Corrections reach 5,156 targets, including 2,468 weak ones;
58 previously clean targets acquire quality flags and remain in the frozen
structural comparison population. A paired control using measured centres for
clean strong targets and the field elsewhere gives 5.168% / 5.796%, with the
same conditional physicality warning. It does not establish a further gain.
These remain sample-specific offline experiments, not production defaults or
proof that the original physical calibration is wrong.

After the user clarified that enstatite served as CrysAlis's geometry calibrant,
a separate physical calibration experiment was authorised. It fits a rigid
six-parameter detector pose and three crystal-orientation parameters to the
1,121 strong training positions, provisionally fixing the supplied CIF cell
(a=8.8101, b=5.1824, c=18.2288 A), wavelength and pixel sizes. The 270 withheld
positions improve from 0.572 to 0.297 pixel RMS; a free-orthorhombic-cell control
gives 0.293 pixels. No empirical residual field or reference intensity enters
these fits. The original CeO2 file is retained. A separate PONI round-trips
through the actual Dioptas importer; noiseless synthetic geometry recovery gives
maximum prediction error below 1e-9 pixels.

Raw HDF5 re-extraction with the fitted geometry updates the prediction centres,
scattering Jacobians, rocking widths, background apertures and LP/diamond
corrections. With the same profile settings and scan-scale protocol, the same
1,280-reflection comparison now gives strong-reference R1=3.943%, versus
CrysAlis=4.051%, and conditional family-holdout R1=4.202%. All full and
training-only displacement tensors are positive. A fixed old-profile-bank
control with zero empirical centre offsets gives 3.892% / 4.040%, supporting
the geometry explanation without selecting that control as the primary result.
Eight originally flagged predictions move off-panel; no old clean cases or
common comparison HKLs are lost. Three flagged scan-endpoint predictions retain
raw measurements but have NaN scan-scaled values rather than extrapolation.
This is a source-specific calibration using a provisional source-derived cell;
similar structural R1 is not proof that individual intensities or physical
instrument parameters are independently more accurate than CrysAlis.

Lower R1 alone is insufficient: inspect family consistency, reference coverage,
profile mismatch, corrected signed distributions and physical displacement tensors.
Scale fitting can absorb illumination changes and some integration bias; it does
not identify the physical cause of a trend or validate every extracted intensity.
