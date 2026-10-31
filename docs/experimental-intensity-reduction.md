# Experimental profile intensity reduction

The experimental rotation package has a portable reduction coordinator for learned
3D profiles, strong-only shape refinement, fixed weak estimators, explicit detector
response, declared sensor centroid maps, equivalent-based scaling and optional
observation-error floors. Image loading and application state belong to Dioptas;
structural refinement is a separate optional validation step.

This is an experimental workflow. R1 agreement does not establish absolute
intensity accuracy, identify detector hardware, or validate a crystal structure.

## Code ownership

| Module | Responsibility |
|---|---|
| `crates/phasesmith-spots/src/profiles_affine.rs` | Exact voxel/parallelogram/exposure masses and fused position/width derivatives |
| `crates/phasesmith-spots/src/intensity_model.rs` | Slab response, conditional GLS accumulation, frozen linear estimators, empirical values/Jacobians and family centering |
| `experimental/profiles.py` | Typed immutable raw shoeboxes, profile learning and fixed-profile fitting |
| `experimental/profile_refinement.py` | SciPy orchestration of declared strong shape fits; fixed weak extraction; centroid-map training adapter |
| `experimental/sensor_offsets.py` | Explicit sections, translations, optional in-plane rotations and three shared symmetric-gradient terms |
| `experimental/intensity_response.py` | Explicit homogeneous-slab assumptions and response provenance |
| `experimental/empirical_scaling.py` | Declared designs, robust within-family fitting, held-out scoring and shared regularization selection |
| `experimental/intensity_uncertainty.py` | Optional family-balanced equivalent-pair observation-error model |
| `experimental/reduction.py` | Configuration, orchestration, separated outputs and CLI |
| `experimental/reduction_io.py` | Pickle-free portable observation/profile-bank snapshots |
| `experimental/reduction_cache.py` | Immutable, atomic, content-addressed numerical artifacts |
| `experimental/intensity_reference.py` | Independent readable NumPy equations and dense small GLS reference |

Python experimental paths in this table are relative to
`python/phasesmith/`. None imports a research dataset or another experiment's
script. SciPy is optional orchestration; it is needed for shape/scaling fits.
The normal package requires no external integration or structure-refinement
program. Build native experimental functionality with
`maturin build --release --features experimental-spots` and install that wheel
into the consuming Python environment.

## Array and coordinate contracts

`ProfileObservations` retains the original `ReflectionTable`, signed signal,
independent pixel variance, background estimate, validity vector, group offsets,
per-group 3×3 background covariance, predicted/independently measured centres,
local scales and optional 2×2 pixel transforms. Invalid pixels have explicit
validity bits; signal significance never defines a weak observation's pixels.
The existing array validation and 20-million-sample budget apply.

Detector coordinates are pixel centres, x then y. Exposure and rocking coordinates
are degrees. Finite grids are x/y/rocking dimensions, stored as z/y/x volumes.
`profile_coordinates="detector"` uses pixel/pixel/degree steps. The Dioptas
`"scattering"` adapter uses degree/degree/degree local grids, a ray-direction
Jacobian and signed rocking zeta. It chains a declared fixed pixel coordinate map
through the local Jacobian. It adds no image flip, rotation or padding.

The selected geometry and the image loader must already agree on image
orientation. In Dioptas this includes the usual display flip. Attenuation is
specified at the selected geometry's actual wavelength; a mismatch is rejected.
Geometry fitting and parallax-coordinate conversion remain separate from count
extraction and intensity correction.

## Finite masses and analytical width derivatives

Each nonnegative voxel mass is integrated over its exact transformed pixel and
full exposure intersection. For a spatial voxel, mass fraction is polygon overlap
area divided by voxel area; the rocking factor is interval overlap divided by
voxel width. Finite support dilates with the grid when a width changes.

`evaluate_profile_shape` returns six columns:

1. mass;
2. derivatives with respect to detector-centre x, y and omega;
3. derivative with respect to log isotropic spatial width;
4. derivative with respect to log rocking width.

For overlap area fraction `w`, voxel bounds `lx,hx,ly,hy` and clipped boundary
chords `left,right,bottom,top`, the spatial log-width derivative is

```
(hx*right - lx*left + hy*top - ly*bottom)/voxel_area - 2*w
```

For rocking bounds `l,h`, sample bounds `a,b`, voxel width `s` and fraction `wz`,
the log-width derivative is

```
(h*1[h < b] - l*1[l > a])/s - wz
```

These terms and centre derivatives are evaluated in the same intersection pass.
Empty/touching intersections have zero mass. At exact coincident boundaries the
strict interior chord convention applies; finite differences must avoid these
nondifferentiable configurations. Complete enclosing boxes have unit mass and
zero width derivative. No target-specific tail clipping renormalizes an observed
subset to unit mass.

## Strong fits and fixed weak estimators

`refine_profiles(obs, bank, choices, ...)` requires explicit per-row choices:
`0` fixed, `1` strong centre plus spatial/rocking widths, `2` strong x/y centre
only. `ShapeFitOptions` declares bounds, regularization, optimization budget,
noise iterations and 1–4 independent workers. Native GLS marginalizes the fitted
background-plane covariance. Rust now owns the bounded profile optimizer,
counting-noise iterations, joint sigma and frozen weak filter; Python is an
optional array adapter. Native empirical scaling also owns the inner solve and
robust outer iterations. See [native fitting](native-profile-scaling-fitting.md)
for equations, independent references and validation. All rows and the
aperture baseline are retained. Failed conditional fits retain baseline values
with the profile failure bit.

The default `ReductionConfig.strong_policy` is `"fixed"`. Optional SNR rules are
explicit configuration; they are not a replacement for pixel/family holdouts
when deciding whether free shape parameters help a new dataset. Never select
shapes by R1, external integrated intensities or desired U values.

For weak updates, weights use a declared fixed expected count level (default
1000) and width (default spatial factor 0.95). Only valid samples with positive
finite-profile mass enter this estimator. The count amplitude stays signed and
linear in the measured signal; position and width are frozen. Reported counting
variance uses the actual independent pixel variances plus background covariance,
rather than the artificial design-count variance. These defaults record a study
policy; validate their suitability before applying them to another experiment.

A supplied sensor map predicts a **physical spot centroid**. The integration
adapter subtracts both the bank shift and the template first moment to obtain
its profile origin. Treating either one as the centroid produces a systematic
miscentering. Weak-fit residual/shape diagnostics that were not refreshed are
NaN, rather than mislabeled copies of the baseline diagnostics.

`calibrate_sensor_offsets` converts independently fitted strong centroids to a
map using a caller-declared training mask and `SensorLayout`. Layout bounds,
centres and coordinate scale are required. There is no camera-name inference.
Translations, in-plane rotations and three shared symmetric-gradient terms are
optional. Sensor fitting also accepts explicit `penalty_offsets` with shape
`(sections,4,2)` in pixels. These define the four spatial regularization probes
independently of assignment rectangles that may include inactive gaps; without
them the section half extents are used. Preserve the declared probes when
comparing calibrated maps. With `q=(xy-section_centre)/coordinate_scale`, the linear centroid
map is `delta = (tx,ty) + roll*(-qy,qx) + (sx*qx+shear*qy, sy*qy+shear*qx)`.
The roll coefficient divided by coordinate scale is the first-order rotation in
radians; the shared coefficients divided by that scale are dimensionless
symmetric gradients. The fit is empirical and needs held-out/independent-calibrant checks;
it is not independent module metrology. `SensorOffsets.to_dict()` produces a
JSON-safe frozen map for `ReductionConfig.sensor`.

For reproducibility, `frozen_parameters` can reapply saved strong positions and
widths without optimizing them again. Conditional joint sigmas are recomputed
from analytical derivatives. They can differ from historical finite-difference
sigmas near sharp voxel boundaries or poorly identified rocking widths even
when the intensities are unchanged. Do not claim bitwise uncertainty parity.

## Physical detector and polarization corrections

`SlabResponse` requires thickness in mm, attenuation in mm⁻¹, wavelength in Å,
material and provenance; optional density records the assumption used to obtain
attenuation. There is no runtime material lookup or detector-name guess.

For incidence cosine `c` and optical depth `tau=mu*t/c`:

```
eta(c) = 1 - exp(-tau)
relative(c) = eta(1)/eta(c)
mean_normal_depth(c) = c/mu - t/(exp(tau)-1)
```

The native pass returns all three values and cosine derivatives. Thin-limit
series and opaque limits avoid unstable subtraction/overflow. Public physical
responses require positive thickness and attenuation. This is uniform-slab
absorption, not measured counting quantum efficiency: threshold, fluorescence,
charge sharing and module electronics are outside the model.

The physical correction uses absolute `1/eta(c)` when a response is declared.
`relative(c)` is available for normalized comparisons. A global normalization
must remain explicit when comparing arbitrary intensity scales.

A declared transverse laboratory polarization axis and fraction define
`P=f*(1-(s1·e)^2)+(1-f)*(1-(s1·(s0×e))^2)`. The rotation-scan intensity correction
is `|axis·(s1×s0)|/P`; a powder-pattern solid-angle factor is not applied to
integrated point reflections. Geometry/polarization assumptions are saved.
The library does not silently add diamond absorption or a correction recovered
from a vendor's output chain.

## Empirical scales and equivalent errors

The declared model is

```
I_pre = J_family * F_frame * D(x,y) * A(incident_ray, outgoing_ray)
F = exp(X_frame*b_frame)
D = 1 + X_detector*b_detector
A = 1 + X_harmonic*b_harmonic
I_corrected = I_pre/(F*D*A)
```

Values and all coefficient derivatives are fused in Rust. Weighted family means
are nuisance parameters removed by native centering. Positive weak/strong
references, masks, bounds, neutral priors, Huber rule and fixed gauges are
explicit. Weak and negative rows receive the fitted correction but never teach
their own log scale. A positive floor defines the optimizer extension outside
the physical domain; the final solution must meet the declared positive bound.

`ScalingDesign` accepts arbitrary bounded typed designs, including the recorded
frame/bilinear-panel/degree-1+2 paired-ray design. There is no radial/B-factor term.
`select_common_strength` fits each integration separately under one model and
selects a common neutral-prior strength using complete validation families and
paired-family standard errors. Structural outputs never enter this choice.
`score_equivalents` reports family-balanced log residuals on reserved families.

The Dioptas recipe's optional `empirical` section requires explicit integer
intensity-equivalence operations and `symmetry_provenance`, plus detector shape.
Metric lattice symmetry is insufficient. Declare inversion/Friedel equivalence
if intended. It uses a deterministic complete-family holdout, fixed frame/node
gauges, neutral prior widths and the explicitly selected `ScalingOptions`.
The default reference relative-variance floor is 0.02 and can be declared as
`relative_variance_floor`. The general dense design currently has a 256-parameter
budget; longer scans need an explicitly reduced design.

Optional `equivalent_error_floor=true` also requires those explicit families.
It solves for a nonnegative beta such that a family-balanced weighted median of
absolute standardized training-pair differences equals the Gaussian median:

```
var(I_i) = statistical_sigma_i^2 + (beta*median_family_intensity)^2
r_ij = (I_i-I_j)/sqrt(sigma_i^2+sigma_j^2+2*(beta*mu_family)^2)
median_weighted(abs(r_ij))/0.6744897501960817 = 1
```

Only complete training families determine beta. Signed finite rows remain;
missing measurements remain missing. Total and statistical sigmas are separate
outputs. This observation-error model does not represent the correlated errors
of learned shapes, sensor maps, geometry or scale coefficients.

## Reuse and safe replay

```python
from phasesmith.experimental.reduction import ReductionConfig, reduce_observations

config = ReductionConfig()  # fixed profiles, no physical/empirical correction
result = reduce_observations(obs, detector_shape=(height, width), config=config,
                             model=bank, cache="disposable-profile-cache")
# result.aperture: unchanged counting baseline
# result.profile: raw signed profile amplitudes and flags
# result.intensity_sigma: explicitly corrected intensities and reported sigma
# result.statistical_sigma: conditional counting/strong-fit sigma before an error floor
# result.metadata: assumptions, hashes, masks, fitted coefficients and cache identity
```

`prepare_profiles` streams caller-supplied count frames once; use original counts
and explicit masks/noise rather than background-subtracted image caches for
counting uncertainty. `profile_geometry` supplies map-aware scattering transforms
when that coordinate model is selected.

`save_case(path, obs, bank)` writes one portable NPZ with JSON metadata and no
pickle, using exclusive creation. `load_case` validates its observation
fingerprint. Banks retain their original teacher fingerprint, allowing an
explicitly transferred bank to remain distinguishable from local training.

A safe fixed-profile CLI accepts a canonical case and JSON containing
`detector_shape` and `reduction`:

```sh
python -m phasesmith.experimental.reduction case.npz --config recipe.json --output fresh-output
```

The output directory must be new. Optional geometry vectors/wavelength belong
in the recipe when physical/scaling corrections are requested. The generic CLI
supports an affine geometry; application-owned nonlinear maps must be passed via
the Python API with their provenance. It does not silently substitute an affine
map when one is required.

`benchmarks/replay_intensity_recipe.py` additionally checks declared source-file
SHA256 values and accepts frozen per-row choices, weak membership, sensor
centroids and optional strong parameters. `benchmarks/replay_empirical_scaling.py`
checks a plain-array fixed scaling case. Outputs are fresh; neither imports
historical study scripts or overwrites their results.

Caches hash observation content, source-frame/mask fingerprints, all declared
settings, profile bank, teacher masks, sensor offsets, frozen parameters and
native/Python implementation identities. Entries have array schemas and payload
checksums, are installed atomically as directories, and are immutable for a key.
A corrupt/incomplete entry raises; a conflicting same-key write raises. Cache
reuse does not replace validation or the original evidence archive.

## Dioptas and validation boundaries

In the experimental Dioptas single-crystal intensity panel, select **Learned 3D
profiles (experimental)** and optionally load a reduction recipe. The existing
window controls specify collected pixels/exposures. Profile preparation requires
the local plane background. Save/load recipe controls retain profile/correction
assumptions; `.scw` also retains the window settings and selected method, with no
numerical result/cache restored from the workflow archive.

The table displays reduced profile intensities. Frame curves, per-image quality
and wider-window checks use the retained aperture measurements and are labeled
accordingly. CSV export contains separate aperture, profile, statistical sigma,
profile flags, physical factors and empirical factors. The normal aperture
workflow remains the default.

Independent NumPy values, centered finite differences away from support
boundaries, normalization, dense GLS controls, signed known-count recovery,
whole-family holdouts, cache corruption tests and GUI adapter tests cover the
new paths. `benchmarks/benchmark_intensity_kernels.py` measures a realistic
many-peak grid and a many-observation empirical pass. GSAS-II has no oracle for
this spot workflow; independent equations and synthetic measurements provide its
numerical controls. Optional real datasets are local regression evidence, not
shipped fixtures or a universal intensity-truth claim.

`benchmarks/shelxl_intensity_validation.py` is an optional local validator. It
requires an explicit self-contained HKLF4 atomic template, signed observations,
SHELXL executable and fresh output directory. It preserves input rows, template
bytes and weights, records executable/input hashes, recomputes R1(all) from FCF
and converts CIF Uij through the cell metric for Cartesian eigenvalues. It adds
no OMIT, restraints or automatic structural choices. Gemmi is needed only for
reading these validation outputs; SHELXL and its license remain external.
