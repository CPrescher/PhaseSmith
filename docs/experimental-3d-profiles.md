# Learned three dimensional reflection profiles

**Supported namespace:** this method is included in the 0.8 single-crystal
capability. Use `phasesmith.single_crystal`; ordinary builds include the native
kernels. The historical filename is retained for existing links. See the
[supported guide](single-crystal.md) for scope and migration. Numerical equations,
finite support and the specific limitations below remain unchanged.


Opt-in `experimental-spots`. This increment learns nonnegative empirical voxel
masses in detector x/y and a scaled rocking coordinate from isolated, strong training
reflections. Detector-region/rotation-region profiles can fall back to a global
profile with explicit model IDs/counts. The default is a detector-coordinate
model; optional scattering-plane transforms and strong-only width transfer are
described below. Neither implies numerical equivalence to DIALS.

## Scattering-plane and frozen-width extension

`scattering_transforms(xy, geometry)` returns a per-reflection 2 x 2 Jacobian A
in degrees/pixel, plus signed zeta. With unit outgoing ray S and incident S0,
e1=normalise(S cross S0), e2=S cross e1, and
A_ij=(180/pi) (e_i dot pixel_axis_j)/|ray|. The pixel footprint maps to
u_xy=diag(scales_xy) A (pixel_xy-centre_xy); the angular interval maps to
u_omega=scale_omega (omega-centre_omega). This is the first-order local
scattering frame of Kabsch (2010), not an exact nonlinear mapping over a detector
pixel. Direct-beam singularities and singular/ill-conditioned A are rejected.

For a finite voxel of mass p_j, its contribution is p_j times the intersection
area of the mapped parallelogram and voxel rectangle divided by voxel area,
times the angular overlap divided by voxel width. Rust uses fixed-size polygon
clipping; the independent NumPy reference enumerates candidate vertices instead.
Identity A takes the original separable path. Reflected/sheared transforms retain
mass. Centre derivatives use rectangle-boundary chord fluxes, followed by the
Jacobian chain rule; mass and partials share the same sample pass. Contact-only
intersections have zero mass. Strict-edge derivative conventions return zero at
coincident boundaries; centred finite-difference checks avoid those points.

Width component k has positive factors f_k and density
p_k(u)=p_0(u/f_k)/product(f_k), with the finite support stretched by f_k.
The default four components use base, 1.6 spatial stretch, 1.6 angular stretch
and both. Each retains unit finite mass; missing pixels never renormalise it.
Strong-reference fitting imposes a common coefficient sign, so component
fractions q_k=beta_k/sum(beta) are nonnegative and sum to one even for signed
intensity. Up to four components use explicit constrained-face enumeration.
Free shape fits are allowed only with `allow_shape_refinement=True` and should
be limited to training/diagnostics: they are biased for weak targets.

`learn_profile_variations` uses actual clean base-profile reference IDs, keeps
strong fitted measurements above the requested SNR, and stores their fractions
and reference prediction positions. `fit_learned_profiles` transfers robust median
fractions from up to 16 nearest references in normalised x/y/omega coordinates;
self-references are omitted on the same source fingerprint. Targets outside the
teachers' angular range receive `SHAPE_EXTRAPOLATED`. Spatial reference density
and profile-learning uncertainty still need further validation.

Target intensity is one signed scalar with P_eff=sum(q_k P_k), with fractions
frozen independently of the target pixels. Thus I=(P_eff' C^-1 d)/(P_eff' C^-1
P_eff), Var(I)=1/(P_eff' C^-1 P_eff); component coefficients are I*q and their
conditional covariance is Var(I)*q*q'. Raw Poisson fitting iterates expected
variances; supplied variances stay fixed. Fit masks, overlap flags and observed
mass use the actual frozen P_eff, including only active components. Covariance
conditions on shape fractions and all other learned quantities. The free-fit
covariance is an unconstrained GLS approximation, not selection uncertainty.

The extension is explicit and leaves the original fixed-profile default intact:

```python
from phasesmith.single_crystal.profiles import (
    scattering_transforms,
    ProfileOptions,
    prepare_profiles,
    learn_profiles,
    learn_profile_variations,
    fit_learned_profiles,
)

A, zeta = scattering_transforms(predictions.values[:, :2], geometry)
# First exclude near-spindle predictions consistently from predictions/A/zeta.
observations = prepare_profiles(
    load_frame,
    omega_bounds,
    predictions,
    mask_loader=load_mask,
    saturation=65535,
    options=integration_options,
    planar_transforms=A,
    scales=np.column_stack([np.ones((len(zeta), 2)), zeta]),
    angular_half_widths=np.minimum(10.0, 2.6 / abs(zeta) + 0.5),
)
base = learn_profiles(
    observations,
    training_mask,
    detector_shape=(height, width),
    options=ProfileOptions(
        shape=(19, 19, 13), step=(0.025, 0.025, 0.25), regions=(1, 1, 1), cutoff=0.0
    ),
)
shapes = learn_profile_variations(observations, base)
result = fit_learned_profiles(observations, base, shapes)
```

These numerical choices describe the Fe7C3 investigation, not universal presets.
A profile's `reference_centres` preserves its measured spatial reference centres;
the angular coordinate remains the prediction. Equal reference weighting is the
default; optional inverse-variance weighting weights normalised profiles by
summed_intensity^2/summed_variance. It is explicit and independently tested, but
was not selected for the final two-scan experiment. All method settings, teacher
membership and finite support must accompany exported measurements.

## Finite support and equations

The profile grid has odd dimensions (nx,ny,nomega), with steps in local coordinates
and edges centred around zero. A voxel contains probability mass p>=0 and
sum(p)=1 **within this finite grid**, with zero density outside. A source sample
is the pixel rectangle [x-.5,x+.5) times [y-.5,y+.5) and its full angular
exposure interval. Its predicted mass is P=sum(pj*overlap_volume/voxel_volume).
No fractional exposure scaling is applied to observed counts. Irregular widths,
reverse scans and fractional centres are represented by interval intersections.

Each reflection supplies signed coordinate scales, defaulting to one. Bounds
are mapped by u_j = scale_j * (coordinate_j - centre_j) and sorted, preserving
mass even for negative scales. Position derivatives include the chain-rule
scale. The measured comparison uses (1,1,zeta), where zeta = spindle dot
(outgoing cross incident)/|outgoing cross incident|. Thus the local rocking
coordinate is zeta*(omega-omega_B), in degrees. This first-order geometry follows
[Kabsch 2010, section 2.3](https://journals.iucr.org/d/issues/2010/02/00/dz5178/index.html).
Detector x/y are not rotated into the full local reciprocal-space basis.
Near-spindle |zeta|<0.05 crossings are excluded explicitly in this benchmark,
with their IDs/HKL retained; they are never assigned an artificial minimum
rocking width. Per-reflection angular collection widths are retained in the
summed table, independent of the profile-grid support.

For interval [a,b] and shifted voxel [l+s,h+s], overlap is
max(0,min(b,h+s)-max(a,l+s)). Away from coincident edges, its derivative is
1[h+s<b]-1[l+s>a] while the overlap is positive, otherwise zero. The native
value and three position derivatives share the same voxel/sample pass. Support
boundaries are piecewise differentiable; tests avoid edge coincidences.

Strong-reference profiles are recentered in x/y using signed signal centroids, then
pooled after division by their measured summed intensity. Iterative projected
least squares reverses the known pixel/exposure averaging operator rather than
assuming each broad exposure measured a point. A diagonal majorizer bounds
the least-squares update. Nonnegative projection and a connected component of
the learned central profile above an explicit relative cutoff define support.
The angular centre stays at the geometric Bragg prediction: the midpoint-weighted
centroid of a coarse rotation image is not an unbiased Bragg-angle measurement
(Kabsch 2010, section 2.4). The final finite profile is normalised. Reference membership, local fallback,
centroid offsets and learned values are retained. Weak observations never teach
the profile and their counts are never selected by a positive-pixel threshold.

For background-subtracted sample vector d, model P and fixed covariance C,
I=(P'C^-1 d)/(P'C^-1 P), Var(I)=1/(P'C^-1 P). I remains signed. Each exposure's
C=diag(V)+X Cov(background_plane) X' retains its shared-background covariance;
a small Woodbury solve avoids a dense pixel covariance. The background and
profile must have disjoint measurement supports. Raw-count IRLS can update the
independent V from fitted background+max(I,0)*P; supplied variances stay fixed.
Reported derivatives and uncertainties condition on those weights, profile,
reference selection, clipping and centres. They exclude profile-learning,
centroid, geometry, inter-reflection and detector-correlation uncertainty.

Coverage is the mass of the learned profile actually observed in valid pixels
and frames. Missing/saturated samples do not renormalise P. Their flags remain,
and inadequate coverage prevents an unqualified fit. Fits and profile flags use
the learned positive foreground only; collection-box overlap or saturation
outside it does not invalidate the profile amplitude. Conversely, actual shared
foreground pixels are flagged. Fitted backgrounds use a disjoint guarded ring
around the collection rectangle. Summed intensities are the
original finite-aperture sums, with their own covariance and flags. Neither
estimator includes LP, absorption, efficiency, symmetry merging or scaling.

## Source basis and validation

Profile fitting against learned normalised shapes follows the general model in
[Leslie 2006 §5.4](https://journals.iucr.org/d/issues/2006/01/00/ba5082/).
[DIALS 2018 §3.4](https://journals.iucr.org/d/issues/2018/02/00/di5011/index.html)
motivates local profiles and a separate summation estimator. The finite-box
operator, projected training solver and covariance equations here are derived
independently. No external implementation is copied. The GSAS-II powder oracle
does not cover these image estimators; independent NumPy equations, synthetic
known counts, mass/moment checks and a black-box DIALS comparison supply evidence.


## Programmable use

```python
import numpy as np
from phasesmith.single_crystal.integration import IntegrationOptions
from phasesmith.single_crystal.profiles import (
    ProfileOptions,
    rotation_factors,
    prepare_profiles,
    learn_profiles,
    fit_profiles,
)

# predictions is a ScanPredictions table for the frozen crystal and scan.
# Select |zeta|>=0.05 predictions before preparing this normalised model.
zeta = rotation_factors(predictions.values[:, :2], geometry)
scales = np.column_stack([np.ones((len(zeta), 2)), zeta])
profile_options = ProfileOptions(shape=(19, 19, 13), step=(1.0, 1.0, 0.25))
observations = prepare_profiles(
    load_frame,
    omega_bounds,
    predictions,
    mask_loader=load_mask,
    saturation=65535,
    options=IntegrationOptions(
        pixel_radius=11,
        omega_half_width=4.5,
        background_model="plane",
        background_gap=3,
        background_margin=6,
    ),
    scales=scales,
    angular_half_widths=np.minimum(10.0, 1.625 / abs(zeta) + 0.5),
)
model = learn_profiles(
    observations, training_mask, detector_shape=(height, width), options=profile_options
)
result = fit_profiles(observations, model)
# result.values: I, sigma, observed_mass, reduced_chi2, foreground_samples,
# correlation, and four conditional objective derivatives (I,x,y,omega).
# result.summation preserves the separate signed rectangular measurement.
```

`training_mask` must reserve independent reflection intensities. The learner
additionally rejects masked, saturated, overlapping, incomplete, background-failed,
weak, poorly centred and endpoint-dominated references. Each model stores the
actual reference IDs, counts, local offsets and source fingerprint. Small local
banks explicitly fall back to bank zero. Input and model arrays are read-only
snapshots. Repeated training/fitting is deterministic.

## Fe7C3 comparison status

The implemented estimator passes independent synthetic equations and known-count
checks. The measured TIFF comparison **fails the declared intensity-agreement
gate**. It must remain experimental; low profile residuals or clear flags alone
do not establish intensity accuracy. See measured comparison (`fe7c3-3d-profile-comparison.md`, local research report).

The bias investigation (`fe7c3-profile-bias-investigation.md`, local research report) implements and tests
scattering-plane geometry and strong-only frozen width transfer. P2 passes the
new family-holdout comparison gate; P1 does not. These changed populations do not
replace the historical 40-pair result. Both estimates remain experimental.

Remaining work includes further spatial shape and angular
shape between reflections and scan blocks, together with the effect of local
background/covariance modelling. Full reciprocal-space spatial coordinates,
profile uncertainty, cross-validation of local templates, joint overlap fitting,
and crystallographic correction/scaling remain separate work.
