# Single-crystal profiles, intensities and scaling

Native owners are `phasesmith-single-crystal::spots` integration, local-background, profile,
intensity and scaling modules; the typed workflow constructs their designs.
Independent Python reference implementations remain validation paths and do not
wrap native evaluation. This reference distinguishes finite detected counts,
conditional profile amplitudes and explicitly corrected/scaled intensities.

## Finite profile mass

A learned odd-sized grid contains voxel masses `p_j >=0` with `sum p_j=1` on its
**finite support**, and zero density outside. A voxel has piecewise-constant
density `p_j/volume_j`. For sample pixel/exposure box `Omega_i`:

```text
P_i = sum_j p_j volume(Omega_i intersect voxel_j) / volume_j
overlap([a,b],[l+s,h+s]) = max(0, min(b,h+s)-max(a,l+s)).
```

The sample spans its full angular exposure, not a point at the exposure midpoint.
Coordinate intervals transform by declared signed scales and are sorted after
mapping. Away from coincident edges, the shifted interval derivative is
`1[h+s<b]-1[l+s>a]` for positive overlap and zero otherwise. Contact-only
intersection has zero mass; each coincident edge contributes zero to this
strict-inequality derivative. Value and position partials share the same sample/voxel
pass. Smooth derivative tests avoid these boundaries.

For outgoing direction S, incident direction S0, and sample axis a, the local
scattering frame uses `e1=unit(S × S0)`, `e2=S × e1` and signed `zeta=e1·a`.
Its spatial Jacobian in degrees per pixel is:

```text
A_ij = (180/pi) (e_i · pixel_axis_j) / |ray|
u_xy = diag(scales_xy) A (pixel_xy-centre_xy)
u_omega = scale_omega (omega-omega_B).
```

This is a first-order local mapping, not exact nonlinear geometry across an
arbitrarily large pixel. Affine pixel footprints intersect voxel rectangles by
polygon clipping; boundary chord fluxes give centre partials with the chain rule.
Direct-beam singularities and ill-conditioned maps fail. No artificial minimum
zeta is substituted for near-spindle measurements.

Width variant k uses positive stretch factors `f_k` and density
`p_k(u)=p_0(u/f_k)/product(f_k)`. A strong-reference mixture has frozen fractions
`q_k>=0`, `sum q_k=1`, and effective mass `P_eff=sum_k q_k P_k`.
Weak target pixels do not learn their own width fractions. Profile learning,
reference membership, regional fallbacks and source identity remain recorded.

## Signed amplitudes and conditional covariance

For background-subtracted samples d, fixed covariance C and fixed profile P:

```text
I_hat = (P.T C^-1 d) / (P.T C^-1 P)
Var(I_hat) = 1 / (P.T C^-1 P)
C = diag(V) + X Cov(background_plane) X.T.
```

The low-rank background term is handled through a Woodbury solve, without a
dense pixel covariance. Background and foreground measurement supports must be
disjoint. The amplitude remains signed. For frozen mixtures, coefficients are
`I_hat*q` and their conditional covariance is `Var(I_hat)*q*q.T`.
Free nonnegative shape fitting is a separate explicit diagnostic/training mode;
its unconstrained covariance is not selection uncertainty.

Raw-count IRLS may update independent variances from
`background+max(I_hat,0)*P`; supplied variances stay fixed. Reported errors
condition on weights, background, profile, centres and selected teachers. They
exclude learned-shape uncertainty, geometry error, inter-reflection covariance
and detector correlations. A gain setting is not a measured photons/ADU calibration.

Observed profile mass is the mass in valid measured pixels/frames. Missing or
saturated samples do not renormalize P. Positive finite support is not a guarantee
that unrecorded tails are negligible. Summation and profile estimators retain
their own supports, flags and uncertainties. Shared foreground, inadequate
coverage and failed background fits remain visible rather than silently removed.

## Corrections and relative scaling

Uncorrected detector counts are not automatically |F|^2. Explicit correction
records supply the chosen geometric/Lorentz, polarization, response, efficiency
or empirical factors with their conventions and provenance. A DAC aperture mask
is an availability decision, not an absorption correction. There is no universal
correction for unknown detector response, partiality or sample absorption.

An empirical positive-teacher model has the form:

```text
log I_j = log J_family(j) + x_j.T beta + residual_j
I_corrected,j = I_j exp(-x_j.T beta).
```

Family means are nuisance parameters eliminated in the bounded native fit.
The design, robust weights, gauge constraints, teacher/held-out family partition
and allowed extrapolation define the actual statistical model. Nonpositive
observations cannot teach a logarithmic fit, but signed target measurements are
retained and scaled rather than clipped away. Uncertainty propagation and any
additional equivalent-error model must be named explicitly.

Profile integration follows the statistical models described by Leslie (2006),
DOI [10.1107/S0907444905039107](https://doi.org/10.1107/S0907444905039107),
and DIALS (2018), DOI [10.1107/S2059798317017235](https://doi.org/10.1107/S2059798317017235).
Scattering-frame/coarse-exposure conventions follow Kabsch (2010),
DOI [10.1107/S0907444909047374](https://doi.org/10.1107/S0907444909047374).
These are methodological sources,
not an implementation dependency or a claim of numerical equivalence to DIALS.
The measured `Fe7C3` comparisons retain their failed gates and development-data status.
