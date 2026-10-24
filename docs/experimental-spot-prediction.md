# Experimental rotation prediction and positional refinement

This adds no material/structure prior, space-group assignment, intensity model or
standard-release API. It consumes the **blindly inferred** lattice and tentative
HKLs from `experimental.indexing`. Instrument geometry, wavelength, omega-zero
convention and axis are held fixed; nine reciprocal-basis coefficients refine
the unconstrained cell and orientation together. A separate optional
[orthorhombic metric constraint](experimental-orthorhombic.md) uses six parameters
and retains the exact integer lattice setting. No GSAS-II/DIALS runtime or
implementation code is used. The equations below follow elastic scattering and
ray/plane geometry; the centroid-refinement workflow is described by
[Waterman et al. (2016)](https://journals.iucr.org/d/issues/2016/04/00/gm5043/).

## Forward model and fused analytical derivatives

Use the same right-handed laboratory/sample basis and zero-based pixel centres
as `experimental.reciprocal`. B has reciprocal vectors as **rows**, inverse Å
without 2π. For an integer row h, `g = h B`; direct rows are `A = inverse(B).T`.
Let b and a be the unit incident beam and rotation axis, λ the wavelength in Å,
and R(a,θ) a right-handed sample rotation. The elastic Ewald equation is

```
F(g,θ) = b . R(a,θ)g + λ |g|² / 2 = 0.
c = (b.a)(a.g)
u = b.g - c; v = b.(a × g); t = -λ |g|²/2 - c
θ± = atan2(v,u) ± acos(t / hypot(u,v))
```

Both crossings are available. `predict_spots` unwraps each root to the nearest
turn around a supplied measured-omega hint; the nearer root is selected, or the
other root when explicitly requested. Turn ties use `floor(turns + 0.5)` and root
distance ties select the plus-acos root. Hints are bounded to ±10^8 degrees.
Zero reflections/nonisolated roots and impossible Ewald intersections are
flagged `NO_CROSSING=1`. Tangencies with
`|b.(a × Rg)| <= 10^-10 max(1,hypot(u,v))` are flagged `TANGENT=2`; their implicit
derivatives are singular. A parallel/backwards panel intersection is
`NO_PANEL=4`. Invalid values/Jacobians are NaN with a nonzero flag, not invented
centroids. Root-choice boundaries and tangencies are excluded from smooth
derivative tests.

The outgoing ray is `s = b + λ Rg`, with unit norm at an Ewald root. For pixel
origin o, fast/slow step vectors f,d and plane normal n, the hit is
`p = s (n.o)/(n.s)`. Pixel coordinates solve `p-o = x f + y d` using the dual
pixel basis. Panel vectors are in metres; x,y in pixels; θ is returned in degrees.

The implicit angular derivative and detector derivative are

```
dθ/dg = -(R.T b + λg) / [b.(a × Rg)]
ds/dg = λ [R + (a × Rg) outer (dθ/dg)]
dp/dg = [(n.o)/(n.s)] [I - s outer n/(n.s)] ds/dg
dg_j/dB_ij = h_i
```

Rust evaluates x,y,omega and all **27 derivatives** with respect to row-major B
in the same reflection pass. NumPy provides an independent readable prediction
reference using sine-phase Ewald roots and least-squares panel projection.
Forward/inverse round trips, arbitrary tilted axes/panels, both roots, turns,
invalid cases and centred finite differences validate the implementation.

## Refinement and validation contract

Each measured item is one original detected-signal-weighted `(x,y,omega)` track
centroid. Its per-frame observations are **not** treated as repeated independent
Bragg events. In particular, a reflection detected over several exposures has
one predicted Bragg centre, not a separate Bragg crossing at every exposure
midpoint. Existing exposure-based centroids remain approximate; rocking widths,
threshold selection and mosaicity are not fitted.

The HKL assignments and training/held-out partition are frozen from blind
indexing. Initially indexed tracks are eligible if their initial prediction is
valid, detector displacement is strictly <25 pixels and omega displacement is
strictly <2 degrees. This eligibility mask stays fixed after fitting. ≥12
training and ≥5 held-out eligible centroids are required, along with rank-three
HKLs and a rank-nine training Jacobian. Excluded/unmatched rows remain exported.

Residuals are calculated minus observed. Scale by `(2 px, 2 px, 0.3 degrees)`;
these are explicit **loss scales, not calibrated standard errors**. The Euclidean
scaled residual per reflection has Huber loss with transition 2.5. Deterministic
IRLS/Levenberg damping fits only training centroids. No validation coordinate
contributes to an update, robust weight, stopping criterion or iteration choice.
Accepted costs decrease strictly; there are at most 40 attempts by default.
Reciprocal-basis drift is limited to 20% of its initial Frobenius norm, and basis
handedness cannot flip. No confidence intervals are claimed from this loss.

Report raw detector-vector RMS, omega RMS and combined scaled RMS on the same
fixed eligible training and validation populations, before and after. Invalid
final validation predictions give explicitly failed/infinite diagnostics and are
never dropped to improve apparent performance. A converged fit can fail to
improve validation. Dioptas keeps the **initial overlay** selected by default in
that case, while retaining the refined trial and comparisons for inspection.

The reported orientation matrix maps canonical orthonormal crystal coordinates
into the sample frame at the recorded omega zero. Direct canonical cell rows are
the lower-triangular Cholesky factor of `A A.T`. This is an explicit frame/setting
convention, not a known conventional cell or absolute instrument-zero calibration.

## Scan prediction, display and export

`predict_scan` enumerates integer HKLs inside a declared reciprocal sphere.
Bounds follow `|h_i| <= qmax |a_i|`, then exact `|h B| <= qmax`. Both Ewald roots
are tested against every exposure. The supported scan envelope is ≤180 degrees,
with at most 10,000 exposures. Raw HKL enumeration is capped at 250,000 and
frame-assigned predictions at 1,000,000 by default; budget failures are explicit
and never truncate a result. The native prediction call is capped at 1,000,000
HKLs, with each index in [-1,000,000,1,000,000].

Detector support is the half-open pixel footprint
`[-0.5,width-0.5) × [-0.5,height-0.5)`. Each exposure accepts
`min(start,end) <= omega < max(start,end)`, including reverse scans. Gaps remain
gaps and overlapping exposures can contain the same centre. These are point
Bragg-centre predictions, not finite rocking/partiality support. Absent, weak,
obstructed or inaccessible reflections can therefore be geometrically predicted.
There is **no** extinction, intensity, absorption or completeness model.

Dioptas derives qmax from the forward panel corners when the direct beam falls
inside the panel footprint; other layouts use the conservative Ewald diameter
and may hit enumeration budgets. The exact qmax, geometry and bounds are stored.
The overlay filters nonfinite pixels, enabled masks and optional zero padding
for the displayed frame. Exported scan predictions describe the rectangular
geometric panel before these display exclusions.

Cyan crosses show predicted Bragg centres in the exposure containing their
calculated omega. Measured per-frame circles can legitimately occur in adjacent
frames because of finite rocking width. Click a prediction for its tentative
HKL/omega. Initial/refined overlays can be switched without refitting. Six
residual plots show detector and omega trends with separate held-out colours.
Source/mask/detection/configuration changes invalidate predictions and plots;
the existing single-worker queue discards stale results.

Exports retain all selected track IDs, original centroids, frozen HKL/partition,
eligibility, initial/final predictions, residuals and failure flags; a separate
CSV lists possible scan reflections for both models. JSON stores both bases,
orientation, fixed geometry, qmax, loss settings, cost history and source/code
hashes. No integrated intensity is produced and original inputs are unchanged.

## Cost and scope

Native values plus derivatives are O(N) time/storage per prediction pass.
Refinement with nine fixed parameters is O(N I), with bounded iterations I and
small 9×9 solves. Scan enumeration costs O(H F) in the current orchestration for
H HKLs and F exposures, with explicit H/F/output bounds. The benchmark covers
roughly 5,000 reflections and full independent held-out refinement, with exact
repeat checks. No existing spot/profile loop changes. The GSAS-II powder oracle
does not cover rotation Ewald prediction or single-crystal centroid refinement.
