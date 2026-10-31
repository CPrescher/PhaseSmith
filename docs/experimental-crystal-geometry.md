# Indexed single-crystal geometry calibration

The opt-in `phasesmith.experimental.geometry_refinement` API refines detector
pose and crystal orientation against confirmed indexed spot centroids. The
scientific basis is centroid refinement as described by [Waterman et al. (2016),
sections 2--4](https://journals.iucr.org/d/issues/2016/04/00/gm5043/index.html).
No reference intensities, structure factors, atomic coordinates, GSAS-II objects
or indexing reassignment enter this fit. Initial indexing is a separate step.

A declared reference cell can replace the starting metric while keeping its
canonical orientation. Otherwise the indexed candidate supplies the metric.
Optionally, three cell lengths vary with the initial angles fixed. This first
API does not enforce every Bravais equality or refine cell angles. A metric
candidate is not proof of material identity, space group or correct indexing.
Reference conditions and axis setting must be checked by the caller.

## Parameters and gauge

`B` has reciprocal vectors in rows, in inverse angstrom, without `2*pi`.
Observed and predicted records are `(x,y,omega)`, in pixels and degrees.
Geometry vectors are pixel-centre origin, fast/slow pixel steps (metres), beam
and spindle axis. The rigid detector rotates about its image-centre pivot and
translates in laboratory millimetres. All rotation parameters use radians in
`R=Rz @ Ry @ Rx`; the two rotations refer to detector and crystal separately.

For a base basis `B0`, `B = diag(exp(length_parameters)) @ B0 @ Rc.T` when
length refinement is enabled, and otherwise `B=B0 @ Rc.T`. For the detector,
`origin=pivot+translation_mm/1000+Rd@(origin0-pivot)`,
`fast=Rd@fast0`, and `slow=Rd@slow0`.
Beam direction, wavelength, pixel metric and spindle axis stay fixed. Omega zero
is absorbed by crystal orientation, avoiding a duplicate degree of freedom.
Detector roll is relative to this spindle convention; rotating detector,
spindle and crystal together about the beam changes coordinates without
changing predicted pixels or omega. Strong correlations are reported.

## Fused derivatives and optimization

The existing Rust predictor solves the Ewald crossing and its B Jacobian.
The new entry point additionally computes detector-vector derivatives in the
same reflection pass. The implicit equation is

```
origin + x*fast + y*slow = t*ray
normal = normalize(cross(fast,slow))
Ki = dual_i - dot(dual_i,ray)*normal/dot(normal,ray)
d(pixel_i)/d(origin) = -Ki
d(pixel_i)/d(fast)   = -x*Ki
d(pixel_i)/d(slow)   = -y*Ki
```

Omega detector partials vanish. Python applies analytical Euler/length/translation
chains and orchestrates optional SciPy bounded soft-L1 minimization. Rust owns
per-reflection projection and fused partial accumulation; there are no new
per-reflection Python production loops. The original predictor does not evaluate
the extra derivatives and retains byte-identical values and B partials.

Residuals use explicit x/y and omega loss scales, not calibrated uncertainty
estimates. At least 30 training and 10 withheld indexed positions spanning 3D
are required. Initial gates are strict `<` inequalities, and work is bounded
by one million records. Parameter bounds are closed; a bound hit prevents
advisory acceptance. Convergence, scaled Jacobian condition, maximum parameter
correlation and withheld residuals remain observable. No optimizer stopping
condition establishes correctness of the supplied HKLs or reference cell.

## Partition, acceptance and export

Callers supply immutable snapshots of training and eligibility masks. Complete
HKL families should stay together; the Dioptas adapter conservatively hashes
absolute HKL triples without asserting intensity symmetry. Withheld positions
never teach the geometry. Earlier indexing/structure information can still make
such a check conditional rather than a completely independent experiment.

The result does not change the input calibration or indexed model. Its advisory
`acceptable` property requires convergence, no bound hit, scaled condition below
1e6, and improvement of the withheld scaled residual. Dioptas presents the
review before exporting or applying it. Saving maps displayed-image geometry
back through the recorded inverse array transform to the original PONI frame.
A real Dioptas import must round-trip the fitted vectors. Original calibration
files cannot be overwritten by this action.

## Validation

The independent NumPy reference computes plane partials by inverting the implicit
3x3 projection matrix; it does not call Rust. Deterministic random cases compare
values and partials, finite differences check every detector component and every
physical parameter chain, and noiseless fixed-/variable-length experiments
recover known poses. Perturbed withheld positions leave fitted parameters
byte-identical. Invalid inputs and nonconverged fits are tested.

`benchmarks/geometry_refinement.py` covers 10,888 valid indexed reflections.
On the recorded macOS run, the original entry took 1.42 ms versus 1.54 ms for the
earlier core; values and B partials were byte-identical. The fused detector
entry took 1.76 ms. These are local workload measurements, not universal speed
claims. GSAS-II's powder oracle does not cover this experiment; independent
geometry equations, known-pose recovery and external reduction comparisons
supply the controls.
