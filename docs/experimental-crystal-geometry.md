# Indexed single-crystal geometry calibration

**Supported namespace:** this method is included in the 0.8 single-crystal
capability. Use `phasesmith.single_crystal`; ordinary builds include the native
kernels. The historical filename is retained for existing links. See the
[supported guide](single-crystal.md) for scope and migration. Numerical equations,
finite support and the specific limitations below remain unchanged.


The opt-in `phasesmith.single_crystal.geometry_refinement` API refines detector
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

Omega detector partials vanish. Rust now applies all Euler/length/translation
chains, evaluates the robust objective/gradient/normal, optimizes the bounded
parameters and computes eligibility, covariance/condition and withheld diagnostics.
Python is an optional validated NumPy boundary/result adapter; production geometry
fitting does not import SciPy. The original predictor retains its existing values,
flags and B derivatives. The independent NumPy/SciPy reference calls no Rust.

For each coordinate residual `r=(prediction-observation)/scale`, the unchanged
soft-L1 objective is `F=f² sum(sqrt(1+(r/f)²)-1)`. We evaluate the equivalent
`sum(r²/(sqrt(1+(r/f)²)+1))` to avoid cancellation near an exact solution.
With physical-parameter Jacobian J, the gradient is
`J.T @ (r/sqrt(1+(r/f)²))`. The robust Gauss--Newton normal is
`J.T @ diag((1+(r/f)²)^(-3/2)) @ J`. Values and exact first partials use the
same predicted reflection; no finite differences enter the native optimizer.

The native solver uses deterministic damped normal steps in scaled coordinates:
0.001 rad for all angles, 0.1 mm for translations and 0.001 for log lengths.
The closed bounds and loss scales are unchanged. Variables at a bound with an
outward descent direction are fixed for that step; other steps are clipped to
the bounds. Initial damping is 1e-3, added to `max(abs(diagonal),1)` of the
scaled normal. Accepted steps strictly decrease F. Actual/predicted reduction
ratios above .75 divide damping by three; ratios below .25 multiply it by four.
Rejected/invalid trials multiply damping by ten; damping above 1e24 stops without
convergence. Cholesky failures increase damping without evaluating another model.
The declared budget includes initial and rejected/invalid objective evaluations.

Convergence uses scaled projected-gradient infinity norm <1e-9, scaled step norm
<=1e-11*(1e-11+scaled-parameter-norm), or relative cost reduction <=1e-11 with
reduction ratio >.25. These differ algorithmically from the former SciPy bounded
trust-region iterations; equivalent scientific outputs, not bitwise fit vectors,
are required. Accepted costs and evaluation count are observable. The scaled
normal eigensystem gives the Jacobian condition; the unscaled robust normal's
Moore--Penrose inverse uses a local 1e-15 relative eigenvalue cutoff. This is
conditional curvature, not a calibrated observational uncertainty. A parameter
within `1e-11*max(1,abs(bound))` of a bound prevents advisory acceptance.

Native Rust callers use `phasesmith::spots::geometry_refinement::{Input, Options,
fit}`; `fit_with_map` accepts a caller-owned Rust `PixelMap` implementation.
The ordinary affine panel takes the identity path without mapping callbacks.
The optional Python binding retains application-owned `pixel_map` protocols by
bridging their batch inverse coordinates/Jacobians; those callbacks are supplied
by the application and are not required for the native solver or MgSiO3 example.
Progress callbacks likewise report native evaluation boundaries and may cancel
by returning an error. Python-held masks, options, observations and returned
arrays remain snapshotted; no host numerical optimizer is called.

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

## Native fitting validation

Standalone Rust tests recover known detector pose, crystal orientation and
optional reciprocal-row log lengths, including an independently supplied native
nonlinear coordinate map. They check the soft-L1 objective gradient by centered
finite differences, closed bounds, evaluation budgets, cancellation and withheld
independence. Independent NumPy implicit projection solves compare both analytical
Jacobians and parameter chains on deterministic random cases; a SciPy control
checks noisy/outlier fits. A blocked-SciPy-import test exercises the production
fitter. No profile term, intensity normalization or integrated moment changes,
and GSAS-II's powder oracle does not cover this position-only calibration.

The complete-fit benchmark now covers the same 10,888-spot/2048-square detector
workload with three deterministic repeats and an optional earlier SciPy runtime
control. Frozen MgSiO3 v9 changes only the geometry backend, with explicit
comparison tolerances in `validation/enstatite-raw-workflow-v9.json` and all
v8 science settings retained. Full original data/products remain external.

On the current macOS workload with OMP/OpenBLAS/Accelerate thread limits all one,
the complete native fit takes 22.96 ms median versus 34.53 ms for the earlier
Python/SciPy fitter (three repeats; all timings retained). Native repeats are
bitwise deterministic; predicted coordinates agree with the earlier fitter
within 1.82e-12 pixels and 5.69e-14 degrees. The unchanged predictor value/B
Jacobian fingerprint also matches the earlier native binary exactly. This is a
synthetic many-reflection benchmark, not a measured-data accuracy certificate.
See `benchmarks/geometry-refinement-rust-macos.json`. Workspace verification
passes 505 Rust tests (34 ignored), 3,225 Python tests (11 skipped, 34 deselected),
formatting and strict all-target/all-feature Clippy.

The complete original-frame MgSiO3 v9 run passes every frozen v8 equivalence gate.
Indexed/source arrays, exported HKLs, use mask and flags are identical; 246
teachers and 4,125 signed observations (220 negative) are retained. The maximum
eligible indexed-prediction change is 2.13e-7 pixels / 6.18e-8 degrees. Native
geometry uses seven evaluations, with withheld RMS .33311 pixels/.165893 degrees.
Intensity/sigma relative L2 changes are 9.21e-6/6.71e-6; the largest intensity
change is .009495 baseline sigma, with 95th percentile 5.79e-7 sigma.

The search again infers Mg16 Si16 O48 at amplitude R1 .0410522 (v8: .0411159).
Maximum/RMS reference errors are .0069006/.0046574 A. After origin/hand alignment,
maximum/RMS candidate-coordinate differences from v8 are .0005351/.0002411 A.
The full worker takes 94.53 s. No reference coordinates/counts, source HKLs or
archived arrays enter the worker; grading and candidate comparisons occur after
it exits. All original source files remain unchanged. These are equivalent
results within frozen tolerances, rather than bitwise identity of fitted
intensities/atomic coordinates. This remains exposed prior-assisted development
data and does not establish independent detector metrology or general solve rate.

A clean NumPy-only installed wheel recovers a known geometry/reference cell with
no SciPy/h5py/hdf5plugin installed. Rust standalone tests independently exercise
the same solver and native nonlinear maps. Reports are
`benchmarks/enstatite-native-geometry-macos.json` and
`benchmarks/geometry-refinement-rust-macos.json`; large data/products stay in the
external directories recorded there. The raw example still uses Python for
orchestration and the existing strong-profile/empirical-scaling optimizers.
