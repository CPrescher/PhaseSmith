# Scientific and numerical conventions

Public APIs prefer unit-bearing field names. Where an algorithm uses a shared
coordinate system, its documentation states that relationship explicitly.

## Units

- Constant-wavelength powder coordinates and peak positions use degrees
  `2θ`; angular profile widths are also degrees.
- Unit-cell lengths and wavelengths use ångströms; cell angles use degrees.
- Reciprocal-cell vectors use inverse ångströms without a `2π` factor.
- Scattering-vector magnitude follows `s = sin(θ) / λ` where documented by the
  scattering APIs.
- Neutron time-of-flight paths use the explicit units named by their fields.

For formula-level definitions, see the versioned
[equations and units guide](https://phasesmith.readthedocs.io/en/stable/equations/).

## Array and derivative layouts

Large numerical inputs use structure-of-arrays borrowed views. Constructors
validate equal lengths, finiteness, monotonic grids, and physical domains
before kernels run.

Profile accumulation stores local derivatives sparsely over each peak's finite
support. [`crate::core::SupportJacobian`] records the first active sample,
prefix offsets, and sample-major derivative rows. Dense materialization is an
explicit potentially large allocation. Shared parameters use a separate dense,
parameter-major [`crate::core::DenseJacobian`].

Structural and refinement APIs expose Jacobian-vector products (JVPs) and
transpose-Jacobian-vector products (VJPs) so optimizers and uncertainty tools
do not need to materialize a full Jacobian.

## Finite support and differentiability

Profile support includes samples exactly on its boundary. Analytical
derivatives hold the selected active sample set fixed; they do not differentiate
the discrete act of adding or removing a support sample.

## Determinism and concurrency

Parallel operations preserve input ordering. [`crate::execution`] uses an
operation-owned Rayon pool and never mutates the global pool. Floating-point
summation order is kept explicit where reproducibility is part of the contract.

## Validation and failures

Fallible public constructors reject non-finite values, invalid domains,
inconsistent shapes, and allocation overflow. File readers and persistence
loaders apply caller-visible resource limits before large allocations. Callers
should preserve structured error categories rather than matching display text.
