# Experimental pixel coordinate maps

**Supported namespace:** this method is included in the 0.8 single-crystal
capability. Use `phasesmith.single_crystal`; ordinary builds include the native
kernels. The historical filename is retained for existing links. See the
[supported guide](single-crystal.md) for scope and migration. Numerical equations,
finite support and the specific limitations below remain unchanged.


`PanelGeometry` retains the native affine source plane and optionally accepts
an application-owned `pixel_map`. The native Rust reciprocal, Ewald projection
and fused analytical derivative kernels are unchanged. Camera interpretation,
response parameters, serialization and image conventions stay outside the core.

A map supplies three vectorized methods on finite real `(n,2)` arrays:

- `to_plane(image_xy)` converts measured image coordinates to the source plane.
- `from_plane(plane_xy)` returns image coordinates for predicted plane crossings.
- `inverse_jacobian(plane_xy)` returns the analytical `(n,2,2)` inverse chain.

Reciprocal conversion applies the forward map before the native calculation.
Prediction applies the inverse map to valid native crossings, then chains the
first two rows of both reciprocal-basis and detector-vector Jacobians. Invalid
crossings keep their flags and NaNs. Refinement retains the same fixed map when
it changes the source detector pose. The map parameters are not refined or
silently reinterpreted as physical detector parameters.

For native prediction `p(B,G)` and inverse pixel map `F`, the image prediction is
`F(p_xy), p_omega`. Its derivative is `diag(F',1) J_native`. The independent
NumPy prediction, reciprocal and implicit detector-partial references include
the same declared coordinate boundary; they make no native calls.

The application must define boundaries and inverse ambiguity. The Dioptas
CrysAlis response uses integer anchors: within an anchor cell the correction is
a constant translation, so `F'=I`. At half-pixel jumps a derivative does not
exist. The jumps can produce overlapping inverse branches. Dioptas chooses the
valid branch nearest the continuous inverse, breaking exact ties by the
lexicographically smaller anchor; it exposes candidate counts for diagnostics.
This is a deterministic processing-coordinate convention, not a new physical
calibration. Tests exclude anchor/root-selection boundaries for centered
finite differences and test the boundaries separately.

Validation covers an independent reference, nontrivial affine coordinate-map
chains for both fused native derivative families, invalid callbacks, unchanged
native flags, fixed-map retention during detector refinement, and forward/
inverse reciprocal closure. Dioptas separately validates the actual nonlinear
response against saved CrysAlis mappings and integration/save/load consumers.
GSAS-II does not cover this detector-coordinate model and is not used.

Run `benchmarks/pixel_coordinate_map.py` for a deterministic many-reflection
throughput comparison. The baseline no-map route does not change arithmetic.
No powder profile, support, integrated intensity, or native accumulation code
is modified by this adapter.
