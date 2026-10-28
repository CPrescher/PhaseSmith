# Symmetry-constrained candidate refinement

This follow-up operates on a recovered full-cell candidate, the **caller-supplied
space group**, corrected observations and species. No archived atom coordinates
are accepted by origin recovery or refinement. It does not select a space group.

## Origin, orbit and coordinate constraints

For an operation `(R,t)` and an arbitrary-origin candidate `x`, search both hands
and a periodic origin shift `s` so `R(x+s)+t` matches a same-element point `x_j+s`.
Equivalently, the periodic matching residual is
`R x + t - x_j + (R-I)s`. Origin hypotheses use a bounded grid and iterative
same-element correspondence/linear least squares; complete bijections and
orbit multiplicities are checked before a model is accepted. Distances are
Cartesian shortest periodic images; excessively skew cells exceed an explicit
integer-search budget rather than using an incorrect componentwise distance.

After aligning, average inverse-related observations to one asymmetric site.
For its stabilizer, the affine projector is the group average of `R x+t-n`, with
the nearest lattice translation `n` fixed at identification. Its tangent is
`P=mean(R)`. A metric-orthonormal basis of `range(P)` parameterizes allowed
coordinate motion. The first site fixes only the continuously free origin
directions of the full group; physical coordinate directions remain free.
Orbit topology is fixed during polishing, and additional special-position
collapses are rejected. The supplied metric must satisfy the group: Cartesian rotations obey
`max(abs(Q^T Q-I)) <= 1e-8` at this boundary (normal solve preparation has its
own metric check). At most 192 operations and 128 full-cell atoms are accepted.

## Anisotropic displacement

Use Cartesian mean-square displacement `U` in angstrom squared and reciprocal
Cartesian vector `q=A^-T h` (no `2pi`). The Debye–Waller factor is
`exp(-2 pi^2 q^T U q)`. For an image rotation, `Q=A R A^-1`, the image tensor is
`Q U Q^T`. These conventions follow the
[IUCr ADP nomenclature report](https://www.iucr.org/who-we-are/commissions/commission-on-crystallographic-nomenclature/published-reports/adp).
Stabilizer constraints require `Q U Q^T=U`; the group-average invariant-tensor
projection follows Grosse-Kunstleve & Adams, J. Appl. Cryst. 35 (2002), 477–480,
[10.1107/S0021889802008580](https://doi.org/10.1107/S0021889802008580).

Project the six orthonormal symmetric Cartesian basis tensors by
`E -> mean(Q E Q^T)` over the stabilizer, then orthonormalize to obtain independent
invariant tensors `B_k`. Parameterize `H=sum(theta_k B_k)` and **`U=exp(H)`**.
This ensures positive definiteness and site-symmetry invariance simultaneously,
without retaining six redundant parameters on high-symmetry sites. Eigenvalues
of U must lie in `[1e-5, 0.2]` angstrom squared; invalid trial steps are rejected.

For `H=V diag(lambda) V^T`, the derivative of U is
`dU=V [K elementwise (V^T dH V)] V^T`, where
`K_ii=exp(lambda_i)` and
`K_ij=(exp(lambda_i)-exp(lambda_j))/(lambda_i-lambda_j)`.
The repeated-eigenvalue limit is evaluated continuously. For each image term,
`dF/dtheta=-2 pi^2 (v^T dU v) F_image`, `v=Q^T q`.
Coordinate derivatives are `2 pi i (h.R.b) F_image`. Values and all derivatives
are evaluated in the same reflection/image pass. Scale profiling and the
unweighted amplitude objective retain the preceding experimental conventions;
negative source intensities remain unchanged.

An independent NumPy implementation uses separate matrix eigendecompositions
and direct complex sums. Native/reference, finite-difference, isotropic-limit,
nonorthogonal-cell, site-symmetry, multiplicity and periodic-origin tests cover
this slice. F000 equals the full-cell sum of atomic scattering at zero q and
does not depend on ADPs. No powder profile/support equations are changed.

The old P1 polishing path remains available. A failed alignment/refinement
retains the P1 candidate and a reason; it must not silently label it as a
symmetry-constrained model. The density map retains its phasing frame, and the
model-to-density origin/hand transformation is recorded explicitly.

## Budgets, export and oracle boundary

There is no reciprocal-radius truncation in the ADP factor: every supplied
reflection is evaluated. The eigenvalue interval is inclusive, with 1e-12
rounding allowance on log eigenvalues; there is no clipping or altered derivative
at its boundary. Trial position collapses are checked at the explicit 1e-8
fractional expansion tolerance. Closest-image searches reject more than 4,096
integer translations; they never silently fall back to fractional wrapping.
Prepared models accept at most 100,000 reflections and two million dense
Jacobian entries. The high-level path additionally requires more independent
reflection families than fitted parameters plus scale; that count is a resource/
identifiability guard, not a rank or uncertainty certificate. ADPs have no reported
standard errors. Iteration/time/cancellation stops remain visible.

For CIF, with reciprocal lengths `b_i = |A^-T e_i|`, define
`M=diag(1/b_i) A^-1` and export `U_cif=M U_cart M^T` in order
11,22,33,23,13,12; `U_equiv=trace(U_cart)/3`. Unique image tensors and coordinates
are also exported in the density frame. Tests check the nonorthogonal anisotropic
exponent and both exported frames through the existing general native
structure-factor evaluator. Exact rational supplied operations accompany JSON
and CIF outputs. Continuous origin gauges and absence of anomalous data leave
handedness/origin ambiguities; the reported choice is deterministic within a build.

No GSAS-II fixture is regenerated: its powder oracle does not cover phasing or
origin recovery. The new fused ADP equations are checked against independent
NumPy sums/finite differences and the separate established general structure-
factor path, plus the checksum-pinned experimental recovery campaign. SHELX is
not called at runtime. No external implementation code was read or translated.

## Relationship to other programs

The phasing engine implements the published Oszlanyi–Suto charge-flipping
family, including its weak-reflection phase perturbation. It does not implement
SHELXT intrinsic phasing, Patterson-superposition starts, automatic space-group
search or SHELXS/SHELXD direct methods. See
[Oszlanyi & Suto 2008](https://doi.org/10.1107/S0108767307046028) and
[Sheldrick 2015, SHELXT](https://doi.org/10.1107/S2053273314026370).
The common sequence of solving in P1, aligning symmetry and polishing atoms does
not imply identical phasing or origin-search algorithms.

Refinement uses independently written dense damped Gauss–Newton least squares
with analytical coordinate/log-ADP derivatives and an eliminated amplitude scale.
Its target is normalized **unweighted amplitudes**, not the weighted F² target
used for routine SHELXL refinement. Symmetry constraints and the Cartesian
Debye–Waller tensor model are standard crystallographic equations; positive
log-tensor coordinates and this optimizer are our implementation choices.
No SHELXL weighting prescription, restraints, riding hydrogen model, occupancy
refinement, extinction/twin model, covariance/standard uncertainties or
Flack-parameter calculation is implemented in this slice. HKLF 4 compatibility
and optional black-box SHELXT verification are the direct SHELX interfaces.
