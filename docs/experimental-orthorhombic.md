# Experimental orthorhombic metric settings

This opt-in branch consumes only a position-derived lattice and its frozen HKLs.
No known unit cell, chemical identity, reference structure or processed UB matrix
is accepted. A near-orthogonal metric does not establish a space group or the
crystal's intensity symmetry. The search cannot prove that the input lattice is
primitive. Missing detections are not used as systematic absences.

## Integer lattice setting

Direct vectors are rows of A. An integer nonsingular setting matrix T gives
`A' = T A`, `h' = h T.T`, `B' = inverse(T).T B`, so `h' B' = h B` exactly.
The conventional cell has index `abs(det T)` and volume multiplied by that index.
The translations `n inverse(T) mod 1`, for integer direct-lattice n, identify
P, A, B, C, I or F centring. Index-2/4 settings retain this reciprocal-lattice
constraint in scan prediction: `h' inverse(T).T` must be integer (numerical
absolute tolerance 1e-8). This is a change of lattice coordinates, not a fitted
space-group extinction law. Other translation subgroups are not accepted.

An integer size reduction records row operations. Primitive coefficient vectors
within [-3,3]^3 (configurable bound 1–4) are sign-normalized and length ordered.
Pairs pass when `abs(u.v)/(|u||v|) <= sin(0.3 degrees)`; triples need all three
pairs to pass, determinant magnitude 1, 2 or 4, and the supported translations.
The tolerance is inclusive and configurable in (0,2] degrees. Right-handed
settings are ranked by lattice index, maximum angular deviation, then integer
matrix entries. No validation residual participates in setting selection.
This finite axis search is not a full Niggli/Le Page Bravais classification.
The definitions follow the direct/reciprocal coordinate identities above and
[IUCr cell conventions](https://dictionary.iucr.org/Unit_cell).

## Orthorhombic positional fit

`refine_crystal(..., symmetry="orthorhombic")` uses three positive reciprocal
lengths and three local sample-frame orientation increments. Normalized initial
reciprocal rows are projected onto their nearest orthogonal matrix by the polar
factor (SVD); row norms initialize the reciprocal lengths. This projected start
is not silently substituted for the original model in before/after residuals.

For a local step `(s, w)`, with rotation vector w in radians,

```
B_new = diag(exp(s)) B R(w).T
R = I + sinc(theta/pi) [w]x + 0.5 sinc(theta/(2 pi))^2 [w]x²
partial B_i / partial s_j = delta_ij B_i
partial B_i / partial w_k = e_k cross B_i
J_s,w = J_B (partial B / partial(s,w))
```

Here sinc is NumPy's normalized sine function and theta=|w|. Orthogonality is
preserved at each accepted step. Native prediction still evaluates all x/y/omega
values and B derivatives in one reflection pass. Python orchestrates the constant
9×6 parameter chain and six-dimensional training solve; no second reflection
pass or new native profile term is introduced. Independent Python forward
predictions and centred differences check the chained derivatives.

The existing fixed-instrument, frozen-assignment Huber/LM and half-open exposure/
panel support conventions remain unchanged. Eligibility is gated using the
original unconstrained model, then frozen. The fit requires Jacobian rank six.
Held-out positions do not influence fitting, stopping or iteration selection.
Original and constrained metrics are reported on the identical population;
failure to improve validation retains the initial overlay as default. This is
not a calibrated likelihood-ratio test or a formal symmetry determination.

## Validation and cost

Synthetic sheared P/C/I/F primitive inputs test integer setting recovery and
reciprocal invariance without any experimental target cell. Centred and primitive
scan predictions must agree physically. Finite differences use independent
forward geometry; synthetic noisy fits test exact 90° angles and independence
from held-out perturbations. No new intensity normalization or peak moment model
is involved. The pinned GSAS-II powder oracle does not cover this experiment.

For M bounded coefficient vectors the metric search uses O(M²) storage and
O(M³) worst-case work, with pairwise angular filtering before triples. The
six-parameter positional fit remains O(N I) for N tracks and bounded iterations I.
`benchmarks/spot_prediction.py` covers thousands of reflections for both fits and
records setting-search time, native prediction time and exact repeat checks.
