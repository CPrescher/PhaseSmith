# Lattice symmetry and constrained cell refinement

**Native Rust kernels:** `phasesmith_single_crystal::lattice_symmetry`, behind
the `grains` feature. No Python binding yet. This step follows
[lattice discovery](experimental-grain-search.md): the grain finder returns a
reduced primitive cell per lattice, and this module finds the conventional
cell and lattice symmetry (Bravais type) of that lattice and refines the cell
under that symmetry's constraints. The older Python
[cell comparison](experimental-cell-comparison.md) answers a related question
with fixed populations and detector-space fits; this module is the native
kernel that Dioptas uses.

Units and conventions follow the grain finder: direct bases are row matrices
in Å (one lattice vector per row), reciprocal positions `q = h B` are in
1/Å without 2π, and integer transforms act on rows
(`conventional = transform · input`, `h_conventional = h_input · transformᵀ`).

```rust
use phasesmith_single_crystal::lattice_symmetry::{
    index, refine_cell, search, RefineOptions, SearchOptions,
};

let matches = search(reduced_primitive_basis, &SearchOptions::default())?;
let best = &matches[0]; // highest symmetry that fits; aP is always last
let hkl = index(&q, best.idealized, best.bravais, 0.015)?;
let (q_used, hkl_used): (Vec<_>, Vec<_>) = q
    .iter()
    .zip(&hkl)
    .filter_map(|(q, h)| h.map(|h| (*q, h)))
    .unzip();
let fit = refine_cell(&q_used, &hkl_used, best.idealized, best.bravais, &RefineOptions::default())?;
println!("{:?} ± {:?}, V = {} ± {}", fit.cell, fit.cell_su, fit.volume, fit.volume_su);
```

## Bravais-lattice search

`search(basis, &options)` returns every one of the 14 Bravais types (aP, mP,
mC, oP, oC, oI, oF, tP, tI, hP, hR, cP, cI, cF) whose metric the lattice fits,
one standard setting per type, highest lattice symmetry first (holohedry
order c 48, hP 24, t 16, hR 12, o 8, m 4, a 2; F, I, C before P within a
system). Each `BravaisMatch` carries the type, the integer transform, the
conventional basis (exactly `transform · input`), the measured cell, the cell
with the constraints imposed, an idealized basis (the constrained cell in the
orientation closest to the measured one, a starting point for refinement) and
the deviations.

Method:

1. **Unique reduced cell.** Selling reduction, then Niggli reduction
   (Křivý & Gruber 1976, *Acta Cryst.* A32, 297), with the floating-point
   tolerances of Grosse-Kunstleve, Sauter & Adams (2004, *Acta Cryst.* A60, 1).
   The Niggli cell is unique, which makes everything after it independent of
   the basis passed in. It is the aP result.
2. **Candidate axes.** All primitive lattice vectors `n · N` with `n` in
   `[-3, 3]³` of the Niggli basis `N` (145 directions, one sign each).
   Coefficients of 3 occur, for example, for the hexagonal c axis of an acute
   rhombohedron.
3. **Axis systems and centring.** Triples in which one axis is perpendicular
   (within the angle tolerance) to the other two give monoclinic (b unique),
   orthogonal and hexagonal (c unique) systems. Their integer matrix `T` must
   have a determinant of 1 to 4. The centring comes from the lattice, not the
   metric: the rows of `T⁻¹` are the fractional coordinates of the input
   basis vectors in the conventional cell, and the group they generate
   modulo 1 must be exactly P, C (½ ½ 0), I (½ ½ ½), F or obverse R
   (⅔ ⅓ ⅓, ⅓ ⅔ ⅔). Exact integer adjugate residues are used. A supercell that
   happens to have a higher metric (for example a doubled axis) has a coset
   such as (0 0 ½) and is never reported.
4. **Metric test.** Lengths that must be equal may differ by at most the
   length tolerance, as `(max − min) / mean`. Angles that must be 90° or 120°
   may deviate by at most the angle tolerance, and so may the **obliquity**
   (Le Page 1982, *J. Appl. Cryst.* 15, 255) of each axis that must be normal
   to a lattice plane: the angle between that axis and the plane's normal
   (the reciprocal axis). The obliquity does not depend on the plane's own
   basis. Without it a triclinic lattice can show a "monoclinic" cell with
   β ≈ 165° made of long, nearly parallel vectors whose 90° angles hold by
   accident.
5. **One setting per type.** Smallest `a² + b² + c²`; symmetry-equivalent
   settings (equal sums) by smaller deviation, then cell, then transform.

| type | setting |
|---|---|
| aP | Niggli-reduced cell |
| mP, mC | unique axis b, β ≥ 90°, shortest a² + c²; mP a ≤ c; mC C-centred |
| oP, oI, oF | a ≤ b ≤ c |
| oC | C-centred, a ≤ b |
| tP, tI | unique axis c |
| hP | unique axis c, γ = 120° |
| hR | hexagonal axes, obverse, γ = 120° |
| cP, cI, cF | a ≤ b ≤ c |

Signs are chosen so that the conventional basis is right handed with β ≥ 90°
and α ≥ 90° (orthogonal and monoclinic) or γ = 120° (hexagonal axes). These
are lattice settings; a space-group setting may permute them (orthoenstatite,
*Pbca* with a = 18.23 Å, comes out as oP 5.18 × 8.82 × 18.23 Å).

### Default tolerances: 1 % and 1°

`SearchOptions::default()` uses `length_tolerance = 0.01` and
`angle_tolerance = 1.0°`. Reasons:

- Autoindexing programs use wider limits, but on *unrefined* cells and only as
  a first filter: cctbx's `iotbx.lattice_symmetry` defaults to a 3° Le Page
  obliquity and DIALS' `refine_bravais_settings` to 5°, after which each
  candidate is refined and judged by its residual. The input here is a lattice
  already refined on many positions by the grain finder, whose cell errors are
  expected to be well below 1 % and 1°.
- A tighter limit keeps fewer pseudo-symmetric metrics; 1° and 1 % still leave
  a wide margin over the scatter of a refined cell.
- PhaseSmith's Python cell comparison uses the same limits (1° and 1 % metric
  strain), so the two agree on what fits.

For a quickly indexed cell, widen both (say 3 % and 3°) and compare the
candidates with `refine_cell`.

## Constrained refinement

`refine_cell(q, hkl, start, bravais, &options)` fits the orientation (three
parameters) and only the free cell parameters of the lattice system: cubic a;
tetragonal, hexagonal and rhombohedral (hexagonal axes) a, c; orthorhombic
a, b, c; monoclinic a, b, c, β; triclinic all six. The model is
`q = R (h B₀(p))`, with `B₀` the reciprocal basis of the cell in the standard
Cartesian frame (a along x, b in the xy plane) and `R` a rotation updated by
small rotation increments. The rotation derivative is `eₖ × q`; the cell
derivative is `R (h ∂B₀/∂p)` with `∂B₀/∂p` from central differences of the
3×3 matrix (tested against full-model differences). Gauss–Newton steps are
halved until the cost does not increase; the fit stops when a cycle lowers
`Σ|Δq|²` by at most `options.tolerance` (relative) or when no step lowers it.

Uncertainties come from the covariance `s² (JᵀJ)⁻¹` with
`s² = Σ|Δq|² / (3N − P)`; all positions carry equal weight. Tied lengths share
one uncertainty, fixed angles have none, and the volume uncertainty is
propagated from the cell parameters. In tests with Gaussian position noise
the pulls `(estimate − truth) / su` have a mean near 0 and a standard
deviation between 0.8 and 1.25, for monoclinic, rhombohedral and
orthorhombic cells. A too-high symmetry shows as a larger RMS: a monoclinic
lattice with β = 91.5° fitted as orthorhombic has about twice the RMS of the
monoclinic fit at 0.002 Å⁻¹ noise.

`index(q, conventional, bravais, tolerance)` returns rounded conventional
indices `q · aᵢ` for positions within `tolerance` of a node that the
centring allows, and None otherwise (and for 000). The basis should be
close; `BravaisMatch::idealized` is a good choice.

## Limits

- **The metric cannot prove the symmetry.** A lattice can fit a
  higher-symmetry metric by accident (pseudo-symmetry, or a tolerance that is
  too wide), and a crystal can have lower symmetry than its lattice. Only the
  diffraction intensities (Laue class, equivalent reflections) confirm it.
  That is why all types that fit are reported with their deviations, and why
  the constrained fit's residual should be compared across candidates.
- The search tests lattice vectors with coefficients up to 3 in the Niggli
  basis and conventional cells of index at most 4; settings outside that bound
  are not considered.
- The refinement weights all positions equally, has no outlier rejection and
  does not refine the instrument; detector-space refinement remains in
  `spots::geometry_refinement`.
- Equal-sum ties between symmetry-equivalent settings of an exactly
  symmetric metric are broken by the integer transform, which depends on the
  input basis; the cell is then identical and the bases differ by a lattice
  symmetry operation.
