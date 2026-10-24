# Experimental geometry correction and blind cell proposals

Only `codex/experimental-spots`, with `--features experimental-spots`. No default
API, runtime dependency, release, GUI or GSAS-II coupling is added. This slice
uses first-principles vector geometry and the published Fourier indexing idea;
it does not reuse indexing implementation code from another package.

## Geometry and units

`PanelGeometry` contains the laboratory location of pixel-centre (0,0), fast
and slow pixel displacements, incident beam and sample rotation axis. All five
vectors use the **same right-handed** laboratory frame. Positions/displacements
are metres; wavelength is **angstroms**. Pixel indices are zero-based centres,
so the PONI constructor includes the half-pixel offset. `x` is column, `y` row.
No image inversion is implicit. For sample angle omega in degrees:

```
p = origin + x fast + y slow
q_lab = (p/|p| - beam/|beam|) / wavelength
q_sample = Rodrigues(axis/|axis|, -omega) q_lab
```

Thus `|q| = 1/d` in inverse angstroms, **without 2*pi**. Omega increases by a
right-handed rotation about the supplied axis. A constant omega-zero offset
only changes overall orientation for this single-axis model, not the cell
metric. The rotation **sign** and axis direction do affect lattice consistency.

The undistorted PONI constructor uses pyFAI's (1,2,3) right-handed basis:
`R=R3(rot3) R2(-rot2) R1(-rot1)`. This convention and pixel geometry are checked
against pyFAI by the external Dioptas adapter tests. That adapter samples its
actual calibration, explicitly maps image flips/quarter turns, and rejects
non-affine detector geometry. No parallax/spatial distortion model or joint
geometry refinement is introduced here.

Each retained per-frame observation is converted **before** combining a linked
track into one equal-observation mean q. Observations are not independent
reflections and never enter the search as separate spots. Per-observation
weights were not retained by the earlier position format, so no intensity
weights are invented. RMS scatter is a diagnostic, not an uncertainty. Finite
exposure midpoints, threshold selection and incorrect links limit accuracy.
Distinct tracks are not merged merely because they are close in q.

## Blind search and boundaries

`find_lattices` accepts measured reciprocal vectors, a training mask and generic
search bounds. There is no material, structure, cell, UB or crystal-system input.
Default bounds are 1.5–30 Å, 256³ grid, 160 Fourier maxima and 28 retained
translation vectors. Length boundaries are inclusive. Index acceptance requires
**strictly** less than 0.15 in every fractional coordinate and 0.015 Å^-1 in
Euclidean reciprocal distance; 000 is excluded.

Training q is deposited by trilinear interpolation on a periodic reciprocal
grid with spacing `1/(2 max_length)`. Points outside its guarded extent are
omitted **only from the FFT proposal**, not continuous refinement or scoring.
The real Fourier transform proposes cosine maxima; all 26 periodic neighbours
are checked, ties excluded. One of each ± pair is retained. Continuous
maximization uses the actual unbinned observations:

```
F(r) = mean cos(2*pi q_i.r)
grad F = -mean 2*pi q_i sin(2*pi q_i.r)
H F = -mean (2*pi)^2 q_i q_i^T cos(2*pi q_i.r)
```

Rust evaluates value, gradient and Hessian together in one ordered pass. Bounded
damped Newton steps require nondecreasing score. Triplets must be independent
and well separated in volume/length product. Up to 80 preliminary bases are
refined by alternating nearest integer assignment and least-squares reciprocal
basis fits. Selection is training-only. Extra supercell volume can capture a
few random spots, so the smallest volume within a declared 2%-of-training-size
consensus slack is preferred. Small-denominator (2,3) subdivisions are tested
against observed support, with at most four rounds. Integer size reduction is
not Niggli reduction; subdivisions do **not prove primitivity** or remove every
possible supercell. Missing classes, twinning and multiple lattices remain hard.

The returned basis has direct vectors as **rows**, in Å; `h ≈ q @ A.T`, and
`q_predicted = h @ inverse(A).T`. Six cell parameters come from its unconstrained
triclinic metric. Equivalent cells can use different axes/signs. No conventional
cell, Bravais class or space group is assigned, and no integrated I(hkl) exists.

Validation positions contribute to neither FFT, translation refinement, basis
selection nor fitting. A candidate passes provisional checks with ≥20 distinct
training HKLs of rank three, ≥20% training/validation coverage, ≥5 validation
matches, and validation fraction ≥ three times the largest shifted control
fraction plus 5 percentage points. Eight deterministic off-origin shifts give
descriptive controls, **not calibrated significance levels**. These checks do
not prove the geometry, chemical identity, uniqueness or completeness. Always
retain failures and competing proposals. Do not tune thresholds to a known cell.

The Dioptas adapter reserves every fifth five-frame block and excludes tracks
crossing its partition boundary. It selects unflagged tracks with ≤8 detected
observations and RMS q spread ≤0.03 Å^-1. When comparing rotation signs, the
selection is identical and passes the spread bound under both signs. These are
explicit initial heuristics, not universal spot-quality criteria.

## Cost, validation and limitations

For N observations, M grid cells and K translation candidates: geometry is
O(N), gridding O(N), FFT O(M log M), neighbourhood scans O(M), continuous
Fourier work O(N K iterations), and bounded triplets O(N K³). Fourier work is
capped at 200 million pairs per call; input is capped at 100,000 positions,
grid at 256³, retained translations at 40. FFT memory is O(M), several hundred
MB at 256³; it is not a full image-stack allocation. The current implementation
uses NumPy FFT/linear algebra orchestration and one deterministic Rust worker;
external BLAS configuration can still affect NumPy's own thread count.

Independent readable NumPy reference equations validate geometry and Fourier
moments, including randomized rotations and centred finite differences. Blind
synthetic orthogonal, skew and hexagonal metrics include noise/outliers;
unstructured clouds must fail. Benchmarks use 3,000 reciprocal positions,
10,000 detector observations and repeated full 256³ searches. GSAS-II's powder
oracle does not cover detector reciprocal geometry or unknown-cell indexing.

Sources: [pyFAI geometry conventions](https://github.com/silx-kit/pyFAI/blob/main/doc/source/geometry_conversion.rst),
[Powell et al., autoindexing (2013)](https://journals.iucr.org/d/issues/2013/07/00/ba5191/),
[DIALS Fourier-method overview](https://dials.diamond.ac.uk/documentation/library_reference/algorithms/dials.algorithms.indexing.html).
Sources inform the equations/algorithm family, not copied program logic.
