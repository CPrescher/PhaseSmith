# Experimental primitive and Bravais comparison

`phasesmith.experimental.cell_comparison.compare_cells(q, inferred_basis,
training=mask, ...)` follows blind lattice discovery. It accepts a **previously
inferred, training-derived** direct basis, not a supplied material cell, and
unindexed Cartesian reciprocal positions (NumPy N×3, inverse Å, no 2π). Optional
measured `(x,y,omega)` centroids and a fixed `PanelGeometry` enable detector-space
fits. Intensity, chemical identity, space group and reference HKLs are absent.
This is opt-in under `experimental-spots`; no standard release or powder API changes.

```python
from phasesmith.experimental.cell_comparison import compare_cells, CellComparisonOptions

result = compare_cells(
    q,
    inferred.direct_basis,
    training=training,
    observed=centroids,
    geometry=geometry,
    options=CellComparisonOptions(max_subcell_index=8),
)
for hypothesis in result.primitives:  # Includes rejected hypotheses and exclusions.
    print(hypothesis.index, hypothesis.cell, hypothesis.excluded_training, hypothesis.supported)
for fit in result.settings:  # Includes distinct monoclinic axes, not just one row/type.
    print(fit.setting.symbol, fit.cell, fit.training_metrics, fit.validation_metrics)
print(result.recommended.setting.symbol)
```

## Two separate comparisons

1. **Primitive hypotheses:** do smaller direct cells retain the observed reciprocal
   lattice? Enumerate all upper row-Hermite reciprocal sublattices through a
   declared index K (default 4, maximum 8). There are 56 including identity through
   4, and 390 through 8. Retain every hypothesis, including failures, and its
   excluded training/validation reflection counts and pointwise membership.
2. **Bravais settings:** which conventional metric constraints fit each supported
   hypothesis? Compare all 14 types: aP, mP, mC, oP, oC, oI, oF, tP, tI, hP, hR,
   cP, cI, cF. `searched_types` retains the complete list even when a type has no
   compatible setting inside the bounded search. Monoclinic uses unique b and C
   centring; rhombohedral uses hexagonal axes, obverse R centring. This does not
   test intensity symmetry, systematic absences beyond lattice centring, or SG.

A primitive hypothesis of reciprocal row-HNF H, det H=d>0, has initial direct
basis `A_p=H^-T A_input`, volume `V_input/d`, reciprocal rows `B_p=H B_input`, and
inferred indices `h_p=h_input H^-1`. Only integral `h_p` belong to that hypothesis.
H is an exact integer record of the rational basis change, **before fitting**.
The returned direct basis is subsequently fitted on its fixed training indices.
The helper `reduce_cell(hypothesis.direct_basis)` supplies a reduced representative.

Parent assignments use the existing native integer-rounding kernel in a
Selling-reduced basis: round `q A_reduced.T` with ties to even, then convert HKLs
back through the verified integer inverse transform. This avoids sensitivity to
large integer shears; it is not an unrestricted Euclidean closest-vector solver.
000 is excluded. Membership has strict
`||q-hB|| < q_tolerance`, default .015 Å^-1. HNF membership uses exact adjugate numerators and integer divisibility. A smaller hypothesis must retain at least 90% of
parent training assignments (inclusive boundary), at least 12 points and rank 3.
Unsupported alternatives are **not** discarded or treated as other grains. This
finite support test cannot prove primitivity when reflections are missing or
commensurate lattices overlap. It cannot establish why excluded spots exist.

## Reduced axes and conventional settings

Native Selling reduction starts from a superbase `(a,b,c,-a-b-c)`. For positive
`b_i dot b_j`, replace `b_i` by `-b_i` and add its old value to the two non-partner
vectors. The sum of four squared lengths decreases by `2 b_i dot b_j`. Repeat
the largest positive pair, stable lexicographic tie order, until all pair products
are <= `tolerance * max squared length` (default 1e-12). Choose the shortest three
vectors, making them right handed by a swap. The exact integer transform U has
absolute determinant one and `A_reduced=U A_input`. Checked coefficient updates,
a 10,000-move limit and a 1e9 coefficient budget fail explicitly. This is
Selling-Delaunay reduction, **not unique Niggli reduction**, and does not prove a
measured lattice primitive. Python's independent reference performs superbase
moves with NumPy Gram matrices. No external implementation was copied.

Search primitive integer directions in `[-L,L]^3` of the reduced basis, default
L=2, maximum 3. Enumerate conventional axis triplets of index 1, 2, 3 or 4.
Classify centring with exact adjugate residues of integer T: direct translations
are rows of `T^-1` modulo Z³. Products are reduced modulo the index first, and
integer determinants use Python integers to avoid overflow. Accept only standard
P/C/I/F/R cosets appropriate to the candidate family. `A_conv=T A_p` and
`h_conv=h_p T.T`; conventional centring preserves the lattice and does **not**
shrink its primitive volume. Predictions in conventional axes must retain this
integer-T restriction (`predict_scan(..., lattice_transform=T)`).

Constrained angles must lie within `max_angle` (default 1°); candidate Gram
projection strain must be <= `max_strain` (default 1%). Metric strain is the largest
absolute principal stretch minus one between projected and supplied direct Gram
metrics; this is **not** Le Page's obliquity. Equal-axis prefilter is a permissive
4× strain relative-length bound; the Gram projection supplies the final test.
Equivalent linear constraint subspaces in the reduced primitive Gram coordinates
are deduplicated by their orthogonal projectors (max entry difference <1e-8).
Working in reduced coordinates prevents large input shears from introducing
spurious numerical duplicates. Different monoclinic unique axes remain separate rows. Equivalent representations
prefer short conventional axes, beta>=90°, and a<=b when applicable, with a
lexicographic integer-transform tie break. Enumeration is bounded; no setting
inside the bounds does not imply a mathematical impossibility outside them.

## Constrained fits and derivatives

Write direct Gram `G=A A.T` as a linear combination of symmetric matrices D:
triclinic has six metric coordinates; unique-b monoclinic has three diagonals
and ac; orthorhombic has three diagonals; tetragonal has a=b and c; hexagonal/R
has `G11=G22=-2G12`, `G13=G23=0`, and c; cubic has one diagonal. Three Cartesian
orientation coordinates give 9/7/6/5/4 total parameters respectively.

Project G onto this linear space in the Frobenius norm, reject non-positive
metrics, and use orthogonal Procrustes for the closest orientation Q. With
`C=chol(G)`, `A=CQ`, `B=C^-T Q`, a local metric perturbation gives

`M=C^-1 dG C^-T`, `dC=C Phi(M)`, `dB=-C^-T dC.T B`,

where Phi retains the strict lower triangle and halves the diagonal. Local
coordinate directions are scaled by `||G||F / ||D_k||F`. Rotational derivatives
are `e_k cross B_rows`. Retraction uses `G+sum(s_k D_k)` and an exact Rodrigues
rotation of Q. Cholesky rejects non-positive trial metrics. The small metric chain
multiplies the existing fused **native spot values and nine analytical basis
partials**; no per-reflection Python loop is added. Reciprocal-only fits use the
vectorised linear `q_pred=hB` Jacobian. Python reference reconstruction from
linear Gram coordinates is independent of this derivative/retraction.

Fixed instrument, fixed HKLs, fixed population per primitive hypothesis. Detector
eligibility is determined once from its unconstrained training-only baseline:
valid crossing, radial xy error strictly <25 px and |omega error| strictly <2°.
All its Bravais models then fit exactly the same training points; invalid trial
crossings reject the step, and invalid final validation predictions produce
infinite residuals, never disappear. At least 12 training points are required.
Reserved data may be empty, in which case metrics report count 0 and NaN.

Damped Gauss-Newton minimises radial Huber cost, cutoff 2.5, default scales
(2 px,2 px,.3°), or (.005,.005,.005) Å^-1 in reciprocal mode. Metric chain and
native analytical derivatives are reused each iteration. Accepted history strictly
decreases; convergence, rank failure and budget termination are explicit. Trial
reciprocal bases may move at most 20% of the initial projected Frobenius norm.

## Explicit proposal ranking

The recommendation first chooses the smallest supported primitive by exact HNF
index, then retained training count and stable hypothesis ID. Within **each one
fixed population**, calculate `S=2*Huber training cost + p*log(3*N_train)` and
`delta=S-min(S_finite)`. A fit is comparable when its score is finite and
`delta <= ranking_score_window` (inclusive; default 10, configurable down to 0).
This window is an explicit heuristic in the existing score units, not a calibrated
probability or significance threshold. It does not change fitted values, support,
metric tolerances or the original training/held-out split.

Within a primitive population:

1. Comparable fits, **highest lattice symmetry first**, then lower S and stable
   fit index. Lattice holohedry orders, including inversion, are a:2, m:4, o:8,
   t:16, hR:12, hP:24, c:48. Centring adds no point symmetry; cP/cI/cF tie here.
2. Finite fits outside the window, by lower S, then higher lattice symmetry and
   stable fit index. A poorly fitting higher-symmetry model cannot outrank the
   comparable group merely because of its symmetry.
3. Invalid fits remain visible at the end of the entire proposal list.

Primitive populations retain the support/index order above; scores and score
deltas are never compared across different retained reflection populations.
Rejected primitive hypotheses stay in `primitives`, with their support exclusions;
they are not promoted into the list of fitted cell proposals.

`result.ranking` contains immutable `CellRank` records with one-based rank,
stable `setting_index`, primitive priority, lattice symmetry order, score delta,
comparable flag and reason. `result.ranked_settings` supplies fits in proposal
order, and `result.recommended` is the first one. The original `settings` pool
and its row IDs do not move, preserving HKL/array/export correspondence. CSV rows
are written in proposal order with both rank and stable row ID; JSON retains the
original pool plus the explicit ranking. The ranking is provisional, not a space
group or an assertion that a higher-symmetry physical crystal has been proven.

The score is a declared robust parameter penalty, **not a calibrated BIC,
likelihood, p-value or symmetry probability**. Held-out data enter neither fits,
primitive support nor ranking. Training-derived input bases and training
partitions are the caller's responsibility; importing an all-data-fitted basis
cannot recover independence. Nonconvergence and residuals remain separate visible
fit diagnostics, not confidence estimates.

## Bounds, validation and cost

Position/hypothesis and position/setting snapshots each have a 5-million-pair
budget; exceeding it raises, never silently truncates. Max 1 million input points,
finite coordinates <=1000 Å^-1, right-handed nonsingular input basis within the
stated magnitude/conditioning budgets. Assignment transforms and their inverses
are bounded by 1e6, as are inferred HKL coordinates. Arrays in returned records are immutable.
No new runtime dependencies: NumPy and the existing opt-in native extension only.
Gemmi is an optional **test-only black-box oracle** (tested 0.7.5), not production.

Tests cover all 14 synthetic Bravais lattices under large integer shears and
Cartesian rotations, noisy full fitting/recommendation for all 14, exact C/I/F/R
primitive recovery from conventional inputs, populated parity rejection, exact
changes of basis/HKL/q invariance, independent Selling and Gram references,
centred finite differences of every metric/orientation parameter, training-only
invariance under held-out perturbations, invalid inputs and budgets. Gemmi's public
APIs independently check Selling minimum energy and proper lattice rotation-group
orders/centring counts. GSAS-II's powder oracle does not cover this operation.
Discrete lattice moves have no smooth profile derivative; intensity normalisation,
intensity moments and powder support are not changed.

For H HNF hypotheses and N positions, membership/least squares costs O(H N).
With D short directions, axis enumeration is bounded by O(D³), per supported
hypothesis, plus small constant-size metric operations. For J surviving settings,
M fit iterations and p<=9 parameters, fitting costs O(J M (N p²+p³)). Snapshot
storage is O(N(H+J)), explicitly budgeted. Selling is a separate bounded small
basis calculation. `benchmarks/cell_comparison.py` measures full comparison on
5,399 positions and 390 hypotheses, 0.182-0.190 s on the recorded Apple Silicon
host, identical output over three repeats. This is a new stage; no prior timing
baseline exists and no speedup is claimed.

## Blind P1 diagnostic, 2026-10-25

Inputs: measured P1 tracks, the earlier training-derived inferred lattice, and
instrument geometry only. No known material/cell, saved UB, published/refined
HKLs, structure or other sample runs. Original 1052/243 split reused; this is an
exploratory continuation, **not a new independent blind benchmark**. Freeze options
before both declared runs. Primary uses weighted xy/omega transformed consistently
to q, followed by detector fits; sensitivity uses previous equal-observation q.

Both runs enumerate 390 primitive hypotheses through index 8. Only identity
retains >=90% support. Every smaller hypothesis is retained as a rejected row.
The supported lattice has five metric candidates: aP, three mP unique-axis choices,
and oP. The other eleven Bravais types have no compatible conventional metric within
the stated search bounds. The parameter penalty prefers oP in both runs.

Primary fitted oP: **4.349107 × 11.647267 × 13.413063 Å**, all angles90°,
**679.44155 Å³**. Parent memberships: 551 training/127 validation; the fixed
crossing/position gate retains 546/127 for *every* setting. Training RMS is
1.579 px and .3350°; validation RMS 2.033 px and .3578°. Scores: aP894.52,
best mP879.98, oP874.85. Held-out scaled RMS is slightly worse for oP than aP
(1.567 vs1.560), so do not claim a validation improvement or independently proven
symmetry. The result favours the simpler orthorhombic metric under a declared
training criterion, while retaining the larger cell. Smaller half-cells exclude
many observed reflections; this stage supplies no evidence to discard them.

Sensitivity q-only: oP4.349166 ×11.647181 ×13.420951 Å, V679.84522 Å³,
550/127 fixed matches. A compact checked-in record is in
`validation/results/p1-cell-comparison-20261025.json`; the reproducible analysis is
`python benchmarks/p1_cell_comparison.py` with the archived position/inferred-basis
snapshots available locally. Full local artifact, tables, transforms, masks, frozen
protocol and input hashes: `runs/p1-bravais-comparison-20261025/`. The report links
CSV/JSON exports and shows all 14 type outcomes and all rejected subcells.

## Primary sources and implementation boundary

- Patterson & Love (1957), [Remarks on the Delaunay reduction](https://doi.org/10.1107/S0365110X57000328).
- [Gemmi's reduction documentation](https://gemmi.readthedocs.io/en/latest/cell.html),
  for terminology and public black-box test APIs; no implementation copied.
- [DIALS Bravais-setting refinement](https://dials.github.io/documentation/programs/dials_refine_bravais_settings.html),
  for the distinction between metric candidates, basis transforms and positional
  residuals. PhaseSmith's bounded enumeration and penalty are its own declared
  method, not a claim of DIALS/Le Page algorithm equivalence.

All new source is first-principles implementation. No GSAS-II or other program's
implementation is copied or mechanically translated, and no oracle object enters
the runtime API. Existing orthorhombic/refinement APIs retain their behaviour.

## Ranked P1 follow-up

The explicit ranking uses the same positions, inferred basis and instrument with
no known material/cell, new cell search or refitting policy. A fresh frozen run is
written to `runs/p1-ranked-cells-20261025/` by:

```sh
python benchmarks/p1_cell_comparison.py --output runs/p1-ranked-cells-20261025
```

For the detector fit, the proposed order is oP first (S=874.854, delta=0), the
three mP settings next (delta=5.131, 7.023, 7.367), then aP (delta=19.667,
outside the default comparable window). The original fit matrices, residuals,
reflection membership and larger-cell conclusion are unchanged. This is an
explicit change to proposal ranking; it does not turn the earlier held-out
residuals into evidence for higher symmetry.

Ranking tests cover higher symmetry with a slightly worse comparable score,
rejection from the preferred group outside the inclusive window, invalid/NaN
fits, stable ties, C/I/F equal point symmetry, separate primitive populations,
held-out invariance and exact rank-to-fit-pool correspondence. Gemmi's independent
lattice rotation orders also check every ranking symmetry order. The existing
multi-reflection benchmark checks exact repeated ranks as well as fit matrices.
Ranking changes no production profile kernel, analytical derivative, intensity
normalisation or diffraction support rule; those numerical tests remain intact.

The compact frozen ranking record is
`validation/results/p1-ranked-cells-20261025.json`; the prior full numerical-fit
record is preserved unchanged. The repeated 5,399-position/390-hypothesis full
comparison plus ranking takes 0.188-0.190 s on the recorded host, with identical
ranks and fitted matrices on three runs (`experimental-cell-ranking-macos.json`).
There is no meaningful observed regression against the earlier 0.184-0.189 s
measurement; this is a small timing sample, not a speedup claim. Ranking adds
bounded population grouping and O(J log J) sorting of J fit proposals, without
another pass over the detector images or a new fit.
