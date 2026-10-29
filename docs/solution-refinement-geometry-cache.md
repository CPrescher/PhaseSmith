# Shared-search refinement geometry preparation

## Diagnosis and scope

A native sampling profile of the complete thiourea element-only solve places the
main cost in `PreparedSymmetricModel::evaluate`, its trigonometric/exponential
work and refinement, rather than FFT phasing. The current dual-space case uses
156 isotropic screening fits and 34 requested-mode fits over 4,068 expanded
reflections. All candidate/label alternatives and the original complete-shortlist
acceptance policy remain necessary: an earlier first-target shortcut misassigned
C/N. This change removes repeated arithmetic within each model, not alternatives.

## Independently derived equations

For a fixed site's reference coordinate r, coordinate tangents b_j, image
operation (R,t), and reciprocal Cartesian vector q, prepare

`phi0 = 2 pi h . (R r + t)`;
`a_j = 2 pi h . (R b_j)`;
`v = Q^T q`, where Q is the Cartesian image rotation;
`w = -2 pi^2 [v_x^2,v_y^2,v_z^2,v_y v_z,v_x v_z,v_x v_y]`.

For a physical displacement tensor U define
`u6 = [U11,U22,U33,U23+U32,U13+U31,U12+U21]`.
At parameter vector p, the image contribution is

`F_image = f(s) exp(w . u6) exp(i (phi0 + sum_j a_j p_j))`.

Its coordinate derivative is `i a_j F_image`. Its log-tensor parameter derivative
is `(w . d(u6)/dp_k) F_image`, with the unchanged analytical matrix-exponential
Frechet derivative. The image value and both derivative types are accumulated in
the same reflection/image pass. Phase/image summation order stays fixed; phase
and quadratic-form arithmetic is reassociated, so round-off equivalence rather
than historical bitwise equality is required. Repeated cached runs remain
bitwise deterministic. Independent NumPy direct sums and centered differences
check values and derivatives away from topology boundaries.

The cache is reflection-major, contiguous and immutable for a prepared model.
It stores ten f64 scalars per reflection/image. A 32 MiB cap on the large reflection/image buffer falls back
to the original direct calculation; it never truncates reflections or atoms.
A small exact topology key is stored once. If public Rust geometry is changed,
evaluation uses the direct calculation instead of stale cached entries. Changing
a tensor basis also disables the isotropic shortcut unless its canonical
isotropic form still matches exactly. These mutation cases have native tests.
Parameter finiteness, physical ADP bounds, unchanged special-position multiplicity,
normalization, profiled amplitude scale and all search/termination criteria remain
unchanged. The structure-factor sum has no finite support cutoff. Existing finite
Gaussian atomicity support is unaffected. GSAS-II supplies no oracle for this
experimental search; existing independent SF/derivative references and exported
original-observation fits are the numerical checks.

An explicit `cached_refinement=False` element-search option retains the direct
kernel for paired ablations. Both kernels retain the same complete candidate
shortlist and beam ranking.

For isotropic screening, `U=u I` and `dU/dp=(du/dp) I`. Orthogonal image rotations
preserve `|q|^2`, so `D=exp(-2 pi^2 |q|^2 u)` and its log-parameter multiplier
`-2 pi^2 |q|^2 du/dp` are evaluated once per site/reflection and reused for every
image. Image values/coordinate/log-U derivatives still accumulate in their
original image order in one pass. The anisotropic path retains the full tensor
contractions above. The direct ablation retains the original rotated-vector
calculation for both modes.

## Controlled recovery and speed comparison

Three serial alternating-order pairs per case use the same dual-space phasing,
800-cycle ceiling, checkpoint schedule, seed 0, 60-second total limit, R1 target
0.06, atom/beam bounds, screening iterations and complete shortlist. The only
comparison switch is `cached_refinement`. Both modes infer counts from element
types, cell and supplied symmetry; formulas and reference coordinates are opened
only after every solve/report is saved. These are reused development datasets.

| Case | Direct median | Cached median | Speedup | Inferred full-cell counts |
|---|---:|---:|---:|---|
| K[BrF4] | 0.732 s | 0.514 s | 1.42x | Br4 F16 K4 |
| BiSeI | 0.487 s | 0.356 s | 1.37x | Bi4 I4 Se4 |
| Thiourea, non-H atoms | 6.573 s | 4.451 s | 1.48x | C4 N8 S4 |

All 18 solves match the reference counts and geometry. Final amplitude R1 values
are 0.0222664, 0.0522851 and 0.0494090, and maximum coordinate errors are 0.005909,
0.002516 and 0.010372 angstrom. All 36 exported CIFs reproduce their candidate R1
within 2e-10 through the general structure-factor calculation. The original
paired quality gates remain unchanged: R1 increase <=1e-7, maximum coordinate
error <0.02 angstrom and paired maximum-error increase <=1e-4 angstrom.

The search performs exactly the same screening/final fit counts in each paired
mode: 42/18, 23/10 and 156/34. Thus the thiourea speedup is not obtained by dropping
C/N alternatives, stopping inside a shortlist or using known composition. Wall
time falls by about 32%. A single additional charge-flipping pair per case also
passes count, geometry and both-CIF gates, with direct/cached times 1.718/1.501 s,
1.058/0.530 s and 5.969/5.240 s. It is a transfer check, not a repeated timing
estimate. Absolute timings should not be compared across separate campaigns;
these speedups use same-build, paired controls.

Frozen reports are `benchmarks/structure-solution-refinement-cache-macos.json`
and `benchmarks/structure-solution-refinement-cache-charge-flipping-macos.json`.
The controller is `benchmarks/solution_refinement_cache_comparison.py`; outputs
are under `runs/refinement-cache-guard-final/` and
`runs/refinement-cache-guard-cf-final/`. Reports include source/native and
input checksums, actual effort, post-solve reference audits and both CIF frames.
Native profile hotspot counts are preserved in
`benchmarks/structure-solution-thiourea-profile.json`; the raw sampling report
remains in `runs/thiourea-search-investigation/native-sample.txt`.

Independent numerical tests compare direct and cached kernels with NumPy sums
on randomized deterministic isotropic/anisotropic models spanning triclinic,
monoclinic, trigonal and cubic symmetries, including special positions and
translated origins. Local tolerances are 4e-10 relative / 2e-12 absolute for
values/Jacobians and 5e-6 / 5e-9 for centered derivative checks. Refined objectives
and R1 must agree within 2e-11 and 2e-10, respectively. A 60,000-reflection,
eight-image example forces the memory-cap fallback and requires bitwise-equal
residuals/Jacobians, without observation truncation. Repeated cached evaluations
are bitwise deterministic. Existing tests and their tolerances are unchanged.

Validation: 1,386 Python tests passed (11 external-data skips, 34 deselections),
392 Rust tests passed (34 ignored), strict workspace Clippy, Cargo formatting and
Python Ruff checks passed. No release or push was made.

The realistic kernel benchmark uses four asymmetric sites and their eight mirror
images, over 2,108 reflections. Direct/cached fused evaluation takes 0.469/0.307 ms
for isotropic models (1.53x) and 0.713/0.514 ms for anisotropic models (1.39x).
Including model preparation and eight refinement iterations gives 5.236/3.904 ms
(1.34x) and 11.565/9.996 ms (1.16x). Cache construction and identity guards are
included; no scalar-only call is used as the performance claim. Full estimates
and raw output are saved in
`benchmarks/structure-solution-refinement-cache-kernel-macos.json`.

A separate default-feature wheel imports without SciPy and exposes no
experimental solver symbols. Stable/default powder runtime dependencies are
unchanged. The experimental editable installation remains available for the
recorded comparisons.
