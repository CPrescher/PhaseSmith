# SHELXM/SHELXD inspection and structure-search assessment

29 October 2026. This is a static inspection and algorithm review; it makes no
new numerical implementation or measured speed/coverage claim.

## Executable and evidence limits

SHELXM is an earlier name for the SHELXD family; see the authors' discussion in
[Weeks et al. (2003)](https://atbweb.stanford.edu/atb_publications/weeks_et_al_2003.pdf).
The newly supplied file `/Users/clemens/Downloads/shelxd.exe` identifies itself as
SHELXD 2013/2 and also contains an XM 2013/2 banner. It is a 26,062,336-byte
Windows PE32+ x86-64 executable with SHA-256
`9b331fef9101abb840ed4368e67baa9ec8aff0871da9f5ab9e1e2cc164c8590d`.
This does not establish the availability of current Linux or macOS builds.

Ghidra 12.2 on macOS imported the PE file and produced approximate C-like output
for all 40 selected entry points in the initial code region. These include
compiler fragments with overlapping decompilations and several runtime helpers;
**40 does not mean 40 independent algorithms, nor a complete recovered program**.
The combined export has 56,670 routine-output lines, much of it duplicated.
The PE symbol/export/debug tables do not supply original routine names. Retained
Fortran traceback descriptors nevertheless identify MAIN__, SX1C and SX1E.
Guessed variable types, compiler outlining, unreachable-block warnings and
inferred function boundaries limit interpretation.

The external project, scripts, status records, address maps, approximate output
and hashes live in
`/Users/clemens/Codex/shelxm-shelxd-audit-2026-10-29/`.
No decompiled output is incorporated into PhaseSmith. Future numerical work must
be independently derived from publications or first principles. The original
executable is unchanged. Neither Wine nor another Windows runtime was present;
no SHELXD solve, controlled ablation or numerical comparison was performed.

## Published algorithm

The [official small-molecule manual](https://shelx.uni-goettingen.de/shelxd_sm_keywords.php)
describes FIND for initial solution and PLOP for completion. It specifies
resolution-shell E normalization, tangent expansion from a retained strong-phase
subset, bounded triplet lists, random omission, and uncertainty-weighted
correlation scoring. Patterson starts use probabilistic vector/translation
sampling. PLOP deletes peaks while correlation improves, stopping by half removal.
Cell contents and FIND peak counts are supplied estimates. These features make
SHELXD different from an element-only count-inference solver. The manual also
supports native HKLF 4 data; macromolecular anomalous differences are not required.

The historical account explains why sampling several good Patterson starts can
outperform always taking the strongest maximum, and why random omission helps
escape locked solutions. It also records cases where Patterson seeding was
counterproductive. An unseeded alternative should therefore remain in a search
portfolio. See [Sheldrick (2008), section 2.3](https://journals.iucr.org/a/issues/2008/01/00/sc5010/index.html).

In general direct-methods notation, a triplet relates phases for h, k and h-k.
A tangent update has the form

`phi_new(h) = arg sum_k w(h,k) exp(i[phi(k)+phi(h-k)])`.

Weights reflect reliability; a trusted phase subset seeds expansion. This is a
general description, not a recovered exact SHELXD weighting rule. It differs
from PhaseSmith's current hybrid update, which changes measured coefficient
magnitudes while retaining the phase from the current atomic map.

## What the binary adds

The following references are to the external address-named exports, not original
Fortran source lines. Roles inferred from code and diagnostics are distinguished
from names supplied by traceback metadata.

| Entry/export | Observation | Confidence and limit |
|---|---|---|
| `140001000.c`, MAIN__ | Allocation, setup and dispatch; parallel machinery and per-thread working arrays | Direct code/traceback evidence; not a runtime scaling measurement |
| `140004879.c`, MAIN__ outlined region, lines 1557 and 1658 | Calls SX1C and then conditionally SX1E | Distinct search/completion ordering agrees with the manual; the precise guard is not fully recovered |
| `140010f20.c`, SX1C, lines 2272–2367 | Weighted sums of observed/calculated amplitudes, means, variances and cross-products, including a second subset accumulation | Direct weighted-correlation arithmetic; exact array roles are inferred |
| `14001a390.c`, inferred SX1C numerical worker, lines 5900–5988 | Phase indices reduced modulo 360; pairs of phase indices feed two weighted lookup sums and normalization | Strong evidence for a binned tangent kernel; exact lookup layout and probability weights remain unresolved |
| `140010f20.c` and its worker | Phase-relation memory-limit diagnostics, Patterson/PSUM messages and iterative peak/cycle work | Supports cached relations and two-space work; diagnostic text alone does not specify equations |
| `1400089a0.c`, SX1E | Separate large numerical stage invoked after SX1C, consistent with completion | Traceback name is explicit; not every completion subloop is reconstructed |
| `140030690.c` / `140030944.c` | Instruction table includes FIND, PLOP, TANG, WEED, PATS, NTPR and CCWT | Direct input-parser evidence, not proof of each option's internal implementation |

The large PE image includes static Intel mathematical, FFT and OpenMP runtimes.
Packed single-precision arithmetic and phase lookup tables are visible in the
selected numerical code. They suggest where this older implementation saves
work, but provide no justification to reduce PhaseSmith's precision or import its
quantization without independent error analysis.

## Comparison with PhaseSmith 4d78439

The source review inspected `dual_space.rs`, `shared_search.rs` and
`symmetry_recycling.rs` directly.

| Mechanism | Current PhaseSmith behavior | Assessment |
|---|---|---|
| Patterson start | Ranked vector and one positive two-map minimum superposition, initially in P1 | A symmetry-aware sampled placement could add start diversity |
| Reciprocal update | Hybrid difference-Fourier coefficients, no direct-methods triplet phase refinement | Tangent expansion is a distinct possible fallback algorithm |
| Atomicity | Count-free RMS-selected positive peaks, bounded Gaussian mask, deterministic omission every third cycle | Preserve count independence; do not import FIND's supplied-N assumption |
| Completion | Density-guided orbit batches, joint labels, staged birth/delete/relabel refinement | Cheap frozen-geometry pruning could screen batches before expensive fits |
| Acceptance | Full requested-mode refinement, penalized ranking, required elements and stable batch checks | Keep these; a high normalized correlation is not a correctness certificate |

For a candidate orbit j with cached contribution F_j, inexpensive pruning can
score `|F_total-F_j|` without recomputing every surviving atom's phase sum.
This follows ordinary structure-factor addition. A prospective implementation
would remove whole symmetry orbits, preserve alternative close C/N assignments,
and retain full anisotropic/original-observation checks for final selection.
It must not use molecular connectivity rules that would reject isolated ions or
valid mineral coordination. No speedup is claimed before paired tests.

The latest frozen dual-space comparison already hands off K[BrF4] and BiSeI after
four cycles, and thiourea after sixteen. Its recycled-path screening/final-fit
counts are 27/15, 14/7 and 50/14. Thus the first speed experiment should be cheap
peak/orbit-list pruning before further refinement. Tangent expansion should be
tested as an additional strategy for difficult, especially light-atom, phasing
starts; adding triplet preparation to every easy solve could cost more than it
saves. Better shell normalization and symmetry-aware Patterson sampling belong
in separately measured ablations rather than a single opaque rewrite.

Any follow-up must keep element-only inputs, explicit missing-reflection
provenance, deterministic seeds, bounded memory and full-cell multiplicity
inference. Verification should include all three current cases, multiple starts,
new light-atom transfer cases and incomplete/noisy data. Report count/label and
coordinate recovery as well as wall time and fit quality. A count-informed
SHELXD comparison, if later run on Windows, must be labeled with its additional
priors. This review changes no numerical behavior, so no numerical test suite
was rerun.
