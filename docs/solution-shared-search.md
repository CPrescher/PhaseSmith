# Experimental shared element and orbit search

This path accepts corrected intensities, a cell, exact symmetry and required
neutral non-H element types. Counts and atomic coordinates are not supplied.
It is independent work from the structure-factor equations below and the existing
published charge-flipping/difference-Fourier components.

A single count-independent charge-flipping map per start uses
`E_h = Fo_h / sqrt(sum_e f_e(s_h)^2)`, with one equal contribution per allowed
element. This is a sharpening heuristic, not a composition estimate. A 75% strongest-peak height threshold proposes the initial prefix. Nearby prefix
lengths are tried for symmetry closure, bounded below by the shortest sampled
orbit multiplicity. This is an initialization heuristic, not a composition fit.
That prefix starts as the heaviest allowed element; local relabel moves can change
it. Several compatible origins are retained. No full-composition combination is
enumerated and no solver restart is made for an edited model.

Each beam state shares a partial structure, its setting and cached complex
structure factors. Signed difference-map maxima generate possible new orbits.
Up to three simultaneous affine fixed-set constraints propose nearby special
positions. Orbit multiplicity follows from exact symmetry expansion, without a
requested count. Bounds on sites, peaks, proposals, beam width, rounds and elapsed
time are explicit. Whole-orbit birth, deletion and element relabel moves compete
with retained parents. Site coordinates and positive ADPs use the existing
analytical constrained refiner. The representative phase sums and their three
coordinate derivatives are calculated once per reflection/orbit and shared
across all proposed element labels:

`Q_h(x) = sum_g exp(2 pi i h . (R_g x + t_g))`

`dQ_h/dx_j = 2 pi i sum_g (R_g^T h)_j exp(2 pi i h . (R_g x + t_g))`.

For element e, `F_h = Fparent_h + f_e(s_h) exp(-8 pi^2 U s_h^2) Q_h`.
Only unique physical images contribute. The initial proposal score uses a common
isotropic U; final selection uses the full refined site tensors. Fixed topology
and stabilizer-compatible directions are required when differentiating a special
position. There is no spatial truncation of a structure-factor contribution.
The signed difference map has the same measured support, undefined-phase zeros
and Nyquist/Friedel conventions as the existing completion implementation.

With `A_h=|F_h|`, `k=(A.Fo)/(A.A)`, `J=sum(k A-Fo)^2/sum(Fo^2)`.
Amplitude derivatives are `Re(conj(F) dF)/|F|` (zero at the declared numerical
zero threshold). The derivative of profiled k is included in the normalized
residual Jacobian. The descriptive rank is `N log(max(J,1e-300))+p log(N)`, with
independent family count N and fitted coordinate/ADP parameters plus scale p.
It is not a calibrated likelihood or a composition probability. Non-overlap is a
strict 0.5 angstrom periodic pair cutoff. Both model edits and subsequent refined
coordinates must pass it. Required types must all be present for a target hit.
Beam retention keeps origin lineages represented, rather than prematurely losing
a setting that only weak atoms distinguish. Partial results and actual effort
must survive cancellation or a time limit.

Independent NumPy equations, deterministic synthetic data, centered finite
differences and measured development-data comparisons validate this path.
GSAS-II does not expose this count/element search and is not an oracle for it.
The existing signed-observation provenance and external SHELXT comparison remain
separate. This does not infer the space group, hydrogens, partial occupancies,
disorder, twinning, or anomalous scattering.

## API, controls and provenance

```python
from phasesmith.experimental.solution import solve_structure_elements, write_solution

result = solve_structure_elements(
    observations,
    cell=cell,
    space_group=group,
    elements=["Br", "F", "K"],
)
print(dict(result.composition))  # inferred full-cell counts
print(result.shared_search.accepted_moves)
write_solution(result, "new-output-folder")
```

The default maximum count is the upward-rounded `cell_volume/8` (at most 128),
a conservative search cap rather than a density measurement. Defaults are one
800-cycle charge-flipping start, four live states, eight edit rounds and a
30-second total budget. Up to 24 positive signed difference-map peaks and eight
transformed original-map peaks seed each state's pool. Up to 32 projection states
per seed, three constraints and 64 distinct valid new orbits are allowed. The best
two proposals for each allowed element enter the refinement schedule. With
staging enabled, a conservative subset receives full anisotropic refinement. Both full and
partial states can survive; all required elements and the R1 target must hold in
the preferred state before success is reported. The source observations remain
unchanged, including negative intensities.

`SolutionOptions` supplies the common phase/refinement budgets and target. This
path requires `phasing="charge_flip"` and constrained ADPs. Legacy count-based
completion, P1 polish and fixed-composition assignment controls do not apply.
`maximum_atoms`, `beam_width` and `search_rounds` are explicit API arguments.
Cancellation and time limits retain completed initial/edited candidates. The
stored map corresponds to the preferred candidate's actual phasing start, and
its exported origin/hand transformation is composed across every model edit.

Schema v5 records `composition_source="inferred_shared_orbit_search"`, the input
element list and actual phase, proposal, label, refinement and edit effort. Only
the selected accepted edit path is called accepted; unsuccessful trials still
contribute to effort counters. The legacy `p1_r1/p1_objective` refinement fields
carry the initial projected seed fit for this path: `initial_projected_common_u`
for the original full-fit schedule and `initial_projected_isotropic_sites` for
the staged schedule. This is explicit in `refinement_baseline_kind`. No independent P1 model
was fitted. Count-constrained runs retain schema v4 and their existing semantics.

This is a local beam search with a strong-peak initializer, not exhaustive
composition inference. Wrong strong prefixes, insufficient origin/beam coverage,
local minima, missing elements, insufficient observations, or budget limits can
prevent a correct result. Supplied exact symmetry remains a substantial prior.
No claim of SHELXT equivalence follows from a successful development case.

## Full-refinement baseline results, 28 October 2026

All three measured cases used only the listed element types, the externally
measured cell and exact supplied symmetry. The same default controls applied.
Reference coordinates and formula counts were read only after all runs for the
case had been solved and saved.

| Case | Inferred full-cell counts | R1 | Time | Maximum coordinate error |
| --- | --- | --- | --- | --- |
| K[BrF4] | Br4 K4 F16 | 0.0222664 | median 3.167 s, three runs | 0.005909 A |
| BiSeI | Bi4 I4 Se4 | 0.0522851 | 2.298 s, one run | 0.002517 A |
| Thiourea, non-H | S4 N8 C4 | 0.0494090 | 27.981 s, one run | 0.010372 A |

The K[BrF4] path began with four strong Br sites, added a four-image K orbit and
then a sixteen-image F orbit. It used one phasing start, four initial settings,
two edit rounds, 16 distinct orbit proposals, 48 shared alternative-element scores
and 78 constrained refinements. No full-composition combination was supplied or
enumerated. The three repeats recovered identical counts and coordinates.
Thiourea needed four edit rounds and two distinct two-image carbon orbits; its
longer time illustrates that coverage and speed remain structure dependent.

The old 45-composition screen took 116.205 seconds. The shared K[BrF4] median is
36.7 times faster in these recorded runs. The old screen used four workers; the
new search uses one native call. This is not a thread-matched speed comparison
with SHELXT. The old count-sweep report remains unchanged. These are reused
small development cases, not a general success-rate benchmark.

The multi-reflection Criterion test uses 16 fixed atoms, a proposed sixteen-image
orbit, 1060 reflections and all three coordinate derivatives. Shared Si/O/Fe
label evaluation measured 159.41–159.56 microseconds against
409.61–414.54 microseconds for three independently prepared label evaluators
(about 2.58 times faster at the central estimates). Preparation is excluded from
both loops. The unchanged single-label value loop measured 136.13–136.32
microseconds, a 1.93% regression relative to its earlier saved Criterion baseline;
that measurement is retained rather than changing a tolerance or discarding it.

Both exported CIF frames for all five measured runs reproduce their native R1
through the general structure-factor implementation within 2e-10. The frozen
report, input/code/native hashes, timing intervals and post-solve geometry audits
are in `benchmarks/structure-solution-shared-search-macos.json`. Full diagnostic
maps and CIFs remain in `runs/shared-search-final/`.

Validation for the saved baseline (`8d1b9e1`): 1270 Python tests passed (11 skipped,
34 deselected); 388 Rust tests passed (34 ignored). Strict workspace clippy,
cargo formatting and selected Python lint checks passed. A fresh default wheel
imported successfully and excluded both experimental native solver entry points.
A separate multi-start regression verifies that the selected map and origin
belong to the winning phasing start, rather than an earlier diagnostic map.

## Staged refinement design

The staged schedule screens each proposed edit with a bounded isotropic fit.
Only screened candidates compete at this stage; an isotropic score is never
compared directly with an anisotropically refined parent's score. Select the
best `refinement_shortlist` scores, include all scores within 2 of that cutoff,
and rescue the best birth introducing each still-missing required element.
The 2-unit window is a conservative heuristic, not a probability interval.
Promote these candidates to the requested final ADP model, then use the existing
final-model score and improvement test. Initial frameworks receive full fits.
Screened states cannot enter the retained beam or satisfy the target.

Promotion preserves screened coordinates, source-atom ordering and composed
origin/hand. For each site, initialize the final invariant log-tensor parameters
from `H_i = log(Uiso_i) I`, with coefficient `trace(H_i B_ij)` in the existing
orthonormal invariant basis. This embeds the isotropic model exactly in the
anisotropic model, apart from clamping roundoff to the existing [1e-5,0.2] U
bounds. All structure-factor values, derivatives, spatial support and final
objectives use the existing kernels; there is no new scattering/profile term.
An independent NumPy selector and tensor-seed reference check the orchestration.
GSAS-II does not expose this search schedule, so it supplies no oracle fixture.

Defaults for anisotropic shared search are `staged_refinement=True`,
`screening_iterations=8` and `refinement_shortlist=3`. The shortlist is per parent,
so different origin lineages are screened independently. `staged_refinement=False`
reproduces the full-fit schedule; requested isotropic refinement bypasses staging.
Only requested-mode fits enter the beam or satisfy the target. During a stop,
unpromoted screening fits are discarded and completed full-mode states survive.
The effort report separates total, screening and final refinements, attempted
promotions and valid screened-out edits. These additive v5 fields do not change
count-constrained schema v4. The shared-search CLI benchmark accepts
`--no-staged-refinement` to reproduce its historical schedule.

The paired benchmark is `benchmarks/solution_staged_comparison.py --output ...`.
It pins the same three reflection files, alternates method order across three
serial repeats, and uses identical inputs and 60-second total solver budgets.
Reference counts and coordinates are opened only after every solve has been
saved. Gates require equal inferred/reference counts, maximum coordinate error
below 0.02 A, paired error increase at most 0.0001 A and R1 increase at most 1e-7.
Both exported CIF frames must reproduce native R1 within 2e-10. These checks
assess preserved structure recovery on reused development data; they cannot
establish a general guarantee that isotropic screening never discards a useful
anisotropic model.

### Paired staged-refinement results, 28 October 2026

The serial alternating-order comparison completed three pairs per dataset, 18
solves in total. All runs recovered the correct full-cell non-H counts, retained
the reference geometry and passed the native/exported-CIF checks.

| Case | Full-fit median | Staged median | Speedup | Full requested-mode fits |
| --- | --- | --- | --- | --- |
| K[BrF4] | 3.250 s | 2.260 s | 1.44x | 78 → 30 |
| BiSeI | 2.314 s | 1.123 s | 2.06x | 57 → 19 |
| Thiourea | 28.288 s | 7.572 s | 3.74x | 242 → 46 |

All paired speedups were positive. Inferred counts and accepted edit kinds were
preserved. R1 differences were below 2.1e-10; maximum reference-coordinate errors
across both schedules were 0.005909 A, 0.002517 A and 0.010372 A respectively.
The number of total fits increases because many are now short isotropic screens;
requested-mode fit counts and elapsed times, rather than total fit count alone,
measure the saving. Geometry and both CIF-frame checks were performed after all
18 models had been saved, using no reference counts or coordinates in search.

The frozen result is `benchmarks/structure-solution-staged-refinement-macos.json`;
full outputs and the before-reference report remain in
`runs/staged-comparison-final/`. Historical shared-search reports remain unchanged.
Validation: 1293 Python tests passed (11 skipped, 34 deselected); 388 Rust tests
passed (34 ignored). Strict workspace clippy, cargo formatting and selected ruff
checks passed. A freshly built default wheel imported without experimental solver
entry points. No structure-factor value/Jacobian loop was changed; the paired
multi-atom, multi-reflection benchmark measures the new scheduling behavior.
