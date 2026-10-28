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
two proposals for each allowed element undergo full refinement. Both full and
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
carry the initial projected common-U fit for this path, explicitly identified by
`refinement_baseline_kind="initial_projected_common_u"`. No independent P1 model
was fitted. Count-constrained runs retain schema v4 and their existing semantics.

This is a local beam search with a strong-peak initializer, not exhaustive
composition inference. Wrong strong prefixes, insufficient origin/beam coverage,
local minima, missing elements, insufficient observations, or budget limits can
prevent a correct result. Supplied exact symmetry remains a substantial prior.
No claim of SHELXT equivalence follows from a successful development case.

## Measured development results, 28 October 2026

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

Validation for this saved implementation: 1270 Python tests passed (11 skipped,
34 deselected); 388 Rust tests passed (34 ignored). Strict workspace clippy,
cargo formatting and selected Python lint checks passed. A fresh default wheel
imported successfully and excluded both experimental native solver entry points.
A separate multi-start regression verifies that the selected map and origin
belong to the winning phasing start, rather than an earlier diagnostic map.
