# Dual-space origin recovery and early structure handoffs

This follows the original count-free implementation in
[solution-count-free-dual-space.md](solution-count-free-dual-space.md).
The RMS selector, Gaussian finite support, mixed-amplitude normalization and
hybrid Fourier update are unchanged. These changes use first-principles symmetry
and orchestration; they do not implement code from an external program.

## BiSeI failure diagnosis

The saved unsuccessful BiSeI dual-space map already contains a four-peak strong
framework compatible with Pnma at the original 0.15 angstrom tolerance. The old
origin fitter tries a quarter-cell starting grid. Its local inversion-matching
iterations can settle on incompatible peak pairings in this nearly rational,
pseudosymmetric framework, rejecting all prefixes despite a valid setting.
Aligning the frozen four-peak framework through inversion-pair seeds gives a
maximum symmetry-matching error of 0.048822 angstrom. This diagnosis uses the
computed map and supplied symmetry, without reference coordinates or atom counts.

For an inversion operation `p -> -p + t`, an anchor and its same-element image
obey `-2 s = p_anchor + p_j - t (mod integers)` under origin shift s. Therefore
candidate origins are

`s = (t - p_anchor - p_j)/2 + b/2 (mod 1)`, for `b in {0,1}^3`.

If the existing grid search finds no complete symmetry match, enumerate these
eight branches for every same-element target of the strongest anchor, in stable
input/branch order and for both hands. Fit each origin with the existing local
least-squares update; require the original full-operation symmetry bijection,
tolerance and site projection. There is no hard-coded eighth-cell origin, group
name or composition. Noncentrosymmetric groups have no such fallback. Existing
successful origin searches keep their prior ordering and results. The maximum
128 input sites bounds fallback work; the shared deadline/cancellation is checked
between seeds. Synthetic Pnma orbits at translated origins, nonzero inversion
translations and the independent NumPy seed equation check this path.

## Phasing/checkpoint contract

Only the shared count-free dual-space path enables handoffs by default. Test
windows after 4,16,64,256,... completed Fourier cycles and at the requested final
ceiling. Retain the first minimum profiled mixed-amplitude SSE map in each
window, excluding omission cycles. Early windows deliberately have no 50-cycle
warm-up exclusion; an initial Patterson map can already be a good framework.
Fixed-count and unmonitored paths retain the previous warm-up convention.

The handoff runs the same symmetry closure, orbit birth/delete/relabel search,
integrated-density assignment and requested ADP refinement. A callback accepts
only when the preferred fully refined model contains every required element and
meets the original unsharpened amplitude R1 target. It finishes the current beam
round and the full refinement shortlist before
applying that criterion, preserving competing assignments for nearby elements.
Isotropic screening alone cannot accept a handoff. This stops at an adequate
candidate; it does not claim exhaustive minimum-complexity selection or prove
structure correctness.

If a handoff fails, resume the same live Fourier arrays, FFT plans, omission seed
and cycle index. Do not restart or switch phasing methods. A regression requires
bitwise-identical full histories/peak traces with declined handoffs versus an
unmonitored run. Independent NumPy complete-cycle tests also check accepted and
rejected windows and selected-map scores. Thus the resource/cycle ceiling is not
multiplied by the number of checkpoints. Cooperative wall time includes downstream
search. Difficult cases can spend more time on unsuccessful handoffs, bounded by
that same total deadline.

Each checkpoint's search retains a beam in one density frame. When a later
checkpoint fails to improve the preferred candidate, restore the prior beam and
its matching density together. Requested and actual cycles, window best cycle,
score, acceptance and adaptive peak counts are exported. The explicit
`dual_checkpoints=False` API option performs the full-budget ablation with the
origin fix. Atom bounds influence only downstream structure search and whether a
handoff is accepted; they do not change a resumed phasing trajectory.

No structure-factor/profile term or derivative has changed. Existing analytical
coordinate/ADP/ball-kernel derivatives, finite-support Gaussian normalization and
moment checks remain applicable. GSAS-II supplies no matching phasing oracle;
independent equations and downstream original-observation fits are the validation.

## Controlled measured-data result, 29 October 2026

Three serial, alternating-order pairs per case use the same pinned HKL data,
element types, cells and exact space groups as the previous comparison. Both
methods have one total phase start, an 800-cycle ceiling, seed 0, a 60-second
wall-time budget, R1 target 0.06, integrated-density guidance and staged ADP
refinement. No formula or reference coordinates enter the search. All 18 solves
and reports are saved before opening reference formulas/coordinates. The sources
are reused development cases, not an independent coverage survey.

| Case | Charge flipping median | Dual-space handoffs median | CF / dual-space time | Actual dual-space cycles |
|---|---:|---:|---:|---:|
| K[BrF4] | 1.705 s | 0.667 s | 2.56 | 4 |
| BiSeI | 0.970 s | 0.435 s | 2.23 | 4 |
| Thiourea, non-H atoms | 5.486 s | 5.761 s | 0.95 | 16 |

All three repetitions of both methods infer Br4 F16 K4, Bi4 I4 Se4 and C4 N8 S4,
respectively. Their amplitude R1 values are 0.0222664, 0.0522851 and 0.0494090;
maximum element-matched coordinate errors are 0.005909, 0.002516 and 0.010372
angstrom. Every solve passes the original paired R1/geometry gates. All 36 CIFs
reproduce their candidate R1 within 2e-10 through the general structure-factor
API, including the density-frame P1 export and supplied-symmetry export.

K[BrF4] and BiSeI are faster than charge flipping in every pair. Thiourea is
about 5% slower in every pair; shared orbit/element refinement remains its cost,
although early phasing avoids most Fourier iterations. No universal speedup is
claimed. Charge flipping remains the default. The new method's demonstrated
coverage is all three reused cases, compared with two for the original count-free
dual-space path.

The frozen primary report is
`benchmarks/structure-solution-dual-space-handoffs-macos.json`; full outputs are
under `runs/dual-space-handoffs-conservative-final/`. It includes native/source
hashes, actual window scores and cycles, all inferred counts, reference geometry
and both CIF audits.

A separate same-build ablation with `dual_checkpoints=False` also recovers all
three cases, using all 800 cycles. This isolates the origin fix from the early
handoff speed change. Dual-space times are 2.453 s, 0.942 s and 5.848 s, recorded
in `benchmarks/structure-solution-dual-space-handoffs-ablation-macos.json`.
The frozen old-map origin diagnostic is
`benchmarks/structure-solution-dual-space-origin-diagnosis.json`, reproducible with
`benchmarks/solution_dual_origin_diagnosis.py` and the saved original failed map.
The diagnostic peak coordinates are observations of that failed map, not inputs
to any reported structure solve.

An additional first-target-fit shortcut was explicitly rejected: stopping inside
the current refinement shortlist gave thiourea C2 N10 S4 at R1 0.0536176, below
the target but with incorrect composition. The count gate caught all three
repetitions. The final implementation completes the original shortlist and beam
round, recovering C4 N8 S4. The failed result and its implementation hashes remain
in `benchmarks/structure-solution-dual-space-rejected-shortcut.json`; this is why
an R1 threshold alone is insufficient and the shortcut is absent from final code.

Validation: 1,370 Python tests passed (11 external-data skips, 34 deselections),
390 Rust tests passed (34 ignored), strict workspace Clippy, Cargo formatting and
Python Ruff checks passed. Independent complete-cycle tests cover accepted and
rejected handoffs; an all-rejected run retains bitwise-identical Fourier history
and adaptive peak counts. Existing fixed-count, charge-flipping, derivative and
support tests pass unchanged. No release or push was made.

The realistic 32^3/2,108-reflection, 50-cycle kernel benchmark reports 63.07 ms
without handoffs, 63.58 ms with every handoff declined (about 0.8% monitoring
cost), and 31.55 ms for charge flipping. Recorded historical-control changes are
+5.7% for the unmonitored dual-space kernel and +11.4% for the unchanged
charge-flipping kernel; these absolute cross-run differences cannot be assigned
to checkpoint logic. The same-run monitoring comparison is the relevant overhead
check. Faster structure solves come from earlier usable maps and fewer cycles,
not a faster atomicity kernel. Full estimates and raw output are in
`benchmarks/structure-solution-dual-space-handoffs-kernel-macos.json`.

A default-feature wheel was built separately and verified to import without
SciPy and expose none of the experimental native solver symbols. The experimental
editable installation used for the benchmarks remains available. In a retained
partial result, the selected trial map cycle follows the retained candidate's
checkpoint; density and refinement transformations therefore refer to that same
map, even if a later checkpoint has a lower Fourier-only SSE.
