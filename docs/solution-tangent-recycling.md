# Tangent refinement within supplied-symmetry recycling

This follow-up changes the placement of the optional tangent/annealing/quartet
step when symmetry recycling is also enabled. It uses the existing independently
implemented phase-invariant equations and symmetry projection, documented in
`solution-tangent-refinement.md` and `solution-symmetry-recycling.md`. It introduces
no new observations, composition inputs, external solver or copied implementation.

## Diagnosed regression

The previous combination perturbed independent Friedel phases during P1
Patterson/dual-space iteration, before the strong framework established an origin
for the supplied space group. The quartet guard controlled one phase-invariant
score but did not enforce agreement between symmetry-equivalent reflections.
The later recycling projection averaged their unit phasors. Disagreement could
therefore cancel useful phase information even after the framework itself fit well.

On the saved thiourea seed-0 comparison, the strongest-framework R1 is effectively
identical (0.2668332457). The mean symmetry/Friedel-aligned unit-phasor length drops
from 0.84249 without tangent updates to 0.75795 with the original combination;
the fraction below 0.5 rises from 9.44% to 19.62%. The recycled map has 20 positive
maxima instead of 16, yet its joint proposal admits only one additional orbit
instead of four. The eventual correct model then requires five serial edit
rounds instead of one, 258/46 screening/final fits instead of 50/14, and about
6 seconds instead of 1.54 seconds. This localizes the cost to downstream model
construction, not the roughly microsecond-scale tangent kernel.

For reflection h, the reported coherence is
`C_h=|mean_g [exp(2 pi i h.t_g) (U(hR_g)+conj(U(-hR_g)))/2]|`,
where `U` is the unit phase after applying the candidate's source-to-symmetry
origin/hand transform. It is a cancellation diagnostic, not a correctness score.
`benchmarks/solution_tangent_recycling_diagnosis.py` computes this from saved maps,
observations and supplied symmetry only. It reads no reference coordinates or
formula. File hashes, settings, phase coherence and search effort are frozen in
`benchmarks/structure-solution-tangent-recycling-diagnosis.json`.

## Revised order and guard

With both `tangent_refinement=True` and `symmetry_recycling=True`, the default
`tangent_in_recycling=True` defers tangent updates until each strong-framework
origin has been recovered. The initial P1 dual-space trajectory is unchanged.
Each retained origin still receives its own map and bounded hypothesis search.
For every existing recycling cycle:

1. Project coefficients onto the supplied symmetry/Friedel relations and restore
   measured Fo amplitudes, giving `F0=P_sym(F)`.
2. Propose the existing annealed tangent correction on the independent measured
   phases of F0. Keep the original triplet/quartet graph, temperature schedule,
   quarter blend and bounded free-phase backtracking.
3. Restore symmetry again: `F1=P_sym(T(F0))`. Accept this combined proposal only
   if `Q(arg F1) <= Q(arg F0)+1e-12`; otherwise retain F0 exactly.
4. Synthesize the map, clip negative density and transform back as before.

The final symmetry/amplitude projection is unchanged. The second quartet check
matters: symmetry averaging can undo an improvement seen in unrestricted phase
space. A synthetic 1,346-reflection screw-symmetry example exercises actual
rejections at this second guard and matches the independent NumPy complete cycles.
This is a discrete projection, with no claimed Jacobian through clipping or phase
branch selection. Existing invariant values and analytical phase gradients remain
fused; refinement derivatives and candidate acceptance gates are unchanged.

Because the first projection restores Fo exactly, the frozen 40% selection uses
the **stored measured Fo**, rather than recomputing complex norms. This prevents
roundoff from changing which of several equal symmetry-related amplitudes is
frozen. On symmetry phase lines, quartet costs for the plus/minus perturbations
can tie exactly. In this coupled step, costs within
`1e-12*(1+abs(cost_plus)+abs(cost_minus))` choose the plus proposal deterministically.
The unconstrained legacy step retains its original strict comparison. These
conventions make native and NumPy complete steps agree even in centric cases;
no global comparison tolerance was relaxed.

The cooling counter starts at zero in each recycling attempt and uses the same
trial seed. The existing recycling-cycle ceiling, atom cap, integration radius,
whole-orbit proposals, label sweeps, shortlists, full anisotropic fits and complete
stability round remain in force. There is no early acceptance of a low-R1 batch,
case-specific dispatch, reference-informed count or pruned competitor beam.

One extra coefficient buffer is bounded by the existing 2,097,152-node FFT cap
(32 MiB of complex values). Relation and symmetry-plan limits remain unchanged.
Cancellation/deadline checks apply between bounded cycles. With tangent updates
disabled, the old projection arithmetic remains available unchanged.

## API, audit and independent checks

```python
result = solve_structure_elements(
    observations, cell=cell, space_group=space_group, elements=element_types,
    options=SolutionOptions(phasing="dual_space", trials=1, cycles=800),
    symmetry_recycling=True, tangent_refinement=True,
)
```

`tangent_in_recycling=False` explicitly retains the earlier P1 tangent placement
for ablation. When recycling is disabled, tangent refinement continues to use
that P1 placement regardless of this preference. Tangent refinement and symmetry
recycling remain individually opt-in. Deferring updates can change which
frameworks are found on other datasets; the legacy placement remains available.

`SharedSearchAudit.tangent_in_recycling` records the requested placement policy.
Actual coupled work is attached to each `RecyclingAudit.tangent`, including trial,
lineage via the enclosing record, relation counts, cycles, proposals, phase
changes, symmetry restorations, second-guard rejections and Q before/after each
combined step. `tangent_trials` continues to describe P1 tangent work and is empty
when all updates were deferred. Phase changes are counted after symmetry
restoration, using a wrapped 1e-12-radian audit threshold and excluding amplitudes
at or below the existing 1e-100 unit-phase threshold. A signed complex zero has
no physical phase and cannot count as a changed coefficient. These thresholds
affect only the counter. No quartet evidence is claimed when its count is zero.

Independent dense NumPy cycles cover P1, inversion, mirror, screw and centered
tetragonal groups, translated/inverted origins, measured-amplitude restoration,
missing-coefficient provenance, zero cycles, stopping and matched-plan validation.
They check actual work counters and bitwise native repeatability as well as maps
and quartet scores. Synthetic shared solves exercise both placements and inferred
counts. Existing randomized invariant/gradient and finite-difference checks still
apply. GSAS-II has no corresponding phase-recycling oracle.

## Comparison protocol

The paired controller records `tangent_in_recycling` for every mode, the unchanged
search budgets, source/native hashes and seed for every solve. A legacy-placement
comparison and a recycling-only comparison each use three serial alternating
pairs per case with seed 0. All solve outputs are saved before reference formulas
and coordinates are opened. Quality gates require exact non-H counts, coordinate
errors below 0.02 angstrom, paired R1 increase at most 1e-7, paired maximum error
increase at most 1e-4 angstrom, and both CIF frames reproducing R1 within 2e-10.

The requested seed campaign is fixed in advance to
`0, 1, 2, 3, 4, 17, 47, 2026`, one start and an 800-cycle/60-second ceiling per
solve. It compares the improved combination with recycling alone on all three
structures, saves every failure and applies the same post-solve gates after all
48 solves. These reused datasets provide a seed robustness check, not a blind
coverage estimate.

```sh
.venv/bin/python benchmarks/solution_tangent_comparison.py --output runs/coupled-legacy --repeats 3 --symmetry-recycling --legacy-tangent-baseline
.venv/bin/python benchmarks/solution_tangent_comparison.py --output runs/coupled-recycling --repeats 3 --symmetry-recycling
.venv/bin/python benchmarks/solution_tangent_comparison.py --output runs/coupled-seeds --repeats 1 --symmetry-recycling --seeds 0 1 2 3 4 17 47 2026
cargo bench -p phasesmith-solution --bench solution -- recycling_10_cycles
```

## Frozen results (macOS arm64, 29 October 2026)

All 84 final saved solves pass the count, geometry, original-observation R1 and
both-CIF quality gates. This includes 24 improved-combination solves over the
eight declared seeds and all repeated timing runs. Correctness does not rely on
reaching the R1 target alone.

The three-pair, seed-0 medians compare the previous unconstrained combination with
the improved combination in the same native build:

| Structure | Previous combination (s) | Improved combination (s) | Previous/improved | Inferred non-H full-cell counts |
| --- | ---: | ---: | ---: | --- |
| K[BrF4] | 0.7817 | 0.7896 | 0.990 | Br4 F16 K4 |
| BiSeI | 0.3083 | 0.2887 | 1.068 | Bi4 I4 Se4 |
| Thiourea | 5.7752 | 1.3347 | 4.327 | C4 N8 S4 |

Thiourea's joint proposals again add four weak orbits per origin. The search
requires one ordinary stability round, 50 screens and 14 full fits, replacing
five rounds, 258 screens and 46 full fits. The initial strong-framework fits,
atom bounds, label search, full shortlist and acceptance checks were retained.
BiSeI again accepts the cycle-4 handoff rather than cycle 16. K[BrF4] keeps the
same candidate and 27/15 fits; its small timing difference is overhead.

A separate three-pair comparison with **recycling alone** gives baseline/improved
medians of 0.7680/0.7856 s, 0.2493/0.2906 s and 1.4836/1.3685 s respectively.
Thus the combined option still adds overhead for the two easy cases; it does
not justify turning either experimental option on by default. The prepared-plan
kernel benchmark likewise costs more per map: 6.425 ms for ten recycling cycles
versus 8.122 ms for ten coupled cycles over eight atoms and 2,108 reflections.
The large end-to-end thiourea improvement comes from avoiding model-search work.

### Eight-seed recovery and remaining timing variation

Every seed `0,1,2,3,4,17,47,2026` recovers the correct structure and counts:

| Structure | Correct improved solves | Improved time range (s) | Maximum reference error over seeds (angstrom) | Amplitude R1 range |
| --- | ---: | ---: | ---: | --- |
| K[BrF4] | 8/8 | 0.778–0.788 | 0.005909 | 0.0222664004093 |
| BiSeI | 8/8 | 0.288–0.913 | 0.002516 | 0.0522850546588 |
| Thiourea | 8/8 | 0.654–7.302 | 0.011363 | 0.0494090499679–0.0494090499684 |

Across these eight seeds, recycling-alone/improved median times are
0.7641/0.7815 s, 0.3984/0.4388 s and 1.7994/1.3025 s: thiourea improves 1.38x in
this seed distribution. Seed 2026 falls from three edit rounds and 117/27 fits to
one round and 23/9 fits. Seeds 3 and 4 retain the long search tail:

| Thiourea seed | Recycling alone (s) | Improved combination (s) | Screens/full fits, both modes |
| --- | ---: | ---: | --- |
| 3 | 6.893 | 7.302 | 232/44 |
| 4 | 6.801 | 6.430 | 246/46 |

These long runs also occur without tangent refinement and are not the earlier
seed-0 interaction regression. No speed improvement for every seed is claimed.
For each of the eight thiourea solves, 1,296–2,880 nonzero-amplitude independent
phases actually change across the retained recycling attempts. The improvement
does not merely switch tangent refinement off. Both centrosymmetric cases have
zero accepted physical phase changes after the symmetry/quartet guards; their
remaining cost is preparation and guarded attempts. K[BrF4] has no eligible
negative quartets, as before. A zero audit score there is unavailable evidence.

The full source/native hashes, per-seed work, initializations, failed handoffs,
map frames, post-solve counts/geometry and CIF checks are retained in:

- `benchmarks/structure-solution-coupled-vs-legacy-tangent-macos.json`
- `benchmarks/structure-solution-coupled-vs-recycling-macos.json`
- `benchmarks/structure-solution-coupled-seeds-macos.json`
- `benchmarks/structure-solution-coupled-kernel-macos.json`

Validation: 1,464 Python tests passed (11 external-data skips, 34 deselections),
394 Rust tests passed (34 ignored), strict workspace Clippy, Cargo formatting,
Ruff and whitespace checks passed. Native/NumPy complete-cycle comparisons use
5e-9 relative and 2e-11 absolute map tolerances; amplitude/symmetry restoration
uses separate 1e-11 relative and 2e-10 absolute assertions. Rejection counters and
native replay are checked exactly. A separate default-feature wheel passes native
geometry and symmetry checks with SciPy blocked and exposes no experimental
solution/recycling/tangent symbols. No new dependencies, release or push are part
of this change.
