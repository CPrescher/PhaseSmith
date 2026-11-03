# Count-free tangent phases, annealing and negative quartets

The combined symmetry-recycling path now places these updates after origin
recovery and checks the restored symmetry phases. See
`solution-tangent-recycling.md` for the follow-up. The measurements below retain
the original unconstrained placement, available with `tangent_in_recycling=False`.

Experimental extension, independently derived from phase-invariant equations. Tangent and seven-magnitude quartet
relations are described in the
[IUCr direct-methods introduction](https://www.iucr.org/what-we-do/education/pamphlets/introduction-direct-methods).
The annealing schedule and sign selection follow the published idea of cooling
phase perturbations and using quartet consistency; see
[Sheldrick (1990)](https://doi.org/10.1107/S0108767390000277).
The bounded choices below are PhaseSmith conventions, not recovered SHELXS
probabilities or a claim of an exact SHELXS reimplementation.

## Conventions and equations

Retain one h from each Friedel pair (`h > -h` lexicographically), with
`phi(-h)=-phi(h)`. All observations, including signed intensities and sigmas,
remain unchanged. Compute epsilon from operations with `hR=h`; use nonnegative
I/epsilon to estimate smooth mean intensities in up to eight resolution shells
(each initially at least 32 independent Friedel pairs), with linear interpolation
between shell centers. `E=sqrt(max(I,0)/(epsilon*mean))`;
`Eupper=sqrt(max(I+2 sigma(I),0)/(epsilon*mean))`. This empirical normalization
has no composition or atom-count input. Very sparse shells and mixed scattering
species limit the statistical interpretation.
Means are evaluated after scaling intensities by the largest positive measured
intensity to avoid overflow. A shell mean below 1e-12 of the global mean uses
that global mean. Coincident shell centers merge with population weights before
interpolation; all-one-resolution data therefore have a single mean. Normalized
values are invariant to a common intensity/sigma scale.

Strong variables have E>=1.2 and I>=2 sigma, at most the 256 largest E. Triplets
use only this set. For each canonical target h retain at most 64 strongest
unordered pairs k,l with k+l=h; discard repeated independent variables.
`w=E_k E_l`, `theta_h=arg sum w exp(i[phi_k+phi_l])`.
A zero sum retains the input phase. Negative quartets contain four distinct
strong independent variables with signed HKLs summing to zero. All three cross
reflections must be measured and have Eupper<0.9; absences, 000 and missing data
cannot qualify. Rank by the product of principal E values times the product of
`1-Eupper/0.9` for the three cross terms. Retain at most 4,096 quartets and 16
per variable. Pair-list and combination caps are explicit resource bounds.
Each weak-cross group retains at most 32 pairs. Discovery stops after 2,000,000
pair combinations or 32,768 eligible candidates, and records `truncated=True`.
These are computation limits, independent of the searched atom count. Preparation
accepts at most 100,000 expanded reflection rows and checks cancellation/deadline
throughout relation discovery. The plan is prepared once and reused across starts.

For diagnostics and independent analytical-gradient tests use
`T=sum w[1-cos(phi_h-phi_k-phi_l)]/sum w` and
`Q=sum v[1+cos(phi_a+phi_b+phi_c+phi_d)]/sum v`.
Absent relation classes contribute zero, with an availability flag/count.
Values and phase gradients are accumulated in the same relation pass:
`dT/dphi_i=sum w sin(S) signed_incidence_i/sum w` and
`dQ/dphi_i=-sum v sin(S) signed_incidence_i/sum v`.
These are phase-consistency diagnostics, not likelihoods or correctness proofs.

Every optional reciprocal phase step freezes the strongest 40% of the selected
variables by current calculated magnitude. For the others propose the tangent
phase. During the first 12 cycles perturb it by either sign of
`delta=min(pi/3, 0.35*0.85^cycle*(-ln u))`, using deterministic stateless u in
(0,1). Choose the sign with smaller incident-quartet Q contribution; without
quartets use the seeded sign. Blend one quarter of the wrapped phase correction
into the atomic phase. Backtrack that blend up to six halvings if global Q
increases, then decline the proposal if Q still increases by more than 1e-12.
After cooling ends use unperturbed tangent proposals. This is a conservative,
explicit annealing variant; the unperturbed search remains available for ablation.

The existing hybrid magnitude update, FFT atomicity support, omission schedule,
whole-orbit editing, full shortlist, ADP derivatives and original-data final
checks are retained. There is no new spatial cutoff or claimed coordinate
Jacobian for this discrete phasing step. No external solver is a runtime dependency;
GSAS-II supplies no corresponding phasing oracle. Independent NumPy relations,
complete steps and phase finite differences provide the numerical reference.

## API and observable work

```python
from phasesmith.single_crystal.solution import SolutionOptions, solve_structure_elements

result = solve_structure_elements(
    observations,
    cell=cell,
    space_group=space_group,
    elements=["Bi", "Se", "I"],
    options=SolutionOptions(phasing="dual_space", trials=1, cycles=800),
    tangent_refinement=True,
)
```

Only element types, integrated observations, cell and exact symmetry are supplied.
Atom counts and assignments still follow the shared orbit search. This option
currently belongs to the element-only solver and requires `phasing="dual_space"`;
unsupported combinations raise an error. It is off by default. The separate
`symmetry_recycling=True` option can be combined with it, but their speed benefits
must be measured together.

`result.shared_search.tangent_trials` records the independent measured variable
count, strong selection, actual triplets/quartets, truncation, completed and
annealed steps, proposals, accepted phase changes, declined steps and per-step
quartet scores. A zero quartet count means there was no eligible measured weak
cross evidence. A zero score without relations is unavailable evidence. The
guard compares each reciprocal proposal with its input; subsequent real-space
projection can increase Q again. Q does not certify structure correctness.
An accepted structure checkpoint ends the trajectory before the next optional
phase step. No extra phase updates are counted after acceptance.

## Validation and comparison protocol

`tests/test_solution_tangent.py` checks normalization against NumPy for five
symmetry groups and twelve orders of intensity scale; sparse and coincident
resolution shells; measured-only triplet/quartet discovery against exhaustive
small-graph enumeration; randomized fused values and phase derivatives against
independent dense incidence matrices; centered finite differences with 2e-9
absolute tolerance; translation/hand invariance; complete annealed/cooled steps
against NumPy; deterministic repeatability; and quartet non-increase. All-zero
phases satisfy the triplet objective while maximally violating an available
negative quartet, checking the reason for the guard. Missing or uncertain cross
reflections cannot qualify. Synthetic P1, P-1 and Pm element-only solves check
actual phase changes and correct inferred counts. Invalid inputs, immutable
plan metadata, cancellation and deadlines have explicit coverage.

`benchmarks/solution_tangent_comparison.py` runs three serial pairs per structure,
alternating order by repeat, with seed 0, one start, 800 cycles, 60 seconds,
R1 target 0.06 and the same screening/refinement budgets. Paired modes differ
only in `tangent_refinement`. A second complete campaign enables symmetry
recycling in both modes. No formula or reference coordinates enter the solves.
Source and native-library hashes, numerical controls, input hashes and raw
effort are saved before references are read. Post-solve gates require exact
full-cell atom counts, maximum matched coordinate error below 0.02 angstrom,
no paired error increase above 1e-4 angstrom or R1 increase above 1e-7, and both
exported CIF frames reproducing R1 within 2e-10. These are reused development
datasets, not a blind coverage estimate. Repeats use the same seed to measure
timing repeatability, not independent search success probability.

Run the campaigns with:

```sh
.venv/bin/python benchmarks/solution_tangent_comparison.py --output runs/tangent-check --repeats 3
.venv/bin/python benchmarks/solution_tangent_comparison.py --output runs/tangent-recycling-check --repeats 3 --symmetry-recycling
cargo bench -p phasesmith-single-crystal --features solution --bench solution -- tangent_
```

## Measured outcome (macOS arm64, 29 October 2026)

All 36 saved solves pass every quality gate, including all 18 tangent-enabled
solves. The recovered non-H full-cell counts, geometry and R1 are:

| Structure | Inferred counts | Maximum matched error (angstrom) | Amplitude R1 |
| --- | --- | ---: | ---: |
| K[BrF4] | Br4 F16 K4 | 0.005909 | 0.0222664004093 |
| BiSeI | Bi4 I4 Se4 | 0.002516 | 0.0522850546588 |
| Thiourea | C4 N8 S4 | 0.010372 | 0.0494090499683 |

Three-pair median wall times, including plan preparation and complete search:

| Structure | Dual-space baseline (s) | With tangent (s) | Baseline/tangent | With recycling baseline (s) | Recycling + tangent (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| K[BrF4] | 0.5395 | 0.5490 | 0.983 | 0.7881 | 0.8004 |
| BiSeI | 0.3722 | 0.4724 | 0.788 | 0.2559 | 0.3212 |
| Thiourea | 4.6447 | 3.4749 | 1.337 | 1.5386 | 6.0270 |

The new option improves thiourea by 1.34x without recycling, with screening/final
fit counts changing from 156/34 to 156/30. K[BrF4] uses the same 42/18 fits and
accepts the unchanged first checkpoint candidate: its small time difference is
overhead. BiSeI changes from a cycle-4 to cycle-16 accepted checkpoint and uses
28/12 fits instead of 23/10, causing a measurable regression.

Combining the options is particularly unfavorable for thiourea: 258/46 fits
replace 50/14 and the median is 3.92x slower. The changed phase trajectory changes
the downstream candidates and search workload. For recycled BiSeI the fit counts
stay 14/7 but acceptance is again delayed to cycle 16. A lower reciprocal phase
inconsistency does not guarantee cheaper subsequent atom assignment. No
compound-specific dispatch, altered target or reduced shortlist hides these
outcomes. `tangent_refinement` remains off by default; these measurements do not
justify replacing the existing default or the faster recycling configuration.

K[BrF4] has 16,056 retained triplets and **zero** eligible negative quartets.
The tangent/annealing update still runs, but no quartet evidence is claimed for
that case. BiSeI has 7,819 triplets / 400 quartets; thiourea 10,716 / 83. None
reaches the quartet discovery work cap. The first accepted checkpoint requires
3 actual tangent steps for K[BrF4] and 15 for the others; later atomic projections
can undo an earlier improvement in Q, as the saved per-step audits show.

Frozen complete reports:

- `benchmarks/structure-solution-tangent-macos.json`
- `benchmarks/structure-solution-tangent-symmetry-recycling-macos.json`
- `benchmarks/structure-solution-tangent-kernel-macos.json`

The realistic eight-atom / 2,108-reflection kernel benchmark measures medians of
41.62 ms to prepare the reusable plan, 38.81 microseconds for fused invariant
values/gradients and 44.85 microseconds for a complete annealed phase step. The
guard evaluates quartet values and derivatives together without repeatedly
evaluating the unrelated triplets during backtracking. The preparation cost is
included in the end-to-end comparisons above; it is not hidden as warm-up.

Validation: 1,439 Python tests passed (11 external-data skips, 34 deselections),
394 Rust tests passed (34 ignored), strict workspace Clippy, Cargo formatting and
Python Ruff checks passed. No global tolerance was loosened. A separate
default-feature wheel is checked with SciPy blocked and without experimental
phasing symbols. No dependencies, release or push are added.
