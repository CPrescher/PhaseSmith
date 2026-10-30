# Reducing repeated phasing work

The baseline is `bfb4c0f`, with both preceding stopping controls enabled. The
archived 35 development cases complete 37,621 scored cycles in 95 starts. Later
starts account for 28,566 cycles (75.93%); 18,882 cycles occur after cycle256 in
those starts. Only seven starts reach CFOM0.9; 60 reach a cycle ceiling, 22 use
the early plateau rule and six meet their time safeguard. Thus the previous
plateau rule removes early work but leaves most later work untouched.

Seven independently audited correct structures finish above the requested
R1=0.06: COD 2011038, 2011590, 2021199, 2011241, 2010705, 2243618 and 1549173.
They consume all four starts even after useful complete models emerge. A phase
quality threshold need not be attainable on measured data; neither low R1 nor
phase quality alone establishes correctness. The diagnosis, per-case slices and
source-report hash are in
`benchmarks/structure-solution-cycle-effort-diagnosis-macos.json`.
The SHELXT listings have fewer listed iterations per trial, with the limitations
recorded by the preceding solver-only comparison. No new SHELXT implementation
code is copied or translated for the policies below.

## Later-start plateau

The opt-in `late_phase_plateau_stopping` control requires work-sliced execution
without adaptive checkpoints. It applies to intermediate starts only:

    floor(T/2) <= j < T-1.

At least the final independent start retains its original full cycle quota,
unless another existing acceptance or the separate consensus rule ends the
search. Single- and two-start searches have no eligible intermediate start.
No cycle quota or global deadline changes. For the standard four starts only
start2 is eligible; start3 retains its 800-cycle fallback ceiling.

Let B(n) be the existing maximum measured-only CFOM over eligible non-omit
cycles among the first n cycles. Handoff the existing best map when

    n >= 256, n modulo 32 = 0,
    all scores finite, B(n) >= 0.8,
    B(n)-B(n-128) <= 0.002.

The longer window protects later improvement. The original early rule and
high-quality target remain. Accepted physical models still require origin,
symmetry, assignment and requested-mode refinement. A handoff is observable as
`late_phase_plateau`. It does not resume the stopped trajectory at full length;
its effect on actual recovery must be evaluated, not assumed.

## Independent-model consensus

The opt-in `model_consensus_stopping` control requires phase-origin selection
and the existing ambiguous-assignment checks. After an independent start, test
the top two retained complete models only if both are supplied-symmetry validated,
not disfavoured by assignment metrics, and come from different actual phase
starts with one from the current start. The most recent two origin audits must
have no unresolved count/assignment ambiguity. Both R1 values must be finite
and <=0.1; their absolute R1 difference must be <=0.001 and the existing penalized
independent-family score difference <=2.

Element-count vectors must agree exactly. Full-cell coordinates must admit a
species-preserving bijection under one common translation and optional inversion
with every matched distance <=0.02 Angstrom. With cell basis A, fractional points
p_i,q_j, hand h in {+1,-1} and translation t, distances are

    d_ij(h,t) = min_{z in Z^3} || A (h p_i + t - q_j - z) ||.

A least-populated element supplies anchors, with deterministic element/index
ordering. Enumerate matching target anchors and both hands, fix t by their
coincidence and match remaining atoms to unused same-element targets. The
existing exact minimum-image routine evaluates periodic distances. This is
conservative: fixed-anchor/greedy matching can reject near/ambiguous agreements.
It does not invent a match without a complete labelled bijection. Production
models already pass the existing 0.5-Angstrom separation gate. Public diagnostic
arrays are bounded at 128 atoms, shape/type/finiteness/species/cell checked, and
cancellation/deadlines are checked between anchors and source atoms.

Agreement is a work-allocation heuristic, not a correctness certificate. A
consistent but wrong interpretation remains possible. Original ranking chooses
the reported model; this rule does not average models or insert reference atoms.
Termination is `independent_model_consensus`. The requested R1 target and
`target_met` computation are unchanged, so a consensus stop above that target
honestly reports `target_met=False`. Audits record comparison count, accepted
stop count, the two actual start indices and maximum matched distance.

A legacy candidate's lineage/setting index is insufficient to establish
independent starts. Private state now records the actual start when entering
the retained beam. Multiple origin/settings from one phase start cannot create
consensus. Serialized legacy candidate metadata and ordering are unchanged.
A dedicated test covers that distinction.

## Validation and comparison

Independent NumPy code exhausts species-preserving permutations on small
fixtures. A metric-eigenvalue bound defines a finite lattice-image cube containing
the exact closest image, rather than calling the Rust distance routine. Random
orthogonal/oblique cells, translations, inversion, reordered labels, genuine
displacements, malformed arrays and count mismatches are checked. The factorial
reference is limited to seven atoms; it is not used by the production search.
Rust tests also cover cancellation/deadlines, actual start provenance and the
longer plateau warm-up. Full synthetic P1/P-1/Pm searches check counts, geometry,
original late quotas and bitwise phase prefixes. Physical equations, analytical
values/derivatives and finite support are unchanged; existing independent
physical/finite-difference tests remain applicable. The pinned GSAS-II powder
oracle does not cover this search policy.

The four-mode runner compares control, later plateau alone, model consensus alone
and both, with every preceding best control enabled. All modes use seed0,
four starts, 800 cycles, native13/outer15-second caps, supplied symmetry and final
anisotropic refinement. All 35 development cases and seven references beyond
the native cap remain in denominators. Leading position rotates9/9/9/8 by case;
all 280 predictions precede four reference barriers. The runner completion guard
is tested before collection. Adapter-only revision33 inherits32 without changing
the registry or 35/65 split. The 65 holdout outcomes remain unopened.

The verifier checks current numerical/binding/build-input and actual native
hashes, all actual source-start pairs, quotas and stop reasons; records every
count vector; and compares the exact prior structure/count/strict-success sets.
No own test, build or kernel work overlaps timed corpus collection. Criterion
`cycle_effort_search_` covers full eight-atom Fe/C/N/O searches on a 32-cubed grid
with 2,108 rows. All policies remain false by default. Results follow after the
complete development comparison; any regressed variant remains labelled.

## Complete development results

Every variant retains the exact previous 19/35 correct structures, 26/35 correct
total non-H counts, 19/35 correct per-element counts and 15/35 strict
0.02-Angstrom structures. Every one of the 35 inferred count vectors agrees with
both the fresh control and previous best, including incorrect/no-model outcomes.
There are four no-model outcomes and no PhaseSmith hard timeout. The holdout
remains closed and all seven above-cap references remain in denominators.

| Mode | Summed solve time | Summed wall time | Solver saving | Wall saving |
| --- | ---: | ---: | ---: | ---: |
| Previous best policy | 46.31714 s | 53.60655 s | — | — |
| Intermediate plateau only | 43.83236 s | 50.86310 s | 5.36% | 5.12% |
| Independent-model consensus only | 38.39080 s | 45.60271 s | 17.11% | 14.93% |
| Both | 35.59809 s | 42.69431 s | 23.14% | 20.36% |

With both policies, completed phase cycles fall 38,120-to-23,242, **39.03% less**.
Workspace attempt counters are 38,121-to-23,244: deadlines can interrupt a cycle
before it is scored. Starts fall 95-to 81. Eight intermediate plateaus hand off
best maps, and nine independent-model matches end searches with unused starts.
Fourteen cases trigger at least one new policy (three trigger both), explaining
why the large cumulative reduction is concentrated in longer searches.

Typical paired solver time falls only 1.00%, and typical paired wall time 2.85%.
The ratio of separate aggregate medians would exaggerate typical improvement
and is not used. There are29 faster and six slower wall-time cases. These are
one balanced collection's host-sensitive timings, not a repeated uncertainty
estimate. Existing time safeguards also produce phase-cycle variation between
modes; it is recorded rather than assumed equal work.

P1 time falls 22.74272-to 15.75511 s (30.72% less). Assignment screens fall 671-to 542,
and final fits124-to 103. Screening time falls 8.58536-to 6.81105 s (20.67% less),
and final refinement8.01466-to 6.76088 s (15.64% less). The result saves repeated
model interpretation/refinement as well as Fourier cycles. The original final-fit
convergence criteria, requested R1, physical equations and analytical derivatives
remain unchanged.

All nine scalar-prefiltered geometric comparisons match. Three consensus stops
(COD 2021292, 2010543, 2011393) remain previously failed reference-audit cases with
incorrect counts/labels. Consensus therefore must not be interpreted as proof
of correctness; it removes repeated work on an interpretation that multiple
starts reproduce. None of these outcomes is promoted to an R1-target success.
The complete counts and independent reference audits remain explicit. A noisy
synthetic regression also exercises a real consensus stop, distinct start IDs,
correct inferred counts and an honestly unmet requested R1 target.

For the 34 completed SHELXT timing pairs, its listings again report 108 trials and
12,106 listed iterations. PhaseSmith uses 91 starts and 36,120 completed cycles
in the fresh control, versus77 starts and 21,242 cycles with both policies.
The ratio of completed PhaseSmith cycles to SHELXT's listed work falls from
about 2.98 to 1.75. N(iter) remains a listing quantity, not an independently
instrumented executed-cycle count. The SHELXT timeout remains in all 35-case
wall/recovery results and its partial listing is preserved separately.

In this development collection the median paired PhaseSmith library-solve /
SHELXT printed-internal-time ratio falls 1.16580-to 0.85130. Including startup,
the all 35-case paired wall ratio falls 1.31287-to 1.06319. These timing boundaries,
reflection grids/limits and isotropic/anisotropic endpoints differ; they do not
establish general equivalence or unseen-data performance.

The isolated full synthetic search has identical printed work in all variants
(four starts,2,000 cycles). Its median is1.12758 s for control and 1.11401 s for
both policies (about1.2% less); this is not evidence for a stopping-work benefit
on that fixture. The complete development campaign, where the rules actually
fire, supplies that evidence. Exact executable/source hashes, raw samples and
logs are retained in the kernel artifact; historical Criterion deltas are not
used as same-build comparisons.

Evidence is recorded under
`benchmarks/structure-solution-cycle-effort-{control,late,consensus,cache,comparison,retention,kernels,diagnosis,validation}-macos.json`
and `structure-solution-cycle-effort-shelxt-{control,late,consensus,cache}-macos.json`.
The verifier checks58 archived/current source and build-input hashes per mode,
actual native hashes, every quota, consensus source pair and unchanged count
vector. All280 predictions precede all four reference barriers. There are no
post-collection source exceptions. Both controls are the selected development
variant and remain opt-in.

Validation: `cargo fmt --check`; workspace Clippy with all targets/features and
warnings denied;424 Rust tests passed,34 ignored;2,645 Python tests passed,
11 skipped,34 deselected. The final Python build enables both experimental
solution and spot modules. The53 focused cycle-policy tests include independent
geometry/phase references and the noisy end-to-end consensus regression.
