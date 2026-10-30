# Search improvements after fused phase workspaces

This review uses the complete 35-case development report at `2ff1e17`, current
Rust source, the published SHELXT paper.
No new structure searches or holdout audits were run. All 65 holdout outcomes
remain closed. These are ranked experiments, not implemented solver changes or
promised performance gains.

## Evidence from the current pipeline

The latest panel spends 32.962 seconds in phase search, 27.922 in final refinement,
8.859 in isotropic screening, 1.530 in origin search, 1.238 in symmetry recycling
and 1.941 in peak completion. These recorded stage totals are not a new CPU
profile and do not include all startup/orchestration costs.

Ten failures have the correct total non-H count but incorrect element counts.
Six incorrect models meet the R1 <=0.06 early-stop target, while six correct
models do not meet it. For example, COD 2021310 returns Br3/O6/N12/C24 at R1
0.03246, against development counts Br3/O6/N3/C33. This establishes an assignment
failure, not proof that all candidate coordinates or atomic topology are already
correct. Reference counts are used only for this retrospective development audit.

The 96 phase starts end with 82 cycle-limit, eight quality-target and six phase
time-limit stops. This does not prove the cycle-limit starts could have stopped
earlier. Current early quotas are the fixed sequence 400/400/800/800, subject to
quality exits and the time backstop. With phase-origin selection enabled, each
start can trigger assignment/refinement before the next start; maps are not
globally pooled first in this path.

## Published SHELXT routes and existing coverage

[Sheldrick (2015), sections 2–5 and figure 1](https://journals.iucr.org/a/issues/2015/01/00/sc5086/)
describes Patterson/dual-space phasing, correlation and weak-reflection quality,
increasing restart effort after poor quality, phase-based symmetry selection,
symmetry recycling, integrated-density element assignment and isotropic
conjugate-gradient refinement. Independent stages can run in parallel. Its
organic bond-angle score is optional and inappropriate as a universal mineral
criterion. It uses the Laue group to test space groups rather than accepting an
exact space-group assignment as final.

PhaseSmith already implements most of the corresponding known-symmetry stages,
including carbon/boron/oxygen density-scale proposals and isotropic competition.
Adding them again is not a new route.

## Priority 1: eliminate duplicate final-refinement evaluations

This is an independent first-principles optimization, not a claim about an
identified SHELXT implementation detail. `prepare_data` already merges each
allowed symmetry/Friedel family, then repeats the same amplitude across its P1
expansion. P1 expansion is needed for phasing. However,
`PreparedSymmetricModel::evaluate` loops over every expanded row even after
symmetry has been imposed on coordinates and anisotropic tensors.

Across these 35 inputs, expanded rows per independent family range from 2.0 to
28.74, with median 3.88. This is potential work reduction in the relevant
reflection loops, not a predicted whole-solver speedup. Low-symmetry expensive
cases may have only two equivalent rows, and linear algebra/setup costs remain.

For family f with multiplicity m_f, observed amplitude o_f and constrained
calculated amplitude a_f(theta), retain the existing expanded-data weighting:

    k = sum_f m_f a_f o_f / sum_f m_f a_f^2
    r_f = sqrt(m_f) (k a_f - o_f) / sqrt(sum_f m_f o_f^2)
    J_f = sqrt(m_f) d(k a_f)/dtheta / sqrt(sum_f m_f o_f^2)
    objective = sum_f r_f^2
    R1 = sum_f m_f |k a_f-o_f| / sum_f m_f o_f.

These preserve the mathematical scale, objective, gradient and normal matrix
of the current expanded objective. Simply dropping equivalent rows without
weights changes the objective when orbit multiplicities differ. The model-rank
sample count stays the independent family count. Original source observations,
P1 phase arrays, signed intensities, correctness gates and resource admissibility
remain unchanged; existing input-derived caps must not be relaxed as part of
the comparison.

A diagnostic folds the current native residuals and analytical Jacobians on ten
synthetic cases: isotropic/anisotropic modes in P1, P-1, Pm, P3121 and Fm-3m,
including special positions, centring, noncubic metrics and unequal family
multiplicities. Scale, R1, objective, gradient and normal matrix agree at the
explicit local 1e-10 relative/2e-12 absolute tolerances. This checks the reduction
algebra; it does not implement a reduced evaluator, test its finite differences
or benchmark its speed. Evaluating only one member changes floating-point
round-off and reduction order, so a production version needs independent values,
derivatives and a complete matched 35-case campaign.

## Priority 2: retain and resolve ambiguous element assignments

The current assignment neighborhood proposes single-site relabels and paired
swaps at fixed coordinates and U, then fits only two neighbors isotropically.
The first final model meeting the R1 target can suppress its runner-up and stop
further starts. This is a concrete explanation for why low R1 does not guarantee
the correct labels, although causation for each failure needs a controlled test.

Test a bounded beam of genuinely different element assignments, with short
coordinate/U relaxation before rejection. Propose a small number of coordinated
relabels of ambiguous sites, including changes that require crossing a
fixed-geometry uphill step; preserve required element types and orbit topology.
Element counts remain inferred and can change with the proposed labels.
Do not enumerate all full-cell count combinations. Carry atom-count alternatives
separately where peak inclusion remains uncertain.

Use the gap between comparable relaxed assignments as an ambiguity diagnostic.
A low R1 with close competing labels should earn a small extra discrimination
budget rather than immediately suppressing alternatives. The gap is not a
correctness certificate. Joint density-scale/U evidence, resolution-dependent
scattering and optional element-appropriate coordination can propose candidates;
chemistry must remain a soft, explicit hypothesis, without importing formulas or
imposing organic valences on minerals. Similar atomic numbers may remain
unidentifiable from a given dataset and should be reported as alternatives.

Changing only a global R1 threshold is poorly motivated: correct and incorrect
models cross the existing target in both directions. First measure whether
retained correct label alternatives are being discarded, never assuming that
more fitting will fix fundamentally ambiguous observations.

## Priority 3: allocate work by phase quality and candidate diversity

Test short independent starts followed by selective continuation of promising
ones, using phase quality, its improvement, symmetry consistency and diversity
of the resulting density/settings. Reuse existing cheap score kernels; avoid
repeated final fits of equivalent candidates from different starts. Keep a
bounded exploration allocation so poor initial quality cannot permanently
exclude a recoverable start.

This differs from the earlier stable-map checkpoint experiment and fixed
half-work quotas. A stationary map can be wrong. Pooling candidates may also
slow easy cases that currently solve on the first start, so use a measured,
bounded policy and retain a first-start fast path. Log all actual work and
missed promotions rather than selecting a new threshold from only successful
cases. This is a higher-risk algorithm change than multiplicity reduction.

## Later scope and validation order

Independent-start parallelism is available as a separate latency experiment;
compare fixed total work/CPU budgets and deterministic result ordering. It does
not establish improved algorithm efficiency. Space-group inference from Laue
symmetry is an important capability gap, with a separate benchmark contract;
the present known-symmetry results do not demonstrate that capability.

Implement and measure multiplicity-weighted refinement first, then assignment
ambiguity handling, then quality/diversity scheduling, one change per matched
experiment. For each production change, use all 35 development cases with the
same caps, seed, reflection inputs and final anisotropic mode; compare recovery
case sets and per-element counts as well as speed. Use several fixed seeds before
freezing the policy. Holdout outcomes remain unopened during development.

The read-only evidence and synthetic checks are preserved in
`benchmarks/structure-solution-next-routes-macos.json`. Reproduce with:

```sh
.venv/bin/python benchmarks/solution_next_routes_probe.py --output /tmp/solution-next-routes.json
```

No production numerical behavior changes in this investigation.
