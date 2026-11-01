# Unknown space-group search: MgSiO3 validation

The subsequent [fast/auto/thorough effort report](space-group-search-speed.md)
records the current execution and timings. This page preserves the original
v31–v33 experiments and their bounded-coverage evidence.

The native original-frame workflow now ranks **Pbca, number 61**, and recovers
**Mg16 Si16 O48** with exact space group and Laue group withheld. The measured
full command takes **36.16–36.50 seconds** after shared-map reuse and four-worker
P1 initialization, compared with **47.64–47.73 seconds** before those changes.
This is exposed development/regression data with a supplied cell and element
types, rather than a blind structure-solution benchmark. Only a bounded subset
of models is searched; uniqueness is not established.

Use the [native workflow guide](single-crystal-native-workflow.md) to reproduce
the run and the [assessment guide](space-group-search.md) for equations and
resource contracts. The new mode is available through the typed Rust API and
native example executable. It does not add a Python or Dioptas discovery frontend.

## Scientific input and coverage

The recipe reads original EIGER counts, source detector calibration and explicit
acquisition/DAC assumptions. Cell 8.8101 × 5.1824 × 18.2288 Å, right angles,
wavelength 0.2846 Å and the element types Mg, Si, O are supplied. Atom counts,
coordinates, source HKLs, exact symmetry and Laue group are withheld from the
worker. All eight source files, 470,052,794 bytes, retain their registered hashes.
Reference CIF grading occurs after the worker exits.

Unknown symmetry changes the scientific reduction path deliberately: predictions
retain primitive reflections that an exact group would exclude; whole metric
orbits protect holdout disjointness without asserting intensity symmetry. The
workflow assesses physical aperture intensities before symmetry-dependent
empirical scaling, then uses the supported provisional Laue group for scaling.
Final absence assessment retains finite signed measurements without an SNR or
positivity filter. Masked/missing reflections never become invented zeros.

| Evidence or output | Result |
| --- | --- |
| Catalogue | All 530 native Hall entries considered; 345 distinct metric-compatible settings after exact-alias coalescing |
| Preliminary intensity symmetry | mmm supported by 1,490 constraints; reduced chi square 1.3404 |
| Final intensity symmetry | mmm reduced chi square 1.2676 |
| Final assessment input | 4,406 measured rows, 3,010 Friedel families |
| Pbca absence evidence | 420 measured forbidden families, two positive >5-sigma violations; clipped log evidence 647.52 |
| Phase screening | 64 settings: 19 consistent, 44 inconsistent, one uninformative; 281 not phase-tested |
| Atomic searches | 12 candidate settings; termination `candidate_budget`; failed/unattempted settings retained |
| Reduced observations for structure search | 4,844 used, including 594 negative |
| Profile learning | 246 teachers, 8,153 retained target groups, 14,933,220 samples |
| Shared P1 map | 64 × 32 × 128; four deterministic starts × 800 cycles; selected start 3 |
| Selected setting | Pbca, Hall `-P 2ac 2ab` |
| Selected composition | Mg16 Si16 O48; ten asymmetric sites, 80 expanded sites |
| Reference coordinate RMS / maximum | 0.004436 / 0.006125 Å; labelled recovery and strict 0.02 Å check pass |
| Own-group amplitude R1 | 0.051433695723409355 |
| Common-measurement amplitude R1 | 0.0616558093202203 |
| Common-set BIC-like score / parameters | −18988.03 / 41 |

The common score includes measurements forbidden by each tested candidate, so
discarding difficult observations cannot improve that candidate's rank. Some
lower groups fit a slightly lower common R1 with more parameters. The approximate
complexity penalty and phase/absence evidence rank Pbca first; none is a
calibrated posterior. Two P1-fallback candidates fail the claimed-group common
symmetry check. A small R1 alone is insufficient evidence of a correct model.

`candidate/space-group.cif` preserves the selected Hall setting and asymmetric
sites. `candidate/best.cif` retains the complete P1 model in the density frame.
Independent Gemmi expansion verifies the setting CIF's finite coordinates and
Mg16 Si16 O48 composition. The selected isotropic polish terminates by its
gradient criterion after four accepted steps; it is not a standalone refinement.

## Measured optimization stages

Controls were committed before each corresponding frame replay. The v30 evidence
plan was extended by v31 phase screening, v32 shared-map model initialization,
then v33 bounded independent phase workers. Original protocols and trial outputs
are preserved; no outcome-driven change to a frozen protocol or tolerance was made.

| Execution | Source | Full command, s | Observer wall, s | Search stage, s | Shared P1 map, s |
| --- | --- | --- | --- | --- | --- |
| v31: independent candidate phasing, two runs | `4d11bbf`, `8678ca1` | 47.73, 47.64 | 47.69, 47.59 | 30.25, 30.48 | 7.56, 7.65 |
| v32: reuse the P1 map | `6028e9f` | 42.26 | 42.22 | 24.71 | 7.64 |
| v33: four ordered P1 workers, two runs | `254b7af` | 36.50, 36.16 | 36.46, 36.12 | 19.01, 19.07 | 2.01, 2.01 |
| v33: standalone, empty inherited environment | `a7c9a07` | 36.08 | Not used | 19.07 | Archived separately |

Reusing the map reduces the search stage by about 19%; four-worker initialization
then reduces it by another 23%. Relative to v31, complete command time falls by
about 24%. These are sequential warm-cache observations on one Apple M4 Pro,
14 logical CPUs and 48 GiB RAM, macOS 26.6.2, native arm64 release builds with
Rust 1.99.1. Compilation and test jobs were stopped during timing. This is not a
cold-cache estimate, statistical confidence interval or cross-platform speed claim.

The v32 path tries 12 settings from the shared map and uses independent fallback
phasing for two settings, totaling eight new fallback starts. Reuse changes
initialization: v31 R1 0.0514133 becomes v32/v33 R1 0.0514337, while labelled
structure recovery remains correct. That change is recorded rather than called
bitwise parity. Serial v32 and parallel v33 retain bitwise-identical measurements,
selected candidates, shared-map arrays, histories, origins and group-search
records after excluding elapsed timers. Repeated v33 and standalone execution
retain the same results. Time-limited/cancelled runs do not promise identical
coverage across different host speeds.

The supplied-Pbca v29 control remains bitwise identical to the previous native
coordinator: 4,125 used observations, 220 negative and R1 0.041128354912261904.
That control takes about 16.06 seconds. Its earlier strict Python/native final
candidate mismatch remains documented in the
[native coordinator validation](single-crystal-native-validation.md); this work
does not change that historical qualification. Comparing its R1 directly with
unknown-group R1 is inappropriate because the measurement selection and internal
isotropic/anisotropic search endpoints differ.

### Full stage breakdown

The second v33 observer run records the following native stage times. Callback
archives, startup, verification and post-run grading account for the remaining
wall time; stage subtotals should not be added again to the complete command.

| Stage | Seconds |
| --- | ---: |
| Shadow masks | 2.905 |
| Detection and linking | 3.138 |
| Indexing and geometry | 0.144 |
| Integration and profile learning | 5.679 |
| Profile fitting and scaling | 4.280 |
| Structure search, including symmetry assessment | 19.068 |
| Native coordinator total | 35.217 |
| Native worker process | 35.273 |
| Source verification, before and after | 0.334 |
| Reference assessment, after process exit | 0.191 |
| Observer wall / complete command | 36.118 / 36.16 |

Within search, shared P1 initialization takes 2.012 seconds, phase/origin tests
sum to 2.598 seconds, and candidate-model runs sum to 12.842 seconds. The selected
Pbca candidate run itself takes 0.261 seconds. Remaining search time includes
evidence construction, common-model scoring and orchestration. Improving the
winning fit alone therefore has limited impact; redundant candidate work is the
larger target.

## What the paper and decompilation establish

[Sheldrick (2015)](https://journals.iucr.org/a/issues/2015/01/00/sc5086/) describes
P1 phasing followed by phase-based symmetry/origin tests, recycling, element
assignment and isotropic polishing. It uses weighted wrapped phase residuals
and retains ambiguous groups. Its P1 merit combines correlation and weak-reflection
information; additional starts depend on solution quality. It warns that lower
R1 can favor an incorrect subgroup or element assignment. PhaseSmith's absence
prescreen is an additional independent choice, not SHELXT's phase-based group test.

The local SHELXT 2018/2 audit was inspected again, including actual pseudocode
branches rather than only routine names. The executable SHA-256 is
`4d712cb17f8952e1f59f476af7c61ad4ab5261e594c75441c7f81dcefe7b5560`.

| Evidence | Concrete observation | Independently implemented action or remaining work |
| --- | --- | --- |
| Publication, sections 3–5 | Shared P1 phases precede symmetry/origin assessment and atom assignment | One map supplies phase screening and direct candidate initialization |
| `_spgnon_.c`, lines 970–1009 and 1948–2024 | Score guards precede origin refinement and saved hits; one branch abandons an accumulated weighted squared residual early | Cheap phase screening is implemented; partial-sum pruning and origin-dimension reduction remain future work |
| `_spgcen_.c`, lines 900–950; `_savhit_.c`, lines 28–58 | Candidate score checks precede refinement/storage | Explicit bounded hypotheses and candidate budget; no inferred binary thresholds copied |
| `_solvp1_.c`, lines 1408–1428 and 2010–2040 | Calls consistent with correlation scoring and calculated-amplitude normalization before a weak-score accumulation | Adaptive correlation/weak-score stopping is a promising separate experiment; current native starts still run 800 cycles |
| Audit README and status inventory | 24 of 25 targeted routines exported; large `_MAIN__` and some outlined helpers unresolved | No claim to a recovered complete algorithm, original types, or execution schedule |

The pseudocode, probes and inferred dataflow remain external under
`/Users/clemens/Codex/shelxt-binary-audit-2026-10-28`. Symbols and decompiler-local
types do not prove semantics. No generated implementation, numerical constants
or inferred data structures were copied or mechanically translated. The
publication and existing independently tested crystallographic kernels govern
the production implementation.

## Direct SHELXT timing comparison

The installed SHELXT 2018/2 x86_64 executable runs via Rosetta. It received the
same 4,844 signed native measurements as fixed-width HKLF4, with export rounding
recorded, cell/elements and mmm inferred by the preceding native assessment.
Exact group, coordinates and correct atom counts were withheld; `UNIT 1 1 1`
is a dummy. The descriptive TITLE contains the example name. No raw-frame
reduction is included in these SHELXT times.

| SHELXT mode, one worker | Process seconds, three runs | Typical reported stages |
| --- | --- | --- |
| Default | 3.732 first launch; 2.247, 2.260 subsequent | P1 0.76 s; group determination 1.38 s; isotropic assignment/polish 0.046 s; total about 2.20 s |
| `-a`, retain more groups | 2.482, 2.415, 2.470 | Total about 2.43 s; five candidates instead of one |

Thus the P1 stage is below a second here; the complete measured SHELXT solution
and group search is about 2.25–2.48 seconds after first launch. The native search
remains substantially slower at about 19 seconds. Both rank Pbca first. SHELXT's
top saved candidate reports R1 0.033 with Mg8 Si24 O48; its positions match the
reference at 0.00663 Å RMS when element labels are ignored, but its typed counts
do not match Mg16 Si16 O48. Those R1 values use different processing/objectives
and cannot establish which model is better.

The logs expose a major workload difference. SHELXT uses a 0.8 Å phasing cutoff,
a 36 × 18 × 72 Fourier grid and three starts of 100, 110 and 121 cycles. The
native initializer uses data to 0.5 Å, a grid with **5.62 times more points** and
four starts of 800 cycles. SHELXT also reports adding missing Fourier coefficients.
This does not establish its exact internal stopping rules or final fit dataset.
It does show that equal input observations do not imply equal computational work.

The next measured experiments should test a coarser phasing grid and adaptive
stopping while preserving the original measurements for final assessment, then
cheap fixed-map/short-isotropic screens before full model fits and redundant
subgroup searches. Cached phase coefficients and dimension-aware origin searches
are further candidates. These changes are not implemented or validated by this
receipt; MgSiO3 alone cannot establish general solution reliability.

## Checks and retained evidence

The final installed release wheel passes **3,292 Python tests** (11 external-data
skips, 34 explicitly deselected), including the executable/reference comparisons.
Rust 1.89.0 passes **536 workspace tests** with all features (34 ignored), plus
three native-example tests. Strict all-target/all-feature Clippy, formatting,
warnings-denied Rustdoc and minimal Rust feature builds pass. The installed wheel
also passes native detection, row search, solution and bundled-filter HDF5 smoke
checks in a NumPy-only environment. Public API, version, equation-document and
skill synchronization checks pass. Strict documentation builds contain 185 HTML
pages, with 23,191 local links checked, no broken targets/fragments and every page
reachable through navigation.

Native evidence equations agree with the independent NumPy reference on
deterministic orthorhombic, monoclinic, hexagonal and cubic cases and on both real
345-setting assessments (maximum absolute log-evidence discrepancy 1.82e-11).
Tests also cover ambiguous absences, genuinely missing rows, all-negative
measurements, catalogue aliases/orientations, phase-map identity guards,
cancellation, input types and serial/parallel reproducibility. Existing fused
coordinate/ADP and phase-origin derivative kernels remain unchanged. The powder
GSAS-II oracle does not cover this discrete assessment/coordinator change.

The machine-readable receipt is
`validation/results/space-group-discovery-0.8-macos.json`. Large source files,
all trials, native binaries, numeric evidence, full candidate records, independent
grading and distributions remain under:

```text
/Users/clemens/ChatGPT/PhaseSmith/space-group-discovery-0.8-20261101
```

Only rebuildable local build/documentation products are removed after checks;
raw data, earlier trials, frozen controls and receipts are retained. Remote
Linux/Windows/macOS CI is configured but was not executed in this local receipt.
Removing this worktree's build directory and temporary documentation site removed
8.19 GB of allocated products; observed filesystem free space increased 7.87 GB.
