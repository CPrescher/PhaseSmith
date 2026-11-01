# Matched-prior symmetry and acceptance validation

This follows the [first 14-case comparison](unknown-group-corpus-validation.md).
The same exposed COD cases and measured HKL values are reused. Both programs
now receive the same Laue class and lattice centring, cell and non-H element
types; exact space group, atom counts and coordinates remain withheld.
No structural model or reference grading enters numerical selection.

**With matched priors, fast and auto recover 11/14 complete element-labelled
structures, including the published group in 10/14.** All four numerically
accepted cases pass both reference checks. SHELXT remains faster on this set;
automatic effort is still conservative and adds work without increasing this
sample's recovery count over fast.

The subsequent [candidate fitting and fallback report](candidate-fitting-speed.md)
measures exact fitting reuse and deferral of redundant group descriptions. It
preserves every retained result and acceptance flag, with a 5.5% isolated fitting
gain and essentially unchanged complete-search time. The measurements below
remain the historical matched-prior baseline.

## Measured results

There are 112 sequential timed executions, totalling 216.70 seconds, with no
hard timeout. Repeated variants use the median within each case, followed by the
median across all 14 cases, including failures. Auto and SHELXT-four have one
execution per case. Startup, assessment and output files are included.

| Execution | Median seconds | Labelled structure | Published group number | Structure and group | Strict labelled positions |
| --- | ---: | ---: | ---: | ---: | ---: |
| PhaseSmith fast, four P1 workers | 1.095 | 11/14 | 11/14 | 10/14 | 10/14 |
| PhaseSmith auto, four P1 workers | 8.082 | 11/14 | 11/14 | 10/14 | 10/14 |
| SHELXT, four workers | 0.483 | 9/14 | 12/14 | 8/14 | 6/14 |
| SHELXT, ten workers | 0.492 | 8/14 | 12/14 | 8/14 | 6/14 |

Primary recovery requires exact non-H element counts, RMS position error <=0.025 Å
and maximum <=0.05 Å after a common origin/inversion. Strict recovery additionally
requires maximum <=0.02 Å. Group agreement is a separate check. The predeclared
secondary criterion ignoring element identities gives positional recovery of
12/14 fast, 12/14 auto, 11/14 SHELXT-four and 10/14 SHELXT-ten. Some failures
therefore concern element assignment, rather than absence of the atomic framework.
No reference is used to select another output model or change a threshold.
Both receive the same curated non-H element list. Native search requires those
types; observed SHELXT models can omit a listed species. Complete labelled
recovery therefore includes element-omission and assignment failures, while the
separate position-only result describes framework recovery.

**A** below means the primary atomic criterion passes, **F** means it fails, and
**—** means no model. The number is the chosen group type, not an agreement flag.
All three native fast repeats have byte-identical full-cell candidate CIFs where
models exist; the no-model outcome repeats too. SHELXT-ten's pass/fail and group
number are unchanged over its three repetitions.

| COD | Reference group | Fast seconds / result | Auto seconds / result | SHELXT-ten seconds / result |
| --- | ---: | --- | --- | --- |
| 2200381 | 221 | 1.821 / A, #221 | 15.691 / A, #221 | 1.299 / A, #221 |
| 2215687 | 221 | 0.262 / A, #221 | 1.356 / A, #221 | 0.489 / F, #221 |
| 2243618 | 194 | 1.131 / A, #194 | 18.104 / A, #194 | 0.891 / A, #194 |
| 2017324 | 194 | 0.432 / A, #194 | 0.447 / A, #194 | 0.488 / A, #194 |
| 1549038 | 4 | 0.839 / A, #4 | 12.830 / A, #4 | 0.493 / F, #4 |
| 2010875 | 4 | 2.360 / F, #4 | 20.754 / F, #4 | 0.492 / F, #4 |
| 2241300 | 56 | 0.660 / A, #56 | 0.676 / A, #56 | 0.471 / A, #56 |
| 2021040 | 33 | 3.273 / A, #33 | 3.278 / A, #33 | 0.889 / F, #33 |
| 2207083 | 139 | 0.210 / A, #139 | 2.243 / A, #139 | 0.257 / F, #141 |
| 2233659 | 139 | 1.669 / A, #121 | 8.345 / A, #121 | 0.498 / A, #139 |
| 2011241 | 2 | 1.495 / F, #1 | 7.819 / F, #1 | 0.471 / A, #2 |
| 2010993 | 2 | 0.370 / A, #2 | 0.374 / A, #2 | 0.250 / A, #2 |
| 2203173 | 167 | 5.211 / A, #167 | 17.210 / A, #167 | 0.666 / F, #155 |
| 2201295 | 152 | 1.059 / — | 9.264 / — | 0.883 / A, #152 |

Compared with the previous unmatched-prior run, labelled recovery increases
from 8 to 11 cases and structure-plus-group recovery from 2 to 10 fast, or
4 to 10 auto. Fast median time changes from 1.016 to 1.095 s; auto changes
from 9.584 to 8.082 s. Total time over the 14 auto runs decreases from 212.21 to
118.39 s (44% observed reduction). This combines better input symmetry and the
policy changes; it is not an isolated arithmetic optimization or an estimate of
general performance. The independently rerun SHELXT controls have unchanged
recovery counts and some timing variation too.

The machine remains an Apple M4 Pro, 14 logical CPUs, 48 GiB RAM, macOS 26.6.2
(25G83). Native is arm64; the pinned, separately supplied SHELXT 2018/2 is x86_64
under Rosetta. Input priors and numerical HKL values now match, while algorithms,
objectives and resource policies differ. Three repeats do not establish cold-cache
statistics or confidence intervals, and small subsecond differences should not
be overinterpreted.

## Acceptance, fallback and remaining failures

Normal auto accepts four first passes and escalates in ten cases: seven fail
the own-group R1 target, two retain group ambiguity, and one has no model. Its
first-pass decisions match standalone fast in every case. All four accepted
cases have correct labelled structures and published group numbers; there is
no observed false acceptance in this sample. This is a small exposed regression
set, not a calibrated guarantee of future correctness.

- The former Pr/In/O false acceptance, COD 2215687, now selects the correct
  Pr3 In1 O1 model in #221. It remains unaccepted because #207 and #221 have
  identical BIC-like scores. No wrong extra O is accepted. Alternative subgroup
  descriptions can remain redundant/ambiguous at this stage; identifying their
  physical equivalence is further work, not a reason to silently remove a gate.
- COD 2017324, 2241300 and 2207083 now recover #194, #56 and #139 respectively,
  rather than accepting the earlier monoclinic subgroup descriptions. The first
  two are accepted; 2207083 retains close group alternatives and is unaccepted.
- COD 2021040 now reaches accepted Pna21 (#33) on fast, instead of needing the
  old 27.56-second auto run to upgrade its Pc model. Its current auto is 3.28 s.
- COD 2233659 retains its correct labelled atomic approximation, maximum error
  0.00245 Å, through auto. P-1 is incompatible with its supplied Laue prior and
  cannot become the replacement. The selected I-42m (#121) still misses the
  published I4/mmm (#139), and the high residual keeps it explicitly unaccepted.
- The remaining complete-structure failures are element assignment in 2010875,
  the unsuccessful 2011241 model, and no model in 2201295. Priors alone do not
  fix these phasing/assignment limits. Only the saved top model is graded.

Both passes remain preserved on all ten natural escalations. None produces an
accepted deeper upgrade on this matched set: the four accepted cases already
finish on fast. Independent forced-budget controls exercise successful deeper
upgrades; those controls are separate from the normal timing table.

Across the normal 14 auto runs, saved counters account for 3.86 s in the shared
P1 map, 6.94 s in origin tests and 106.70 s in candidate model work, including
assignment, fitting and independent phasing/restarts. Search coordination after
assessment totals 117.53 s; assessment, process startup and archives bring this
to 118.39 s. Model work remains about **90%** of total auto time. The next speed
work should address that stage and unnecessary fallback on otherwise good
models, while preserving truthful ambiguity and recovery checks.

## Native contract

`DiscoveryOptions.symmetry_prior: Option<SymmetryPrior>` is an explicit Rust
input. Its complete `laue_rotations`, exact `centring_translations` and nonempty
`provenance` refer to the original direct/index basis. Translations include zero,
lie in [0,1), and use reduced rationals with denominators dividing 24. The closed
conventional prior has at most 48 Laue rotations and four centring translations.
Identity, inversion, closure, duplicates and rotational/lattice compatibility
are checked. The prior must match a metric-compatible catalogue setting.

For a candidate space group G, define

```text
L(G) = rotations(G) union -rotations(G)
C(G) = { t : (identity,t) belongs to G }
eligible(G) iff L(G) = L(prior) and C(G) = C(prior).
```

This is exact group algebra. The Laue prior supplies intensity equivalence; it
does not assert that the atomic structure has inversion. Screw and glide
translations remain unknown. Settings outside the prior retain `prior_excluded`
diagnostics rather than disappearing or consuming model/phase candidate slots.
The diagnostic inversion-only phase test remains available independently.

The initial phase-map processing group is the symmorphic set
`{(R,t) : R in L(prior), t in C(prior)}`. Inferred-symmetry input without a prior
remains supported. Actual measured Laue constraints, chi square, missing coverage
and contradictions are retained unchanged: a supplied prior does not manufacture
redundancy or relabel insufficient evidence as measured support. Original signed
observations remain the source for common-set scoring, including forbidden rows.
Missing reflections never become measured zeros.

The existing adapter accepts `priors.symmetry` with these three fields. Its JSON
translations are arrays of three `[numerator,denominator]` pairs. No exact-group
or model fields are admitted in that record. Search JSON reports the supplied
prior and both `laue_group_supplied` and `centring_prior_supplied` truthfully.

## Acceptance and fallback

The own-group target R1 <=0.06, common-set R1 <=0.08 and phase alpha <=0.3 remain
unchanged. Additional gates reject a candidate when:

- its group fails the supplied Laue/centring prior or measured evidence contradicts
  that supplied Laue class;
- no prior was supplied and a strict higher Laue superset remains supported or
  insufficiently measured, rather than contradicted;
- there are not more independent common Friedel families than fitted parameters;
- different group types remain in the configured BIC-like ambiguity interval;
- an acentric model has no completed inversion-phase test, or that test remains
  consistent with inversion (alpha <=0.3).

The inversion test is performed explicitly before the bounded setting screen,
even when P-1 is ineligible as a model or ranks beyond the setting limit. It
uses the existing cached phase-origin equations and consumes the same bounded
phase/time budget. Laue and atomic centrosymmetry are separate decisions.

These gates establish numerical plausibility with stated unresolved evidence,
not correct chemistry or unique structure. No R1 threshold can guarantee those.
Automatic effort escalates after rejection using the same original shared deadline.
An accepted deeper result can replace the first candidate. If neither pass is
accepted, an existing first approximation is retained; a lower R1/BIC alone is
insufficient to upgrade it. If the first pass has no model, the deeper available
approximation can be retained. Both pass reports and available maps survive,
with an explicit `selection_reason`.

The numerical profile, Fourier, structure-factor and coordinate/ADP derivative
kernels are unchanged. This change concerns group algebra, initialization priors
and orchestration decisions. New profile derivative, normalization and support
equations are therefore not introduced. GSAS-II's powder oracle does not cover
this workflow. Exact independent Python prior eligibility and native policy tests
cover the new boundaries; existing numerical kernel/reference tests still apply.

## Frozen comparison

`benchmarks/unknown_group_corpus.py prepare --matched-priors` constructs native
priors **from the same translation-free SYMM/LATT instructions sent to SHELXT**.
It does not extract screw/glide translations from a reference group for native
input. The case selection, 128 MiB fitting memory, four P1 workers, fixed seed,
8-second fast/180-second auto budgets, cycle/model caps and primary grading
tolerances remain unchanged. Each case has three fast runs, one auto run,
one SHELXT-four run and three SHELXT-ten runs, collected sequentially after builds
and tests stop. The prior mismatch in the previous comparison is removed.
The existing diagnostic host opens unused MgSiO3 HDF5/PONI metadata as in the
first comparison; that initialization remains included in native process time.
No raw frame is read during these intensity-file searches.

The first report's inputs, predictions and receipts remain unchanged. This is a
combined regression test of the supplied-prior and acceptance changes, not an
isolated causal estimate of either change or an untouched holdout. Every new
prediction is saved and hashed before reference grading. Additional unlabelled
position grading remains a separately labelled diagnostic, with the primary
element-labelled distance/count criteria unchanged.

Full data and numerical products stay outside Git under
`/Users/clemens/ChatGPT/PhaseSmith/matched-prior-symmetry-0.8-20261101/`.

## Additional regression controls and checks

Normal MgSiO3 searches, with no symmetry prior and with supplied mmm/P,
continue to accept Pbca on the first pass. Warm process measurements are
0.499–0.506 s without a prior and 0.498–0.504 s with one. All six saved normal
controls have the same candidate CIF bytes, candidate-density arrays and shared
phase-map arrays as the validated v40 run: R1 0.0514043045582282, Mg16 Si16 O48.
The first execution at the newly copied binary path took 0.937 s and is retained
separately; a written diagnostic plan precedes the additional warm repetitions.

Two separate controls deliberately restrict only the fast total/per-candidate
budget to 1 microsecond, leaving data, priors, acceptance, full-resolution
controls and the original 180-second shared budget unchanged:

| Control | Actual outcome | Forced diagnostic time |
| --- | --- | ---: |
| MgSiO3 | Fast has no model; full four-by-800 pass accepts Pbca, correct Mg16 Si16 O48; maximum position error 0.00612 Å | 18.80 s |
| COD 2021040, acentric | Fast has no model; full four-by-800 pass accepts Pna21, correct labelled structure; maximum position error 0.01387 Å | 25.51 s |

These verify accepted deeper upgrades with the current guards, including a
non-MgSiO3 acentric case. Both preserve the failed first pass and report
`selection_reason="accepted_pass"`. They are functional tests, excluded from
normal case timings and recovery totals. Cooperative initialization outlasts
the deliberately tiny fast limit by about 11–13 ms; that elapsed time is charged
to the original total deadline, rather than granting a new full budget.

Validation passes: 548 workspace Rust tests with all features, four native
example tests, 3,318 installed-wheel Python tests (11 external-data skips and
34 normal deselections), strict all-target/all-feature Clippy, formatting/Ruff,
Rust 1.89 workflow checking, default/minimal facade checks, public API/version
and reference-document checks. Eight native/independent Python prior comparisons
include P, I, R and F lattices, both centric and acentric types, while exact
native prior validation roundtrips all 230 standard groups. The strict
documentation build also passes.

Frozen records are `validation/matched-prior-search-policy-v1.json`,
`validation/unknown-group-effort-corpus-v2.json`, and
`validation/results/matched-prior-search-v1-macos.json`. All 979 normal prediction
fingerprints remain unchanged through grading, as do the previous comparison's
protocol and receipt. The executed source snapshot, native binary, installed
wheel, input audit, complete models, unsuccessful passes and logs are retained
externally. After the freeze, Rust documentation comments clarify `prior_excluded`
and the prior-selected processing index; decision/numerical source is unchanged.
No package dependency or profile/structure-factor kernel is added or changed.
