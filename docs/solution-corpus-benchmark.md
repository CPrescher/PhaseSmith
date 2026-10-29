# Experimental structure-solution corpus and first independent comparison

The numerical structure solver is frozen at commit `8bc4efc`. This change builds
an experimental-data benchmark and adapters; it does not tune the search against
the new structures. The original K[BrF4], BiSeI and thiourea cases remain previously
exposed development evidence and are excluded from this new panel.

## Data and split

`validation/solution-corpus-v5.json` registers 100 experimental reflection/reference
pairs from 100 separate DOI/formula families. A deterministic space-group-stratified
selection uses COD metadata and source eligibility, before any solver outcome.
The inventory preserves rejected candidates and reasons. The sources are COD's
[documented API](https://wiki.crystallography.net/RESTful_API/), individual coordinate
and reflection files, and its Granada mirror when the main server refuses a
connection. COD data are [dedicated to CC0](https://www.crystallography.net/);
original authors, titles and DOIs remain in every record. Of 260 inspected
candidates, 160 exclusions remain in the inventory with their reasons.

| Crystal system | Datasets |
| --- | ---: |
| Triclinic | 18 |
| Monoclinic | 29 |
| Orthorhombic | 17 |
| Tetragonal | 16 |
| Trigonal | 8 |
| Hexagonal | 6 |
| Cubic | 6 |

There are 27 space-group types, 39 light-element and 61 heavier-element cases;
66 contain hydrogen, which is excluded from both solvers' required element lists
and recovery scoring. There are 4–112 non-H atoms per full cell and 50–7,459
measured observations per source. This is a first structured panel, not an unbiased
sample of COD or complete coverage of crystallographic difficulties. Initial
metadata queries emphasize 2000, 2015 and 2018, supplemented by targeted symmetry
queries. Full discovery snapshots and their checksums are retained locally.

DOI/formula connected components define families. A fixed SHA-256 split assigns
**35 development cases and 65 held-out cases**. No related family crosses the
split. Only development cases are evaluated in the first comparison. Held-out
solver outcomes remain unopened. Curation reads coordinate CIFs for eligibility;
the numerical workers receive only separate input packets, never coordinates,
formula quantities, expected counts, reference paths or refinement models.

The clean initial tier requires measured single-crystal X-ray intensity/F-squared
observations with positive sigmas, an unambiguous wavelength, full non-H occupancy,
known reference-setting symmetry and at most eight non-H elements / 128 non-H
atoms. Explicit disorder/twinning, non-reference settings, non-X-ray/powder data,
amplitude-only or calculated-only files and unusable observations are logged as
exclusions. No Fcalc is substituted for measured data and no bad-sigma rows are
silently dropped. These source/format checks do not independently prove every
published model is physically correct.

## Reproducibility and input translation

Raw files are unchanged. Every raw/reference and prepared input has a SHA-256.
`fetch` retrieves the pinned raw files and reconstructs only byte-identical
prepared inputs; changed upstream data fail verification instead of silently
regenerating a benchmark. The explicit seal rechecks all 100 sources, operations,
occupancies, input packets, splits and common input hashes. Existing manifests,
locks and output directories cannot silently be overwritten.

Both programs consume the same fixed-width HKLF4 text. Numeric fields always
contain an explicit decimal point: otherwise Fortran F8.2 input can apply implicit
scaling. A bounded formatter retains as much precision as eight columns allow,
with at most 5e-4 relative change to either intensity or sigma. Every case records
its actual quantization. Signs, row order, duplicates and negative measurements
are preserved. Both engines see the same round-tripped values, while raw source
measurements remain available separately.

Each worker validates an exact whitelist: ID, cell, wavelength, space-group name
and number, the external spelling of that group, and element types. The exact
reference-setting operations from PhaseSmith's numeric group API are checked
against Gemmi for **all 230 space-group types**. This avoids ambiguous extended
symbol syntax without changing the production solver or supplying atomic models.

SHELXT instructions use its correct `LATT` centering/inversion conventions and a
reduced SYMM set, tested to reconstruct all 230 reference groups. `UNIT 1 ... 1`
is a syntactically required dummy; no actual composition is supplied. `-s` fixes
the supplied group, and `-t1` plus environment limits fixes one thread. External
screw-axis spellings such as `P2(1)` are normalized only for output interpretation.

The installed user-provided SHELXT is pinned to 2018/2, SHA-256
`4d712cb17f8952e1f59f476af7c61ad4ab5261e594c75441c7f81dcefe7b5560`.
Its executable is not distributed. It runs through the host's x86_64 compatibility
runtime; PhaseSmith is native arm64. Timings are observations on this host, not an
instruction-set-matched comparison.

## Search and timing protocol

Jobs run serially with alternating program order and a **15-second hard wall
ceiling including startup**. PhaseSmith uses dual-space search, symmetry recycling
and constrained tangent refinement, seed 0, four starts, 800 cycles per start,
R1 target 0.06 and an internal 13-second search deadline, allowing time to save
results. SHELXT uses its normal search controls with one thread and fixed symmetry.
These are different internal search/refinement algorithms; iteration counts and
reported R1 values are not treated as equal work or equal objectives.

The atom ceiling is derived only from inputs:

`Nmax=min(128,ceil(V/8),floor((floor(2000000/M)-7)/5))`,

where V is the supplied cell volume and M is the number of observed allowed
symmetry/Friedel-expanded reflection rows. This follows the existing dense-model
memory bound `(5*Nmax+7)*M <= 2,000,000`. The preparation is outside both timed
jobs. No reference atom count enters this bound. After solving, cases whose
reference exceeds it are explicitly identified; they remain in the total
coverage denominator. The experiment therefore exposes this production resource
limit rather than hiding it or using known composition to configure the search.

Every stdout, input, chosen model, early handoff, termination and timeout is saved.
A hard timeout terminates the job's process group. A nonzero exit or an external
return code of zero without a usable model is retained as a failure. There are no
success-only timing averages. Timing measures the model available at stop, not
the first instant at which a correct intermediate candidate may have existed.

The first collection resumed after an adapter rejected an explicit origin-choice
suffix. Completed jobs were reused unchanged; the remainder use the equivalent
numeric reference-setting API. Source snapshots and collection-stage IDs are
retained, and resume checks prohibit changes to the numerical solver, executable,
inputs, budget or split. Output-format corrections operate on the already chosen
`sample_a.res` file, never the reference-best alternative. No numerical search
changes or reference-informed reruns are part of this campaign.

## Independent success criteria

All predictions are saved before the post-solve reference audit. The external
program's own best `_a` model and PhaseSmith's own best candidate are assessed;
reference information never chooses among hypotheses.

Success requires exact full-cell non-H element counts and an element-preserving
coordinate match, allowing periodic images, translation and inversion. Maximum
error must be at most 0.05 angstrom and RMS at most 0.025 angstrom. A separate
0.02-angstrom strict result is also reported. These broad-panel thresholds were
fixed before outcomes and do not replace earlier three-case tests. Output cells
that change setting are explicitly marked unassessed rather than compared in an
incorrect basis. Absolute structure is outside this nonresonant benchmark.

The independent assignment uses Gemmi periodic distances and SciPy's Hungarian
algorithm. A common optional score evaluates both full-cell models with the same
native nonresonant scattering, fixed Uiso=0.01 angstrom squared and independently
fitted scales. It includes a signed-intensity weighted normalized SSE and an
amplitude R1. This is a common geometry/label score, **not** a common refinement,
a published R1, or a substitute for coordinate/count validation.

## Reference precision audit

The prediction collection used registry v4. Registry v5 changes reference-count
interpretation only, retaining a SHA-256 link to v4; all IDs, splits, raw files and
solver input hashes are identical. Nine deposited models contain rounded
special-position coordinates such as 0.3333/0.6667. Expanding them with a 1e-7
fractional duplicate threshold falsely counted coincident symmetry images as
separate atoms. The reference adapter now merges images of the **same deposited
site** within 5e-4 fractional distance. All 100 expanded per-element counts then
agree with the separately registered COD cell formula; the manifest records that
check for every case and names all nine corrections. No source coordinate is
modified or different deposited sites merged.

SHELXT's five-decimal coordinate output uses a separate 1e-4 fractional image
threshold, and its actual LATT/SYMM operators determine origin choice. Tests cover
rounded special positions, special-site occupancy and ambiguous group labels.
These are file-precision corrections; the coordinate recovery gates above remain
unchanged. Re-audit uses the saved chosen models and original timings, without
executing either solver. Earlier collection/audit records and source snapshots
remain available locally; the committed result is the final v5 reference audit.

## Development results

[Machine-readable results](../benchmarks/structure-solution-corpus-development-macos.json)
retain all 70 timed jobs, per-case models, failures, limits, source hashes and
collection/audit implementation hashes. None of the 65 held-out outcomes was run.

| Measure (35 development cases, seed 0) | PhaseSmith | SHELXT 2018/2 |
| --- | ---: | ---: |
| Exact non-H counts/labels and coordinate recovery | 5/35 | 13/35 |
| Stricter maximum coordinate error ≤0.02 Å | 5/35 | 10/35 |
| Correct total non-H atom count, ignoring labels/coordinates | 9/35 | 29/35 |
| Median wall time, all outcomes | 13.220 s | 0.889 s |
| 95th-percentile wall time, all outcomes | 13.263 s | 2.014 s |
| Hard timeouts | 0 | 1 |

Three cases are recovered by both, two only by PhaseSmith, ten only by SHELXT,
and twenty by neither under the full criterion. PhaseSmith returns wrong
counts/labels for 28, one coordinate mismatch and one no-model outcome. SHELXT
returns wrong counts/labels for 20, one coordinate mismatch and one no-model
outcome. The exact-label gate is demanding when only element types are supplied:
SHELXT has the right total count but wrong element counts in 15 cases; PhaseSmith
has that outcome in three. A model file or a low R1 is not counted as recovery.

Seven references exceed PhaseSmith's input-derived dense-model capacity. Among
the remaining 28, recovery is 5/28 versus 11/28. All seven remain in the primary
35-case denominator. Both the capacity boundary and the mismatch between recovered
counts, element assignments and geometry are now concrete development targets.

This establishes **no general parity with SHELXT**. These are fixed-budget,
element-only, known-symmetry results on one deliberately restricted panel and one
host, with different internal algorithms and executable architectures. The ratio
of all-outcome median runtimes is about 14.9; it is not a ratio of matched
successful solve times. Repeating seeds or changing budgets would be a new,
explicitly labelled development experiment.

The next work should first reduce whole-orbit count-selection failures and the
dense reflection/parameter memory constraint, then improve element assignment
using likelihood/refinement evidence. Each change should be compared on these
same 35 development cases with failures retained; only a frozen improved version
should open the 65-case holdout. This panel is validation and algorithm-development
data, not machine-learning training data. No learned model is introduced.

## Commands

Use a release build with the `experimental-solution` feature (see
[build instructions](experimental-structure-solution.md)), Gemmi 0.7.5 and the
optional SciPy refinement dependency. The core package still imports without
these benchmark-only tools.

```sh
# Verify or retrieve immutable inputs; never silently regenerate changed sources.
.venv/bin/python benchmarks/solution_corpus_benchmark.py fetch

# Reproduce the declared development screen (choose a new output directory).
.venv/bin/python benchmarks/solution_corpus_benchmark.py run \
  --manifest validation/solution-corpus-v5.json \
  --output runs/solution-corpus-development-new --seconds 15

# Resume a collection only with identical numeric solver/input/budget settings.
.venv/bin/python benchmarks/solution_corpus_benchmark.py run \
  --manifest validation/solution-corpus-v5.json \
  --output runs/solution-corpus-development-new --seconds 15 --resume
```

The committed result was re-audited without rerunning searches:

```sh
.venv/bin/python benchmarks/solution_corpus_reaudit.py \
  --manifest validation/solution-corpus-v5.json \
  --run runs/solution-corpus-v4-development \
  --output runs/solution-corpus-v4-development/comparison-new-audit.json
```

That command requires the locally retained original collection. A fresh clone
can retrieve inputs with `fetch` and perform a new timed collection with `run`;
it cannot reconstruct historical wall times. The historical collection used v4
locks 2/3; the current adapters and corrected reference registry are sealed by
`validation/solution-corpus-v5.benchmark4.lock.json`. The later count-policy CLI
is sealed separately by `validation/solution-corpus-v5.benchmark5.lock.json`,
which current commands use by default. Raw data, bulky solver output
and discovery snapshots are kept in ignored data/run directories; the committed
manifests and result retain their hashes and source URLs. The user-supplied pinned
SHELXT executable is required separately.

The holdout path additionally requires `--split holdout --open-holdout`; its
protocol records the opening before any solve. A holdout run is a final evaluation
of a frozen version, not material for further tuning while calling it held out.
The original three cases and the 35 new development cases can support subsequent
failure-driven work. No held-out cases should be moved after outcomes are known.

Validation: 21 corpus adapter/protocol tests cover input fidelity, split leakage,
timeouts, reference matching, special-position precision and all 230 reference
groups. The complete Rust suite passes 394 tests; formatting and strict Clippy
checks pass. The full Python suite and repository lint checks also pass (external
data/oracle tests retain their normal opt-in skips).

The subsequent count-selection investigation and same-build ablation are in
[the count-selection follow-up](solution-count-selection.md). The results above
remain the original frozen screen; they are not silently replaced by later runs.
