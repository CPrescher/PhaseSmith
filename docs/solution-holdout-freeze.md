# Frozen structure-solution validation protocol

The numerical solver is fixed at commit `870d61a` with its already tested release
extension. Freezing means committing the implementation and the complete test
protocol, then retaining hashes of sources, binaries, inputs and controls.
The protocol commit can add validation tooling/documentation without changing
the solver. No numerical rebuild, option selection or tuning is permitted during
this comparison.

## Algorithm attribution

The algorithm sources are published crystallographic methods, including
G. M. Sheldrick (2015), *Acta Crystallographica* A71, 3–8,
[doi:10.1107/S2053273314026370](https://doi.org/10.1107/S2053273314026370),
for Patterson-seeded dual-space phasing, phase-based symmetry/origin reasoning
and density interpretation, and G. M. Sheldrick (1990), *Acta Crystallographica*
A46, 467–473,
[doi:10.1107/S0108767390000277](https://doi.org/10.1107/S0108767390000277),
for phase annealing and negative-quartet consistency. The independent equations,
NumPy references and explicit PhaseSmith choices are documented in the individual
method notes. Search scheduling, bounded count/label repairs, numerical caches
and their constants include PhaseSmith-specific choices; the papers are not
claimed to specify the complete implementation.

SHELXT 2018/2 is solely the separately supplied external test comparator. Its
executable, selected outputs, runtime and correctness are recorded for testing;
it is neither redistributed nor a runtime dependency of PhaseSmith.

## Predeclared comparison

- All 65 cases from the immutable family-separated holdout; no exclusions or
  case-specific retuning based on solver outcomes.
- Exact best configuration from the cached-repair development panel, including
  both repair flags; one PhaseSmith worker and one comparator thread.
- Same measured, quantized HKLF4 observations, element types, cell, wavelength
  and supplied reference-setting symmetry for both programs. Composition counts
  and coordinates remain inaccessible to numerical workers.
- Seed 0, four starts, 800 cycles, final anisotropic refinement, completion off,
  a 13-second native safeguard and a 15-second process ceiling. The original
  input-derived atom/memory caps and all stopping/acceptance rules remain fixed.
- Serial jobs alternate which program runs first in each case. No task-owned
  test, build, profiling or kernel jobs overlap timed collection.
- All 130 chosen predictions must be saved before any reference grading. Every
  yield rechecks committed source and binary hashes. No fallback runs, alternative
  seeds, extra trials or reference-selected models are permitted.
- Primary structure correctness requires exact full-cell non-H element counts
  and periodic element-preserving coordinate recovery: maximum error 0.05 A,
  RMS error 0.025 A, allowing common origin shift/inversion. Strict 0.02 A
  recovery is secondary. Total counts and full element-count vectors are separate
  outcomes; residual fit alone never certifies a correct structure.
- Report all-outcome median, p95 and summed process wall time, paired timing
  ratios, timeouts, returned models, full structure and count accuracy. The
  common-success subset is secondary and explicitly identified. Above-cap,
  failed and unassessed outcomes remain in all 65-case primary denominators.
- Use the already tested reference/output auditor unchanged. Adapter errors are
  recorded, not corrected by changing predictions or rerunning a numerical search.

`validation/solution-holdout-freeze-v1.json` pins the exact configuration,
source/binary/data hashes, prior development evidence, all 65 IDs, algorithm
citations and the external comparator. The protocol must be committed and the
working tree clean before `--open-holdout` starts collection. The user explicitly
authorized opening this holdout for this frozen comparison. The earlier immutable
corpus locks record their historical unopened status; the new run records the
opening without rewriting those locks.

```sh
.venv/bin/python benchmarks/solution_holdout_frozen.py --open-holdout --output runs/holdout-frozen-v1
.venv/bin/python benchmarks/solution_corpus_reaudit.py --manifest validation/solution-corpus-v5.json --run runs/holdout-frozen-v1 --output runs/holdout-frozen-v1/comparison-verified.json
.venv/bin/python benchmarks/solution_holdout_report.py --run runs/holdout-frozen-v1
```

The two-stage existing auditor's initial grading and normalized post-solve
grading both occur after the prediction barrier; no solver is run by the
normalization pass. This is one fixed-seed, fixed-budget transfer test, not a
complete search-capability or instruction-set-matched speed claim. After opening,
the 65 cases become exposed validation evidence. Future tuning requires a new
untouched test set.
