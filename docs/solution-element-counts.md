# Bounded element-only count experiment

`benchmarks/solution_element_counts.py` accepts corrected measured CIF HKL loops,
a cell, an exact space group and neutral non-H element types. It does not accept
atom counts or reference coordinates. Python orchestrates repeated calls to the
existing count-constrained Rust solver; no numerical equations or native kernels
were changed. This is a bounded experiment, not general unknown-composition
solution or full SHELXT equivalence.

The experiment expands trial points whose coordinates are eighths, thirds or
0.137/0.271 using the supplied operations. The distinct orbit multiplicities are
count choices for each required element. Sampling is not an exhaustive Wyckoff
table for arbitrary space-group settings, and sums of several orbit sizes are
not explicitly generated. Every supplied element must be present. Counts and
coordinates remain free to differ between hypotheses; the native output need
not contain exactly one asymmetric site per element.

A declared volume bound limits the total count:
`min(128, ceil(cell_volume / minimum_volume_per_atom))`. The default lower volume
is 8 angstrom cubed per atom. This is an explicit search limit, not a physical
density estimate or an inferred chemical constraint. Proposals are ordered by
total atom count, then lexicographic counts with elements in decreasing atomic
number. Up to 512 proposals and six element types are allowed.

Each hypothesis uses one dual-space start, 1200 maximum cycles, completion and
constrained anisotropic refinement. The default time cap is 12 seconds per
hypothesis, with four concurrent workers. Caps can interrupt a hypothesis before
refinement and make boundary outcomes machine/load dependent. No early stop is
made when a count hypothesis meets the amplitude target. All outcomes are saved.

The final ranking accepts only candidates refined to the supplied symmetry with
no full-cell atom pair closer than 0.5 angstrom. The periodic cutoff check uses
reciprocal-metric image bounds; equality at the cutoff is accepted. Candidates
are ranked by `N log(J) + k log(N)`, with `N` the independent merged family count,
`J` the existing normalized amplitude least-squares objective and `k` the number
of coordinate/ADP parameters plus profiled amplitude scale. This is a descriptive
complexity penalty, not a calibrated BIC, likelihood, composition probability or
proof of correctness. Stronger/wrong-count candidates that fail symmetry remain
visible, but are not ranked.

The command saves the input element list, all proposed counts and outcomes,
implementation/native hashes, and a snapshot of its own source. Reference
comparison is deliberately a separate, subsequent operation after the full
ranked report has been saved. Source formulas and reference coordinates must
never be used to select a count proposal, tune its settings or select the winner.

## Reproduce the KBrF4 development-data experiment

```sh
.venv/bin/python benchmarks/solution_element_counts.py \
  --hkl validation/data/structure-solution/cod-1549359/1549359.hkl \
  --cell 6.0999 6.0999 11.0509 90 90 90 \
  --space-group I4/mcm --elements Br F K \
  --output runs/element-only-count-search
```

The existing pinned reflection-loop file is translated without modifying signed
measured F-squared or sigma values. The source structure is reused development
data, so this demonstrates a capability on this case rather than independent
general validation. The known cell and exact space group remain substantial
priors; only the composition counts have been removed from the supplied inputs.

## Recorded result, 28 October 2026

The supplied element list was `Br F K`. Sampled I4/mcm orbit sizes were
4, 8, 16 and 32. The upward-rounded volume cap was 52 atoms, producing 45 count
hypotheses. The selected hypothesis was generated at index 7: Br4 K4 F16,
equivalent to four K[BrF4] formula units. It was the only candidate with successful
supplied-symmetry refinement and no overlap. Its R1 was 0.02226640044 with 12
coordinate/ADP/scale parameters and 214 independent reflection families.

The final pass took 116.21 seconds with four workers. Fourteen hypotheses used
their one-start trial budget, thirty reached the 12-second time cap, and the
selected hypothesis met its R1 target. All 45 were attempted; the capped trials
are not exhaustive searches of their respective compositions. An initial pass
selected the same counts and identical R1. This does not prove unique composition
or exclude solutions requiring other starts, multiple orbit counts or more time.

Only after saving the entire ranking was the deposited structural CIF opened.
Equal counts were required before one-to-one element matching. All 24 atoms
matched within 0.005909 angstrom (RMS 0.004825 angstrom), allowing origin/hand
alignment. Both the P1 full-cell CIF and supplied-symmetry asymmetric-unit CIF
reproduced the native R1 through the separate general structure-factor path
within 7e-13. The frozen report is
`benchmarks/structure-solution-element-only-macos.json`; full outputs and the
post-solve audit remain in `runs/element-only-count-search-final/`.

Verification: 1246 Python tests passed (11 skipped, 34 deselected), 387 Rust tests
passed (34 ignored), and strict workspace clippy, cargo formatting and selected
Python lint checks passed. No native numerical kernel changed; the experiment
itself records realistic end-to-end timings.
