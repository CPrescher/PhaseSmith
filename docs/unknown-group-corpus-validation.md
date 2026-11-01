# Cross-structure fast/automatic search validation

This is a fixed, exposed-data regression comparison of the current v40
fast/automatic policy, rather than a new holdout or a tuned solver benchmark.
The numerical executable is unchanged from the
[MgSiO3 adaptive-effort validation](space-group-search-speed.md).

**The fast path generalizes to some other atomic structures, but this test does
not validate general automatic space-group recovery.** Natural fallback runs
correctly and improves group recovery, while the current acceptance/selection
policy still has material failures. The MgSiO3 speed advantage does not
generalize to this sample.

The [matched-prior follow-up](matched-prior-search-validation.md) preserves these
first-run results and repeats the same cases with equal Laue/centring priors,
new symmetry/acceptance guards and retained unsuccessful fallback candidates.

## Results

There are 112 timed executions, taking 310.81 seconds altogether. No execution
hits the hard timeout. For repeated variants, calculate each case's median
first, then the median across all 14 cases, including unsuccessful results.
Auto and the four-worker SHELXT control have one run per case.

| Execution | Median seconds | Labelled atomic structure | Published group number | Both structure and group |
| --- | ---: | ---: | ---: | ---: |
| PhaseSmith fast, four P1 workers | 1.016 | 8/14 | 3/14 | 2/14 |
| PhaseSmith auto, four P1 workers | 9.584 | 8/14 | 5/14 | 4/14 |
| SHELXT, four workers | 0.484 | 9/14 | 12/14 | 8/14 |
| SHELXT, ten workers | 0.560 | 8/14 | 12/14 | 8/14 |

The frozen primary criterion requires the complete element-labelled structure.
As an explicitly **post-hoc diagnostic**, ignoring element identities while
retaining the same site-count and distance criteria gives atomic-position
recovery of 9/14 fast, 9/14 auto, 11/14 SHELXT-four and 10/14 SHELXT-ten.
Thus some primary failures concern element assignment. This secondary check
changes no model, primary result or solver setting. It was specified in a saved
diagnostic plan before that additional scoring.

In the table below, **A** means the saved model passes the primary atomic
criterion, **F** means it fails, and **—** means no model was saved. The following
number is the selected space-group type, **not** a claim of agreement. A correct
expanded structure in a subgroup remains a group-recovery failure here.
SHELXT-ten's atomic pass/fail and group number are stable over all three repeats;
native fast's saved full-cell CIFs are byte-identical across all repeats that
have models. The two no-model fast outcomes repeat consistently too.

| COD | System | Reference group | Fast seconds / result | Auto seconds / result | SHELXT-ten seconds / result |
| --- | --- | ---: | --- | --- | --- |
| 2200381 | cubic | 221 | 1.364 / — | 11.101 / A, #207 | 1.466 / A, #221 |
| 2215687 | cubic | 221 | 0.129 / F, #10 | 0.178 / F, #10 | 0.450 / F, #221 |
| 2243618 | hexagonal | 194 | 0.567 / F, #11 | 15.707 / F, #174 | 0.883 / A, #194 |
| 2017324 | hexagonal | 194 | 0.180 / A, #11 | 0.185 / A, #11 | 0.449 / A, #194 |
| 1549038 | monoclinic | 4 | 3.728 / A, #4 | 25.760 / A, #4 | 0.458 / F, #4 |
| 2010875 | monoclinic | 4 | 5.412 / F, #4 | 52.082 / F, #4 | 0.668 / F, #4 |
| 2241300 | orthorhombic | 56 | 0.669 / A, #13 | 0.672 / A, #13 | 0.662 / A, #56 |
| 2021040 | orthorhombic | 33 | 1.645 / A, #7 | 27.558 / A, #33 | 0.884 / F, #33 |
| 2207083 | tetragonal | 139 | 0.127 / A, #12 | 0.126 / A, #12 | 0.239 / F, #141 |
| 2233659 | tetragonal | 139 | 0.394 / A, #12 | 7.851 / F, #2 | 0.450 / A, #139 |
| 2011241 | triclinic | 2 | 1.488 / F, #1 | 8.066 / F, #1 | 0.448 / A, #2 |
| 2010993 | triclinic | 2 | 0.343 / A, #2 | 0.343 / A, #2 | 0.235 / A, #2 |
| 2203173 | trigonal | 167 | 1.645 / A, #148 | 37.993 / A, #167 | 0.668 / F, #155 |
| 2201295 | trigonal | 152 | 2.675 / — | 24.589 / — | 0.874 / A, #152 |

## What fallback does and where it fails

Auto's first-pass acceptance agrees with standalone fast in all 14 cases.
It accepts five fast candidates and naturally escalates in the other nine.
Both pass archives survive. These are actual data-driven escalations, additional
to the earlier forced-timeout/forced-rejection MgSiO3 tests.

- **Useful accepted fallback:** COD 2021040 moves from a correct expanded
  structure in Pc (#7), own R1 0.06052, to the correct Pna21 (#33), R1 0.05769.
  The latter passes acceptance and has maximum position error 0.01387 Å.
- **Useful unaccepted fallback:** COD 2203173 moves from R-3 (#148) to the
  correct R-3c (#167); maximum position error improves from 0.01698 to 0.00479 Å.
  Own R1 remains about 0.122, so it is honestly retained as unaccepted.
  COD 2200381 also gains a correct expanded structure, but selects P432 (#207)
  instead of Pm-3m (#221), and remains unaccepted.
- **Premature acceptance:** COD 2215687 has only 50 rows and no measured Laue
  constraints. Fast accepts P2/m (#10), common R1 0.03614, with Pr3 In1 O2;
  the reference has Pr3 In1 O1 in Pm-3m. Auto consequently never escalates.
  SHELXT finds the published group but misses O, so it too fails complete
  labelled recovery on this case. Low R1 establishes neither chemistry nor
  the correct space group.
- **Subgroup exits:** COD 2017324, 2241300 and 2207083 recover the labelled
  expanded structures quickly but accept #11/#13/#12 instead of #194/#56/#139.
  Their higher-symmetry groups remain unresolved, despite good coordinates.
- **Worse selected approximation:** COD 2233659's fast model has correct
  labelled positions, maximum error 0.00391 Å, but fails the numerical gate.
  Auto selects a P-1 approximation with maximum error 0.14401 Å because its
  common R1 (0.103 versus 0.227) and BIC-like score are better. Neither pass
  is accepted and the earlier model is retained in the other-pass archive.
  A better objective has not improved structural truth in this example.

Eight cases lack enough measured constraints for their published nontrivial
Laue class; seven have zero and one has nine, below the frozen minimum of 20.
SHELXT's supplied Laue/centring prior is therefore a substantial advantage on
these archived inputs. This explains part of the gap, without establishing
that it explains every failure. For example, COD 2201295 has 523 supported
Laue constraints, but the full P1 map gives the correct group's best phase alpha
about 0.710, above 0.3, so that group is screened out. Its failure also requires
better phasing/model initialization, not just a supplied symmetry prior.

## Where the time goes and next work

Across the 14 auto executions, saved native counters account for about 6.17 s
of shared P1 map work, 15.17 s of origin testing, and 189.77 s of model search
including assignment, fitting and any independent phasing/restarts. The complete
search coordinator after assessment totals 211.15 s; measured symmetry assessment,
process/output/initialization bring that to 212.21 s. These counters identify **candidate model work**, rather than the
already short shared-map stage, as the next performance target on this sample.

The evidence supports this order of work:

1. Support explicit Laue/centring priors for archived intensity input and compare
   both solvers under that matched contract; preserve the separate unmerged
   symmetry-inference route and truthful insufficient-evidence states.
2. Check higher symmetry of a recovered atomic model before a fast subgroup
   exit, and make unresolved symmetry/chemistry visible in acceptance. Do not
   equate a target R1 with solved structure, or silently force a group from the
   metric alone.
3. Improve failed-map recovery and bounded candidate assignment/fitting, while
   preserving both approximate passes and measuring whether selection improves
   physical recovery. Use new validation cases before claiming unseen-data gain.

No numerical policy was changed or tuned on these outcomes. The report records
the current limitation; it does not claim those next steps are implemented.

## Inputs and comparison contract

Select two measured COD cases per crystal system from the existing 100-case
corpus. Within each system, order by `SHA256("unknown-effort-v1:" + id)`;
take the first, then the first from the other light/heavy tier if available,
otherwise the next case. This selects 14 cases, four light and ten heavy,
with 50–2,652 supplied intensity rows. Selection never reads earlier solver
outcomes. These cases were already exposed by earlier validation.

Both programs receive the same fixed-width HKLF4 values, including their
uncertainties and any negative intensities. Native JSON is parsed from that
very HKL file, so there is no precision advantage. Cell and non-H element
types are supplied; correct atom counts and coordinates are withheld.

SHELXT additionally receives the **published Laue rotations and lattice
centring**, expressed with translation-free SYMM operations, a positive LATT,
neutral TITLE and dummy `UNIT 1 ... 1`. All screw/glide translations are removed;
there is no exact-group `-s` restriction. PhaseSmith instead infers intensity
symmetry from its supplied observations and admits all metric-compatible
catalogue settings. These are equal numerical intensities with **unequal
symmetry priors**, not an equal-work speed comparison. The comparator contract
follows the Laue-group input described in
[Sheldrick (2015)](https://doi.org/10.1107/S2053273314026370).

The archived published HKL sets can have very few remaining symmetry-equivalent
observations. They are not equivalent to fresh unmerged raw-frame reductions;
missing equivalents provide no measured Laue evidence. No missing reflection
is converted to a measured zero, and no symmetry-expanded pseudo-observations
are added to supply the missing evidence.

The native controls are unchanged: four independent P1 workers, four starts,
256 cycles and an inclusive 0.8 Å initial cutoff on fast; at most four models,
two seconds per model and eight seconds for that pass. Auto retains the shared
180-second total budget and, on rejection, uses full-resolution four-by-800
phasing, up to 64 phase settings and 12 models. Dense fitting memory is 128 MiB.
No threshold, seed, candidate cap or chemistry is adjusted after an outcome.

Run fast three times, auto once, SHELXT with four workers once and with ten
workers three times for every case, sequentially. Times include process startup,
symmetry assessment, unsuccessful work and saved products. No task-owned build,
test or profiler overlaps collection. Native uses the existing diagnostic host;
its unused MgSiO3 HDF5/PONI metadata initialization is included in process wall
time, but it reads no raw frame and uses only the supplied case observations
for solution. This is a validation harness, not a new generic HKL CLI.

The machine is an Apple M4 Pro, 14 logical CPUs, 48 GiB RAM, macOS 26.6.2
(25G83). Native is arm64; separately installed SHELXT 2018/2 is x86_64 under
Rosetta. Three repeats describe observed variation, not confidence intervals
or cold-cache performance.

## Independent grading

Save all predictions before opening reference structures for grading. Use the
existing independent element-labelled assignment auditor without changing its
tolerances: exact non-H element counts, RMS coordinate error <=0.025 Å and
maximum error <=0.05 Å after a common origin shift and optional inversion.
The secondary strict criterion is maximum error <=0.02 Å. Space-group number
agreement is reported separately; a lower-symmetry model can have the correct
expanded coordinates without establishing the correct group. Changed cell
settings remain explicitly unassessed by this fixed-basis auditor.

Native numerical acceptance is a separate observation from reference recovery.
Do not grade success by low R1 alone. SHELXT's own R1 and the native mixed/full
data fit are different objectives and should not be compared as interchangeable
quality scores. Only each program's saved top model is scored; later alternatives
are not selected using the reference.

## Reproduction and artifacts

The harness is `benchmarks/unknown_group_corpus.py`, with `prepare`, `run` and
`grade` commands. Prepare requires the pinned corpus repository, source manifest,
v40 diagnostic packet template, native binary, null-rotation JSON and installed
SHELXT path. Run and grade take the newly created external output directory.
The frozen protocol records source, input and executable hashes and every case.
Source CIFs/HKLs, complete predictions, logs and raw grading products remain at
`/Users/clemens/ChatGPT/PhaseSmith/cross-structure-effort-0.8-20261101/`.
They are not runtime dependencies or repository additions.

The compact, versioned records are
`validation/unknown-group-effort-corpus-v1.json` and
`validation/results/unknown-group-effort-corpus-v1-macos.json`. A saved prediction
barrier hashes 975 artifacts before reference grading; all remain unchanged.
Every source/input fingerprint and the numerical binary fingerprint are retained.
The current benchmark harness additionally preserves copied executable permissions,
reads the actual saved CIF cell during grading and labels no-model outcomes
explicitly; these changes do not rerun or alter a numerical prediction. The
exact harness used for freezing/launching remains in the external archive.

Validation: 42 targeted Python tests pass, including four comparisons with the
native symmetry-assessment command and comparator-input symmetry checks. Ruff and the strict documentation build
pass. Rust numerical sources and package dependencies are unchanged.
