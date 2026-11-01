# Fast, automatic and thorough structure search

The MgSiO3 example now defaults to **automatic effort**: first try the short
coarse-grid path, then use finer phasing and a broader search only if the saved
candidate fails explicit acceptance. Measured standalone automatic searches take
**0.50–0.51 s warm**, including measured symmetry assessment and output files. The
same-machine SHELXT controls take **0.64–0.67 s with ten workers**. Both find Pbca;
PhaseSmith recovers the labelled Mg16 Si16 O48 structure.

These are exposed development/regression observations on one Apple M4 Pro,
rather than general speed or reliability claims. SHELXT runs x86_64 through
Rosetta; PhaseSmith is native arm64. Their phase processing, objectives and
hypothesis coverage differ. The complete original-frame workflow still takes
**17.36–17.85 s** with the independent observer, or **16.49 s** for the directly
launched native command in the final replay. Raw-frame reduction is additional
work compared with SHELXT's intensity-file input.

## Choose effort

Use the [native workflow guide](single-crystal-native-workflow.md) for building
and source-data registration. With frozen protocol v40:

```sh
target/release/examples/single_crystal_workflow --mgsio3 \
  '/path/to/MgSiO3 Enstatite Calibration Crystal' \
  validation/enstatite-raw-workflow-v40.json \
  /path/outside/the/repository/new-run --effort auto
```

| Option | Execution |
| --- | --- |
| `fast` | Coarse first attempt only; retain rejection/failure without a deeper attempt |
| `auto` | Try fast; run the configured full-resolution/broader pass only after acceptance fails |
| `thorough` | Start directly with the configured full-resolution/broader pass |

The optional Python benchmark observer accepts the same `--effort` choice. Rust
hosts use `DiscoveryOptions.effort`, `EffortOptions`, `FastSearchLimits` and
`SearchEffort`. Earlier protocols without effort controls retain their frozen
execution. This adds a native coordinator/CLI option, not a Python or Dioptas
discovery frontend.

The v40 fast pass uses genuine measured reflections with initial d >= 0.8 Å,
four independent starts of at most 256 cycles, up to four shared-map model
settings, two seconds per setting and eight seconds overall. It has no independent
phasing restart. All original signed measurements remain available for final
full-resolution fitting, common-model scoring and absence evidence.

Acceptance requires a complete model meeting the existing own-group numerical
target (R1 <=0.06), an accepted common-set symmetry model, common-data R1 <=0.08
and measured-map phase alpha <=0.3. These are explicit plausible-candidate
criteria, not proof of the correct chemistry, unique symmetry or a converged
standalone structural refinement. Force `thorough` when broader alternative
assessment is wanted even after a plausible fast result.

On rejection, automatic effort uses the configured full-resolution profile:
four starts of at most 800 cycles, up to 64 phase-tested settings and 12 atomic
settings, with independent restart available. Both passes share the original
remaining 180-second cooperative search budget. Cancellation and exhaustion of
that total budget do not start another pass. The example retains its 210-second
hard search-process deadline. Cooperative initialization can outlast an extremely
small per-pass budget; every elapsed interval remains charged and reported.

`space-group-search.json` records requested/selected effort, acceptance, actual
cycles, grids, model attempts and the escalation reason. When both passes run,
the complete other pass survives under `effort.other_pass`, including its failed
and unattempted settings. An available other-pass phase density is saved as
`space-group-other-pass-phase-map.npz`. The retained model and setting continue to
own the top-level products. If neither pass is accepted, any retained approximate
model remains explicitly unaccepted.

## Staged measurements

All controls were committed before their corresponding new replays. The
[frozen plan](space-group-speed-plan.md) records equations, parameter conventions,
boundaries, fallback and validation gates. Scientific reduction controls and
original source files remain unchanged. Times include unsuccessful work.

| Stage | Full observed command, s | Native search stage, s | Main finding |
| --- | ---: | ---: | --- |
| Previous v33 | 36.16–36.50 | 19.01–19.07 | Shared map, four independent workers |
| v34: cache density Fourier coefficients | 36.81 | 19.09 | Origin tests 2.38 s versus about 2.60 s; no demonstrated overall gain beyond host variation |
| v35: interleaved tests and plausible light-centric exit | 21.69 | 3.78 | Avoid later settings after a plausible model; internal map/model coordinator 2.38 s |
| v36: coarse phasing and existing quality handoffs | 19.76 | 2.07 | P1 0.322 s; final fitting retains full data |
| v37: prepared exact absences and common shell contributions | 18.58–19.01 | 1.57–1.59 | Identical absence decisions and evidence |
| v38: buffered JSON archives | 17.08 | 0.684 | Sampled profile identified per-token unbuffered writes |
| v39: deterministic 256-cycle portfolio handoff | 17.00–17.24 | about 0.48 | Avoid full-length work after a sufficiently good initial map |
| v40: explicit fast-first automatic effort, two runs | 17.36–17.85 | 0.486–0.493 | Accept first pass; no full-resolution phasing or broader model pass |

v39 preserves its independent quality-gated restart as an intermediate profile.
Current v40 tests the short map against an actual model before any deeper work.
Timing variation in reduction stages is appreciable; the table does not claim
that each smaller full-command difference is statistically established.

### Matched intensity-file comparison

Both programs receive the same 4,844 signed native measurements, including 594
negative values. SHELXT receives standard fixed-width HKLF4 with the previously
recorded rounding; PhaseSmith retains native precision. Cell and element types
are supplied; exact group, source HKLs, coordinates and correct atom counts are
withheld. SHELXT receives mmm inferred by the preceding native assessment, uses
dummy `UNIT 1 1 1` and a neutral TITLE. Native processing includes assessment of
all 345 metric-compatible settings and inferred intensity symmetry.

| Execution | Workers | Measured process seconds |
| --- | --- | --- |
| PhaseSmith automatic, final three warm runs | Four P1 workers; serial origin/model work | 0.510, 0.508, 0.503 |
| SHELXT default | One | 2.325, 2.309, 2.319 |
| SHELXT default | Four | 0.861, 0.889, 0.898 |
| SHELXT default | Ten | 0.661, 0.644, 0.668 |

These sequential warm-cache measurements ran after compilation/tests stopped.
New executable first-launch overhead is retained separately in intermediate
trials (final guarded binary 0.876 s first launch, then 0.510/0.508/0.503 s;
v39 0.913 s first launch before 0.500/0.490 s). No cold-cache confidence
interval or general equal-work speed ratio follows. SHELXT's saved candidates
and independent element/position assessments are retained; its reported R1 is
not the same native full-data objective. A lower R1 alone cannot establish a
better model. In these saved controls, the one-worker top model has Mg8 Si24
O48; four- and ten-worker models have the correct Mg16 Si16 O48 counts and pass
the labelled reference assessment. The faster multicore comparison therefore
includes successful typed SHELXT models.

### Full final replay breakdown

The second v40 observer replay records:

| Stage | Seconds |
| --- | ---: |
| Shadow masks | 2.980 |
| Detection and linking | 3.248 |
| Indexing and geometry | 0.133 |
| Integration and profile learning | 5.795 |
| Profile fitting and scaling | 4.036 |
| Complete search stage | 0.493 |

The first automatic pass runs on a 32 × 16 × 64 initial grid; its completed
cycle counts are 256, 140, 256 and 256. The selected map is start 1. Only the
inversion-only diagnostic and Pbca undergo phase tests, and one model setting is
fitted. Original-data fitting uses the 64 × 32 × 128 map representation and all
original selected measurements. Bounded omissions remain explicit and uniqueness
is never asserted. Search is now a small fraction of full workflow time;
integration/profile learning, scaling and image stages dominate.

The synthetic multi-peak microbenchmarks also run independently: all 345 settings
on 4,168 measured rows take about 4.43 ms; cached origin search on a 32-cubed,
eight-peak field takes 2.02 ms versus 2.60 ms with repeated transforms; periodic
interpolation from 32-cubed to 64-cubed takes 2.29 ms. These kernel timings are
separate from complete workflow measurements.

## Recovery and escalation checks

Every staged replay retains bitwise-identical masks, positions, indexing,
profile-bank samples, use flags, signed intensities and uncertainties relative to
v33. v34 cache reuse preserves all original candidate records and phase results.
Coarse initialization deliberately changes the final candidate: R1 becomes
0.0514043045582282, with labelled reference RMS/max errors 0.004263/0.006179 Å.
Repeated final automatic runs retain identical candidates and selected density;
the explicit effort wrapper preserves those numerical results.

The final selected setting is Pbca, Hall `-P 2ac 2ab`; the CIF contains ten
asymmetric sites and expands independently to Mg16 Si16 O48, 80 sites. The
reference is read after worker exit. The original supplied-Pbca control retains
R1 0.041128354912261904 and bitwise native candidate/reduction parity. Its earlier
strict Python/native final-candidate discrepancy remains a historical
qualification in the [coordinator validation](single-crystal-native-validation.md).

Frozen v41 deliberately exhausts the fast pass after one microsecond. Automatic
effort records `no_model`/`max_runtime`, performs no fast model attempt, then
runs the full-resolution 64 × 32 × 128 grid with four 800-cycle starts and
12 settings. It recovers the correct Pbca structure with the original broader
candidate, R1 0.051433695723409355. This negative control takes 34.45–34.92 s including
reduction; it is not a normal performance result. Separate mode controls verify
that `fast` does not escalate after failure and `thorough` bypasses the fast pass.

V42 checks rejection of an existing complete candidate, rather than a timeout:
an intentionally unattainable common-R1 ceiling of 1e-12 rejects the fast model
and triggers the full-resolution pass. Neither pass silently relaxes the gate.
The final outcome stays unaccepted, retains the better approximate model, and
archives the other complete pass and its density. This is an acceptance-negative
control, not a performance measurement.

The independent NumPy interpolation agrees on deterministic random grids and
analytic periodic fields, including mixed Nyquist planes and unequal-axis
upsampling. Prepared integer absence tests agree with the original exact
cyclotomic detector for every standard group 1–230 and alternative settings;
nonconventional denominators retain the original detector. Real evidence remains
identical to the independently checked baseline. Existing fused values/derivatives
remain unchanged; the powder GSAS-II oracle does not cover these discrete
assessment and scheduling changes.

Final checks pass: 543 Rust workspace tests with all features on Rust 1.89.0
(34 ignored), four native-example tests, and 3,297 installed-wheel Python tests
(11 external-data skips, 34 explicitly deselected). Strict Clippy, formatting,
minimal Rust features, warnings-denied Rustdoc, API/version/equation/skill
synchronization and strict documentation builds pass. Deterministic native/NumPy
comparisons run through the real executable. Normal NumPy-only wheel smoke also
verifies bundled Bitshuffle/LZ4 HDF5 decoding. Remote CI is configured but was
not run as part of this local evidence.

The publication basis for the shared P1/phase-origin/light-centric sequence is
[Sheldrick (2015)](https://doi.org/10.1107/S2053273314026370). The exact stabilizer
and interpolation equations are independently derived in the frozen plan. Full SHELXT feature
coverage, universal structure recovery, and standalone refinement are not claimed.

The receipt is `validation/results/fast-space-group-search-0.8-macos.json`.
Large inputs, all trials, standalone binaries, profiles, mode controls, distributions
and full verification logs remain external:

```text
/Users/clemens/ChatGPT/PhaseSmith/fast-space-group-search-0.8-20261101
```

The source distribution was rebuilt from its extracted directory with a fresh
Cargo target; all 14 local crate manifests resolved inside that extraction.
Its wheel passes the same NumPy-only smoke. After preserving benchmark samples,
binaries, distributions and logs, removing owned build/extraction/documentation
products reclaimed 8.16 GB of observed filesystem free space. The preserved
standalone binary and installed source-built wheel pass smoke checks afterward.
