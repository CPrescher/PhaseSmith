# Frozen unknown-symmetry speed experiments

The target is solution/group-search latency comparable with SHELXT, measured
separately from original-frame reduction. The exposed MgSiO3 baseline is v33:
about 19 s search and 36 s full command. Preserve signed observations, original
final-assessment resolution, genuine missing coverage, counts-free element input,
bounded omissions and the prior exact-group execution path. Do not use source
HKLs, reference counts/coordinates or reference R1 to select scientific controls.

Three protocols are frozen before their new replays:

1. **v34:** extract the same complex Fourier density once for all group-origin
   tests. Identical transform arithmetic and phase support; compare map arrays,
   origin/model results and histories bitwise with v33. Repeated source-only
   diagnostic searches quantify the saving before full raw replay.
2. **v35:** allow the published light-element centrosymmetric early exit. Require
   a P-minus-one phase alpha below the stated 0.3 screen, a centrosymmetric
   candidate satisfying the existing own-group R1 target, an accepted common-set
   symmetry model, and no supplied element heavier than Sc. Report this heuristic
   termination and all unsearched groups; never infer uniqueness. Disable it for
   continued bounded alternative searches. Interleave phase tests and model
   attempts in measured-evidence order, first checking the inversion-only phase
   diagnostic, so an accepted early model also avoids later origin tests.
   This follows Sheldrick (2015), section
   5, independently implemented using existing tested kernels.
3. **v36:** phase only genuine measured reflections with d >= 0.8 A (inclusive),
   retaining the original measurements for identity and all final assessment.
   Use the existing Wilson normalization, correlation/weak-score merit and
   quality/checkpoint stopping, with the existing maximum four starts/800 cycles.
   Interpolate the selected periodic density by Fourier zero padding onto the
   original-data grid for symmetry recycling and final full-resolution fitting.
   Padding is a representation change, not fabricated observed intensities.

References: [Sheldrick (2015), sections 2, 3 and 5](https://doi.org/10.1107/S2053273314026370).
The cutoff is motivated by the independently observed SHELXT processing workload;
it is an exposed-data experiment, not a universal safe cutoff or generalization
claim. If speed/recovery gates fail, retain failures and report them rather than
tuning this frozen policy using the reference.

Required evidence: serial/repeated determinism without time interruption;
independent Fourier interpolation and phase-quality comparisons; invalid inputs,
cutoff boundaries, high-Z/acentric/poor-map early-exit rejection and cancellation;
original-frame reduction parity; independently graded labelled structure and
selected setting; full stage timers and matched SHELXT input/timing qualifications.
Use synthetic multi-peak benchmarks and applicable exposed regression cases for
broader checks. Existing exact-group tests and numerical tolerances stay fixed.
Run all-feature Rust tests, strict Clippy, formatting and installed-wheel pytest.
The powder GSAS-II oracle does not cover these discrete search policies.

## Coarse-map representation

The initial cutoff is inclusive in reciprocal metric: keep measured row h when
`|B* h|² <= 1/d_cut²`. Original rows, uncertainties and their order remain the
map's exact identity. Neither filtering nor interpolation changes any measurement
used for final model fitting, common-set scoring or absence assessment.

For source grid N, compute `C(k) = DFT_positive(rho)(k) / product(N)`.
Copy each represented signed frequency onto the finer grid M; on every upsampled
even-grid Nyquist axis split its coefficient equally across +N/2 and -N/2.
With r such axes each of the 2^r images receives C/2^r. The unnormalized negative
DFT gives the finer real map. This preserves the periodic interpolant, DC/mean,
Hermitian symmetry and source samples on integral refinements, including mixed
Nyquist planes. It adds no observations. Equal-grid input returns an exact copy.
Independent NumPy FFT and analytic periodic-polynomial tests check this linear
representation; existing phase-quality derivatives and Fourier kernels retain
their independent reference/finite-difference coverage.

Large arrays, source files, binaries and trials stay outside Git under
`/Users/clemens/ChatGPT/PhaseSmith/fast-space-group-search-0.8-20261101`.

## v37: exact absence assessment execution

The v36 replay exposes about 1.4 s between the complete search stage and its
0.65 s model-search coordinator. Test a fourth stage with identical scientific
controls and bitwise gates. For a validated closed space group, operations fixing
h form its stabilizer; `exp(2 pi i h.t)` is a one-dimensional character of that
finite group. Its sum vanishes exactly iff at least one stabilizer operation has
`h.t` nonintegral. Cosets have the same cancellation, proving the general-position
absence criterion. Prepare integer translation numerators over 24 for conventional
operations with denominators dividing 24 and bounded rotations. Use the existing
exact cyclotomic implementation as fallback for every other operation set.
Cross-check all 230 standard groups and alternative settings against that original
implementation and the independent Python complex-sum reference. This changes no
reflection decision, measurement, likelihood or accumulation order. Compute the
common shell contribution once per measured family rather than once per setting;
retain the existing deferred overflow rejection when an absent family uses it.

## v38: buffer native JSON archives

A retained sampled profile of v37 places substantial wall time in per-token
JSON writes to an unbuffered file. Buffer these writes, explicitly flush and
propagate I/O failures. Preserve exclusive creation, JSON formatting and exact
serialized bytes; no numerical or coverage policy changes. Measure before any
further phasing change.

## v39: deterministic portfolio handoff

The measured four-worker SHELXT control is about 0.9 s and ten-worker control
about 0.65 s, so the one-worker comparison alone is insufficient. Test a portfolio
handoff at the existing 256-cycle checkpoint and measured quality target 0.9.
Run four independent initial starts to at most 256 cycles, retaining their
existing per-start adaptive stopping. If any reaches the existing quality target,
select the best completed map in deterministic start order. Otherwise restart
the four original seeds with the full 800-cycle ceiling and remaining original
deadline, retaining and comparing all maps. Worst-case work therefore includes
four short plus four full starts. Save every actual seed, cycle ceiling, history
and checkpoint; map reuse records the actual selected seed. This is explicit
quality-controlled work scheduling, not a reduction in final-resolution data or
a new correctness threshold. Preserve old profiles when this option is absent.
