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

Large arrays, source files, binaries and trials stay outside Git under
`/Users/clemens/ChatGPT/PhaseSmith/fast-space-group-search-0.8-20261101`.
