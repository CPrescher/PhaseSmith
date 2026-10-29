# Missing reflections and resolution extension

This experiment adds opt-in `missing_reflections=False` and
`resolution_extension=1.0` (reciprocal-radius multiplier, maximum 1.5) controls to
the shared phase-origin path. It retains chemical assignment and iterative
refinement as independently controlled features.

The previous solver already lets unmeasured allowed coefficients vary inside the
measured sphere. That is the control, not a zero-filled baseline. The new missing
reflection option adds a bounded, Wilson-normalized conditional prediction rule.
Extension adds allowed coefficients beyond the measured sphere and, when needed,
a larger FFT grid. Neither option creates observed intensities or changes the
original HKLF4 arrays, scoring families, atom caps, refinement observations,
weak-reflection tests or origin tests.

The physical basis is density-modification prediction of unmeasured complex
coefficients (Caliandro et al., 2005, DOI 10.1107/S090744490500404X;
Caliandro et al., 2007, DOI 10.1107/S0021889807034073) and its use in SHELXT
(Sheldrick, 2015, DOI 10.1107/S2053273314026370). The conditional second moment
E(R²|Rp)=1+sigma_A²(Rp²-1) motivates shrinkage toward the Wilson expectation.
The bounded estimator, warm-up, grid bounds and taper below are PhaseSmith
choices implemented from equations, not recovered SHELXT code.

For a map coefficient z, an expected amplitude e from the fitted observed Wilson
trend, a shell model second moment v and intensity correlation c clipped to [0,1],
define R²=|z|²/(e²*v), cap R² at 16, and set

    a = gain * c * e * sqrt(1-c+c*min(R²,16)),
    z_pred = a*z/sqrt(|z|² + 0.01*e²*v).

`gain` includes a deterministic warm-up/ramp and an outer extension-band cosine
taper. For the first eight P1 cycles, internal gaps retain their old free values
and the extension band is zero. Over the next sixteen cycles the gap update
blends its old free coefficient with the bounded prediction, while the extension
band ramps directly from zero. Beyond the measured reciprocal radius Qobs, the
taper is (1+cos(pi*t))/2 with t=(Q²-Qobs²)/(Qext²-Qobs²), clamped to [0,1].
Aligned supplied-symmetry recycling uses full gain after phase projection and
once more at its final projection. Thus the final extended map remains tapered. The extra c factor suppresses unsupported predictions. Soft phase
regularization tends to zero at a zero coefficient; it never invents a phase.
Statistics use measured independent symmetry/Friedel families only. At most eight
radial bins in Q²/Qobs² are used; bins with fewer than 12 measured independent
families fall back to measured global moments. A 1e-10 tolerance in the scaled
bin coordinate puts numerical boundary ties on the same side for equivalent
reflections. Model second moments are bounded to [1e-12,1e12]. The
expected intensity uses supplied neutral element types, epsilon factors and the
fitted resolution trend, with the end-shell residual held constant beyond the
measured range. Extension is not a claim of additional measured resolution.

The complex update and its 2x2 Cartesian Jacobian are evaluated together. Writing
r=|z|, d=sqrt(r²+0.01*e²*v), f=a/d, its derivative is

    J = f I + (a'/(d*r) - a/d³) [Re(z),Im(z)] [Re(z),Im(z)]ᵀ,

with the appropriate zero limit at r=0. The amplitude derivative is zero above
the R² cap; finite-difference tests avoid that boundary. Moment estimates are
fixed inputs to this local derivative, not differentiable observation estimates.

Origin/DC remains free and is not normalized by the Wilson fit. Known absences
and coefficients outside the support remain zero. Measured coefficients retain
the existing reciprocal projection/hybrid update. Integrated peak masses are
still calculated from the original measured, unsharpened amplitudes, avoiding
interpretation of extrapolated coefficients as measured electrons.

## Support, stages and resource boundaries

The existing sphere includes its boundary with a relative squared-frequency
1e-12 tolerance. The extension sphere uses the same convention; its outer taper
is zero at and beyond the requested boundary. Nyquist planes are excluded.
Automatic grids remain bounded to 128 on each axis and 2,097,152 voxels. If the
requested extension would cross an actual grid's Nyquist plane, the effective
radius is explicitly reduced using the conservative cell-length bound and is
recorded separately from the request. A grid unable to contain the measured
sphere fails. The default grid calculation is preserved exactly when extension
is 1.0. No atom-count or dense-Jacobian cap is relaxed.

Prediction has separate domains for mixed G=sqrt(E*F) P1 phasing and unsharpened
Fo symmetry recycling. With observed Wilson expected intensity W, their
expected amplitudes are W^(1/4) and W^(1/2), respectively, divided by the original
measured-amplitude RMS for that domain. The measured targets are never replaced.
Both domains use only original independent families to estimate moments and
intensity correlations. The correlated conditional second-moment expression is
regularized empirically rather than claimed as an exact SHELXT reproduction.

The audit distinguishes original measured row count, newly regularized missing
pairs, extended pairs, actual grid and effective predicted resolution. A zero
`missing_pairs` when the option is off does not mean the measurements are
complete: ordinary internal free coefficients still exist. Per-solve counters
sum prediction calls, nonzero predicted pair updates, cap hits and update time.
Prediction time includes shell statistics and overlaps the parent phasing or
recycling timer; preparation and FFT-grid growth remain in wall time. No
predicted coefficient is counted as an observed refinement constraint.

## Validation and development-only comparison

Tests cover independent Wilson extrapolation with symmetry epsilon factors,
local values/Jacobians and centered finite differences including zero/capped
coefficients, global scale equivariance, measured/DC/fixed-zero preservation,
Hermitian updates and full NumPy/native dual-space-cycle parity. End-to-end
P1, inversion and mirror cases recover full counts and coordinates under all
three feature combinations while preserving original measured arrays. The
GSAS-II powder oracle does not cover this single-crystal prediction stage.

The fixed experiment uses all 35 development cases, seed 0, four starts, an
800-cycle ceiling, the same 13-second internal/15-second outer budgets and one
numerical thread. Chemical assignment, iterative refinement, normalization,
phase quality and phase-origin selection stay on in every mode. Compare current
control, missing-only, extension-only at factor 1.2, and both. The factor was
chosen before inspecting this campaign's outcomes. The extension comparison
includes any required grid enlargement; it does not isolate finer sampling from
additional Fourier support. Every denominator retains all cases, including
resource failures. Per-panel predictions are saved before their reference audit;
normalized re-audits follow the four complete collections. Adapter-only seal
revision 12 preserves the revision-11 registry and split. The 65 holdout outcomes
remain unopened.

All four panels used compiled native SHA-256
`5a04b184a9beebdf6d7c8c29e4d9a5a5238dea6be76ec66d65907a17749de60a`.
The runtime source hashes still match their saved snapshots. Full predictions
and audits are in `benchmarks/structure-solution-prediction-{mode}-macos.json`
for `control`, `missing`, `extension` and `combined`; the guarded comparison is
`benchmarks/structure-solution-prediction-comparison-macos.json`.

| All 35 development cases | Current control | Missing only | Extension only | Both |
| --- | ---: | ---: | ---: | ---: |
| Correct labelled structures | 14 | 14 | 15 | 14 |
| Exact total non-H counts | 26 | 27 | 26 | 27 |
| Exact per-element full-cell counts | 15 | 15 | 15 | 15 |
| Strict 0.02-angstrom recoveries | 11 | 11 | 12 | 12 |
| Median wall seconds, all outcomes | 3.388 | 4.455 | 4.244 | 4.268 |
| 95th percentile wall seconds | 7.680 | 7.384 | 9.068 | 7.804 |
| Hard timeouts | 0 | 0 | 0 | 0 |

Missing-only gains no structures and loses none, while recovering the total
count of COD 2011426; its element-count set is still wrong. Extension-only gains
COD 4326074 with no lost recoveries. Both gains COD 4326074 but loses COD 2108342.
For the lost case the full counts stay correct, but the maximum coordinate error
increases from 0.0237 to 3.5153 angstrom despite reported R1 values of 0.05931 and
0.05963. This is an actual structural failure, not a loosened-tolerance issue.
The original primary gates remain unchanged.

Relative to control, median all-outcome runtime is 31.5% higher for missing-only,
25.3% higher for extension-only and 26.0% higher for both. Among the 14 common
control/extension successes, medians are 1.578 versus 1.609 seconds. The 13 common
control/combined successes give 1.676 versus 1.669 seconds. These are single-seed
serial collections, not repeated paired timing estimates or general reliability
claims. No mode provides a demonstrated overall speed/recovery improvement on
this panel; neither new control is enabled by default.

### Separating padding from predicted data

Extension enlarges the FFT grid in 21 cases, by as much as eight times the voxel
count. For COD 4326074 the grid grows from 16x16x32 to 16x16x64, with 623 predicted
extension pairs. A further **padding-only** diagnostic uses that larger grid but
keeps the original measured cutoff, original free internal gaps and no new
prediction rule. It also recovers the correct structure. Its chosen prediction
is saved before reference audit, using the same native build and solver settings
apart from grid shape. Input/implementation hashes and the full selected model
are retained in `benchmarks/structure-solution-prediction-grid-probe-macos.json`.
Reproduce with `benchmarks/solution_prediction_grid_probe.py`.

Thus the extra recovery cannot be attributed to extrapolated coefficients: a
finer grid is sufficient in this diagnostic. This is one selected development
case, not a fifth 35-case panel or independent success-rate/timing evidence. It
motivates a separate grid/origin-search study rather than automatic extrapolation.
The only distinct total-count gain from the bounded gap rule remains incorrectly
labelled. Predicted reflections have not established an independent structural
recovery benefit over padding alone here.

### Work and remaining limits

Quality-scored Fourier cycles change from 44,885 in control to 47,331 for missing,
29,563 for extension and 29,583 for both. Recorded prediction-update time is
0.427, 0.834 and 0.869 seconds across the corresponding complete panels, including
6.87, 63.49 and 67.07 million nonzero pair updates. The explicit update is cheap;
changed trajectories, larger FFTs/maps and subsequent fitting dominate cost.
Requested-mode fit time is 35.753 seconds for control, 40.069 for missing,
41.466 for extension and 36.423 for both. A low amplitude residual or correct
atom count still does not certify correct coordinates.

All 35 cases remain in every denominator; seven exceed the original input-derived
atom cap. All requested extension factors of 1.2 fit the actual grids in this
campaign without clipping. SHELXT in the combined collection recovers 13 labelled
structures and 29 total counts at a 0.855-second median, with one hard timeout;
its space-group search and refinement model differ. The 65 corpus holdout
outcomes remain unopened.

Validation passes: 1,634 Python tests (11 external-data skips, 34 deselected),
398 Rust tests (34 ignored), all-feature/all-target Clippy with warnings denied,
Rust formatting, Ruff and `git diff --check`. No original correctness or global
numerical tolerances were relaxed. New finite-difference tests use a smaller
step at the zero/perfect-correlation limit, whose derivative convergence is
first order; they do not loosen an existing tolerance.

On a multi-peak 32-cubed grid with 2,108 measured rows, preparation takes
961.35 microseconds (957.42–966.01 interval) and a prediction update including
shell statistics takes 26.89 microseconds (26.57–27.03 interval). Raw Criterion
samples, confidence intervals and source/executable hashes are in
`benchmarks/structure-solution-prediction-kernels-macos.json`. These measure the
new kernel, not an end-to-end speedup. Neither tests nor corpus collection ran
concurrently with the timed microbenchmark.

To reproduce each full panel, use `solution_corpus_benchmark.py run --manifest
validation/solution-corpus-v5.json --seconds 15 --wilson-normalization
--phase-quality-selection --phase-origin-selection --chemical-assignment
--iterative-refinement --output NEW_DIRECTORY`. Add `--missing-reflections`,
`--resolution-extension 1.2`, or both for the other modes. Run
`solution_corpus_reaudit.py` only after collection; use
`solution_phase_ablation.py` to compare the complete saved reports.
