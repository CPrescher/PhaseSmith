# GSAS-II performance comparison

`benchmarks/compare_gsasii.py`,
`benchmarks/compare_gsasii_structural.py`, and
`benchmarks/compare_gsasii_qarr.py`, and
`benchmarks/compare_gsasii_pbso4.py` provide reproducible speed
comparisons between PhaseSmith and the exact GSAS-II revision recorded in
`oracle/PINNED_GSASII.json`. GSAS-II remains a separately installed external
validation oracle and is never imported by the normal package.

## Profile-only workload

The default case contains 200 monochromatic constant-wavelength reflections on
a sorted 5,001-sample grid. Both implementations return all of the following:

- the accumulated symmetric CW profile;
- support-block local derivatives for integrated intensity and position;
- dense shared derivatives for U, V, W, X, and Y.

Each timed operation includes reflection-dependent width calculation, TCH
support calculation, binary support lookup, output allocation, profile and
analytical derivative evaluation, and accumulation. Warmup calls, interpreter
startup, process creation, and NPZ transfer are outside the timed region.
PhaseSmith uses one fused native batch call. The pinned GSAS-II worker uses
one `GSASIIpwd.getdPsVoigt` call per active reflection and performs batch
orchestration in Python, which is the interface GSAS-II exposes for this
profile-level comparison.

This scope is intentionally narrower than a complete powder calculation or
refinement. It does not compare structure factors, reflection generation,
background models, constraints, optimizers, project-file I/O, or GUI work. The
reported ratio must therefore be described as profile-and-derivative
throughput, not as total Rietveld-refinement speed.

## Structural-intensity workload

The separate structural benchmark uses a synthetic 32-site P1 crystal with C,
O, Si, and Fe atoms, 256 GSAS-II-generated powder families, and monochromatic
neutron radiation. It reports two independently timed cases:

- structure-factor values through integrated reflection intensities;
- the same structural calculation composed with a 20,001-sample symmetric CW
  profile and analytical intensity, position, U, V, W, X, and Y derivatives.

Both programs receive the same atom coordinates, occupancies, isotropic
displacements, cell, HKLs, multiplicities, wavelength, instrument parameters,
finite-support convention, and sample grid. The GSAS-II project, reflection
generation, dictionary preparation, process startup, and NPZ transfer are done
once outside the timed region. The PhaseSmith phase and prepared pattern
are likewise constructed outside the timed region. The first Rietveld timing
uses the public values-only structure-factor API; the second uses the fused
native `PreparedStructuralPattern` path.

The structural comparison deliberately uses neutron scattering and a neutral
integrated-intensity correction. GSAS-II tabulates coherent scattering lengths
in `10^-12 cm`, while PhaseSmith's independently sourced table uses fm;
the adapter therefore applies the exact squared-unit conversion factor of 100.
Because the tabulations are independently sourced and rounded, normalized
maximum errors for F-squared and intensity must remain below `1.25e-4`, not
machine precision. Reflection geometry is checked at `2e-14`; the unweighted
profile kernel derivative remains gated at `6e-6`; intensity-weighted profile
and derivative arrays include the table difference and are gated at `1.5e-4`.

This is a controlled structure-to-profile throughput comparison, not a full
Rietveld refinement. It excludes background evaluation, constraints,
optimization, project I/O, and GUI work. It also does not time reflection-list
generation because the list is prepared input to both structural kernels.

## QARR complete-workflow workload

The QARR comparison runs the checksum-pinned IUCr QARR 1g pattern and its three
CIF phases through each package's explicit native workflow. The external worker
uses the public GSAS-II scripting API, disables the exactly redundant overall
sample scale, and stages background/phase scales, U/V/W/zero, isotropic
size/microstrain, and atomic displacement parameters. Its ablation switches can
place SH/L below the pinned calculation floor or hold sample broadening or
displacement fixed. Pinned GSAS-II clamps the applicable powder-profile SH/L to
0.002, so the below-minimum case validates that behavior rather than claiming
to disable FCJ. The optional trace-mean anisotropic ablation is the sole pinned,
revision-gated internal probe because the public wrapper does not expose that
ADP representation change.

The default discarded complete warmup also populates GSAS-II's process-external
Matplotlib/font cache. The report separates GSAS-II import, project/CIF setup,
every refinement stage, finalization, total workflow, and external-process wall
time. It records phase fractions with standard uncertainties, residuals,
correlation, parameter and reflection counts, exact input hashes, and the
complete recipe. Repetitions run in fresh subprocesses and must produce
identical scientific outputs.

Before timing is reported, the driver also gates the two final scientific
records against each other. Sample and reflection counts must match exactly;
the maximum absolute phase-fraction difference must not exceed `0.02`, the
Poisson-weighted and unit-weight Rwp differences must not exceed `0.03` and
`0.02`, and the profile-correlation difference must not exceed `0.01`.
These are complete-workflow tolerances for intentionally different
parameterizations, not matched-kernel numerical tolerances.

This is deliberately labeled a native-workflow comparison rather than a
same-parameterization benchmark. PhaseSmith uses a fixed Smooth Bruckner
background and fixed CIF anisotropic tensors, while the GSAS-II recipe refines
ten Chebyshev background terms and may refine anisotropic parameters. Those
differences are reported in JSON and must not be hidden behind a single timing
ratio.

The 2026-08-08 live pinned check measured a maximum phase-fraction difference
of `0.007249`, Poisson-weighted Rwp difference of `0.014365`, unit-weight Rwp
difference of `0.005580`, and profile-correlation difference of `0.000921`;
all cross-implementation checks passed.

## PbSO4 real X-ray/neutron workload

The PbSO4 comparison uses the official commit-pinned GSAS-II combined-
refinement tutorial inputs: a packed constant-wavelength Cu Kα X-ray pattern,
a packed 1.909 Å neutron pattern, their legacy instrument files, and the PbSO4
CIF. This expands real-data coverage beyond QARR to two radiation probes on the
same crystalline specimen.

The pinned GSAS-II worker follows the published staged recipe: background,
cell, X-ray H-strain, X-ray size/microstrain, sample/atom parameters, angular
limits, and U/V/W. It jointly refines one structure against both histograms.
PhaseSmith currently performs two independent native refinements. Each uses a
fixed Smooth Bruckner baseline plus a refined three-term Chebyshev residual
correction. Its neutron workflow refines the lattice; its X-ray doublet
workflow keeps the supplied lattice fixed. The result therefore reports—not
obscures—the difference between a joint richer model and two independent
current-capability workflows.

PhaseSmith's optional intelligent planner discloses and runs cumulative
scale/background, position, structure, and final-polish stages from only the
parameter families authorized by the request. The residual polynomial remains
active cumulatively after its first stage. GSAS-II's stages remain the explicit
official tutorial recipe. Neither numerical solver chooses these sequences
internally.
The PhaseSmith neutron path uses the constant-wavelength powder Lorentz factor
`1/(sin(theta) sin(2 theta))`; using a neutral intensity correction is not a
valid parity workflow.

Both workflows now use the same documented Debye--Scherrer displacement
parameterization for neutron peak positions: a fixed 650 mm goniometer radius,
X perpendicular to the incident beam, and Y parallel to it, with X/Y in
micrometres. PhaseSmith refines X/Y analytically in its position stage. The
pinned result reports X=1578.824 µm and Y=49.924 µm for GSAS-II's joint fit;
PhaseSmith's independent neutron fit reports X=910.803 µm and Y=-529.510 µm.
They are not equality-gated because the shared structure, background, and
probe coupling differ. Peak residuals and the refined cell remain the relevant
cross-implementation gates.

The reference CIF cell is a=8.480, b=5.398, c=6.958 Å. The corrected pinned
check returned PhaseSmith's neutron cell a=8.470449, b=5.391700,
c=6.951482 Å and GSAS-II's joint cell a=8.473965, b=5.393891, c=6.954432 Å;
their maximum relative difference was 0.000424. PhaseSmith/GSAS-II Poisson Rwp
values were 10.346%/10.573% for X-ray and 4.217%/4.535% for neutron.
Corresponding correlations were 0.99555/0.99554 and 0.99664/0.99669. The
probe-specific Rwp-delta gates are at most 0.006 and the correlation gates are
at most 0.002.

On the same release host, one discarded warmup and three measured repetitions
gave an 8.191 s PhaseSmith median (8.204 s p95) for both disclosed staged
workflows together and a 12.943 s GSAS-II median (12.944 s p95) for the full
six-stage joint workflow. The GSAS-II/PhaseSmith median ratio was 1.580x. These
are host-specific observed timings, not cross-machine acceptance thresholds.

## Run it

Build PhaseSmith in release mode, then use GSAS-II's separate interpreter
and compatible binary directory:

```shell
maturin develop --release --uv
uv run python benchmarks/compare_gsasii.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --json-output gsasii-speed-comparison.json

uv run python benchmarks/compare_gsasii_structural.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --json-output gsasii-structural-speed-comparison.json

uv run python benchmarks/compare_gsasii_qarr.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --phasesmith-threads 2 \
  --data-directory validation/data/iucr-qarr-1g \
  --json-output gsasii-qarr-comparison.json

uv run python benchmarks/compare_gsasii_pbso4.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --data-directory validation/data/gsasii-pbso4-cw \
  --json-output gsasii-pbso4-comparison.json
```

The equivalent `GSASII_PYTHON`, `PHASESMITH_GSASII_ROOT`, and
`PHASESMITH_GSASII_BINARY_DIR` environment variables are supported. The external
worker refuses any GSAS-II checkout other than revision
`c0bc79b259cdf0065480b5fbd57674ddf12c4a23`.

Before printing a speed ratio, each runner performs its documented numerical
checks. JSON reports record raw timings, median, p95, software versions,
platform, workload dimensions, numerical errors, and the ratio `GSAS-II median
/ PhaseSmith median`. A ratio above one means PhaseSmith was faster
for that specific workload on that machine.

The self-hosted pinned-oracle workflow runs both commands and uploads text and
machine-readable JSON reports. Results should be compared on the same
otherwise-idle host; VM type, CPU power policy, Python/NumPy versions, and build
mode are part of the result, not noise to omit.

## Reference structural run

On 2026-08-06, a 100-repetition release run on the development Apple Silicon
macOS host produced:

| Timed case | Rietveld median | GSAS-II median | GSAS-II / Rietveld |
| --- | ---: | ---: | ---: |
| Structure-factor values and intensities | 0.292 ms | 0.731 ms | 2.508x |
| Structure through profile and CW derivatives | 0.633 ms | 5.211 ms | 8.231x |

The corresponding p95 values were 0.308/0.747 ms and 0.649/5.285 ms,
respectively. The normalized maximum errors were `4.683e-5` for complex F,
`9.134e-5` for both F-squared and integrated intensity, `9.222e-5` for the
profile, and at most `9.385e-5` across the reported derivative groups. These
numbers characterize this workload and host; the committed runner and JSON
schema, not the observed ratio, are the durable performance contract.

## Reference QARR run

The first reviewed single-run comparison on 2026-08-07 returned PhaseSmith
fractions of Al2O3 33.250%, ZnO 32.936%, and CaF2 33.814%, versus GSAS-II
31.479%, 33.652%, and 34.869%. Maximum errors from the independently weighed
fractions were 1.880 and 0.558 percentage points. Poisson-weighted Rwp was
19.679% and 18.389%; unit-weight Rwp was 13.282% and 13.732%.

Measured total native-workflow times were 8.73 s for PhaseSmith and 2.93 s for
GSAS-II on that host. This result motivates structural-linearization reuse and
the missing-physics ablations; it does not override the matched kernel results
above.

After bounded native linearization was enabled for stable stages, a reviewed
two-repetition run preserved those scientific outputs and reduced the
PhaseSmith median to 7.60 s. GSAS-II measured 2.99 s in the same run. QARR stage
2 retains matrix-free products until its rounding-sensitive flat basin is
addressed by optimizer damping and acceptance hardening.

After the structural path gained the exact continuous FCJ convolution, a
reviewed one-repetition release run with no warmup returned essentially the
same scientific result: PhaseSmith fractions 33.253%, 32.934%, and 33.813%,
Poisson Rwp 19.672%, unit-weight Rwp 13.273%, and maximum weighed-fraction
error 1.883 percentage points. PhaseSmith required 158.79 s. The pinned
GSAS-II workflow returned 31.479%, 33.652%, and 34.869%, Rwp 18.389%, and
required 10.59 s including an 8.21 s cold setup in that run. This is not a
matched-kernel comparison: PhaseSmith evaluates the published continuous
equal-height FCJ model while pinned GSAS-II uses its discretized one-parameter
SH/L implementation. The result identifies FCJ convolution reuse and
vectorization as a concrete performance requirement.

After fixed CIF anisotropic tensors, convergence-tested adaptive FCJ
quadrature, and removal of the obsolete QARR-only matrix-free override, the
2026-08-08 warmed comparison returned PhaseSmith fractions 30.754%, 34.229%,
and 35.018%, 19.826% Poisson Rwp, and 13.174% unit-weight Rwp. Pinned GSAS-II
returned 31.479%, 33.652%, and 34.869%, 18.389% Poisson Rwp, and 13.733%
unit-weight Rwp. PhaseSmith took 1.673 s and GSAS-II took 2.718 s, so the ratio
`GSAS-II / PhaseSmith` was 1.624x. This is a complete native-workflow result,
not a claim that the two programs optimize identical models or stopping rules.

The subsequent deterministic multicore checkpoint adds a public bounded
`ExecutionPolicy` and reuses special-position coordinate models across guarded
trial states. On the same development host, warmed PhaseSmith-only medians over
three measured runs are 1.245 s with one thread, 0.961 s with two, 0.965 s with
three, and 0.966 s with automatic selection. All settings return the same
scientific record and 19.826% Poisson Rwp. Relative to the preceding pinned
GSAS-II median of 2.718 s, the two-thread PhaseSmith measurement is 2.83x
faster; this last ratio combines two reviewed runs rather than claiming a new
same-process comparison. Reproduce a fresh paired run with
`benchmarks/compare_gsasii_qarr.py --phasesmith-threads 2`.

The Rust-only application-boundary migration was re-gated on 2026-08-08 after
Python values, dense products, JVPs, and VJPs were delegated to the shared
native multiphase workflow. A release build with no warmup and two measured
repetitions returned the exact same timing-free scientific fingerprint for one
and two threads (`77746021bc30373315705b435568305b773bdedb264e78f3538ca2f791ab3831`).
The one-thread median was 1.394 s and the two-thread median was 0.933 s, a 1.49x
speedup. Both returned 19.826% Poisson Rwp, 13.174% unit-weight Rwp, and a
maximum weighed-fraction error of 0.617 percentage points. These host timings
are diagnostic; the exact cross-thread fingerprint and existing scientific
thresholds are the durable migration gate.

## Pinned FCJ strategy review

The performance investigation inspected exact revision
`c0bc79b259cdf0065480b5fbd57674ddf12c4a23` without importing any GSAS-II code
into PhaseSmith. The pinned workflow narrows every reflection to a calculated
support window, evaluates FCJ in a compiled numerical routine, selects among
cached even-order Gauss--Legendre tables from an axial-span/width heuristic,
and has a derivative routine that returns the profile and its position, width,
and asymmetry derivatives together. Those are execution-strategy observations,
not ported implementation details.

PhaseSmith independently retains its published regular-height FCJ integral,
two typed sample/detector ratios, `f64` arithmetic, and exact support union. It
adopts only the general lesson that integration effort should track resolved
aberration: the production 8/48-point rule is selected by axial span divided by
the transformed TCH FWHM and accepted against an independent 256-point NumPy
integral. This keeps the licensing boundary and the numerical model explicit.
