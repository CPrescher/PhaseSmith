# Blocked normal matrices

Sequential five-second native profiles of the preceding best `1fe9889` show
normal-system preparation as the largest refinement leaf: 1,557/4,142 main-thread
samples in COD 2010543 and 418/4,156 in COD 2011241. COD 2243618 remains dominated
by Fourier work. These representative samples establish an opportunity, not a
whole-corpus allocation. Drivers, actual native SHA/UUID, raw sample hashes and
collapsed counts are recorded in
`benchmarks/structure-solution-blocked-normal-profiles-macos.json`.

## Equations and execution

For residual r and parameter-major Jacobian J (P rows, M reflections), the
existing diagonally scaled damped system is

    d_i = max(sum_m J_im^2, 1e-8)
    s_i = 1 / sqrt(d_i)
    H_ik = s_i s_k sum_m J_im J_km
    b_i = -s_i sum_m J_im r_m
    (H + lambda I) x = b,   step_i = s_i x_i.

The opt-in `blocked_normal_matrices=True` control requires
`reuse_normal_matrices=True` and its existing iterative/cache dependencies. It
changes only the Gram sum. Lower block tiles of at most 32 by 32 are formed with
safe ndarray 0.17.2 `general_mat_mul`; lower entries are scaled in the original
order and mirrored. Output views write directly into the existing normal matrix,
and J views borrow existing contiguous storage. The zero-right-hand-side path
allocates no normal matrix. Small dense fits and ineligible shapes retain their
previous implementation. Default execution is unchanged.

The experimental solution crate adds a pinned ndarray dependency already present
in the lock through the NumPy binding. The powder core has no new dependency.
Only the `std` feature is selected: no BLAS, Rayon or matrixmultiply threading.
The backend uses architecture-specific SIMD, with portable fallback. No external
SHELX or GSAS-II code is copied or translated; the operation is the normal-equation
identity above. Workspace `unsafe_code=forbid` remains in force for our code.

The existing cache limits remain P>=32, M>=2P and P^2*8<=8 MiB, inside the
existing 2,000,000 Jacobian-entry boundary. There is no second full matrix.
The backend temporarily packs panels; for the pinned default DGEMM KC=256 and
32-by-32 tiles its two panels occupy at most 128 KiB, plus alignment and small
backend scratch. Alternative compile-time backend block overrides may change
that panel bound. Cancellation/deadlines are checked before each tile and in
all existing solver loops. Stopped work does not launch a legacy rebuild.

Damping, scaled RHS, iteration ceilings, curvature checks and backtracking are
unchanged. Every cached candidate step is reapplied through the original J
operator and must meet its original relative squared-residual bound (1e-16, or
1e-10 at the iteration ceiling). A failed blocked solve first replaces the
matrix with the original eight-lane Gram calculation and retries checked CG;
that legacy matrix stays for further damping attempts at this linearization.
The existing matrix-free fallback follows if necessary. Accepted model changes
invalidate the cache exactly as before; candidates/fits never share a cache.

Backend summation order can change floating-point bits and selected model paths.
Bitwise equality with the preceding implementation is not assumed. Repeated
identical blocked executions remain deterministic. Actual backend preparations
and legacy rebuilds are exported as `blocked_normal_builds` and
`blocked_normal_fallbacks`, separately from total matrix builds/reuses and the
existing cached-step/matrix-free-fallback counters. Failed fits that produce no
result retain the existing aggregate-counter limitation.

## Validation

The independent NumPy normal-equation references remain unchanged and cover
random, rank-deficient, tiny/zero columns, scale variation and damping 1e-8 to 10.
Checks compare the original residual bound and predicted data rather than
requiring uniquely identified near-singular parameter values. Rust tests compare
lower/partial-edge tiles against legacy Gram entries with absolute tolerance
3e-14, exercise zero RHS, cancellation/deadlines and cache invalidation, and
corrupt a blocked matrix to verify the original-operator guard and legacy retry.
Complete multiatom isotropic/anisotropic P1/P-1/Pm fits are compared with the
independent physical evaluator, including centered finite-difference derivatives.
Three synthetic element-only full searches check recovered counts/geometry and
public control validation. Physical values/analytical derivatives and support
are unchanged. The GSAS-II powder oracle does not cover this linear solver.

Criterion `blocked_normal_` covers complete 40-step eight-atom anisotropic fits
and complete eight-atom Fe/C/N/O element-only searches (2,108 reflection rows,
32-cubed phase grid), with actual work counters in the raw log. Comparisons use
control/treatment within the same executable, not historical Criterion deltas.

`benchmarks/solution_blocked_normal_interleaved.py` runs all 35 development cases
serially with alternating 18/17 leading order. Only the blocked-normal flag
differs. Both modes retain short phase starts, all preceding accepted controls,
seed zero, four starts, 800 cycles, native13/outer15-second caps, known supplied
symmetry and anisotropic final refinement. All 140 predictions are saved before
both reference-audit barriers. Seven above-cap references remain in denominators.
Adapter-only revision31 inherits30 with unchanged registry/split; the 65 holdout
outcomes remain unopened. No task-owned tests, builds or profiling overlap timed
collection. Exact prior successful-case sets and every inferred atom-count vector
are checked against both the fresh control and previous best report.

## Complete development results

The exact previous **19/35 successful structures** are retained against both
fresh control and the preceding best. Total non-H counts remain **26/35**,
per-element counts **19/35**, and strict 0.02-Angstrom recovery **15/35**. All four
successful-case sets agree. Every one of the 35 inferred count vectors is
unchanged, including incorrect and no-model outcomes. Four cases still return
no model; no PhaseSmith hard timeout occurs. All seven above-cap references
remain included, and the 65 holdout outcomes remain unopened.

| Measurement | Previous Gram | Blocked Gram | Change |
| --- | ---: | ---: | ---: |
| Summed wall time, all 35 cases | 61.64892 s | 58.62529 s | 4.90% less |
| Typical paired control/treatment ratio | 1 | 1.02549 | 2.49% less time |
| Median wall time, all outcomes (descriptive) | 1.31727 s | 1.27119 s | 3.50% less |
| Final refinement time, summed | 10.50487 s | 8.32317 s | 20.77% less |
| Isotropic screening time, summed | 11.34420 s | 10.61453 s | 6.43% less |
| P1 search time, summed | 24.49795 s | 24.56820 s | 0.29% more |

There are 29 faster and six slower cases. The median paired ratio, rather than
the ratio of two aggregate medians, defines typical speed here. These are one
balanced serial collection's host-sensitive timings, not a repeated uncertainty
estimate or a comparison with an older build's absolute times.

Both modes perform 95 phase starts, 667 assignment screens, 125 final fits and
250 ambiguity relaxations. Existing per-start time guards produce cycle
variation in three cases: 39,390 versus 39,433 completed Fourier cycles, an
increase of 43 in the treatment. Every quota and cap is identical. The treatment
uses 5,217 blocked preparations and 861 normal reuses; no legacy-Gram or
matrix-free cache fallback is needed in these completed fits. Normal CG work can
change with summation order: final cached solves are 2,373 versus 2,369, and
assignment cached solves 4,587 versus 4,585. No equal-work bitwise claim is made.

Twenty-two serialized model records change in floating-point values; 13 are
identical including the four no-model records. Geometry/count correctness is
checked independently; changed bits are not hidden as identical models. The
retention artifact records every case, exact recovery sets and all count vectors.

The isolated Criterion complete 40-step fit median is 59.6655-to-50.9094 ms,
**14.68% less time**. Its accepted-step count remains 28. The complete small
search median is 197.5618-to-198.4121 ms, **0.43% more**, with the same work.
That search prepares only four eligible normal systems and does not demonstrate
a meaningful end-to-end benefit on its own. The full development comparison is
the evidence for the net 4.90% gain. Refinement improves more than overall time
because the unchanged P1 phase work remains the largest aggregate stage.

Evidence is retained in
`benchmarks/structure-solution-blocked-normal-{control,cache,comparison,retention,schedule,kernels,profiles,validation}-macos.json`.
The schedule verifier checks all 140 predictions precede both audit barriers,
18/17 leading order, 57 current source/build-input hashes per mode and matching
actual native hashes. Dependency manifests and Python native bindings are now
included in the snapshot as well as numerical source and adapters.

Validation: `cargo fmt --check`, workspace Clippy with all targets/features and
warnings denied, **420 Rust tests passed, 34 ignored**, and **2,555 Python tests
passed, 11 skipped, 34 deselected**. Python was built with both experimental
solution and spot modules. The option remains false by default. This is another
bounded development improvement, not general unseen-data or SHELXT parity.
