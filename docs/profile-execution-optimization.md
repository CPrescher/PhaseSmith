# Bounded profile execution optimizations

These changes preserve the finite voxel model, mask/reference selection,
iteration counts, derivatives, accumulation order and scientific controls.
They change execution and local work reuse only. Large data/products remain
outside Git. Every stage has its own archived benchmark and fresh MgSiO3 run.

## Stage 1: immutable learning geometry cache

For a fixed grid, teacher centre, detector map, local scale and sample footprint,
the exact overlap `A_ij` is constant throughout learning. The forward operation
is `expected_i = sum_j A_ij p_j`; the original gradient accumulates
`reference_weight_i * A_ij * (target_i - expected_i)` into each global/local
bank. Cache the original nonzero `(voxel, overlap)` visits in source sample and
voxel order. Initialization, both forward bank sums and backward accumulation
retain the original floating-point operation sequence. No rounding, interpolation,
renormalization or new finite-support convention is introduced.

The native `profiles::Execution` / optional Python `ProfileExecution` controls
additional geometry-cache bytes (default 64 MiB; zero disables caching; maximum
1 GiB). Count before allocating; use exact vector reservations and reject an
allocation exceeding the declared cap. If admission or reservation fails, use
the original streaming equations. Cache rows include only accepted teachers,
and the cache is local to the immutable call: it is never reused after geometry,
reference selection, grid or source data changes. Actual admitted capacity,
row count and intersection count are reported separately from the profile model.

The independent NumPy reference remains unchanged because the equations are
unchanged. Native and Python tests compare cached/streaming paths bitwise for
axis-aligned and affine/signed-rocking inputs, reference weighting and tiny
budgets. Existing independent value/partial/support/normalization tests remain
authoritative. GSAS-II does not provide this empirical single-crystal oracle.

`benchmarks/profile_execution.py` measures the complete real learned bank and
baseline amplitude integration on the preserved 7,277-reflection shoebox case.
Three deterministic repeats, explicit execution controls, admitted cache bytes
and process peak RSS are reported. Fresh full runs use the existing observational
wall-time wrapper and compare every numerical array and saved candidate with
retained v11. Timings and correctness are reported separately at each stage.
