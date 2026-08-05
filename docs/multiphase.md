# Multi-phase pattern composition

Implementation unit 6 introduces the first domain-level pattern calculation.
It composes typed phases above the reflection-profile kernel; the Rust core
continues to receive only flat contiguous reflection and contribution arrays.

## Intensity and derivative convention

For phase `p` and reflection `i`, let `I[p,i]` be the supplied base integrated
intensity, `s[p] >= 0` the phase scale, `m[p,i]` the product of optional sample
intensity modifiers, and `P[p,i](x)` the normalized broadened profile. The
profile part of the calculated pattern is

```text
Y_profile(x) = sum_p sum_i I[p,i] s[p] m[p,i] P[p,i](x).
Y_total(x) = Y_profile(x) + Y_background(x).
```

The background array is added exactly once outside the peak kernel. It has no
parameter row in this unit. Negative base intensities are accepted so
derivative, difference, and cancellation calculations are representable;
physical refinement workflows may impose stronger bounds.

Each phase contributes the dense analytical scale row

```text
dY/ds[p] = sum_i I[p,i] m[p,i] P[p,i](x).
```

Provider width and intensity chains retain their existing equations and are
prefixed by durable phase ID, for example
`phase[alpha].physics.isotropic_microstrain.rms`. Instrument rows remain first,
then each phase contributes its scale row followed by its provider rows in
input order. The local support blocks use the same flattened phase/reflection
order and are labeled by `(phase_id, reflection_id)`.

## One-call flattening

`calculate_pattern` validates phases, calls each optional batch provider once,
and flattens the following arrays in stable input order:

```text
positions
base integrated intensities
sample Gaussian variances and Lorentzian FWHMs
sample intensity multipliers
all position and named-parameter derivative chains
```

It then makes one native support-limited accumulation call. `PreparedPattern`
performs validation, provider evaluation, and flattening at construction and
reuses the immutable arrays for repeated calculations. It does not hide mutable
optimizer state.

Optional phase-separated curves are diagnostics. They are reconstructed from
the intensity column of the same native local support blocks, so enabling them
does not issue per-phase native calls. Their sum agrees with the fused profile
within the explicitly tested floating-point summation tolerance.

## Public ownership

- `ReflectionBatch` owns unique durable reflection IDs and a validated
  `ReflectionGeometryBatch`.
- `Phase` owns its durable ID, display name, reflections, non-negative scale,
  and optional provider.
- `PowderPattern` owns the sorted grid, optional observations, positive
  uncertainties, mask, and supplied background.
- `PatternCalculationResult` returns total/profile/background arrays, derivative
  storage, phase offsets, reflection keys, and optional phase components.

No GSAS-II project object, nested dictionary, or file state is represented in
these types.
