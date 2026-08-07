# Repository Guide for Coding Agents

## Read first

Read `PROJECT_BRIEF.md` before changing architecture or numerical behavior. The
brief is the source of truth for scope, milestones, validation, and licensing
boundaries.

## Hard constraints

- GSAS-II is a pinned external oracle only. Never add it as a runtime dependency
  or make its dictionaries, globals, project files, or per-reflection Python
  loops part of the core design.
- Do not copy or mechanically translate GSAS-II implementation code. Implement
  equations from citable publications or first principles, and validate with
  black-box numerical comparisons.
- The Rust kernel owns production profile evaluation and fused accumulation.
  Python reference code must remain independent and readable, not wrap Rust.
- Values and analytical derivatives must be evaluated in the same peak/sample
  pass. New profile terms need corresponding derivative and finite-difference
  tests.
- Finite support is observable behavior. State its convention, handle boundary
  semantics explicitly, and test it.
- Preserve deterministic results unless a documented performance decision and
  tolerance analysis justify a change.

## Repository conventions

- Rust production code lives under `crates/`; Python source uses the `python/`
  layout; integration tests live in `tests/`.
- Keep the numerical core dependency-light. Domain translation, serialization,
  oracle interaction, and convenience APIs belong outside `phasesmith-core`.
- Prefer typed structs and contiguous arrays over nested mappings.
- Public Python functions accept and return NumPy arrays. Validate dtype, shape,
  finiteness, sortedness, and parameter ranges at the boundary.
- Use `cargo fmt`, `cargo clippy --workspace --all-targets --all-features`,
  `cargo test --workspace --all-features`, and `pytest` before completing a
  numerical change.
- Add or update a benchmark when modifying a hot loop. Benchmarks must cover a
  realistic multi-peak grid, not only a scalar profile call.
- Keep tolerances explicit and local to assertions. Do not loosen a global
  tolerance to hide a regression.

## Numerical change checklist

1. Write the equation and parameter conventions in documentation.
2. Add or update the independent Python reference.
3. Implement the Rust value and analytical derivatives together.
4. Test special cases and invalid inputs.
5. Compare Rust and Python on randomized deterministic cases.
6. Check analytical derivatives with centered finite differences away from
   support boundaries.
7. Check normalization/integrated intensity and peak moments.
8. Add an oracle fixture or explain why the oracle does not cover the change.
9. Run benchmarks and report meaningful regressions.

## GSAS-II oracle rules

- Pin an exact GSAS-II revision and record provenance in `oracle/`.
- Prefer the public scripting API. Put unavoidable private access behind the
  small probe adapter in `python/phasesmith/oracle/`; keep it version-gated and
  covered by fixture-schema tests.
- Oracle adapters extract plain arrays and records: `X`, `Ycalc`, background,
  and reflection lists. No GSAS-II objects cross into the numerical core.
- Do not silently regenerate golden data. Make regeneration an explicit command
  and review numerical diffs.
- Never require GSAS-II to import or use the normal `phasesmith` package.

## Change discipline

Do not combine unrelated cleanup with numerical work. Preserve user changes in
the working tree. Architectural changes should update `PROJECT_BRIEF.md` in the
same commit so later agents inherit the reasoning.
