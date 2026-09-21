# Pawley repair and promotion review, 2026-09-21

The dense Intel convergence failure and installed validation path failures
identified by the consolidation review are repaired. Scientific acceptance
thresholds, caller fit budgets, profile equations, finite support and golden
fixtures are unchanged. The numerical contract is documented in
[boundary feasibility](pawley-boundary-feasibility.md).

## Cause and result

The active-set subproblem used an absolute feasibility floor and an absolute
negative-motion cutoff. Near a zero Lorentzian-width face, it admitted outward
motion large enough to make trial widths negative. Physical validation rejected
those trials correctly, but repeated rejection raised damping and ended in
stagnation. The repair uses scale-relative feasibility and recognizes negative
blocking motion at every step size. Caching row and vector norms avoids repeated
work in the constraint loop. Convergence still requires its existing certificate.

| Intel dense boundary regression | Before | Repaired |
| --- | ---: | ---: |
| Termination | stagnated | converged |
| Evaluations | 99 | 5 |
| Infeasible trials | 74 | 0 |
| Rwp | 6.452e-12 | 7.152e-17 |
| Projected-gradient norm | 1.001e-9 | 7.953e-15 |

The original test keeps its error and convergence assertions and now requires
the actual projected-gradient certificate. A separate analytic projection test
covers coupled constraints at step scales from 1 to 1e-20. Existing independent
NumPy objective/derivative and exhaustive active-face comparisons remain intact.

TOF validation accepts an explicit `--manifest`; fixed-spectrum validation now
accepts `--oracle-fixture`. Their defaults resolve relative to the working
directory, and missing external assets produce clear argument errors. Both
installed CLIs pass from outside the repository with explicit asset paths.
The scheduled workflow and documented commands use those paths explicitly.

## Completed local validation

- 970 Python tests pass on each of ARM64 and Intel (Rosetta), with 11 external
  dataset skips and 34 marker deselections on each architecture.
- 365 Rust tests/doctests pass; 34 opt-in tests are ignored by the ordinary run.
  The separate Rust/Python differential run passes 11 tests.
- Strict Clippy, formatting, Ruff, generated mathematics/skill checks and strict
  documentation pass. Public API and persistence contracts pass in pytest.
- Both canonical QARR 1g entry-point tests pass with the pinned real data.
- The final native ARM64 binary SHA-256 is
  `b083b0be1151462b9a8fdf788d54875a1fa12a3ac4c3f617e909357c186d442e`.

## Measured gates and benchmarks

All predeclared measured Pawley gates pass with exact repeated arrays/history:
sucrose Rwp 0.066054, Echidna fixed-cell 0.284869, Echidna cell/profile 0.218412,
POWGEN TOF 0.216797 and joint nickel TOF 0.020989. Sucrose takes about 136.9
seconds per repeat, below the unchanged 240-second limit. Fixed-spectrum ceria
and the declared pinned-oracle engineering checks also pass. The documented
strict oracle profile-equivalence exception remains; it is not concealed by
this solver fix. Matrix-free rank/covariance limitations remain unchanged.

The cached and uncached fixes have exactly matching measured scientific records
when wall times are excluded. Final measurements ran sequentially after local
builds/tests, with one BLAS worker. Three-repeat whole-fit benchmark medians:

| 256 reflections, 10,001 samples | Original solver (s) | Repaired solver (s) |
| --- | ---: | ---: |
| Dense, fixed geometry | 0.6954 | 0.6903 |
| Dense, joint width + FCJ | 1.0764 | 1.0661 |
| Matrix-free, fixed geometry | 0.04090 | 0.04033 |
| Matrix-free, joint width + FCJ | 0.09746 | 0.09747 |

No runtime regression is observed in this comparison; variations within about
1.5% are not evidence of a speed improvement from this small desktop sample.
The new 256-peak zero-width boundary case converges in five evaluations for
both modes, with relative profile errors below 1.1e-15 (dense about 0.993 s,
matrix-free about 0.088 s). Large fixed-spectrum and joint TOF benchmarks also
pass their numerical and exact-repeatability checks.

Raw reports, before/after Intel diagnostics, wheel hashes, paired benchmark
measurements and the exact installed CLI command ledger are retained in
`validation/results/pawley-fix-20260921-*.json`. Initial failed prototypes and
complete build/test logs remain in the external preservation directory.

## Distribution verification

The initial repair passes [normal CI](https://github.com/CPrescher/PhaseSmith/actions/runs/35584278779)
and the entire [distribution dry run](https://github.com/CPrescher/PhaseSmith/actions/runs/35584305718),
including macOS Intel. The final cached implementation and merged development
history pass [CI](https://github.com/CPrescher/PhaseSmith/actions/runs/35585324892)
and the [final distribution run](https://github.com/CPrescher/PhaseSmith/actions/runs/35585352467).
All configured jobs pass, including Intel/ARM64 macOS, Windows, Linux x86-64,
Linux AArch64 construction and the fresh sdist installation. Publishing jobs
are skipped. These results complete the Pawley promotion gates. The distribution
run tests `d0ce20a`; the handoff adds only documentation and evidence.
No tag, package publication or merge into main is included.
