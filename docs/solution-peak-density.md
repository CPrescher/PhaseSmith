# Integrated peak density as an assignment guide

This experimental guide uses measured amplitudes and phases of the shared search's
current count-independent map. It does not integrate the sharpened map itself.
No atomic counts, reference coordinates, or absolute electron-density scale are
supplied. A Fourier synthesis with F000 absent is an arbitrary-scale, finite-data
map. Its spherical integrals are relative signals, not measured electron counts.

For Cartesian reciprocal vector q and a ball of radius R,

`K(q,R) = (4 pi R^3/3) 3 [sin(t)-t cos(t)]/t^3`, `t=2 pi |q| R`.

The t=0 limit is the ball volume. The smooth small-t series avoids cancellation.
The radial derivative is `dK/dR = 4 pi R^2 sinc(t)`. Apply K to measured complex
coefficients with phases extracted from the current real map, then synthesize the
periodic integrated-density grid with normalization 1/cell_volume. Missing and
undefined-phase coefficients, including unmeasured F000, stay zero. No resolution
extension or additional observations are introduced. Friedel mates and strict
Nyquist bounds are validated. This is the exact ball integral of the band-limited
Fourier field at grid nodes. Its support is the closed Cartesian ball; the sphere
boundary has measure zero in this analytic integral.

At arbitrary fractional positions, use periodic trilinear interpolation of that
grid. Value, all three fractional-coordinate derivatives and the radial derivative
are accumulated in the same eight-node pass. Coordinate derivatives are piecewise;
at grid planes the cell on the positive side defines the derivative. Centered finite differences avoid these grid planes. Tests separately check the ball-volume
normalization, harmonic response, radius derivative, periodicity and scale.

Relative scale is estimated from the median integrated mass per atomic number of
the current strongest assigned species. Orbit averaging reduces phase asymmetry.
Nonpositive or inconsistent evidence remains uninformative. This is a provisional
assignment prior: nearby atomic numbers remain alternatives, and missing-element
and diffraction-fit rescue paths protect candidates despite a weak map. Final
selection still uses the existing requested-mode refinement and amplitude score.
The guide can be disabled to compare the prior staged search directly. It is not
SHELXT's element-assignment implementation and uses no decompiled program code.

The scale anchors are current asymmetric orbits with assigned Z at least 75% of
the largest current Z. Use positive orbit-average masses; if no anchors survive
or their mass/Z ratios span more than a factor two, use no scale. At a proposed
orbit, `z_est = mean_mass/scale` and `e_j = abs(log(z_est/Z_j))`. Admit labels with
`e_j <= min(e)+log(1.25)` or `e_j <= log(1.5)`, keeping close C/N alternatives.
Nonpositive mass or unavailable scale admits all labels. These tolerances are
explicit heuristic uncertainty bands, not chemical probabilities.

Guidance prunes only the existing top-two diffraction-amplitude birth proposals
per type. It does not fill vacated slots with new, lower-ranked topologies. Always
rescue the leading amplitude proposal for every missing required type, and the
best amplitude proposal across all types. Relabels to missing types are also
retained. Deletion trials and existing assignments are never removed by the prior.

In staged mode, the base final-fit shortlist is capped by the number of admissible
labels at the leading amplitude proposal, at least one. The existing near-tie
window and missing-element rescues still expand it. Thus a confident density
assignment reduces both label screens and unnecessary expensive promotions;
ambiguous or uninformative evidence keeps the larger budget. This is a bounded
heuristic and can change search trajectories, so benchmark validation must include
counts and reference geometry, not just residuals.

Defaults are `integrated_peak_density=True` and
`peak_density_radius_angstrom=0.7`. Set `integrated_peak_density=False` to reproduce
the prior staged controller. The source observations, phasing map, initial peak
prefix, exact supplied symmetry and final fit equations stay the same. Additive
v5 diagnostics record sphere radius, map/integral counts, label rejections/rescues,
net birth/relabel fits omitted and each parent calibration and shortlist cap.
No reference atoms or formula counts enter calibration.

The idea of integrated density for assigning elements is described by Sheldrick
(2015), section 4, [DOI 10.1107/S2053273314026370](https://journals.iucr.org/a/issues/2015/01/00/sc5086/index.html).
The Fourier ball transform, interpolation and conservative controller here are
independent implementations from the equations above. GSAS-II does not expose this
assignment controller and supplies no oracle fixture for it.

## Validation and timing, 28 October 2026

`benchmarks/solution_peak_density_comparison.py` runs the same pinned measured
reflection files as the staged comparison, with three serial alternating pairs
per case. Both sides use staged refinement, one 800-cycle phase start and the
same 60-second total budget; only integrated-density guidance differs. Formula
and coordinate comparisons are deferred until every result has been saved.
All 18 solves passed the count, geometry and exported-CIF checks.

| Case | Staged baseline median | Density-guided median | Speedup | Full final-mode fits |
| --- | --- | --- | --- | --- |
| K[BrF4] | 2.269 s | 1.696 s | 1.34x | 30 → 18 |
| BiSeI | 1.150 s | 0.973 s | 1.18x | 19 → 10 |
| Thiourea, non-H | 7.546 s | 5.658 s | 1.33x | 46 → 32 |

Every pair improved in wall time. Inferred counts, source-matched geometry and
R1 were preserved; R1 values were identical across modes on these runs. Maximum
coordinate errors across both methods were 0.005909 A, 0.002517 A and 0.010372 A.
Both CIF frames for all runs reproduce native R1 within 2e-10. Screening fits
also fell from 74 to 48, 56 to 29 and 240 to 160 respectively. The frozen report
is `benchmarks/structure-solution-peak-density-macos.json`; complete diagnostic
outputs and source snapshots remain under `runs/peak-density-comparison-final/`.
These are reused development cases, not an independent general success-rate test.

A first guide version pruned labels but refilled vacated candidate slots and
retained the fixed promotion budget. It preserved the structures but slowed
KBrF4 and BiSeI. Those probes remain under `runs/peak-density-first/` and
`runs/peak-density-paired-probe/`. The final controller prunes the existing pool
and lets assignment ambiguity govern the promotion budget, with rescue paths.
The gains above therefore reflect fewer model fits, not a faster FFT or reduced
numerical precision.

The multi-peak 32-cubed/2108-reflection Criterion map-construction benchmark took
0.984–0.993 milliseconds. Evaluating 32 centers with value, center gradient and
radius derivative took 0.504–0.515 microseconds with a prepared map. Preparation
and evaluation are timed separately; both use a real multi-reflection field.

Validation: 1333 Python tests passed (11 skipped, 34 deselected); 388 Rust tests
passed (34 ignored). Strict workspace clippy, cargo formatting and selected Python
lint checks passed. Forty new tests cover quadrature/direct-Fourier parity,
center/radius finite differences, sphere-volume normalization, grid-plane
semantics, missing phases/F000, periodicity, broad/narrow equal-charge peaks,
invalid controls, uncertainty masks and synthetic count recovery. The broader
suite includes strongly anisotropic recovery and origin/frame checks. A fresh
default wheel imports without the experimental solver or density integrator.
