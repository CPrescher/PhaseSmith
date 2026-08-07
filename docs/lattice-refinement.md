# Lattice refinement and guarded reflection domains

This document fixes the public parameter, derivative, and topology conventions
for CIF-backed Le Bail lattice refinement. CIF parsing remains an optional I/O
adapter. Cell geometry, symmetry constraints, reflection generation, profile
values, and profile derivatives do not call GSAS-II.

## Script-first entry point

The shortest single-phase workflow is:

```python
from phasesmith.refinement import lebail

request = lebail.LeBailInput.from_cif(
    observed_pattern,
    cw_instrument,
    "phase.cif",
    phase_id="alpha",
    refine_lattice=True,
    lattice_relative_bound=0.05,
    lattice_angle_bound_deg=5.0,
)
result = lebail.refine(request)

print(result.metrics.rwp)
print(result.phases[0].structure.cell)
print(result.phases[0].structure.diagnostics)
```

`LeBailInput.from_cif()` uses the first and last pattern coordinates as the
visible monochromatic two-theta interval. It retains the parser-independent
structure source and diagnostics, marks the reflection batch as generated, and
constructs a bounded parameter record. `LeBailPhase.from_cif()` remains the
lower-level constructor and defaults to a fixed cell for backward
compatibility. Explicit-reflection `Phase` and `LeBailInput` construction are
unchanged.

## Independent lattice variables

`LatticeParameterization` derives the allowed variables from the exact metric
constraints of the supplied space-group setting. It does not infer a
conventional unique axis from the crystal-system name alone.

| System/setting | Independent variables |
| --- | --- |
| triclinic | `a, b, c, alpha, beta, gamma` |
| monoclinic | `a, b, c` and the setting's one free angle |
| orthorhombic | `a, b, c` |
| tetragonal or hexagonal axes | equal-plane length and unique-axis length |
| rhombohedral axes | common length and common angle |
| cubic | common length |

Lengths are ångströms and angles are degrees. Parameters have stable keys such
as `lattice/alpha/a_angstrom`. Every refinable lattice variable has finite
bounds. A trial outside its bounds is rejected by the existing parameter
transform before a physical cell is constructed.

## Geometry and analytical derivatives

For Miller vector `h`, direct metric `G`, reciprocal metric `G* = G^-1`, and

```text
q = h^T G* h
d = q^(-1/2),
```

the cell derivative is

```text
dG*/dp = -G* (dG/dp) G*
dd/dp  = -(1/2) q^(-3/2) h^T (dG*/dp) h.
```

The parameterization supplies the exact chain from each independent variable
to `(a, b, c, alpha, beta, gamma)`. Monochromatic coordinates follow Bragg's
law,

```text
2theta = 2 asin(lambda / (2d)),
d(2theta)/dd = -(180/pi) lambda / (d^2 sqrt(1 - (lambda/(2d))^2)),
```

where the derivative is degrees per ångström. The TOF utility follows the
existing calibration convention

```text
t = zero + DIFC d + DIFA d^2 + DIFB/d,
dt/dd = DIFC + 2 DIFA d - DIFB/d^2.
```

The Le Bail pattern column is the chain product of this coordinate derivative
and the local position derivative returned by the fused Rust profile
accumulator. That local derivative already includes the position dependence of
the selected CW width model. Geometry derivatives are tested against centered
finite differences for triclinic, non-standard monoclinic, orthorhombic,
tetragonal, hexagonal, both trigonal settings, and cubic cells. CW and TOF use
the same d-spacing derivative.

## Guard and regeneration semantics

The visible interval is closed: a reflection at either requested two-theta
boundary is visible. The prepared domain is broader than that interval. It
uses finite lattice bounds plus eigenvalue bounds on the direct and reciprocal
metrics to derive a d-spacing range that covers any family which can become
visible within the allowed parameter box. `guard_scale` (default `1.001`)
expands that mathematically bounded range further; it is not a fit parameter.

Reflection topology is fixed while one Jacobian and line-search trial are
evaluated. After a profile step is accepted, the domain is regenerated at the
accepted cell. Existing intensities are transferred by stable Miller-family ID,
new families receive `initial_intensity`, and removed IDs are reported in the
iteration warnings. Guard-only reflections with no sampled support retain
their current intensity instead of being silently reset to zero. Ordinary
explicit reflections preserve the established Le Bail behavior.

The domain configuration and current generated topology are included in
persistence format 3. Restart accepts changed reflection IDs only when the
input and checkpoint have identical symmetry, bounds, wavelength, visible
range, and guard policy.

## Source and validation boundary

The metric identities, reciprocal differentiation, Bragg law, and TOF
calibration chain above are implemented independently from their definitions
and from the existing native cell/profile contracts. No GSAS-II source code or
data structure was used. The pinned GSAS-II oracle can validate reflection
positions and final `Ycalc` when its separate checkout is available, but it
does not define parameter ownership, guard topology, or optimizer iteration
identity. Those engine-specific contracts are covered by independent finite
differences, stable-ID boundary tests, exact synthetic recovery, persistence
restart tests, and the realistic lattice-Le Bail benchmark.

The first release-build baseline on Apple silicon (Python 3.13.5, 20,001
samples, 128 guarded/98 visible tetragonal families) measured median times of
1.915 ms for domain regeneration, 3.908 ms for one fixed-intensity Le Bail
iteration, and 18.149 ms for one two-parameter lattice/profile iteration. The
benchmark reports these cases separately because the lattice iteration performs
analytical column assembly, a damped solve, trial pattern calculations, and
accepted-domain handling in addition to intensity redistribution.

The general crystallographic definitions and the remaining external-oracle
protocol are listed in [the crystallography plan](crystallography-plan.md).
