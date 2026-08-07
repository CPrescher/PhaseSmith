# Exact symmetry and reflection generation

Status: implemented and internally validated in implementation unit 12.

This layer is file-independent. It accepts exact operations supplied by callers
or, later, by the optional CIF adapter. It does not contain a space-group name
database and does not depend on GSAS-II, Gemmi, or Python crystallography
objects.

## Operation convention

An operation acts on fractional column coordinates as

```text
x' = R x + t  (mod 1)
```

`R` is an integer unimodular 3 by 3 matrix and each component of `t` is a
reduced rational number. Translations are normalized to `[0, 1)`. A
`SpaceGroup` must contain the identity, contain no normalized duplicates, and
be closed under exact composition. The reciprocal orbit action is

```text
h' = R^T h.
```

Because every inverse rotation belongs to the closed group, this produces the
same orbit as the alternative inverse-transpose convention. It also directly
matches

```text
F(h) = exp(2 pi i h.t) F(R^T h).
```

## Sites, absences, and families

`SpaceGroup.expand_sites()` applies every operation to each asymmetric site.
Special positions are deduplicated with an explicit periodic coordinate
tolerance; the default is `1e-10` fractional coordinate. Exact operations do
not make measured floating atom coordinates exact, so this tolerance is part
of the public contract.

For a candidate `h`, operations are grouped by common transformed index
`R^T h`. Within each group the translation character sum is

```text
sum_g exp(2 pi i h.t_g).
```

The implementation converts rational phases to an integer polynomial and
reduces it modulo the relevant cyclotomic polynomial. A general-position
systematic absence is reported only when every grouped sum is exactly zero;
there is no floating extinction tolerance.

Families use a sorted reciprocal orbit and a stable ID `hkl:h,k,l`. Friedel
mates are merged by default for ordinary non-resonant powder work. Set
`merge_friedel=False` when anomalous or other physics makes `h` and `-h`
distinct. Accidental equal-d families retain separate IDs and rows.

## Metric constraints

Every rotation requires

```text
R^T G R = G
```

for the direct metric `G`. The library derives exact integer equations in
component order

```text
(g11, g22, g33, g23, g13, g12)
```

and an exact integer nullspace basis. The basis is a usable linear
parameterization of every symmetry-compatible metric; coefficients must still
produce a positive-definite metric. Crystal-system labels are derived from the
rotational topology rather than used to impose hard-coded standard-axis
assumptions. This permits non-standard monoclinic, trigonal, and other settings.
Reflection generation rejects a supplied cell that violates the exact
constraints beyond a local relative tolerance of `1e-10`.

## Inclusive physical ranges

`PreparedReflectionGenerator` caches exact group topology and accepts:

- `DSpacingRange(min_angstrom, max_angstrom)`;
- `ScatteringVectorRange(min, max)` with `Q = 2 pi / d`;
- `CwTwoThetaRange(min_deg, max_deg, wavelength_angstrom)` using monochromatic
  Bragg geometry `2 theta = 2 asin(lambda / (2 d))`;
- `TofRange(...)` using
  `t(d) = zero + DIFC d + DIFA d^2 + DIFB / d` and an explicit safe d search
  interval.

All endpoints are inclusive with a local allowance of 64 floating-point
epsilons. This makes monochromatic-beam generation a first-class path rather
than a later adapter.

For the maximum reciprocal length `q_max = 1 / d_min`, a conservative lower
bound on the smallest reciprocal-metric eigenvalue is

```text
lambda_min(G*) >= 1 / trace(G).
```

The implementation combines that bound with the exact ellipsoid projections

```text
|h_i| <= q_max sqrt(G_ii)
```

and a roundoff guard to form a safe triclinic integer box. A configurable
candidate limit rejects unreasonable ranges before enumeration.

## Python example

```python
from fractions import Fraction

import numpy as np
import phasesmith

identity = phasesmith.SymmetryOperation.identity()
centring = phasesmith.SymmetryOperation(
    np.eye(3, dtype=int),
    (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)),
)
group = phasesmith.SpaceGroup([identity, centring])
generator = phasesmith.PreparedReflectionGenerator(group)

cell = phasesmith.UnitCell(4.2, 4.2, 4.2, 90.0, 90.0, 90.0)
reflections = generator.generate(
    cell,
    phasesmith.CwTwoThetaRange(10.0, 120.0, wavelength_angstrom=1.5406),
)
```

The returned NumPy arrays contain stable IDs, canonical `hkl`, multiplicity,
d-spacing, reciprocal length, and all six analytical cell derivatives. Arrays
are copied or natively allocated and exposed read-only.

## Validation and performance

Rust tests cover exact composition/closure, P-1 special positions, I centring,
P2_1 screw extinction, deterministic orbits, range equivalence, cell mismatch,
and candidate limits. Independent Python tests additionally cover F centring,
glide and three-fold screw extinctions, every crystal system, a non-standard
monoclinic setting, randomized triclinic reflection sets, and finite-difference
cell derivatives.

The release benchmark uses a prepared eight-operation C-centred orthorhombic
group, a `9.1 x 11.2 x 13.4` ångström cell, and `d = 0.5..14` ångströms. It
generates 3,174 allowed families per iteration. Benchmark numbers are recorded
in `IMPLEMENTATION_PLAN.md`; they are development-machine observations, not an
API guarantee.
