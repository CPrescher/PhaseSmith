# Pinned GSAS-II structural-parameter study

Status: documentation survey complete; live pinned P1 perturbation fixture
pending an available GSAS-II checkout.

This study exists to understand the observable parameter and reflection-table
conventions of pinned GSAS-II revision
`c0bc79b259cdf0065480b5fbd57674ddf12c4a23` (tag `#5838`). It is not an
equation source and does not authorize copying implementation code or data
structures.

## Documentation observations

The [pinned GSAS-II object/variable documentation](https://gsas-ii.readthedocs.io/en/latest/objvarorg.html)
documents these externally visible concepts:

| Concept | Pinned GSAS-II observation | Rietveld Engine unit-11 convention |
| --- | --- | --- |
| Cell | `A0` through `A5` are reciprocal-metric components | Direct `a`, `b`, `c`, `alpha`, `beta`, `gamma` with analytical metric conversion |
| Coordinates | `Ax`, `Ay`, `Az` are fractional atom coordinates | Fractional `site.<id>.x/y/z` |
| Occupancy | `Afrac` is atomic site fraction | Dimensionless `site.<id>.occupancy` |
| Site multiplicity | `Amul`; atom record multiplicity is read-only through scripting | Symmetry-derived value, never an independently refinable parameter |
| Isotropic displacement | `AUiso`; atom records expose `Uiso` | `site.<id>.u_iso` in square ångströms |
| Anisotropic displacement | `AU11` ... with six stored components | Reserved until symmetry-constrained anisotropic implementation |
| Parameter identity | `p:h:<var>:n`, omitting unused ownership fields | Typed owner plus durable phase/site/reflection ID |

The same pinned documentation defines the ordinary powder `RefList` columns as
`h,k,l`, multiplicity, d-spacing, position, Gaussian width, Lorentzian width,
observed/calculated squared structure factors, reflection phase, total
intensity correction, preferred orientation, transmission, and extinction.
The oracle adapter may extract named copies of these fields, but the numerical
library does not reproduce the row layout.

GSAS-II documentation also notes that nonphysical occupancy or `Uiso` results
are diagnostic and are not automatically hidden by constraints; optional
parameter limits can freeze a variable at a specified boundary. Rietveld
Engine will make physical bounds explicit in refinement specifications and
will report rank/bound warnings. Exact trial-step behavior will be decided only
after the live perturbation study.

## Live P1 study schema

The external generator will create a one-atom P1 constant-wavelength project
and perturb, one at a time:

1. all six cell degrees of freedom through the public phase interface;
2. fractional `x`, `y`, and `z`;
3. site occupancy;
4. isotropic `Uiso`;
5. phase/histogram scale.

For each base/plus/minus state it records plain arrays/records only:

- parameter name, displayed value, refine flag, and perturbation;
- `X`, `Ycalc`, and background;
- powder reflection `hkl`, multiplicity, d-spacing, position, widths,
  calculated squared structure factor, phase, and correction columns;
- input and output checksums, pinned revision, Python/NumPy versions, and
  adapter schema version.

The public scripting API is used first. Any missing intermediate receives one
narrowly named, revision-gated copied-value probe. The generator imports no
`rietveld` module. Fixture regeneration is explicit and reviewed.

## Required comparisons

- direct-cell to reciprocal-metric conversion and derivative signs;
- `Uiso` exponent and unit convention;
- occupancy and phase-scale normalization;
- reflection phase convention;
- the distinction between `Fcalc^2` and the final corrected integrated
  reflection intensity;
- centered finite-difference response in `Fcalc^2`, position, and `Ycalc`.

Until this live fixture is generated, GSAS-II-specific equivalence is not
claimed. Closed forms, independent NumPy equations, finite differences, and
adjoint checks remain sufficient to continue implementing the independent P1
foundation.
