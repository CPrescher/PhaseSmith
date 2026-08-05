# Crystallography foundation: cells and P1 structure factors

Implementation unit 11 establishes the file-independent numerical boundary for
crystallography. It intentionally accepts caller-supplied complex scattering
amplitudes. Element tables, symmetry expansion, reflection generation,
integrated-intensity corrections, CIF import, and profile fusion belong to
later units.

## Unit-cell convention

The public cell is

```text
(a, b, c, alpha, beta, gamma)
```

with lengths in ångströms and angles in degrees. `alpha` is the angle between
`b` and `c`, `beta` between `a` and `c`, and `gamma` between `a` and `b`.
Trigonometric calculations convert degrees to radians explicitly.

The direct metric is

```text
G = [[a^2,       ab cos(gamma), ac cos(beta)],
     [ab cos(g), b^2,           bc cos(alpha)],
     [ac cos(b), bc cos(a),     c^2]].
```

The reciprocal metric is `G* = inverse(G)`. For Miller column vector `h`,

```text
q^2 = h^T G* h
d   = 1 / sqrt(q^2).
```

Reciprocal vectors use cycles per ångström, without a `2 pi` factor. The cell
volume is `sqrt(det(G))`.

For any direct-cell parameter `p`, derivatives are evaluated analytically:

```text
dG*/dp = -G* (dG/dp) G*
dq^2/dp = h^T (dG*/dp) h
dd/dp = -(1/2) d^3 dq^2/dp
dV/dp = (V/2) trace(G* dG/dp).
```

The stable derivative order is `a`, `b`, `c`, `alpha`, `beta`, `gamma`; angle
rows include the radians-per-degree chain.

## P1 structure-factor convention

For reflection `h` and independent P1 site `j`, the foundation kernel receives
the complex scattering amplitude `f_hj` and calculates

```text
T_hj = exp(-2 pi^2 Uiso_j q_h^2)
A_hj = occupancy_j f_hj T_hj exp(2 pi i h dot x_j)
F_h = sum_j A_hj
I_h = scale |F_h|^2.
```

`Uiso` is in square ångströms and fractional coordinates are dimensionless.
`I_h` has multiplicity and all Lorentz/polarization/sample corrections set to
one. Those factors are added once, through typed owners, in implementation unit
15.

Caller-supplied `f_hj` is held fixed with respect to cell parameters in this
slice. A later scattering model supplies `df/ds`, where
`s = sqrt(q^2) / 2`, and adds that chain without changing this API convention.

The structural derivative identity is

```text
dI/dp = 2 scale Re(conjugate(F) dF/dp)
```

with an additional `dI/dscale = |F|^2`. Coordinate derivatives use
`2 pi i h_component A_hj`; occupancy derivatives use the unoccupied site
amplitude; isotropic-displacement derivatives use `-2 pi^2 q^2 A_hj`.

## Parameter and array layout

The native and Python derivative order is:

1. six cell parameters;
2. `x`, `y`, `z` for every site in input order;
3. occupancy for every site;
4. `Uiso` for every site;
5. phase scale.

Python labels site rows with durable IDs such as `site.si1.x` and
`site.si1.u_iso`. Derivative arrays are parameter-major with shape
`(parameter_count, reflection_count)`.

The dense result is an explicit diagnostic and small-problem interface, guarded
by a configurable element limit. Production refinement uses native
`p1_jacobian_vector_product(...)` and
`p1_intensity_transpose_jacobian_vector_product(...)`, which do not materialize
the dense matrix.

## Validation

The implementation is checked with:

- cubic closed forms and randomized triclinic cells;
- an independent NumPy metric/structure-factor implementation;
- centered finite differences for all cell, coordinate, occupancy, `Uiso`, and
  scale rows;
- forward/dense and reverse/dense comparisons plus the adjoint identity;
- integer lattice translations and common origin shifts;
- shape, finiteness, physical-domain, zero-reflection, and dense-memory errors;
- realistic values/JVP/VJP Rust benchmarks.

For the recorded 5,000-reflection, 64-site release benchmark, median times are
approximately 2.502 ms for values, 2.843 ms for one JVP, and 6.895 ms for one
intensity VJP. Throughput is measured in atom/reflection contributions and was
approximately 127.9, 112.6, and 46.4 million contributions per second,
respectively, on the development machine.

The pinned GSAS-II documentation and live behavioral study are used to map
external parameter conventions. They do not supply these equations or execute
inside the library.
