# Crystallography foundation: cells and P1 structure factors

This page defines PhaseSmith's file-independent unit-cell and P1
structure-factor conventions. The foundation accepts caller-supplied complex
scattering amplitudes; the higher-level crystallography APIs add symmetry,
reflection generation, built-in scattering models, intensity corrections, CIF
import, and profile calculation.

## Unit-cell convention

The public cell is \((a,b,c,\alpha,\beta,\gamma)\), with lengths in
ångströms and angles in degrees. \(\alpha\) is the angle between \(b\) and
\(c\), \(\beta\) between \(a\) and \(c\), and \(\gamma\) between \(a\) and
\(b\).
Trigonometric calculations convert degrees to radians explicitly.

The direct metric is

\[
G =
\begin{pmatrix}
a^2 & ab\cos\gamma & ac\cos\beta \\
ab\cos\gamma & b^2 & bc\cos\alpha \\
ac\cos\beta & bc\cos\alpha & c^2
\end{pmatrix}.
\]

The reciprocal metric is \(G^* = G^{-1}\). For Miller column vector
\(\mathbf{h}\),

\[
q^2 = \mathbf{h}^{\mathsf T}G^*\mathbf{h},
\qquad
d = \frac{1}{\sqrt{q^2}}.
\]

Reciprocal vectors use cycles per ångström, without a \(2\pi\) factor. The
cell volume is \(V=\sqrt{\det G}\).

For any direct-cell parameter \(p\), derivatives are evaluated analytically:

\[
\begin{aligned}
\frac{\partial G^*}{\partial p}
  &= -G^*\frac{\partial G}{\partial p}G^*, \\
\frac{\partial q^2}{\partial p}
  &= \mathbf{h}^{\mathsf T}
     \frac{\partial G^*}{\partial p}\mathbf{h}, \\
\frac{\partial d}{\partial p}
  &= -\frac{1}{2}d^3\frac{\partial q^2}{\partial p}, \\
\frac{\partial V}{\partial p}
  &= \frac{V}{2}\operatorname{tr}\!\left(
     G^*\frac{\partial G}{\partial p}\right).
\end{aligned}
\]

The stable derivative order is \(a,b,c,\alpha,\beta,\gamma\); angle rows
include the radians-per-degree chain.

## P1 structure-factor convention

For reflection \(\mathbf{h}\) and independent P1 site \(j\), the foundation
kernel receives the complex scattering amplitude \(f_{hj}\) and calculates

\[
\begin{aligned}
T_{hj} &= \exp\!\left(-2\pi^2 U_{\mathrm{iso},j}q_h^2\right), \\
A_{hj} &= o_j f_{hj}T_{hj}
          \exp\!\left(2\pi i\,\mathbf{h}\!\cdot\!\mathbf{x}_j\right), \\
F_h &= \sum_j A_{hj}, \\
I_h &= S\lvert F_h\rvert^2.
\end{aligned}
\]

Here \(o_j\) is occupancy and \(S\) is phase scale.
\(U_{\mathrm{iso}}\) is in square ångströms and fractional coordinates are
dimensionless. In this foundation calculation, multiplicity and all
Lorentz/polarization/sample corrections are one. Higher-level structural
calculations apply those factors exactly once through typed models.

Caller-supplied \(f_{hj}\) is held fixed with respect to cell parameters.
Built-in scattering models supply \(\mathrm{d}f/\mathrm{d}s\), where
\(s=\sqrt{q^2}/2\), and add that chain without changing this API convention.

The structural derivative identity is

\[
\frac{\partial I}{\partial p}
= 2S\,\operatorname{Re}\!\left(
  \overline{F}\frac{\partial F}{\partial p}\right).
\]

For phase scale, \(\partial I/\partial S=\lvert F\rvert^2\). Coordinate
derivatives use \(2\pi i h_kA_{hj}\); occupancy derivatives use the
unoccupied site amplitude; isotropic-displacement derivatives use
\(-2\pi^2q^2A_{hj}\).

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
