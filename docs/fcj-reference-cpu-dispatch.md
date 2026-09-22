# FCJ reference CPU dispatch

The final 0.7 release push check failed two NumPy 1.26 comparisons although
another runner passed the identical commit. This was reproduced in all four
[controlled hosted probes](https://github.com/CPrescher/PhaseSmith/actions/runs/35716606148),
covering Intel Xeon Platinum 8573C and AMD EPYC 9V45/9V74 CPUs with AVX-512.
Each probe used the same reviewed native wheel and Python 3.12/NumPy 1.26.4:

| Reference math path | Focused result on each of four runners |
| --- | --- |
| Original, default NumPy CPU dispatch | 2 failed, 4 passed |
| Original, AVX-512 disabled in a fresh process | 6 passed |
| Scalar libm trigonometry, default NumPy CPU dispatch | 6 passed |

The failing cases were the fast FCJ small-span axial derivative comparison
(`ratio=0.001`) and the fixed-doublet global Jacobian comparison. The first
had scaled error about 2.6993e-9 against the unchanged 2e-9 limit; the second
failed 2 of 18,009 entries against `rtol=8e-11, atol=8e-9`.

## Numerical cause and repair

The same equations remain in use. For Bragg position `b` and normalized axial
height `z`, the apparent angle is

\[
a=\arccos(\cos b\sqrt{1+z^2}).
\]

For a wavelength ratio `r`, the component position is

\[
2\theta_r=2\arcsin(r\sin\theta).
\]

Angles inside trigonometric functions are radians; profile positions and
intrinsic widths are degrees. FCJ normalization and its analytical quotient
rule derivatives are unchanged.

NumPy's vector and scalar trigonometric paths can differ at the last bit.
Subtracting the apparent angle from a nearby sample coordinate, and forming
normalized axial derivatives, can amplify this difference. NumPy documents
[CPU dispatch controls](https://numpy.org/doc/1.26/reference/simd/build-options.html)
and the upstream [AVX-512 math difference](https://github.com/numpy/numpy/issues/23523).
The controlled on/off probes establish its effect on these two tests; the
initial unexplained failure was not dismissed by rerunning until green.

Only the independent reference's small angular arrays now use Python `math`
scalar libm calls. Profile arrays, accumulation, and intrinsic exponential
math remain readable NumPy operations. The reference still implements its
own equations and derivatives and never delegates calculation to Rust.
This specifies a consistent angular path across NumPy CPU dispatch choices,
not bitwise equality across all operating-system libm implementations.

Production Rust code, finite support, boundaries, physical models, solver
acceptance, public API and persistence formats are unchanged. Existing
scientific and differential tolerances are unchanged. No oracle fixture is
regenerated: the pinned GSAS-II comparison does not test NumPy CPU dispatch.
The reference is also used by offline calibration/diagnostics, so those
consumers can see last-bit changes; native refinement histories are unaffected.

## Regression coverage

The existing randomized FCJ comparisons, finite differences, high-precision
angular integrals, normalization and doublet tests remain required. A new
subprocess regression compares the reference under default and disabled
AVX-512 dispatch: component geometry must be identical, and FCJ fields must
satisfy the existing scaled 2e-9 derivative envelope. CI records the minimum
NumPy job's CPU features and runs its normal full suite with SIMD enabled.
