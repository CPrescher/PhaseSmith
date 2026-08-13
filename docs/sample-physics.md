# Sample broadening, preferred orientation, and provider contract

This document is the equation and convention ledger for implementation unit 5.
It separates reflection-batch physics from the intrinsic TCH peak primitive.
All provider outputs are evaluated before the native peak/sample loop and are
plain contiguous arrays.

## Composable quantities

For reflection `i`, a provider returns

```text
q_sample[i]       additive Gaussian variance in degree(2theta)^2
l_sample[i]       additive Lorentzian FWHM in degree(2theta)
m_intensity[i]    multiplicative integrated-intensity correction
```

The production kernel combines these with the instrument contributions as

```text
q_total = q_instrument + q_sample
g_total = sqrt(8 ln 2) sqrt(q_total)
l_total = l_instrument + l_sample
I_total = I_base m_intensity.
```

Gaussian variances add under convolution; Lorentzian FWHMs add. This is the
angular-space form of the Gaussian-quadrature and Lorentzian-linear composition
given in section 2, equations 5--6 of Beyer *et al.*, *Acta Cryst.* A **78**
(2022), 10--20, [doi:10.1107/S205327332101088X](https://doi.org/10.1107/S205327332101088X).
That paper also records the reciprocal-space constant size and linear strain
dependencies and their mapping to angular `sec(theta)` and `tan(theta)` terms.

Providers also return derivatives of all three quantities with respect to the
base `two_theta` position and every named provider parameter. The native kernel
therefore includes width variation and intensity-modifier variation in the
local position derivative without calling provider code inside the hot loop.

## Isotropic crystallite size

The first built-in size model assigns the Scherrer contribution to the
Lorentzian component:

```text
theta = two_theta pi / 360
D_A = 10 D_nm
l_size = (180/pi) K lambda_A / (D_A cos(theta)).
```

Here `D_nm` is the coherent-domain size, `K` is a fixed dimensionless shape
factor, and `lambda_A` is the radiation wavelength in ångströms. The convention
uses FWHM in radians before conversion to degrees. The source equation is the
Scherrer relation as restated in section 3, equation 6 of Singh *et al.*,
*J. Appl. Cryst.* **41** (2008), 1057--1064,
[doi:10.1107/S0021889808031766](https://doi.org/10.1107/S0021889808031766).
The provider deliberately calls this a coherent-domain size, not particle or
grain size. `D_nm = infinity` is the exact disabled limit.

The analytical chains are

```text
dl_size/dD_nm = -l_size/D_nm
dl_size/d(two_theta) = l_size (pi/360) tan(theta).
```

## Isotropic RMS microstrain

The first strain model is explicitly Gaussian. Let `epsilon` be the standard
deviation of `delta d/d`. Differentiating Bragg's law gives the standard
deviation of `2theta` in radians as `2 epsilon tan(theta)`. Consequently

```text
q_strain = [(180/pi) 2 epsilon tan(theta)]^2.
```

This definition avoids the common ambiguity between FWHM strain, integral
breadth, and RMS strain. Its Gaussian FWHM is
`(180/pi) 4 sqrt(2 ln 2) epsilon tan(theta)`. Zero microstrain is the exact
disabled limit. The first-order `tan(theta)` dependency follows the discussion
in section 2 of Beyer *et al.* cited above and the differentiated Bragg relation
in Singh *et al.* equation 7.

The analytical chains are evaluated without division by `epsilon`:

```text
dq_strain/depsilon = 2 C epsilon tan(theta)^2
dq_strain/d(two_theta) = 2 C epsilon^2 tan(theta) sec(theta)^2 (pi/360)
C = [2 (180/pi)]^2.
```

## Isotropic Lorentzian microstrain

The second strain model is explicitly Lorentzian. Its dimensionless
`microstrain` parameter is the coefficient of the angular Lorentzian FWHM:

```text
l_strain = (180/pi) microstrain tan(theta)
dl_strain/dmicrostrain = (180/pi) tan(theta)
dl_strain/d(two_theta) = 0.5 microstrain sec(theta)^2.
```

This convention is separate from Gaussian RMS microstrain; the two parameter
values are not interchangeable. Lorentzian FWHMs add linearly, so this term is
fused with Scherrer size and instrument X/Y broadening in the same peak/sample
pass. Zero is the exact disabled limit. The `microstrain` value maps to the
GSAS-II isotropic `Mustrain` record as `microstrain = Mustrain * 1e-6`; this
mapping is validated only by black-box arrays from the pinned oracle.

## Stephens orthorhombic anisotropic microstrain

`StephensOrthorhombicBroadening` is the first symmetry-specific production
slice of the Stephens model. It accepts only orthorhombic cells and the six
independent coefficients, in the fixed order
`(S400, S040, S004, S220, S202, S022)`, in ångström⁻⁴. For reflection
`(h,k,l)`, PhaseSmith defines the physical variance of the inverse squared
d-spacing as

```text
P_hkl = S400 h^4 + S040 k^4 + S004 l^4
      + S220 h^2 k^2 + S202 h^2 l^2 + S022 k^2 l^2
A_hkl = (180/pi)^2 d_hkl^4 tan(theta)^2
H_hkl = sqrt(8 ln(2) A_hkl P_hkl).
```

Thus `A_hkl P_hkl` is the angular variance in degree(2theta)² and `H_hkl`
is its Gaussian-equivalent FWHM. This follows the Stephens construction of
the variance of `1/d_hkl²` as a symmetry-constrained fourth-order polynomial
(P. W. Stephens, *J. Appl. Cryst.* **32** (1999), 281–289,
[doi:10.1107/S0021889898006001](https://doi.org/10.1107/S0021889898006001)).
Recent IUCr discussion emphasizes that program coefficient normalizations are
not interchangeable, which is why PhaseSmith exposes physical ångström⁻⁴
coefficients rather than adopting a legacy program's stored numbers.

The explicit `lorentzian_fraction = eta` is a PhaseSmith profile-composition
contract, not an additional Stephens coefficient:

```text
q_stephens = (1-eta)^2 A_hkl P_hkl
l_stephens = eta H_hkl,                0 <= eta <= 1.
```

`eta=0` is purely Gaussian; `eta=1` is purely Lorentzian. Coefficients may be
signed, because symmetry polynomials can contain cancelling terms, but every
active reflection must have `P_hkl >= 0`. PhaseSmith rejects a materially
negative reflection variance. It clamps only roundoff-sized negative values;
positive Lorentzian mixing also requires strictly positive active variances so
the coefficient derivative of the square root remains defined.

Values and all chains are evaluated together. For basis monomial `B_j`,

```text
dq/dS_j = (1-eta)^2 A B_j
dl/dS_j = l B_j / (2 P_hkl)
dq/deta = -2(1-eta) A P_hkl
dl/deta = H_hkl
d log(A)/d(two_theta_deg) = (pi/180) / [sin(theta) cos(theta)]
dq/dcell_p = 4 q (dd/dcell_p) / d
dl/dcell_p = 2 l (dd/dcell_p) / d.
```

The structural position chain supplies the remaining Bragg-angle dependence
during lattice refinement. Synthetic tests check the Python and Rust values,
all coefficient/mixing derivatives, position derivatives, direct cell chains,
and the combined structural lattice chain by centered finite differences away
from support boundaries. Other Laue classes remain future, separately reviewed
slices; this class must not be used as a generic triclinic polynomial.

## March--Dollase preferred orientation

For preferred reciprocal-lattice direction `a`, reflection vector `h`, and
reciprocal metric `G*`, define

```text
c = |h^T G* a| / sqrt[(h^T G* h)(a^T G* a)]
A = r^2 c^2 + (1-c^2)/r
M = A^(-3/2).
```

`M` multiplies the integrated reflection intensity. `r > 0` is the March ratio;
`r = 1` is the exact random-orientation limit. This is the axially symmetric
March distribution advocated by Dollase, *J. Appl. Cryst.* **19** (1986),
267--272, [doi:10.1107/S0021889886089458](https://doi.org/10.1107/S0021889886089458).
The reciprocal-metric expression is an independent coordinate derivation of
the angle in that equation.

```text
dM/dr = -(3/2) A^(-5/2) [2 r c^2 - (1-c^2)/r^2].
```

The axis and reciprocal metric are fixed geometry in this milestone, not
refinement parameters. Their angle calculation is separately checked by finite
differences and symmetry-equivalent reflection tests.

## Provider API version 1

A provider is passed explicitly to a calculation and is called once for a
typed reflection batch. It returns the three arrays above, position chains,
stable parameter names, and parameter-major derivative arrays. Version 1 is a
Python protocol and plain-data schema, not a promise that Rust has a stable
dynamic ABI. Multiple providers compose by adding width contributions and
multiplying intensity modifiers with the full product rule.

Package entry-point discovery may later locate providers, but discovery never
changes the numerical core's behavior implicitly. A provider ID, provider
version, API version, and plain configuration are sufficient for persistence;
live Python or native objects are never serialized.

## Third-party broadening example

No subclassing or registration is required. A user model implements the
structural protocol and is passed explicitly as `physics=`. This example adds
the empirical Lorentzian law `L = strength / d` and exposes its analytical
parameter derivative:

```python
import numpy as np
from phasesmith import PhysicsContribution, ProviderDescriptor, calculate_cw_pattern


class ReciprocalDBroadening:
    descriptor = ProviderDescriptor("example.reciprocal-d", "1.0")

    def __init__(self, strength_deg_angstrom: float):
        self.strength = float(strength_deg_angstrom)

    def evaluate(self, context):
        d = context.reflections.d_spacing_angstrom
        count = d.size
        zeros = np.zeros(count)
        return PhysicsContribution(
            gaussian_variance_deg2=zeros,
            lorentzian_fwhm_deg=self.strength / d,
            intensity_multiplier=np.ones(count),
            d_gaussian_variance_d_position=zeros,
            d_lorentzian_fwhm_d_position=zeros,
            d_intensity_multiplier_d_position=zeros,
            parameter_names=("strength_deg_angstrom",),
            d_gaussian_variance_d_parameters=np.zeros((1, count)),
            d_lorentzian_fwhm_d_parameters=(1.0 / d)[None, :],
            d_intensity_multiplier_d_parameters=np.zeros((1, count)),
        )


result = calculate_cw_pattern(x, reflections, instrument, physics=ReciprocalDBroadening(0.004))
```

The provider executes once for the reflection batch. Its arrays and derivative
chains then enter the same Rust accumulator as built-in physics, so Le Bail and
later Rietveld orchestration can consume the resulting calculation without a
model-specific branch. A model that changes the intrinsic sampled line-shape
formula, rather than its widths or intensity, needs a compiled kernel provider
for production throughput; a NumPy implementation remains suitable as an
independent reference and prototype.
