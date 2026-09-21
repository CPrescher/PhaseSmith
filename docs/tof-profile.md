# Neutron time-of-flight profile

The TOF layer maps reflection d-spacing to a bin-center time coordinate and to
four peak-shape quantities, then convolves the normalized TCH pseudo-Voigt with
normalized back-to-back exponentials. Public coordinates and widths are
microseconds; d-spacing is ångströms; exponential rates are inverse
microseconds.

The physical profile family follows R. B. Von Dreele, J. D. Jorgensen and
C. G. Windsor, *J. Appl. Cryst.* **15** (1982), 581–589,
[doi:10.1107/S0021889882012722](https://doi.org/10.1107/S0021889882012722).
The modern calibration form and its parameter roles are independently stated
in equations 5–9 of A. Huq *et al.*, “POWGEN: rebuild of a third-generation
powder diffractometer at the Spallation Neutron Source,” *J. Appl. Cryst.*
**52** (2019), 1189–1201,
[doi:10.1107/S160057671901121X](https://doi.org/10.1107/S160057671901121X).
The TCH component-width transform is documented separately in
[`tch-profile.md`](tch-profile.md).

GSAS-II is not an equation source. Its pinned executable behavior validates the
coefficient translation, profile values, derivatives, and bin-coordinate
convention.

## Calibration and broadening equations

For d-spacing `d > 0`, the compatibility model is

```text
position = zero + difC d + difA d^2 + difB / d
alpha    = A / d
beta     = beta0 + beta1 / d^4 + betaq / d^2
sigma^2  = sigma0 + sigma1 d^2 + sigma2 d^4 + sigmaq d
g        = sqrt(8 ln 2) sqrt(sigma^2)
l        = Z + X d + Y d^2.
```

`g` and `l` are Gaussian and Lorentzian component FWHMs passed to the TCH
transform. The model requires positive `difC`, `alpha`, `beta`, and `sigma^2`,
plus non-negative `l`; invalid derived values are errors rather than clamps.

The `sigmaq d` term is an explicitly named GSAS-II compatibility convention.
Some published instrument models instead use a reciprocal-d term (for example
equation 7 of the POWGEN paper). A future published-law variant must therefore
be a distinct typed instrument model; `sigmaq_us2_per_angstrom` will never be
silently reinterpreted.

All derivatives of these equations with respect to d-spacing and all 15
instrument coefficients are evaluated analytically. Their stable global order
is

```text
zero, difc, difa, difb, alpha, beta0, beta1, betaq,
sigma0, sigma1, sigma2, sigmaq, x, y, z.
```

## Normalized asymmetric primitive

Let `q(u; g, l)` be the normalized TCH profile and let `s = x - position`.
The normalized back-to-back exponential density is

```text
b(v) = alpha beta / (alpha + beta) * exp(alpha v),  v < 0
     = alpha beta / (alpha + beta) * exp(-beta v), v >= 0.
```

The profile is the convolution

```text
p(s) = integral b(v) q(s - v; g, l) dv.
```

For deterministic finite tails, the implementation substitutes a dimensionless
exponential coordinate `t`, truncates it at `0 <= t <= L`, and renormalizes by
`1 - exp(-L)`. The defaults use `L = 20`. This gives left and right mixture
weights `beta/(alpha+beta)` and `alpha/(alpha+beta)` respectively.

The standalone primitive uses eight fixed panels of 48-point Gauss–Legendre
quadrature. In the fused finite-support kernel, four 48-point panels are mapped
onto each exact active integration interval rather than masking fixed nodes.
The independent Python reference uses NumPy-generated high-order Gauss–Legendre
nodes and does not call Rust.

Profile value plus derivatives with respect to position, alpha, beta, Gaussian
FWHM, and Lorentzian FWHM are accumulated from the same quadrature evaluations.
The fused reflection kernel chains those direct derivatives to d-spacing and
all instrument coefficients in the same sample pass.

## Finite support and bins

For transformed TCH FWHM `H`, `R = support_fwhm H`, and exponential cutoff
`L`, a reflection has observable support

```text
left  = position - R - L / alpha
right = position + R + L / beta.
```

Grid samples exactly on either boundary belong to the reflection's support
block. The TCH factor itself is limited to `abs(delta) <= R`. Analytical
derivatives hold the outer active sample block fixed, but include the moving
internal quadrature bounds, weights, and `R(H)` chain for position, rates, and
component widths. At an exact internal clamp boundary the derivative is defined
as zero; centered finite-difference tests therefore use samples away from those
non-smooth boundaries.

The public `tof_us` array always contains bin centers. GSAS-II can ingest TOF bin
boundaries with bin-width-multiplied Y and sigma. The typed SLOG FXYE reader
converts adjacent boundaries to arithmetic centers and divides Y and sigma by
the bin width, matching the density convention used by the profile kernel and
GSAS-II's scripting `getdata` arrays. The final boundary is not an output
sample; zero-intensity or zero-sigma bins are explicitly excluded. The `PNT`
fixture records and tests this translation explicitly.

The production input boundary accepts three reduced-data conventions:

- plain `tof_us, intensity_density[, sigma_density]` center columns;
- GSAS `SLOG ... FXYE` boundaries with width-multiplied Y and sigma; and
- GSAS `CONST ... STD` lower boundaries with packed integrated counts.

Both GSAS forms are converted to center/density arrays before the model sees
them. This conversion is an I/O concern, not a POWGEN assumption.

## Local derivatives and oracle scope

Local support-block columns are integrated intensity and d-spacing. Shared
instrument rows are dense. This is the same `AccumulationResult` contract as
constant-wavelength calculations, so refinement code does not need a TOF-only
Jacobian representation.

The pinned `tof_v1` fixture contains public `X`, `Ycalc`, background, and an
18-column TOF reflection list. Private probes are limited to selected
`getEpsVoigt` values and derivatives. The production code is independently
implemented from the equations above and compared numerically; GSAS-II is not a
runtime dependency and no GSAS-II implementation code is copied.

## Native pattern and Le Bail workflow

`phasesmith-workflows` owns a separate fixed-instrument TOF path built around
`TofPatternRecord`, `TofLeBailPhase`, and `TofLeBailInput`. It does not cast a
microsecond grid into `PatternRecord.x_deg`. Reflections retain d-spacing as
their durable coordinate, and each calculation returns the core accumulation
containing intensity/d-spacing support columns and all 15 dense instrument
rows.

`refine_tof_lebail` performs nonnegative multiplicative redistribution on the
actual, potentially nonuniform, bin-center grid. Trapezoidal cell widths enter
the discrete extraction sums; masks and optional one-sigma uncertainties are
applied locally. The instrument and reflection topology remain fixed during
this workflow.

An optional `TofChebyshevBackground` is a refinable residual added on top of
the fixed pattern background. A Smooth Bruckner estimate can therefore remain
the broad, non-differentiable baseline while the Chebyshev terms absorb smooth
low-order mismatch. On the explicit microsecond domain `[lower, upper]`, define
`u = 2 (tof - lower) / (upper - lower) - 1`; the background is
`sum_k c_k T_k(u)`. Its exact coefficient derivatives are `T_k(u)`. After each
nonnegative intensity redistribution, the coefficients are updated by a
deterministic weighted linear least-squares solve using the same mask and
uncertainty convention as the residual. The final calculation returns the
sample-major basis and combined fixed-plus-residual background. Omitting the model preserves the
previous fixed-background behavior. This is a complete Le Bail
intensity/background extraction capability, not a claim of TOF structural
Rietveld refinement.

Legacy GSAS type-1 and type-3 instrument files are translated by the bounded
`phasesmith-io` adapter used by applications and validation. Its `ICONS` fields
map to `difC`, `difA`, `zero`, and an unused value;
legacy records do not supply `difB`. This potentially ambiguous mapping is
verified through the pinned GSAS-II bank import. For profile function 3, the
first three `PRCF11` values map to
`alpha`, `beta0`, and `beta1`; the first two `PRCF12` values map to `sigma1`
and `sigma2`. Unsupported coefficients are explicit zeros. Legacy records stop
at this adapter and do not enter the core instrument model.

When present, the independently documented second `BNKPAR` field is the
nominal detector-bank scattering angle (the `TTHETA` field in the
[GSAS manual record specification](https://subversion.xray.aps.anl.gov/EXPGUI/gsas/all/GSAS%20Manual.pdf)).
The adapter returns it as a validated
`TofBankGeometry(two_theta_deg=...)` alongside the profile calibration. This
geometry is deliberately separate from `TofInstrument`: an empirical `DIFC`
does not uniquely supply a physical angle. Old profile-only files without
`BNKPAR` remain readable with `bank_geometry=None`; a structural TOF request
must instead require geometry explicitly. The checksum-pinned POWGEN and LANL
acceptance datasets exercise `90.000°` and `88.05°`, respectively.

For profile function 1, the first value in each four-value legacy group is
unused by the supported law: `PRCF 1` values 2--4 map to
`alpha`/`beta0`/`beta1`, and `PRCF 2` values 2--3 map to `sigma1`/`sigma2`.
The checksum-pinned LANL nickel tutorial exercises this translation and packed
constant-step input independently of POWGEN.

When a legacy bank explicitly supplies `I ITYP 4` and all twelve `ICOFF`
coefficients, the adapter also returns a public `TofIncidentSpectrum`. This
record uses a microsecond validity interval, evaluates the published
Maxwellian/Chebyshev intensity and analytical TOF derivative through an
independent NumPy implementation, and can normalize a `TofPowderPattern` by
dividing observations, uncertainties, and any supplied background by the same
positive spectrum. Absent/type-0 records return `None`; unsupported nonzero
functions are rejected. Other beamline adapters can construct the identical
public record without using a GSAS file.

Facility-neutral means a beamline can provide reduced center/density arrays,
the typed 15 coefficients, `TofBankGeometry`, and an optional explicit
`TofIncidentSpectrum` without adopting GSAS filenames. It does not mean
that specialized tails, tabulated resolution functions, or every historical
profile function are silently approximated by this model.

## Runtime, cancellation, and continuation

The fixed-instrument workflow uses the common native refinement runtime.
Cancellation is cooperative and checked before cycles and model evaluations.
If it arrives after a candidate has begun but before acceptance, that candidate
is discarded and the result contains the previous complete accepted state.
Each accepted checkpoint contains phase intensities, optional Chebyshev
coefficients, and the complete deterministic history. A checkpoint may resume
only against the same phase IDs, reflection topology, d-spacings, phase scales,
background identity/domain/order, and a total cycle budget at least as large as
its completed count.

Rust hosts use `refine_tof_lebail_with_runtime` to attach event and checkpoint
sinks. Python callers pass `cancellation=`, `checkpoint=`, and `progress=` to
`refine_tof_lebail`. Progress callbacks receive plain start, accepted-iteration,
and termination records; Python is never invoked from the peak/sample pass.
Native project format 3 persists this checkpoint through an explicit
microsecond histogram/experiment variant. It stores reflection topology,
current intensities, the optional refinable background, options, and complete
accepted residual history in bounded typed NPZ arrays; no TOF coordinate is
placed in the angle-domain histogram record. Rust hosts use
`save_tof_lebail_project` and `load_tof_lebail_project` with a validated
`TofLeBailProjectState`.

Two or more detector banks use the separate atomic contract documented in
[`tof-multibank.md`](tof-multibank.md). It shares exact fixed-cell reflection
geometry, retains experimental/intensity state per bank, computes aggregate
metrics from all included observations, and checkpoints the complete bank set.

Rust lattice adapters use `tof_lattice_geometry` to evaluate the shared-cell
chain without finite-differencing the profile. For each reflection,

```text
tof(d) = zero + difC d + difA d^2 + difB / d
d(tof)/dp = (difC + 2 difA d - difB / d^2) d(d)/dp
```

`d(d)/dp` comes from the analytical reciprocal-metric derivative and the
setting-aware independent-cell Jacobian. The result exposes both d-spacing and
TOF-position derivatives in reflection-major, independent-parameter order.
The bank calibration appears only in the second chain, so the same cell and
d-spacing derivative are reusable across detector banks.

The public Python path keeps the same unit boundary:

```python
from phasesmith.refinement.tof_lebail import TofLeBailInput, refine_tof_lebail

request = TofLeBailInput.from_files(
    "PG3_17541.gsa",
    "PGHR_60-2015A.prm",
    "LaB6.cif",
    bank=2,
)
result = refine_tof_lebail(request)
print(result.metrics.rwp)
```

`TofLeBailInput.from_files` generates fixed-cell reflection families from the
CIF and attaches an optional refinable Chebyshev residual to the fixed supplied
background. It does not refine the lattice, instrument coefficients, atomic
structure, or multiple detector banks. Cooperative cancellation/progress and
the Python-free native persistence boundary are separate completed layers
around this first public application slice. Rust hosts can use the separate
[shared multi-bank lattice workflow](tof-lattice-refinement.md); its Python
facade and project persistence are follow-on work.
