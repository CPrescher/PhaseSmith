# Constant-wavelength U/V/W/X/Y broadening

The constant-wavelength model maps reflection position to Gaussian and
Lorentzian component widths, then applies the independently implemented TCH
transform in [`tch-profile.md`](tch-profile.md). The angular dependence follows
the conventional Caglioti/TCH form summarized, for example, in equation 53 of
Dinnebier and Scardi, *J. Appl. Cryst.* **54** (2021), 1811–1831,
[doi:10.1107/S1600576721009183](https://doi.org/10.1107/S1600576721009183).

## Public units and equations

All profile coordinates and component FWHMs are degrees `2theta`. Let

```text
theta = (two_theta / 2) * pi / 180
t = tan(theta)
s = sec(theta)

q = U t^2 + V t + W
g = sqrt(8 ln 2) sqrt(q)
l = X s + Y t.
```

`q` is Gaussian variance in degrees squared, so `U`, `V`, and `W` also have
degree-squared units. `X` and `Y` are in degrees. The public parameter names are
therefore `u_deg2`, `v_deg2`, `w_deg2`, `x_deg`, and `y_deg`.

GSAS-II stores `U/V/W` in centidegrees squared and `X/Y` in centidegrees. The
pinned reflection-list probe confirms

```text
sigma2_centideg2 = U_gsas t^2 + V_gsas t + W_gsas
gamma_centideg   = X_gsas s + Y_gsas t + sample terms.
```

The second relation's `X/Y` ordering differs from some publications. The
GSAS-II adapter converts `U/V/W` by `1e-4` and `X/Y` by `1e-2`; the Rust kernel
uses only the public degree convention. Sample-size and microstrain terms are
outside this unit.

The model requires `0 < two_theta < 180`, `q > 0`, and `l >= 0`. Invalid
derived widths are errors; they are not silently clamped. Wavelength is stored
and validated in the typed instrument model for later reflection generation,
but does not enter profile evaluation when `two_theta` is already supplied.

## Analytical derivatives

With `c = sqrt(8 ln 2)` and `k = pi/360`,

```text
dg/dq = c / (2 sqrt(q))

dg/dU = dg/dq t^2
dg/dV = dg/dq t
dg/dW = dg/dq

dl/dX = s
dl/dY = t

dt/d(two_theta) = k s^2
ds/d(two_theta) = k s t
dq/d(two_theta) = (2 U t + V) dt/d(two_theta)
dg/d(two_theta) = dg/dq dq/d(two_theta)
dl/d(two_theta) = X ds/d(two_theta) + Y dt/d(two_theta).
```

For a reflection contribution `I p(x - position, g(position), l(position))`,
the local position derivative includes translation and width variation:

```text
d/dposition = I [-dp/ddelta
                 + dp/dg dg/dposition
                 + dp/dl dl/dposition].
```

The local support Jacobian stores intensity and position columns. Contributions
for `U`, `V`, `W`, `X`, and `Y` are accumulated directly into five dense global
rows in the same peak/sample pass.
