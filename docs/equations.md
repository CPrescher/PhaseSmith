# Symmetric pseudo-Voigt equations

Let \(\Delta=x-\mu\), let \(H>0\) be the full width at half maximum, and let
\(0\leq\eta\leq1\). The normalized components are

\[
\begin{aligned}
G(\Delta,H)
  &= \frac{\sqrt{4\ln 2/\pi}}{H}
     \exp\!\left(-4\ln 2\frac{\Delta^2}{H^2}\right), \\
L(\Delta,H)
  &= \frac{2}{\pi H}
     \frac{1}{1+4\Delta^2/H^2}, \\
p(\Delta,H,\eta)
  &= \eta L(\Delta,H)+(1-\eta)G(\Delta,H).
\end{aligned}
\]

Both \(G\) and \(L\) have unit integral over the real line and reach half their
central height at \(\lvert\Delta\rvert=H/2\). Their mixture therefore has
FWHM \(H\).

With \(a=4\ln2\), \(z=\Delta/H\), and \(q=1+4z^2\), the derivatives are

\[
\begin{aligned}
\frac{\partial G}{\partial\Delta}
  &= -\frac{2a\Delta}{H^2}G,
&
\frac{\partial L}{\partial\Delta}
  &= -\frac{8\Delta}{H^2q}L, \\
\frac{\partial G}{\partial H}
  &= \frac{G}{H}\left(-1+2az^2\right),
&
\frac{\partial L}{\partial H}
  &= \frac{L}{H}\left(-1+\frac{8z^2}{q}\right), \\
\frac{\partial p}{\partial\eta} &= L-G.
\end{aligned}
\]

For a peak contribution \(y=I\,p(x-\mu,H,\eta)\), the parameter derivatives
are

\[
\begin{aligned}
\frac{\partial y}{\partial I} &= p, \\
\frac{\partial y}{\partial\mu}
  &= -I\frac{\partial p}{\partial\Delta}, \\
\frac{\partial y}{\partial H}
  &= I\frac{\partial p}{\partial H}, \\
\frac{\partial y}{\partial\eta} &= I(L-G).
\end{aligned}
\]

Evaluation is restricted to
\(\lvert\Delta\rvert\leq \mathtt{support\_fwhm}\,H\). This deliberately
truncates the normalized infinite-domain function without renormalizing it.
Consequently, \(I\) denotes the infinite-support integrated intensity,
while the sampled pattern contains the analytically predictable in-window
fraction. The derivative treats the selected samples as fixed; distributional
derivatives at the moving cutoff are outside the API contract.

## Derivative storage

The calculated pattern remains dense. Per-peak derivatives are stored only for
the inclusive active interval found by two binary searches. For peak `p`,
`starts[p]` is its first active sample and `offsets[p]:offsets[p + 1]` selects
sample-major rows in the order `(intensity, position, H, eta)`. Empty and
entirely out-of-grid supports have zero-length blocks. Dense
`(peak, parameter, sample)` storage is an explicit compatibility conversion.
