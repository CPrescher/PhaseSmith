# Quantitative phase analysis

## Public convention

PhaseSmith converts compatible refined phase scales into normalized crystalline
weight fractions using the Hill--Howard relation

\[
W_p = \frac{S_p (Z M V)_p}{\sum_i S_i (Z M V)_i},
\]

where `S` is the refined phase scale, `Z` is the number of formula units per
unit cell, `M` is the formula mass in grams per mole, and `V` is the unit-cell
volume in cubic angstroms. The common unit conversion cancels during
normalization.

The equation source is R. J. Hill and C. J. Howard, “Quantitative phase analysis
from neutron powder diffraction data using the Rietveld method,” *Journal of
Applied Crystallography* **20** (1987), 467–474,
DOI [`10.1107/S0021889887086199`](https://doi.org/10.1107/S0021889887086199).

`phasesmith.quantitative.weight_fractions_from_scale` is the NumPy-first API.
`QuantitativePhase` plus `quantitative_phase_analysis` adds durable phase labels
for scripts and reports. This calculation is deliberately outside the
refinement solver: it interprets final scales but does not alter them.

`quantitative_phase_analysis_with_covariance` propagates a compatible
phase-scale covariance analytically. For (k_i=(ZMV)_i) and
(C=\sum_i S_i k_i), its Jacobian is

\[
\frac{\partial W_i}{\partial S_j}
= \frac{\delta_{ij} k_i - W_i k_j}{C},
\]

and the returned fraction covariance is (J\,\mathrm{Cov}(S)\,J^T). The
independent Python and production Rust implementations are each checked
against centered scale differences.

## Preconditions and exclusions

All supplied scales must come from the same calculation and use identical
structure-factor, multiplicity, correction, and scale normalization
conventions. PhaseSmith's structural phase scale multiplies the integrated
`multiplicity * correction * |F|^2` contribution, which is compatible when all
phases use the same built-in scattering/intensity path.

The normalized result describes only the supplied crystalline phases. It does
not quantify amorphous or unidentified material unless the experimental design
adds and models a suitable internal standard. Propagated uncertainty describes
only the supplied scale covariance and crystallographic metadata are treated as
fixed; it is not a complete experimental uncertainty budget.
