# Identifiability, termination, and scientific interpretation

Use this reference when staging a fit, reviewing an audit directory, or explaining
results. Keep execution status, numerical termination, and scientific acceptance
as separate conclusions. Report the evidence available for each.

## Identifiability before more iterations

Parameters are identifiable only to the extent that the observations and
independent constraints distinguish their effects. A bounds check, a smooth
profile, or a small residual does not establish this. Review at least:

- **Scale and occupancy:** both change structural intensity. Releasing both
  without a justified composition/occupancy model can trade one against the other.
- **Lattice and wavelength:** peak positions depend on their ratio through Bragg's
  law. An independent wavelength or lattice anchor is needed for absolute values.
- **Zero shift and specimen displacement:** their effects can be strongly
  correlated over the measured angular range; justify geometry and calibration.
- **Background and broad/weak peaks:** flexible background terms can absorb
  intensity and change composition estimates even while Rwp improves.
- **Instrument and sample widths:** Gaussian U and isotropic RMS strain share
  `tan(theta)^2` variance dependence. Instrument X and Scherrer size share
  `sec(theta)` Lorentzian dependence; Y and Lorentzian strain share `tan(theta)`.
  Multiple phases can constrain differences without fixing an absolute separation.
- **Structural and intensity corrections:** coordinates, displacement parameters,
  occupancy, orientation, and scale can compensate over limited or overlapping
  reflections. A larger authorized selection is not a reason to free everything
  before scale, background, and positions are established.

Inspect independent constraints, reported rank, unresolved correlations, bounds,
parameter movement, and model sensitivity together. A missing rank or covariance
means unavailable evidence, not zero uncertainty or proven singularity. Full rank
is a local numerical result; it does not certify model correctness or global
uniqueness. Covariance conditions on the chosen data weights, model, and constraints
and does not include unmodeled systematic uncertainty.

If the authorized selection is scientifically redundant, explain the needed
anchor or narrower selection to the project owner. External proposals cannot
change constraints or omit authorized terms from their final stage. Correct the
project/authorization and generate a new plan rather than hiding the problem in
an otherwise schema-valid recipe.

## Explicit empirical Gaussian convention

`EmpiricalGaussianConvention` is a CW preparation option when independent
instrument calibration is absent. The user chooses a reference phase and a
positive RMS-strain anchor. For phase p,

\[
\begin{aligned}
q_p(\theta) &= (U+C\epsilon_p^2)\tan^2\theta + V\tan\theta + W, \\
C &= (2\cdot180/\pi)^2.
\end{aligned}
\]

For reference r and chosen anchor a, the transformation is
\(U' = U + C(\epsilon_r^2-a^2)\) and
\(\epsilon_p'^2 = \epsilon_p^2-\epsilon_r^2+a^2\). It preserves initial total
Gaussian widths while fixing the reference strain in later stages. It removes
this one redundant direction, not every parameter correlation.

Each phase must have the supported built-in isotropic Gaussian strain provider;
negative transformed strain variance, invalid instrument widths, or conflicting
settings/constraints are errors. Do not clip variances to force acceptance.
Reference phase and anchor are explicit assumptions, not measured properties;
there is no universal reference material or anchor value to choose automatically.
The starting-profile identity does not guarantee identical later fits because
constraints and feasible parameter space change. Assess sensitivity to the anchor,
including phase fractions, under a separately authorized comparison.

Retain the convention in project/checkpoint/result provenance. Report Gaussian
instrument widths, sample strains, and their uncertainties as conditional on that
choice. Do not claim independently measured instrument response or absolute sample
microstrain from an empirical decomposition.

## Quantitative phase analysis

Raw phase scales are not mass fractions. `phasesmith.quantitative` converts
compatible scales using \(W_p=S_p(ZMV)_p/\sum_i S_i(ZMV)_i\), where Z is formula
units per cell, M is formula mass in g/mol, and V is cell volume in cubic angstroms.
All scales must come from the same calculation with compatible structure-factor,
multiplicity, correction, and scale normalization conventions. Verify Z, M, and V;
normalizing raw scales alone generally gives the wrong composition.

The fractions normalize only the supplied crystalline phases. They do not measure
amorphous or unidentified material; that requires an appropriate experimental
design with a suitable modeled internal standard. Propagated scale covariance
treats crystallographic metadata as fixed and remains conditional on the model;
it does not supply a complete experimental uncertainty budget.

## Three distinct outcomes

1. **Execution:** `completed` means the workflow reached an accepted final stage
   under its configured termination policy. CLI run status 0 represents accepted
   completion; status 3 is an auditable stopped/incomplete result; status 2 is a
   request/boundary failure. None certifies a scientifically acceptable fit.
2. **Optimization:** inspect every stage's termination and evaluation/iteration
   history, not only the last polish. Version-1 automation's safe accepted reasons
   include `converged` and `stagnated`, so completed execution can contain a
   non-converged stage. `converged` means the implemented stopping test was met;
   it is not a global-optimum or comprehensive stationarity certificate.
3. **Science:** assess whether the experiment, model, residuals, parameters, and
   independent evidence support the stated purpose. A deliberately bounded fit
   may be useful without convergence, but disclose that limitation and preserve
   its predeclared acceptance criteria. Never rename a stopped fit as converged.

`max_iterations`, `max_evaluations`, and `max_runtime` indicate exhausted budgets;
`cancelled` indicates a requested stop. `stagnated`, `repeated_rejections`,
`diverged`, or `numerical_failure` need investigation of the last accepted state,
parameter domain, and history. More budget is not an automatic remedy. A final
converged scale-only stage cannot erase earlier width/structural stagnation.
Recoverable stops retain accepted state; they do not install a failed trial.

## Read the correct report

`phasesmith review OUTPUT_DIRECTORY` produces a digest-bound workflow review whose
status is always `review_required`. It includes per-stage termination/Rwp changes,
scaled parameter motion, bound contacts, rank and correlations, and, when pattern
CSV was saved, lag-1 residual correlation and ten regional RMS summaries. Missing
CSV means residual-shape evidence is unavailable. No warnings does not mean an
automatically accepted fit. Its bound-contact rule uses a scale-dependent tolerance.

`project.fit_report()` or `build_fit_report(result, original_pattern)` produces the
separate `phasesmith.fit-report.v1` CW report. Supply the original coordinate grid;
the result does not retain grid identity. It preserves actual fit weights and masks,
ranks nonempty coordinate intervals by chi-square contribution, and reports a
Durbin--Watson statistic that never joins samples across excluded intervals. This
is different from the workflow review's lag-1 statistic and regional RMS summaries.
FitReport's active-bound advice means exact equality; it also explicitly discloses
unavailable rank/covariance and advises review for every non-converged termination.

Residuals are calculated minus observed. Weighted residuals are sigma units only
when the supplied uncertainty model warrants it; unit weights do not supply that
interpretation. Durbin--Watson with masks is descriptive, with no universal
acceptance threshold. Localized error identifies where to inspect curves, background,
and reflection positions. It cannot establish missing phases, a specific physical
cause, or the parameter expected to improve the fit: `attribution_available` is
false. The standalone `diagnose_residuals` also accepts TOF coordinates; this does
not make `FitReport` or the automation workflow a TOF refinement interface.

Compare Rwp only alongside weights, masks, data range, background, model freedom,
profile support, and relevant independent accuracy. Inspect composition and other
target quantities separately: lower Rwp can coexist with worse phase fractions.
Do not invent universal Rwp, correlation, or uncertainty cutoffs. Explain changes
in residual structure, bound contacts, empirical assumptions, and repeatability.
Use retained audit evidence to justify a specific next experiment or revised model;
any next execution requires its own plan and the skill's approval boundary.

## Further manual detail

Online development pages may differ from an installed release:

- [Fit evidence](https://phasesmith.readthedocs.io/en/latest/fit-report/)
- [Empirical Gaussian convention](https://phasesmith.readthedocs.io/en/latest/empirical-gaussian/)
- [Quantitative phase analysis](https://phasesmith.readthedocs.io/en/latest/quantitative-phase-analysis/)
- [Refinement runtime](https://phasesmith.readthedocs.io/en/latest/refinement-runtime/)
- [Structural refinement](https://phasesmith.readthedocs.io/en/latest/rietveld/)
