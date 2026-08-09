# Calculations and refinement workflows

Choose the narrowest layer that matches the scientific task. Lower layers are
useful for custom algorithms; workflow types provide validated orchestration
for standard analyses.

## Profile-only calculation

Use [`crate::core`] when reflection positions and integrated intensities are
already known. It supplies symmetric TCH/pseudo-Voigt, CW U/V/W/X/Y, FCJ,
wavelength-component, neutron TOF, and smooth-background kernels.

## Structure-to-pattern calculation

Use [`crate::engine`] when starting from a unit cell, atom sites, symmetry, and
a radiation/scattering selection. Prepared types cache validated topology and
tables for repeated values/JVP/VJP evaluations:

- [`crate::engine::PreparedStructuralPhase`] prepares one structural phase;
- [`crate::engine::PreparedStructuralSpectrum`] adds wavelength components;
- [`crate::engine::PreparedStructuralMultiphase`] composes phases;
- [`crate::engine::PreparedStructuralModel`] is the reusable high-level model.

One-shot `calculate_*` functions are convenient for isolated evaluations.
Prepared objects are preferable inside optimizers and interactive applications.

## Le Bail extraction

[`crate::workflows::LeBailInput`] combines an observed
[`crate::model::PatternRecord`], an instrument, and one or more fixed or
dynamically generated reflection phases. The common sequence is:

1. construct and validate the input;
2. select [`crate::workflows::LeBailOptions`];
3. call [`crate::workflows::refine_lebail`] for a simple run, or
   [`crate::workflows::refine_lebail_with_runtime`] for cancellation, event,
   and checkpoint integration;
4. retain the returned result and checkpoint rather than parsing log text.

## Rietveld refinement

[`crate::workflows::RietveldInput`] owns the observed pattern, structural
phases, background, and calculation options. The workflow crate separates:

- calculation with [`crate::workflows::calculate_rietveld_pattern`];
- prepared objective products for optimizer integration;
- single-histogram structural refinement;
- general profile/background/lattice/structure refinement;
- joint multi-histogram refinement;
- staged recipes for application-facing workflows.

The runnable `joint_pbso4` example in the `phasesmith-workflows` package shows
a complete native X-ray/neutron workflow over the pinned validation dataset.

## Runtime controls

Long-running entry points have `_with_runtime` variants accepting
[`crate::workflows::RefinementRuntime`]. Runtime contracts provide explicit
limits, cancellation, progress events, monotonic timing, and checkpoint sinks.
They are synchronous and application-neutral: a GUI or async service decides
which worker/task mechanism invokes them.
