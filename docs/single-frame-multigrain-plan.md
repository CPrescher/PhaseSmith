# Future milestone: single-frame multi-grain analysis

Extend the opt-in single-crystal workflow to discover and measure several grains
from one detector exposure, following the snapshot-analysis problem encountered
in serial crystallography. This is deferred work building on the rotation-series
position, prediction and integration infrastructure. It is not implemented by
the existing rotation grain search and does not change the current release scope.

The initial peak list supplies evidence for discovering grain orientations; it
must not define the final reflection population. After a grain is indexed,
predict and measure its reflections throughout the declared reciprocal-space
range and detector footprint, including initially undetected, weak, zero and
negative observations. Improved weak-spot discovery matters especially when a
weak grain would otherwise remain undiscovered: a single frame has no adjacent
rotation exposures from which to recover its orientation. For the current
rotation workflow, peak-finder replacement remains lower priority when the
existing reliable spots already establish the relevant lattices.

The future workflow should provide:

- **Peak discovery:** compare the current detector with isolated, pinned PF8
  and PF9 implementations. Evaluate background-derived detection noise,
  separate strong-seed and lower growth thresholds, component-level
  significance and size checks, and radial versus robust local background
  estimates. Multiscale matched filtering is a further candidate for diffuse
  weak spots. Control false detections rather than optimizing peak count alone;
  detection masks must not determine the final weak-reflection integration
  support. The preliminary weak-peak comparison motivates these experiments
  but does not establish single-frame grain recovery or final intensity quality.
- **Multiple-lattice discovery and assignment:** find competing orientations
  with explicit grain budgets, retaining unassigned spots and ambiguous
  ownership. Declare whether grains share a cell or may have distinct cells,
  and whether cell parameters are supplied or inferred. Sparse single-frame
  data may not identify an unconstrained cell; retain alternative hypotheses
  and report insufficient evidence rather than forcing a unique solution.
- **Joint positional refinement and prediction:** refine identifiable grain
  parameters against a shared calibrated detector model. Any detector refinement
  needs independent constraints and validation. Predict reflections for every
  accepted grain before intensity extraction, preserving assignment uncertainty.
- **Intensity extraction and overlaps:** fit signed intensities at predicted
  positions using fixed or strong-reference profiles and explicit background
  models. Avoid freely fitting weak-spot centres and widths to noise. Account
  for all grains in background exclusions and fit overlapping contributions
  jointly where identifiable; flag unresolved intensity partitions. Report
  measured finite-support spot counts separately from complete-reflection
  intensity estimates. A single exposure can be partial; any partiality
  correction, extrapolation or merging across snapshots needs its own declared
  physical model and validation.

Exit gates must prioritize correct grain recovery and false-grain rejection,
orientation/cell accuracy, reflection-assignment accuracy and ambiguity, and
prediction coverage. Include weak-grain/strong-grain mixtures, overlapping spots,
asymmetric backgrounds, masks and noise-only controls, with independent synthetic
truth and reserved measured examples. Test extraction of deliberately omitted
discovery peaks, signed weak/null intensity bias and uncertainty, finite-support
boundaries, deterministic results and realistic multi-grain runtime/memory.
More detected spots alone is not a promotion criterion. Production numerical
kernels remain Rust-owned with independent Python references and the existing
derivative, support and benchmark requirements.

Use [CrystFEL](https://www.desy.de/~twhite/crystfel/) and
[FDIP/PF9 (Gevorkov et al., 2024)](https://doi.org/10.3390/cryst14020164)
as methodological references and external comparators. Their GPL implementations
must remain outside the MIT production implementation; independently implement
documented equations and validate against pinned external outputs, without
introducing an upstream runtime dependency.
