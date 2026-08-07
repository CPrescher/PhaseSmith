# Real-data validation plan

## Purpose

Real patterns test boundaries that synthetic and scalar profile cases do not:
text-format rounding, background preprocessing, reflection coverage, realistic
overlap, refinement stability, phase-scale interpretation, and complete script
ergonomics. External data remains opt-in and checksum-pinned. No validation
workflow imports GSAS-II or makes it a PhaseSmith runtime dependency.

## Implemented checkpoint

The first checkpoint adds four reusable layers:

1. `phasesmith.io.powder` reads ordinary two/three-column files and unpacked
   constant-wavelength GSAS FXYE banks into immutable NumPy arrays. FXYE
   centidegrees are converted explicitly to public degrees.
2. `phasesmith.validation.datasets` records stable dataset IDs, source and
   license notes, commit-pinned HTTPS locations, byte sizes, and SHA-256 hashes.
   Fetch and offline verification are separate explicit calls.
3. `phasesmith.quantitative` implements the published Hill--Howard `S Z M V`
   conversion for compatible phase scales.
4. `tools/validate_real_data.py` runs machine-readable checks and can write a
   finite JSON result without storing external inputs in the repository.

The official APS 11-BM sucrose tutorial is the currently supported complete
case: monochromatic FXYE input, native Smooth Bruckner background, P 21
reflection generation, non-negative Le Bail extraction, and analytical
U/V/W/X/Y refinement. The gate checks profile improvement, `Rwp <= 0.22`,
observed/calculated correlation of at least 0.98, and finite non-negative
integrated intensities. The threshold is a PhaseSmith regression gate for the
present model, not an equivalence tolerance against another program.

The original IUCr QARR 1g case was deliberately a readiness check. It verified
the real 5–150 degree pattern and Cu K-alpha doublet metadata, then reported a
blocked structural-doublet capability. That calculation-layer block is now
removed; a real three-phase fit and acceptance assessment is the next
checkpoint. No monochromatic approximation is substituted for the doublet.

## Implemented fixed-spectrum structural checkpoint

The fixed-spectrum prerequisite now provides:

1. Typed fixed `ComponentRadiation` with positive component intensities and a
   reference-wavelength profile instrument.
2. Component-specific Bragg positions, Lorentz--polarization values, source
   weights, and optional sample physics. Structure factors retain their
   wavelength-independent scattering-vector convention.
3. One fused native structural value/JVP/VJP batch per component. Python loops
   only over the small spectrum and never over reflections or samples.
4. Fixed-cell CIF reflection generation using the exact union visible in any
   component while keeping every shared family physical for every wavelength.
5. Fixed-spectrum Rietveld scale/site/profile/background/sample derivatives,
   component-indexed reports, and format-6 persistence with formats 1--5
   migration.
6. Explicit rejection of component-wavelength and guarded lattice refinement
   until their shared topology derivative contract is implemented.

## Remaining QARR execution sequence

1. Construct the three QARR phases from the pinned CIFs, use the calibrated Cu
   K-alpha1/K-alpha2 spectrum, and verify reflection coverage phase by phase.
2. Review and record each phase's `Z`, formula mass, and cell volume before any
   scale-to-weight conversion.
3. Establish the background and shared instrument state in explicit stages,
   then refine structural phase scales. Structured events, cancellation,
   budgets, checkpoints, and last-accepted-state behavior remain mandatory.
4. Inspect difference curves, phase contributions, scale correlations,
   termination reasons, and reflection diagnostics before calculating weight
   fractions.
5. Convert final scales through the Hill--Howard layer using the reviewed `Z`,
   formula masses, and cell volumes. Record covariance propagation separately
   once scale covariance is available.
6. Freeze acceptance only after diagnostic review. The initial target is an
   absolute error no greater than two weight-percentage points for each of the
   three weighed phases, alongside pattern residuals, reflection diagnostics,
   and numerical stability checks. Tighter limits should be based on repeated
   runs and published round-robin dispersion, not selected post hoc.
7. Add an optional same-scope GSAS-II timing comparison in its isolated pinned
   environment. Validate values first and report startup, setup, calculation,
   and refinement timing scopes separately. Normal installation and CI remain
   independent of GSAS-II.
8. After fixed-component real-data acceptance, implement refinable wavelength
   ratios and a multi-wavelength lattice guard. Move the component-level loop
   across the PyO3 boundary only if benchmark evidence justifies the ABI.

## Review gates

Every new real-data case must document the source URL, immutable revision where
available, hashes, redistribution decision, radiation, geometry, coordinate
units, included range, preprocessing, refined parameters, stopping reason, and
acceptance criteria. A failed or unsupported capability is reported as such;
golden results are never silently regenerated or relaxed.
