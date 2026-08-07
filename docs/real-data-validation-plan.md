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

The IUCr QARR 1g case now runs the real 5–150 degree pattern, Cu K-alpha
doublet, three CIF structures, native structural values/JVP/VJP products, and
Hill--Howard QPA. No monochromatic approximation is substituted for the
doublet.

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

The QARR prerequisite review additionally identified and implemented two typed
structural-intensity inputs: caller-supplied fixed X-ray `f' + i f''` offsets
and polarized Bragg--Brentano LP with the instrument polarization mapped
directly to `P`. Both execute in the native value/JVP/VJP path. Persistence
format 7 stores these models and loads formats 1--6.

## Accepted QARR checkpoint

The deterministic three-stage workflow first refines phase scales plus shared
U/V/W/zero, then adds isotropic displacement, size, microstrain, and preferred
orientation, and finally polishes the three linear phase scales with nonlinear
parameters fixed. The reviewed result is:

- Al2O3 33.250%, ZnO 32.936%, CaF2 33.814%;
- maximum absolute weighed-fraction error 1.881 percentage points (limit 2);
- Poisson-weighted Rwp 0.19679 (limit 0.20);
- unit-weight Rwp 0.13282 (limit 0.15);
- background-subtracted profile correlation 0.99069 (limit 0.98).

The two residual gates are intentionally separate: assigning
`sigma=sqrt(max(counts, 1))` changes the weighting and must not be compared to a
unit-weight prototype number. Current approximations are recorded in the
machine-readable report: Al2O3 anisotropic displacement is replaced by
trace-mean Uiso, Cu K-alpha1 fixed dispersion is reused for K-alpha2, and the
supplied SH/L=0.002 FCJ asymmetry and absorption are not yet active.

## Remaining QARR sequence

1. Add covariance propagation for final phase fractions once the polished scale
   covariance is retained.
2. Add an optional same-scope GSAS-II timing comparison in its isolated pinned
   environment. Validate values first and report startup, setup, calculation,
   and refinement timing scopes separately. Normal installation and CI remain
   independent of GSAS-II.
3. After fixed-component real-data acceptance, implement refinable wavelength
   ratios and a multi-wavelength lattice guard. Move the component-level loop
   across the PyO3 boundary only if benchmark evidence justifies the ABI.

## Review gates

Every new real-data case must document the source URL, immutable revision where
available, hashes, redistribution decision, radiation, geometry, coordinate
units, included range, preprocessing, refined parameters, stopping reason, and
acceptance criteria. A failed or unsupported capability is reported as such;
golden results are never silently regenerated or relaxed.
