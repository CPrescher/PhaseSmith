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

The IUCr QARR 1g case is deliberately a readiness check. It verifies the real
5–150 degree pattern and Cu K-alpha doublet metadata, then reports a blocked
structural-doublet capability. It does not fit a monochromatic approximation
and call that quantitative validation.

## QARR implementation sequence

The next numerical slice will make QARR executable in this order:

1. Extend the structural constant-wavelength experiment from one wavelength to
   typed `WavelengthComponents`. Each component gets its own Bragg position,
   scattering vector, X-ray scattering factors, Lorentz--polarization factor,
   and normalized source weight.
2. Fuse component contributions in Rust. Shared phase, site, scale, lattice,
   size, strain, preferred-orientation, and background derivatives accumulate
   across components in one structural JVP/VJP contract; no Python loop is
   introduced per reflection or sample.
3. Add analytical derivatives for refinable component wavelengths and ratios
   only after fixed, externally calibrated components pass. Test values and
   derivatives against the independent NumPy equations and centered finite
   differences away from support/topology boundaries.
4. Extend reflection-domain guards using the minimum and maximum component
   wavelengths, so accepted lattice/wavelength motion cannot silently add or
   remove an unguarded visible family.
5. Migrate persistence and `RietveldProject` records with a versioned radiation
   union. Old monochromatic records load as one-component spectra.
6. Construct the three QARR phases from the pinned CIFs, refine background and
   shared instrument terms in explicit stages, then refine structural phase
   scales. Structured events, cancellation, budgets, checkpoints, and last
   accepted-state behavior remain mandatory.
7. Convert final scales through the Hill--Howard layer using reviewed `Z`,
   formula masses, and refined cell volumes. Record covariance propagation as
   a separate result once scale covariance is available.
8. Freeze acceptance only after diagnostic review. The initial target is an
   absolute error no greater than two weight-percentage points for each of the
   three weighed phases, alongside pattern residuals, reflection diagnostics,
   and numerical stability checks. Tighter limits should be based on repeated
   runs and published round-robin dispersion, not selected post hoc.
9. Add an optional same-scope GSAS-II timing comparison in its isolated pinned
   environment. Validate values first and report startup, setup, calculation,
   and refinement timing scopes separately. Normal installation and CI remain
   independent of GSAS-II.

## Review gates

Every new real-data case must document the source URL, immutable revision where
available, hashes, redistribution decision, radiation, geometry, coordinate
units, included range, preprocessing, refined parameters, stopping reason, and
acceptance criteria. A failed or unsupported capability is reported as such;
golden results are never silently regenerated or relaxed.
