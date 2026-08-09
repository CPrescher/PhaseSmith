# Real-data validation plan

## Purpose

Real patterns test boundaries that synthetic and scalar profile cases do not:
text-format rounding, background preprocessing, reflection coverage, realistic
overlap, refinement stability, phase-scale interpretation, and complete script
ergonomics. External data remains opt-in and checksum-pinned. No validation
workflow imports GSAS-II or makes it a PhaseSmith runtime dependency.

## Implemented checkpoint

The implemented checkpoint provides Rust-owned layers with Python adapters:

1. `phasesmith-io` reads columns, GSAS FXYE, and GSAS STD into validated owned
   records; Python reborrows those arrays as NumPy.
2. `phasesmith-validation` owns dataset provenance, byte sizes, SHA-256 hashes,
   stable reports, and all native real-data runners.
3. `phasesmith-workflows::quantitative_phase_analysis` implements the published
   Hill--Howard `S Z M V` conversion for compatible phase scales.
4. `phasesmith-validation` provides a standalone JSON CLI. Public Python calls
   delegate ordinary runs through PyO3 while independent reference/callback
   paths remain available.

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

The deterministic pure-Rust three-stage workflow first refines phase scales plus
shared U/V/W/zero, then adds isotropic displacement, size, microstrain, and
preferred orientation, and finally polishes the three linear phase scales with
nonlinear parameters fixed. The fixed-anisotropic reviewed result is:

- Al2O3 30.466%, ZnO 34.110%, CaF2 35.424%;
- maximum absolute weighed-fraction error 1.004 percentage points (limit 2);
- Poisson-weighted Rwp 0.19844 (limit 0.20);
- unit-weight Rwp 0.13181 (limit 0.15);
- background-subtracted profile correlation 0.99062 (limit 0.98).

The two residual gates are intentionally separate: assigning
`sigma=sqrt(max(counts, 1))` changes the weighting and must not be compared to a
unit-weight prototype number. Current approximations are recorded in the
machine-readable report: Al2O3 CIF tensors are evaluated directly and fixed,
sites lacking CIF displacement values start at `Uiso=0.005 Å²` and refine,
Cu K-alpha1 fixed dispersion is reused for K-alpha2, SH/L=0.002 uses the
documented equal-height FCJ mapping, and absorption is not yet active. The
reviewed release run ends safely by stagnation before the stage-2 evaluation
budget. One test
repeats the full native workflow exactly and another compares its scientific
measurements with the independent Python implementation.

PbSO4 now covers both probes in Rust: the monochromatic neutron runner and an
exact fixed-doublet X-ray runner retaining all 383 reflection families. The
joint example combines this spectrum with the neutron histogram, and the
analytical mixed-radiation objective has dedicated Rust coverage.

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
