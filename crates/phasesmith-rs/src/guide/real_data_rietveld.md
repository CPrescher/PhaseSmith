# Real-data Rietveld walkthrough: `PbSO4`

This walkthrough follows the repository's complete Rust-only
`joint_pbso4` example. It refines the same `PbSO4` structure against a measured
constant-wavelength X-ray pattern and a measured neutron pattern. Structural
coordinates and displacement values are shared; instrument, profile,
background, correction, and phase scale remain histogram-local.

The canonical runnable source is
[`crates/phasesmith-workflows/examples/joint_pbso4.rs`](https://github.com/CPrescher/PhaseSmith/blob/main/crates/phasesmith-workflows/examples/joint_pbso4.rs).
From a repository checkout, run:

```text
cargo run --release -p phasesmith-workflows --example joint_pbso4 -- \
  validation/data/gsasii-pbso4-cw
```

The example is a measured-data regression workload, not a claim that every
legacy GSAS instrument or correction term is represented. Read its printed
sample count, termination, objective, residual metrics, and accepted cell as
acceptance evidence; elapsed time alone is not a refinement result.

A current one-worker run reports the following scientific summary (elapsed
time is intentionally omitted):

```text
samples=8378
iterations=40
evaluations=121
termination=max_iterations
initial_objective=4.499083933654e5
final_objective=4.129023112834e5
joint_rwp=0.2878307497
joint_rp=0.2365660813
cell_a_angstrom=8.4800000000
cell_b_angstrom=5.3980000000
cell_c_angstrom=6.9580000000
```

`max_iterations` is a bounded stop, not convergence. It is acceptable for this
regression command because the example separately gates objective improvement,
Rwp, sample count, and the fixed shared cell. A scientific application should
not relabel that termination as converged.

## Walkthrough 1: import and inspect before modelling

The directory contains one CIF plus packed GSAS X-ray and neutron powder
files. CIF and pattern ingestion are separate because a structure does not
define an experiment.

```ignore
let structure = read_cif_file(
    directory.join("PbSO4-Wyckoff.cif"),
    None,
    true,
    CifReadLimits::default(),
)?.structure;

let imported = read_powder_file(
    directory.join("PBSO4.XRA"),
    PowderFormat::GsasStd,
    1,
    PowderReadLimits::default(),
)?;
```

The example then selects the validated angular interval and copies the
matching x, observed intensity, and optional uncertainty rows. This is where a
real application would also apply an explicit mask. It estimates a smooth
Bruckner envelope and stores it as the pattern's fixed preprocessing
background. A small analytical Chebyshev residual background is added later
and may be refined. Keeping those two backgrounds separate prevents accidental
double counting.

Before proceeding in your own program, inspect:

- `structure.diagnostics`, even in strict mode;
- `structure.cell`, symmetry operation count, and independent site identities;
- powder `sample_count`, x range/order, uncertainty presence, and any reader
  provenance;
- whether cropping, masking, and preprocessing preserve aligned array lengths.

## Walkthrough 2: turn a CIF into a physical phase

A CIF has no reflection range. The example derives one from each histogram's
visible two-theta range and wavelength:

```ignore
let reflections = PreparedReflectionGenerator::new(
    structure.space_group.clone(),
    true,
    500_000,
)?.generate(
    structure.cell,
    ReflectionRange::CwTwoTheta {
        min_deg: angular_range[0],
        max_deg: angular_range[1],
        wavelength_angstrom,
    },
)?;
```

The `true` setting groups Friedel-related families and the final argument is an
explicit reflection-resource bound. The returned HKLs and multiplicities,
together with the CIF's independent atom sites, form an
[`crate::engine::StructuralPhaseDefinition`].

Probe choice is made here, not by the CIF reader. The X-ray histogram uses
[`crate::engine::BuiltInScatteringModel::XrayNonResonant`] and a polarized
Bragg--Brentano Lorentz-polarization correction. The neutron histogram uses
[`crate::engine::BuiltInScatteringModel::NeutronNuclear`] and the
constant-wavelength neutron Lorentz correction. Charge and isotope information
retained by CIF import is included in each scattering key when present.

The definition is wrapped in [`crate::workflows::RietveldPhase`] with stable
phase and site IDs. The X-ray phase also owns isotropic size and microstrain
models. These are typed sample-physics models and are recomputed after accepted
lattice/topology changes; they are not opaque arrays copied from a foreign
project file.

## Walkthrough 3: assemble and check each histogram

Each [`crate::workflows::RietveldInput`] owns:

- the observed [`crate::model::PatternRecord`];
- a [`crate::core::ConstantWavelengthInstrument`];
- optional wavelength components and FCJ axial geometry;
- an explicit position correction;
- an analytical residual background;
- the ordered structural phases.

The X-ray input uses the fixed 1.5405/1.5443 Å spectrum. The neutron input is
monochromatic at 1.909 Å. That distinction is part of the typed request and is
validated before calculation.

The example estimates a positive starting phase scale from a weighted linear
projection, installs it in a new validated phase definition, and then performs
a forward calculation with [`crate::workflows::calculate_rietveld_pattern`].
Do the forward calculation before optimization:

```ignore
let calculation = calculate_rietveld_pattern(
    &input,
    &RietveldCalculationOptions::new(30.0, true, execution.clone())?,
)?;

println!("initial Rwp = {}", calculation.metrics.rwp);
```

Check `profile_y`, `background_y`, total `y`, and each phase calculation on the
same grid. A solver cannot repair a wrong radiation model, wavelength,
background ownership decision, or phase identity.

## Walkthrough 4: select parameters deliberately

Each histogram supplies a [`crate::workflows::RietveldParameterSelection`]. In
this workload the structural selection releases coordinates, isotropic U, and
phase scale; it also releases the attached analytical background. Instrument
and lattice values remain fixed for this particular regression:

```ignore
let selection = RietveldParameterSelection::new(
    RietveldStructuralSelection {
        coordinates: true,
        u_iso: true,
        phase_scale: true,
        ..RietveldStructuralSelection::default()
    },
    Vec::new(), // no instrument parameters
    true,       // refine analytical background coefficients
    false,      // keep sample-physics parameters fixed
)?;
```

This record is authorization, not a suggestion. The solver cannot activate an
unselected family. Lattice bounds have one entry per phase; this example uses
`None`, matching its fixed lattice selection. Constraints, when present, use
stable [`crate::workflows::ParameterKey`] identities rather than positional
columns.

[`crate::workflows::JointRietveldHistogram`] adds a stable histogram ID and
calculation policy. The joint layout shares structural values only when phase
and site identities match. Histogram-local phase scales and backgrounds remain
distinct automatically.

## Walkthrough 5: refine and judge the accepted state

The joint solver minimizes the sum of both weighted objectives. Every trial is
calculated for both histograms and accepted only if the complete joint
objective decreases.

```ignore
let result = refine_joint_rietveld(
    &histograms,
    &[], // optional physical constraints
    options,
    None, // checkpoint for a new run
    None, // cooperative cancellation token
)?;
```

The returned object owns the accepted state. Do not reconstruct it by parsing
log messages. At minimum, review:

| Field | Acceptance question |
| --- | --- |
| `termination_reason` | Did it converge, or merely reach a budget/cancellation boundary? |
| `history` | Did every accepted objective decrease, and were steps/backtracks plausible? |
| `metrics.rwp`, `rp`, `chi_square` | Did the fit improve in the weighting convention actually requested? |
| `histograms` | Are the accepted physical models and local backgrounds sensible? |
| `parameters` / `free_keys` | Which physical values moved, and which degrees of freedom were solved? |
| `checkpoint` | Can the exact accepted contract be continued or persisted? |
| `evaluations` | Was the result obtained inside the declared work budget? |

The example additionally asserts that the objective decreased, both
histograms received the same shared cell, the expected number of samples was
included, Rwp stayed inside its regression gate, and the cell stayed near its
reference range. These are workload-specific checks. Your application should
add scientifically appropriate bounds, composition/site checks, difference
plots, correlation review, and independent validation data.

## Adapting this example

For one histogram, retain the import, phase construction, input, and forward
calculation steps, then use [`crate::workflows::refine_general_rietveld`]. It
adds complete instrument/background/sample/structural selection, constraints,
covariance diagnostics, and a typed checkpoint. For several histograms, use
the joint path when structural identity is genuinely shared; do not duplicate
one histogram merely to increase its statistical weight.

For another CIF, audit the supported input contract in
[`crate::guide::cif_inputs`]. For runtime callbacks, staged release, and
continuation, proceed to [`crate::guide::refinement_operations`].
