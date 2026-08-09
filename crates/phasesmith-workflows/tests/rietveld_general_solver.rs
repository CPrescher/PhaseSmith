//! Constraint-aware complete native Rietveld solver contracts.

use std::path::PathBuf;
use std::process::Command;

use phasesmith_core::{ConstantWavelengthInstrument, OwnedCwContributions};
use phasesmith_crystallography::{IntegratedIntensityCorrectionModel, UnitCell};
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPhaseDefinition,
};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_io::space_group_by_number;
use phasesmith_model::{PatternRecord, RecordId};
use phasesmith_workflows::{
    AffineConstraint, BackgroundModel, CancellationToken, Constraint, FixedConstraint,
    LatticeBounds, LatticeParameterization, LatticeReflectionDomain, ParameterKey,
    PolynomialBackground, RefinementLimits, RietveldCalculationOptions, RietveldCovarianceOptions,
    RietveldGeneralRefinementError, RietveldInput, RietveldInstrumentParameter,
    RietveldParameterSelection, RietveldPhase, RietveldRefinementOptions,
    RietveldSamplePhysicsModel, RietveldStructuralSelection, TerminationReason,
    calculate_rietveld_pattern, refine_general_rietveld,
};

fn instrument() -> ConstantWavelengthInstrument {
    ConstantWavelengthInstrument {
        wavelength_angstrom: 1.5406,
        u_deg2: 2.0e-4,
        v_deg2: -1.0e-4,
        w_deg2: 1.2e-4,
        x_deg: 1.5e-3,
        y_deg: 3.0e-3,
    }
}

fn phase(scale: f64, occupancy: f64) -> RietveldPhase {
    let definition = StructuralPhaseDefinition {
        cell: UnitCell {
            a_angstrom: 4.7,
            b_angstrom: 4.7,
            c_angstrom: 4.7,
            alpha_deg: 90.0,
            beta_deg: 90.0,
            gamma_deg: 90.0,
        },
        space_group: space_group_by_number(221).unwrap().space_group,
        hkl: vec![[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0], [2, 1, 0]],
        multiplicity: vec![6, 12, 8, 6, 24],
        fractional_xyz: vec![[0.0, 0.0, 0.0]],
        occupancy: vec![occupancy],
        u_iso_angstrom2: vec![0.01],
        anisotropic_mask: vec![false],
        u_aniso_cif_angstrom2: vec![[0.0; 6]],
        scattering_species: vec!["Si".to_owned()],
        scattering_real_offset: Vec::new(),
        scattering_imag_offset: Vec::new(),
        scale,
        coordinate_tolerance: 1.0e-10,
        scattering_model: BuiltInScatteringModel::XrayNonResonant,
        correction_model: IntegratedIntensityCorrectionModel::Neutral,
    };
    RietveldPhase::new_with_site_ids(
        RecordId::new("alpha").unwrap(),
        "Alpha",
        vec![RecordId::new("Si1").unwrap()],
        definition.clone(),
        OwnedCwContributions::neutral(definition.hkl.len()),
    )
    .unwrap()
}

fn dynamic_phase(wavelength_angstrom: f64) -> (RietveldPhase, LatticeBounds) {
    let mut definition = phase(1.0, 1.0).definition().clone();
    definition.correction_model = IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp {
        wavelength_angstrom,
    };
    let parameterization =
        LatticeParameterization::new(definition.space_group.clone(), definition.cell).unwrap();
    let values = parameterization.values_from_cell(definition.cell).unwrap();
    let bounds = LatticeBounds::new(
        &parameterization,
        values.iter().map(|value| value - 0.1).collect(),
        values.iter().map(|value| value + 0.1).collect(),
    )
    .unwrap();
    let domain = LatticeReflectionDomain::new(
        parameterization,
        bounds.clone(),
        wavelength_angstrom,
        [20.0, 95.0],
        0.0,
        true,
        100_000,
        1.05,
    )
    .unwrap();
    let phase = RietveldPhase::from_lattice_domain(
        RecordId::new("alpha").unwrap(),
        "Alpha",
        vec![RecordId::new("Si1").unwrap()],
        definition,
        domain,
    )
    .unwrap();
    (phase, bounds)
}

fn calculation() -> RietveldCalculationOptions {
    RietveldCalculationOptions::new(20.0, true, ExecutionPolicy::new(Some(1), 2).unwrap()).unwrap()
}

fn options(max_iterations: usize) -> RietveldRefinementOptions {
    RietveldRefinementOptions::new(
        calculation(),
        RefinementLimits::new(max_iterations, 2_000, None, 20).unwrap(),
        1,
        1.0e-12,
        1.0e-10,
        1.0e-6,
        10.0,
        0.3,
        1.0e-10,
        40,
        1.0,
        10,
    )
    .unwrap()
}

fn request(pattern: PatternRecord, phase: RietveldPhase, coefficients: Vec<f64>) -> RietveldInput {
    RietveldInput::new_with_background(
        pattern,
        instrument(),
        None,
        MonochromaticPositionCorrection {
            zero_shift_deg: 0.0,
            bragg_brentano_mm: None,
            debye_scherrer_micrometre: None,
        },
        BackgroundModel::Polynomial(PolynomialBackground::new("main", coefficients).unwrap()),
        vec![phase],
    )
    .unwrap()
}

fn input_from_truth(
    starting_phase: RietveldPhase,
    starting_background: Vec<f64>,
    truth_phase: RietveldPhase,
    truth_background: Vec<f64>,
) -> RietveldInput {
    let x_deg = (0..3_001)
        .map(|index| 20.0 + f64::from(index) * 0.025)
        .collect::<Vec<_>>();
    let empty = PatternRecord::new(
        x_deg.clone(),
        Some(vec![0.0; x_deg.len()]),
        Some(vec![0.5; x_deg.len()]),
        None,
        Some(vec![0.0; x_deg.len()]),
    )
    .unwrap();
    let truth = request(empty, truth_phase, truth_background);
    let observed = calculate_rietveld_pattern(&truth, &calculation()).unwrap();
    request(
        PatternRecord::new(
            x_deg,
            Some(observed.y),
            truth.pattern.uncertainty,
            None,
            Some(truth.pattern.background_y),
        )
        .unwrap(),
        starting_phase,
        starting_background,
    )
}

fn selection(structural: RietveldStructuralSelection) -> RietveldParameterSelection {
    RietveldParameterSelection::new(structural, Vec::new(), true, false).unwrap()
}

fn background_key(index: usize) -> ParameterKey {
    ParameterKey::new("background", "main", format!("coefficient_{index}")).unwrap()
}

#[test]
fn affine_background_and_phase_scale_recover_with_physical_covariance() {
    let input = input_from_truth(
        phase(0.7, 1.0),
        vec![0.6, 0.2],
        phase(1.3, 1.0),
        vec![1.0, 0.4],
    );
    let selection = selection(RietveldStructuralSelection {
        phase_scale: true,
        ..RietveldStructuralSelection::default()
    });
    let constraint = Constraint::Affine(
        AffineConstraint::new(background_key(1), background_key(0), 0.5, -0.1).unwrap(),
    );
    let result = refine_general_rietveld(
        &input,
        &selection,
        &[None],
        std::slice::from_ref(&constraint),
        &options(10),
        RietveldCovarianceOptions::default(),
        None,
        None,
    )
    .unwrap();
    assert_eq!(result.termination_reason, TerminationReason::Converged);
    assert!((result.input.phases[0].definition().scale - 1.3).abs() < 2.0e-9);
    let coefficients = phasesmith_workflows::DifferentiableBackground::coefficients(
        result.input.background.as_ref().unwrap(),
    );
    assert!((coefficients[0] - 1.0).abs() < 2.0e-9);
    assert!((coefficients[1] - 0.4).abs() < 2.0e-9);
    assert_eq!(
        result.free_keys,
        [
            background_key(0),
            ParameterKey::new("phase", "alpha", "scale").unwrap()
        ]
    );
    assert_eq!(result.jacobian_rank, Some(2));
    let covariance = result.covariance.as_ref().unwrap();
    assert_eq!(covariance.size, 3);
    let source = covariance.values[0];
    assert!((covariance.values[4] - 0.25 * source).abs() < 1.0e-12 * source.abs().max(1.0));
    assert!((covariance.values[1] - 0.5 * source).abs() < 1.0e-12 * source.abs().max(1.0));
    assert!(result.unresolved_correlations.is_empty());
    for row in &result.history {
        assert!(row.rp.is_finite());
        assert!(row.rwp.is_finite());
        assert!((row.chi_square - 2.0 * row.objective).abs() < 1.0e-12);
        assert!(
            (row.reduced_chi_square - row.chi_square / 2_999.0).abs()
                < 1.0e-12 * row.reduced_chi_square.abs().max(1.0)
        );
    }

    let partial = refine_general_rietveld(
        &input,
        &selection,
        &[None],
        std::slice::from_ref(&constraint),
        &options(1),
        RietveldCovarianceOptions::default(),
        None,
        None,
    )
    .unwrap();
    let resumed = refine_general_rietveld(
        &input,
        &selection,
        &[None],
        &[constraint],
        &options(10),
        RietveldCovarianceOptions::default(),
        Some(&partial.checkpoint),
        None,
    )
    .unwrap();
    assert_eq!(resumed.input, result.input);
    assert_eq!(resumed.history, result.history);
}

#[test]
fn rank_deficiency_reports_correlations_without_a_misleading_inverse() {
    let input = input_from_truth(phase(1.0, 0.8), vec![0.0], phase(1.0, 0.8), vec![0.0]);
    let selection = selection(RietveldStructuralSelection {
        occupancy: true,
        phase_scale: true,
        ..RietveldStructuralSelection::default()
    });
    let result = refine_general_rietveld(
        &input,
        &selection,
        &[None],
        &[],
        &options(4),
        RietveldCovarianceOptions::new(true, 8, 0.999_999).unwrap(),
        None,
        None,
    )
    .unwrap();
    assert_eq!(result.termination_reason, TerminationReason::Converged);
    assert_eq!(result.jacobian_rank, Some(2));
    assert_eq!(result.free_keys.len(), 3);
    assert!(result.covariance.is_none());
    assert!(
        result
            .unresolved_correlations
            .iter()
            .any(|value| value.correlation.abs() >= 0.999_999)
    );
}

#[test]
fn value_dependent_parameter_metadata_does_not_invalidate_the_final_checkpoint() {
    let input = input_from_truth(phase(1.0, 0.6), vec![0.0], phase(1.0, 0.9), vec![0.0]);
    let selection = selection(RietveldStructuralSelection {
        occupancy: true,
        ..RietveldStructuralSelection::default()
    });
    let result = refine_general_rietveld(
        &input,
        &selection,
        &[None],
        &[],
        &options(8),
        RietveldCovarianceOptions::default(),
        None,
        None,
    )
    .unwrap();
    assert!(result.history.len() >= 2);
    assert!((result.input.phases[0].definition().occupancy[0] - 0.9).abs() < 2.0e-9);
    assert_eq!(result.checkpoint.input, result.input);
}

#[test]
fn selected_wavelength_updates_remain_restart_compatible() {
    let starting_wavelength = 1.5406;
    let truth_wavelength = 1.541;
    let (starting_phase, bounds) = dynamic_phase(starting_wavelength);
    let (truth_phase, _) = dynamic_phase(truth_wavelength);
    let x_deg = (0..3_001)
        .map(|index| 20.0 + f64::from(index) * 0.025)
        .collect::<Vec<_>>();
    let empty = PatternRecord::new(
        x_deg.clone(),
        Some(vec![0.0; x_deg.len()]),
        Some(vec![0.5; x_deg.len()]),
        None,
        Some(vec![0.0; x_deg.len()]),
    )
    .unwrap();
    let mut truth_instrument = instrument();
    truth_instrument.wavelength_angstrom = truth_wavelength;
    let truth = RietveldInput::new_with_background(
        empty,
        truth_instrument,
        None,
        MonochromaticPositionCorrection {
            zero_shift_deg: 0.0,
            bragg_brentano_mm: None,
            debye_scherrer_micrometre: None,
        },
        BackgroundModel::Polynomial(PolynomialBackground::new("main", vec![0.0]).unwrap()),
        vec![truth_phase],
    )
    .unwrap();
    let observed = calculate_rietveld_pattern(&truth, &calculation()).unwrap();
    let input = request(
        PatternRecord::new(
            x_deg,
            Some(observed.y),
            truth.pattern.uncertainty,
            None,
            Some(truth.pattern.background_y),
        )
        .unwrap(),
        starting_phase,
        vec![0.0],
    );
    let selection = RietveldParameterSelection::new(
        RietveldStructuralSelection::default(),
        vec![RietveldInstrumentParameter::WavelengthAngstrom],
        false,
        false,
    )
    .unwrap();
    let result = refine_general_rietveld(
        &input,
        &selection,
        &[Some(bounds)],
        &[],
        &options(8),
        RietveldCovarianceOptions::default(),
        None,
        None,
    )
    .unwrap();
    assert!((result.input.instrument.wavelength_angstrom - truth_wavelength).abs() < 2.0e-8);
    let domain_wavelength = result.input.phases[0]
        .reflection_domain()
        .unwrap()
        .wavelength_angstrom();
    assert!((domain_wavelength - result.input.instrument.wavelength_angstrom).abs() < f64::EPSILON);
}

#[test]
fn unsatisfied_constraints_and_changed_restart_contracts_are_rejected() {
    let input = input_from_truth(phase(0.8, 1.0), vec![0.5], phase(1.1, 1.0), vec![0.8]);
    let selection = selection(RietveldStructuralSelection {
        phase_scale: true,
        ..RietveldStructuralSelection::default()
    });
    let fixed = Constraint::Fixed(FixedConstraint::new(background_key(0), 0.8).unwrap());
    assert!(matches!(
        refine_general_rietveld(
            &input,
            &selection,
            &[None],
            &[fixed],
            &options(4),
            RietveldCovarianceOptions::default(),
            None,
            None,
        ),
        Err(RietveldGeneralRefinementError::UnsatisfiedConstraint { .. })
    ));

    let result = refine_general_rietveld(
        &input,
        &selection,
        &[None],
        &[],
        &options(1),
        RietveldCovarianceOptions::default(),
        None,
        None,
    )
    .unwrap();
    let changed = Constraint::Fixed(FixedConstraint::new(background_key(0), 0.5).unwrap());
    assert!(matches!(
        refine_general_rietveld(
            &input,
            &selection,
            &[None],
            &[changed],
            &options(4),
            RietveldCovarianceOptions::default(),
            Some(&result.checkpoint),
            None,
        ),
        Err(RietveldGeneralRefinementError::InvalidCheckpoint { .. }
            | RietveldGeneralRefinementError::UnsatisfiedConstraint { .. })
    ));
    let mut changed_request = input.clone();
    changed_request.instrument.u_deg2 = 3.0e-4;
    assert!(matches!(
        refine_general_rietveld(
            &changed_request,
            &selection,
            &[None],
            &[],
            &options(4),
            RietveldCovarianceOptions::default(),
            Some(&result.checkpoint),
            None,
        ),
        Err(RietveldGeneralRefinementError::InvalidCheckpoint { .. })
    ));
}

#[test]
fn instrument_and_sample_parameters_refine_through_the_complete_solver() {
    let x_deg = (0..3_501)
        .map(|index| 20.0 + f64::from(index) * 0.02)
        .collect::<Vec<_>>();
    let pattern = PatternRecord::new(
        x_deg.clone(),
        Some(vec![0.0; x_deg.len()]),
        Some(vec![0.5; x_deg.len()]),
        None,
        Some(vec![0.0; x_deg.len()]),
    )
    .unwrap();
    let mut truth = request(
        pattern,
        phase(1.0, 1.0).with_sample_physics(RietveldSamplePhysicsModel::IsotropicMicrostrain {
            rms_microstrain: 8.0e-4,
        }),
        vec![0.0],
    );
    truth.position_correction.zero_shift_deg = 0.018;
    let observed = calculate_rietveld_pattern(&truth, &calculation()).unwrap();
    let mut input = request(
        PatternRecord::new(
            x_deg,
            Some(observed.y),
            truth.pattern.uncertainty,
            None,
            Some(truth.pattern.background_y),
        )
        .unwrap(),
        phase(1.0, 1.0).with_sample_physics(RietveldSamplePhysicsModel::IsotropicMicrostrain {
            rms_microstrain: 5.0e-4,
        }),
        vec![0.0],
    );
    input.position_correction.zero_shift_deg = -0.01;
    let selection = RietveldParameterSelection::new(
        RietveldStructuralSelection::default(),
        vec![RietveldInstrumentParameter::ZeroShiftDeg],
        false,
        true,
    )
    .unwrap();
    let result = refine_general_rietveld(
        &input,
        &selection,
        &[None],
        &[],
        &options(12),
        RietveldCovarianceOptions::default(),
        None,
        None,
    )
    .unwrap();
    assert_eq!(result.termination_reason, TerminationReason::Converged);
    assert!((result.input.position_correction.zero_shift_deg - 0.018).abs() < 2.0e-9);
    let sample = result.input.phases[0]
        .sample_physics()
        .unwrap()
        .parameters()
        .unwrap();
    assert!((sample[0].value - 8.0e-4).abs() < 2.0e-10);
    assert!(result.calculation.metrics.rwp < 2.0e-8);
}

#[test]
fn cancellation_returns_the_unchanged_restartable_complete_state() {
    let input = input_from_truth(phase(0.7, 1.0), vec![0.4], phase(1.2, 1.0), vec![0.9]);
    let selection = selection(RietveldStructuralSelection {
        phase_scale: true,
        ..RietveldStructuralSelection::default()
    });
    let cancellation = CancellationToken::default();
    cancellation.request("test cancellation").unwrap();
    let result = refine_general_rietveld(
        &input,
        &selection,
        &[None],
        &[],
        &options(8),
        RietveldCovarianceOptions::default(),
        None,
        Some(cancellation),
    )
    .unwrap();
    assert_eq!(result.termination_reason, TerminationReason::Cancelled);
    assert!(result.history.is_empty());
    assert_eq!(result.input, input);
    assert_eq!(result.checkpoint.input, input);
    assert_eq!(result.checkpoint.completed_iterations, 0);
    assert!(RietveldCovarianceOptions::new(true, 0, 0.9).is_err());
}

#[test]
fn accepted_dense_trial_is_the_next_current_linearization() {
    let input = input_from_truth(phase(0.7, 1.0), vec![0.0], phase(1.3, 1.0), vec![0.0]);
    let selection = RietveldParameterSelection::new(
        RietveldStructuralSelection {
            phase_scale: true,
            ..RietveldStructuralSelection::default()
        },
        Vec::new(),
        false,
        false,
    )
    .unwrap();
    let options = RietveldRefinementOptions::new(
        calculation(),
        RefinementLimits::new(2, 4, None, 20).unwrap(),
        1,
        1.0e-12,
        1.0e-10,
        1.0e-6,
        10.0,
        0.3,
        1.0e-10,
        1,
        0.1,
        0,
    )
    .unwrap();
    let result = refine_general_rietveld(
        &input,
        &selection,
        &[None],
        &[],
        &options,
        RietveldCovarianceOptions::new(false, 1, 1.0).unwrap(),
        None,
        None,
    )
    .unwrap();

    // One initial fused linearization and two accepted trial linearizations.
    // Each accepted trial becomes the next current state, and the final
    // accepted calculation is returned without evaluating it again.
    assert_eq!(result.evaluations, 3);
    assert_eq!(result.history.len(), 2);
    assert!(result.input.phases[0].definition().scale > 0.7);
}

#[test]
#[allow(clippy::too_many_lines)]
fn complete_solver_matches_python_affine_history_when_configured() {
    let Ok(python) = std::env::var("PHASESMITH_NUMPY_PYTHON") else {
        return;
    };
    let input = input_from_truth(
        phase(0.7, 1.0),
        vec![0.6, 0.2],
        phase(1.3, 1.0),
        vec![1.0, 0.4],
    );
    let selection = selection(RietveldStructuralSelection {
        phase_scale: true,
        ..RietveldStructuralSelection::default()
    });
    let constraint = Constraint::Affine(
        AffineConstraint::new(background_key(1), background_key(0), 0.5, -0.1).unwrap(),
    );
    let native = refine_general_rietveld(
        &input,
        &selection,
        &[None],
        std::slice::from_ref(&constraint),
        &options(10),
        RietveldCovarianceOptions::new(false, 64, 1.0 - 1.0e-10).unwrap(),
        None,
        None,
    )
    .unwrap();
    let script = r#"
import numpy as np
import phasesmith
from phasesmith.refinement import AffineConstraint, PolynomialBackground
from phasesmith.refinement import rietveld

cell = phasesmith.UnitCell(4.7, 4.7, 4.7, 90.0, 90.0, 90.0)
group = phasesmith.space_group_by_number(221).space_group
structure = phasesmith.CrystalStructure(
    "alpha", "P m -3 m", cell, group,
    (phasesmith.AtomSite("Si1", "Si1", "Si", "Si", (0.0, 0.0, 0.0), 1.0, 0.01),),
)
reflections = phasesmith.StructuralReflectionBatch(
    ("1,0,0", "1,1,0", "1,1,1", "2,0,0", "2,1,0"),
    ((1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,1,0)),
    (6, 12, 8, 6, 24),
)
def phase(scale):
    return rietveld.RietveldPhase(
        "alpha", "Alpha", structure, reflections, phasesmith.XrayNonResonant(),
        phasesmith.NeutralIntegratedIntensityCorrection(), scale,
    )
instrument = phasesmith.ConstantWavelengthInstrument(1.5406, 2e-4, -1e-4, 1.2e-4, 1.5e-3, 3e-3)
experiment = phasesmith.ConstantWavelengthExperiment.x_ray(instrument)
x = 20.0 + np.arange(3001, dtype=np.float64) * 0.025
empty = phasesmith.PowderPattern(x, observed_y=np.zeros_like(x), uncertainty=np.full_like(x, 0.5))
truth_background = PolynomialBackground("main", (1.0, 0.4))
observed = rietveld.calculate(empty, experiment, (phase(1.3),), background=truth_background).y
pattern = phasesmith.PowderPattern(x, observed_y=observed, uncertainty=np.full_like(x, 0.5))
selected = rietveld.RietveldParameterSelection(
    phase_scale=True, lattice=False, coordinates=False, occupancy=False, u_iso=False,
    background=True,
)
starting_background = PolynomialBackground("main", (0.6, 0.2))
starting = (phase(0.7),)
parameters = rietveld.build_parameter_set(
    starting, (None,), selected, experiment=experiment, background=starting_background,
)
constraint = AffineConstraint(
    rietveld.background_parameter_key("main", 1),
    rietveld.background_parameter_key("main", 0),
    0.5, -0.1,
)
request = rietveld.RietveldInput(
    pattern, experiment, starting, (None,), parameters, (constraint,), selected, starting_background,
)
options = rietveld.RietveldOptions(
    limits=rietveld.RefinementLimits(max_iterations=10, max_evaluations=2000, max_consecutive_rejections=20),
    min_iterations=1, objective_tolerance=1e-12, parameter_tolerance=1e-10,
    initial_damping=1e-6, damping_increase=10.0, damping_decrease=0.3,
    cg_tolerance=1e-10, max_cg_iterations=40, max_scaled_parameter_step=1.0,
    max_backtracks=10, use_uncertainty=True, support_fwhm=20.0,
    estimate_covariance=False, execution=phasesmith.ExecutionPolicy(1, 2),
)
result = rietveld.refine(request, options)
print(format(result.background.coefficients[0], ".17g"), format(result.background.coefficients[1], ".17g"), format(result.phases[0].scale, ".17g"))
print(len(result.history))
print(" ".join(format(row.objective, ".17g") for row in result.history))
"#;
    let python_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python");
    let output = Command::new(python)
        .args(["-c", script])
        .env("PYTHONPATH", python_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Python complete Rietveld oracle failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    let final_values = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|value| value.parse::<f64>().unwrap())
        .collect::<Vec<_>>();
    let native_background = phasesmith_workflows::DifferentiableBackground::coefficients(
        native.input.background.as_ref().unwrap(),
    );
    assert!((native_background[0] - final_values[0]).abs() < 3.0e-9);
    assert!((native_background[1] - final_values[1]).abs() < 3.0e-9);
    assert!((native.input.phases[0].definition().scale - final_values[2]).abs() < 3.0e-9);
    let history_count = lines.next().unwrap().parse::<usize>().unwrap();
    let objectives = lines
        .next()
        .unwrap_or_default()
        .split_whitespace()
        .map(|value| value.parse::<f64>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(history_count, objectives.len());
    assert_eq!(native.history.len(), history_count);
    for (native, python) in native.history.iter().zip(objectives) {
        assert!((native.objective - python).abs() <= 2.0e-8 * python.abs().max(1.0));
    }
}
