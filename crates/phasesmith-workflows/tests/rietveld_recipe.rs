//! Native explicit and intelligent staged Rietveld workflow contracts.

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

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
    LatticeBounds, ParameterKey, PolynomialBackground, RefinementEvent, RefinementEventKind,
    RefinementLimits, RietveldCalculationOptions, RietveldCovarianceOptions,
    RietveldGeneralCheckpoint, RietveldInput, RietveldInstrumentParameter,
    RietveldParameterSelection, RietveldPhase, RietveldRecipe, RietveldRecipeError,
    RietveldRecipeMode, RietveldRecipeSinks, RietveldRefinementOptions, RietveldStage,
    RietveldStructuralSelection, TerminationReason, calculate_rietveld_pattern,
    intelligent_rietveld_recipe, run_rietveld_recipe, run_rietveld_recipe_with_sinks,
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

fn phase(scale: f64) -> RietveldPhase {
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
        occupancy: vec![1.0],
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

fn request(
    pattern: PatternRecord,
    phase: RietveldPhase,
    zero_shift_deg: f64,
    background: Option<Vec<f64>>,
) -> RietveldInput {
    let position = MonochromaticPositionCorrection {
        zero_shift_deg,
        bragg_brentano_mm: None,
        debye_scherrer_micrometre: None,
    };
    if let Some(coefficients) = background {
        RietveldInput::new_with_background(
            pattern,
            instrument(),
            None,
            position,
            BackgroundModel::Polynomial(PolynomialBackground::new("main", coefficients).unwrap()),
            vec![phase],
        )
        .unwrap()
    } else {
        RietveldInput::new(pattern, instrument(), None, position, vec![phase]).unwrap()
    }
}

fn shifted_input(background: Option<(Vec<f64>, Vec<f64>)>) -> RietveldInput {
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
    let truth_background = background.as_ref().map(|value| value.1.clone());
    let truth = request(empty, phase(1.2), 0.03, truth_background);
    let observed = calculate_rietveld_pattern(&truth, &calculation()).unwrap();
    let starting_background = background.map(|value| value.0);
    request(
        PatternRecord::new(
            x_deg,
            Some(observed.y),
            truth.pattern.uncertainty,
            None,
            Some(truth.pattern.background_y),
        )
        .unwrap(),
        phase(0.8),
        -0.01,
        starting_background,
    )
}

fn maximum(background: bool) -> RietveldParameterSelection {
    RietveldParameterSelection::new(
        RietveldStructuralSelection {
            phase_scale: true,
            ..RietveldStructuralSelection::default()
        },
        vec![RietveldInstrumentParameter::ZeroShiftDeg],
        background,
        false,
    )
    .unwrap()
}

#[test]
fn intelligent_planner_is_cumulative_advice_and_matches_python_when_configured() {
    let input = shifted_input(None);
    let recipe =
        intelligent_rietveld_recipe(&input, &maximum(false), "intelligent-cumulative").unwrap();
    assert_eq!(recipe.mode(), RietveldRecipeMode::Intelligent);
    assert_eq!(
        recipe
            .stages()
            .iter()
            .map(RietveldStage::name)
            .collect::<Vec<_>>(),
        ["scale_background", "positions"]
    );
    assert!(recipe.stages()[0].selection().instrument.is_empty());
    assert_eq!(
        recipe.stages()[1].selection().instrument,
        [RietveldInstrumentParameter::ZeroShiftDeg]
    );
    assert!(recipe.planner_notes()[0].contains("advisory workflow orchestration"));

    let Ok(python) = std::env::var("PHASESMITH_NUMPY_PYTHON") else {
        return;
    };
    let script = r#"
from dataclasses import replace
import numpy as np
import phasesmith
from phasesmith.refinement import intelligent_rietveld_recipe, rietveld

cell = phasesmith.UnitCell(4.7, 4.7, 4.7, 90.0, 90.0, 90.0)
structure = phasesmith.CrystalStructure(
    "alpha", "P m -3 m", cell, phasesmith.space_group_by_number(221).space_group,
    (phasesmith.AtomSite("Si1", "Si1", "Si", "Si", (0, 0, 0), 1.0, 0.01),),
)
reflections = phasesmith.StructuralReflectionBatch(
    ("1,0,0",), ((1,0,0),), (6,),
)
phase = rietveld.RietveldPhase(
    "alpha", "Alpha", structure, reflections, phasesmith.XrayNonResonant(),
    phasesmith.NeutralIntegratedIntensityCorrection(), 0.8,
)
instrument = phasesmith.ConstantWavelengthInstrument(1.5406, 2e-4, -1e-4, 1.2e-4, 1.5e-3, 3e-3)
experiment = replace(phasesmith.ConstantWavelengthExperiment.x_ray(instrument), zero_shift_deg=-0.01)
pattern = phasesmith.PowderPattern(np.linspace(20, 30, 11), observed_y=np.zeros(11), uncertainty=np.ones(11))
selection = rietveld.RietveldParameterSelection(
    phase_scale=True, lattice=False, coordinates=False, occupancy=False, u_iso=False,
    instrument_parameters=("zero_shift_deg",), background=False,
)
parameters = rietveld.build_parameter_set((phase,), (None,), selection, experiment=experiment)
request = rietveld.RietveldInput(pattern, experiment, (phase,), (None,), parameters, (), selection)
recipe = intelligent_rietveld_recipe(request)
print(" ".join(stage.name for stage in recipe.stages))
print(" ".join(recipe.stages[1].selection.instrument_parameters))
"#;
    let output = Command::new(python)
        .args(["-c", script])
        .env(
            "PYTHONPATH",
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python"),
        )
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Python planner oracle failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "scale_background positions\nzero_shift_deg\n"
    );
}

#[test]
fn native_recipe_refines_scale_then_position_and_only_diagnoses_the_final_stage() {
    let input = shifted_input(None);
    let maximum = maximum(false);
    let recipe = intelligent_rietveld_recipe(&input, &maximum, "native-two-stage").unwrap();
    let workflow = run_rietveld_recipe(
        &input,
        &maximum,
        &[None],
        &[],
        &recipe,
        &options(20),
        RietveldCovarianceOptions::default(),
        None,
    )
    .unwrap();
    assert!(workflow.completed());
    assert_eq!(workflow.stages().len(), 2);
    assert!(workflow.stages()[0].result.jacobian_rank.is_none());
    assert!(workflow.stages()[1].result.jacobian_rank.is_some());
    assert_eq!(
        workflow.stages()[1].starting_rwp.to_bits(),
        workflow.stages()[0]
            .result
            .calculation
            .metrics
            .rwp
            .to_bits()
    );
    let final_result = workflow.final_result();
    assert!((final_result.input.phases[0].definition().scale - 1.2).abs() < 3.0e-8);
    assert!((final_result.input.position_correction.zero_shift_deg - 0.03).abs() < 3.0e-8);
    assert!(final_result.calculation.metrics.rwp < 2.0e-7);
    assert_eq!(
        workflow.last_accepted_stage().unwrap().result.input,
        final_result.input
    );
}

#[test]
fn recipe_forwards_structured_events_and_complete_checkpoints_across_stages() {
    let input = shifted_input(None);
    let maximum = maximum(false);
    let recipe = intelligent_rietveld_recipe(&input, &maximum, "observed").unwrap();
    let observed_events = Arc::new(Mutex::new(Vec::new()));
    let event_target = Arc::clone(&observed_events);
    let checkpoint_count = Arc::new(Mutex::new(0_usize));
    let checkpoint_target = Arc::clone(&checkpoint_count);
    let sinks = RietveldRecipeSinks::default()
        .with_event_sink(move |event: &RefinementEvent| {
            event_target
                .lock()
                .unwrap()
                .push((event.kind(), event.stage().to_owned()));
            Ok(())
        })
        .with_checkpoint_sink(move |_checkpoint: &RietveldGeneralCheckpoint| {
            *checkpoint_target.lock().unwrap() += 1;
            Ok(())
        });
    let workflow = run_rietveld_recipe_with_sinks(
        &input,
        &maximum,
        &[None],
        &[],
        &recipe,
        &options(20),
        RietveldCovarianceOptions::new(false, 64, 0.999).unwrap(),
        None,
        Some(&sinks),
    )
    .unwrap();
    assert!(workflow.completed());
    let events = observed_events.lock().unwrap();
    for stage in ["scale_background", "positions"] {
        assert!(events.contains(&(RefinementEventKind::Start, stage.to_owned())));
        assert!(events.contains(&(RefinementEventKind::Termination, stage.to_owned())));
    }
    assert!(*checkpoint_count.lock().unwrap() >= 2);
}

#[test]
fn recipe_rejects_unauthorized_families_and_invalid_metadata() {
    let input = shifted_input(None);
    let maximum = maximum(false);
    let unauthorized = RietveldParameterSelection::new(
        RietveldStructuralSelection {
            coordinates: true,
            ..RietveldStructuralSelection::default()
        },
        Vec::new(),
        false,
        false,
    )
    .unwrap();
    let stage = RietveldStage::new(
        "coordinates",
        unauthorized,
        vec!["Deliberately unauthorized.".to_owned()],
    )
    .unwrap();
    let recipe = RietveldRecipe::new(
        "unauthorized",
        vec![stage],
        RietveldRecipeMode::Explicit,
        Vec::new(),
    )
    .unwrap();
    assert!(matches!(
        run_rietveld_recipe(
            &input,
            &maximum,
            &[None],
            &[],
            &recipe,
            &options(4),
            RietveldCovarianceOptions::default(),
            None,
        ),
        Err(RietveldRecipeError::UnauthorizedSelection { .. })
    ));
    assert!(RietveldStage::new(" bad", maximum.clone(), vec!["Reason.".to_owned()]).is_err());
    assert!(
        RietveldRecipe::new(
            "empty",
            Vec::new(),
            RietveldRecipeMode::Explicit,
            Vec::new()
        )
        .is_err()
    );
}

#[test]
fn recipe_rejects_a_selected_constraint_without_its_source_family() {
    let input = shifted_input(Some((vec![0.8], vec![1.0])));
    let maximum = maximum(true);
    let background_only = RietveldParameterSelection::new(
        RietveldStructuralSelection::default(),
        Vec::new(),
        true,
        false,
    )
    .unwrap();
    let stage = RietveldStage::new(
        "background_only",
        background_only,
        vec!["Exercise dependency validation.".to_owned()],
    )
    .unwrap();
    let recipe = RietveldRecipe::new(
        "missing-source",
        vec![stage],
        RietveldRecipeMode::Explicit,
        Vec::new(),
    )
    .unwrap();
    let constraint = Constraint::Affine(
        AffineConstraint::new(
            ParameterKey::new("background", "main", "coefficient_0").unwrap(),
            ParameterKey::new("phase", "alpha", "scale").unwrap(),
            1.0,
            0.0,
        )
        .unwrap(),
    );
    assert!(matches!(
        run_rietveld_recipe(
            &input,
            &maximum,
            &[None],
            &[constraint],
            &recipe,
            &options(4),
            RietveldCovarianceOptions::default(),
            None,
        ),
        Err(RietveldRecipeError::MissingConstraintDependency { .. })
    ));
    let unsatisfied = Constraint::Fixed(
        FixedConstraint::new(ParameterKey::new("phase", "alpha", "scale").unwrap(), 1.0).unwrap(),
    );
    assert!(matches!(
        run_rietveld_recipe(
            &input,
            &maximum,
            &[None],
            &[unsatisfied],
            &recipe,
            &options(4),
            RietveldCovarianceOptions::default(),
            None,
        ),
        Err(RietveldRecipeError::UnsatisfiedConstraint { .. })
    ));
}

#[test]
fn rejected_or_cancelled_stage_does_not_become_an_accepted_state() {
    let input = shifted_input(None);
    let maximum = maximum(false);
    let stage = RietveldStage::new(
        "cancelled",
        maximum.clone(),
        vec!["Exercise safe workflow stop.".to_owned()],
    )
    .unwrap();
    let recipe = RietveldRecipe::new(
        "cancelled",
        vec![stage],
        RietveldRecipeMode::Explicit,
        Vec::new(),
    )
    .unwrap();
    let cancellation = CancellationToken::default();
    cancellation.request("test cancellation").unwrap();
    let workflow = run_rietveld_recipe(
        &input,
        &maximum,
        &[None],
        &[],
        &recipe,
        &options(4),
        RietveldCovarianceOptions::default(),
        Some(&cancellation),
    )
    .unwrap();
    assert!(!workflow.completed());
    assert_eq!(
        workflow.final_result().termination_reason,
        TerminationReason::Cancelled
    );
    assert!(workflow.last_accepted_stage().is_none());
    assert_eq!(workflow.final_result().input, input);
}

#[test]
fn planner_skips_empty_preparatory_stages_and_supports_evaluate_only() {
    let input = shifted_input(None);
    let coordinates = RietveldParameterSelection::new(
        RietveldStructuralSelection {
            coordinates: true,
            ..RietveldStructuralSelection::default()
        },
        Vec::new(),
        false,
        false,
    )
    .unwrap();
    let recipe = intelligent_rietveld_recipe(&input, &coordinates, "coordinates").unwrap();
    assert_eq!(recipe.stages().len(), 1);
    assert_eq!(recipe.stages()[0].name(), "structure");
    assert!(recipe.stages()[0].selection().structural.coordinates);

    let empty = RietveldParameterSelection::default();
    let recipe = intelligent_rietveld_recipe(&input, &empty, "evaluate").unwrap();
    assert_eq!(recipe.stages()[0].name(), "evaluate_only");
    let workflow = run_rietveld_recipe(
        &input,
        &empty,
        &[None::<LatticeBounds>],
        &[],
        &recipe,
        &options(2),
        RietveldCovarianceOptions::default(),
        None,
    )
    .unwrap();
    assert!(workflow.completed());
    assert_eq!(
        workflow.final_result().termination_reason,
        TerminationReason::Converged
    );
}
