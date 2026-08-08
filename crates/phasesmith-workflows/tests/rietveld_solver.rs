//! Bounded native structural Rietveld solver contracts.

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
    CancellationToken, LatticeBounds, LatticeParameterization, RefinementEventKind,
    RefinementLimits, RefinementRuntime, RietveldCalculationOptions, RietveldInput, RietveldPhase,
    RietveldRefinementOptions, RietveldStructuralSelection, TerminationReason,
    calculate_rietveld_pattern, refine_rietveld, refine_rietveld_with_runtime,
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

fn phase(scale: f64, a_angstrom: f64) -> RietveldPhase {
    let definition = StructuralPhaseDefinition {
        cell: UnitCell {
            a_angstrom,
            b_angstrom: a_angstrom,
            c_angstrom: a_angstrom,
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

fn input_from_truth(starting: RietveldPhase, truth: RietveldPhase) -> RietveldInput {
    let x_deg = (0..4_001)
        .map(|index| 20.0 + f64::from(index) * 0.02)
        .collect::<Vec<_>>();
    let seed_pattern = PatternRecord::new(
        x_deg.clone(),
        Some(vec![0.0; x_deg.len()]),
        Some(vec![0.5; x_deg.len()]),
        None,
        Some(x_deg.iter().map(|value| 0.1 + value / 1_000.0).collect()),
    )
    .unwrap();
    let seed_input = request(seed_pattern, vec![truth]);
    let observed = calculate_rietveld_pattern(&seed_input, &calculation()).unwrap();
    request(
        PatternRecord::new(
            x_deg,
            Some(observed.y),
            seed_input.pattern.uncertainty,
            None,
            Some(seed_input.pattern.background_y),
        )
        .unwrap(),
        vec![starting],
    )
}

fn request(pattern: PatternRecord, phases: Vec<RietveldPhase>) -> RietveldInput {
    RietveldInput::new(
        pattern,
        instrument(),
        None,
        MonochromaticPositionCorrection {
            zero_shift_deg: 0.0,
            bragg_brentano_mm: None,
            debye_scherrer_micrometre: None,
        },
        phases,
    )
    .unwrap()
}

fn calculation() -> RietveldCalculationOptions {
    RietveldCalculationOptions::new(20.0, true, ExecutionPolicy::new(Some(1), 2).unwrap()).unwrap()
}

fn options(max_iterations: usize, max_evaluations: usize) -> RietveldRefinementOptions {
    RietveldRefinementOptions::new(
        calculation(),
        RefinementLimits::new(max_iterations, max_evaluations, None, 20).unwrap(),
        1,
        1.0e-12,
        1.0e-10,
        1.0e-6,
        10.0,
        0.3,
        1.0e-10,
        30,
        1.0,
        8,
    )
    .unwrap()
}

#[test]
fn phase_scale_recovery_is_bounded_and_deterministic() {
    let input = input_from_truth(phase(0.45, 4.7), phase(1.4, 4.7));
    let selection = RietveldStructuralSelection {
        phase_scale: true,
        ..RietveldStructuralSelection::default()
    };
    let first = refine_rietveld(&input, selection, &[None], &options(8, 500), None, None).unwrap();
    let second = refine_rietveld(&input, selection, &[None], &options(8, 500), None, None).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.termination_reason, TerminationReason::Converged);
    assert!((first.phases[0].definition().scale - 1.4).abs() < 1.0e-10);
    assert!(first.calculation.metrics.rwp < 1.0e-11);
    assert!(!first.history.is_empty());
    assert_eq!(first.checkpoint.completed_iterations, first.history.len());
}

#[test]
fn cubic_lattice_recovery_and_checkpoint_restart_match_uninterrupted() {
    let input = input_from_truth(phase(1.0, 4.699), phase(1.0, 4.7));
    let parameterization = LatticeParameterization::new(
        input.phases[0].definition().space_group.clone(),
        input.phases[0].definition().cell,
    )
    .unwrap();
    let bounds = LatticeBounds::new(&parameterization, vec![4.5], vec![4.9]).unwrap();
    let selection = RietveldStructuralSelection {
        lattice: true,
        ..RietveldStructuralSelection::default()
    };
    let uninterrupted = refine_rietveld(
        &input,
        selection,
        &[Some(bounds.clone())],
        &options(8, 800),
        None,
        None,
    )
    .unwrap();
    let partial = refine_rietveld(
        &input,
        selection,
        &[Some(bounds.clone())],
        &options(1, 200),
        None,
        None,
    )
    .unwrap();
    let resumed = refine_rietveld(
        &input,
        selection,
        &[Some(bounds)],
        &options(8, 800),
        Some(&partial.checkpoint),
        None,
    )
    .unwrap();

    assert_eq!(resumed.phases, uninterrupted.phases);
    assert_eq!(resumed.history, uninterrupted.history);
    let recovered = resumed.phases[0].definition().cell.a_angstrom;
    assert!(
        (recovered - 4.7).abs() < 2.0e-8,
        "recovered={recovered:.17e}, termination={:?}, history={:?}",
        resumed.termination_reason,
        resumed.history
    );
}

#[test]
fn cancellation_and_evaluation_budget_return_last_accepted_state() {
    let input = input_from_truth(phase(0.45, 4.7), phase(1.4, 4.7));
    let selection = RietveldStructuralSelection {
        phase_scale: true,
        ..RietveldStructuralSelection::default()
    };
    let token = CancellationToken::default();
    token.request("test cancellation").unwrap();
    let cancelled = refine_rietveld(
        &input,
        selection,
        &[None],
        &options(8, 500),
        None,
        Some(token),
    )
    .unwrap();
    assert_eq!(cancelled.termination_reason, TerminationReason::Cancelled);
    assert!(cancelled.history.is_empty());

    let bounded = refine_rietveld(&input, selection, &[None], &options(8, 2), None, None).unwrap();
    assert_eq!(
        bounded.termination_reason,
        TerminationReason::MaxEvaluations
    );
    assert!(bounded.history.is_empty());

    let mut excluded = input.clone();
    excluded.pattern.mask = Some(vec![false; excluded.pattern.sample_count()]);
    let no_observations =
        refine_rietveld(&excluded, selection, &[None], &options(8, 100), None, None).unwrap();
    assert_eq!(
        no_observations.termination_reason,
        TerminationReason::NoObservations
    );
    assert!(no_observations.history.is_empty());

    let partial =
        refine_rietveld(&input, selection, &[None], &options(1, 100), None, None).unwrap();
    let mut invalid = partial.checkpoint.clone();
    let key = invalid.parameters.specs()[0].key().clone();
    let mut replacement = std::collections::BTreeMap::new();
    replacement.insert(key, 99.0);
    invalid.parameters = invalid.parameters.replace_values(&replacement).unwrap();
    assert!(
        refine_rietveld(
            &input,
            selection,
            &[None],
            &options(8, 500),
            Some(&invalid),
            None,
        )
        .is_err()
    );
}

#[test]
fn caller_owned_runtime_receives_events_and_accepted_checkpoints() {
    let input = input_from_truth(phase(0.45, 4.7), phase(1.4, 4.7));
    let selection = RietveldStructuralSelection {
        phase_scale: true,
        ..RietveldStructuralSelection::default()
    };
    let options = options(8, 500);
    let events = Arc::new(Mutex::new(Vec::new()));
    let observed_events = events.clone();
    let checkpoints = Arc::new(Mutex::new(Vec::new()));
    let observed_checkpoints = checkpoints.clone();
    let mut runtime = RefinementRuntime::new(options.limits, None).unwrap();
    runtime.set_event_sink(move |event: &phasesmith_workflows::RefinementEvent| {
        observed_events.lock().unwrap().push(event.kind());
        Ok(())
    });
    runtime.set_checkpoint_sink(
        move |checkpoint: &phasesmith_workflows::RietveldCheckpoint| {
            observed_checkpoints
                .lock()
                .unwrap()
                .push(checkpoint.completed_iterations);
            Ok(())
        },
    );
    let result =
        refine_rietveld_with_runtime(&input, selection, &[None], &options, None, &mut runtime)
            .unwrap();

    let events = events.lock().unwrap();
    assert_eq!(events.first(), Some(&RefinementEventKind::Start));
    assert!(events.contains(&RefinementEventKind::StepAccepted));
    assert_eq!(events.last(), Some(&RefinementEventKind::Termination));
    assert_eq!(
        *checkpoints.lock().unwrap(),
        (1..=result.history.len()).collect::<Vec<_>>()
    );
}
