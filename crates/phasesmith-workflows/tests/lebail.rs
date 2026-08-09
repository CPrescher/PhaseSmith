//! Native fixed-reflection Le Bail workflow and Python differential checks.

use std::process::Command;
use std::sync::{Arc, Mutex};

use phasesmith_core::ConstantWavelengthInstrument;
use phasesmith_crystallography::UnitCell;
use phasesmith_execution::ExecutionPolicy;
use phasesmith_io::space_group_by_number;
use phasesmith_model::PatternRecord;
use phasesmith_workflows::{
    AffineConstraint, CancellationToken, Constraint, FixedConstraint, LatticeBounds,
    LatticeParameterization, LatticeReflectionDomain, LeBailCheckpoint, LeBailInput, LeBailOptions,
    LeBailPhase, RefinementLimits, RefinementRuntime, TerminationReason,
    build_lebail_parameter_set, build_lebail_parameter_set_with_lattice, calculate_lebail_pattern,
    iterate_lebail_once, lebail_lattice_parameter_key, lebail_reflection_position_key,
    refine_lebail, refine_lebail_with_runtime,
};

fn instrument() -> ConstantWavelengthInstrument {
    ConstantWavelengthInstrument {
        wavelength_angstrom: 1.5406,
        u_deg2: 2.0e-4,
        v_deg2: -1.0e-4,
        w_deg2: 2.0e-4,
        x_deg: 1.5e-3,
        y_deg: 3.0e-3,
    }
}

fn execution() -> ExecutionPolicy {
    ExecutionPolicy::new(Some(1), 2).unwrap()
}

fn options(max_iterations: usize) -> LeBailOptions {
    LeBailOptions::new(
        max_iterations,
        2.min(max_iterations),
        1.0e-6,
        1.0e-8,
        1.0,
        1.0e-15,
        1.0e-12,
        true,
        1.0 - 1.0e-10,
        false,
        20.0,
        execution(),
    )
    .unwrap()
}

fn phase(phase_id: &str, positions: &[f64], intensities: &[f64]) -> LeBailPhase {
    let wavelength = instrument().wavelength_angstrom;
    LeBailPhase::new(
        phase_id,
        format!("{phase_id} phase"),
        (0..positions.len())
            .map(|index| format!("{phase_id}-{index}"))
            .collect(),
        (0..positions.len())
            .map(|index| [i32::try_from(index + 1).unwrap(), 1, 0])
            .collect(),
        positions
            .iter()
            .map(|position| wavelength / (2.0 * (0.5 * position.to_radians()).sin()))
            .collect(),
        positions.to_vec(),
        intensities.to_vec(),
        1.0,
        Vec::new(),
    )
    .unwrap()
}

fn tetragonal_cell(a_angstrom: f64, c_angstrom: f64) -> UnitCell {
    UnitCell {
        a_angstrom,
        b_angstrom: a_angstrom,
        c_angstrom,
        alpha_deg: 90.0,
        beta_deg: 90.0,
        gamma_deg: 90.0,
    }
}

fn lattice_domain(reference: UnitCell) -> LatticeReflectionDomain {
    let group = space_group_by_number(123).expect("P 4/mmm").space_group;
    let parameterization = LatticeParameterization::new(group, reference).expect("parameters");
    let bounds = LatticeBounds::around(&parameterization, 0.04, 5.0).expect("bounds");
    LatticeReflectionDomain::new(
        parameterization,
        bounds,
        instrument().wavelength_angstrom,
        [20.0, 90.0],
        1.0,
        true,
        50_000_000,
        1.001,
    )
    .expect("lattice domain")
}

fn dynamic_phase(cell: UnitCell, domain: LatticeReflectionDomain) -> LeBailPhase {
    let phase =
        LeBailPhase::from_lattice_domain("alpha", "Alpha phase", cell, 1.0, domain).unwrap();
    let intensities = phase
        .hkl()
        .iter()
        .map(|hkl| {
            let marker =
                hkl[0].unsigned_abs() + 2 * hkl[1].unsigned_abs() + 3 * hkl[2].unsigned_abs();
            2.0 + f64::from(marker % 7)
        })
        .collect::<Vec<_>>();
    phase.with_integrated_intensities(&intensities).unwrap()
}

#[allow(clippy::cast_precision_loss)]
fn linspace(start: f64, endpoint: f64, count: usize) -> Vec<f64> {
    let spacing = (endpoint - start) / (count - 1) as f64;
    (0..count)
        .map(|index| {
            if index + 1 == count {
                endpoint
            } else {
                start + index as f64 * spacing
            }
        })
        .collect()
}

fn observed_pattern(
    x: Vec<f64>,
    truth: &[LeBailPhase],
    background: Vec<f64>,
    uncertainty: Option<Vec<f64>>,
    mask: Option<Vec<bool>>,
) -> PatternRecord {
    let blank = PatternRecord::new(
        x.clone(),
        None,
        uncertainty.clone(),
        mask.clone(),
        Some(background.clone()),
    )
    .unwrap();
    let calculated =
        calculate_lebail_pattern(&blank, instrument(), truth, 20.0, &execution()).unwrap();
    PatternRecord::new(x, Some(calculated.y), uncertainty, mask, Some(background)).unwrap()
}

#[test]
fn isolated_reflections_converge_with_display_components() {
    let x = linspace(20.0, 80.0, 6_001);
    let positions = [30.0, 50.0, 70.0];
    let truth = vec![phase("alpha", &positions, &[10.0, 6.0, 3.0])];
    let starting = vec![phase("alpha", &positions, &[1.0, 1.0, 1.0])];
    let pattern = observed_pattern(x.clone(), &truth, vec![0.2; x.len()], None, None);
    let result = refine_lebail(
        &LeBailInput::new(pattern.clone(), instrument(), starting).unwrap(),
        &options(10),
        None,
    )
    .unwrap();

    assert_eq!(result.termination_reason, TerminationReason::Converged);
    assert_close_slice(
        &result
            .intensities
            .iter()
            .map(|item| item.integrated_intensity)
            .collect::<Vec<_>>(),
        &[10.0, 6.0, 3.0],
        2.0e-11,
    );
    assert_close_slice(
        &result.calculation.y,
        pattern.observed_y.as_ref().unwrap(),
        4.0e-13,
    );
    assert_eq!(
        result.calculation.reflection_keys,
        [
            ("alpha".to_owned(), "alpha-0".to_owned()),
            ("alpha".to_owned(), "alpha-1".to_owned()),
            ("alpha".to_owned(), "alpha-2".to_owned()),
        ]
    );
    assert_close_slice(
        &result.calculation.phase_components[0].y,
        &result.calculation.profile_y,
        2.0e-15,
    );
}

#[test]
fn uncertainty_mask_resume_and_unobserved_behavior_are_deterministic() {
    let x = linspace(38.0, 43.0, 5_001);
    let positions = [40.0, 40.06, 41.4];
    let truth = vec![phase("alpha", &positions, &[9.0, 4.0, 6.0])];
    let starting = vec![phase("alpha", &positions, &[2.0, 7.0, 1.0])];
    let uncertainty = x.iter().map(|value| 0.5 + 0.01 * (value - x[0])).collect();
    let pattern = observed_pattern(x, &truth, vec![0.0; 5_001], Some(uncertainty), None);
    let input = LeBailInput::new(pattern, instrument(), starting).unwrap();
    let partial = refine_lebail(&input, &options(3), None).unwrap();
    let resumed = refine_lebail(&input, &options(30), Some(&partial.checkpoint)).unwrap();
    let uninterrupted = refine_lebail(&input, &options(30), None).unwrap();
    assert_eq!(resumed.history, uninterrupted.history);
    assert_eq!(
        resumed.checkpoint.intensities,
        uninterrupted.checkpoint.intensities
    );
    assert_eq!(resumed.calculation.y, uninterrupted.calculation.y);
    let first = iterate_lebail_once(&input, &options(30), None).unwrap();
    let second = iterate_lebail_once(&input, &options(30), Some(&first.checkpoint)).unwrap();
    let direct = refine_lebail(&input, &options(2), None).unwrap();
    assert_eq!(second.history, direct.history);
    assert_eq!(second.checkpoint.intensities, direct.checkpoint.intensities);

    let x = linspace(39.0, 43.0, 4_001);
    let truth = vec![phase("alpha", &[40.0, 42.0], &[5.0, 7.0])];
    let starting = vec![phase("alpha", &[40.0, 42.0, 80.0], &[0.0, 0.0, 0.0])];
    let mask = x.iter().map(|value| *value < 41.0).collect();
    let pattern = observed_pattern(x, &truth, vec![0.0; 4_001], None, Some(mask));
    let result = refine_lebail(
        &LeBailInput::new(pattern, instrument(), starting).unwrap(),
        &options(20),
        None,
    )
    .unwrap();
    assert!((result.intensities[0].integrated_intensity - 5.0).abs() < 2.0e-9);
    assert_eq!(result.intensities[1].integrated_intensity.to_bits(), 0);
    assert_eq!(result.intensities[2].integrated_intensity.to_bits(), 0);
    assert!(result.history.last().unwrap().warnings[0].contains("2 reflections"));
}

#[test]
fn invalid_state_is_rejected_and_worker_budgets_are_deterministic() {
    assert!(
        LeBailPhase::new(
            "bad",
            "Bad phase",
            vec!["r0".to_owned()],
            vec![[1, 0, 0]],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            1.0,
            Vec::new(),
        )
        .is_err()
    );
    assert!(
        LeBailOptions::new(
            1,
            2,
            1.0e-6,
            1.0e-8,
            1.0,
            1.0e-15,
            1.0e-12,
            true,
            0.9,
            false,
            20.0,
            execution(),
        )
        .is_err()
    );

    let x = linspace(38.0, 43.0, 5_001);
    let positions = [40.0, 40.06, 41.4];
    let truth = vec![phase("alpha", &positions, &[9.0, 4.0, 6.0])];
    let starting = vec![phase("alpha", &positions, &[2.0, 7.0, 1.0])];
    let pattern = observed_pattern(x, &truth, vec![0.0; 5_001], None, None);
    let input = LeBailInput::new(pattern, instrument(), starting).unwrap();
    let serial = refine_lebail(&input, &options(30), None).unwrap();
    let mut parallel_options = options(30);
    parallel_options.execution = ExecutionPolicy::new(Some(2), 2).unwrap();
    let parallel = refine_lebail(&input, &parallel_options, None).unwrap();
    assert_eq!(
        serial.checkpoint.intensities,
        parallel.checkpoint.intensities
    );
    assert_eq!(serial.history, parallel.history);
    assert_eq!(serial.calculation.y, parallel.calculation.y);
}

#[test]
fn dynamic_phase_selection_and_wavelength_contracts_are_explicit() {
    let cell = tetragonal_cell(4.0, 6.0);
    let dynamic = dynamic_phase(cell, lattice_domain(cell));
    assert!(
        build_lebail_parameter_set(
            instrument(),
            std::slice::from_ref(&dynamic),
            &[],
            false,
            true,
        )
        .is_err()
    );
    let fixed = phase("fixed", &[40.0], &[1.0]);
    assert!(
        build_lebail_parameter_set_with_lattice(instrument(), &[fixed], &[], false, false, true,)
            .is_err()
    );
    let x = linspace(20.0, 90.0, 1_001);
    let pattern = observed_pattern(
        x.clone(),
        std::slice::from_ref(&dynamic),
        vec![0.0; x.len()],
        None,
        None,
    );
    let mut wrong_wavelength = instrument();
    wrong_wavelength.wavelength_angstrom = 1.0;
    assert!(LeBailInput::new(pattern, wrong_wavelength, vec![dynamic]).is_err());
}

#[test]
fn analytical_position_instrument_and_constraint_updates_match_the_domain() {
    let x = linspace(39.0, 43.0, 4_001);
    let truth = vec![phase("alpha", &[40.0, 42.0], &[5.0, 7.0])];
    let starting = vec![phase("alpha", &[39.99, 41.99], &[5.0, 7.0])];
    let pattern = observed_pattern(x, &truth, vec![0.0; 4_001], None, None);
    let parameters = build_lebail_parameter_set(instrument(), &starting, &[], false, true).unwrap();
    let first = lebail_reflection_position_key("alpha", "alpha-0").unwrap();
    let second = lebail_reflection_position_key("alpha", "alpha-1").unwrap();
    let constraints = vec![Constraint::Affine(
        AffineConstraint::new(second, first, 1.0, 2.0).unwrap(),
    )];
    let input =
        LeBailInput::new_with_parameters(pattern, instrument(), starting, parameters, constraints)
            .unwrap();
    let selected = options(30).with_profile_controls(1.0e-10, 1.0, 8).unwrap();
    let result = refine_lebail(&input, &selected, None).unwrap();
    assert!((result.phases[0].two_theta_deg()[0] - 40.0).abs() < 4.0e-7);
    assert!((result.phases[0].two_theta_deg()[1] - 42.0).abs() < 4.0e-7);
    assert!(
        result
            .history
            .iter()
            .any(|record| !record.parameter_changes.is_empty())
    );
    assert!(result.covariance.is_some());
    let partial_options = options(2).with_profile_controls(1.0e-10, 1.0, 8).unwrap();
    let partial = refine_lebail(&input, &partial_options, None).unwrap();
    let resumed = refine_lebail(&input, &selected, Some(&partial.checkpoint)).unwrap();
    assert_eq!(resumed.history, result.history);
    assert_eq!(resumed.calculation.y, result.calculation.y);
    assert_eq!(resumed.parameters, result.parameters);

    let positions = [30.0, 60.0, 100.0];
    let x = linspace(20.0, 110.0, 9_001);
    let truth = vec![phase("alpha", &positions, &[8.0, 5.0, 3.0])];
    let starting = vec![phase("alpha", &positions, &[8.0, 5.0, 3.0])];
    let pattern = observed_pattern(x, &truth, vec![0.0; 9_001], None, None);
    let mut broad = instrument();
    broad.w_deg2 = 5.0e-4;
    let parameters =
        build_lebail_parameter_set(broad, &starting, &["w_deg2"], false, false).unwrap();
    let input =
        LeBailInput::new_with_parameters(pattern, broad, starting, parameters, Vec::new()).unwrap();
    let result = refine_lebail(
        &input,
        &options(30).with_profile_controls(1.0e-10, 1.0, 8).unwrap(),
        None,
    )
    .unwrap();
    assert!((result.instrument.w_deg2 - instrument().w_deg2).abs() < 2.0e-8);
    assert!(result.metrics.rwp < 2.0e-5);

    let x = linspace(39.0, 41.0, 1_001);
    let phases = vec![phase("alpha", &[40.0], &[8.0])];
    let pattern = observed_pattern(x, &phases, vec![0.0; 1_001], None, None);
    let parameters = build_lebail_parameter_set(instrument(), &phases, &[], true, false).unwrap();
    let input =
        LeBailInput::new_with_parameters(pattern, instrument(), phases, parameters, Vec::new())
            .unwrap();
    let result = refine_lebail(&input, &options(3), None).unwrap();
    assert!(result.covariance.is_none());
    assert!(
        result
            .history
            .iter()
            .flat_map(|record| &record.warnings)
            .any(|warning| warning.contains("not identifiable independently"))
    );
}

#[test]
fn bounded_lattice_refinement_recovers_cell_and_restarts_exactly() {
    let starting_cell = tetragonal_cell(3.995, 6.0);
    let true_cell = tetragonal_cell(4.0, 6.0);
    let domain = lattice_domain(starting_cell);
    let starting = vec![dynamic_phase(starting_cell, domain.clone())];
    let truth = vec![
        starting[0]
            .regenerate_lattice_at_cell(true_cell)
            .expect("truth cell"),
    ];
    let x = linspace(20.0, 90.0, 7_001);
    let pattern = observed_pattern(x.clone(), &truth, vec![0.0; x.len()], None, None);
    let parameters =
        build_lebail_parameter_set_with_lattice(instrument(), &starting, &[], false, false, true)
            .unwrap();
    let c_key = lebail_lattice_parameter_key("alpha", "c_angstrom").unwrap();
    let constraints = vec![Constraint::Fixed(
        FixedConstraint::new(c_key, 6.0).expect("fixed c"),
    )];
    let input =
        LeBailInput::new_with_parameters(pattern, instrument(), starting, parameters, constraints)
            .unwrap();
    let selected = options(30).with_profile_controls(1.0e-10, 0.05, 8).unwrap();
    let result = refine_lebail(&input, &selected, None).unwrap();
    let refined_cell = result.phases[0].cell().expect("dynamic cell");
    assert!((refined_cell.a_angstrom - 4.0).abs() < 2.0e-6);
    assert!((refined_cell.b_angstrom - 4.0).abs() < 2.0e-6);
    assert!(result.metrics.rwp < 2.0e-5);
    assert!(
        result
            .history
            .iter()
            .flat_map(|record| &record.parameter_changes)
            .any(
                |change| change.key == lebail_lattice_parameter_key("alpha", "a_angstrom").unwrap()
            )
    );

    let partial = refine_lebail(
        &input,
        &options(2).with_profile_controls(1.0e-10, 0.05, 8).unwrap(),
        None,
    )
    .unwrap();
    let resumed = refine_lebail(&input, &selected, Some(&partial.checkpoint)).unwrap();
    assert_eq!(resumed.history, result.history);
    assert_eq!(resumed.phases, result.phases);
    assert_eq!(resumed.calculation.y, result.calculation.y);
    assert_eq!(resumed.parameters, result.parameters);
}

#[test]
fn dynamic_checkpoint_allows_changed_topology_only_for_the_same_domain() {
    let reference = tetragonal_cell(4.0, 6.0);
    let group = space_group_by_number(123).expect("P 4/mmm").space_group;
    let parameterization = LatticeParameterization::new(group, reference).expect("parameters");
    let bounds = LatticeBounds::around(&parameterization, 0.2, 5.0).expect("bounds");
    let domain = LatticeReflectionDomain::new(
        parameterization.clone(),
        bounds.clone(),
        instrument().wavelength_angstrom,
        [20.0, 90.0],
        1.0,
        true,
        50_000_000,
        1.001,
    )
    .expect("domain");
    let starting = dynamic_phase(reference, domain.clone());
    let changed = bounds
        .corner_values()
        .into_iter()
        .map(|values| parameterization.to_cell(&values).expect("corner cell"))
        .map(|cell| dynamic_phase(cell, domain.clone()))
        .find(|phase| phase.reflection_ids() != starting.reflection_ids())
        .expect("wide anisotropic bounds should exercise a topology change");
    let x = linspace(20.0, 90.0, 2_001);
    let pattern = observed_pattern(
        x.clone(),
        std::slice::from_ref(&starting),
        vec![0.0; x.len()],
        None,
        None,
    );
    let input_parameters = build_lebail_parameter_set_with_lattice(
        instrument(),
        std::slice::from_ref(&starting),
        &[],
        false,
        false,
        true,
    )
    .unwrap();
    let input = LeBailInput::new_with_parameters(
        pattern,
        instrument(),
        vec![starting],
        input_parameters,
        Vec::new(),
    )
    .unwrap();
    let checkpoint_parameters = build_lebail_parameter_set_with_lattice(
        instrument(),
        std::slice::from_ref(&changed),
        &[],
        false,
        false,
        true,
    )
    .unwrap();
    let checkpoint = LeBailCheckpoint {
        completed_iterations: 0,
        phases: vec![changed.clone()],
        instrument: instrument(),
        intensities: changed.integrated_intensity().to_vec(),
        parameters: Some(checkpoint_parameters),
        previous_rwp: f64::INFINITY,
        history: Vec::new(),
    };
    refine_lebail(&input, &options(1), Some(&checkpoint))
        .expect("identical domains permit changed accepted topology");

    let incompatible_domain = LatticeReflectionDomain::new(
        parameterization,
        bounds,
        instrument().wavelength_angstrom,
        [20.0, 90.0],
        1.0,
        true,
        50_000_000,
        1.01,
    )
    .expect("incompatible domain policy");
    let incompatible = dynamic_phase(changed.cell().unwrap(), incompatible_domain);
    let incompatible_parameters = build_lebail_parameter_set_with_lattice(
        instrument(),
        std::slice::from_ref(&incompatible),
        &[],
        false,
        false,
        true,
    )
    .unwrap();
    let incompatible_checkpoint = LeBailCheckpoint {
        phases: vec![incompatible.clone()],
        intensities: incompatible.integrated_intensity().to_vec(),
        parameters: Some(incompatible_parameters),
        ..checkpoint
    };
    assert!(refine_lebail(&input, &options(1), Some(&incompatible_checkpoint)).is_err());
}

#[test]
fn profile_backtracking_obeys_the_host_evaluation_budget() {
    let x = linspace(39.0, 41.0, 1_001);
    let truth = vec![phase("alpha", &[40.0], &[8.0])];
    let starting = vec![phase("alpha", &[39.985], &[8.0])];
    let pattern = observed_pattern(x, &truth, vec![0.0; 1_001], None, None);
    let parameters = build_lebail_parameter_set(instrument(), &starting, &[], false, true).unwrap();
    let input =
        LeBailInput::new_with_parameters(pattern, instrument(), starting, parameters, Vec::new())
            .unwrap();
    let limits = RefinementLimits::new(30, 2, None, 2).unwrap();
    let mut runtime = RefinementRuntime::new(limits, None).unwrap();
    let result = refine_lebail_with_runtime(
        &input,
        &options(30).with_profile_controls(1.0e-10, 1.0, 8).unwrap(),
        None,
        &mut runtime,
    )
    .unwrap();
    assert_eq!(result.termination_reason, TerminationReason::MaxEvaluations);
    assert!(result.history.is_empty());
    assert_eq!(runtime.evaluations(), 2);
}

#[test]
fn coincident_reflections_retain_partition_and_report_joint_rank() {
    let x = linspace(39.0, 41.0, 4_001);
    let truth = vec![
        phase("alpha", &[40.0], &[4.0]),
        phase("beta", &[40.0], &[8.0]),
    ];
    let starting = vec![
        phase("alpha", &[40.0], &[1.0]),
        phase("beta", &[40.0], &[2.0]),
    ];
    let pattern = observed_pattern(x, &truth, vec![0.0; 4_001], None, None);
    let mut selected = options(10);
    selected.diagnose_rank_deficiency = true;
    let result = refine_lebail(
        &LeBailInput::new(pattern, instrument(), starting).unwrap(),
        &selected,
        None,
    )
    .unwrap();
    assert_close_slice(
        &result
            .intensities
            .iter()
            .map(|item| item.integrated_intensity)
            .collect::<Vec<_>>(),
        &[4.0, 8.0],
        2.0e-11,
    );
    assert_eq!(result.rank_deficient_groups.len(), 1);
    assert_eq!(result.rank_deficient_groups[0].rank, 1);
    assert_eq!(
        result.rank_deficient_groups[0].reflection_keys,
        [
            ("alpha".to_owned(), "alpha-0".to_owned()),
            ("beta".to_owned(), "beta-0".to_owned()),
        ]
    );
}

#[test]
fn runtime_cancellation_events_and_checkpoint_delivery_are_integrated() {
    let x = linspace(39.0, 41.0, 1_001);
    let truth = vec![phase("alpha", &[40.0], &[8.0])];
    let starting = vec![phase("alpha", &[40.0], &[1.0])];
    let pattern = observed_pattern(x, &truth, vec![0.0; 1_001], None, None);
    let input = LeBailInput::new(pattern, instrument(), starting).unwrap();
    let token = CancellationToken::default();
    token.request("desktop_stop").unwrap();
    let limits = RefinementLimits::new(10, 11, None, 2).unwrap();
    let mut runtime = RefinementRuntime::new(limits, Some(token)).unwrap();
    let cancelled = refine_lebail_with_runtime(&input, &options(10), None, &mut runtime).unwrap();
    assert_eq!(cancelled.termination_reason, TerminationReason::Cancelled);
    assert!(cancelled.history.is_empty());

    let delivered = Arc::new(Mutex::new(Vec::new()));
    let observed = delivered.clone();
    let mut runtime = RefinementRuntime::new(limits, None).unwrap();
    runtime.set_checkpoint_sink(move |checkpoint: &phasesmith_workflows::LeBailCheckpoint| {
        observed
            .lock()
            .unwrap()
            .push(checkpoint.completed_iterations);
        Ok(())
    });
    let result = refine_lebail_with_runtime(&input, &options(10), None, &mut runtime).unwrap();
    assert_eq!(
        *delivered.lock().unwrap(),
        (1..=result.history.len()).collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "requires an installed NumPy Python interpreter"]
fn fixed_reflection_history_matches_python_when_configured() {
    let Ok(python) = std::env::var("PHASESMITH_NUMPY_PYTHON") else {
        return;
    };
    let x = linspace(38.0, 43.0, 5_001);
    let positions = [40.0, 40.06, 41.4];
    let truth = vec![phase("alpha", &positions, &[9.0, 4.0, 6.0])];
    let starting = vec![phase("alpha", &positions, &[2.0, 7.0, 1.0])];
    let uncertainty = x.iter().map(|value| 0.5 + 0.01 * (value - x[0])).collect();
    let pattern = observed_pattern(x, &truth, vec![0.0; 5_001], Some(uncertainty), None);
    let result = refine_lebail(
        &LeBailInput::new(pattern, instrument(), starting).unwrap(),
        &options(30),
        None,
    )
    .unwrap();

    let script = r#"
import numpy as np
import phasesmith
from phasesmith.refinement import lebail

instrument = phasesmith.ConstantWavelengthInstrument(1.5406, 2e-4, -1e-4, 2e-4, 1.5e-3, 3e-3)
x = np.linspace(38.0, 43.0, 5001)
positions = np.array([40.0, 40.06, 41.4])

def phase(values):
    d = instrument.wavelength_angstrom / (2.0 * np.sin(np.deg2rad(positions / 2.0)))
    reflections = phasesmith.ReflectionBatch(
        [f"alpha-{index}" for index in range(3)],
        np.array([[1, 1, 0], [2, 1, 0], [3, 1, 0]], dtype=np.int64),
        d, positions, np.asarray(values, dtype=np.float64),
    )
    return phasesmith.Phase("alpha", "alpha phase", reflections)

blank = phasesmith.PowderPattern(x)
observed = phasesmith.calculate_pattern(blank, instrument, (phase([9.0, 4.0, 6.0]),)).y
pattern = phasesmith.PowderPattern(
    x, observed_y=observed, uncertainty=0.5 + 0.01 * (x - x[0])
)
result = lebail.refine(
    lebail.LeBailInput(pattern, instrument, (phase([2.0, 7.0, 1.0]),)),
    lebail.LeBailOptions(max_iterations=30, execution=phasesmith.ExecutionPolicy(threads=1)),
)
print(result.termination_reason.value)
print(" ".join(format(item.integrated_intensity, ".17g") for item in result.intensities))
for item in result.history:
    print(item.iteration, format(item.rwp, ".17g"), format(item.maximum_relative_intensity_change, ".17g"))
"#;
    let output = Command::new(python)
        .args(["-c", script])
        .env_remove("PYTHONPATH")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Python Le Bail oracle failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    assert_eq!(lines.next().unwrap(), result.termination_reason.as_str());
    let python_intensities = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|value| value.parse::<f64>().unwrap())
        .collect::<Vec<_>>();
    assert_close_slice(
        &result
            .intensities
            .iter()
            .map(|item| item.integrated_intensity)
            .collect::<Vec<_>>(),
        &python_intensities,
        5.0e-12,
    );
    let python_history = lines
        .map(|line| {
            let values = line.split_whitespace().collect::<Vec<_>>();
            (
                values[0].parse::<usize>().unwrap(),
                values[1].parse::<f64>().unwrap(),
                values[2].parse::<f64>().unwrap(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(python_history.len(), result.history.len());
    for (native, python) in result.history.iter().zip(python_history) {
        assert_eq!(native.iteration, python.0);
        assert!((native.rwp - python.1).abs() < 2.0e-14);
        assert!((native.maximum_relative_intensity_change - python.2).abs() < 2.0e-12);
    }
}

#[test]
#[ignore = "requires an installed NumPy Python interpreter"]
fn analytical_profile_history_matches_python_when_configured() {
    let Ok(python) = std::env::var("PHASESMITH_NUMPY_PYTHON") else {
        return;
    };
    let x = linspace(39.0, 41.0, 4_001);
    let truth = vec![phase("alpha", &[40.0], &[8.0])];
    let starting = vec![phase("alpha", &[39.985], &[8.0])];
    let pattern = observed_pattern(x, &truth, vec![0.0; 4_001], None, None);
    let parameters = build_lebail_parameter_set(instrument(), &starting, &[], false, true).unwrap();
    let input =
        LeBailInput::new_with_parameters(pattern, instrument(), starting, parameters, Vec::new())
            .unwrap();
    let result = refine_lebail(
        &input,
        &options(30).with_profile_controls(1.0e-10, 1.0, 8).unwrap(),
        None,
    )
    .unwrap();

    let script = r#"
import numpy as np
import phasesmith
from phasesmith.refinement import lebail

instrument = phasesmith.ConstantWavelengthInstrument(1.5406, 2e-4, -1e-4, 2e-4, 1.5e-3, 3e-3)
x = np.linspace(39.0, 41.0, 4001)

def phase(position):
    d = instrument.wavelength_angstrom / (2.0 * np.sin(np.deg2rad(position / 2.0)))
    reflections = phasesmith.ReflectionBatch(["alpha-0"], [[1, 1, 0]], [d], [position], [8.0])
    return phasesmith.Phase("alpha", "alpha phase", reflections)

observed = phasesmith.calculate_pattern(phasesmith.PowderPattern(x), instrument, (phase(40.0),)).y
pattern = phasesmith.PowderPattern(x, observed_y=observed)
starting = (phase(39.985),)
parameters = lebail.build_parameter_set(instrument, starting, reflection_positions=True)
result = lebail.refine(
    lebail.LeBailInput(pattern, instrument, starting, parameters),
    lebail.LeBailOptions(
        max_iterations=30,
        max_scaled_parameter_step=1.0,
        execution=phasesmith.ExecutionPolicy(threads=1),
    ),
)
print(format(result.phases[0].reflections.two_theta_deg[0], ".17g"))
print(format(result.metrics.rwp, ".17g"))
for item in result.history:
    changes = sum(abs(change.scaled_change) for change in item.parameter_changes)
    print(item.iteration, format(item.scaled_profile_step_norm, ".17g"), format(changes, ".17g"))
"#;
    let output = Command::new(python)
        .args(["-c", script])
        .env_remove("PYTHONPATH")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Python profile oracle failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    let python_position = lines.next().unwrap().parse::<f64>().unwrap();
    let python_rwp = lines.next().unwrap().parse::<f64>().unwrap();
    assert!((result.phases[0].two_theta_deg()[0] - python_position).abs() < 2.0e-12);
    assert!((result.metrics.rwp - python_rwp).abs() < 2.0e-12);
    let python_history = lines.collect::<Vec<_>>();
    assert_eq!(result.history.len(), python_history.len());
    for (native, python) in result.history.iter().zip(python_history) {
        let values = python.split_whitespace().collect::<Vec<_>>();
        assert_eq!(native.iteration, values[0].parse::<usize>().unwrap());
        assert!(
            (native.scaled_profile_step_norm - values[1].parse::<f64>().unwrap()).abs() < 2.0e-10
        );
        let native_changes = native
            .parameter_changes
            .iter()
            .map(|change| change.scaled_change.abs())
            .sum::<f64>();
        assert!((native_changes - values[2].parse::<f64>().unwrap()).abs() < 2.0e-10);
    }
}

#[test]
#[ignore = "requires an installed NumPy Python interpreter"]
fn bounded_lattice_history_matches_python_when_configured() {
    let Ok(python) = std::env::var("PHASESMITH_NUMPY_PYTHON") else {
        return;
    };
    let starting_cell = tetragonal_cell(3.995, 6.0);
    let domain = lattice_domain(starting_cell);
    let starting = vec![dynamic_phase(starting_cell, domain)];
    let truth = vec![
        starting[0]
            .regenerate_lattice_at_cell(tetragonal_cell(4.0, 6.0))
            .unwrap(),
    ];
    let x = linspace(20.0, 90.0, 7_001);
    let pattern = observed_pattern(x.clone(), &truth, vec![0.0; x.len()], None, None);
    let parameters =
        build_lebail_parameter_set_with_lattice(instrument(), &starting, &[], false, false, true)
            .unwrap();
    let constraint = Constraint::Fixed(
        FixedConstraint::new(
            lebail_lattice_parameter_key("alpha", "c_angstrom").unwrap(),
            6.0,
        )
        .unwrap(),
    );
    let input = LeBailInput::new_with_parameters(
        pattern,
        instrument(),
        starting,
        parameters,
        vec![constraint],
    )
    .unwrap();
    let result = refine_lebail(
        &input,
        &options(30).with_profile_controls(1.0e-10, 0.05, 8).unwrap(),
        None,
    )
    .unwrap();

    let stdout = run_python_lattice_oracle(&python);
    let mut lines = stdout.lines();
    let python_a = lines.next().unwrap().parse::<f64>().unwrap();
    let python_rwp = lines.next().unwrap().parse::<f64>().unwrap();
    let python_ids = lines.next().unwrap();
    assert!((result.phases[0].cell().unwrap().a_angstrom - python_a).abs() < 3.0e-11);
    assert!((result.metrics.rwp - python_rwp).abs() < 3.0e-11);
    assert_eq!(result.phases[0].reflection_ids().join("|"), python_ids);
    let python_history = lines.collect::<Vec<_>>();
    assert_eq!(result.history.len(), python_history.len());
    for (native, line) in result.history.iter().zip(python_history) {
        let values = line.split_whitespace().collect::<Vec<_>>();
        assert_eq!(native.iteration, values[0].parse::<usize>().unwrap());
        assert!((native.rwp - values[1].parse::<f64>().unwrap()).abs() < 3.0e-11);
        assert!(
            (native.maximum_relative_intensity_change - values[2].parse::<f64>().unwrap()).abs()
                < 3.0e-10
        );
        assert!(
            (native.scaled_profile_step_norm - values[3].parse::<f64>().unwrap()).abs() < 3.0e-10
        );
        assert_eq!(
            native.parameter_changes.len(),
            values[4].parse::<usize>().unwrap()
        );
        assert_eq!(native.warnings.len(), values[5].parse::<usize>().unwrap());
    }
}

fn run_python_lattice_oracle(python: &str) -> String {
    let script = r#"
from dataclasses import replace
import numpy as np
import phasesmith
from phasesmith.refinement import (
    CwLatticeReflectionDomain,
    FixedConstraint,
    LatticeParameterBounds,
    LatticeParameterization,
    lebail,
)

instrument = phasesmith.ConstantWavelengthInstrument(1.5406, 2e-4, -1e-4, 2e-4, 1.5e-3, 3e-3)
group = phasesmith.space_group_by_number(123).space_group
cell = phasesmith.UnitCell(3.995, 3.995, 6.0, 90.0, 90.0, 90.0)
structure = phasesmith.CrystalStructure("lattice-test", "Lattice test", cell, group)
parameterization = LatticeParameterization(group, cell)
bounds = LatticeParameterBounds.around(parameterization, relative_length=0.04)
domain = CwLatticeReflectionDomain(group, parameterization, bounds, 1.5406, 20.0, 90.0)
starting = lebail.LeBailPhase.from_structure(
    structure,
    phase_id="alpha",
    wavelength_angstrom=1.5406,
    two_theta_min_deg=20.0,
    two_theta_max_deg=90.0,
    lattice_bounds=bounds,
)
hkl = starting.reflections.hkl
marker = np.abs(hkl[:, 0]) + 2 * np.abs(hkl[:, 1]) + 3 * np.abs(hkl[:, 2])
intensities = 2.0 + marker % 7
starting = replace(
    starting,
    reflections=phasesmith.ReflectionBatch(
        starting.reflections.reflection_ids,
        hkl,
        starting.reflections.d_spacing_angstrom,
        starting.reflections.two_theta_deg,
        intensities,
    ),
)
true_cell = parameterization.to_cell([4.0, 6.0])
generated = domain.generate(true_cell, starting.reflections)
truth = replace(
    starting,
    structure=replace(starting.structure, cell=true_cell),
    reflections=generated.reflections,
)
x = np.linspace(20.0, 90.0, 7001)
observed = phasesmith.calculate_pattern(phasesmith.PowderPattern(x), instrument, (truth,)).y
pattern = phasesmith.PowderPattern(x, observed_y=observed, background=np.zeros_like(x))
parameters = lebail.build_parameter_set(instrument, (starting,), lattice_parameters=True)
constraints = (FixedConstraint(lebail.lattice_parameter_key("alpha", "c_angstrom"), 6.0),)
result = lebail.refine(
    lebail.LeBailInput(pattern, instrument, (starting,), parameters, constraints),
    lebail.LeBailOptions(
        max_iterations=30,
        min_iterations=2,
        intensity_tolerance=1e-6,
        rwp_tolerance=1e-8,
        profile_damping=1e-10,
        max_scaled_parameter_step=0.05,
        max_profile_backtracks=8,
        execution=phasesmith.ExecutionPolicy(threads=1, minimum_parallel_tasks=2),
    ),
)
print(format(result.phases[0].structure.cell.a_angstrom, ".17g"))
print(format(result.metrics.rwp, ".17g"))
print("|".join(result.phases[0].reflections.reflection_ids))
for item in result.history:
    print(
        item.iteration,
        format(item.rwp, ".17g"),
        format(item.maximum_relative_intensity_change, ".17g"),
        format(item.scaled_profile_step_norm, ".17g"),
        len(item.parameter_changes),
        len(item.warnings),
    )
"#;
    let output = Command::new(python)
        .args(["-c", script])
        .env_remove("PYTHONPATH")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Python lattice oracle failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn assert_close_slice(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index={index}, actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.1e}"
        );
    }
}
