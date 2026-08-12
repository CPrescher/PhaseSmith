#![allow(missing_docs)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use phasesmith_core::{
    OwnedCwContributions, TofBankGeometry, TofInstrument, TofInstrumentParameter,
};
use phasesmith_crystallography::{
    IntegratedIntensityCorrectionModel, Rational, SpaceGroup, SymmetryOperation, UnitCell,
};
use phasesmith_engine::{BuiltInScatteringModel, StructuralPhaseDefinition};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::{
    ProjectRecord, RecordId, StructuralPhaseRecord, TofExperimentRecord, TofHistogramRecord,
    TofPatternRecord,
};
use phasesmith_workflows::{
    CancellationToken, LatticeBounds, LatticeParameterization, ParameterBounds,
    PreparedStructuralTofMultiBankObjective, RefinementLimits, RefinementRuntime, RietveldPhase,
    RietveldStructuralSelection, StructuralTofBank, StructuralTofMultiBankAnalysis,
    StructuralTofMultiBankCheckpoint, StructuralTofMultiBankError, StructuralTofMultiBankInput,
    StructuralTofMultiBankLayout, StructuralTofMultiBankProjectState,
    StructuralTofMultiBankRefinementOptions, TerminationReason, TofChebyshevBackground,
    TofInstrumentParameterBound, refine_structural_tof_multibank,
    refine_structural_tof_multibank_with_runtime,
};

fn id(value: &str) -> RecordId {
    RecordId::new(value).expect("valid test ID")
}

fn cell() -> UnitCell {
    UnitCell {
        a_angstrom: 4.7,
        b_angstrom: 5.1,
        c_angstrom: 6.2,
        alpha_deg: 82.0,
        beta_deg: 87.0,
        gamma_deg: 74.0,
    }
}

fn instrument(zero_us: f64) -> TofInstrument {
    TofInstrument {
        zero_us,
        difc_us_per_angstrom: 5_000.0,
        difa_us_per_angstrom2: 0.2,
        difb_us_angstrom: 0.0,
        alpha_coefficient: 0.2,
        beta0_per_us: 0.03,
        beta1_angstrom4_per_us: 0.001,
        betaq_angstrom2_per_us: 0.0,
        sigma0_us2: 25.0,
        sigma1_us2_per_angstrom2: 4.0,
        sigma2_us2_per_angstrom4: 0.1,
        sigmaq_us2_per_angstrom: 0.0,
        x_us_per_angstrom: 1.0,
        y_us_per_angstrom2: 0.1,
        z_us: 0.5,
    }
}

fn phase() -> RietveldPhase {
    let definition = StructuralPhaseDefinition {
        cell: cell(),
        space_group: SpaceGroup::new(vec![SymmetryOperation::identity()]).expect("P1"),
        hkl: vec![[1, 0, 1], [2, 1, 1], [1, 2, 3]],
        multiplicity: vec![2, 4, 2],
        fractional_xyz: vec![[0.17, 0.23, 0.31], [0.37, 0.11, 0.19]],
        occupancy: vec![0.82, 0.55],
        u_iso_angstrom2: vec![0.012, 0.018],
        anisotropic_mask: vec![false, false],
        u_aniso_cif_angstrom2: vec![[0.0; 6]; 2],
        scattering_species: vec!["Si".to_owned(), "O".to_owned()],
        scattering_real_offset: Vec::new(),
        scattering_imag_offset: Vec::new(),
        scale: 1.0,
        coordinate_tolerance: 1.0e-10,
        scattering_model: BuiltInScatteringModel::NeutronNuclear,
        correction_model: IntegratedIntensityCorrectionModel::Neutral,
    };
    let reflections = definition.hkl.len();
    RietveldPhase::new_with_site_ids(
        id("phase"),
        "TOF phase",
        vec![id("si"), id("o")],
        definition,
        OwnedCwContributions::neutral(reflections),
    )
    .expect("phase")
}

fn bank(bank_id: &str, angle: f64, zero: f64, scale: f64) -> StructuralTofBank {
    let tof_us = (0..2_401)
        .map(|index| 1_000.0 + 10.0 * f64::from(index))
        .collect::<Vec<_>>();
    let pattern = TofPatternRecord::new(
        tof_us.clone(),
        Some(vec![0.0; tof_us.len()]),
        Some(
            (0..tof_us.len())
                .map(|index| 0.8 + 0.001 * f64::from(u32::try_from(index % 37).unwrap()))
                .collect(),
        ),
        Some((0..tof_us.len()).map(|index| index % 19 != 0).collect()),
        Some(vec![0.15; tof_us.len()]),
    )
    .expect("pattern");
    StructuralTofBank {
        bank_id: id(bank_id),
        pattern,
        instrument: instrument(zero),
        geometry: TofBankGeometry {
            two_theta_deg: angle,
        },
        correction_model: IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz {
            two_theta_deg: angle,
        },
        scale,
        scale_bounds: ParameterBounds::new(0.2, 3.0).expect("scale bounds"),
        refine_scale: true,
        background: Some(
            TofChebyshevBackground::new(
                id(&format!("{bank_id}-background")),
                vec![0.2, -0.03],
                [tof_us[0], *tof_us.last().unwrap()],
            )
            .expect("background"),
        ),
        refine_background: true,
        instrument_bounds: vec![
            TofInstrumentParameterBound::new(TofInstrumentParameter::Zero, -10.0, 10.0)
                .expect("zero bounds"),
        ],
    }
}

fn input() -> StructuralTofMultiBankInput {
    let phase = phase();
    let parameterization = LatticeParameterization::new(
        phase.definition().space_group.clone(),
        phase.definition().cell,
    )
    .expect("parameterization");
    StructuralTofMultiBankInput {
        phase,
        structural_selection: RietveldStructuralSelection {
            lattice: true,
            coordinates: true,
            occupancy: true,
            u_iso: true,
            phase_scale: false,
        },
        lattice_bounds: Some(
            LatticeBounds::around(&parameterization, 0.05, 3.0).expect("lattice bounds"),
        ),
        banks: vec![
            bank("bank-1", 88.05, 1.2, 1.3),
            bank("bank-2", 120.0, -0.7, 0.9),
        ],
        support_fwhm: 20.0,
        tail_log: 20.0,
        use_uncertainty: true,
        execution: ExecutionPolicy::new(Some(1), 16).expect("execution"),
    }
}

fn with_synthetic_observations(
    mut input: StructuralTofMultiBankInput,
) -> StructuralTofMultiBankInput {
    let calculated = PreparedStructuralTofMultiBankObjective::new(input.clone())
        .expect("initial objective")
        .calculate()
        .expect("initial calculation");
    for (bank, calculation) in input.banks.iter_mut().zip(calculated.banks) {
        bank.pattern.observed_y = Some(
            calculation
                .y
                .iter()
                .enumerate()
                .map(|(sample, value)| {
                    value
                        + 0.002
                            * (f64::from(u32::try_from(sample).expect("sample fits u32")) * 0.17)
                                .sin()
                })
                .collect(),
        );
    }
    input
}

fn scale_zero_solver_request() -> (StructuralTofMultiBankInput, Vec<f64>, Vec<f64>) {
    let mut request = input();
    request.structural_selection = RietveldStructuralSelection::default();
    request.lattice_bounds = None;
    for bank in &mut request.banks {
        bank.refine_background = false;
    }
    let layout = StructuralTofMultiBankLayout::new(&request).unwrap();
    assert_eq!(layout.parameters().specs().len(), 4);
    let initial = layout
        .parameters()
        .specs()
        .iter()
        .map(phasesmith_workflows::ParameterSpec::value)
        .collect::<Vec<_>>();
    let truth_values = vec![1.55, 2.3, 0.72, -1.8];
    let truth = layout.apply_values(&request, &truth_values).unwrap();
    let calculated = PreparedStructuralTofMultiBankObjective::new(truth)
        .unwrap()
        .calculate()
        .unwrap();
    for (bank, result) in request.banks.iter_mut().zip(calculated.banks) {
        bank.pattern.observed_y = Some(result.y);
    }
    (request, truth_values, initial)
}

fn single_bank_scale_zero_solver_request() -> (StructuralTofMultiBankInput, Vec<f64>) {
    let mut request = input();
    request.banks.truncate(1);
    request.structural_selection = RietveldStructuralSelection::default();
    request.lattice_bounds = None;
    request.banks[0].refine_background = false;
    let layout = StructuralTofMultiBankLayout::new(&request).unwrap();
    assert_eq!(layout.parameters().specs().len(), 2);
    let truth_values = vec![1.55, 2.3];
    let truth = layout.apply_values(&request, &truth_values).unwrap();
    let calculated = PreparedStructuralTofMultiBankObjective::new(truth)
        .unwrap()
        .calculate()
        .unwrap();
    request.banks[0].pattern.observed_y = Some(calculated.banks[0].y.clone());
    (request, truth_values)
}

fn special_position_coordinate_solver_request() -> StructuralTofMultiBankInput {
    let mirror = SpaceGroup::new(vec![
        SymmetryOperation::identity(),
        SymmetryOperation::new([[1, 0, 0], [0, 1, 0], [0, 0, -1]], [Rational::zero(); 3]).unwrap(),
    ])
    .unwrap();
    let mut definition = phase().definition().clone();
    definition.cell = UnitCell {
        alpha_deg: 90.0,
        beta_deg: 90.0,
        gamma_deg: 90.0,
        ..definition.cell
    };
    definition.space_group = mirror;
    definition.fractional_xyz[0][2] = 0.0;
    let phase = RietveldPhase::new_with_site_ids(
        id("mirror-phase"),
        "Mirror phase",
        vec![id("si-mirror"), id("o-general")],
        definition,
        OwnedCwContributions::neutral(3),
    )
    .unwrap();
    let mut request = input();
    request.phase = phase;
    request.structural_selection = RietveldStructuralSelection {
        coordinates: true,
        ..RietveldStructuralSelection::default()
    };
    request.lattice_bounds = None;
    request.banks.truncate(1);
    request.banks[0].refine_scale = false;
    request.banks[0].refine_background = false;
    request.banks[0].instrument_bounds.clear();
    let layout = StructuralTofMultiBankLayout::new(&request).unwrap();
    assert_eq!(
        layout
            .parameters()
            .specs()
            .iter()
            .map(|spec| spec.key().name())
            .collect::<Vec<_>>(),
        ["q0", "q1", "x", "y", "z"]
    );
    let truth = layout
        .apply_values(&request, &[0.025, -0.018, 0.37, 0.11, 0.19])
        .unwrap();
    let calculated = PreparedStructuralTofMultiBankObjective::new(truth)
        .unwrap()
        .calculate()
        .unwrap();
    request.banks[0].pattern.observed_y = Some(calculated.banks[0].y.clone());
    request
}

fn solver_options(max_iterations: usize) -> StructuralTofMultiBankRefinementOptions {
    StructuralTofMultiBankRefinementOptions::new(
        RefinementLimits::new(max_iterations, 200, None, 8).unwrap(),
        1,
        1.0e-12,
        1.0e-9,
        1.0e-3,
        10.0,
        0.3,
        1.0,
        8,
    )
    .unwrap()
}

#[test]
#[allow(clippy::too_many_lines)]
fn joint_products_match_differences_and_the_adjoint_identity() {
    let request = with_synthetic_observations(input());
    let objective = PreparedStructuralTofMultiBankObjective::new(request.clone()).unwrap();
    let layout = objective.layout();
    assert_eq!(layout.parameters().specs().len(), 24);
    let direction = (0..layout.parameters().specs().len())
        .map(|index| 2.0e-4 * f64::from(u32::try_from(index + 1).unwrap()))
        .collect::<Vec<_>>();
    let products = objective.jvp(&direction).unwrap();
    let weights = request
        .banks
        .iter()
        .map(|bank| {
            bank.pattern
                .tof_us
                .iter()
                .map(|value| (value * 1.0e-3).sin())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let reverse = objective.vjp(&weights).unwrap();
    let forward_dot = products
        .iter()
        .zip(&weights)
        .map(|(product, weight)| {
            product
                .derivative
                .iter()
                .zip(weight)
                .map(|(left, right)| left * right)
                .sum::<f64>()
        })
        .sum::<f64>();
    let reverse_dot = direction
        .iter()
        .zip(&reverse)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    assert!((forward_dot - reverse_dot).abs() < 2.0e-10 * forward_dot.abs().max(1.0));

    let values = layout
        .parameters()
        .specs()
        .iter()
        .map(phasesmith_workflows::ParameterSpec::value)
        .collect::<Vec<_>>();
    let step = 1.0e-6;
    let plus_values = values
        .iter()
        .zip(&direction)
        .map(|(value, direction)| value + step * direction)
        .collect::<Vec<_>>();
    let minus_values = values
        .iter()
        .zip(&direction)
        .map(|(value, direction)| value - step * direction)
        .collect::<Vec<_>>();
    let plus = layout.apply_values(&request, &plus_values).unwrap();
    let minus = layout.apply_values(&request, &minus_values).unwrap();
    let plus = PreparedStructuralTofMultiBankObjective::new(plus)
        .unwrap()
        .calculate()
        .unwrap();
    let minus = PreparedStructuralTofMultiBankObjective::new(minus)
        .unwrap()
        .calculate()
        .unwrap();
    for ((analytical, plus), minus) in products.iter().zip(plus.banks).zip(minus.banks) {
        for sample in 0..analytical.derivative.len() {
            let finite = (plus.y[sample] - minus.y[sample]) / (2.0 * step);
            let scale = finite.abs().max(1.0);
            assert!(
                (analytical.derivative[sample] - finite).abs() <= 3.0e-4 * scale,
                "bank={} sample={sample} analytical={} finite={finite}",
                analytical.bank_id,
                analytical.derivative[sample],
            );
        }
    }

    let gradient = objective.gradient().unwrap();
    let finite_objective = (plus.objective - minus.objective) / (2.0 * step);
    let directional_gradient = gradient
        .gradient
        .iter()
        .zip(&direction)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    assert!(
        (directional_gradient - finite_objective).abs() < 5.0e-4 * finite_objective.abs().max(1.0)
    );

    let normal = objective.normal_product(&direction, 0.3).unwrap();
    let weighted = request
        .banks
        .iter()
        .zip(&products)
        .map(|(bank, product)| {
            product
                .derivative
                .iter()
                .enumerate()
                .map(|(sample, value)| {
                    if bank.pattern.mask.as_ref().unwrap()[sample] {
                        let sigma = bank.pattern.uncertainty.as_ref().unwrap()[sample];
                        value / (sigma * sigma)
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let expected_normal = objective.vjp(&weighted).unwrap();
    for ((actual, expected), tangent) in normal.iter().zip(expected_normal).zip(&direction) {
        assert!((actual - (expected + 0.3 * tangent)).abs() < 2.0e-10 * actual.abs().max(1.0));
    }
}

#[test]
fn contracts_reject_implicit_or_incompatible_physics() {
    let mut request = input();
    request.banks.clear();
    assert!(matches!(
        request.validate(),
        Err(StructuralTofMultiBankError::TooFewBanks)
    ));

    let mut request = input();
    request.banks[1].bank_id = request.banks[0].bank_id.clone();
    assert!(matches!(
        request.validate(),
        Err(StructuralTofMultiBankError::DuplicateBankId)
    ));

    let mut request = input();
    request.banks[0].correction_model =
        IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz {
            two_theta_deg: 90.0,
        };
    assert!(matches!(
        request.validate(),
        Err(StructuralTofMultiBankError::InvalidBankContract(_))
    ));

    let mut request = input();
    request.structural_selection.phase_scale = true;
    assert!(matches!(
        StructuralTofMultiBankLayout::new(&request),
        Err(StructuralTofMultiBankError::InvalidPhaseContract(_))
    ));
}

#[test]
fn single_bank_solver_recovers_local_scale_and_zero() {
    let (request, truth_values) = single_bank_scale_zero_solver_request();
    let result = refine_structural_tof_multibank(&request, solver_options(20), None, None).unwrap();

    assert_eq!(result.input.banks.len(), 1);
    assert_eq!(result.calculation.banks.len(), 1);
    assert_eq!(result.termination_reason, TerminationReason::Converged);
    assert!(result.calculation.objective < 1.0e-12);
    for (spec, truth) in result.parameters.specs().iter().zip(truth_values) {
        assert!((spec.value() - truth).abs() < 2.0e-6, "{}", spec.key());
    }
    result.checkpoint.validate_for(&request).unwrap();
}

#[test]
fn special_position_coordinates_use_incremental_state_and_resume_exactly() {
    let request = special_position_coordinate_solver_request();
    let partial = refine_structural_tof_multibank(&request, solver_options(2), None, None).unwrap();
    assert!(!partial.history.is_empty());
    assert!(
        partial.input.phase.definition().fractional_xyz[0]
            .iter()
            .zip(request.phase.definition().fractional_xyz[0])
            .any(|(after, before)| (after - before).abs() > 1.0e-12)
    );
    assert!(
        partial.parameters.specs()[..2]
            .iter()
            .all(|spec| spec.value().to_bits() == 0.0_f64.to_bits())
    );
    partial.checkpoint.validate_for(&request).unwrap();

    let resumed = refine_structural_tof_multibank(
        &request,
        solver_options(20),
        Some(&partial.checkpoint),
        None,
    )
    .unwrap();
    let uninterrupted =
        refine_structural_tof_multibank(&request, solver_options(20), None, None).unwrap();
    assert_eq!(resumed.history, uninterrupted.history);
    assert_eq!(resumed.input, uninterrupted.input);
    assert_eq!(resumed.parameters, uninterrupted.parameters);
    assert_eq!(resumed.checkpoint, uninterrupted.checkpoint);
}

#[test]
fn bounded_solver_recovers_bank_scales_and_zero_terms() {
    let (request, truth_values, initial) = scale_zero_solver_request();
    let result = refine_structural_tof_multibank(&request, solver_options(20), None, None).unwrap();
    assert_eq!(result.termination_reason, TerminationReason::Converged);
    assert!(!result.history.is_empty());
    assert!(result.calculation.objective < 1.0e-12);
    for ((spec, truth), start) in result
        .parameters
        .specs()
        .iter()
        .zip(truth_values)
        .zip(initial)
    {
        assert!((spec.value() - truth).abs() < 2.0e-6, "{}", spec.key());
        assert_ne!(spec.value().to_bits(), start.to_bits());
    }
    result.checkpoint.validate_for(&request).unwrap();
}

#[test]
fn checkpoint_resume_is_exact_and_corruption_is_rejected() {
    let (request, _, _) = scale_zero_solver_request();
    let partial = refine_structural_tof_multibank(&request, solver_options(2), None, None).unwrap();
    assert_eq!(partial.termination_reason, TerminationReason::MaxIterations);
    assert_eq!(partial.history.len(), 2);
    let resumed = refine_structural_tof_multibank(
        &request,
        solver_options(20),
        Some(&partial.checkpoint),
        None,
    )
    .unwrap();
    let uninterrupted =
        refine_structural_tof_multibank(&request, solver_options(20), None, None).unwrap();
    assert_eq!(resumed.history, uninterrupted.history);
    assert_eq!(resumed.input, uninterrupted.input);
    assert_eq!(resumed.parameters, uninterrupted.parameters);
    assert_eq!(resumed.checkpoint, uninterrupted.checkpoint);

    let mut corrupt = partial.checkpoint;
    corrupt.objective += 1.0;
    assert!(corrupt.validate_for(&request).is_err());
}

#[test]
fn cancellation_and_evaluation_limit_return_the_last_accepted_state() {
    let (request, _, _) = scale_zero_solver_request();
    let cancellation = CancellationToken::default();
    let requested = cancellation.clone();
    let checkpoints = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&checkpoints);
    let options = solver_options(20);
    let mut runtime = RefinementRuntime::<StructuralTofMultiBankCheckpoint>::new(
        options.limits,
        Some(cancellation),
    )
    .unwrap();
    runtime.set_checkpoint_sink(move |checkpoint: &StructuralTofMultiBankCheckpoint| {
        captured.lock().unwrap().push(checkpoint.clone());
        if checkpoint.completed_iterations() == 2 {
            requested
                .request("structural TOF test stop")
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    });
    let stopped =
        refine_structural_tof_multibank_with_runtime(&request, options, None, &mut runtime)
            .unwrap();
    assert_eq!(stopped.termination_reason, TerminationReason::Cancelled);
    assert_eq!(stopped.history.len(), 2);
    assert_eq!(stopped.checkpoint, checkpoints.lock().unwrap()[1]);

    let bounded_options = StructuralTofMultiBankRefinementOptions::new(
        RefinementLimits::new(20, 2, None, 8).unwrap(),
        1,
        1.0e-12,
        1.0e-9,
        1.0e-3,
        10.0,
        0.3,
        1.0,
        8,
    )
    .unwrap();
    let bounded = refine_structural_tof_multibank(&request, bounded_options, None, None).unwrap();
    assert_eq!(
        bounded.termination_reason,
        TerminationReason::MaxEvaluations
    );
    assert!(bounded.history.is_empty());
    assert_eq!(bounded.input, request);
}

#[test]
fn structural_tof_project_owns_exact_histograms_phase_and_checkpoint() {
    let (request, _, _) = scale_zero_solver_request();
    let options = solver_options(20);
    let refined = refine_structural_tof_multibank(&request, options, None, None).unwrap();
    let project = ProjectRecord {
        project_id: id("structural-tof-project"),
        revision: 3,
        name: "Structural TOF project".to_owned(),
        histograms: Vec::new(),
        tof_histograms: request
            .banks
            .iter()
            .map(|bank| TofHistogramRecord {
                histogram_id: bank.bank_id.clone(),
                name: bank.bank_id.as_str().to_owned(),
                pattern: bank.pattern.clone(),
                experiment: TofExperimentRecord::new(bank.instrument).unwrap(),
                phase_ids: vec![request.phase.phase_id().clone()],
            })
            .collect(),
        phases: vec![StructuralPhaseRecord {
            phase_id: request.phase.phase_id().clone(),
            name: request.phase.name().to_owned(),
            definition: request.phase.definition().clone(),
            required_providers: Vec::new(),
        }],
        metadata: BTreeMap::new(),
    };
    let analysis = StructuralTofMultiBankAnalysis {
        analysis_id: id("structural-tof-analysis"),
        input: request.clone(),
        options,
        checkpoint: Some(refined.checkpoint),
    };
    let state = StructuralTofMultiBankProjectState {
        project,
        analyses: vec![analysis],
    };
    state.validate().unwrap();

    let mut duplicate = state.clone();
    duplicate.analyses.push(duplicate.analyses[0].clone());
    assert!(duplicate.validate().is_err());

    let mut changed_histogram = state.clone();
    changed_histogram.project.tof_histograms[0]
        .pattern
        .background_y[0] += 1.0;
    assert!(changed_histogram.validate().is_err());

    let mut changed_phase = state.clone();
    changed_phase.project.phases[0].name.push_str(" changed");
    assert!(changed_phase.validate().is_err());

    let mut changed_checkpoint = state;
    changed_checkpoint.analyses[0]
        .checkpoint
        .as_mut()
        .unwrap()
        .objective += 1.0;
    assert!(changed_checkpoint.validate().is_err());
}
