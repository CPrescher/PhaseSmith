//! Shared analytical cell refinement over distinct TOF detector banks.

use std::sync::{Arc, Mutex};

use phasesmith_core::TofInstrument;
use phasesmith_crystallography::UnitCell;
use phasesmith_execution::ExecutionPolicy;
use phasesmith_io::space_group_by_number;
use phasesmith_model::{RecordId, TofPatternRecord};
use phasesmith_workflows::{
    CancellationToken, LatticeBounds, LatticeParameterization, RefinementLimits, RefinementRuntime,
    TerminationReason, TofChebyshevBackground, TofLeBailBank, TofLeBailInput, TofLeBailOptions,
    TofLeBailPhase, TofMultiBankInput, TofMultiBankLatticeCheckpoint, TofMultiBankLatticeInput,
    TofMultiBankLatticeOptions, TofSharedLatticePhase, calculate_tof_lebail_pattern,
    refine_tof_multibank_lattice, refine_tof_multibank_lattice_with_runtime, tof_lattice_geometry,
};

fn id(value: &str) -> RecordId {
    RecordId::new(value).unwrap()
}

fn cell(a: f64) -> UnitCell {
    UnitCell {
        a_angstrom: a,
        b_angstrom: a,
        c_angstrom: a,
        alpha_deg: 90.0,
        beta_deg: 90.0,
        gamma_deg: 90.0,
    }
}

fn instrument(difc: f64, zero: f64) -> TofInstrument {
    TofInstrument {
        zero_us: zero,
        difc_us_per_angstrom: difc,
        difa_us_per_angstrom2: -0.2,
        difb_us_angstrom: 0.3,
        alpha_coefficient: 0.18,
        beta0_per_us: 0.04,
        beta1_angstrom4_per_us: 0.000_5,
        betaq_angstrom2_per_us: 0.001,
        sigma0_us2: 1.0,
        sigma1_us2_per_angstrom2: 10.0,
        sigma2_us2_per_angstrom4: 0.05,
        sigmaq_us2_per_angstrom: 0.2,
        x_us_per_angstrom: 0.3,
        y_us_per_angstrom2: 0.05,
        z_us: 0.4,
    }
}

fn options(cycles: usize) -> TofMultiBankLatticeOptions {
    let lebail = TofLeBailOptions::new(
        cycles,
        1.0,
        1.0e-12,
        1.0e-15,
        20.0,
        20.0,
        true,
        ExecutionPolicy::new(Some(1), 2).unwrap(),
    )
    .unwrap();
    TofMultiBankLatticeOptions::new(lebail, 1.0e-10, 0.01, 8).unwrap()
}

fn phase(
    parameterization: &LatticeParameterization,
    phase_cell: UnitCell,
    instrument: TofInstrument,
    intensities: Vec<f64>,
    scale: f64,
) -> TofLeBailPhase {
    let hkl = vec![[1, 0, 0], [1, 1, 0], [1, 1, 1]];
    let geometry = tof_lattice_geometry(parameterization, phase_cell, &hkl, instrument).unwrap();
    TofLeBailPhase::new(
        id("alpha"),
        "shared alpha",
        vec!["100".to_owned(), "110".to_owned(), "111".to_owned()],
        hkl,
        geometry.d_spacing_angstrom,
        intensities,
        scale,
    )
    .unwrap()
}

struct SyntheticRequest {
    input: TofMultiBankLatticeInput,
    truth_a: f64,
}

#[allow(clippy::too_many_lines)]
fn request() -> SyntheticRequest {
    let initial_cell = cell(3.995);
    let truth_cell = cell(4.0);
    let group = space_group_by_number(221).unwrap().space_group;
    let parameterization = LatticeParameterization::new(group, initial_cell).unwrap();
    let bounds = LatticeBounds::around(&parameterization, 0.02, 1.0).unwrap();
    let model =
        TofSharedLatticePhase::new(id("alpha"), parameterization.clone(), bounds, initial_cell)
            .unwrap();
    let bank_specs = [
        (
            "bank-1",
            instrument(5_000.0, -0.7),
            (0..2_801)
                .map(|index| 9_000.0 + 5.0 * f64::from(index))
                .collect::<Vec<_>>(),
            vec![120.0, 75.0, 210.0],
            1.0,
            vec![1.8, 0.12],
            29,
        ),
        (
            "bank-2",
            instrument(4_400.0, 1.3),
            (0..2_401)
                .map(|index| 7_500.0 + 5.0 * f64::from(index))
                .collect::<Vec<_>>(),
            vec![55.0, 180.0, 95.0],
            1.4,
            vec![1.1, -0.08, 0.03],
            31,
        ),
    ];
    let mut banks = Vec::new();
    for (bank_id, instrument, grid, intensities, scale, background, mask_stride) in bank_specs {
        let domain = [grid[0], grid[grid.len() - 1]];
        let blank = TofPatternRecord::new(
            grid.clone(),
            Some(vec![0.0; grid.len()]),
            Some(vec![1.0; grid.len()]),
            None,
            Some(vec![0.2; grid.len()]),
        )
        .unwrap();
        let truth = TofLeBailInput::new(
            blank,
            instrument,
            vec![phase(
                &parameterization,
                truth_cell,
                instrument,
                intensities.clone(),
                scale,
            )],
        )
        .unwrap()
        .with_refinable_background(
            TofChebyshevBackground::new(
                id(&format!("background-{bank_id}")),
                background.clone(),
                domain,
            )
            .unwrap(),
        )
        .unwrap();
        let observed = calculate_tof_lebail_pattern(&truth, &options(1).lebail)
            .unwrap()
            .y;
        let pattern = TofPatternRecord::new(
            grid,
            Some(observed),
            Some(vec![1.0; truth.pattern.sample_count()]),
            Some(
                (0..truth.pattern.sample_count())
                    .map(|index| index % mask_stride != 0)
                    .collect(),
            ),
            Some(vec![0.2; truth.pattern.sample_count()]),
        )
        .unwrap();
        let input = TofLeBailInput::new(
            pattern,
            instrument,
            vec![phase(
                &parameterization,
                initial_cell,
                instrument,
                intensities,
                scale,
            )],
        )
        .unwrap()
        .with_refinable_background(
            TofChebyshevBackground::new(
                id(&format!("background-{bank_id}")),
                vec![0.0; background.len()],
                domain,
            )
            .unwrap(),
        )
        .unwrap();
        banks.push(TofLeBailBank {
            bank_id: id(bank_id),
            input,
        });
    }
    SyntheticRequest {
        input: TofMultiBankLatticeInput {
            multibank: TofMultiBankInput { banks },
            lattice_phases: vec![model],
        },
        truth_a: truth_cell.a_angstrom,
    }
}

#[test]
fn summed_analytical_lattice_step_recovers_one_cell_from_distinct_banks() {
    let request = request();
    let result = refine_tof_multibank_lattice(&request.input, &options(24)).unwrap();

    let refined = result.lattice_phases[0].cell.a_angstrom;
    assert!((refined - request.truth_a).abs() < 2.0e-8, "a={refined}");
    assert!(result.metrics.rwp < 2.0e-7, "rwp={}", result.metrics.rwp);
    assert!(
        result
            .history
            .iter()
            .any(|record| record.scaled_lattice_step_norm > 0.0)
    );
    assert_eq!(result.banks.len(), 2);
    assert_eq!(
        result.banks[0].phases[0].d_spacing_angstrom(),
        result.banks[1].phases[0].d_spacing_angstrom()
    );
}

#[test]
fn fused_profile_lattice_column_matches_centered_cell_differences() {
    let request = request();
    let bank = &request.input.multibank.banks[0];
    let model = &request.input.lattice_phases[0];
    let base = calculate_tof_lebail_pattern(&bank.input, &options(1).lebail).unwrap();
    let values = model
        .parameterization()
        .values_from_cell(model.initial_cell())
        .unwrap();
    assert_eq!(values.len(), 1);
    let geometry = tof_lattice_geometry(
        model.parameterization(),
        model.initial_cell(),
        bank.input.phases[0].hkl(),
        bank.input.instrument,
    )
    .unwrap();
    let local = &base.accumulation.derivatives.local;
    let mut analytical = vec![0.0; bank.input.pattern.sample_count()];
    for reflection in 0..bank.input.phases[0].reflection_ids().len() {
        let begin = local.offsets[reflection];
        let end = local.offsets[reflection + 1];
        let start = local.starts[reflection];
        let chain = geometry.d_d_spacing_d_parameters[reflection];
        for active in begin..end {
            analytical[start + active - begin] +=
                local.values[active * local.parameter_count + 1] * chain;
        }
    }
    let step = 1.0e-7;
    let plus_cell = model
        .parameterization()
        .to_cell(&[values[0] + step])
        .unwrap();
    let minus_cell = model
        .parameterization()
        .to_cell(&[values[0] - step])
        .unwrap();
    let mut plus_input = bank.input.clone();
    plus_input.phases[0] = phase(
        model.parameterization(),
        plus_cell,
        bank.input.instrument,
        bank.input.phases[0].integrated_intensity().to_vec(),
        bank.input.phases[0].scale(),
    );
    let mut minus_input = bank.input.clone();
    minus_input.phases[0] = phase(
        model.parameterization(),
        minus_cell,
        bank.input.instrument,
        bank.input.phases[0].integrated_intensity().to_vec(),
        bank.input.phases[0].scale(),
    );
    let plus = calculate_tof_lebail_pattern(&plus_input, &options(1).lebail).unwrap();
    let minus = calculate_tof_lebail_pattern(&minus_input, &options(1).lebail).unwrap();
    let mut compared = 0;
    for (sample, analytical_value) in analytical.iter().copied().enumerate() {
        let finite = (plus.y[sample] - minus.y[sample]) / (2.0 * step);
        if base.profile_y[sample] < 1.0e-3 || finite.abs().max(analytical_value.abs()) < 1.0e-9 {
            continue;
        }
        let tolerance = 2.0e-7 + 5.0e-6 * finite.abs().max(analytical_value.abs());
        assert!(
            (analytical_value - finite).abs() < tolerance,
            "sample {sample}: {analytical_value} != {finite}"
        );
        compared += 1;
    }
    assert!(compared > 100);
}

#[test]
fn lattice_checkpoint_continuation_is_atomic_and_exact() {
    let request = request();
    let selected = options(10);
    let uninterrupted = refine_tof_multibank_lattice(&request.input, &selected).unwrap();
    let cancellation = CancellationToken::default();
    let requested = cancellation.clone();
    let checkpoints = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&checkpoints);
    let evaluations = 10 * (selected.max_lattice_backtracks + 4);
    let limits = RefinementLimits::new(10, evaluations, None, 1).unwrap();
    let mut runtime =
        RefinementRuntime::<TofMultiBankLatticeCheckpoint>::new(limits, Some(cancellation))
            .unwrap();
    runtime.set_checkpoint_sink(move |checkpoint: &TofMultiBankLatticeCheckpoint| {
        captured
            .lock()
            .unwrap()
            .push(checkpoint.completed_iterations);
        if checkpoint.completed_iterations == 3 {
            requested
                .request("atomic lattice stop")
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    });
    let stopped =
        refine_tof_multibank_lattice_with_runtime(&request.input, &selected, None, &mut runtime)
            .unwrap();
    assert_eq!(stopped.termination_reason, TerminationReason::Cancelled);
    assert_eq!(stopped.checkpoint.completed_iterations, 3);
    assert_eq!(stopped.checkpoint.banks.len(), 2);
    assert_eq!(stopped.checkpoint.lattice_phases.len(), 1);
    assert_eq!(*checkpoints.lock().unwrap(), [1, 2, 3]);

    let mut continuation = RefinementRuntime::new(limits, None).unwrap();
    let resumed = refine_tof_multibank_lattice_with_runtime(
        &request.input,
        &selected,
        Some(&stopped.checkpoint),
        &mut continuation,
    )
    .unwrap();
    assert_eq!(resumed.history, uninterrupted.history);
    assert_eq!(resumed.lattice_phases, uninterrupted.lattice_phases);
    assert_eq!(resumed.banks, uninterrupted.banks);
    assert_eq!(resumed.metrics, uninterrupted.metrics);
}

#[test]
fn lattice_model_rejects_missing_or_duplicate_phase_selection() {
    let request = request();
    let mut missing = request.input.clone();
    missing.lattice_phases[0] = TofSharedLatticePhase::new(
        id("missing"),
        missing.lattice_phases[0].parameterization().clone(),
        missing.lattice_phases[0].bounds().clone(),
        missing.lattice_phases[0].initial_cell(),
    )
    .unwrap();
    assert!(missing.validate().is_err());

    let mut duplicate = request.input.clone();
    duplicate
        .lattice_phases
        .push(duplicate.lattice_phases[0].clone());
    assert!(duplicate.validate().is_err());

    let mut mismatched = request.input;
    let original = &mismatched.lattice_phases[0];
    mismatched.lattice_phases[0] = TofSharedLatticePhase::new(
        original.phase_id().clone(),
        original.parameterization().clone(),
        original.bounds().clone(),
        cell(4.001),
    )
    .unwrap();
    assert!(mismatched.validate().is_err());
}
