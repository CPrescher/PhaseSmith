//! Selected analytical bank-local instrument refinement over fixed-cell TOF banks.

use std::sync::{Arc, Mutex};

use phasesmith_core::{TofInstrument, TofInstrumentParameter};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::{RecordId, TofPatternRecord};
use phasesmith_workflows::{
    CancellationToken, RefinementLimits, RefinementRuntime, TerminationReason,
    TofBankInstrumentModel, TofInstrumentParameterBound, TofLeBailBank, TofLeBailInput,
    TofLeBailOptions, TofLeBailPhase, TofMultiBankInput, TofMultiBankInstrumentCheckpoint,
    TofMultiBankInstrumentInput, TofMultiBankInstrumentOptions, calculate_tof_lebail_pattern,
    refine_tof_multibank_instrument, refine_tof_multibank_instrument_with_runtime,
};

fn id(value: &str) -> RecordId {
    RecordId::new(value).unwrap()
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

fn phase(intensities: Vec<f64>, scale: f64) -> TofLeBailPhase {
    TofLeBailPhase::new(
        id("alpha"),
        "fixed alpha",
        vec!["large".to_owned(), "middle".to_owned(), "small".to_owned()],
        vec![[1, 0, 0], [1, 1, 0], [2, 0, 0]],
        vec![1.8, 1.3, 0.95],
        intensities,
        scale,
    )
    .unwrap()
}

fn one_reflection_phase(intensity: f64) -> TofLeBailPhase {
    TofLeBailPhase::new(
        id("alpha"),
        "one-reflection alpha",
        vec!["one".to_owned()],
        vec![[1, 0, 0]],
        vec![1.3],
        vec![intensity],
        1.0,
    )
    .unwrap()
}

fn options(cycles: usize) -> TofMultiBankInstrumentOptions {
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
    TofMultiBankInstrumentOptions::new(lebail, 1.0e-10, 0.2, 8, 0.999_999).unwrap()
}

struct SyntheticRequest {
    input: TofMultiBankInstrumentInput,
    truth: Vec<TofInstrument>,
}

#[allow(clippy::too_many_lines)]
fn request() -> SyntheticRequest {
    let truth = vec![instrument(5_000.0, -0.7), instrument(4_400.0, 1.3)];
    let initial = [instrument(5_000.0, 1.6), instrument(4_394.0, 1.3)];
    let specs = [
        (
            "bank-1",
            (0..1_751)
                .map(|index| 3_500.0 + 4.0 * f64::from(index))
                .collect::<Vec<_>>(),
            vec![120.0, 75.0, 210.0],
            1.0,
            29,
        ),
        (
            "bank-2",
            (0..1_601)
                .map(|index| 3_000.0 + 4.0 * f64::from(index))
                .collect::<Vec<_>>(),
            vec![55.0, 180.0, 95.0],
            1.4,
            31,
        ),
    ];
    let mut banks = Vec::new();
    for (bank_index, (bank_id, grid, intensities, scale, mask_stride)) in
        specs.into_iter().enumerate()
    {
        let blank = TofPatternRecord::new(
            grid.clone(),
            Some(vec![0.0; grid.len()]),
            Some(vec![1.0; grid.len()]),
            None,
            Some(vec![0.25; grid.len()]),
        )
        .unwrap();
        let truth_input = TofLeBailInput::new(
            blank,
            truth[bank_index],
            vec![phase(intensities.clone(), scale)],
        )
        .unwrap();
        let observed = calculate_tof_lebail_pattern(&truth_input, &options(1).lebail)
            .unwrap()
            .y;
        let pattern = TofPatternRecord::new(
            grid,
            Some(observed),
            Some(vec![1.0; truth_input.pattern.sample_count()]),
            Some(
                (0..truth_input.pattern.sample_count())
                    .map(|index| index % mask_stride != 0)
                    .collect(),
            ),
            Some(vec![0.25; truth_input.pattern.sample_count()]),
        )
        .unwrap();
        banks.push(TofLeBailBank {
            bank_id: id(bank_id),
            input: TofLeBailInput::new(
                pattern,
                initial[bank_index],
                vec![phase(intensities, scale)],
            )
            .unwrap(),
        });
    }
    let models = vec![
        TofBankInstrumentModel::new(
            id("bank-1"),
            vec![
                TofInstrumentParameterBound::new(TofInstrumentParameter::Zero, -5.0, 5.0).unwrap(),
            ],
        )
        .unwrap(),
        TofBankInstrumentModel::new(
            id("bank-2"),
            vec![
                TofInstrumentParameterBound::new(TofInstrumentParameter::Difc, 4_300.0, 4_500.0)
                    .unwrap(),
            ],
        )
        .unwrap(),
    ];
    SyntheticRequest {
        input: TofMultiBankInstrumentInput {
            multibank: TofMultiBankInput { banks },
            instrument_models: models,
        },
        truth,
    }
}

#[test]
fn selected_local_instrument_columns_recover_distinct_bank_truth() {
    let request = request();
    let result = refine_tof_multibank_instrument(&request.input, &options(18)).unwrap();

    assert!((result.instruments[0].instrument.zero_us - request.truth[0].zero_us).abs() < 2.0e-7);
    assert!(
        (result.instruments[1].instrument.difc_us_per_angstrom
            - request.truth[1].difc_us_per_angstrom)
            .abs()
            < 2.0e-7
    );
    assert!(result.metrics.rwp < 2.0e-7, "rwp={}", result.metrics.rwp);
    assert_eq!(result.diagnostics.parameter_count, 2);
    assert_eq!(result.diagnostics.jacobian_rank, 2);
    assert_eq!(result.diagnostics.maximum_absolute_correlation, Some(0.0));
    assert!(result.diagnostics.unresolved_correlations.is_empty());
    assert!(result.history.iter().any(|row| {
        row.instrument_parameter_changes
            .iter()
            .any(|change| change.parameter == TofInstrumentParameter::Zero)
    }));
}

#[test]
fn instrument_checkpoint_continuation_is_atomic_and_exact() {
    let request = request();
    let selected = options(10);
    let uninterrupted = refine_tof_multibank_instrument(&request.input, &selected).unwrap();
    let cancellation = CancellationToken::default();
    let requested = cancellation.clone();
    let checkpoints = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&checkpoints);
    let evaluations = 10 * (selected.max_instrument_backtracks + 4);
    let limits = RefinementLimits::new(10, evaluations, None, 1).unwrap();
    let mut runtime =
        RefinementRuntime::<TofMultiBankInstrumentCheckpoint>::new(limits, Some(cancellation))
            .unwrap();
    runtime.set_checkpoint_sink(move |checkpoint: &TofMultiBankInstrumentCheckpoint| {
        captured
            .lock()
            .unwrap()
            .push(checkpoint.completed_iterations);
        if checkpoint.completed_iterations == 3 {
            requested
                .request("atomic instrument stop")
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    });
    let stopped =
        refine_tof_multibank_instrument_with_runtime(&request.input, &selected, None, &mut runtime)
            .unwrap();
    assert_eq!(stopped.termination_reason, TerminationReason::Cancelled);
    assert_eq!(stopped.checkpoint.completed_iterations, 3);
    assert_eq!(*checkpoints.lock().unwrap(), [1, 2, 3]);

    let mut continuation = RefinementRuntime::new(limits, None).unwrap();
    let resumed = refine_tof_multibank_instrument_with_runtime(
        &request.input,
        &selected,
        Some(&stopped.checkpoint),
        &mut continuation,
    )
    .unwrap();
    assert_eq!(resumed.history, uninterrupted.history);
    assert_eq!(resumed.instruments, uninterrupted.instruments);
    assert_eq!(resumed.banks, uninterrupted.banks);
    assert_eq!(resumed.metrics, uninterrupted.metrics);
}

#[test]
fn instrument_selection_rejects_missing_duplicate_and_out_of_bounds_models() {
    let request = request();
    let duplicate_bound =
        TofInstrumentParameterBound::new(TofInstrumentParameter::Zero, -10.0, 10.0).unwrap();
    assert!(
        TofBankInstrumentModel::new(id("bank-1"), vec![duplicate_bound, duplicate_bound],).is_err()
    );

    let mut missing = request.input.clone();
    missing.instrument_models[0] =
        TofBankInstrumentModel::new(id("missing"), vec![duplicate_bound]).unwrap();
    assert!(missing.validate().is_err());

    let mut outside = request.input;
    outside.instrument_models[0] = TofBankInstrumentModel::new(
        id("bank-1"),
        vec![TofInstrumentParameterBound::new(TofInstrumentParameter::Zero, -5.0, 0.0).unwrap()],
    )
    .unwrap();
    assert!(outside.validate().is_err());
}

#[test]
fn same_bank_column_correlations_are_reported_explicitly() {
    let mut request = request().input;
    request.instrument_models[0] = TofBankInstrumentModel::new(
        id("bank-1"),
        vec![
            TofInstrumentParameterBound::new(TofInstrumentParameter::Zero, -5.0, 5.0).unwrap(),
            TofInstrumentParameterBound::new(TofInstrumentParameter::Difc, 4_900.0, 5_100.0)
                .unwrap(),
        ],
    )
    .unwrap();
    request.instrument_models.truncate(1);
    let mut selected = options(2);
    selected.unresolved_correlation = 0.5;
    let result = refine_tof_multibank_instrument(&request, &selected).unwrap();

    assert_eq!(result.diagnostics.parameter_count, 2);
    assert_eq!(result.diagnostics.jacobian_rank, 2);
    assert!(
        result
            .diagnostics
            .maximum_absolute_correlation
            .is_some_and(|value| value > 0.5)
    );
    assert_eq!(result.diagnostics.unresolved_correlations.len(), 1);
    let pair = &result.diagnostics.unresolved_correlations[0];
    assert_eq!(pair.left_bank_id, id("bank-1"));
    assert_eq!(pair.left_parameter, TofInstrumentParameter::Zero);
    assert_eq!(pair.right_parameter, TofInstrumentParameter::Difc);
}

#[test]
fn rank_deficient_calibration_selection_is_returned_not_hidden() {
    let grid = (0..501)
        .map(|index| 5_800.0 + 2.0 * f64::from(index))
        .collect::<Vec<_>>();
    let instruments = [instrument(5_000.0, -0.7), instrument(5_000.0, 0.5)];
    let mut banks = Vec::new();
    for (index, instrument) in instruments.into_iter().enumerate() {
        let blank = TofPatternRecord::new(
            grid.clone(),
            Some(vec![0.0; grid.len()]),
            Some(vec![1.0; grid.len()]),
            None,
            None,
        )
        .unwrap();
        let truth =
            TofLeBailInput::new(blank, instrument, vec![one_reflection_phase(100.0)]).unwrap();
        let observed = calculate_tof_lebail_pattern(&truth, &options(1).lebail)
            .unwrap()
            .y;
        let pattern = TofPatternRecord::new(
            grid.clone(),
            Some(observed),
            Some(vec![1.0; grid.len()]),
            None,
            None,
        )
        .unwrap();
        banks.push(TofLeBailBank {
            bank_id: id(&format!("rank-bank-{}", index + 1)),
            input: TofLeBailInput::new(pattern, instrument, vec![one_reflection_phase(100.0)])
                .unwrap(),
        });
    }
    let selected = TofBankInstrumentModel::new(
        id("rank-bank-1"),
        vec![
            TofInstrumentParameterBound::new(TofInstrumentParameter::Zero, -5.0, 5.0).unwrap(),
            TofInstrumentParameterBound::new(TofInstrumentParameter::Difc, 4_900.0, 5_100.0)
                .unwrap(),
        ],
    )
    .unwrap();
    let input = TofMultiBankInstrumentInput {
        multibank: TofMultiBankInput { banks },
        instrument_models: vec![selected],
    };
    let result = refine_tof_multibank_instrument(&input, &options(1)).unwrap();

    assert_eq!(result.diagnostics.parameter_count, 2);
    assert_eq!(result.diagnostics.jacobian_rank, 1);
    assert!(
        result
            .diagnostics
            .maximum_absolute_correlation
            .is_some_and(|value| (value - 1.0).abs() < 1.0e-12)
    );
    assert_eq!(result.diagnostics.unresolved_correlations.len(), 1);
}
