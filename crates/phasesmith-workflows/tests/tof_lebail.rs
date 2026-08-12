//! Fixed-instrument native TOF Le Bail workflow tests.

use std::sync::{Arc, Mutex};

use phasesmith_core::TofInstrument;
use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::{RecordId, TofPatternRecord};
use phasesmith_workflows::{
    CancellationToken, ChebyshevBackground, DifferentiableBackground, RefinementEventKind,
    RefinementLimits, RefinementRuntime, TerminationReason, TofChebyshevBackground,
    TofLeBailCheckpoint, TofLeBailInput, TofLeBailOptions, TofLeBailPhase,
    calculate_tof_lebail_pattern, refine_tof_lebail, refine_tof_lebail_with_runtime,
};

fn instrument() -> TofInstrument {
    TofInstrument {
        zero_us: -0.7,
        difc_us_per_angstrom: 5_000.0,
        difa_us_per_angstrom2: -1.5,
        difb_us_angstrom: 0.8,
        alpha_coefficient: 0.18,
        beta0_per_us: 0.04,
        beta1_angstrom4_per_us: 0.000_5,
        betaq_angstrom2_per_us: 0.001,
        sigma0_us2: 1.0,
        sigma1_us2_per_angstrom2: 12.0,
        sigma2_us2_per_angstrom4: 0.05,
        sigmaq_us2_per_angstrom: 0.2,
        x_us_per_angstrom: 0.3,
        y_us_per_angstrom2: 0.05,
        z_us: 0.4,
    }
}

fn phase(intensities: Vec<f64>) -> TofLeBailPhase {
    TofLeBailPhase::new(
        RecordId::new("phase-a").unwrap(),
        "synthetic phase",
        vec!["100".to_owned(), "110".to_owned(), "111".to_owned()],
        vec![[1, 0, 0], [1, 1, 0], [1, 1, 1]],
        vec![0.72, 0.93, 1.17],
        intensities,
        1.0,
    )
    .unwrap()
}

fn options(cycles: usize) -> TofLeBailOptions {
    TofLeBailOptions::new(
        cycles,
        1.0,
        1.0e-12,
        1.0e-15,
        20.0,
        20.0,
        true,
        ExecutionPolicy::new(Some(1), 2).unwrap(),
    )
    .unwrap()
}

fn synthetic_request() -> TofLeBailInput {
    let x = (0..1_401)
        .map(|index| {
            let fraction = f64::from(index) / 1_400.0;
            3_200.0 + 3_200.0 * fraction.powf(1.15)
        })
        .collect::<Vec<_>>();
    let blank =
        TofPatternRecord::new(x.clone(), Some(vec![0.0; x.len()]), None, None, None).unwrap();
    let truth =
        TofLeBailInput::new(blank, instrument(), vec![phase(vec![120.0, 75.0, 210.0])]).unwrap();
    let observed = calculate_tof_lebail_pattern(&truth, &options(1))
        .unwrap()
        .profile_y;
    let pattern = TofPatternRecord::new(x, Some(observed), None, None, None).unwrap();
    TofLeBailInput::new(pattern, instrument(), vec![phase(vec![0.0; 3])]).unwrap()
}

#[test]
fn synthetic_nonuniform_tof_extraction_recovers_intensities() {
    let x = (0..2_801)
        .map(|index| {
            let fraction = f64::from(index) / 2_800.0;
            3_200.0 + 3_200.0 * fraction.powf(1.15)
        })
        .collect::<Vec<_>>();
    let background = x
        .iter()
        .map(|value| 2.0 + 1.0e-4 * (value - 3_200.0))
        .collect::<Vec<_>>();
    let blank =
        TofPatternRecord::new(x.clone(), Some(vec![0.0; x.len()]), None, None, None).unwrap();
    let truth =
        TofLeBailInput::new(blank, instrument(), vec![phase(vec![120.0, 75.0, 210.0])]).unwrap();
    let truth_calculation = calculate_tof_lebail_pattern(&truth, &options(1)).unwrap();
    let observed = truth_calculation
        .profile_y
        .iter()
        .zip(&background)
        .map(|(profile, background)| profile + background)
        .collect::<Vec<_>>();
    let pattern = TofPatternRecord::new(
        x,
        Some(observed),
        Some(vec![1.0; background.len()]),
        None,
        Some(background),
    )
    .unwrap();
    let request = TofLeBailInput::new(pattern, instrument(), vec![phase(vec![0.0; 3])]).unwrap();

    let result = refine_tof_lebail(&request, &options(25)).unwrap();

    assert_eq!(result.history.len(), 25);
    assert!(result.background.is_none());
    assert!(result.calculation.background_basis.is_none());
    assert!(result.metrics.rwp < 1.0e-10, "rwp={}", result.metrics.rwp);
    let actual = result
        .intensities
        .iter()
        .map(|item| item.integrated_intensity)
        .collect::<Vec<_>>();
    for (actual, expected) in actual.iter().zip([120.0, 75.0, 210.0]) {
        assert!((actual - expected).abs() < 1.0e-7, "{actual} != {expected}");
    }
    assert_eq!(
        result.calculation.reflection_keys,
        [
            ("phase-a".to_owned(), "100".to_owned()),
            ("phase-a".to_owned(), "110".to_owned()),
            ("phase-a".to_owned(), "111".to_owned()),
        ]
    );
    assert_eq!(
        result
            .calculation
            .accumulation
            .derivatives
            .global
            .as_ref()
            .unwrap()
            .parameter_count,
        15
    );
}

#[test]
fn tof_workflow_rejects_angle_like_or_invalid_state_at_the_typed_boundary() {
    let pattern = TofPatternRecord::new(
        vec![1_000.0, 1_001.0],
        Some(vec![1.0, 1.0]),
        None,
        None,
        None,
    )
    .unwrap();
    assert!(TofLeBailInput::new(pattern, instrument(), Vec::new()).is_err());
    let pattern = TofPatternRecord::new(
        vec![1_000.0, 1_001.0],
        Some(vec![1.0, 1.0]),
        None,
        None,
        None,
    )
    .unwrap();
    let mut invalid_profile = instrument();
    invalid_profile.alpha_coefficient = -1.0;
    assert!(TofLeBailInput::new(pattern, invalid_profile, vec![phase(vec![1.0; 3])]).is_err());
    assert!(
        TofLeBailPhase::new(
            RecordId::new("phase").unwrap(),
            "phase",
            vec!["100".to_owned()],
            vec![[1, 0, 0]],
            vec![0.0],
            vec![1.0],
            1.0,
        )
        .is_err()
    );
    assert!(
        TofChebyshevBackground::new(
            RecordId::new("background").unwrap(),
            Vec::new(),
            [1_000.0, 1_001.0],
        )
        .is_err()
    );
    let outside = TofChebyshevBackground::new(
        RecordId::new("background").unwrap(),
        vec![0.0],
        [1_000.0, 1_000.5],
    )
    .unwrap();
    let pattern = TofPatternRecord::new(
        vec![1_000.0, 1_001.0],
        Some(vec![1.0, 1.0]),
        None,
        None,
        None,
    )
    .unwrap();
    assert!(
        TofLeBailInput::new(pattern, instrument(), vec![phase(vec![1.0; 3])])
            .unwrap()
            .with_refinable_background(outside)
            .is_err()
    );
}

#[test]
fn chebyshev_background_and_intensities_are_refined_together() {
    let x = (0..2_801)
        .map(|index| {
            let fraction = f64::from(index) / 2_800.0;
            3_200.0 + 3_200.0 * fraction.powf(1.15)
        })
        .collect::<Vec<_>>();
    let domain = [x[0], x[x.len() - 1]];
    let fixed_background = x
        .iter()
        .map(|value| 0.5 + 1.0e-4 * (value - domain[0]))
        .collect::<Vec<_>>();
    let blank = TofPatternRecord::new(
        x.clone(),
        Some(vec![0.0; x.len()]),
        None,
        None,
        Some(fixed_background.clone()),
    )
    .unwrap();
    let truth_background = TofChebyshevBackground::new(
        RecordId::new("tof-background").unwrap(),
        vec![2.2, 0.7, -0.25, 0.08],
        domain,
    )
    .unwrap();
    let truth = TofLeBailInput::new(blank, instrument(), vec![phase(vec![120.0, 75.0, 210.0])])
        .unwrap()
        .with_refinable_background(truth_background.clone())
        .unwrap();
    let truth_calculation = calculate_tof_lebail_pattern(&truth, &options(1)).unwrap();
    let pattern = TofPatternRecord::new(
        x,
        Some(truth_calculation.y),
        Some(vec![1.0; 2_801]),
        None,
        Some(fixed_background.clone()),
    )
    .unwrap();
    let starting_background = TofChebyshevBackground::new(
        RecordId::new("tof-background").unwrap(),
        vec![0.0; 4],
        domain,
    )
    .unwrap();
    let request = TofLeBailInput::new(pattern, instrument(), vec![phase(vec![0.0; 3])])
        .unwrap()
        .with_refinable_background(starting_background)
        .unwrap();

    let result = refine_tof_lebail(&request, &options(80)).unwrap();

    assert!(result.metrics.rwp < 1.0e-7, "rwp={}", result.metrics.rwp);
    let background = result.background.as_ref().unwrap();
    for (actual, expected) in background
        .coefficients()
        .iter()
        .zip(truth_background.coefficients())
    {
        assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
    }
    for (actual, expected) in result.intensities.iter().zip([120.0, 75.0, 210.0]) {
        assert!(
            (actual.integrated_intensity - expected).abs() < 1.0e-4,
            "{} != {expected}",
            actual.integrated_intensity
        );
    }
    let residual = background.calculate(&request.pattern.tof_us).unwrap();
    for ((actual, fixed), residual) in result
        .calculation
        .background_y
        .iter()
        .zip(fixed_background)
        .zip(residual)
    {
        assert!((actual - fixed - residual).abs() < 2.0e-12);
    }
    assert!(
        result
            .history
            .iter()
            .any(|cycle| cycle.maximum_absolute_background_change > 0.0)
    );
}

#[test]
fn chebyshev_coefficient_derivatives_match_centered_differences() {
    let x = vec![1_000.0, 1_250.0, 1_750.0, 2_000.0];
    let domain = [1_000.0, 2_000.0];
    let coefficients = vec![2.0, -0.3, 0.08, 0.02];
    let model = TofChebyshevBackground::new(
        RecordId::new("tof-background").unwrap(),
        coefficients.clone(),
        domain,
    )
    .unwrap();
    let pattern =
        TofPatternRecord::new(x, Some(vec![0.0; 4]), Some(vec![1.0; 4]), None, None).unwrap();
    let request = TofLeBailInput::new(pattern, instrument(), vec![phase(vec![1.0; 3])])
        .unwrap()
        .with_refinable_background(model)
        .unwrap();
    let calculation = calculate_tof_lebail_pattern(&request, &options(1)).unwrap();
    let independent = ChebyshevBackground::new("independent", coefficients.clone(), domain)
        .unwrap()
        .calculate(&request.pattern.tof_us)
        .unwrap();
    assert_eq!(calculation.background_y, independent);
    let basis = calculation.background_basis.unwrap();
    let step = 1.0e-6;
    for coefficient in 0..coefficients.len() {
        let mut plus = coefficients.clone();
        let mut minus = coefficients.clone();
        plus[coefficient] += step;
        minus[coefficient] -= step;
        let plus =
            TofChebyshevBackground::new(RecordId::new("tof-background").unwrap(), plus, domain)
                .unwrap()
                .calculate(&request.pattern.tof_us)
                .unwrap();
        let minus =
            TofChebyshevBackground::new(RecordId::new("tof-background").unwrap(), minus, domain)
                .unwrap()
                .calculate(&request.pattern.tof_us)
                .unwrap();
        for sample in 0..basis.rows {
            let finite = (plus[sample] - minus[sample]) / (2.0 * step);
            let analytical = basis.values[sample * basis.columns + coefficient];
            assert!((finite - analytical).abs() < 2.0e-10);
        }
    }
}

#[test]
fn cancellation_returns_an_accepted_checkpoint_and_resume_matches_uninterrupted() {
    let request = synthetic_request();
    let selected_options = options(12);
    let uninterrupted = refine_tof_lebail(&request, &selected_options).unwrap();
    let cancellation = CancellationToken::default();
    let requested = cancellation.clone();
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&events);
    let limits = RefinementLimits::new(12, 36, None, 1).unwrap();
    let mut runtime =
        RefinementRuntime::<TofLeBailCheckpoint>::new(limits, Some(cancellation)).unwrap();
    runtime.set_event_sink(move |event: &phasesmith_workflows::RefinementEvent| {
        captured.lock().unwrap().push(event.kind());
        Ok(())
    });
    runtime.set_checkpoint_sink(move |checkpoint: &TofLeBailCheckpoint| {
        if checkpoint.completed_iterations == 4 {
            requested
                .request("test stop")
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    });

    let stopped =
        refine_tof_lebail_with_runtime(&request, &selected_options, None, &mut runtime).unwrap();
    assert_eq!(stopped.termination_reason, TerminationReason::Cancelled);
    assert_eq!(stopped.history.len(), 4);
    assert_eq!(stopped.checkpoint.completed_iterations, 4);
    let event_kinds = events.lock().unwrap();
    assert_eq!(event_kinds.first(), Some(&RefinementEventKind::Start));
    assert_eq!(event_kinds.last(), Some(&RefinementEventKind::Termination));
    assert_eq!(
        event_kinds
            .iter()
            .filter(|kind| **kind == RefinementEventKind::Iteration)
            .count(),
        4
    );
    drop(event_kinds);

    let mut continuation = RefinementRuntime::new(limits, None).unwrap();
    let resumed = refine_tof_lebail_with_runtime(
        &request,
        &selected_options,
        Some(&stopped.checkpoint),
        &mut continuation,
    )
    .unwrap();
    assert_eq!(resumed.termination_reason, TerminationReason::MaxIterations);
    assert_eq!(resumed.history, uninterrupted.history);
    assert_eq!(resumed.intensities, uninterrupted.intensities);
    assert_eq!(resumed.calculation.y, uninterrupted.calculation.y);
}

#[test]
fn invalid_tof_checkpoint_is_rejected_before_continuation() {
    let request = synthetic_request();
    let selected_options = options(3);
    let result = refine_tof_lebail(&request, &selected_options).unwrap();
    let mut invalid = result.checkpoint;
    invalid.completed_iterations -= 1;
    let limits = RefinementLimits::new(3, 9, None, 1).unwrap();
    let mut runtime = RefinementRuntime::new(limits, None).unwrap();
    let error =
        refine_tof_lebail_with_runtime(&request, &selected_options, Some(&invalid), &mut runtime)
            .unwrap_err();
    assert!(error.to_string().contains("checkpoint iteration count"));
}
