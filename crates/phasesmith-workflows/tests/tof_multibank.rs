//! Atomic shared-geometry multi-bank TOF Le Bail workflow tests.

use std::sync::{Arc, Mutex};

use phasesmith_core::TofInstrument;
use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::{RecordId, TofPatternRecord};
use phasesmith_workflows::{
    CancellationToken, RefinementEventKind, RefinementLimits, RefinementRuntime, TerminationReason,
    TofChebyshevBackground, TofLeBailBank, TofLeBailInput, TofLeBailOptions, TofLeBailPhase,
    TofMultiBankCheckpoint, TofMultiBankError, TofMultiBankInput, calculate_tof_lebail_pattern,
    refine_tof_multibank, refine_tof_multibank_with_runtime,
};

fn id(value: &str) -> RecordId {
    RecordId::new(value).unwrap()
}

fn instrument(difc: f64, zero: f64) -> TofInstrument {
    TofInstrument {
        zero_us: zero,
        difc_us_per_angstrom: difc,
        difa_us_per_angstrom2: -1.2,
        difb_us_angstrom: 0.5,
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
        "shared alpha",
        vec!["100".to_owned(), "110".to_owned(), "111".to_owned()],
        vec![[1, 0, 0], [1, 1, 0], [1, 1, 1]],
        vec![0.72, 0.93, 1.17],
        intensities,
        scale,
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

fn bank(
    bank_id: &str,
    instrument: TofInstrument,
    grid: Vec<f64>,
    truth_intensities: Vec<f64>,
    scale: f64,
    background_coefficients: &[f64],
    mask_stride: usize,
) -> TofLeBailBank {
    let domain = [grid[0], grid[grid.len() - 1]];
    let blank = TofPatternRecord::new(
        grid.clone(),
        Some(vec![0.0; grid.len()]),
        Some(vec![1.0; grid.len()]),
        None,
        Some(vec![0.25; grid.len()]),
    )
    .unwrap();
    let truth = TofLeBailInput::new(blank, instrument, vec![phase(truth_intensities, scale)])
        .unwrap()
        .with_refinable_background(
            TofChebyshevBackground::new(
                id(&format!("background-{bank_id}")),
                background_coefficients.to_vec(),
                domain,
            )
            .unwrap(),
        )
        .unwrap();
    let observed = calculate_tof_lebail_pattern(&truth, &options(1)).unwrap().y;
    let pattern = TofPatternRecord::new(
        grid,
        Some(observed),
        Some(vec![1.0; truth.pattern.sample_count()]),
        Some(
            (0..truth.pattern.sample_count())
                .map(|index| index % mask_stride != 0)
                .collect(),
        ),
        Some(vec![0.25; truth.pattern.sample_count()]),
    )
    .unwrap();
    let input = TofLeBailInput::new(pattern, instrument, vec![phase(vec![0.0; 3], scale)])
        .unwrap()
        .with_refinable_background(
            TofChebyshevBackground::new(
                id(&format!("background-{bank_id}")),
                vec![0.0; background_coefficients.len()],
                domain,
            )
            .unwrap(),
        )
        .unwrap();
    TofLeBailBank {
        bank_id: id(bank_id),
        input,
    }
}

fn request() -> TofMultiBankInput {
    let first_grid = (0..1_601)
        .map(|index| 3_100.0 + 2.1 * f64::from(index))
        .collect();
    let second_grid = (0..1_401)
        .map(|index| {
            let fraction = f64::from(index) / 1_400.0;
            2_750.0 + 2_650.0 * fraction.powf(1.12)
        })
        .collect();
    TofMultiBankInput {
        banks: vec![
            bank(
                "bank-1",
                instrument(5_000.0, -0.7),
                first_grid,
                vec![120.0, 75.0, 210.0],
                1.0,
                &[2.0, 0.2],
                29,
            ),
            bank(
                "bank-2",
                instrument(4_400.0, 1.3),
                second_grid,
                vec![55.0, 180.0, 95.0],
                1.4,
                &[1.2, -0.15, 0.04],
                31,
            ),
        ],
    }
}

#[test]
fn atomic_multibank_extraction_recovers_local_intensities_and_backgrounds() {
    let request = request();
    let result = refine_tof_multibank(&request, &options(110)).unwrap();

    assert_eq!(result.banks.len(), 2);
    assert_eq!(result.history.len(), 110);
    assert!(result.metrics.rwp < 2.0e-7, "rwp={}", result.metrics.rwp);
    let bank_chi_square: f64 = result
        .banks
        .iter()
        .map(|bank| bank.metrics.chi_square)
        .sum();
    assert!((result.metrics.chi_square - bank_chi_square).abs() < 1.0e-12);
    for (bank, expected) in result
        .banks
        .iter()
        .zip([[120.0, 75.0, 210.0], [55.0, 180.0, 95.0]])
    {
        assert!(
            bank.metrics.rwp < 3.0e-7,
            "{} rwp={}",
            bank.bank_id,
            bank.metrics.rwp
        );
        for (actual, expected) in bank.intensities.iter().zip(expected) {
            assert!(
                (actual.integrated_intensity - expected).abs() < 3.0e-4,
                "{} != {expected}",
                actual.integrated_intensity
            );
        }
    }
    assert_eq!(result.banks[0].bank_id, id("bank-1"));
    assert_eq!(result.banks[1].bank_id, id("bank-2"));
    assert_ne!(
        result.banks[0].phases[0].integrated_intensity(),
        result.banks[1].phases[0].integrated_intensity()
    );
    assert_eq!(
        result.banks[0].phases[0].d_spacing_angstrom(),
        result.banks[1].phases[0].d_spacing_angstrom()
    );
}

#[test]
fn cancellation_is_atomic_and_continuation_matches_uninterrupted() {
    let request = request();
    let selected = options(12);
    let uninterrupted = refine_tof_multibank(&request, &selected).unwrap();
    let cancellation = CancellationToken::default();
    let requested = cancellation.clone();
    let checkpoints = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&checkpoints);
    let limits = RefinementLimits::new(12, 36, None, 1).unwrap();
    let mut runtime =
        RefinementRuntime::<TofMultiBankCheckpoint>::new(limits, Some(cancellation)).unwrap();
    runtime.set_checkpoint_sink(move |checkpoint: &TofMultiBankCheckpoint| {
        captured
            .lock()
            .unwrap()
            .push(checkpoint.completed_iterations);
        if checkpoint.completed_iterations == 4 {
            requested
                .request("atomic stop")
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    });

    let stopped =
        refine_tof_multibank_with_runtime(&request, &selected, None, &mut runtime).unwrap();
    assert_eq!(stopped.termination_reason, TerminationReason::Cancelled);
    assert_eq!(stopped.checkpoint.completed_iterations, 4);
    assert_eq!(*checkpoints.lock().unwrap(), [1, 2, 3, 4]);
    assert_eq!(stopped.checkpoint.banks.len(), 2);

    let mut continuation = RefinementRuntime::new(limits, None).unwrap();
    let resumed = refine_tof_multibank_with_runtime(
        &request,
        &selected,
        Some(&stopped.checkpoint),
        &mut continuation,
    )
    .unwrap();
    assert_eq!(resumed.history, uninterrupted.history);
    assert_eq!(resumed.banks, uninterrupted.banks);
    assert_eq!(resumed.metrics, uninterrupted.metrics);
}

#[test]
fn shared_geometry_and_bank_identity_are_strictly_validated() {
    let mut mismatch = request();
    let phase = &mismatch.banks[1].input.phases[0];
    mismatch.banks[1].input.phases[0] = TofLeBailPhase::new(
        phase.phase_id().clone(),
        phase.name(),
        phase.reflection_ids().to_vec(),
        phase.hkl().to_vec(),
        vec![0.72, 0.93, 1.18],
        phase.integrated_intensity().to_vec(),
        phase.scale(),
    )
    .unwrap();
    assert!(matches!(
        mismatch.validate(),
        Err(TofMultiBankError::SharedTopologyMismatch { .. })
    ));

    let mut duplicate = request();
    duplicate.banks[1].bank_id = duplicate.banks[0].bank_id.clone();
    assert!(matches!(
        duplicate.validate(),
        Err(TofMultiBankError::DuplicateBankId { .. })
    ));

    let one = TofMultiBankInput {
        banks: vec![request().banks.remove(0)],
    };
    assert!(matches!(
        one.validate(),
        Err(TofMultiBankError::TooFewBanks)
    ));
}

#[test]
fn progress_events_report_one_joint_iteration_per_atomic_acceptance() {
    let request = request();
    let selected = options(3);
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&events);
    let limits = RefinementLimits::new(3, 9, None, 1).unwrap();
    let mut runtime = RefinementRuntime::<TofMultiBankCheckpoint>::new(limits, None).unwrap();
    runtime.set_event_sink(move |event: &phasesmith_workflows::RefinementEvent| {
        captured.lock().unwrap().push(event.kind());
        Ok(())
    });
    refine_tof_multibank_with_runtime(&request, &selected, None, &mut runtime).unwrap();
    let events = events.lock().unwrap();
    assert_eq!(events.first(), Some(&RefinementEventKind::Start));
    assert_eq!(events.last(), Some(&RefinementEventKind::Termination));
    assert_eq!(
        events
            .iter()
            .filter(|kind| **kind == RefinementEventKind::Iteration)
            .count(),
        3
    );
}
