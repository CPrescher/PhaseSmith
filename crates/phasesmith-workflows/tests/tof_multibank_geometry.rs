//! Joint shared-cell and bank-local instrument TOF refinement.

use std::sync::{Arc, Mutex};

use phasesmith_core::{TofInstrument, TofInstrumentParameter};
use phasesmith_crystallography::UnitCell;
use phasesmith_execution::ExecutionPolicy;
use phasesmith_io::space_group_by_number;
use phasesmith_model::{RecordId, TofPatternRecord};
use phasesmith_workflows::{
    CancellationToken, LatticeBounds, LatticeParameterization, RefinementLimits, RefinementRuntime,
    TerminationReason, TofBankInstrumentModel, TofGeometryParameterKey,
    TofInstrumentParameterBound, TofLeBailBank, TofLeBailInput, TofLeBailOptions, TofLeBailPhase,
    TofMultiBankGeometryCheckpoint, TofMultiBankGeometryInput, TofMultiBankGeometryOptions,
    TofMultiBankInput, TofMultiBankLatticeInput, TofSharedLatticePhase,
    calculate_tof_lebail_pattern, refine_tof_multibank_geometry,
    refine_tof_multibank_geometry_with_runtime, tof_lattice_geometry,
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

fn phase(
    parameterization: &LatticeParameterization,
    phase_cell: UnitCell,
    instrument: TofInstrument,
    intensities: Vec<f64>,
    scale: f64,
) -> TofLeBailPhase {
    let hkl = vec![[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0]];
    let geometry = tof_lattice_geometry(parameterization, phase_cell, &hkl, instrument).unwrap();
    TofLeBailPhase::new(
        id("alpha"),
        "shared alpha",
        vec![
            "100".to_owned(),
            "110".to_owned(),
            "111".to_owned(),
            "200".to_owned(),
        ],
        hkl,
        geometry.d_spacing_angstrom,
        intensities,
        scale,
    )
    .unwrap()
}

fn options(cycles: usize) -> TofMultiBankGeometryOptions {
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
    TofMultiBankGeometryOptions::new(lebail, 1.0e-10, 0.08, 10, 0.5).unwrap()
}

struct SyntheticRequest {
    input: TofMultiBankGeometryInput,
    truth_cell: UnitCell,
    truth_instruments: Vec<TofInstrument>,
}

#[allow(clippy::too_many_lines)]
fn request() -> SyntheticRequest {
    let initial_cell = cell(3.992);
    let truth_cell = cell(4.0);
    let group = space_group_by_number(221).unwrap().space_group;
    let parameterization = LatticeParameterization::new(group, initial_cell).unwrap();
    let bounds = LatticeBounds::around(&parameterization, 0.02, 1.0).unwrap();
    let lattice_phase =
        TofSharedLatticePhase::new(id("alpha"), parameterization.clone(), bounds, initial_cell)
            .unwrap();
    let truth_instruments = vec![instrument(5_000.0, -0.7), instrument(4_400.0, 1.3)];
    let initial_instruments = [instrument(5_000.0, 1.1), instrument(4_400.0, -0.2)];
    let specs = [
        (
            "bank-1",
            (0..2_301)
                .map(|index| 7_000.0 + 5.0 * f64::from(index))
                .collect::<Vec<_>>(),
            vec![120.0, 75.0, 210.0, 90.0],
            1.0,
            29,
        ),
        (
            "bank-2",
            (0..2_101)
                .map(|index| 6_000.0 + 5.0 * f64::from(index))
                .collect::<Vec<_>>(),
            vec![55.0, 180.0, 95.0, 140.0],
            1.4,
            31,
        ),
    ];
    let mut banks = Vec::new();
    let mut instrument_models = Vec::new();
    for (bank_index, (bank_id, grid, intensities, scale, mask_stride)) in
        specs.into_iter().enumerate()
    {
        let blank = TofPatternRecord::new(
            grid.clone(),
            Some(vec![0.0; grid.len()]),
            Some(vec![1.0; grid.len()]),
            None,
            None,
        )
        .unwrap();
        let truth = TofLeBailInput::new(
            blank,
            truth_instruments[bank_index],
            vec![phase(
                &parameterization,
                truth_cell,
                truth_instruments[bank_index],
                intensities.clone(),
                scale,
            )],
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
            None,
        )
        .unwrap();
        banks.push(TofLeBailBank {
            bank_id: id(bank_id),
            input: TofLeBailInput::new(
                pattern,
                initial_instruments[bank_index],
                vec![phase(
                    &parameterization,
                    initial_cell,
                    initial_instruments[bank_index],
                    intensities,
                    scale,
                )],
            )
            .unwrap(),
        });
        instrument_models.push(
            TofBankInstrumentModel::new(
                id(bank_id),
                vec![
                    TofInstrumentParameterBound::new(TofInstrumentParameter::Zero, -5.0, 5.0)
                        .unwrap(),
                ],
            )
            .unwrap(),
        );
    }
    SyntheticRequest {
        input: TofMultiBankGeometryInput {
            lattice: TofMultiBankLatticeInput {
                multibank: TofMultiBankInput { banks },
                lattice_phases: vec![lattice_phase],
            },
            instrument_models,
        },
        truth_cell,
        truth_instruments,
    }
}

#[test]
fn one_joint_system_recovers_shared_cell_and_local_zero_terms() {
    let request = request();
    let result = refine_tof_multibank_geometry(&request.input, &options(30)).unwrap();

    assert!(
        (result.lattice_phases[0].cell.a_angstrom - request.truth_cell.a_angstrom).abs() < 2.0e-7
    );
    for (actual, truth) in result.instruments.iter().zip(&request.truth_instruments) {
        assert!((actual.instrument.zero_us - truth.zero_us).abs() < 2.0e-7);
    }
    assert!(result.metrics.rwp < 3.0e-7, "rwp={}", result.metrics.rwp);
    assert_eq!(result.diagnostics.parameter_count, 3);
    assert_eq!(result.diagnostics.jacobian_rank, 3);
    assert!(
        result
            .diagnostics
            .unresolved_correlations
            .iter()
            .any(|pair| {
                matches!(pair.left, TofGeometryParameterKey::Lattice { .. })
                    && matches!(pair.right, TofGeometryParameterKey::Instrument { .. })
            })
    );
    assert!(result.history.iter().any(|row| {
        !row.lattice_parameter_changes.is_empty() && !row.instrument_parameter_changes.is_empty()
    }));
}

#[test]
fn joint_geometry_checkpoint_continuation_is_atomic_and_exact() {
    let request = request();
    let selected = options(12);
    let uninterrupted = refine_tof_multibank_geometry(&request.input, &selected).unwrap();
    let cancellation = CancellationToken::default();
    let requested = cancellation.clone();
    let checkpoints = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&checkpoints);
    let evaluations = 12 * (selected.max_geometry_backtracks + 4);
    let limits = RefinementLimits::new(12, evaluations, None, 1).unwrap();
    let mut runtime =
        RefinementRuntime::<TofMultiBankGeometryCheckpoint>::new(limits, Some(cancellation))
            .unwrap();
    runtime.set_checkpoint_sink(move |checkpoint: &TofMultiBankGeometryCheckpoint| {
        captured
            .lock()
            .unwrap()
            .push(checkpoint.completed_iterations);
        if checkpoint.completed_iterations == 4 {
            requested
                .request("atomic joint geometry stop")
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    });
    let stopped =
        refine_tof_multibank_geometry_with_runtime(&request.input, &selected, None, &mut runtime)
            .unwrap();
    assert_eq!(stopped.termination_reason, TerminationReason::Cancelled);
    assert_eq!(stopped.checkpoint.completed_iterations, 4);
    assert_eq!(*checkpoints.lock().unwrap(), [1, 2, 3, 4]);

    let mut continuation = RefinementRuntime::new(limits, None).unwrap();
    let resumed = refine_tof_multibank_geometry_with_runtime(
        &request.input,
        &selected,
        Some(&stopped.checkpoint),
        &mut continuation,
    )
    .unwrap();
    assert_eq!(resumed.history, uninterrupted.history);
    assert_eq!(resumed.lattice_phases, uninterrupted.lattice_phases);
    assert_eq!(resumed.instruments, uninterrupted.instruments);
    assert_eq!(resumed.banks, uninterrupted.banks);
    assert_eq!(resumed.metrics, uninterrupted.metrics);
    assert_eq!(resumed.diagnostics, uninterrupted.diagnostics);
}
