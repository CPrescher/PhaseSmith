//! Detached native refinement job and event contract tests.

use std::collections::BTreeMap;
use std::sync::{Arc, Barrier, mpsc};
use std::time::{Duration, Instant};

use phasesmith_core::{ConstantWavelengthInstrument, OwnedCwContributions};
use phasesmith_crystallography::{IntegratedIntensityCorrectionModel, UnitCell};
use phasesmith_desktop::{
    CalculationManager, CalculationOptionsInput, DesktopErrorCode, DesktopEvent,
    DesktopProjectStore, JobManager, JobState, SeriesDtype,
};
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPhaseDefinition,
};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_io::space_group_by_number;
use phasesmith_model::{
    ExperimentRecord, FixedWavelengthSpectrum, HistogramRecord, PatternRecord, ProjectRecord,
    RadiationDefinition, RadiationProbe, RecordId, StructuralPhaseRecord,
};
use phasesmith_workflows::{
    RefinementLimits, RietveldAnalysis, RietveldCalculationOptions, RietveldCovarianceOptions,
    RietveldInput, RietveldParameterSelection, RietveldPhase, RietveldProjectState,
    RietveldRefinementOptions,
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

fn phase() -> RietveldPhase {
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
        hkl: vec![[1, 0, 0]],
        multiplicity: vec![6],
        fractional_xyz: vec![[0.0, 0.0, 0.0]],
        occupancy: vec![1.0],
        u_iso_angstrom2: vec![0.01],
        anisotropic_mask: vec![false],
        u_aniso_cif_angstrom2: vec![[0.0; 6]],
        scattering_species: vec!["Si".to_owned()],
        scattering_real_offset: Vec::new(),
        scattering_imag_offset: Vec::new(),
        scale: 1.0,
        coordinate_tolerance: 1.0e-10,
        scattering_model: BuiltInScatteringModel::XrayNonResonant,
        correction_model: IntegratedIntensityCorrectionModel::Neutral,
    };
    RietveldPhase::new_with_site_ids(
        RecordId::new("alpha").unwrap(),
        "Alpha",
        vec![RecordId::new("Si1").unwrap()],
        definition,
        OwnedCwContributions::neutral(1),
    )
    .unwrap()
}

fn options() -> RietveldRefinementOptions {
    RietveldRefinementOptions::new(
        RietveldCalculationOptions::new(20.0, true, ExecutionPolicy::new(Some(1), 1).unwrap())
            .unwrap(),
        RefinementLimits::new(4, 100, None, 10).unwrap(),
        1,
        1.0e-8,
        1.0e-7,
        1.0e-6,
        10.0,
        0.3,
        1.0e-6,
        20,
        0.25,
        4,
    )
    .unwrap()
}

fn project_state() -> RietveldProjectState {
    let phase = phase();
    let x_deg = (0..101)
        .map(|index| 20.0 + f64::from(index) * 0.1)
        .collect::<Vec<_>>();
    let pattern = PatternRecord::new(
        x_deg.clone(),
        Some(vec![0.0; x_deg.len()]),
        None,
        None,
        None,
    )
    .unwrap();
    let position = MonochromaticPositionCorrection {
        zero_shift_deg: 0.0,
        bragg_brentano_mm: None,
        debye_scherrer_micrometre: None,
    };
    let input = RietveldInput::new(
        pattern.clone(),
        instrument(),
        None,
        position,
        vec![phase.clone()],
    )
    .unwrap();
    RietveldProjectState {
        project: ProjectRecord {
            project_id: RecordId::new("project").unwrap(),
            revision: 99,
            name: "Runnable".to_owned(),
            histograms: vec![HistogramRecord {
                histogram_id: RecordId::new("histogram").unwrap(),
                name: "Histogram".to_owned(),
                pattern,
                experiment: ExperimentRecord::new(
                    instrument(),
                    RadiationDefinition::Monochromatic {
                        probe: RadiationProbe::Xray,
                        wavelength_angstrom: instrument().wavelength_angstrom,
                    },
                    None,
                    position,
                )
                .unwrap(),
                phase_ids: vec![phase.phase_id().clone()],
            }],
            phases: vec![StructuralPhaseRecord {
                phase_id: phase.phase_id().clone(),
                name: phase.name().to_owned(),
                definition: phase.definition().clone(),
                required_providers: Vec::new(),
            }],
            metadata: BTreeMap::new(),
        },
        analyses: vec![RietveldAnalysis {
            histogram_id: RecordId::new("histogram").unwrap(),
            input,
            selection: RietveldParameterSelection::default(),
            lattice_bounds: vec![None],
            constraints: Vec::new(),
            options: options(),
            covariance: RietveldCovarianceOptions::default(),
            checkpoint: None,
        }],
    }
}

fn runnable_store() -> DesktopProjectStore {
    let store = DesktopProjectStore::new();
    store.create_project("project", "Empty").unwrap();
    store.replace_project(0, project_state()).unwrap();
    store
}

fn receive_completion(receiver: &mpsc::Receiver<DesktopEvent>) -> DesktopEvent {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let event = receiver.recv_timeout(remaining).unwrap();
        if matches!(
            event,
            DesktopEvent::RefinementCompleted { .. } | DesktopEvent::RefinementFailed { .. }
        ) {
            return event;
        }
    }
}

#[test]
fn completion_emits_events_but_requires_explicit_revision_checked_acceptance() {
    let store = runnable_store();
    let (sender, receiver) = mpsc::channel();
    let jobs = JobManager::new(store.clone(), move |event: &DesktopEvent| {
        sender
            .send(event.clone())
            .map_err(|error| error.to_string())
    });

    let started = jobs.start_refinement(1, "histogram").unwrap();
    let completed = receive_completion(&receiver);
    assert!(matches!(
        completed,
        DesktopEvent::RefinementCompleted {
            job_id,
            project_revision: 1,
            ..
        } if job_id == started.job_id
    ));
    assert_eq!(store.snapshot().unwrap().revision(), 1);
    let status = jobs.job_status(started.job_id).unwrap();
    assert_eq!(status.state, JobState::Completed);
    assert!(status.outcome.is_some());
    assert!(serde_json::to_value(&status).is_ok());

    let accepted = jobs.accept_refinement(started.job_id, 1).unwrap();
    assert_eq!(accepted.accepted_revision, Some(2));
    assert_eq!(store.snapshot().unwrap().revision(), 2);
    assert!(
        store.snapshot().unwrap().state().analyses[0]
            .checkpoint
            .is_some()
    );
    jobs.discard_job(started.job_id).unwrap();
    assert_eq!(
        jobs.job_status(started.job_id).unwrap_err().code,
        DesktopErrorCode::UnknownJob
    );
}

#[test]
fn accepting_fixed_spectrum_refinement_preserves_the_radiation_contract() {
    let mut state = project_state();
    let spectrum = FixedWavelengthSpectrum::new(vec![1.5406, 1.5444], vec![1.0, 0.5]).unwrap();
    state.project.histograms[0].experiment.radiation = RadiationDefinition::FixedSpectrum {
        probe: RadiationProbe::Xray,
        spectrum: spectrum.clone(),
    };
    state.analyses[0].input = RietveldInput::new_fixed_spectrum(
        state.project.histograms[0].pattern.clone(),
        instrument(),
        spectrum.clone(),
        None,
        state.project.histograms[0].experiment.position_correction,
        vec![phase()],
    )
    .unwrap();
    state.validate().unwrap();
    let store = DesktopProjectStore::new();
    store.create_project("project", "Empty").unwrap();
    store.replace_project(0, state).unwrap();
    let (sender, receiver) = mpsc::channel();
    let jobs = JobManager::new(store.clone(), move |event: &DesktopEvent| {
        sender
            .send(event.clone())
            .map_err(|error| error.to_string())
    });

    let started = jobs.start_refinement(1, "histogram").unwrap();
    receive_completion(&receiver);
    jobs.accept_refinement(started.job_id, 1).unwrap();

    assert_eq!(
        store.snapshot().unwrap().state().project.histograms[0]
            .experiment
            .radiation,
        RadiationDefinition::FixedSpectrum {
            probe: RadiationProbe::Xray,
            spectrum,
        }
    );
}

#[test]
fn project_and_refinement_arrays_use_described_little_endian_binary_payloads() {
    let store = runnable_store();
    let project_series = store.project_series(1, "histogram").unwrap();
    assert_eq!(
        project_series
            .iter()
            .map(|value| value.series_id.as_str())
            .collect::<Vec<_>>(),
        ["x_deg", "observed_y", "fixed_background_y"]
    );
    assert!(serde_json::to_value(&project_series).is_ok());
    let x = store
        .project_series_payload(1, "histogram", "x_deg")
        .unwrap();
    assert_eq!(x.descriptor().dtype, SeriesDtype::Float64Le);
    assert_eq!(x.bytes().len(), x.descriptor().byte_length);
    assert_eq!(
        f64::from_le_bytes(x.bytes()[..8].try_into().unwrap()).to_bits(),
        20.0_f64.to_bits()
    );
    assert_eq!(
        store
            .project_series_payload(1, "histogram", "missing")
            .unwrap_err()
            .code,
        DesktopErrorCode::UnknownSeries
    );

    let (sender, receiver) = mpsc::channel();
    let jobs = JobManager::new(store, move |event: &DesktopEvent| {
        sender
            .send(event.clone())
            .map_err(|error| error.to_string())
    });
    let started = jobs.start_refinement(1, "histogram").unwrap();
    receive_completion(&receiver);
    let descriptors = jobs.refinement_series(started.job_id).unwrap();
    assert!(
        descriptors
            .iter()
            .any(|value| value.series_id == "calculated_y")
    );
    assert!(
        descriptors
            .iter()
            .any(|value| value.series_id == "phase/alpha/profile_y")
    );
    let mask = jobs
        .refinement_series_payload(started.job_id, "included_mask")
        .unwrap();
    assert_eq!(mask.descriptor().dtype, SeriesDtype::Uint8);
    assert!(mask.bytes().iter().all(|value| *value == 1));
    let calculated = jobs
        .refinement_series_payload(started.job_id, "calculated_y")
        .unwrap();
    assert_eq!(calculated.bytes().len(), 101 * 8);
    assert_eq!(calculated.into_bytes().len(), 101 * 8);
}

#[test]
fn stale_completion_never_overwrites_a_newer_edit() {
    let store = runnable_store();
    let (sender, receiver) = mpsc::channel();
    let jobs = JobManager::new(store.clone(), move |event: &DesktopEvent| {
        sender
            .send(event.clone())
            .map_err(|error| error.to_string())
    });
    let started = jobs.start_refinement(1, "histogram").unwrap();
    receive_completion(&receiver);
    let snapshot = store.snapshot().unwrap();
    let mut edited = snapshot.state().clone();
    edited.project.name = "Edited while refining".to_owned();
    store.replace_project(1, edited).unwrap();

    let error = jobs.accept_refinement(started.job_id, 2).unwrap_err();
    assert_eq!(error.code, DesktopErrorCode::RevisionConflict);
    assert_eq!(
        store.snapshot().unwrap().state().project.name,
        "Edited while refining"
    );
}

#[test]
fn cancellation_is_first_reason_wins_and_returns_a_normal_completion() {
    let store = runnable_store();
    let entered_start = Arc::new(Barrier::new(2));
    let release_start = Arc::new(Barrier::new(2));
    let (sender, receiver) = mpsc::channel();
    let entered = Arc::clone(&entered_start);
    let release = Arc::clone(&release_start);
    let jobs = JobManager::new(store, move |event: &DesktopEvent| {
        sender
            .send(event.clone())
            .map_err(|error| error.to_string())?;
        if matches!(
            event,
            DesktopEvent::RefinementProgress { event, .. } if event.kind == "start"
        ) {
            entered.wait();
            release.wait();
        }
        Ok(())
    });
    let started = jobs.start_refinement(1, "histogram").unwrap();
    entered_start.wait();
    assert_eq!(
        jobs.discard_job(started.job_id).unwrap_err().code,
        DesktopErrorCode::InvalidJobState
    );
    assert!(
        jobs.cancel_refinement(started.job_id, "user_requested")
            .unwrap()
            .first_request
    );
    assert!(
        !jobs
            .cancel_refinement(started.job_id, "ignored_second_reason")
            .unwrap()
            .first_request
    );
    release_start.wait();

    let completed = receive_completion(&receiver);
    assert!(matches!(
        completed,
        DesktopEvent::RefinementCompleted { outcome, .. }
            if outcome.termination_reason == "cancelled"
    ));
}

#[test]
fn lifecycle_close_cancels_running_jobs_and_releases_retained_arrays() {
    let store = runnable_store();
    let calculations = CalculationManager::new(store.clone());
    let calculation = calculations
        .calculate_histogram(1, "histogram", CalculationOptionsInput::default())
        .unwrap();
    let entered_start = Arc::new(Barrier::new(2));
    let release_start = Arc::new(Barrier::new(2));
    let entered = Arc::clone(&entered_start);
    let release = Arc::clone(&release_start);
    let jobs = JobManager::new(store.clone(), move |event: &DesktopEvent| {
        if matches!(
            event,
            DesktopEvent::RefinementProgress { event, .. } if event.kind == "start"
        ) {
            entered.wait();
            release.wait();
        }
        Ok(())
    });
    let started = jobs.start_refinement(1, "histogram").unwrap();
    entered_start.wait();

    let closed = store
        .close_project_with_cleanup(1, &jobs, &calculations)
        .unwrap();
    assert_eq!(closed.discarded_jobs, 1);
    assert_eq!(closed.discarded_calculations, 1);
    assert_eq!(
        store.snapshot().unwrap_err().code,
        DesktopErrorCode::NoProject
    );
    assert_eq!(
        jobs.job_status(started.job_id).unwrap_err().code,
        DesktopErrorCode::UnknownJob
    );
    assert_eq!(
        calculations
            .calculation_series(calculation.calculation_id)
            .unwrap_err()
            .code,
        DesktopErrorCode::UnknownCalculation
    );
    release_start.wait();
}

#[test]
fn duplicate_active_job_is_rejected_and_sink_failure_isolated() {
    let store = runnable_store();
    let entered_start = Arc::new(Barrier::new(2));
    let release_start = Arc::new(Barrier::new(2));
    let entered = Arc::clone(&entered_start);
    let release = Arc::clone(&release_start);
    let jobs = JobManager::new(store, move |event: &DesktopEvent| {
        if matches!(
            event,
            DesktopEvent::RefinementProgress { event, .. } if event.kind == "start"
        ) {
            entered.wait();
            release.wait();
        }
        Err("webview closed".to_owned())
    });
    let started = jobs.start_refinement(1, "histogram").unwrap();
    entered_start.wait();
    let duplicate = jobs.start_refinement(1, "histogram").unwrap_err();
    assert_eq!(duplicate.code, DesktopErrorCode::JobAlreadyRunning);
    release_start.wait();

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let status = jobs.job_status(started.job_id).unwrap();
        if status.state != JobState::Running {
            assert_eq!(status.state, JobState::Completed);
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
}

#[test]
fn invalid_histogram_and_unknown_job_are_structured() {
    let store = runnable_store();
    let jobs = JobManager::new(store, |_event: &DesktopEvent| Ok(()));
    assert_eq!(
        jobs.start_refinement(1, "missing").unwrap_err().code,
        DesktopErrorCode::UnknownAnalysis
    );
    assert_eq!(
        jobs.job_status(999).unwrap_err().code,
        DesktopErrorCode::UnknownJob
    );
}
