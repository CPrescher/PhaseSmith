//! Native format-3 TOF histogram and resumable Le Bail state coverage.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use phasesmith_core::TofInstrument;
use phasesmith_engine::crystallography::{
    IntegratedIntensityCorrectionModel, SpaceGroup, SymmetryOperation, UnitCell,
};
use phasesmith_engine::{BuiltInScatteringModel, StructuralPhaseDefinition};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::{
    ProjectRecord, RecordId, StructuralPhaseRecord, TofExperimentRecord, TofHistogramRecord,
    TofPatternRecord,
};
use phasesmith_persistence::{
    PROJECT_FORMAT_VERSION, PROJECT_MANIFEST_NAME, ProjectReadLimits, ProjectSaveOptions,
    ProjectSummaryReport, load_project, load_rietveld_project, load_tof_lebail_project,
    save_tof_lebail_project,
};
use phasesmith_workflows::{
    TofChebyshevBackground, TofLeBailAnalysis, TofLeBailInput, TofLeBailOptions, TofLeBailPhase,
    TofLeBailProjectState, TofProjectError, calculate_tof_lebail_pattern, refine_tof_lebail,
};

static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[test]
fn format_three_round_trips_tof_histogram_analysis_and_checkpoint() {
    let state = state();
    let directory = temporary_path("tof-round-trip");
    save_tof_lebail_project(&directory, &state, ProjectSaveOptions::default()).unwrap();

    let restored = load_tof_lebail_project(&directory, ProjectReadLimits::default()).unwrap();
    assert_eq!(restored, state);
    assert_eq!(
        load_project(&directory, ProjectReadLimits::default()).unwrap(),
        state.project
    );
    assert!(
        load_rietveld_project(&directory, ProjectReadLimits::default())
            .unwrap()
            .analyses
            .is_empty()
    );

    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(directory.join(PROJECT_MANIFEST_NAME)).unwrap()).unwrap();
    assert_eq!(manifest["format_version"], PROJECT_FORMAT_VERSION);
    assert_eq!(
        manifest["project"]["tof_histograms"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(manifest["tof_lebail_analyses"].as_array().unwrap().len(), 1);
    assert!(
        manifest["arrays"]
            .as_object()
            .unwrap()
            .keys()
            .any(|name| name.contains("checkpoint"))
    );

    let report = ProjectSummaryReport::from_project(&state.project).unwrap();
    assert_eq!(report.histogram_count, 1);
    assert_eq!(report.total_sample_count, 401);
    assert_eq!(report.histograms[0].coordinate_kind, "tof_us");
    assert_eq!(report.histograms[0].probe, "neutron");
    cleanup(directory);
}

#[test]
fn tof_project_rejects_duplicate_ownership_and_histogram_drift() {
    let mut duplicate = state();
    duplicate.analyses.push(duplicate.analyses[0].clone());
    assert!(matches!(
        duplicate.validate(),
        Err(TofProjectError::DuplicateAnalysis { .. })
    ));

    let mut drifted = state();
    drifted.analyses[0].input.pattern.background_y[0] = 1.0;
    assert!(matches!(
        drifted.validate(),
        Err(TofProjectError::HistogramStateMismatch { .. })
    ));
}

fn state() -> TofLeBailProjectState {
    let options = TofLeBailOptions::new(
        2,
        1.0,
        1.0e-12,
        1.0e-15,
        20.0,
        20.0,
        true,
        ExecutionPolicy::new(Some(1), 2).unwrap(),
    )
    .unwrap()
    .with_redistribution_uncertainty(false);
    let tof_us = (0..401)
        .map(|index| 3_000.0 + 10.0 * f64::from(index))
        .collect::<Vec<_>>();
    let blank = TofPatternRecord::new(
        tof_us.clone(),
        Some(vec![0.0; tof_us.len()]),
        Some(vec![1.0; tof_us.len()]),
        None,
        None,
    )
    .unwrap();
    let truth = TofLeBailInput::new(blank, instrument(), vec![tof_phase(vec![80.0, 120.0])])
        .unwrap()
        .with_refinable_background(
            TofChebyshevBackground::new(id("background"), vec![2.0, 0.2], [3_000.0, 7_000.0])
                .unwrap(),
        )
        .unwrap();
    let observed = calculate_tof_lebail_pattern(&truth, &options).unwrap().y;
    let pattern = TofPatternRecord::new(
        tof_us,
        Some(observed),
        Some(vec![1.0; 401]),
        Some((0..401).map(|index| index != 0).collect()),
        None,
    )
    .unwrap();
    let input = TofLeBailInput::new(pattern.clone(), instrument(), vec![tof_phase(vec![1.0; 2])])
        .unwrap()
        .with_refinable_background(
            TofChebyshevBackground::new(id("background"), vec![1.0, 0.0], [3_000.0, 7_000.0])
                .unwrap(),
        )
        .unwrap();
    let checkpoint = refine_tof_lebail(&input, &options).unwrap().checkpoint;
    let project = ProjectRecord {
        project_id: id("tof-project"),
        revision: 3,
        name: "TOF project".to_owned(),
        histograms: Vec::new(),
        tof_histograms: vec![TofHistogramRecord {
            histogram_id: id("bank-1"),
            name: "Bank 1".to_owned(),
            pattern,
            experiment: TofExperimentRecord::new(instrument()).unwrap(),
            phase_ids: vec![id("alpha")],
        }],
        phases: vec![structural_phase()],
        metadata: BTreeMap::from([("coordinate".to_owned(), "microseconds".to_owned())]),
    };
    TofLeBailProjectState {
        project,
        analyses: vec![TofLeBailAnalysis {
            histogram_id: id("bank-1"),
            input,
            options,
            checkpoint: Some(checkpoint),
        }],
    }
}

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

fn tof_phase(intensities: Vec<f64>) -> TofLeBailPhase {
    TofLeBailPhase::new(
        id("alpha"),
        "Phase alpha",
        vec!["100".to_owned(), "110".to_owned()],
        vec![[1, 0, 0], [1, 1, 0]],
        vec![0.72, 1.17],
        intensities,
        1.0,
    )
    .unwrap()
}

fn structural_phase() -> StructuralPhaseRecord {
    StructuralPhaseRecord {
        phase_id: id("alpha"),
        name: "Phase alpha".to_owned(),
        definition: StructuralPhaseDefinition {
            cell: UnitCell {
                a_angstrom: 5.0,
                b_angstrom: 5.0,
                c_angstrom: 5.0,
                alpha_deg: 90.0,
                beta_deg: 90.0,
                gamma_deg: 90.0,
            },
            space_group: SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap(),
            hkl: vec![[1, 0, 0], [1, 1, 0]],
            multiplicity: vec![2, 4],
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
            scattering_model: BuiltInScatteringModel::NeutronNuclear,
            correction_model: IntegratedIntensityCorrectionModel::Neutral,
        },
        required_providers: Vec::new(),
    }
}

fn id(value: &str) -> RecordId {
    RecordId::new(value).unwrap()
}

fn temporary_path(label: &str) -> PathBuf {
    let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "phasesmith-persistence-{label}-{}-{sequence}",
        std::process::id()
    ))
}

fn cleanup(path: PathBuf) {
    if path.exists() {
        fs::remove_dir_all(path).unwrap();
    }
}
