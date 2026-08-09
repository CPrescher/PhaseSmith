//! Native powder import tests for the desktop adapter.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use phasesmith_desktop::{
    DesktopErrorCode, DesktopExperimentInput, DesktopPositionCorrectionInput, DesktopProjectStore,
    DesktopRadiationProbe, PowderFormatInput, PowderHistogramImportRequest,
};
use phasesmith_io::PowderReadLimits;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn temporary_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "phasesmith-desktop-import-{label}-{}-{}",
        std::process::id(),
        TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ))
}

fn experiment() -> DesktopExperimentInput {
    DesktopExperimentInput {
        probe: DesktopRadiationProbe::Xray,
        wavelength_angstrom: 1.5406,
        u_deg2: 0.01,
        v_deg2: 0.0,
        w_deg2: 0.01,
        x_deg: 0.01,
        y_deg: 0.0,
        axial_geometry: None,
        position_correction: DesktopPositionCorrectionInput::default(),
    }
}

fn request(path: PathBuf, histogram_id: &str) -> PowderHistogramImportRequest {
    PowderHistogramImportRequest {
        path,
        histogram_id: histogram_id.to_owned(),
        name: "Observed pattern".to_owned(),
        format: PowderFormatInput::Auto,
        bank: 1,
        experiment: experiment(),
        phase_ids: Vec::new(),
    }
}

#[test]
fn imports_columns_into_an_exact_new_revision() {
    let path = temporary_path("columns.xy");
    std::fs::write(&path, "10 100 2\n11 121 3\n12 144 4\n").unwrap();
    let store = DesktopProjectStore::new();
    store.create_project("project", "Imported").unwrap();

    let response = store
        .import_powder_histogram(0, request(path.clone(), "histogram-1"))
        .unwrap();

    assert_eq!(response.revision, 1);
    assert_eq!(response.histogram_id, "histogram-1");
    assert_eq!(response.sample_count, 3);
    assert_eq!(response.format, PowderFormatInput::Columns);
    assert_eq!(response.bank, None);
    let snapshot = store.snapshot().unwrap();
    let histogram = &snapshot.state().project.histograms[0];
    assert_eq!(histogram.pattern.x_deg, [10.0, 11.0, 12.0]);
    assert_eq!(
        histogram.pattern.observed_y.as_deref(),
        Some(&[100.0, 121.0, 144.0][..])
    );
    assert_eq!(
        histogram.pattern.uncertainty.as_deref(),
        Some(&[2.0, 3.0, 4.0][..])
    );
    assert_eq!(
        histogram
            .experiment
            .instrument
            .wavelength_angstrom
            .to_bits(),
        1.5406_f64.to_bits()
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn import_failures_and_stale_revisions_preserve_the_current_snapshot() {
    let path = temporary_path("invalid.xy");
    std::fs::write(&path, "10 1\n9 2\n").unwrap();
    let store = DesktopProjectStore::new();
    store.create_project("project", "Unchanged").unwrap();

    let stale = store
        .import_powder_histogram(1, request(path.clone(), "histogram-1"))
        .unwrap_err();
    assert_eq!(stale.code, DesktopErrorCode::RevisionConflict);
    let invalid = store
        .import_powder_histogram(0, request(path.clone(), "histogram-1"))
        .unwrap_err();
    assert_eq!(invalid.code, DesktopErrorCode::Import);
    assert_eq!(store.snapshot().unwrap().revision(), 0);
    assert!(
        store
            .snapshot()
            .unwrap()
            .state()
            .project
            .histograms
            .is_empty()
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn resource_limits_and_geometry_shape_are_checked_before_installation() {
    let path = temporary_path("limited.xy");
    std::fs::write(&path, "10 1\n11 2\n").unwrap();
    let store = DesktopProjectStore::new();
    store.create_project("project", "Limited").unwrap();
    let limited = store
        .import_powder_histogram_with_limits(
            0,
            request(path.clone(), "histogram-1"),
            PowderReadLimits {
                max_rows: 1,
                ..PowderReadLimits::default()
            },
        )
        .unwrap_err();
    assert_eq!(limited.code, DesktopErrorCode::Import);

    let mut malformed = request(path.clone(), "histogram-1");
    malformed
        .experiment
        .position_correction
        .sample_displacement_mm = Some(0.1);
    let geometry = store.import_powder_histogram(0, malformed).unwrap_err();
    assert_eq!(geometry.code, DesktopErrorCode::InvalidProject);
    assert!(
        store
            .snapshot()
            .unwrap()
            .state()
            .project
            .histograms
            .is_empty()
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn command_records_have_stable_snake_case_json() {
    let value = serde_json::to_value(request(PathBuf::from("pattern.fxye"), "hist")).unwrap();
    assert_eq!(value["format"], "auto");
    assert_eq!(value["experiment"]["probe"], "xray");
    let restored: PowderHistogramImportRequest = serde_json::from_value(value).unwrap();
    assert_eq!(restored.histogram_id, "hist");
}
