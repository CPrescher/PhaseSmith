//! Revision-owned desktop report export tests.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use phasesmith_desktop::{DesktopErrorCode, DesktopProjectStore};
use phasesmith_persistence::project_summary_json;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn temporary_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "phasesmith-desktop-report-{label}-{}-{}",
        std::process::id(),
        TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ))
}

#[test]
fn exports_the_exact_revision_as_canonical_json() {
    let store = DesktopProjectStore::new();
    store.create_project("project", "Report project").unwrap();
    let path = temporary_path("summary.json");

    let response = store.export_project_report(0, &path, false).unwrap();

    assert_eq!(response.revision, 0);
    assert!(PathBuf::from(&response.path).is_absolute());
    let expected = project_summary_json(&store.snapshot().unwrap().state().project).unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), expected);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn stale_and_existing_destinations_never_overwrite_caller_data() {
    let store = DesktopProjectStore::new();
    store.create_project("project", "Report project").unwrap();
    let path = temporary_path("protected.json");
    std::fs::write(&path, "caller-owned").unwrap();

    let stale = store.export_project_report(1, &path, true).unwrap_err();
    assert_eq!(stale.code, DesktopErrorCode::RevisionConflict);
    let protected = store.export_project_report(0, &path, false).unwrap_err();
    assert_eq!(protected.code, DesktopErrorCode::Persistence);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "caller-owned");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn explicit_overwrite_replaces_an_existing_report() {
    let store = DesktopProjectStore::new();
    store.create_project("project", "Report project").unwrap();
    let path = temporary_path("overwrite.json");
    std::fs::write(&path, "old report").unwrap();

    store.export_project_report(0, &path, true).unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("\"project_id\": \"project\""));
    assert!(!content.contains("old report"));
    std::fs::remove_file(path).unwrap();
}
