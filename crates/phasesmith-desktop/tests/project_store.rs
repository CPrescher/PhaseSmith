//! Revisioned desktop project-store contract tests.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use phasesmith_desktop::{DesktopErrorCode, DesktopProjectStore};
use phasesmith_model::{ProjectRecord, RecordId};
use phasesmith_persistence::{ProjectReadLimits, ProjectSaveOptions};
use phasesmith_workflows::RietveldProjectState;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn temporary_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "phasesmith-desktop-{label}-{}-{}",
        std::process::id(),
        TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ))
}

fn replacement(snapshot: &RietveldProjectState, name: &str) -> RietveldProjectState {
    let mut next = snapshot.clone();
    name.clone_into(&mut next.project.name);
    next
}

fn unrelated_project() -> RietveldProjectState {
    RietveldProjectState {
        project: ProjectRecord {
            project_id: RecordId::new("other").unwrap(),
            revision: 0,
            name: "Other".to_owned(),
            histograms: Vec::new(),
            phases: Vec::new(),
            metadata: BTreeMap::new(),
        },
        analyses: Vec::new(),
    }
}

#[test]
fn create_snapshot_summary_and_close_are_revision_checked() {
    let store = DesktopProjectStore::new();
    let missing = store.snapshot().unwrap_err();
    assert_eq!(missing.code, DesktopErrorCode::NoProject);

    let opened = store
        .create_project("desktop-project", "Desktop project")
        .unwrap();
    assert_eq!(opened.project_id, "desktop-project");
    assert_eq!(opened.revision, 0);
    let snapshot = store.snapshot().unwrap();
    let summary = store.project_summary().unwrap();
    assert_eq!(snapshot.revision(), 0);
    assert_eq!(summary.project_id, "desktop-project");
    assert_eq!(summary.revision, 0);

    let stale = store.close_project(1).unwrap_err();
    assert_eq!(stale.code, DesktopErrorCode::RevisionConflict);
    assert_eq!(stale.expected_revision, Some(1));
    assert_eq!(stale.actual_revision, Some(0));
    store.close_project(0).unwrap();
    assert_eq!(
        store.snapshot().unwrap_err().code,
        DesktopErrorCode::NoProject
    );
}

#[test]
fn replacement_increments_revision_and_keeps_old_snapshots_immutable() {
    let store = DesktopProjectStore::new();
    store.create_project("project", "Before").unwrap();
    let before = store.snapshot().unwrap();
    let after = store
        .replace_project(0, replacement(before.state(), "After"))
        .unwrap();

    assert_eq!(before.revision(), 0);
    assert_eq!(before.state().project.name, "Before");
    assert_eq!(after.revision(), 1);
    assert_eq!(after.state().project.name, "After");

    let stale = store
        .replace_project(0, replacement(before.state(), "Stale"))
        .unwrap_err();
    assert_eq!(stale.code, DesktopErrorCode::RevisionConflict);
    assert_eq!(store.snapshot().unwrap().state().project.name, "After");
}

#[test]
fn replacement_rejects_project_identity_changes_without_mutating_state() {
    let store = DesktopProjectStore::new();
    store.create_project("project", "Current").unwrap();
    let error = store.replace_project(0, unrelated_project()).unwrap_err();
    assert_eq!(error.code, DesktopErrorCode::ProjectIdentityMismatch);
    assert_eq!(store.snapshot().unwrap().state().project.name, "Current");
}

#[test]
fn exact_snapshot_replacement_rejects_close_and_reopen_at_the_same_revision() {
    let store = DesktopProjectStore::new();
    store.create_project("project", "Original").unwrap();
    let original = store.snapshot().unwrap();
    store.create_project("project", "Reopened").unwrap();

    let error = store
        .replace_snapshot(&original, replacement(original.state(), "Late result"))
        .unwrap_err();

    assert_eq!(error.code, DesktopErrorCode::RevisionConflict);
    assert_eq!(error.expected_revision, Some(0));
    assert_eq!(error.actual_revision, Some(0));
    assert_eq!(store.snapshot().unwrap().state().project.name, "Reopened");
}

#[test]
fn concurrent_replacements_allow_exactly_one_revision_winner() {
    let store = DesktopProjectStore::new();
    store.create_project("project", "Initial").unwrap();
    let initial = store.snapshot().unwrap().shared_state();
    let barrier = Arc::new(Barrier::new(3));
    let workers = ["First", "Second"].map(|name| {
        let store = store.clone();
        let initial = Arc::clone(&initial);
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            let next = replacement(&initial, name);
            barrier.wait();
            store.replace_project(0, next)
        })
    });
    barrier.wait();
    let outcomes = workers.map(|worker| worker.join().unwrap());
    assert_eq!(outcomes.iter().filter(|result| result.is_ok()).count(), 1);
    let error = outcomes.into_iter().find_map(Result::err).unwrap();
    assert_eq!(error.code, DesktopErrorCode::RevisionConflict);
    assert_eq!(store.snapshot().unwrap().revision(), 1);
}

#[test]
fn save_and_open_round_trip_an_exact_revision() {
    let store = DesktopProjectStore::new();
    store.create_project("project", "Saved").unwrap();
    let destination = temporary_path("round-trip");
    let saved = store
        .save_project(0, &destination, ProjectSaveOptions::default())
        .unwrap();
    assert_eq!(saved.revision, 0);
    assert!(std::path::Path::new(&saved.path).is_absolute());

    let stale = store
        .save_project(1, temporary_path("stale"), ProjectSaveOptions::default())
        .unwrap_err();
    assert_eq!(stale.code, DesktopErrorCode::RevisionConflict);

    let restored = DesktopProjectStore::new();
    let opened = restored
        .open_project(&destination, ProjectReadLimits::default())
        .unwrap();
    assert_eq!(opened.project_id, "project");
    assert_eq!(opened.revision, 0);
    assert_eq!(
        restored.snapshot().unwrap().state(),
        store.snapshot().unwrap().state()
    );
    std::fs::remove_dir_all(destination).unwrap();
}

#[test]
fn failed_open_preserves_the_current_project() {
    let store = DesktopProjectStore::new();
    store.create_project("project", "Still open").unwrap();
    let error = store
        .open_project(temporary_path("missing"), ProjectReadLimits::default())
        .unwrap_err();
    assert_eq!(error.code, DesktopErrorCode::Persistence);
    assert_eq!(store.snapshot().unwrap().state().project.name, "Still open");
}
