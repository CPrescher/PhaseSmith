//! Python-free application state and transport records for desktop hosts.
//!
//! This crate deliberately has no Tauri, webview, `PyO3`, `NumPy`, or `CPython`
//! dependency. A presentation adapter can expose these workflow-sized methods
//! as commands without owning scientific validation or mutable solver state.

#![forbid(unsafe_code)]

mod jobs;
mod series;

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::Path;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use phasesmith_model::{ProjectRecord, RecordId};
use phasesmith_persistence::{
    ProjectReadLimits, ProjectSaveOptions, ProjectSummaryReport, load_rietveld_project,
    save_rietveld_project,
};
use phasesmith_workflows::RietveldProjectState;
use serde::Serialize;

pub use jobs::{
    CancelJobResponse, DesktopEvent, DesktopEventSink, DiagnosticRecord, JobId, JobManager,
    JobStarted, JobState, JobStatus, RefinementEventRecord, RefinementOutcome,
};
pub use series::{BinaryPayload, BinarySeriesDescriptor, SeriesDtype, SeriesOwner};

/// Stable desktop-command failure category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DesktopErrorCode {
    /// No project is currently open.
    NoProject,
    /// A command targeted an obsolete project revision.
    RevisionConflict,
    /// A replacement attempted to change the stable project identity.
    ProjectIdentityMismatch,
    /// A revision could not be incremented.
    RevisionOverflow,
    /// A supplied project or record was invalid.
    InvalidProject,
    /// A requested histogram has no runnable native analysis.
    UnknownAnalysis,
    /// A requested refinement job does not exist.
    UnknownJob,
    /// A second job targeted an already-running analysis snapshot.
    JobAlreadyRunning,
    /// A job was not in the lifecycle state required by the command.
    InvalidJobState,
    /// Valid state requires a later native workflow capability.
    UnsupportedOperation,
    /// Native project persistence failed.
    Persistence,
    /// A requested binary display series does not exist.
    UnknownSeries,
    /// Internal shared state was poisoned by a panicking host callback.
    StateUnavailable,
}

/// Serializable, stable error returned by desktop workflow methods.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DesktopError {
    /// Machine-readable failure category.
    pub code: DesktopErrorCode,
    /// Human-readable failure description.
    pub message: String,
    /// Caller-supplied revision for a conflict.
    pub expected_revision: Option<u64>,
    /// Current revision for a conflict.
    pub actual_revision: Option<u64>,
}

impl DesktopError {
    pub(crate) fn simple(code: DesktopErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            expected_revision: None,
            actual_revision: None,
        }
    }

    pub(crate) fn conflict(expected_revision: u64, actual_revision: u64) -> Self {
        let message = if expected_revision == actual_revision {
            format!("project snapshot at revision {expected_revision} is no longer current")
        } else {
            format!(
                "project revision conflict: expected {expected_revision}, current revision is {actual_revision}"
            )
        };
        Self {
            code: DesktopErrorCode::RevisionConflict,
            message,
            expected_revision: Some(expected_revision),
            actual_revision: Some(actual_revision),
        }
    }

    /// Construct a stable host-infrastructure failure at an adapter boundary.
    #[must_use]
    pub fn host_failure(message: impl Into<String>) -> Self {
        Self::simple(DesktopErrorCode::StateUnavailable, message)
    }
}

impl Display for DesktopError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for DesktopError {}

/// Immutable project snapshot retained safely across asynchronous work.
#[derive(Clone, Debug)]
pub struct ProjectSnapshot {
    state: Arc<RietveldProjectState>,
}

impl ProjectSnapshot {
    /// Return the revision evaluated by this snapshot.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.state.project.revision
    }

    /// Borrow the complete validated native state.
    #[must_use]
    pub fn state(&self) -> &RietveldProjectState {
        &self.state
    }

    /// Clone the shared immutable state for a background worker.
    #[must_use]
    pub fn shared_state(&self) -> Arc<RietveldProjectState> {
        Arc::clone(&self.state)
    }

    pub(crate) fn same_instance(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }
}

/// Result of creating or loading a desktop project.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct OpenProjectResponse {
    /// Stable project ID.
    pub project_id: String,
    /// Installed project revision.
    pub revision: u64,
    /// Human-readable project name.
    pub name: String,
}

impl OpenProjectResponse {
    fn from_state(state: &RietveldProjectState) -> Self {
        Self {
            project_id: state.project.project_id.as_str().to_owned(),
            revision: state.project.revision,
            name: state.project.name.clone(),
        }
    }
}

/// Result of saving one exact immutable revision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SaveProjectResponse {
    /// Revision written to disk.
    pub revision: u64,
    /// Absolute destination returned by the native persistence codec.
    pub path: String,
}

#[derive(Default)]
struct StoreState {
    current: Option<Arc<RietveldProjectState>>,
}

/// Revision-checked owner of the current native desktop project.
#[derive(Clone, Default)]
pub struct DesktopProjectStore {
    inner: Arc<RwLock<StoreState>>,
}

impl DesktopProjectStore {
    /// Create an empty store with no open project.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create and install a new empty revision-zero project.
    ///
    /// # Errors
    ///
    /// Returns [`DesktopErrorCode::InvalidProject`] for an invalid ID or name.
    pub fn create_project(
        &self,
        project_id: &str,
        name: &str,
    ) -> Result<OpenProjectResponse, DesktopError> {
        let state = RietveldProjectState {
            project: ProjectRecord {
                project_id: RecordId::new(project_id).map_err(|error| {
                    DesktopError::simple(DesktopErrorCode::InvalidProject, error.to_string())
                })?,
                revision: 0,
                name: name.to_owned(),
                histograms: Vec::new(),
                phases: Vec::new(),
                metadata: BTreeMap::default(),
            },
            analyses: Vec::new(),
        };
        state.validate().map_err(|error| {
            DesktopError::simple(DesktopErrorCode::InvalidProject, error.to_string())
        })?;
        let response = OpenProjectResponse::from_state(&state);
        self.write()?.current = Some(Arc::new(state));
        Ok(response)
    }

    /// Load, fully validate, and install a native project from disk.
    ///
    /// A failed load leaves the currently open project unchanged.
    ///
    /// # Errors
    ///
    /// Returns a persistence or shared-state error.
    pub fn open_project(
        &self,
        path: impl AsRef<Path>,
        limits: ProjectReadLimits,
    ) -> Result<OpenProjectResponse, DesktopError> {
        let state = load_rietveld_project(path, limits).map_err(|error| {
            DesktopError::simple(DesktopErrorCode::Persistence, error.to_string())
        })?;
        let response = OpenProjectResponse::from_state(&state);
        self.write()?.current = Some(Arc::new(state));
        Ok(response)
    }

    /// Return an immutable snapshot of the current project.
    ///
    /// # Errors
    ///
    /// Returns [`DesktopErrorCode::NoProject`] when no project is open.
    pub fn snapshot(&self) -> Result<ProjectSnapshot, DesktopError> {
        let current = self.read()?.current.clone().ok_or_else(|| {
            DesktopError::simple(DesktopErrorCode::NoProject, "no project is open")
        })?;
        Ok(ProjectSnapshot { state: current })
    }

    /// Return a display-safe summary for the current immutable snapshot.
    ///
    /// # Errors
    ///
    /// Returns a no-project, validation, or persistence error.
    pub fn project_summary(&self) -> Result<ProjectSummaryReport, DesktopError> {
        let snapshot = self.snapshot()?;
        ProjectSummaryReport::from_project(&snapshot.state().project).map_err(|error| {
            DesktopError::simple(DesktopErrorCode::InvalidProject, error.to_string())
        })
    }

    /// List binary display-series descriptors for one histogram snapshot.
    ///
    /// # Errors
    ///
    /// Returns a project, revision, histogram, size, or shared-state error.
    pub fn project_series(
        &self,
        expected_revision: u64,
        histogram_id: &str,
    ) -> Result<Vec<BinarySeriesDescriptor>, DesktopError> {
        let snapshot = self.snapshot()?;
        if snapshot.revision() != expected_revision {
            return Err(DesktopError::conflict(
                expected_revision,
                snapshot.revision(),
            ));
        }
        series::project_descriptors(snapshot.state(), histogram_id)
    }

    /// Encode one project display series as an owned binary payload.
    ///
    /// # Errors
    ///
    /// Returns a project, revision, histogram, series, size, or shared-state error.
    pub fn project_series_payload(
        &self,
        expected_revision: u64,
        histogram_id: &str,
        series_id: &str,
    ) -> Result<BinaryPayload, DesktopError> {
        let snapshot = self.snapshot()?;
        if snapshot.revision() != expected_revision {
            return Err(DesktopError::conflict(
                expected_revision,
                snapshot.revision(),
            ));
        }
        series::project_payload(snapshot.state(), histogram_id, series_id)
    }

    /// Replace the current project only when the caller evaluated its exact revision.
    ///
    /// The adapter owns revision increments: `next.project.revision` is ignored and
    /// replaced with `expected_revision + 1` after identity and domain validation.
    ///
    /// # Errors
    ///
    /// Returns a revision conflict, identity mismatch, invalid-project, overflow,
    /// or shared-state error. A failed replacement never changes current state.
    pub fn replace_project(
        &self,
        expected_revision: u64,
        next: RietveldProjectState,
    ) -> Result<ProjectSnapshot, DesktopError> {
        let starting = self.snapshot()?;
        if starting.revision() != expected_revision {
            return Err(DesktopError::conflict(
                expected_revision,
                starting.revision(),
            ));
        }
        self.replace_snapshot(&starting, next)
    }

    /// Replace the current project only if an exact immutable snapshot is still current.
    ///
    /// This stronger compare-and-swap boundary prevents an asynchronous result
    /// from overwriting a project that was closed and reopened at the same numeric
    /// revision.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::replace_project`].
    pub fn replace_snapshot(
        &self,
        starting: &ProjectSnapshot,
        mut next: RietveldProjectState,
    ) -> Result<ProjectSnapshot, DesktopError> {
        let expected_revision = starting.revision();
        if next.project.project_id != starting.state().project.project_id {
            return Err(DesktopError::simple(
                DesktopErrorCode::ProjectIdentityMismatch,
                "replacement project ID differs from the current project",
            ));
        }
        next.project.revision = expected_revision.checked_add(1).ok_or_else(|| {
            DesktopError::simple(
                DesktopErrorCode::RevisionOverflow,
                "project revision cannot be incremented",
            )
        })?;
        next.validate().map_err(|error| {
            DesktopError::simple(DesktopErrorCode::InvalidProject, error.to_string())
        })?;
        let next = Arc::new(next);
        let mut store = self.write()?;
        let current = store.current.as_ref().ok_or_else(|| {
            DesktopError::simple(DesktopErrorCode::NoProject, "no project is open")
        })?;
        let actual_revision = current.project.revision;
        if actual_revision != expected_revision || !Arc::ptr_eq(current, &starting.state) {
            return Err(DesktopError::conflict(expected_revision, actual_revision));
        }
        store.current = Some(Arc::clone(&next));
        Ok(ProjectSnapshot { state: next })
    }

    /// Save the exact requested revision without holding the store lock during I/O.
    ///
    /// # Errors
    ///
    /// Returns a revision conflict, no-project, shared-state, or persistence error.
    pub fn save_project(
        &self,
        expected_revision: u64,
        path: impl AsRef<Path>,
        options: ProjectSaveOptions,
    ) -> Result<SaveProjectResponse, DesktopError> {
        let snapshot = self.snapshot()?;
        if snapshot.revision() != expected_revision {
            return Err(DesktopError::conflict(
                expected_revision,
                snapshot.revision(),
            ));
        }
        let path = save_rietveld_project(path, snapshot.state(), options).map_err(|error| {
            DesktopError::simple(DesktopErrorCode::Persistence, error.to_string())
        })?;
        Ok(SaveProjectResponse {
            revision: snapshot.revision(),
            path: path.to_string_lossy().into_owned(),
        })
    }

    /// Close the current project only at the requested revision.
    ///
    /// # Errors
    ///
    /// Returns a revision conflict, no-project, or shared-state error.
    pub fn close_project(&self, expected_revision: u64) -> Result<(), DesktopError> {
        let mut store = self.write()?;
        let actual_revision = store
            .current
            .as_ref()
            .ok_or_else(|| DesktopError::simple(DesktopErrorCode::NoProject, "no project is open"))?
            .project
            .revision;
        if actual_revision != expected_revision {
            return Err(DesktopError::conflict(expected_revision, actual_revision));
        }
        store.current = None;
        Ok(())
    }

    fn read(&self) -> Result<RwLockReadGuard<'_, StoreState>, DesktopError> {
        self.inner.read().map_err(|_| {
            DesktopError::simple(
                DesktopErrorCode::StateUnavailable,
                "desktop project state lock is poisoned",
            )
        })
    }

    fn write(&self) -> Result<RwLockWriteGuard<'_, StoreState>, DesktopError> {
        self.inner.write().map_err(|_| {
            DesktopError::simple(
                DesktopErrorCode::StateUnavailable,
                "desktop project state lock is poisoned",
            )
        })
    }
}
