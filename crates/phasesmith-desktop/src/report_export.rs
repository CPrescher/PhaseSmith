//! Revision-owned native project report export.

use std::path::Path;

use phasesmith_persistence::{ProjectReportSaveOptions, write_project_summary_json_with_options};
use serde::Serialize;

use crate::{DesktopError, DesktopErrorCode, DesktopProjectStore};

/// Result of exporting one exact project summary revision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ReportExportResponse {
    /// Project revision represented by the report.
    pub revision: u64,
    /// Absolute report destination.
    pub path: String,
}

impl DesktopProjectStore {
    /// Export the stable array-free JSON summary for one exact revision.
    ///
    /// File I/O happens without holding the project-state lock. The report
    /// remains an honest snapshot if a later edit occurs while it is written.
    ///
    /// # Errors
    ///
    /// Returns a project/revision, validation, destination, filesystem, or
    /// shared-state error.
    pub fn export_project_report(
        &self,
        expected_revision: u64,
        path: impl AsRef<Path>,
        overwrite: bool,
    ) -> Result<ReportExportResponse, DesktopError> {
        let snapshot = self.snapshot()?;
        if snapshot.revision() != expected_revision {
            return Err(DesktopError::conflict(
                expected_revision,
                snapshot.revision(),
            ));
        }
        let absolute = std::path::absolute(path.as_ref()).map_err(|error| {
            DesktopError::simple(
                DesktopErrorCode::Persistence,
                format!("cannot resolve report destination: {error}"),
            )
        })?;
        write_project_summary_json_with_options(
            &snapshot.state().project,
            &absolute,
            ProjectReportSaveOptions { overwrite },
        )
        .map_err(|error| DesktopError::simple(DesktopErrorCode::Persistence, error.to_string()))?;
        Ok(ReportExportResponse {
            revision: snapshot.revision(),
            path: absolute.to_string_lossy().into_owned(),
        })
    }
}
