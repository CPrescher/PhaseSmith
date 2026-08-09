//! Thin Tauri command and event translation for the native desktop adapter.

#![forbid(unsafe_code)]
// Tauri's generated command ABI owns deserialized arguments and state guards.
#![allow(clippy::needless_pass_by_value)]

use phasesmith_desktop::{
    BinarySeriesDescriptor, CalculationManager, CalculationOptionsInput, CalculationResponse,
    CancelJobResponse, CifPhaseImportRequest, CifPhaseImportResponse, DesktopError, DesktopEvent,
    DesktopProjectStore, JobManager, JobStarted, JobStatus, OpenProjectResponse,
    PowderHistogramImportRequest, PowderHistogramImportResponse, SaveProjectResponse,
};
use phasesmith_persistence::{
    PROJECT_FORMAT_VERSION, ProjectReadLimits, ProjectSaveOptions, ProjectSummaryReport,
};
use serde::Serialize;
use tauri::ipc::Response;
use tauri::{AppHandle, Emitter, Manager, State};

const JOB_EVENT_NAME: &str = "phasesmith://refinement-event";

struct AppState {
    projects: DesktopProjectStore,
    jobs: JobManager,
    calculations: CalculationManager,
}

/// Static native-runtime information shown by the desktop diagnostics panel.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RuntimeInfo {
    /// Native project wire version.
    pub project_format_version: u32,
    /// Desktop runtime deliberately has no Python dependency.
    pub uses_python: bool,
    /// Stable event name used for job notifications.
    pub refinement_event_name: &'static str,
}

#[tauri::command]
fn runtime_info() -> RuntimeInfo {
    RuntimeInfo {
        project_format_version: PROJECT_FORMAT_VERSION,
        uses_python: false,
        refinement_event_name: JOB_EVENT_NAME,
    }
}

#[tauri::command]
fn create_project(
    state: State<'_, AppState>,
    project_id: String,
    name: String,
) -> Result<OpenProjectResponse, DesktopError> {
    state.projects.create_project(&project_id, &name)
}

#[tauri::command]
async fn open_project(
    state: State<'_, AppState>,
    path: String,
) -> Result<OpenProjectResponse, DesktopError> {
    let projects = state.projects.clone();
    tauri::async_runtime::spawn_blocking(move || {
        projects.open_project(path, ProjectReadLimits::default())
    })
    .await
    .map_err(|error| DesktopError::host_failure(format!("project-load task failed: {error}")))?
}

#[tauri::command]
fn project_summary(state: State<'_, AppState>) -> Result<ProjectSummaryReport, DesktopError> {
    state.projects.project_summary()
}

#[tauri::command]
async fn save_project(
    state: State<'_, AppState>,
    expected_revision: u64,
    path: String,
    overwrite: bool,
) -> Result<SaveProjectResponse, DesktopError> {
    let projects = state.projects.clone();
    tauri::async_runtime::spawn_blocking(move || {
        projects.save_project(expected_revision, path, ProjectSaveOptions { overwrite })
    })
    .await
    .map_err(|error| DesktopError::host_failure(format!("project-save task failed: {error}")))?
}

#[tauri::command]
async fn import_powder_histogram(
    state: State<'_, AppState>,
    expected_revision: u64,
    request: PowderHistogramImportRequest,
) -> Result<PowderHistogramImportResponse, DesktopError> {
    let projects = state.projects.clone();
    tauri::async_runtime::spawn_blocking(move || {
        projects.import_powder_histogram(expected_revision, request)
    })
    .await
    .map_err(|error| DesktopError::host_failure(format!("powder-import task failed: {error}")))?
}

#[tauri::command]
async fn import_cif_phase(
    state: State<'_, AppState>,
    expected_revision: u64,
    request: CifPhaseImportRequest,
) -> Result<CifPhaseImportResponse, DesktopError> {
    let projects = state.projects.clone();
    tauri::async_runtime::spawn_blocking(move || {
        projects.import_cif_phase(expected_revision, request)
    })
    .await
    .map_err(|error| DesktopError::host_failure(format!("CIF-import task failed: {error}")))?
}

#[tauri::command]
async fn calculate_histogram(
    state: State<'_, AppState>,
    expected_revision: u64,
    histogram_id: String,
    options: CalculationOptionsInput,
) -> Result<CalculationResponse, DesktopError> {
    let calculations = state.calculations.clone();
    tauri::async_runtime::spawn_blocking(move || {
        calculations.calculate_histogram(expected_revision, &histogram_id, options)
    })
    .await
    .map_err(|error| DesktopError::host_failure(format!("calculation task failed: {error}")))?
}

#[tauri::command]
fn calculation_series(
    state: State<'_, AppState>,
    calculation_id: u64,
) -> Result<Vec<BinarySeriesDescriptor>, DesktopError> {
    state.calculations.calculation_series(calculation_id)
}

#[tauri::command]
fn calculation_series_bytes(
    state: State<'_, AppState>,
    calculation_id: u64,
    series_id: String,
) -> Result<Response, DesktopError> {
    state
        .calculations
        .calculation_series_payload(calculation_id, &series_id)
        .map(|payload| Response::new(payload.into_bytes()))
}

#[tauri::command]
fn discard_calculation(
    state: State<'_, AppState>,
    calculation_id: u64,
) -> Result<(), DesktopError> {
    state.calculations.discard_calculation(calculation_id)
}

#[tauri::command]
fn close_project(state: State<'_, AppState>, expected_revision: u64) -> Result<(), DesktopError> {
    state.projects.close_project(expected_revision)
}

#[tauri::command]
fn project_series(
    state: State<'_, AppState>,
    expected_revision: u64,
    histogram_id: String,
) -> Result<Vec<BinarySeriesDescriptor>, DesktopError> {
    state
        .projects
        .project_series(expected_revision, &histogram_id)
}

#[tauri::command]
fn project_series_bytes(
    state: State<'_, AppState>,
    expected_revision: u64,
    histogram_id: String,
    series_id: String,
) -> Result<Response, DesktopError> {
    state
        .projects
        .project_series_payload(expected_revision, &histogram_id, &series_id)
        .map(|payload| Response::new(payload.into_bytes()))
}

#[tauri::command]
fn start_refinement(
    state: State<'_, AppState>,
    expected_revision: u64,
    histogram_id: String,
) -> Result<JobStarted, DesktopError> {
    state
        .jobs
        .start_refinement(expected_revision, &histogram_id)
}

#[tauri::command]
fn cancel_refinement(
    state: State<'_, AppState>,
    job_id: u64,
    reason: String,
) -> Result<CancelJobResponse, DesktopError> {
    state.jobs.cancel_refinement(job_id, &reason)
}

#[tauri::command]
fn job_status(state: State<'_, AppState>, job_id: u64) -> Result<JobStatus, DesktopError> {
    state.jobs.job_status(job_id)
}

#[tauri::command]
fn accept_refinement(
    state: State<'_, AppState>,
    job_id: u64,
    expected_revision: u64,
) -> Result<JobStatus, DesktopError> {
    state.jobs.accept_refinement(job_id, expected_revision)
}

#[tauri::command]
fn discard_job(state: State<'_, AppState>, job_id: u64) -> Result<(), DesktopError> {
    state.jobs.discard_job(job_id)
}

#[tauri::command]
fn refinement_series(
    state: State<'_, AppState>,
    job_id: u64,
) -> Result<Vec<BinarySeriesDescriptor>, DesktopError> {
    state.jobs.refinement_series(job_id)
}

#[tauri::command]
fn refinement_series_bytes(
    state: State<'_, AppState>,
    job_id: u64,
    series_id: String,
) -> Result<Response, DesktopError> {
    state
        .jobs
        .refinement_series_payload(job_id, &series_id)
        .map(|payload| Response::new(payload.into_bytes()))
}

/// Build and run the native `PhaseSmith` Tauri application.
///
/// # Panics
///
/// Panics only when Tauri cannot initialize or run the native application host.
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let handle: AppHandle = app.handle().clone();
            let projects = DesktopProjectStore::new();
            let calculations = CalculationManager::new(projects.clone());
            let jobs = JobManager::new(projects.clone(), move |event: &DesktopEvent| {
                handle
                    .emit(JOB_EVENT_NAME, event)
                    .map_err(|error| error.to_string())
            });
            app.manage(AppState {
                projects,
                jobs,
                calculations,
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            runtime_info,
            create_project,
            open_project,
            project_summary,
            save_project,
            import_powder_histogram,
            import_cif_phase,
            calculate_histogram,
            calculation_series,
            calculation_series_bytes,
            discard_calculation,
            close_project,
            project_series,
            project_series_bytes,
            start_refinement,
            cancel_refinement,
            job_status,
            accept_refinement,
            discard_job,
            refinement_series,
            refinement_series_bytes,
        ])
        .run(tauri::generate_context!())
        .expect("PhaseSmith Tauri runtime failed");
}

#[cfg(test)]
mod tests {
    use super::{JOB_EVENT_NAME, runtime_info};

    #[test]
    fn runtime_diagnostics_make_the_python_free_contract_explicit() {
        let info = runtime_info();
        assert!(!info.uses_python);
        assert_eq!(info.refinement_event_name, JOB_EVENT_NAME);
        assert!(info.project_format_version >= 2);
    }
}
