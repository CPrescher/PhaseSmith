//! Detached native refinement jobs and serializable desktop events.

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;

use phasesmith_model::{RadiationDefinition, RecordId};
use phasesmith_workflows::{
    CancellationToken, DiagnosticValue, RefinementEvent, RefinementRuntime,
    RietveldGeneralRefinementResult, RietveldProjectState, refine_general_rietveld_with_runtime,
};
use serde::Serialize;

use crate::{DesktopError, DesktopErrorCode, DesktopProjectStore, ProjectSnapshot};

/// Process-local stable refinement job identifier.
pub type JobId = u64;

/// JSON-compatible refinement diagnostic value.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum DiagnosticRecord {
    /// UTF-8 text.
    String(String),
    /// Boolean flag.
    Bool(bool),
    /// Signed integer.
    Integer(i64),
    /// Unsigned integer.
    Unsigned(u64),
    /// Finite floating-point value.
    Float(f64),
    /// JSON null.
    Null,
}

impl From<&DiagnosticValue> for DiagnosticRecord {
    fn from(value: &DiagnosticValue) -> Self {
        match value {
            DiagnosticValue::String(value) => Self::String(value.clone()),
            DiagnosticValue::Bool(value) => Self::Bool(*value),
            DiagnosticValue::Integer(value) => Self::Integer(*value),
            DiagnosticValue::Unsigned(value) => Self::Unsigned(*value),
            DiagnosticValue::Float(value) => Self::Float(*value),
            DiagnosticValue::Null => Self::Null,
        }
    }
}

/// Stable serialized view of one native solver event.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RefinementEventRecord {
    /// Stable event-kind label.
    pub kind: String,
    /// Workflow stage label.
    pub stage: String,
    /// Attempted iteration counter.
    pub attempted_iteration: usize,
    /// Accepted iteration counter.
    pub accepted_iterations: usize,
    /// Model-product/evaluation counter.
    pub evaluations: usize,
    /// Monotonic elapsed seconds.
    pub elapsed_seconds: f64,
    /// Human-readable event message.
    pub message: String,
    /// Stable ordered diagnostics.
    pub diagnostics: Vec<(String, DiagnosticRecord)>,
}

impl From<&RefinementEvent> for RefinementEventRecord {
    fn from(event: &RefinementEvent) -> Self {
        Self {
            kind: event.kind().as_str().to_owned(),
            stage: event.stage().to_owned(),
            attempted_iteration: event.attempted_iteration(),
            accepted_iterations: event.accepted_iterations(),
            evaluations: event.evaluations(),
            elapsed_seconds: event.elapsed_seconds(),
            message: event.message().to_owned(),
            diagnostics: event
                .diagnostics()
                .iter()
                .map(|(key, value)| (key.clone(), DiagnosticRecord::from(value)))
                .collect(),
        }
    }
}

/// Finite scalar summary of a completed native refinement.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RefinementOutcome {
    /// Stable bounded termination category.
    pub termination_reason: String,
    /// Accepted iteration count.
    pub completed_iterations: usize,
    /// Model-product/evaluation count for this invocation.
    pub evaluations: usize,
    /// Unweighted profile residual, omitted when not finite.
    pub rp: Option<f64>,
    /// Weighted profile residual, omitted when not finite.
    pub rwp: Option<f64>,
    /// Chi-square, omitted when not finite.
    pub chi_square: Option<f64>,
    /// Reduced chi-square, omitted when not finite.
    pub reduced_chi_square: Option<f64>,
    /// Final objective, omitted when not finite.
    pub objective: Option<f64>,
    /// Final weighted scaled-free Jacobian rank when evaluated.
    pub jacobian_rank: Option<usize>,
}

impl RefinementOutcome {
    fn from_result(result: &RietveldGeneralRefinementResult) -> Self {
        let finite = |value: f64| value.is_finite().then_some(value);
        Self {
            termination_reason: result.termination_reason.as_str().to_owned(),
            completed_iterations: result.checkpoint.completed_iterations,
            evaluations: result.evaluations,
            rp: finite(result.calculation.metrics.rp),
            rwp: finite(result.calculation.metrics.rwp),
            chi_square: finite(result.calculation.metrics.chi_square),
            reduced_chi_square: finite(result.calculation.metrics.reduced_chi_square),
            objective: finite(result.checkpoint.objective),
            jacobian_rank: result.jacobian_rank,
        }
    }
}

/// Event emitted by a detached desktop refinement job.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DesktopEvent {
    /// One structured native solver event.
    RefinementProgress {
        /// Process-local job ID.
        job_id: JobId,
        /// Immutable project revision evaluated by the job.
        project_revision: u64,
        /// Histogram being refined.
        histogram_id: String,
        /// Native event translated to JSON-safe records.
        event: RefinementEventRecord,
    },
    /// Refinement completed normally and is ready for explicit acceptance.
    RefinementCompleted {
        /// Process-local job ID.
        job_id: JobId,
        /// Immutable project revision evaluated by the job.
        project_revision: u64,
        /// Histogram that was refined.
        histogram_id: String,
        /// Finite scalar outcome.
        outcome: RefinementOutcome,
    },
    /// Refinement failed unexpectedly without changing project state.
    RefinementFailed {
        /// Process-local job ID.
        job_id: JobId,
        /// Immutable project revision evaluated by the job.
        project_revision: u64,
        /// Histogram that was refined.
        histogram_id: String,
        /// Human-readable native failure.
        message: String,
    },
}

/// Thread-safe event destination implemented by the eventual Tauri emitter.
pub trait DesktopEventSink: Send + Sync {
    /// Emit one immutable event or return a host-facing delivery failure.
    ///
    /// # Errors
    ///
    /// Returns a message when the presentation host cannot consume the event.
    fn emit(&self, event: &DesktopEvent) -> Result<(), String>;
}

impl<F> DesktopEventSink for F
where
    F: Fn(&DesktopEvent) -> Result<(), String> + Send + Sync,
{
    fn emit(&self, event: &DesktopEvent) -> Result<(), String> {
        self(event)
    }
}

/// Stable lifecycle state for one detached job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    /// Native refinement is executing.
    Running,
    /// A complete result is retained for explicit acceptance.
    Completed,
    /// Native refinement failed and retained no result.
    Failed,
}

/// Response returned immediately after starting a job.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct JobStarted {
    /// Process-local job ID.
    pub job_id: JobId,
    /// Stable project ID.
    pub project_id: String,
    /// Immutable project revision evaluated by the job.
    pub project_revision: u64,
    /// Histogram being refined.
    pub histogram_id: String,
}

/// Pollable job status containing no bulk numerical arrays.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct JobStatus {
    /// Process-local job ID.
    pub job_id: JobId,
    /// Immutable project revision evaluated by the job.
    pub project_revision: u64,
    /// Histogram being refined.
    pub histogram_id: String,
    /// Current lifecycle state.
    pub state: JobState,
    /// Complete outcome when successful.
    pub outcome: Option<RefinementOutcome>,
    /// Failure message when unsuccessful.
    pub error: Option<String>,
    /// Revision created by explicit result acceptance, when accepted.
    pub accepted_revision: Option<u64>,
}

/// Result of requesting cooperative cancellation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CancelJobResponse {
    /// Process-local job ID.
    pub job_id: JobId,
    /// Whether this call installed the first cancellation reason.
    pub first_request: bool,
}

struct JobRecord {
    source: ProjectSnapshot,
    histogram_id: RecordId,
    cancellation: CancellationToken,
    state: JobState,
    result: Option<Arc<RietveldGeneralRefinementResult>>,
    error: Option<String>,
    accepted_revision: Option<u64>,
}

/// Owns detached native refinement jobs above a revisioned project store.
#[derive(Clone)]
pub struct JobManager {
    projects: DesktopProjectStore,
    jobs: Arc<Mutex<BTreeMap<JobId, JobRecord>>>,
    next_job_id: Arc<AtomicU64>,
    event_sink: Arc<dyn DesktopEventSink>,
}

impl JobManager {
    /// Construct a manager with a thread-safe presentation event sink.
    #[must_use]
    pub fn new(projects: DesktopProjectStore, event_sink: impl DesktopEventSink + 'static) -> Self {
        Self {
            projects,
            jobs: Arc::new(Mutex::new(BTreeMap::new())),
            next_job_id: Arc::new(AtomicU64::new(1)),
            event_sink: Arc::new(event_sink),
        }
    }

    /// Start native refinement for one analysis in the exact requested snapshot.
    ///
    /// Completion retains a result but never edits current project state.
    ///
    /// # Errors
    ///
    /// Returns a project/revision, histogram, duplicate-job, state, or thread error.
    pub fn start_refinement(
        &self,
        expected_revision: u64,
        histogram_id: &str,
    ) -> Result<JobStarted, DesktopError> {
        let source = self.projects.snapshot()?;
        if source.revision() != expected_revision {
            return Err(DesktopError::conflict(expected_revision, source.revision()));
        }
        let histogram_id = RecordId::new(histogram_id).map_err(|error| {
            DesktopError::simple(DesktopErrorCode::InvalidProject, error.to_string())
        })?;
        let analysis = source
            .state()
            .analyses
            .iter()
            .find(|analysis| analysis.histogram_id == histogram_id)
            .cloned()
            .ok_or_else(|| {
                DesktopError::simple(
                    DesktopErrorCode::UnknownAnalysis,
                    format!("no native Rietveld analysis exists for histogram {histogram_id}"),
                )
            })?;
        analysis.validate().map_err(|error| {
            DesktopError::simple(DesktopErrorCode::InvalidProject, error.to_string())
        })?;
        let mut jobs = self.lock_jobs()?;
        if jobs.values().any(|job| {
            job.state == JobState::Running
                && job.source.same_instance(&source)
                && job.histogram_id == histogram_id
        }) {
            return Err(DesktopError::simple(
                DesktopErrorCode::JobAlreadyRunning,
                format!("a refinement job is already running for histogram {histogram_id}"),
            ));
        }
        let job_id = self
            .next_job_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| {
                DesktopError::simple(DesktopErrorCode::RevisionOverflow, "job ID space exhausted")
            })?;
        let cancellation = CancellationToken::default();
        jobs.insert(
            job_id,
            JobRecord {
                source: source.clone(),
                histogram_id: histogram_id.clone(),
                cancellation: cancellation.clone(),
                state: JobState::Running,
                result: None,
                error: None,
                accepted_revision: None,
            },
        );
        drop(jobs);

        let started = JobStarted {
            job_id,
            project_id: source.state().project.project_id.as_str().to_owned(),
            project_revision: source.revision(),
            histogram_id: histogram_id.as_str().to_owned(),
        };
        let jobs = Arc::clone(&self.jobs);
        let sink = Arc::clone(&self.event_sink);
        let thread_context = started.clone();
        let spawn = thread::Builder::new()
            .name(format!("phasesmith-refine-{job_id}"))
            .spawn(move || {
                run_job(JobExecution {
                    jobs,
                    sink,
                    context: thread_context,
                    analysis,
                    cancellation,
                });
            });
        if let Err(error) = spawn {
            self.lock_jobs()?.remove(&job_id);
            return Err(DesktopError::simple(
                DesktopErrorCode::StateUnavailable,
                format!("cannot start refinement worker: {error}"),
            ));
        }
        Ok(started)
    }

    /// Request cooperative cancellation of a running job.
    ///
    /// # Errors
    ///
    /// Returns an unknown/not-running job, invalid reason, or shared-state error.
    pub fn cancel_refinement(
        &self,
        job_id: JobId,
        reason: &str,
    ) -> Result<CancelJobResponse, DesktopError> {
        let jobs = self.lock_jobs()?;
        let job = jobs.get(&job_id).ok_or_else(|| {
            DesktopError::simple(
                DesktopErrorCode::UnknownJob,
                format!("unknown refinement job {job_id}"),
            )
        })?;
        if job.state != JobState::Running {
            return Err(DesktopError::simple(
                DesktopErrorCode::InvalidJobState,
                format!("refinement job {job_id} is not running"),
            ));
        }
        let first_request = job.cancellation.request(reason).map_err(|error| {
            DesktopError::simple(DesktopErrorCode::InvalidProject, error.to_string())
        })?;
        Ok(CancelJobResponse {
            job_id,
            first_request,
        })
    }

    /// Return the current status of one job.
    ///
    /// # Errors
    ///
    /// Returns an unknown-job or shared-state error.
    pub fn job_status(&self, job_id: JobId) -> Result<JobStatus, DesktopError> {
        let jobs = self.lock_jobs()?;
        let job = jobs.get(&job_id).ok_or_else(|| {
            DesktopError::simple(
                DesktopErrorCode::UnknownJob,
                format!("unknown refinement job {job_id}"),
            )
        })?;
        Ok(status(job_id, job))
    }

    /// Explicitly install one completed result if its exact source snapshot is current.
    ///
    /// # Errors
    ///
    /// Returns a job-state, revision, unsupported-sharing, validation, or state error.
    pub fn accept_refinement(
        &self,
        job_id: JobId,
        expected_revision: u64,
    ) -> Result<JobStatus, DesktopError> {
        let (source, histogram_id, result) = {
            let jobs = self.lock_jobs()?;
            let job = jobs.get(&job_id).ok_or_else(|| {
                DesktopError::simple(
                    DesktopErrorCode::UnknownJob,
                    format!("unknown refinement job {job_id}"),
                )
            })?;
            if job.state != JobState::Completed {
                return Err(DesktopError::simple(
                    DesktopErrorCode::InvalidJobState,
                    format!("refinement job {job_id} has no completed result"),
                ));
            }
            let result = job.result.as_ref().ok_or_else(|| {
                DesktopError::simple(
                    DesktopErrorCode::StateUnavailable,
                    format!("completed refinement job {job_id} retained no result"),
                )
            })?;
            (
                job.source.clone(),
                job.histogram_id.clone(),
                Arc::clone(result),
            )
        };
        let current = self.projects.snapshot()?;
        if current.revision() != expected_revision {
            return Err(DesktopError::conflict(
                expected_revision,
                current.revision(),
            ));
        }
        if source.revision() != expected_revision {
            return Err(DesktopError::conflict(
                source.revision(),
                current.revision(),
            ));
        }
        let mut next = source.state().clone();
        install_result(&mut next, &histogram_id, &result)?;
        let mut jobs = self.lock_jobs()?;
        let job = jobs.get_mut(&job_id).ok_or_else(|| {
            DesktopError::simple(
                DesktopErrorCode::UnknownJob,
                format!("unknown refinement job {job_id}"),
            )
        })?;
        let accepted = self.projects.replace_snapshot(&source, next)?;
        job.accepted_revision = Some(accepted.revision());
        Ok(status(job_id, job))
    }

    /// Remove a completed or failed job and release its retained numerical result.
    ///
    /// # Errors
    ///
    /// Returns an unknown/running-job or shared-state error.
    pub fn discard_job(&self, job_id: JobId) -> Result<(), DesktopError> {
        let mut jobs = self.lock_jobs()?;
        let job = jobs.get(&job_id).ok_or_else(|| {
            DesktopError::simple(
                DesktopErrorCode::UnknownJob,
                format!("unknown refinement job {job_id}"),
            )
        })?;
        if job.state == JobState::Running {
            return Err(DesktopError::simple(
                DesktopErrorCode::InvalidJobState,
                format!("running refinement job {job_id} cannot be discarded"),
            ));
        }
        jobs.remove(&job_id);
        Ok(())
    }

    fn lock_jobs(&self) -> Result<MutexGuard<'_, BTreeMap<JobId, JobRecord>>, DesktopError> {
        self.jobs.lock().map_err(|_| {
            DesktopError::simple(
                DesktopErrorCode::StateUnavailable,
                "refinement job state lock is poisoned",
            )
        })
    }
}

struct JobExecution {
    jobs: Arc<Mutex<BTreeMap<JobId, JobRecord>>>,
    sink: Arc<dyn DesktopEventSink>,
    context: JobStarted,
    analysis: phasesmith_workflows::RietveldAnalysis,
    cancellation: CancellationToken,
}

fn run_job(execution: JobExecution) {
    let JobExecution {
        jobs,
        sink,
        context,
        analysis,
        cancellation,
    } = execution;
    let execution = catch_unwind(AssertUnwindSafe(|| {
        let mut runtime = RefinementRuntime::new(analysis.options.limits, Some(cancellation))
            .map_err(|error| error.to_string())?;
        let progress_sink = Arc::clone(&sink);
        let progress_context = context.clone();
        runtime.set_event_sink(move |event: &RefinementEvent| {
            progress_sink.emit(&DesktopEvent::RefinementProgress {
                job_id: progress_context.job_id,
                project_revision: progress_context.project_revision,
                histogram_id: progress_context.histogram_id.clone(),
                event: RefinementEventRecord::from(event),
            })
        });
        refine_general_rietveld_with_runtime(
            &analysis.input,
            &analysis.selection,
            &analysis.lattice_bounds,
            &analysis.constraints,
            &analysis.options,
            analysis.covariance,
            analysis.checkpoint.as_ref(),
            &mut runtime,
        )
        .map_err(|error| error.to_string())
    }));
    let outcome = match execution {
        Ok(Ok(result)) => Ok(Arc::new(result)),
        Ok(Err(message)) => Err(message),
        Err(_) => Err("native refinement worker panicked".to_owned()),
    };
    let event = match outcome {
        Ok(result) => {
            let summary = RefinementOutcome::from_result(&result);
            if let Ok(mut jobs) = jobs.lock()
                && let Some(job) = jobs.get_mut(&context.job_id)
            {
                job.state = JobState::Completed;
                job.result = Some(result);
            }
            DesktopEvent::RefinementCompleted {
                job_id: context.job_id,
                project_revision: context.project_revision,
                histogram_id: context.histogram_id,
                outcome: summary,
            }
        }
        Err(message) => {
            if let Ok(mut jobs) = jobs.lock()
                && let Some(job) = jobs.get_mut(&context.job_id)
            {
                job.state = JobState::Failed;
                job.error = Some(message.clone());
            }
            DesktopEvent::RefinementFailed {
                job_id: context.job_id,
                project_revision: context.project_revision,
                histogram_id: context.histogram_id,
                message,
            }
        }
    };
    let _ = sink.emit(&event);
}

fn status(job_id: JobId, job: &JobRecord) -> JobStatus {
    JobStatus {
        job_id,
        project_revision: job.source.revision(),
        histogram_id: job.histogram_id.as_str().to_owned(),
        state: job.state,
        outcome: job.result.as_deref().map(RefinementOutcome::from_result),
        error: job.error.clone(),
        accepted_revision: job.accepted_revision,
    }
}

fn install_result(
    state: &mut RietveldProjectState,
    histogram_id: &RecordId,
    result: &RietveldGeneralRefinementResult,
) -> Result<(), DesktopError> {
    let phase_ids = result
        .input
        .phases
        .iter()
        .map(|phase| phase.phase_id().clone())
        .collect::<Vec<_>>();
    if state.analyses.iter().any(|analysis| {
        analysis.histogram_id != *histogram_id
            && analysis
                .input
                .phases
                .iter()
                .any(|phase| phase_ids.contains(phase.phase_id()))
    }) {
        return Err(DesktopError::simple(
            DesktopErrorCode::UnsupportedOperation,
            "accepting a phase shared by multiple histograms requires the joint-refinement model",
        ));
    }
    let analysis = state
        .analyses
        .iter_mut()
        .find(|analysis| analysis.histogram_id == *histogram_id)
        .ok_or_else(|| {
            DesktopError::simple(
                DesktopErrorCode::InvalidProject,
                format!("no native Rietveld analysis exists for histogram {histogram_id}"),
            )
        })?;
    analysis.input.clone_from(&result.input);
    analysis.checkpoint = Some(result.checkpoint.clone());

    let histogram = state
        .project
        .histograms
        .iter_mut()
        .find(|histogram| histogram.histogram_id == *histogram_id)
        .ok_or_else(|| {
            DesktopError::simple(
                DesktopErrorCode::InvalidProject,
                format!("project histogram {histogram_id} is missing"),
            )
        })?;
    histogram.experiment.instrument = result.input.instrument;
    histogram.experiment.axial_geometry = result.input.axial_geometry;
    histogram.experiment.position_correction = result.input.position_correction;
    let probe = histogram.experiment.radiation.probe();
    histogram.experiment.radiation = RadiationDefinition::Monochromatic {
        probe,
        wavelength_angstrom: result.input.instrument.wavelength_angstrom,
    };

    for phase in &result.input.phases {
        let stored = state
            .project
            .phases
            .iter_mut()
            .find(|stored| stored.phase_id == *phase.phase_id())
            .ok_or_else(|| {
                DesktopError::simple(
                    DesktopErrorCode::InvalidProject,
                    format!("project phase {} is missing", phase.phase_id()),
                )
            })?;
        phase.name().clone_into(&mut stored.name);
        stored.definition.clone_from(phase.definition());
    }
    Ok(())
}
