//! Method-independent bounded runtime, cancellation, and structured events.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Stable refinement termination categories shared by every method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminationReason {
    /// Convergence criterion was met.
    Converged,
    /// Iteration budget was exhausted.
    MaxIterations,
    /// No included observations remain.
    NoObservations,
    /// Numerical evaluation or solve failed.
    NumericalFailure,
    /// Cooperative cancellation was requested.
    Cancelled,
    /// Wall-clock budget was exhausted.
    MaxRuntime,
    /// Model-evaluation budget was exhausted.
    MaxEvaluations,
    /// No improving progress could be made.
    Stagnated,
    /// Objective diverged under the method policy.
    Diverged,
    /// Consecutive rejected-step budget was exhausted.
    RepeatedRejections,
}

impl TerminationReason {
    /// Return the stable scripting/wire label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Converged => "converged",
            Self::MaxIterations => "max_iterations",
            Self::NoObservations => "no_observations",
            Self::NumericalFailure => "numerical_failure",
            Self::Cancelled => "cancelled",
            Self::MaxRuntime => "max_runtime",
            Self::MaxEvaluations => "max_evaluations",
            Self::Stagnated => "stagnated",
            Self::Diverged => "diverged",
            Self::RepeatedRejections => "repeated_rejections",
        }
    }
}

/// Stable machine-readable event categories.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefinementEventKind {
    /// Workflow started.
    Start,
    /// One candidate state was evaluated.
    Trial,
    /// Candidate was accepted.
    StepAccepted,
    /// Candidate was rejected.
    StepRejected,
    /// Iteration boundary completed.
    Iteration,
    /// Accepted-state checkpoint completed.
    Checkpoint,
    /// Recoverable warning.
    Warning,
    /// Normal bounded termination.
    Termination,
    /// Unexpected workflow failure.
    Failure,
}

impl RefinementEventKind {
    /// Return the stable scripting/wire label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Trial => "trial",
            Self::StepAccepted => "step_accepted",
            Self::StepRejected => "step_rejected",
            Self::Iteration => "iteration",
            Self::Checkpoint => "checkpoint",
            Self::Warning => "warning",
            Self::Termination => "termination",
            Self::Failure => "failure",
        }
    }
}

/// Finite JSON-scalar diagnostic value.
#[derive(Clone, Debug, PartialEq)]
pub enum DiagnosticValue {
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

impl DiagnosticValue {
    fn validate(&self) -> Result<(), RuntimeError> {
        if matches!(self, Self::Float(value) if !value.is_finite()) {
            return Err(RuntimeError::InvalidEvent {
                message: "floating-point diagnostics must be finite".to_owned(),
            });
        }
        Ok(())
    }
}

/// One immutable orchestration-boundary event.
#[derive(Clone, Debug, PartialEq)]
pub struct RefinementEvent {
    kind: RefinementEventKind,
    stage: String,
    attempted_iteration: usize,
    accepted_iterations: usize,
    evaluations: usize,
    elapsed_seconds: f64,
    message: String,
    diagnostics: Vec<(String, DiagnosticValue)>,
}

impl RefinementEvent {
    /// Validate and construct a finite machine-readable event.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::InvalidEvent`] for invalid labels, counters,
    /// elapsed time, duplicate/empty diagnostic keys, or non-finite floats.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: RefinementEventKind,
        stage: impl Into<String>,
        attempted_iteration: usize,
        accepted_iterations: usize,
        evaluations: usize,
        elapsed_seconds: f64,
        message: impl Into<String>,
        diagnostics: Vec<(String, DiagnosticValue)>,
    ) -> Result<Self, RuntimeError> {
        let stage = stage.into();
        let message = message.into();
        if stage.trim().is_empty() {
            return Err(invalid_event("event stage must be non-empty"));
        }
        if accepted_iterations > attempted_iteration {
            return Err(invalid_event(
                "accepted iterations cannot exceed attempted iterations",
            ));
        }
        if !elapsed_seconds.is_finite() || elapsed_seconds < 0.0 {
            return Err(invalid_event(
                "event elapsed time must be non-negative and finite",
            ));
        }
        if message.is_empty() {
            return Err(invalid_event("event message must be non-empty"));
        }
        let mut keys = BTreeSet::new();
        for (key, value) in &diagnostics {
            if key.is_empty() {
                return Err(invalid_event("diagnostic keys must be non-empty"));
            }
            if !keys.insert(key.clone()) {
                return Err(RuntimeError::InvalidEvent {
                    message: format!("duplicate diagnostic key {key:?}"),
                });
            }
            value.validate()?;
        }
        Ok(Self {
            kind,
            stage,
            attempted_iteration,
            accepted_iterations,
            evaluations,
            elapsed_seconds,
            message,
            diagnostics,
        })
    }

    /// Return the event category.
    #[must_use]
    pub const fn kind(&self) -> RefinementEventKind {
        self.kind
    }

    /// Borrow the stage label.
    #[must_use]
    pub fn stage(&self) -> &str {
        &self.stage
    }

    /// Return the attempted iteration counter.
    #[must_use]
    pub const fn attempted_iteration(&self) -> usize {
        self.attempted_iteration
    }

    /// Return the accepted iteration counter.
    #[must_use]
    pub const fn accepted_iterations(&self) -> usize {
        self.accepted_iterations
    }

    /// Return the evaluation counter.
    #[must_use]
    pub const fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// Return elapsed monotonic seconds.
    #[must_use]
    pub const fn elapsed_seconds(&self) -> f64 {
        self.elapsed_seconds
    }

    /// Borrow the human-readable message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Borrow diagnostics in stable insertion order.
    #[must_use]
    pub fn diagnostics(&self) -> &[(String, DiagnosticValue)] {
        &self.diagnostics
    }
}

#[derive(Debug, Default)]
struct CancellationState {
    requested: AtomicBool,
    reason: Mutex<Option<String>>,
}

/// Thread-safe first-request-wins cooperative cancellation token.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    state: Arc<CancellationState>,
}

impl CancellationToken {
    /// Request cancellation and return true only for the first request.
    ///
    /// The reason is stored before the atomic flag becomes visible, so another
    /// thread cannot observe cancellation without its first reason.
    ///
    /// # Errors
    ///
    /// Returns [`CancellationError`] for an empty reason or poisoned state lock.
    pub fn request(&self, reason: impl Into<String>) -> Result<bool, CancellationError> {
        let reason = reason.into();
        if reason.trim().is_empty() {
            return Err(CancellationError::InvalidReason);
        }
        let mut stored = self
            .state
            .reason
            .lock()
            .map_err(|_| CancellationError::Poisoned)?;
        if stored.is_some() {
            return Ok(false);
        }
        *stored = Some(reason);
        self.state.requested.store(true, Ordering::Release);
        Ok(true)
    }

    /// Return whether cancellation was requested.
    #[must_use]
    pub fn is_requested(&self) -> bool {
        self.state.requested.load(Ordering::Acquire)
    }

    /// Return the first cancellation reason.
    ///
    /// # Errors
    ///
    /// Returns [`CancellationError::Poisoned`] if the internal lock is poisoned.
    pub fn reason(&self) -> Result<Option<String>, CancellationError> {
        self.state
            .reason
            .lock()
            .map(|reason| reason.clone())
            .map_err(|_| CancellationError::Poisoned)
    }
}

/// Invalid cancellation token operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancellationError {
    /// Cancellation reason is empty or whitespace-only.
    InvalidReason,
    /// Internal reason lock was poisoned by a panicking thread.
    Poisoned,
}

impl Display for CancellationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidReason => formatter.write_str("cancellation reason must be non-empty"),
            Self::Poisoned => formatter.write_str("cancellation reason lock is poisoned"),
        }
    }
}

impl Error for CancellationError {}

/// Hard upper bounds independent of convergence criteria.
#[derive(Clone, Copy, Debug, PartialEq)]
// Field names intentionally mirror the stable Python and persistence contract.
#[allow(clippy::struct_field_names)]
pub struct RefinementLimits {
    max_iterations: usize,
    max_evaluations: usize,
    max_runtime_seconds: Option<f64>,
    max_consecutive_rejections: usize,
}

impl RefinementLimits {
    /// Validate strictly positive counters and optional positive finite time.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::InvalidLimits`] for zero counters or invalid time.
    pub fn new(
        max_iterations: usize,
        max_evaluations: usize,
        max_runtime_seconds: Option<f64>,
        max_consecutive_rejections: usize,
    ) -> Result<Self, RuntimeError> {
        if max_iterations == 0 || max_evaluations == 0 || max_consecutive_rejections == 0 {
            return Err(RuntimeError::InvalidLimits);
        }
        if max_runtime_seconds.is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0) {
            return Err(RuntimeError::InvalidLimits);
        }
        Ok(Self {
            max_iterations,
            max_evaluations,
            max_runtime_seconds,
            max_consecutive_rejections,
        })
    }

    /// Return the attempted-iteration ceiling.
    #[must_use]
    pub const fn max_iterations(self) -> usize {
        self.max_iterations
    }

    /// Return the model-evaluation ceiling.
    #[must_use]
    pub const fn max_evaluations(self) -> usize {
        self.max_evaluations
    }

    /// Return the optional elapsed-time ceiling.
    #[must_use]
    pub const fn max_runtime_seconds(self) -> Option<f64> {
        self.max_runtime_seconds
    }

    /// Return the consecutive-rejection ceiling.
    #[must_use]
    pub const fn max_consecutive_rejections(self) -> usize {
        self.max_consecutive_rejections
    }
}

impl Default for RefinementLimits {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            max_evaluations: 1_000,
            max_runtime_seconds: None,
            max_consecutive_rejections: 20,
        }
    }
}

/// Monotonic second source injectable for deterministic tests.
pub trait RuntimeClock: Send + Sync {
    /// Return a finite nondecreasing timestamp in seconds.
    fn now_seconds(&self) -> f64;
}

/// Process-local monotonic clock backed by [`Instant`].
#[derive(Debug)]
pub struct MonotonicClock {
    origin: Instant,
}

impl MonotonicClock {
    /// Start a new monotonic clock origin.
    #[must_use]
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeClock for MonotonicClock {
    fn now_seconds(&self) -> f64 {
        self.origin.elapsed().as_secs_f64()
    }
}

/// Synchronous event consumer called only at orchestration boundaries.
pub trait RefinementEventSink: Send {
    /// Consume one immutable event or return a host-facing failure message.
    ///
    /// # Errors
    ///
    /// Returns a message when the host cannot consume the event. The runtime
    /// isolates the failure and detaches this sink.
    fn emit(&mut self, event: &RefinementEvent) -> Result<(), String>;
}

impl<F> RefinementEventSink for F
where
    F: FnMut(&RefinementEvent) -> Result<(), String> + Send,
{
    fn emit(&mut self, event: &RefinementEvent) -> Result<(), String> {
        self(event)
    }
}

/// Application-owned durable sink for one typed accepted-state checkpoint.
pub trait CheckpointSink<C>: Send {
    /// Persist one complete checkpoint or return a host-facing failure message.
    ///
    /// # Errors
    ///
    /// Returns a message when durable checkpoint delivery fails.
    fn checkpoint(&mut self, checkpoint: &C) -> Result<(), String>;
}

impl<C, F> CheckpointSink<C> for F
where
    F: FnMut(&C) -> Result<(), String> + Send,
{
    fn checkpoint(&mut self, checkpoint: &C) -> Result<(), String> {
        self(checkpoint)
    }
}

/// Normal cooperative/budget termination returned at a safe boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RefinementStop {
    /// Stable termination category.
    pub reason: TerminationReason,
    /// Human-readable boundary message.
    pub message: String,
}

impl Display for RefinementStop {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for RefinementStop {}

/// Stateful method-independent refinement boundary guard.
pub struct RefinementRuntime<C = ()> {
    limits: RefinementLimits,
    cancellation: Option<CancellationToken>,
    event_sink: Option<Box<dyn RefinementEventSink>>,
    checkpoint_sink: Option<Box<dyn CheckpointSink<C>>>,
    clock: Arc<dyn RuntimeClock>,
    started_at: f64,
    attempted_iteration: usize,
    accepted_iterations: usize,
    evaluations: usize,
    consecutive_rejections: usize,
    event_sink_error: Option<String>,
}

impl<C> RefinementRuntime<C> {
    /// Construct a runtime with a process-local monotonic clock.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::InvalidClock`] if the clock origin is invalid.
    pub fn new(
        limits: RefinementLimits,
        cancellation: Option<CancellationToken>,
    ) -> Result<Self, RuntimeError> {
        Self::with_clock(limits, cancellation, Arc::new(MonotonicClock::new()))
    }

    /// Construct a runtime with an injectable thread-safe clock.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::InvalidClock`] if the first timestamp is non-finite.
    pub fn with_clock(
        limits: RefinementLimits,
        cancellation: Option<CancellationToken>,
        clock: Arc<dyn RuntimeClock>,
    ) -> Result<Self, RuntimeError> {
        let started_at = clock.now_seconds();
        if !started_at.is_finite() {
            return Err(RuntimeError::InvalidClock);
        }
        Ok(Self {
            limits,
            cancellation,
            event_sink: None,
            checkpoint_sink: None,
            clock,
            started_at,
            attempted_iteration: 0,
            accepted_iterations: 0,
            evaluations: 0,
            consecutive_rejections: 0,
            event_sink_error: None,
        })
    }

    /// Attach a synchronous event sink. Sink failures are isolated and recorded.
    pub fn set_event_sink(&mut self, sink: impl RefinementEventSink + 'static) {
        self.event_sink = Some(Box::new(sink));
        self.event_sink_error = None;
    }

    /// Attach an application-owned typed checkpoint sink.
    pub fn set_checkpoint_sink(&mut self, sink: impl CheckpointSink<C> + 'static) {
        self.checkpoint_sink = Some(Box::new(sink));
    }

    /// Restore accepted/attempted counters from a validated checkpoint.
    ///
    /// Evaluation and rejection budgets restart for the continuation call.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::InvalidResume`] if work already began or the
    /// completed count exceeds the iteration budget.
    pub fn resume_accepted(&mut self, completed_iterations: usize) -> Result<(), RuntimeError> {
        if self.attempted_iteration != 0
            || self.accepted_iterations != 0
            || self.evaluations != 0
            || completed_iterations > self.limits.max_iterations
        {
            return Err(RuntimeError::InvalidResume);
        }
        self.attempted_iteration = completed_iterations;
        self.accepted_iterations = completed_iterations;
        Ok(())
    }

    /// Return finite non-negative elapsed seconds.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::InvalidClock`] for non-finite or backward time.
    pub fn elapsed_seconds(&self) -> Result<f64, RuntimeError> {
        let elapsed = self.clock.now_seconds() - self.started_at;
        if !elapsed.is_finite() || elapsed < 0.0 {
            return Err(RuntimeError::InvalidClock);
        }
        Ok(elapsed)
    }

    /// Emit one validated event and isolate a failing event sink.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] for clock or event validation failures. A sink
    /// failure does not invalidate numerical state and is stored separately.
    pub fn emit(
        &mut self,
        kind: RefinementEventKind,
        stage: impl Into<String>,
        message: impl Into<String>,
        diagnostics: Vec<(String, DiagnosticValue)>,
    ) -> Result<RefinementEvent, RuntimeError> {
        let event = RefinementEvent::new(
            kind,
            stage,
            self.attempted_iteration,
            self.accepted_iterations,
            self.evaluations,
            self.elapsed_seconds()?,
            message,
            diagnostics,
        )?;
        let sink_result = self.event_sink.as_mut().map(|sink| sink.emit(&event));
        if let Some(Err(message)) = sink_result {
            self.event_sink_error = Some(message);
            self.event_sink = None;
        }
        Ok(event)
    }

    /// Stop at a safe boundary for cancellation, time, or evaluation budget.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Stopped`] for a normal bounded stop, or a clock/
    /// cancellation-state error.
    pub fn check_boundary(&self) -> Result<(), RuntimeError> {
        if let Some(token) = &self.cancellation
            && token.is_requested()
        {
            let message = token
                .reason()
                .map_err(RuntimeError::Cancellation)?
                .unwrap_or_else(|| "user requested cancellation".to_owned());
            return Err(RuntimeError::Stopped(RefinementStop {
                reason: TerminationReason::Cancelled,
                message,
            }));
        }
        if let Some(limit) = self.limits.max_runtime_seconds {
            let elapsed = self.elapsed_seconds()?;
            if elapsed >= limit {
                return Err(RuntimeError::Stopped(RefinementStop {
                    reason: TerminationReason::MaxRuntime,
                    message: "runtime limit reached".to_owned(),
                }));
            }
        }
        if self.evaluations >= self.limits.max_evaluations {
            return Err(RuntimeError::Stopped(RefinementStop {
                reason: TerminationReason::MaxEvaluations,
                message: "model-evaluation limit reached".to_owned(),
            }));
        }
        Ok(())
    }

    /// Begin one strictly increasing attempted iteration.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] for order violations or a normal bounded stop.
    pub fn begin_iteration(&mut self, attempted_iteration: usize) -> Result<(), RuntimeError> {
        if attempted_iteration <= self.attempted_iteration {
            return Err(RuntimeError::InvalidIterationOrder);
        }
        if attempted_iteration > self.limits.max_iterations {
            return Err(RuntimeError::Stopped(RefinementStop {
                reason: TerminationReason::MaxIterations,
                message: "iteration limit reached".to_owned(),
            }));
        }
        self.attempted_iteration = attempted_iteration;
        self.check_boundary()
    }

    /// Reserve one model evaluation after checking every safe-boundary budget.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] for a normal bounded stop or invalid clock/token.
    pub fn begin_evaluation(&mut self) -> Result<(), RuntimeError> {
        self.check_boundary()?;
        self.evaluations = self
            .evaluations
            .checked_add(1)
            .ok_or(RuntimeError::CounterOverflow)?;
        Ok(())
    }

    /// Accept at most one step in the current attempted iteration.
    ///
    /// State becomes accepted before the optional checkpoint sink is called, so
    /// a sink failure never rolls numerical state backward.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] for acceptance-order, counter, checkpoint, event,
    /// or clock failures.
    pub fn accept_step(&mut self, checkpoint: Option<&C>) -> Result<(), RuntimeError> {
        if self.accepted_iterations >= self.attempted_iteration {
            return Err(RuntimeError::DuplicateAcceptance);
        }
        self.accepted_iterations = self
            .accepted_iterations
            .checked_add(1)
            .ok_or(RuntimeError::CounterOverflow)?;
        self.consecutive_rejections = 0;
        if let Some(checkpoint) = checkpoint
            && let Some(sink) = self.checkpoint_sink.as_mut()
        {
            sink.checkpoint(checkpoint)
                .map_err(|message| RuntimeError::CheckpointSink { message })?;
            self.emit(
                RefinementEventKind::Checkpoint,
                "checkpoint",
                "accepted-state checkpoint completed",
                Vec::new(),
            )?;
        }
        Ok(())
    }

    /// Record a rejection and enforce its consecutive budget.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Stopped`] when the rejection limit is reached.
    pub fn reject_step(&mut self) -> Result<(), RuntimeError> {
        self.consecutive_rejections = self
            .consecutive_rejections
            .checked_add(1)
            .ok_or(RuntimeError::CounterOverflow)?;
        if self.consecutive_rejections >= self.limits.max_consecutive_rejections {
            return Err(RuntimeError::Stopped(RefinementStop {
                reason: TerminationReason::RepeatedRejections,
                message: "consecutive rejected-step limit reached".to_owned(),
            }));
        }
        Ok(())
    }

    /// Return the attempted iteration counter.
    #[must_use]
    pub const fn attempted_iteration(&self) -> usize {
        self.attempted_iteration
    }

    /// Return the accepted iteration counter.
    #[must_use]
    pub const fn accepted_iterations(&self) -> usize {
        self.accepted_iterations
    }

    /// Return the model-evaluation counter.
    #[must_use]
    pub const fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// Return consecutive rejected steps since the last acceptance.
    #[must_use]
    pub const fn consecutive_rejections(&self) -> usize {
        self.consecutive_rejections
    }

    /// Borrow the isolated event-sink error, if any.
    #[must_use]
    pub fn event_sink_error(&self) -> Option<&str> {
        self.event_sink_error.as_deref()
    }

    /// Return whether an event sink remains attached.
    #[must_use]
    pub const fn has_event_sink(&self) -> bool {
        self.event_sink.is_some()
    }
}

/// Invalid runtime configuration, event, control order, or host callback.
#[derive(Debug)]
pub enum RuntimeError {
    /// Limits are not strictly positive/finite.
    InvalidLimits,
    /// Clock is non-finite or moved backward.
    InvalidClock,
    /// Event record is invalid.
    InvalidEvent {
        /// Stable explanation.
        message: String,
    },
    /// Checkpoint counters cannot be restored in the current state.
    InvalidResume,
    /// Attempted iterations did not increase strictly.
    InvalidIterationOrder,
    /// More than one accepted step was recorded for an attempted iteration.
    DuplicateAcceptance,
    /// A runtime counter overflowed.
    CounterOverflow,
    /// Cancellation token state failed.
    Cancellation(CancellationError),
    /// A normal cancellation or hard-budget stop.
    Stopped(RefinementStop),
    /// Application checkpoint sink failed after state acceptance.
    CheckpointSink {
        /// Sink-provided failure message.
        message: String,
    },
}

impl Display for RuntimeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLimits => {
                formatter.write_str("refinement limits must be positive and finite")
            }
            Self::InvalidClock => {
                formatter.write_str("refinement clock must be finite and monotonic")
            }
            Self::InvalidEvent { message } | Self::CheckpointSink { message } => {
                formatter.write_str(message)
            }
            Self::InvalidResume => {
                formatter.write_str("refinement counters cannot be resumed in this state")
            }
            Self::InvalidIterationOrder => {
                formatter.write_str("attempted iterations must increase strictly")
            }
            Self::DuplicateAcceptance => {
                formatter.write_str("at most one step may be accepted per attempted iteration")
            }
            Self::CounterOverflow => formatter.write_str("refinement runtime counter overflow"),
            Self::Cancellation(error) => Display::fmt(error, formatter),
            Self::Stopped(stop) => Display::fmt(stop, formatter),
        }
    }
}

impl Error for RuntimeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Cancellation(error) => Some(error),
            Self::Stopped(stop) => Some(stop),
            _ => None,
        }
    }
}

fn invalid_event(message: &str) -> RuntimeError {
    RuntimeError::InvalidEvent {
        message: message.to_owned(),
    }
}
