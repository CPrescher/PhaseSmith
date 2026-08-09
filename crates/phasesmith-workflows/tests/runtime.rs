//! Native bounded-runtime, cancellation, event, and checkpoint contracts.

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use phasesmith_workflows::{
    CancellationToken, DiagnosticValue, RefinementEvent, RefinementEventKind, RefinementLimits,
    RefinementRuntime, RuntimeClock, RuntimeError, TerminationReason,
};

#[derive(Debug)]
struct FakeClock {
    bits: AtomicU64,
}

impl FakeClock {
    fn new(value: f64) -> Self {
        Self {
            bits: AtomicU64::new(value.to_bits()),
        }
    }

    fn set(&self, value: f64) {
        self.bits.store(value.to_bits(), Ordering::Release);
    }
}

impl RuntimeClock for FakeClock {
    fn now_seconds(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::Acquire))
    }
}

#[test]
fn cancellation_is_thread_safe_and_first_reason_wins() {
    let token = CancellationToken::default();
    let worker_token = token.clone();
    let worker = thread::spawn(move || worker_token.request("stop_button").unwrap());
    assert!(worker.join().unwrap());
    assert!(!token.request("later_reason").unwrap());
    assert!(token.is_requested());
    assert_eq!(token.reason().unwrap().as_deref(), Some("stop_button"));
    assert!(token.request(" ").is_err());
}

#[test]
fn events_are_finite_ordered_and_logger_failure_is_isolated() {
    let event = RefinementEvent::new(
        RefinementEventKind::Iteration,
        "rietveld",
        3,
        2,
        12,
        1.25,
        "iteration completed",
        vec![
            ("rwp".to_owned(), DiagnosticValue::Float(0.052)),
            ("accepted".to_owned(), DiagnosticValue::Bool(true)),
            ("warning".to_owned(), DiagnosticValue::Null),
        ],
    )
    .unwrap();
    assert_eq!(event.kind().as_str(), "iteration");
    assert_eq!(event.diagnostics()[0].0, "rwp");
    assert!(
        RefinementEvent::new(
            RefinementEventKind::Warning,
            "rietveld",
            0,
            0,
            0,
            0.0,
            "bad diagnostic",
            vec![("value".to_owned(), DiagnosticValue::Float(f64::NAN))],
        )
        .is_err()
    );

    let mut runtime = RefinementRuntime::<()>::with_clock(
        RefinementLimits::default(),
        None,
        Arc::new(FakeClock::new(100.0)),
    )
    .unwrap();
    runtime.set_event_sink(|_: &RefinementEvent| Err("closed log stream".to_owned()));
    runtime
        .emit(
            RefinementEventKind::Start,
            "rietveld",
            "started",
            Vec::new(),
        )
        .unwrap();
    assert_eq!(runtime.event_sink_error(), Some("closed log stream"));
    assert!(!runtime.has_event_sink());
}

#[test]
fn runtime_enforces_evaluation_time_cancellation_iteration_and_rejection_limits() {
    let clock = Arc::new(FakeClock::new(100.0));
    let token = CancellationToken::default();
    let limits = RefinementLimits::new(2, 2, Some(5.0), 2).unwrap();
    let mut runtime =
        RefinementRuntime::<()>::with_clock(limits, Some(token.clone()), clock.clone()).unwrap();
    runtime.begin_iteration(1).unwrap();
    runtime.begin_evaluation().unwrap();
    runtime.begin_evaluation().unwrap();
    assert_stop(
        runtime.begin_evaluation().unwrap_err(),
        TerminationReason::MaxEvaluations,
    );

    let mut runtime = RefinementRuntime::<()>::with_clock(
        RefinementLimits::new(2, 10, Some(5.0), 2).unwrap(),
        Some(token.clone()),
        clock.clone(),
    )
    .unwrap();
    clock.set(105.0);
    assert_stop(
        runtime.begin_iteration(1).unwrap_err(),
        TerminationReason::MaxRuntime,
    );

    clock.set(106.0);
    let cancellation = CancellationToken::default();
    let mut runtime = RefinementRuntime::<()>::with_clock(
        RefinementLimits::new(1, 10, None, 2).unwrap(),
        Some(cancellation.clone()),
        clock.clone(),
    )
    .unwrap();
    cancellation.request("quit_key").unwrap();
    let error = runtime.begin_iteration(1).unwrap_err();
    assert_eq!(error.to_string(), "quit_key");
    assert_stop(error, TerminationReason::Cancelled);

    let mut runtime = RefinementRuntime::<()>::with_clock(
        RefinementLimits::new(1, 10, None, 2).unwrap(),
        None,
        clock,
    )
    .unwrap();
    runtime.begin_iteration(1).unwrap();
    assert_stop(
        runtime.begin_iteration(2).unwrap_err(),
        TerminationReason::MaxIterations,
    );
    runtime.reject_step().unwrap();
    assert_stop(
        runtime.reject_step().unwrap_err(),
        TerminationReason::RepeatedRejections,
    );
}

#[test]
fn checkpoint_failure_keeps_state_accepted_and_resume_is_guarded() {
    let clock = Arc::new(FakeClock::new(10.0));
    let mut runtime =
        RefinementRuntime::<String>::with_clock(RefinementLimits::default(), None, clock.clone())
            .unwrap();
    runtime.set_checkpoint_sink(|checkpoint: &String| {
        assert_eq!(checkpoint, "iteration-1");
        Err("disk full".to_owned())
    });
    runtime.begin_iteration(1).unwrap();
    assert!(matches!(
        runtime.accept_step(Some(&"iteration-1".to_owned())),
        Err(RuntimeError::CheckpointSink { .. })
    ));
    assert_eq!(runtime.accepted_iterations(), 1);

    let checkpoints = Arc::new(Mutex::new(Vec::new()));
    let observed = checkpoints.clone();
    let mut runtime =
        RefinementRuntime::<String>::with_clock(RefinementLimits::default(), None, clock).unwrap();
    runtime.set_checkpoint_sink(move |checkpoint: &String| {
        observed.lock().unwrap().push(checkpoint.clone());
        Ok(())
    });
    runtime.resume_accepted(3).unwrap();
    runtime.begin_iteration(4).unwrap();
    runtime
        .accept_step(Some(&"iteration-4".to_owned()))
        .unwrap();
    assert_eq!(runtime.accepted_iterations(), 4);
    assert_eq!(*checkpoints.lock().unwrap(), ["iteration-4"]);
    assert!(runtime.resume_accepted(4).is_err());
}

#[test]
#[ignore = "requires an installed NumPy Python interpreter"]
fn python_runtime_stop_sequence_matches_native_when_configured() {
    let Ok(python) = std::env::var("PHASESMITH_NUMPY_PYTHON") else {
        return;
    };
    let native = native_stop_sequence();
    let script = r#"
from phasesmith import CancellationToken
from phasesmith.refinement import RefinementLimits, RefinementRuntime, RefinementStopped

class Clock:
    value = 100.0
    def __call__(self):
        return self.value

clock = Clock()
reasons = []
runtime = RefinementRuntime(RefinementLimits(max_iterations=2, max_evaluations=2, max_runtime_seconds=5.0, max_consecutive_rejections=2), clock=clock)
runtime.begin_iteration(1)
runtime.begin_evaluation(); runtime.begin_evaluation()
try: runtime.begin_evaluation()
except RefinementStopped as error: reasons.append(error.reason.value)

runtime = RefinementRuntime(RefinementLimits(max_iterations=2, max_evaluations=10, max_runtime_seconds=5.0, max_consecutive_rejections=2), clock=clock)
clock.value = 105.0
try: runtime.begin_iteration(1)
except RefinementStopped as error: reasons.append(error.reason.value)

clock.value = 106.0
token = CancellationToken()
runtime = RefinementRuntime(RefinementLimits(max_consecutive_rejections=2), cancellation=token, clock=clock)
token.request("quit_key")
try: runtime.begin_iteration(1)
except RefinementStopped as error: reasons.append(error.reason.value)

runtime = RefinementRuntime(RefinementLimits(max_iterations=1, max_consecutive_rejections=2), clock=clock)
runtime.begin_iteration(1)
try: runtime.begin_iteration(2)
except RefinementStopped as error: reasons.append(error.reason.value)
runtime.reject_step()
try: runtime.reject_step()
except RefinementStopped as error: reasons.append(error.reason.value)
print(" ".join(reasons))
"#;
    let output = Command::new(python)
        .args(["-c", script])
        .env_remove("PYTHONPATH")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Python runtime oracle failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap().trim(), native);
}

fn native_stop_sequence() -> String {
    let clock = Arc::new(FakeClock::new(100.0));
    let mut reasons = Vec::new();
    let mut runtime = RefinementRuntime::<()>::with_clock(
        RefinementLimits::new(2, 2, Some(5.0), 2).unwrap(),
        None,
        clock.clone(),
    )
    .unwrap();
    runtime.begin_iteration(1).unwrap();
    runtime.begin_evaluation().unwrap();
    runtime.begin_evaluation().unwrap();
    reasons.push(stop_reason(runtime.begin_evaluation().unwrap_err()).as_str());

    let mut runtime = RefinementRuntime::<()>::with_clock(
        RefinementLimits::new(2, 10, Some(5.0), 2).unwrap(),
        None,
        clock.clone(),
    )
    .unwrap();
    clock.set(105.0);
    reasons.push(stop_reason(runtime.begin_iteration(1).unwrap_err()).as_str());

    clock.set(106.0);
    let token = CancellationToken::default();
    let mut runtime = RefinementRuntime::<()>::with_clock(
        RefinementLimits::new(100, 1_000, None, 2).unwrap(),
        Some(token.clone()),
        clock.clone(),
    )
    .unwrap();
    token.request("quit_key").unwrap();
    reasons.push(stop_reason(runtime.begin_iteration(1).unwrap_err()).as_str());

    let mut runtime = RefinementRuntime::<()>::with_clock(
        RefinementLimits::new(1, 1_000, None, 2).unwrap(),
        None,
        clock,
    )
    .unwrap();
    runtime.begin_iteration(1).unwrap();
    reasons.push(stop_reason(runtime.begin_iteration(2).unwrap_err()).as_str());
    runtime.reject_step().unwrap();
    reasons.push(stop_reason(runtime.reject_step().unwrap_err()).as_str());
    reasons.join(" ")
}

fn assert_stop(error: RuntimeError, expected: TerminationReason) {
    assert_eq!(stop_reason(error), expected);
}

fn stop_reason(error: RuntimeError) -> TerminationReason {
    match error {
        RuntimeError::Stopped(stop) => stop.reason,
        other => panic!("expected normal runtime stop, got {other}"),
    }
}
