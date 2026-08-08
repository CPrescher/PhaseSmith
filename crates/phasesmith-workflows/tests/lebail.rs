//! Native fixed-reflection Le Bail workflow and Python differential checks.

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use phasesmith_core::ConstantWavelengthInstrument;
use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::PatternRecord;
use phasesmith_workflows::{
    CancellationToken, LeBailInput, LeBailOptions, LeBailPhase, RefinementLimits,
    RefinementRuntime, TerminationReason, calculate_lebail_pattern, iterate_lebail_once,
    refine_lebail, refine_lebail_with_runtime,
};

fn instrument() -> ConstantWavelengthInstrument {
    ConstantWavelengthInstrument {
        wavelength_angstrom: 1.5406,
        u_deg2: 2.0e-4,
        v_deg2: -1.0e-4,
        w_deg2: 2.0e-4,
        x_deg: 1.5e-3,
        y_deg: 3.0e-3,
    }
}

fn execution() -> ExecutionPolicy {
    ExecutionPolicy::new(Some(1), 2).unwrap()
}

fn options(max_iterations: usize) -> LeBailOptions {
    LeBailOptions::new(
        max_iterations,
        2.min(max_iterations),
        1.0e-6,
        1.0e-8,
        1.0,
        1.0e-15,
        1.0e-12,
        true,
        1.0 - 1.0e-10,
        false,
        20.0,
        execution(),
    )
    .unwrap()
}

fn phase(phase_id: &str, positions: &[f64], intensities: &[f64]) -> LeBailPhase {
    let wavelength = instrument().wavelength_angstrom;
    LeBailPhase::new(
        phase_id,
        format!("{phase_id} phase"),
        (0..positions.len())
            .map(|index| format!("{phase_id}-{index}"))
            .collect(),
        (0..positions.len())
            .map(|index| [i32::try_from(index + 1).unwrap(), 1, 0])
            .collect(),
        positions
            .iter()
            .map(|position| wavelength / (2.0 * (0.5 * position.to_radians()).sin()))
            .collect(),
        positions.to_vec(),
        intensities.to_vec(),
        1.0,
        Vec::new(),
    )
    .unwrap()
}

#[allow(clippy::cast_precision_loss)]
fn linspace(start: f64, endpoint: f64, count: usize) -> Vec<f64> {
    let spacing = (endpoint - start) / (count - 1) as f64;
    (0..count)
        .map(|index| {
            if index + 1 == count {
                endpoint
            } else {
                start + index as f64 * spacing
            }
        })
        .collect()
}

fn observed_pattern(
    x: Vec<f64>,
    truth: &[LeBailPhase],
    background: Vec<f64>,
    uncertainty: Option<Vec<f64>>,
    mask: Option<Vec<bool>>,
) -> PatternRecord {
    let blank = PatternRecord::new(
        x.clone(),
        None,
        uncertainty.clone(),
        mask.clone(),
        Some(background.clone()),
    )
    .unwrap();
    let calculated =
        calculate_lebail_pattern(&blank, instrument(), truth, 20.0, &execution()).unwrap();
    PatternRecord::new(x, Some(calculated.y), uncertainty, mask, Some(background)).unwrap()
}

#[test]
fn isolated_reflections_converge_with_display_components() {
    let x = linspace(20.0, 80.0, 6_001);
    let positions = [30.0, 50.0, 70.0];
    let truth = vec![phase("alpha", &positions, &[10.0, 6.0, 3.0])];
    let starting = vec![phase("alpha", &positions, &[1.0, 1.0, 1.0])];
    let pattern = observed_pattern(x.clone(), &truth, vec![0.2; x.len()], None, None);
    let result = refine_lebail(
        &LeBailInput::new(pattern.clone(), instrument(), starting).unwrap(),
        &options(10),
        None,
    )
    .unwrap();

    assert_eq!(result.termination_reason, TerminationReason::Converged);
    assert_close_slice(
        &result
            .intensities
            .iter()
            .map(|item| item.integrated_intensity)
            .collect::<Vec<_>>(),
        &[10.0, 6.0, 3.0],
        2.0e-11,
    );
    assert_close_slice(
        &result.calculation.y,
        pattern.observed_y.as_ref().unwrap(),
        4.0e-13,
    );
    assert_eq!(
        result.calculation.reflection_keys,
        [
            ("alpha".to_owned(), "alpha-0".to_owned()),
            ("alpha".to_owned(), "alpha-1".to_owned()),
            ("alpha".to_owned(), "alpha-2".to_owned()),
        ]
    );
    assert_close_slice(
        &result.calculation.phase_components[0].y,
        &result.calculation.profile_y,
        2.0e-15,
    );
}

#[test]
fn uncertainty_mask_resume_and_unobserved_behavior_are_deterministic() {
    let x = linspace(38.0, 43.0, 5_001);
    let positions = [40.0, 40.06, 41.4];
    let truth = vec![phase("alpha", &positions, &[9.0, 4.0, 6.0])];
    let starting = vec![phase("alpha", &positions, &[2.0, 7.0, 1.0])];
    let uncertainty = x.iter().map(|value| 0.5 + 0.01 * (value - x[0])).collect();
    let pattern = observed_pattern(x, &truth, vec![0.0; 5_001], Some(uncertainty), None);
    let input = LeBailInput::new(pattern, instrument(), starting).unwrap();
    let partial = refine_lebail(&input, &options(3), None).unwrap();
    let resumed = refine_lebail(&input, &options(30), Some(&partial.checkpoint)).unwrap();
    let uninterrupted = refine_lebail(&input, &options(30), None).unwrap();
    assert_eq!(resumed.history, uninterrupted.history);
    assert_eq!(
        resumed.checkpoint.intensities,
        uninterrupted.checkpoint.intensities
    );
    assert_eq!(resumed.calculation.y, uninterrupted.calculation.y);
    let first = iterate_lebail_once(&input, &options(30), None).unwrap();
    let second = iterate_lebail_once(&input, &options(30), Some(&first.checkpoint)).unwrap();
    let direct = refine_lebail(&input, &options(2), None).unwrap();
    assert_eq!(second.history, direct.history);
    assert_eq!(second.checkpoint.intensities, direct.checkpoint.intensities);

    let x = linspace(39.0, 43.0, 4_001);
    let truth = vec![phase("alpha", &[40.0, 42.0], &[5.0, 7.0])];
    let starting = vec![phase("alpha", &[40.0, 42.0, 80.0], &[0.0, 0.0, 0.0])];
    let mask = x.iter().map(|value| *value < 41.0).collect();
    let pattern = observed_pattern(x, &truth, vec![0.0; 4_001], None, Some(mask));
    let result = refine_lebail(
        &LeBailInput::new(pattern, instrument(), starting).unwrap(),
        &options(20),
        None,
    )
    .unwrap();
    assert!((result.intensities[0].integrated_intensity - 5.0).abs() < 2.0e-9);
    assert_eq!(result.intensities[1].integrated_intensity.to_bits(), 0);
    assert_eq!(result.intensities[2].integrated_intensity.to_bits(), 0);
    assert!(result.history.last().unwrap().warnings[0].contains("2 reflections"));
}

#[test]
fn invalid_state_is_rejected_and_worker_budgets_are_deterministic() {
    assert!(
        LeBailPhase::new(
            "bad",
            "Bad phase",
            vec!["r0".to_owned()],
            vec![[1, 0, 0]],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            1.0,
            Vec::new(),
        )
        .is_err()
    );
    assert!(
        LeBailOptions::new(
            1,
            2,
            1.0e-6,
            1.0e-8,
            1.0,
            1.0e-15,
            1.0e-12,
            true,
            0.9,
            false,
            20.0,
            execution(),
        )
        .is_err()
    );

    let x = linspace(38.0, 43.0, 5_001);
    let positions = [40.0, 40.06, 41.4];
    let truth = vec![phase("alpha", &positions, &[9.0, 4.0, 6.0])];
    let starting = vec![phase("alpha", &positions, &[2.0, 7.0, 1.0])];
    let pattern = observed_pattern(x, &truth, vec![0.0; 5_001], None, None);
    let input = LeBailInput::new(pattern, instrument(), starting).unwrap();
    let serial = refine_lebail(&input, &options(30), None).unwrap();
    let mut parallel_options = options(30);
    parallel_options.execution = ExecutionPolicy::new(Some(2), 2).unwrap();
    let parallel = refine_lebail(&input, &parallel_options, None).unwrap();
    assert_eq!(
        serial.checkpoint.intensities,
        parallel.checkpoint.intensities
    );
    assert_eq!(serial.history, parallel.history);
    assert_eq!(serial.calculation.y, parallel.calculation.y);
}

#[test]
fn coincident_reflections_retain_partition_and_report_joint_rank() {
    let x = linspace(39.0, 41.0, 4_001);
    let truth = vec![
        phase("alpha", &[40.0], &[4.0]),
        phase("beta", &[40.0], &[8.0]),
    ];
    let starting = vec![
        phase("alpha", &[40.0], &[1.0]),
        phase("beta", &[40.0], &[2.0]),
    ];
    let pattern = observed_pattern(x, &truth, vec![0.0; 4_001], None, None);
    let mut selected = options(10);
    selected.diagnose_rank_deficiency = true;
    let result = refine_lebail(
        &LeBailInput::new(pattern, instrument(), starting).unwrap(),
        &selected,
        None,
    )
    .unwrap();
    assert_close_slice(
        &result
            .intensities
            .iter()
            .map(|item| item.integrated_intensity)
            .collect::<Vec<_>>(),
        &[4.0, 8.0],
        2.0e-11,
    );
    assert_eq!(result.rank_deficient_groups.len(), 1);
    assert_eq!(result.rank_deficient_groups[0].rank, 1);
    assert_eq!(
        result.rank_deficient_groups[0].reflection_keys,
        [
            ("alpha".to_owned(), "alpha-0".to_owned()),
            ("beta".to_owned(), "beta-0".to_owned()),
        ]
    );
}

#[test]
fn runtime_cancellation_events_and_checkpoint_delivery_are_integrated() {
    let x = linspace(39.0, 41.0, 1_001);
    let truth = vec![phase("alpha", &[40.0], &[8.0])];
    let starting = vec![phase("alpha", &[40.0], &[1.0])];
    let pattern = observed_pattern(x, &truth, vec![0.0; 1_001], None, None);
    let input = LeBailInput::new(pattern, instrument(), starting).unwrap();
    let token = CancellationToken::default();
    token.request("desktop_stop").unwrap();
    let limits = RefinementLimits::new(10, 11, None, 2).unwrap();
    let mut runtime = RefinementRuntime::new(limits, Some(token)).unwrap();
    let cancelled = refine_lebail_with_runtime(&input, &options(10), None, &mut runtime).unwrap();
    assert_eq!(cancelled.termination_reason, TerminationReason::Cancelled);
    assert!(cancelled.history.is_empty());

    let delivered = Arc::new(Mutex::new(Vec::new()));
    let observed = delivered.clone();
    let mut runtime = RefinementRuntime::new(limits, None).unwrap();
    runtime.set_checkpoint_sink(move |checkpoint: &phasesmith_workflows::LeBailCheckpoint| {
        observed
            .lock()
            .unwrap()
            .push(checkpoint.completed_iterations);
        Ok(())
    });
    let result = refine_lebail_with_runtime(&input, &options(10), None, &mut runtime).unwrap();
    assert_eq!(
        *delivered.lock().unwrap(),
        (1..=result.history.len()).collect::<Vec<_>>()
    );
}

#[test]
fn fixed_reflection_history_matches_python_when_configured() {
    let Ok(python) = std::env::var("PHASESMITH_NUMPY_PYTHON") else {
        return;
    };
    let x = linspace(38.0, 43.0, 5_001);
    let positions = [40.0, 40.06, 41.4];
    let truth = vec![phase("alpha", &positions, &[9.0, 4.0, 6.0])];
    let starting = vec![phase("alpha", &positions, &[2.0, 7.0, 1.0])];
    let uncertainty = x.iter().map(|value| 0.5 + 0.01 * (value - x[0])).collect();
    let pattern = observed_pattern(x, &truth, vec![0.0; 5_001], Some(uncertainty), None);
    let result = refine_lebail(
        &LeBailInput::new(pattern, instrument(), starting).unwrap(),
        &options(30),
        None,
    )
    .unwrap();

    let script = r#"
import numpy as np
import phasesmith
from phasesmith.refinement import lebail

instrument = phasesmith.ConstantWavelengthInstrument(1.5406, 2e-4, -1e-4, 2e-4, 1.5e-3, 3e-3)
x = np.linspace(38.0, 43.0, 5001)
positions = np.array([40.0, 40.06, 41.4])

def phase(values):
    d = instrument.wavelength_angstrom / (2.0 * np.sin(np.deg2rad(positions / 2.0)))
    reflections = phasesmith.ReflectionBatch(
        [f"alpha-{index}" for index in range(3)],
        np.array([[1, 1, 0], [2, 1, 0], [3, 1, 0]], dtype=np.int64),
        d, positions, np.asarray(values, dtype=np.float64),
    )
    return phasesmith.Phase("alpha", "alpha phase", reflections)

blank = phasesmith.PowderPattern(x)
observed = phasesmith.calculate_pattern(blank, instrument, (phase([9.0, 4.0, 6.0]),)).y
pattern = phasesmith.PowderPattern(
    x, observed_y=observed, uncertainty=0.5 + 0.01 * (x - x[0])
)
result = lebail.refine(
    lebail.LeBailInput(pattern, instrument, (phase([2.0, 7.0, 1.0]),)),
    lebail.LeBailOptions(max_iterations=30, execution=phasesmith.ExecutionPolicy(threads=1)),
)
print(result.termination_reason.value)
print(" ".join(format(item.integrated_intensity, ".17g") for item in result.intensities))
for item in result.history:
    print(item.iteration, format(item.rwp, ".17g"), format(item.maximum_relative_intensity_change, ".17g"))
"#;
    let python_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python");
    let output = Command::new(python)
        .args(["-c", script])
        .env("PYTHONPATH", python_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Python Le Bail oracle failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    assert_eq!(lines.next().unwrap(), result.termination_reason.as_str());
    let python_intensities = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|value| value.parse::<f64>().unwrap())
        .collect::<Vec<_>>();
    assert_close_slice(
        &result
            .intensities
            .iter()
            .map(|item| item.integrated_intensity)
            .collect::<Vec<_>>(),
        &python_intensities,
        5.0e-12,
    );
    let python_history = lines
        .map(|line| {
            let values = line.split_whitespace().collect::<Vec<_>>();
            (
                values[0].parse::<usize>().unwrap(),
                values[1].parse::<f64>().unwrap(),
                values[2].parse::<f64>().unwrap(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(python_history.len(), result.history.len());
    for (native, python) in result.history.iter().zip(python_history) {
        assert_eq!(native.iteration, python.0);
        assert!((native.rwp - python.1).abs() < 2.0e-14);
        assert!((native.maximum_relative_intensity_change - python.2).abs() < 2.0e-12);
    }
}

fn assert_close_slice(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index={index}, actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.1e}"
        );
    }
}
