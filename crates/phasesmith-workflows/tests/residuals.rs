//! Native residual semantic and cross-interface contracts.

use std::path::PathBuf;
use std::process::Command;

use phasesmith_model::PatternRecord;
use phasesmith_workflows::{ResidualError, ResidualOptions, evaluate_residuals};

#[test]
fn masked_uncertainty_weighted_metrics_match_the_powder_convention() {
    let pattern = pattern();
    let result = evaluate_residuals(
        &pattern,
        &[11.0, 18.0, 27.0, 44.0],
        ResidualOptions {
            use_uncertainty: true,
            parameter_count: 1,
        },
    )
    .unwrap();
    assert_eq!(result.included, [true, false, true, true]);
    assert_eq!(result.residual, [1.0, -2.0, -3.0, 4.0]);
    assert_eq!(result.weighted_residual, [1.0, -1.0, -1.0, 1.0]);
    assert_close(result.chi_square, 3.0, 0.0);
    assert_close(result.reduced_chi_square, 1.5, 0.0);
    assert_close(result.rp, 8.0 / 80.0, 0.0);
    let denominator = 10.0_f64.powi(2) + (30.0_f64 / 3.0).powi(2) + (40.0_f64 / 4.0).powi(2);
    assert_close(result.rwp, (3.0 / denominator).sqrt(), 1.0e-16);
}

#[test]
fn unit_weights_zero_denominators_degrees_and_invalid_inputs_are_explicit() {
    let zero = PatternRecord::new(
        vec![1.0, 2.0],
        Some(vec![0.0, 0.0]),
        Some(vec![1.0, 2.0]),
        Some(vec![true, false]),
        None,
    )
    .unwrap();
    let result = evaluate_residuals(
        &zero,
        &[1.0, 4.0],
        ResidualOptions {
            use_uncertainty: false,
            parameter_count: 1,
        },
    )
    .unwrap();
    assert!(result.rp.is_infinite());
    assert!(result.rwp.is_infinite());
    assert!(result.reduced_chi_square.is_infinite());
    assert_eq!(result.weighted_residual, result.residual);

    let missing = PatternRecord::new(vec![1.0], None, None, None, None).unwrap();
    assert!(matches!(
        evaluate_residuals(&missing, &[0.0], ResidualOptions::default()),
        Err(ResidualError::MissingObservations)
    ));
    assert!(matches!(
        evaluate_residuals(&pattern(), &[1.0], ResidualOptions::default()),
        Err(ResidualError::CalculatedLengthMismatch { .. })
    ));
    assert!(matches!(
        evaluate_residuals(
            &pattern(),
            &[11.0, f64::NAN, 27.0, 44.0],
            ResidualOptions::default(),
        ),
        Err(ResidualError::NonFiniteCalculated { index: 1 })
    ));
}

#[test]
fn python_residual_evaluation_matches_native_when_configured() {
    let Ok(python) = std::env::var("PHASESMITH_NUMPY_PYTHON") else {
        return;
    };
    let native = evaluate_residuals(
        &pattern(),
        &[11.0, 18.0, 27.0, 44.0],
        ResidualOptions {
            use_uncertainty: true,
            parameter_count: 1,
        },
    )
    .unwrap();
    let expected = native
        .residual
        .iter()
        .chain(&native.weighted_residual)
        .copied()
        .chain([
            native.rp,
            native.rwp,
            native.chi_square,
            native.reduced_chi_square,
        ])
        .collect::<Vec<_>>();
    let script = r"
from phasesmith import PowderPattern
from phasesmith.refinement import ResidualOptions, evaluate_residuals

pattern = PowderPattern(
    [1.0, 2.0, 3.0, 4.0],
    observed_y=[10.0, 20.0, 30.0, 40.0],
    uncertainty=[1.0, 2.0, 3.0, 4.0],
    mask=[True, False, True, True],
)
result = evaluate_residuals(
    pattern,
    [11.0, 18.0, 27.0, 44.0],
    ResidualOptions(parameter_count=1),
)
for value in (*result.residual, *result.weighted_residual, result.rp, result.rwp, result.chi_square, result.reduced_chi_square):
    print(repr(float(value)))
";
    let python_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python");
    let output = Command::new(python)
        .args(["-c", script])
        .env("PYTHONPATH", python_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Python residual oracle failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let reference = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|value| value.parse::<f64>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(reference.len(), expected.len());
    for (actual, reference) in expected.iter().zip(reference) {
        assert_close(*actual, reference, 2.0e-16);
    }
}

fn pattern() -> PatternRecord {
    PatternRecord::new(
        vec![1.0, 2.0, 3.0, 4.0],
        Some(vec![10.0, 20.0, 30.0, 40.0]),
        Some(vec![1.0, 2.0, 3.0, 4.0]),
        Some(vec![true, false, true, true]),
        None,
    )
    .unwrap()
}

fn assert_close(actual: f64, expected: f64, absolute_tolerance: f64) {
    assert!((actual - expected).abs() <= absolute_tolerance);
}
