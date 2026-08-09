//! Analytical background values, derivatives, composition, and Python parity.

use std::process::Command;

use phasesmith_workflows::{
    AmorphousBackground, AmorphousPeak, BackgroundError, BackgroundModel, ChebyshevBackground,
    CompositeBackground, DifferentiableBackground, PointBackground, PolynomialBackground,
};

#[test]
fn every_model_basis_matches_centered_coefficient_differences() {
    let models = models();
    let x = linspace(10.0, 80.0, 351);
    for model in models {
        let basis = model.basis(&x).unwrap();
        let coefficients = model.coefficients();
        assert_eq!(basis.rows, x.len());
        assert_eq!(basis.columns, coefficients.len());
        for (column, value) in coefficients.iter().enumerate() {
            let step = 1.0e-6 * value.abs().max(1.0);
            let mut plus = coefficients.clone();
            let mut minus = coefficients.clone();
            plus[column] += step;
            minus[column] -= step;
            let plus_y = model
                .replace_coefficients(&plus)
                .unwrap()
                .calculate(&x)
                .unwrap();
            let minus_y = model
                .replace_coefficients(&minus)
                .unwrap()
                .calculate(&x)
                .unwrap();
            let analytical = basis.column(column).unwrap();
            for ((plus_value, minus_value), derivative) in
                plus_y.iter().zip(minus_y).zip(analytical)
            {
                let finite = (plus_value - minus_value) / (2.0 * step);
                assert_close(derivative, finite, 2.0e-7, 2.0e-9);
            }
        }
    }
}

#[test]
fn points_have_constant_ends_and_composites_preserve_parameter_order() {
    let points =
        PointBackground::new("points", vec![20.0, 40.0, 60.0], vec![2.0, 5.0, 3.0]).unwrap();
    let actual = points
        .calculate(&[10.0, 20.0, 30.0, 40.0, 60.0, 70.0])
        .unwrap();
    assert_eq!(actual, [2.0, 2.0, 3.5, 5.0, 3.0, 3.0]);

    let composite = composite();
    assert_eq!(
        composite.parameter_names(),
        [
            "power.coefficient_0",
            "power.coefficient_1",
            "chebyshev.coefficient_0",
            "chebyshev.coefficient_1",
            "chebyshev.coefficient_2",
            "points.value_0",
            "points.value_1",
            "points.value_2",
            "points.value_3",
            "glass.peak_0.area",
            "glass.peak_0.center_deg",
            "glass.peak_0.fwhm_deg",
        ]
    );
    assert!(!composite.basis_is_invariant());
}

#[test]
fn validation_and_domains_fail_structurally() {
    assert!(matches!(
        ChebyshevBackground::new("bad", vec![1.0], [20.0, 20.0]),
        Err(BackgroundError::InvalidDomain)
    ));
    assert!(matches!(
        PointBackground::new("bad", vec![20.0, 20.0], vec![1.0, 2.0]),
        Err(BackgroundError::InvalidKnots)
    ));
    assert!(matches!(
        AmorphousPeak::new(1.0, 40.0, 0.0),
        Err(BackgroundError::InvalidAmorphousPeak)
    ));
    let chebyshev = ChebyshevBackground::new("cheb", vec![1.0], [10.0, 80.0]).unwrap();
    assert!(matches!(
        chebyshev.calculate(&[9.0, 20.0]),
        Err(BackgroundError::GridOutsideDomain)
    ));
    assert!(matches!(
        chebyshev.calculate(&[20.0, 19.0]),
        Err(BackgroundError::UnorderedGrid)
    ));
    assert!(
        CompositeBackground::new(
            "duplicate",
            vec![
                BackgroundModel::Polynomial(PolynomialBackground::new("same", vec![1.0]).unwrap()),
                BackgroundModel::Point(
                    PointBackground::new("same", vec![0.0, 1.0], vec![1.0, 1.0]).unwrap()
                ),
            ],
        )
        .is_err()
    );
}

#[test]
#[ignore = "requires an installed NumPy Python interpreter"]
fn python_background_values_and_basis_match_native_when_configured() {
    let Ok(python) = std::env::var("PHASESMITH_NUMPY_PYTHON") else {
        return;
    };
    let x = vec![10.0, 20.0, 30.0, 40.0, 50.0, 65.0, 80.0];
    let model = composite();
    let basis = model.basis(&x).unwrap();
    let expected = model
        .calculate(&x)
        .unwrap()
        .into_iter()
        .chain(basis.values)
        .collect::<Vec<_>>();
    let script = r#"
import numpy as np
from phasesmith.refinement import (
    AmorphousBackground, AmorphousPeak, ChebyshevBackground,
    CompositeBackground, PointBackground, PolynomialBackground,
)

x = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 65.0, 80.0])
model = CompositeBackground("combined", (
    PolynomialBackground("power", (1.0, 0.2)),
    ChebyshevBackground("chebyshev", (2.0, -0.3, 0.08), (10.0, 80.0)),
    PointBackground("points", (10.0, 25.0, 50.0, 80.0), (1.0, 2.0, 1.5, 2.5)),
    AmorphousBackground("glass", (AmorphousPeak(4.0, 45.0, 10.0),)),
))
for value in (*model.calculate(x), *model.basis(x).ravel()):
    print(repr(float(value)))
"#;
    let output = Command::new(python)
        .args(["-c", script])
        .env_remove("PYTHONPATH")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Python background oracle failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let reference = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|value| value.parse::<f64>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(reference.len(), expected.len());
    for (actual, reference) in expected.iter().zip(reference) {
        assert_close(*actual, reference, 2.0e-14, 2.0e-14);
    }
}

fn models() -> Vec<BackgroundModel> {
    vec![
        BackgroundModel::Polynomial(
            PolynomialBackground::new("power", vec![2.0, -0.3, 0.08]).unwrap(),
        ),
        BackgroundModel::Chebyshev(
            ChebyshevBackground::new("chebyshev", vec![2.0, -0.3, 0.08], [10.0, 80.0]).unwrap(),
        ),
        BackgroundModel::Point(
            PointBackground::new(
                "points",
                vec![10.0, 25.0, 50.0, 80.0],
                vec![1.0, 2.0, 1.5, 2.5],
            )
            .unwrap(),
        ),
        BackgroundModel::Amorphous(
            AmorphousBackground::new(
                "amorphous",
                vec![
                    AmorphousPeak::new(12.0, 38.0, 11.0).unwrap(),
                    AmorphousPeak::new(5.0, 62.0, 8.0).unwrap(),
                ],
            )
            .unwrap(),
        ),
    ]
}

fn composite() -> CompositeBackground {
    CompositeBackground::new(
        "combined",
        vec![
            BackgroundModel::Polynomial(
                PolynomialBackground::new("power", vec![1.0, 0.2]).unwrap(),
            ),
            BackgroundModel::Chebyshev(
                ChebyshevBackground::new("chebyshev", vec![2.0, -0.3, 0.08], [10.0, 80.0]).unwrap(),
            ),
            BackgroundModel::Point(
                PointBackground::new(
                    "points",
                    vec![10.0, 25.0, 50.0, 80.0],
                    vec![1.0, 2.0, 1.5, 2.5],
                )
                .unwrap(),
            ),
            BackgroundModel::Amorphous(
                AmorphousBackground::new(
                    "glass",
                    vec![AmorphousPeak::new(4.0, 45.0, 10.0).unwrap()],
                )
                .unwrap(),
            ),
        ],
    )
    .unwrap()
}

#[allow(clippy::cast_precision_loss)]
fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    let denominator = (count - 1) as f64;
    (0..count)
        .map(|index| start + (end - start) * index as f64 / denominator)
        .collect()
}

fn assert_close(actual: f64, expected: f64, relative_tolerance: f64, absolute_tolerance: f64) {
    let tolerance = absolute_tolerance + relative_tolerance * expected.abs();
    assert!((actual - expected).abs() <= tolerance);
}
