//! Native parameter and constraint semantic contracts.

use std::process::Command;

use phasesmith_workflows::{
    AffineConstraint, Constraint, ConstraintError, ConstraintTransform, FixedConstraint,
    LinearConstraint, LinearTerm, ParameterBounds, ParameterError, ParameterKey, ParameterSet,
    ParameterSpec,
};

#[test]
fn fixed_and_affine_constraints_expand_and_differentiate_in_stable_order() {
    let (parameters, keys) = parameter_set();
    let transform = ConstraintTransform::new(
        parameters,
        vec![
            Constraint::Fixed(FixedConstraint::new(keys[1].clone(), -2.0e-4).unwrap()),
            Constraint::Affine(
                AffineConstraint::new(keys[2].clone(), keys[0].clone(), 0.5, 1.0e-5).unwrap(),
            ),
        ],
    )
    .unwrap();
    assert_eq!(transform.free_keys(), [keys[0].clone(), keys[3].clone()]);
    assert_eq!(transform.pack().unwrap(), [2.0, 1.0]);
    let values = transform.unpack(&[3.0, 1.5], false).unwrap();
    assert_close(values[&keys[0]], 3.0e-4, 1.0e-18);
    assert_close(values[&keys[1]], -2.0e-4, 1.0e-18);
    assert_close(values[&keys[2]], 1.6e-4, 1.0e-18);
    assert_close(values[&keys[3]], 1.5, 1.0e-15);

    let derivative = transform.derivative_matrix().unwrap();
    assert_eq!(derivative.rows, 4);
    assert_eq!(derivative.columns, 2);
    assert_eq!(derivative.row(0).unwrap(), [1.0e-4, 0.0]);
    assert_eq!(derivative.row(1).unwrap(), [0.0, 0.0]);
    assert_eq!(derivative.row(2).unwrap(), [0.5e-4, 0.0]);
    assert_eq!(derivative.row(3).unwrap(), [0.0, 1.0]);
}

#[test]
fn multi_source_constraints_chain_exact_rows() {
    let (parameters, keys) = parameter_set();
    let linear = LinearConstraint::new(
        keys[2].clone(),
        vec![
            LinearTerm::new(keys[0].clone(), 0.5).unwrap(),
            LinearTerm::new(keys[1].clone(), 0.2).unwrap(),
        ],
        4.0e-5,
    )
    .unwrap();
    let transform = ConstraintTransform::new(parameters, vec![Constraint::Linear(linear)]).unwrap();
    assert_eq!(
        transform.free_keys(),
        [keys[0].clone(), keys[1].clone(), keys[3].clone()]
    );
    let values = transform.unpack(&transform.pack().unwrap(), false).unwrap();
    assert_close(values[&keys[2]], 1.2e-4, 1.0e-18);
    let derivative = transform.derivative_matrix().unwrap();
    assert_eq!(derivative.row(0).unwrap(), [1.0e-4, 0.0, 0.0]);
    assert_eq!(derivative.row(1).unwrap(), [0.0, 1.0e-4, 0.0]);
    assert_eq!(derivative.row(2).unwrap(), [0.5e-4, 0.2e-4, 0.0]);
    assert_eq!(derivative.row(3).unwrap(), [0.0, 0.0, 1.0]);
}

#[test]
fn cycles_duplicates_unknowns_bounds_and_invalid_scalars_are_structured() {
    let (parameters, keys) = parameter_set();
    let cyclic = vec![
        Constraint::Affine(
            AffineConstraint::new(keys[0].clone(), keys[1].clone(), 1.0, 0.0).unwrap(),
        ),
        Constraint::Affine(
            AffineConstraint::new(keys[1].clone(), keys[0].clone(), 1.0, 0.0).unwrap(),
        ),
    ];
    assert!(matches!(
        ConstraintTransform::new(parameters.clone(), cyclic),
        Err(ConstraintError::UnresolvedDependency { .. })
    ));
    assert!(matches!(
        ConstraintTransform::new(
            parameters.clone(),
            vec![
                Constraint::Fixed(FixedConstraint::new(keys[0].clone(), 0.0).unwrap()),
                Constraint::Fixed(FixedConstraint::new(keys[0].clone(), 1.0).unwrap()),
            ],
        ),
        Err(ConstraintError::DuplicateTarget { .. })
    ));
    let transform = ConstraintTransform::new(parameters, Vec::new()).unwrap();
    assert!(matches!(
        transform.unpack(&[-1.0, -1.0, 1.2, 1.0], false),
        Err(ConstraintError::ExpandedValueOutsideBounds { .. })
    ));
    let clipped = transform.unpack(&[-1.0, -1.0, 1.2, 1.0], true).unwrap();
    assert_close(clipped[&keys[0]], 0.0, 0.0);

    assert!(matches!(
        ParameterBounds::new(f64::NAN, 1.0),
        Err(ParameterError::InvalidBounds)
    ));
    assert!(ParameterKey::new(" bad", "owner", "name").is_err());
    assert!(FixedConstraint::new(keys[0].clone(), f64::INFINITY).is_err());
}

#[test]
#[ignore = "requires an installed NumPy Python interpreter"]
fn python_constraint_transform_matches_native_when_configured() {
    let Ok(python) = std::env::var("PHASESMITH_NUMPY_PYTHON") else {
        return;
    };
    let (parameters, keys) = parameter_set();
    let constraint = Constraint::Linear(
        LinearConstraint::new(
            keys[2].clone(),
            vec![
                LinearTerm::new(keys[0].clone(), 0.5).unwrap(),
                LinearTerm::new(keys[1].clone(), 0.2).unwrap(),
            ],
            4.0e-5,
        )
        .unwrap(),
    );
    let transform = ConstraintTransform::new(parameters, vec![constraint]).unwrap();
    let packed = transform.pack().unwrap();
    let unpacked = transform.unpack(&packed, false).unwrap();
    let matrix = transform.derivative_matrix().unwrap();
    let native = packed
        .iter()
        .copied()
        .chain(keys.iter().map(|key| unpacked[key]))
        .chain(matrix.values.iter().copied())
        .collect::<Vec<_>>();

    let script = r#"
from phasesmith.refinement import (
    Bounds, ConstraintTransform, LinearConstraint, ParameterKey, ParameterSet, ParameterSpec
)

keys = (
    ParameterKey("instrument", "bank-1", "u"),
    ParameterKey("instrument", "bank-1", "v"),
    ParameterKey("instrument", "bank-1", "w"),
    ParameterKey("phase", "alpha", "scale"),
)
parameters = ParameterSet([
    ParameterSpec(keys[0], 2.0e-4, "degree^2", Bounds(0.0, 1.0), 1.0e-4),
    ParameterSpec(keys[1], -1.0e-4, "degree^2", Bounds(-1.0, 1.0), 1.0e-4),
    ParameterSpec(keys[2], 1.2e-4, "degree^2", Bounds(0.0, 1.0), 1.0e-4),
    ParameterSpec(keys[3], 1.0, "dimensionless", Bounds(0.0, 10.0), 1.0),
])
transform = ConstraintTransform(parameters, (
    LinearConstraint(keys[2], ((keys[0], 0.5), (keys[1], 0.2)), 4.0e-5),
))
packed = transform.pack()
unpacked = transform.unpack(packed)
for value in (*packed, *(unpacked[key] for key in keys), *transform.derivative_matrix().ravel()):
    print(repr(float(value)))
"#;
    let output = Command::new(python)
        .args(["-c", script])
        .env_remove("PYTHONPATH")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Python constraint oracle failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let reference = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|value| value.parse::<f64>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(reference.len(), native.len());
    for (actual, expected) in native.iter().zip(reference) {
        assert_close(*actual, expected, 1.0e-18);
    }
}

fn parameter_set() -> (ParameterSet, [ParameterKey; 4]) {
    let keys = [
        key("instrument", "bank-1", "u"),
        key("instrument", "bank-1", "v"),
        key("instrument", "bank-1", "w"),
        key("phase", "alpha", "scale"),
    ];
    let narrow = ParameterBounds::new(-1.0, 1.0).unwrap();
    let parameters = ParameterSet::new(vec![
        ParameterSpec::new(
            keys[0].clone(),
            2.0e-4,
            "degree^2",
            ParameterBounds::new(0.0, 1.0).unwrap(),
            1.0e-4,
            true,
        )
        .unwrap(),
        ParameterSpec::new(keys[1].clone(), -1.0e-4, "degree^2", narrow, 1.0e-4, true).unwrap(),
        ParameterSpec::new(
            keys[2].clone(),
            1.2e-4,
            "degree^2",
            ParameterBounds::new(0.0, 1.0).unwrap(),
            1.0e-4,
            true,
        )
        .unwrap(),
        ParameterSpec::new(
            keys[3].clone(),
            1.0,
            "dimensionless",
            ParameterBounds::new(0.0, 10.0).unwrap(),
            1.0,
            true,
        )
        .unwrap(),
    ])
    .unwrap();
    (parameters, keys)
}

fn key(module: &str, owner: &str, name: &str) -> ParameterKey {
    ParameterKey::new(module, owner, name).unwrap()
}

fn assert_close(actual: f64, expected: f64, absolute_tolerance: f64) {
    assert!((actual - expected).abs() <= absolute_tolerance);
}
