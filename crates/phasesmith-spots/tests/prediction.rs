//! Native forward-prediction singularity and contract checks.
use phasesmith_spots::prediction::{NO_CROSSING, TANGENT, predict};
use phasesmith_spots::reciprocal::Geometry;

fn geometry() -> Geometry {
    Geometry {
        origin: [0.0, 0.0, 1.0],
        fast: [0.0, 0.01, 0.0],
        slow: [0.01, 0.0, 0.0],
        axis: [1.0, 0.0, 0.0],
        beam: [0.0, 0.0, 1.0],
        wavelength: 1.0,
    }
}

#[test]
fn impossible_and_tangent_crossings_are_explicit() {
    let basis = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let p = predict(
        geometry(),
        basis,
        &[[0, 0, 0], [0, 0, 2]],
        &[0.0, 0.0],
        false,
    )
    .unwrap();
    assert_eq!(p[0].flags, NO_CROSSING);
    assert_eq!(p[1].flags, TANGENT);
    assert!(p[0].values.iter().all(|v| v.is_nan()));
    assert!(p[1].jacobian.iter().all(|v| v.is_nan()));
    assert!(predict(geometry(), basis, &[[1, 1, 1]], &[], false).is_err());
    assert!(predict(geometry(), [[0.0; 3]; 3], &[], &[], false).is_err());
}
