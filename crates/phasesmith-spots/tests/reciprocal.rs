//! Native geometry invalid-input and Fourier boundary checks.
use phasesmith_spots::reciprocal::{Geometry, fourier};

fn panel() -> Geometry {
    Geometry {
        origin: [0.0, 0.0, 1.0],
        fast: [0.0, 1.0, 0.0],
        slow: [1.0, 0.0, 0.0],
        beam: [0.0, 0.0, 1.0],
        axis: [1.0, 0.0, 0.0],
        wavelength: 1.0,
    }
}

#[test]
fn direct_beam_and_invalid_geometry() {
    assert_eq!(panel().convert(&[[0.0, 0.0, 45.0]]).unwrap(), [[0.0; 3]]);
    assert!(panel().convert(&[[f64::NAN, 0.0, 0.0]]).is_err());
    let mut bad = panel();
    bad.axis = [0.0; 3];
    assert!(bad.convert(&[]).is_err());
    bad = panel();
    bad.slow = bad.fast;
    assert!(bad.convert(&[]).is_err());
}

#[test]
fn fourier_boundary_and_exact_origin() {
    assert!(fourier(&[], &[[0.0; 3]]).is_err());
    assert!(fourier(&[[f64::NAN; 3]], &[[0.0; 3]]).is_err());
    let result = fourier(&[[1.0, 2.0, 3.0]], &[[0.0; 3]]).unwrap();
    assert!((result[0][0] - 1.0).abs() < 1e-15);
    assert_eq!(&result[0][1..4], &[0.0; 3]);
    let result2 = fourier(&[[1.0, 2.0, 3.0]], &[[1.0, 0.0, 0.0]]).unwrap();
    assert!((result2[0][0] - 1.0).abs() < 1e-14);
}
