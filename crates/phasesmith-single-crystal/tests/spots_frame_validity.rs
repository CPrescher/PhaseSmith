#![cfg(feature = "spots")]

//! Standalone native masking tests; no Python runtime or numerical dependency.
#![allow(clippy::float_cmp)] // Exact median/count ownership invariants are intentional.

use phasesmith_single_crystal::spots::frame_validity::{
    self as mask, DacOptions, Error, SeatOptions, Shape,
};
use phasesmith_single_crystal::spots::reciprocal::Geometry;

#[test]
fn block_medians_include_real_zeros_and_partial_edges_without_mutating_counts() {
    let shape = Shape {
        height: 3,
        width: 4,
        block: 2,
    };
    let values = vec![
        0.,
        2.,
        70_000.,
        f64::NAN,
        4.,
        6.,
        8.,
        10.,
        12.,
        14.,
        16.,
        18.,
    ];
    let mut excluded = vec![false; values.len()];
    excluded[2] = true;
    let out = mask::coarsen(&values, &excluded, shape).unwrap();
    assert_eq!(out.values, [3., 9., 13., 17.]);
    assert!(out.available);
    assert_eq!(values[0], 0.);
    assert_eq!(values[2], 70_000.);
    assert!(values[3].is_nan());
    assert!(mask::coarsen(&[-1.; 12], &excluded, shape).is_err());
    let hidden = mask::coarsen(&[-1.; 12], &[true; 12], shape).unwrap();
    assert!(!hidden.available && hidden.values.iter().all(|v| *v == 0.));
}

#[test]
fn rust_only_shadow_estimation_retains_uniform_dimming_and_blocks_moving_seats() {
    let shape = Shape {
        height: 192,
        width: 256,
        block: 16,
    };
    let cells = 12 * 16;
    let mut coarse = vec![80.; 20 * cells];
    for frame in 0..5 {
        for y in 0..12 {
            for x in 8..16 {
                coarse[frame * cells + y * 16 + x] = 15.;
            }
        }
    }
    coarse[19 * cells..].fill(10.);
    let result = mask::estimate(&coarse, &[true; 20], shape, SeatOptions::default()).unwrap();
    assert!(result.excluded[6 * 16 + 11]);
    assert!(!result.excluded[6 * 16 + 2]);
    assert!(result.excluded[10 * cells..11 * cells].iter().all(|v| !v));
    assert!(result.excluded[19 * cells..].iter().all(|v| !v));
    assert_eq!(result.reference_indices, [7, 8, 9, 10, 11]);
    assert!((result.detection_scales[19] - 0.125).abs() < 1e-7);
}

#[test]
fn unknown_dark_frames_are_distinct_from_explicitly_unavailable_frames() {
    let shape = Shape {
        height: 192,
        width: 256,
        block: 16,
    };
    let mut coarse = vec![80.; 5 * 192];
    coarse[..192].fill(0.);
    assert!(matches!(
        mask::estimate(&coarse, &[true; 5], shape, SeatOptions::default()),
        Err(Error::Unavailable(_))
    ));
    let result = mask::estimate(
        &coarse,
        &[false, true, true, true, true],
        shape,
        SeatOptions::default(),
    )
    .unwrap();
    assert!(result.excluded[..192].iter().all(|v| *v));
    assert_eq!(result.known_unavailable_frames, [0]);
}

#[test]
fn default_reference_rounding_is_ties_to_even_without_python() {
    let shape = Shape {
        height: 192,
        width: 256,
        block: 16,
    };
    let result = mask::estimate(
        &vec![80.; 51 * 192],
        &[true; 51],
        shape,
        SeatOptions::default(),
    )
    .unwrap();
    assert_eq!(result.reference_indices[0], 18); // .37*50 == 18.5
    assert_eq!(*result.reference_indices.last().unwrap(), 30); // .59*50 == 29.5
}

fn geometry() -> Geometry {
    Geometry {
        origin: [0.01, 0., 1.],
        fast: [1e-5, 0., 0.],
        slow: [0., 1e-5, 0.],
        beam: [0., 0., 1.],
        axis: [0., 1., 0.],
        wavelength: 1.,
    }
}

#[test]
fn geometric_mask_covers_complete_exposure_closed_edges_and_reverse_scans() {
    let shape = Shape {
        height: 1,
        width: 1,
        block: 1,
    };
    let options = DacOptions {
        axis_at_zero: [0., 0., 1.],
        half_opening_degrees: 35.,
    };
    let closed = mask::dac_mask(geometry(), [34., 35.], shape, options).unwrap();
    assert!(!closed.excluded[0]);
    let full = mask::dac_mask(geometry(), [34., 36.], shape, options).unwrap();
    assert!(full.excluded[0]);
    let reverse = mask::dac_mask(geometry(), [36., 34.], shape, options).unwrap();
    assert_eq!(full.excluded, reverse.excluded);
}

#[test]
fn coarse_mask_expansion_preserves_the_full_partial_edge_footprint() {
    let mask = mask::Mask {
        shape: Shape {
            height: 3,
            width: 5,
            block: 2,
        },
        excluded: vec![false, true, false, false, false, true],
    };
    let expanded = mask.expand().unwrap();
    assert_eq!(expanded.len(), 15);
    assert!(expanded[2] && expanded[3] && expanded[7] && expanded[8] && expanded[14]);
    assert_eq!(expanded.iter().filter(|v| **v).count(), 5);
}

#[test]
fn invalid_dimensions_openings_and_unavailable_references_are_errors() {
    assert!(
        Shape {
            height: 0,
            width: 5,
            block: 1
        }
        .coarse()
        .is_err()
    );
    assert!(
        Shape {
            height: 100_000,
            width: 100_000,
            block: 1
        }
        .coarse()
        .is_err()
    );
    let shape = Shape {
        height: 192,
        width: 256,
        block: 16,
    };
    assert!(matches!(
        mask::estimate(
            &vec![1.; 5 * 192],
            &[true; 5],
            shape,
            SeatOptions::default()
        ),
        Err(Error::Unavailable(_))
    ));
    assert!(
        mask::dac_mask(
            geometry(),
            [0., 1.],
            shape,
            DacOptions {
                axis_at_zero: [0.; 3],
                half_opening_degrees: 35.
            }
        )
        .is_err()
    );
    assert!(
        mask::dac_mask(
            geometry(),
            [0., 1.],
            shape,
            DacOptions {
                axis_at_zero: [0., 0., 1.],
                half_opening_degrees: 90.
            }
        )
        .is_err()
    );
}
