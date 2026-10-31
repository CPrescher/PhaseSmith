//! Cached and streaming profile equations must retain the same floating-point order.
use phasesmith_spots::{
    Bounds,
    profiles::{self, Execution, Grid, Observations},
};

#[test]
#[allow(clippy::too_many_lines)]
fn learning_cache_is_exact_and_tiny_budgets_fall_back() {
    let grid = Grid {
        shape: [5; 3],
        step: [1.; 3],
    };
    let mut profile = vec![0.; 125];
    for z in 1..4_i32 {
        for y in 1..4_i32 {
            for x in 1..4_i32 {
                profile[usize::try_from((z * 5 + y) * 5 + x).unwrap()] =
                    (-0.5 * f64::from((x - 2).pow(2) + (y - 2).pow(2) + (z - 2).pow(2))).exp();
            }
        }
    }
    let mass: f64 = profile.iter().sum();
    for p in &mut profile {
        *p /= mass;
    }
    let centre = [10.13, 10.17, 0.11];
    let mut samples = Vec::new();
    let mut offsets = vec![0];
    let mut boxes = Vec::new();
    let mut angles = Vec::new();
    let mut ids = Vec::new();
    for (id, intensity) in [20000., 25000., 30000., 35000.].into_iter().enumerate() {
        for f in 0..7 {
            let angle = [f64::from(f) - 3.5, f64::from(f) - 2.5];
            let mut bounds = Vec::new();
            for y in 7..14 {
                for x in 7..14 {
                    bounds.push([
                        f64::from(x) - 0.5,
                        f64::from(x) + 0.5,
                        f64::from(y) - 0.5,
                        f64::from(y) + 0.5,
                        angle[0],
                        angle[1],
                    ]);
                }
            }
            samples.extend(
                profiles::evaluate(&profile, grid, &bounds, centre)
                    .unwrap()
                    .into_iter()
                    .map(|p| [intensity * p[0], 4., 20.]),
            );
            offsets.push(samples.len());
            boxes.push(Bounds {
                x0: 7,
                y0: 7,
                x1: 14,
                y1: 14,
            });
            angles.push(angle);
            ids.push(id);
        }
    }
    let valid = vec![true; samples.len()];
    let covariance = vec![[0.; 9]; ids.len()];
    let flags = vec![0; ids.len()];
    for affine in [false, true] {
        let planar = if affine {
            [[0.9, -0.2, 0.1, 1.05]; 4]
        } else {
            [[1., 0., 0., 1.]; 4]
        };
        let scales = if affine {
            [[0.8, 1.1, -0.9]; 4]
        } else {
            [[1.; 3]; 4]
        };
        let o = Observations {
            samples: &samples,
            valid: &valid,
            offsets: &offsets,
            boxes: &boxes,
            ids: &ids,
            angles: &angles,
            covariance: &covariance,
            centres: &[centre; 4],
            flags: &flags,
            planar: &planar,
            scales: &scales,
        };
        for weighted in [false, true] {
            let run = |budget| {
                profiles::learn_with_execution(
                    &o,
                    &[0; 4],
                    &[true; 4],
                    grid,
                    2,
                    2,
                    20.,
                    0.,
                    5,
                    weighted,
                    Execution {
                        workers: 1,
                        geometry_cache_bytes: budget,
                    },
                )
                .unwrap()
            };
            let baseline = run(0);
            for budget in [1, 1024 * 1024] {
                let cached = run(budget);
                assert_eq!(baseline.profiles, cached.profiles);
                assert_eq!(baseline.shifts, cached.shifts);
                assert_eq!(baseline.counts, cached.counts);
                assert_eq!(baseline.used, cached.used);
                assert!(cached.execution.cache_bytes <= budget);
                assert_eq!(cached.execution.cache_entries > 0, budget > 1);
            }
            for workers in [2, 4] {
                let parallel = profiles::learn_with_execution(
                    &o,
                    &[0, 1, 0, 1],
                    &[true; 4],
                    grid,
                    3,
                    2,
                    20.,
                    0.,
                    5,
                    weighted,
                    Execution {
                        workers,
                        geometry_cache_bytes: 1024 * 1024,
                    },
                )
                .unwrap();
                let streaming = profiles::learn_with_execution(
                    &o,
                    &[0, 1, 0, 1],
                    &[true; 4],
                    grid,
                    3,
                    2,
                    20.,
                    0.,
                    5,
                    weighted,
                    Execution {
                        workers: 1,
                        geometry_cache_bytes: 0,
                    },
                )
                .unwrap();
                assert_eq!(streaming.profiles, parallel.profiles);
                assert_eq!(parallel.execution.workers, workers.min(3));
                assert!(parallel.execution.cache_bytes <= 1024 * 1024);
            }
        }
        for expected in [false, true] {
            let (values, flags) = profiles::fit(
                &o,
                &profile,
                grid,
                &[0; 4],
                &[[0.; 3]],
                0.9,
                expected,
                1.,
                0.,
            )
            .unwrap();
            for workers in [1, 2, 4] {
                for budget in [1, 1024 * 1024] {
                    let actual = profiles::fit_with_execution(
                        &o,
                        &profile,
                        grid,
                        &[0; 4],
                        &[[0.; 3]],
                        0.9,
                        expected,
                        1.,
                        0.,
                        Execution {
                            workers,
                            geometry_cache_bytes: budget,
                        },
                    )
                    .unwrap();
                    assert_eq!(actual.flags, flags);
                    for (a, b) in actual.values.iter().flatten().zip(values.iter().flatten()) {
                        assert_eq!(a.to_bits(), b.to_bits());
                    }
                    assert!(actual.execution.cache_peak_bound_bytes <= budget);
                }
            }
        }
    }
}
