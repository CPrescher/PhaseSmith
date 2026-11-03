#![cfg(feature = "spots")]

//! Native-only constrained fits, signed rows, finite support and deterministic workers.
use phasesmith_single_crystal::spots::{
    Bounds, box_optimizer as opt, empirical_fit,
    intensity_model::Design,
    profile_refinement::{self as shape, Noise},
    profiles::{self, Grid, Observations},
};

#[test]
fn constrained_optimizers_respect_fixed_active_bounds_and_budgets() {
    for quasi in [false, true] {
        let options = opt::Options {
            iterations: 100,
            evaluations: 200,
            ftol: 1e-12,
            gtol: 1e-8,
            xtol: 1e-13,
            cost_floor: 1.,
        };
        let evaluate = |p: &[f64]| {
            opt::least_squares(
                &[p[0] - 2., p[1] + 3., p[2] - 5.],
                &[1., 0., 0., 0., 1., 0., 0., 0., 1.],
                3,
            )
        };
        let fit = if quasi {
            opt::quasi_newton(&[0.; 3], &[-1., -4., 0.], &[1., 4., 0.], options, evaluate)
        } else {
            opt::minimize(&[0.; 3], &[-1., -4., 0.], &[1., 4., 0.], options, evaluate)
        }
        .unwrap();
        assert!(fit.converged);
        for (&a, b) in fit.parameters.iter().zip([1., -3., 0.]) {
            assert!((a - b).abs() < 1e-7);
        }
        assert!(fit.history.windows(2).all(|v| v[1] < v[0]));
        assert!(fit.evaluations <= 200);
        assert!(opt::minimize(&[0.; 3], &[1.; 3], &[2.; 3], options, evaluate).is_err());
    }
}

#[test]
#[allow(clippy::many_single_char_names)]
fn empirical_recovery_and_signed_holdout_are_native() {
    let n = 160;
    let widths = [3, 2, 2];
    let p = 7;
    let f: Vec<_> = (0..n * 3)
        .map(|i| (f64::from(u16::try_from(i).unwrap()).powi(2) * 0.173).sin())
        .collect();
    let d: Vec<_> = (0..n * 2)
        .map(|i| (f64::from(u16::try_from(i).unwrap()) * 0.307).cos() * 0.1)
        .collect();
    let a: Vec<_> = (0..n * 2)
        .map(|i| (f64::from(u16::try_from(i).unwrap()) * 0.417).sin() * 0.1)
        .collect();
    let design = Design {
        rows: n,
        widths,
        matrices: [&f, &d, &a],
    };
    let truth = [0.03, -0.02, 0.01, 0.04, -0.03, 0.02, -0.01];
    let v = phasesmith_single_crystal::spots::intensity_model::evaluate(&design, &truth, 1e-8)
        .unwrap()
        .0;
    let mut y: Vec<_> = v
        .iter()
        .enumerate()
        .map(|(i, v)| (6. + f64::from(u16::try_from(i / 4).unwrap()) * 0.01 + v).exp())
        .collect();
    y[n - 2] = -20.;
    y[n - 1] = 0.;
    let variance = vec![0.01; n];
    let groups: Vec<_> = (0..n).map(|i| i / 4).collect();
    let references: Vec<_> = (0..n).map(|i| i < n - 8).collect();
    let fit = empirical_fit::fit(
        &empirical_fit::Input {
            design,
            penalty: &[],
            intensity: &y,
            variance: &variance,
            groups: &groups,
            reference: &references,
            lower: &[-0.5; 7],
            upper: &[0.5; 7],
            start: &[0.; 7],
        },
        empirical_fit::Options {
            strength: 0.,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(fit.converged);
    assert_eq!(fit.coefficients.len(), p);
    for (&a, b) in fit.coefficients.iter().zip(truth) {
        assert!((a - b).abs() < 3e-6, "{a} != {b}");
    }
    assert_eq!(fit.references, references);
    assert!(fit.correction.iter().all(|v| v.is_finite() && *v > 0.));
    assert!(fit.correction[n - 2] * y[n - 2] < 0.);
}

#[test]
#[allow(clippy::too_many_lines)]
fn strong_and_frozen_weak_batch_native_determinism_and_cancellation() {
    let grid = Grid {
        shape: [5; 3],
        step: [1.; 3],
    };
    let mut profile = vec![0.; 125];
    for z in 1..4 {
        for y in 1..4 {
            for x in 1..4 {
                let r = f64::from(
                    u16::try_from((x - 2_i32).pow(2) + (y - 2_i32).pow(2) + (z - 2_i32).pow(2))
                        .unwrap(),
                );
                profile[usize::try_from((z * 5 + y) * 5 + x).unwrap()] = (-0.5 * r).exp();
            }
        }
    }
    let total: f64 = profile.iter().sum();
    for v in &mut profile {
        *v /= total;
    }
    let centre = [10.13, 10.17, 0.11];
    let mut samples = Vec::new();
    let mut bounds = Vec::new();
    let mut offsets = vec![0];
    let mut boxes = Vec::new();
    let mut angles = Vec::new();
    let mut ids = Vec::new();
    for (id, intensity) in [20000., 30000., 0., -20.].into_iter().enumerate() {
        for frame in 0..7 {
            let angle = [f64::from(frame) - 3.5, f64::from(frame) - 2.5];
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
            let masses =
                profiles::evaluate(&profile, grid, &bounds[bounds.len() - 49..], centre).unwrap();
            samples.extend(masses.iter().map(|m| [intensity * m[0], 4., 20.]));
            boxes.push(Bounds {
                x0: 7,
                y0: 7,
                x1: 14,
                y1: 14,
            });
            angles.push(angle);
            ids.push(id);
            offsets.push(samples.len());
        }
    }
    let mut centres = [centre; 4];
    centres[0][0] += 0.17;
    centres[0][1] -= 0.13;
    centres[0][2] += 0.07;
    let valid = vec![true; samples.len()];
    let flags = vec![0; ids.len()];
    let covariance = vec![[0.1, 0., 0., 0., 0.05, 0., 0., 0., 0.07]; ids.len()];
    let o = Observations {
        samples: &samples,
        valid: &valid,
        offsets: &offsets,
        boxes: &boxes,
        ids: &ids,
        angles: &angles,
        covariance: &covariance,
        centres: &centres,
        flags: &flags,
        scales: &[[1.; 3]; 4],
        planar: &[[1., 0., 0., 1.]; 4],
    };
    let input = shape::Input {
        observations: o,
        profiles: &profile,
        grid,
        assignments: &[0; 4],
        shifts: &[[0.; 3]],
        minimum_mass: 0.9,
        noise: Noise {
            expected: false,
            gain: 1.,
            read: 0.,
        },
        choices: &[1, 2, 0, 0],
        weak: &[false, false, true, true],
        sensor_offsets: &[[0.; 2]; 4],
        frozen: None,
        baseline: None,
    };
    let options = shape::Options {
        weak_spatial_width: 1.,
        ..Default::default()
    };
    let a = shape::refine(&input, options, |_, _| Ok(())).unwrap();
    let b = shape::refine(
        &input,
        shape::Options {
            workers: 4,
            ..options
        },
        |_, _| Ok(()),
    )
    .unwrap();
    assert_eq!(a.parameters, b.parameters);
    assert_eq!(a.flags, b.flags);
    for (row, expected) in a.values.iter().zip([20000., 30000., 0., -20.]) {
        assert!((row[0] - expected).abs() < 0.05);
        assert!(row[1] > 0.);
    }
    for (a, b) in a.values.iter().flatten().zip(b.values.iter().flatten()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert!(a.flags.iter().all(|f| f & 16384 == 0));
    assert!(
        shape::refine(&input, options, |_, _| Err(
            phasesmith_single_crystal::spots::SpotError("cancel".into())
        ))
        .is_err()
    );
}

#[test]
fn bounded_bfgs_cauchy_and_subspace_recover_curved_and_active_minima() {
    for upper in [0.8_f64, 2.] {
        let fit = opt::quasi_newton(
            &[-1.2, 1.],
            &[-2., -2.],
            &[upper, 2.],
            opt::Options {
                iterations: 300,
                evaluations: 1500,
                ftol: 1e-12,
                gtol: 1e-8,
                xtol: 1e-13,
                cost_floor: 1.,
            },
            |p| {
                opt::least_squares(
                    &[10. * (p[1] - p[0] * p[0]), 1. - p[0]],
                    &[-20. * p[0], 10., -1., 0.],
                    2,
                )
            },
        )
        .unwrap();
        let x = upper.min(1.);
        assert!(fit.converged, "{:?}", fit.parameters);
        assert!((fit.parameters[0] - x).abs() < 2e-6);
        assert!((fit.parameters[1] - x * x).abs() < 2e-6);
        assert!(fit.history.windows(2).all(|p| p[1] <= p[0]));
    }
}
