#![cfg(feature = "spots")]

//! Standalone native geometry recovery, analytical objective and bound contracts.
use phasesmith_single_crystal::spots::geometry_refinement::{self as native, Input, Options};
use phasesmith_single_crystal::spots::{prediction, reciprocal::Geometry};

fn geometry() -> Geometry {
    Geometry {
        origin: [-0.08, -0.08, 0.2],
        fast: [0., 8e-5, 0.],
        slow: [8e-5, 0., 0.],
        beam: [0., 0., 1.],
        axis: [-0.97, 0.12, 0.2],
        wavelength: 0.4,
    }
}
fn basis() -> [[f64; 3]; 3] {
    [[0.2, 0.01, 0.], [0.01, 0.16, 0.01], [0., 0.02, 0.12]]
}
fn truth(lengths: bool) -> Vec<f64> {
    [
        0.0002, -0.0001, 0.0003, 0.03, -0.04, 0.02, 0.0008, -0.0006, 0.0004, 0.0005, -0.0003,
        0.0002,
    ][..if lengths { 12 } else { 9 }]
        .to_vec()
}
fn pivot() -> [f64; 3] {
    let g = geometry();
    std::array::from_fn(|j| g.origin[j] + 1023.5 * (g.fast[j] + g.slow[j]))
}
fn sample(lengths: bool) -> (Vec<[i64; 3]>, Vec<[f64; 3]>, Vec<bool>) {
    let mut all = Vec::new();
    for h in -5..=5 {
        for k in -5..=5 {
            for l in -5..=5 {
                all.push([h, k, l]);
            }
        }
    }
    let hints: Vec<_> = (0..all.len())
        .map(|i| -200. + f64::from(u16::try_from(i % 170).unwrap()))
        .collect();
    let state =
        native::parameter_state(&truth(lengths), basis(), geometry(), pivot(), lengths).unwrap();
    let predictions =
        prediction::predict(state.geometry, state.basis, &all, &hints, false).unwrap();
    let mut hkl = Vec::new();
    let mut observed = Vec::new();
    for (h, p) in all.into_iter().zip(predictions) {
        if p.flags == 0 && p.values[..2].iter().all(|v| v.abs() < 5000.) {
            hkl.push(h);
            observed.push(p.values);
        }
    }
    let training = (0..hkl.len()).map(|i| i % 5 != 0).collect();
    (hkl, observed, training)
}
fn input<'a>(
    hkl: &'a [[i64; 3]],
    observed: &'a [[f64; 3]],
    training: &'a [bool],
    eligible: &'a [bool],
    lengths: bool,
) -> Input<'a> {
    Input {
        observed,
        hkl,
        training,
        eligible,
        basis: basis(),
        geometry: geometry(),
        image_shape: [2048, 2048],
        reference_cell: None,
        refine_lengths: lengths,
    }
}

#[test]
fn native_known_pose_and_lengths_recover_without_an_interpreter() {
    for lengths in [false, true] {
        let (h, o, t) = sample(lengths);
        let eligible = vec![true; h.len()];
        let data = input(&h, &o, &t, &eligible, lengths);
        let mut progress = Vec::new();
        let fit = native::fit(&data, Options::default(), |i, _| {
            progress.push(i);
            Ok(())
        })
        .unwrap();
        assert!(fit.converged && !fit.at_bounds && fit.scaled_condition < 1e4);
        for (p, target) in fit.parameters.iter().zip(truth(lengths)) {
            assert!((p - target).abs() < 1e-7);
        }
        assert!(fit.rms[1][1][0] < 2e-7 && fit.rms[1][1][1] < 2e-8);
        assert!(fit.history.windows(2).all(|p| p[1] < p[0]));
        assert_eq!(progress, (0..fit.evaluations).collect::<Vec<_>>());
        let mut changed = o.clone();
        for (row, train) in changed.iter_mut().zip(&t) {
            if !train {
                row[0] += 0.1;
            }
        }
        let again = native::fit(
            &input(&h, &changed, &t, &eligible, lengths),
            Options::default(),
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(fit.parameters, again.parameters);
    }
}

#[test]
#[allow(clippy::many_single_char_names)] // h, observations, p and two finite-difference costs.
fn robust_objective_gradient_matches_centered_differences() {
    for lengths in [false, true] {
        let (h, mut o, t) = sample(lengths);
        o[5][0] += 2.;
        let eligible = vec![true; h.len()];
        let data = input(&h, &o, &t, &eligible, lengths);
        let selected: Vec<_> = (0..h.len()).filter(|&i| t[i]).collect();
        let p: Vec<_> = truth(lengths).iter().map(|v| v * 0.7).collect();
        let (_, gradient, normal) =
            native::objective(&data, basis(), pivot(), &selected, &p, Options::default()).unwrap();
        for k in 0..p.len() {
            let mut plus = p.clone();
            let mut minus = p.clone();
            plus[k] += 1e-7;
            minus[k] -= 1e-7;
            let a = native::objective(
                &data,
                basis(),
                pivot(),
                &selected,
                &plus,
                Options::default(),
            )
            .unwrap()
            .0;
            let b = native::objective(
                &data,
                basis(),
                pivot(),
                &selected,
                &minus,
                Options::default(),
            )
            .unwrap()
            .0;
            let numerical = (a - b) / 2e-7;
            assert!(
                (numerical - gradient[k]).abs() < 5e-4 + 3e-6 * gradient[k].abs(),
                "column {k}: {numerical} != {}",
                gradient[k]
            );
            assert!(normal[k * p.len() + k] > 0.);
        }
    }
}

#[test]
fn budget_bounds_invalid_inputs_and_cancellation_remain_explicit() {
    let (h, o, t) = sample(false);
    let e = vec![true; h.len()];
    let data = input(&h, &o, &t, &e, false);
    let fit = native::fit(
        &data,
        Options {
            max_evaluations: 1,
            ..Options::default()
        },
        |_, _| Ok(()),
    )
    .unwrap();
    assert!(!fit.converged && fit.evaluations == 1);
    let hints: Vec<_> = o.iter().map(|p| p[2]).collect();
    let exact: Vec<_> = prediction::predict(geometry(), basis(), &h, &hints, false)
        .unwrap()
        .into_iter()
        .map(|p| p.values)
        .collect();
    let optimum = native::fit(
        &input(&h, &exact, &t, &e, false),
        Options {
            max_evaluations: 1,
            ..Options::default()
        },
        |_, _| Ok(()),
    )
    .unwrap();
    assert!(optimum.converged && optimum.evaluations == 1);

    let options = Options {
        translation_limit_mm: 0.01,
        ..Options::default()
    };
    let fit = native::fit(&data, options, |_, _| Ok(())).unwrap();
    assert!(fit.parameters[3..6].iter().all(|v| v.abs() <= 0.01) && fit.at_bounds);
    assert!(
        native::fit(&data, Options::default(), |_, _| Err(
            phasesmith_single_crystal::spots::SpotError("cancel".into())
        ))
        .is_err()
    );
    let mut bad = input(&h, &o, &t, &e, false);
    bad.observed = &o[..3];
    assert!(native::fit(&bad, Options::default(), |_, _| Ok(())).is_err());
    assert!(
        native::objective(
            &data,
            basis(),
            pivot(),
            &[h.len()],
            &[0.; 9],
            Options::default()
        )
        .is_err()
    );
    assert!(native::reference_basis(basis(), [5., 7., 9., 5., 5., 175.]).is_err());
}

#[test]
fn a_native_nonlinear_pixel_map_keeps_fitting_independent_of_python() {
    struct Warp;
    impl native::PixelMap for Warp {
        fn inverse(
            &self,
            points: &[[f64; 2]],
        ) -> Result<Vec<native::MappedPoint>, phasesmith_single_crystal::spots::SpotError> {
            Ok(points
                .iter()
                .map(|p| native::MappedPoint {
                    image: [1.1 * p[0] + 1e-5 * p[1] * p[1] + 3., 0.9 * p[1] - 2.],
                    jacobian: [[1.1, 2e-5 * p[1]], [0., 0.9]],
                })
                .collect())
        }
    }
    let (h, mut observed, t) = sample(false);
    for p in &mut observed {
        let x = p[0];
        let y = p[1];
        p[0] = 1.1 * x + 1e-5 * y * y + 3.;
        p[1] = 0.9 * y - 2.;
    }
    let eligible = vec![true; h.len()];
    let fit = native::fit_with_map(
        &input(&h, &observed, &t, &eligible, false),
        Options::default(),
        &Warp,
        |_, _| Ok(()),
    )
    .unwrap();
    assert!(fit.converged && fit.rms[1][1][0] < 2e-7);
    for (p, target) in fit.parameters.iter().zip(truth(false)) {
        assert!((p - target).abs() < 1e-7);
    }
}
