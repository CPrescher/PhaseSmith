//! Native Pawley scientific and recovery gates.
#![allow(clippy::float_cmp)] // Exact bounds and restart invariants are intentional.
use phasesmith_core::ConstantWavelengthInstrument;
use phasesmith_model::PatternRecord;
use phasesmith_workflows::*;

fn input(positions: Vec<f64>, areas: Vec<f64>, signed: bool) -> PawleyInput {
    let x: Vec<f64> = (0..1201).map(|i| 39.0 + f64::from(i) * 0.002).collect();
    let pattern = PatternRecord::new(
        x.clone(),
        Some(vec![0.0; x.len()]),
        Some(vec![1.0; x.len()]),
        None,
        None,
    )
    .unwrap();
    let instrument = ConstantWavelengthInstrument {
        wavelength_angstrom: 1.54,
        u_deg2: 0.0,
        v_deg2: 0.0,
        w_deg2: 0.001,
        x_deg: 0.002,
        y_deg: 0.0,
    };
    let phase = PawleyPhase {
        id: "phase".into(),
        reflection_ids: (0..areas.len()).map(|i| i.to_string()).collect(),
        two_theta_deg: positions,
        intensities: areas,
        hkl: vec![],
        lattice: None,
    };
    let parameters =
        pawley_parameters(std::slice::from_ref(&phase), instrument, None, signed).unwrap();
    PawleyInput {
        fixed_spectrum: None,
        pattern,
        instrument,
        axial: None,
        phases: vec![phase],
        background: None,
        signed_intensities: signed,
        parameters,
        constraints: vec![],
    }
}
fn calc(input: &PawleyInput) -> PawleyEvaluation {
    let t = ConstraintTransform::new(input.parameters.clone(), input.constraints.clone()).unwrap();
    evaluate_pawley(input, &t.pack().unwrap(), 20.0, true, 50_000_000).unwrap()
}
fn observations(mut request: PawleyInput, areas: Vec<f64>) -> PawleyInput {
    let mut truth = request.clone();
    truth.phases[0].intensities = areas;
    truth.parameters = pawley_parameters(
        &truth.phases,
        truth.instrument,
        None,
        truth.signed_intensities,
    )
    .unwrap();
    request.pattern.observed_y = Some(calc(&truth).calculated_y);
    request
}
#[test]
fn recovers_overlapping_areas_from_zero_and_reports_covariance() {
    let request = observations(
        input(vec![40.0, 40.06], vec![0.0, 0.0], false),
        vec![3.0, 8.0],
    );
    let result = refine_pawley(&request, &PawleyOptions::default()).unwrap();
    assert_eq!(result.termination_reason, TerminationReason::Converged);
    assert!((result.evaluation.intensities[0] - 3.0).abs() < 1e-7);
    assert!((result.evaluation.intensities[1] - 8.0).abs() < 1e-7);
    assert_eq!(result.rank, Some(2));
    assert!(result.covariance.is_some());
}
#[test]
fn coincidence_has_rank_one_and_identifiable_sum() {
    let request = observations(
        input(vec![40.0, 40.0], vec![0.0, 0.0], false),
        vec![3.0, 8.0],
    );
    let result = refine_pawley(&request, &PawleyOptions::default()).unwrap();
    assert!((result.evaluation.intensities.iter().sum::<f64>() - 11.0).abs() < 1e-7);
    assert_eq!(result.rank, Some(1));
    assert!(result.covariance.is_none());
}
#[test]
fn signed_negative_and_bound_active_fits() {
    let request = observations(input(vec![40.0], vec![0.0], true), vec![-3.0]);
    let result = refine_pawley(&request, &PawleyOptions::default()).unwrap();
    assert!((result.evaluation.intensities[0] + 3.0).abs() < 1e-8);
    let mut nonnegative = input(vec![40.0], vec![0.0], false);
    nonnegative.pattern.observed_y = request.pattern.observed_y;
    let result = refine_pawley(&nonnegative, &PawleyOptions::default()).unwrap();
    assert_eq!(result.evaluation.intensities[0], 0.0);
    assert_eq!(result.active_bounds, vec![0]);
    assert!(result.covariance.is_none());
}
#[test]
fn mask_and_unobserved_families_preserve_values() {
    let mut request = observations(
        input(vec![40.0, 80.0], vec![0.0, 12.0], false),
        vec![3.0, 12.0],
    );
    request.pattern.mask = Some((0..1201).map(|i| i != 100).collect());
    request.pattern.observed_y.as_mut().unwrap()[100] = 1e20;
    let result = refine_pawley(&request, &PawleyOptions::default()).unwrap();
    assert!((result.evaluation.intensities[0] - 3.0).abs() < 1e-7);
    assert_eq!(result.evaluation.intensities[1], 12.0);
    assert_eq!(result.observed_free_parameters, 1);
}
#[test]
fn explicit_ratio_constraint_removes_ambiguity() {
    let mut request = observations(
        input(vec![40.0, 40.0], vec![0.0, 0.0], false),
        vec![3.0, 6.0],
    );
    request.constraints.push(Constraint::Affine(
        AffineConstraint::new(
            pawley_key("intensity", "phase", "1").unwrap(),
            pawley_key("intensity", "phase", "0").unwrap(),
            2.0,
            0.0,
        )
        .unwrap(),
    ));
    let result = refine_pawley(&request, &PawleyOptions::default()).unwrap();
    assert!((result.evaluation.intensities[0] - 3.0).abs() < 1e-7);
    assert!((result.evaluation.intensities[1] - 6.0).abs() < 1e-7);
    assert_eq!(result.rank, Some(1));
    assert!(result.covariance.is_some());
}
#[test]
fn intensity_columns_are_analytic_at_zero_and_memory_is_bounded() {
    let request = observations(input(vec![40.0], vec![0.0], false), vec![3.0]);
    let t = ConstraintTransform::new(request.parameters.clone(), vec![]).unwrap();
    let z = t.pack().unwrap();
    let e = calc(&request);
    let mut plus = z.clone();
    plus[0] += 1e-4;
    let p = evaluate_pawley(&request, &plus, 20.0, true, 50_000_000).unwrap();
    for i in 0..1201 {
        assert!(
            ((p.calculated_y[i] - e.calculated_y[i]) / 1e-4 - e.jacobian.as_ref().unwrap()[(i, 0)])
                .abs()
                < 1e-10
        );
    }
    assert!(evaluate_pawley(&request, &z, 20.0, true, 10).is_err());
}
#[test]
fn accepted_checkpoint_resumes_exactly_and_rejects_changed_data() {
    let request = observations(
        input(vec![40.0, 40.06], vec![0.0, 0.0], false),
        vec![3.0, 8.0],
    );
    let options = PawleyOptions::default();
    let full = refine_pawley(&request, &options).unwrap();
    let token = CancellationToken::default();
    let sink_token = token.clone();
    let mut runtime = RefinementRuntime::new(RefinementLimits::default(), Some(token)).unwrap();
    runtime.set_checkpoint_sink(move |_: &PawleyCheckpoint| {
        sink_token.request("test").unwrap();
        Ok(())
    });
    let first = refine_pawley_with_runtime(&request, &options, None, &mut runtime).unwrap();
    assert_eq!(first.termination_reason, TerminationReason::Cancelled);
    let mut runtime = RefinementRuntime::new(RefinementLimits::default(), None).unwrap();
    let resumed =
        refine_pawley_with_runtime(&request, &options, Some(&first.checkpoint), &mut runtime)
            .unwrap();
    assert_eq!(full.checkpoint, resumed.checkpoint);
    let mut changed = request.clone();
    changed.pattern.observed_y.as_mut().unwrap()[0] += 1.0;
    let mut runtime = RefinementRuntime::new(RefinementLimits::default(), None).unwrap();
    assert!(
        refine_pawley_with_runtime(&changed, &options, Some(&first.checkpoint), &mut runtime)
            .is_err()
    );
}

#[test]
fn support_products_match_dense_and_are_adjoint_with_tied_areas() {
    let mut request = input(vec![40.0, 40.06, 80.0], vec![3.0, 6.0, 1.0], false);
    request.constraints.push(Constraint::Affine(
        AffineConstraint::new(
            pawley_key("intensity", "phase", "1").unwrap(),
            pawley_key("intensity", "phase", "0").unwrap(),
            2.0,
            0.0,
        )
        .unwrap(),
    ));
    let e = calc(&request);
    let op = &e.jacobian_operator;
    let v: Vec<f64> = (0..op.free_count())
        .map(|i| (f64::from(u32::try_from(i).unwrap()) + 0.7).sin())
        .collect();
    let u: Vec<f64> = (0..op.sample_count())
        .map(|i| (f64::from(u32::try_from(i).unwrap()) * 0.37).cos())
        .collect();
    let jv = op.jvp(&v).unwrap();
    let jtu = op.vjp(&u).unwrap();
    let dense_jv = e.jacobian.as_ref().unwrap() * nalgebra::DVector::from_column_slice(&v);
    let dense_jtu =
        e.jacobian.as_ref().unwrap().transpose() * nalgebra::DVector::from_column_slice(&u);
    for (a, b) in jv.iter().zip(dense_jv.iter()) {
        assert!((a - b).abs() < 2e-14);
    }
    for (a, b) in jtu.iter().zip(dense_jtu.iter()) {
        assert!((a - b).abs() < 2e-13);
    }
    let lhs: f64 = u.iter().zip(&jv).map(|(a, b)| a * b).sum();
    let rhs: f64 = v.iter().zip(&jtu).map(|(a, b)| a * b).sum();
    assert!((lhs - rhs).abs() < 2e-13);
    let weights: Vec<f64> = (0..op.sample_count())
        .map(|i| if i % 7 == 0 { 0.0 } else { 0.5 })
        .collect();
    for (j, norm) in op.column_norms(&weights).unwrap().iter().enumerate() {
        let expected = (0..weights.len())
            .map(|i| (weights[i] * e.jacobian.as_ref().unwrap()[(i, j)]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((norm - expected).abs() < 2e-14);
    }
    assert!(op.jvp(&[]).is_err());
    assert!(op.vjp(&vec![f64::NAN; op.sample_count()]).is_err());
    assert!(op.column_norms(&vec![-1.0; op.sample_count()]).is_err());
    assert!(op.materialize(1).is_err());
}

#[test]
fn hard_support_local_convergence_has_two_sided_objective_evidence() {
    let mut request = input(vec![40.0], vec![1.0], false);
    let profile =
        phasesmith_core::CwProfileParameters::from_instrument(40.0, request.instrument).unwrap();
    let edge = (40.0 + 20.0 * profile.tch.total_fwhm).next_up();
    request.pattern = PatternRecord::new(
        vec![39.8, 40.0, 40.2, edge],
        Some(vec![0.0; 4]),
        None,
        None,
        None,
    )
    .unwrap();
    request.parameters = ParameterSet::new(
        request
            .parameters
            .specs()
            .iter()
            .map(|s| {
                ParameterSpec::new(
                    s.key().clone(),
                    s.value(),
                    s.unit(),
                    s.bounds(),
                    s.scale(),
                    s.key().name() == "w_deg2",
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let mut observed = calc(&request).calculated_y;
    observed[1] -= 0.01;
    request.pattern.observed_y = Some(observed);
    let options = PawleyOptions::default();
    let transform = ConstraintTransform::new(request.parameters.clone(), vec![]).unwrap();
    let free = transform.pack().unwrap();
    let current = calc(&request);
    // In this one-width problem the central sample wants a broader peak,
    // but entering the rightmost sample raises the complete objective.
    for sign in [-1.0, 1.0] {
        let mut trial = free.clone();
        trial[0] += sign * 1e-10;
        let neighbor = evaluate_pawley(&request, &trial, 20.0, true, 50_000_000).unwrap();
        assert!(neighbor.residuals.chi_square > current.residuals.chi_square);
    }
    let checkpoint = PawleyCheckpoint {
        input: request.clone(),
        options: options.clone(),
        free,
        chi_square_history: vec![current.residuals.chi_square],
        damping: options.damping,
        linear_initialized: true,
        support_local: true,
    };
    let mut runtime = RefinementRuntime::new(RefinementLimits::default(), None).unwrap();
    let fit =
        refine_pawley_with_runtime(&request, &options, Some(&checkpoint), &mut runtime).unwrap();
    assert_eq!(fit.termination_reason, TerminationReason::Converged);
    assert_eq!(
        fit.diagnostics.convergence_criterion,
        Some("support_projected_gradient")
    );
    assert!(
        fit.covariance_limitation
            .as_ref()
            .unwrap()
            .contains("support")
    );
    // A downward jump must not be classified as a barrier.
    let mut changed = request.clone();
    let tail = profile.tch.evaluate(edge - 40.0).value;
    changed.pattern.observed_y.as_mut().unwrap()[3] = 2.0 * tail;
    let point = calc(&changed);
    let mut checkpoint = checkpoint;
    checkpoint.input = changed.clone();
    checkpoint.chi_square_history = vec![point.residuals.chi_square];
    let mut runtime = RefinementRuntime::new(RefinementLimits::default(), None).unwrap();
    let fit =
        refine_pawley_with_runtime(&changed, &options, Some(&checkpoint), &mut runtime).unwrap();
    assert!(
        !fit.diagnostics
            .convergence_criterion
            .unwrap_or("")
            .starts_with("support_")
    );
    assert!(fit.evaluation.residuals.chi_square < point.residuals.chi_square);
}

#[test]
fn simultaneous_support_events_require_non_cancelling_jumps() {
    let mut request = input(vec![40.0, 60.0], vec![1.0, 1.0], false);
    let edges: Vec<_> = [40.0, 60.0]
        .iter()
        .map(|&p| {
            let profile =
                phasesmith_core::CwProfileParameters::from_instrument(p, request.instrument)
                    .unwrap();
            (p + 20.0 * profile.tch.total_fwhm).next_up()
        })
        .collect();
    request.pattern = PatternRecord::new(
        vec![40.0, edges[0], 60.0, edges[1]],
        Some(vec![0.0; 4]),
        None,
        None,
        None,
    )
    .unwrap();
    request.parameters = ParameterSet::new(
        request
            .parameters
            .specs()
            .iter()
            .map(|s| {
                ParameterSpec::new(
                    s.key().clone(),
                    s.value(),
                    s.unit(),
                    s.bounds(),
                    s.scale(),
                    s.key().name() == "w_deg2",
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let mut observed = calc(&request).calculated_y;
    observed[0] -= 0.01;
    observed[2] -= 0.01;
    request.pattern.observed_y = Some(observed);
    let options = PawleyOptions {
        solver: phasesmith_workflows::PawleySolver::MatrixFree,
        ..PawleyOptions::default()
    };
    let free = ConstraintTransform::new(request.parameters.clone(), vec![])
        .unwrap()
        .pack()
        .unwrap();
    let current = calc(&request);
    for sign in [-1.0, 1.0] {
        let mut trial = free.clone();
        trial[0] += sign * 1e-10;
        assert!(
            evaluate_pawley(&request, &trial, 20.0, true, 50_000_000)
                .unwrap()
                .residuals
                .chi_square
                > current.residuals.chi_square
        );
    }
    let mut checkpoint = PawleyCheckpoint {
        input: request.clone(),
        options: options.clone(),
        free,
        chi_square_history: vec![current.residuals.chi_square],
        damping: options.damping,
        linear_initialized: true,
        support_local: true,
    };
    let mut runtime = RefinementRuntime::new(RefinementLimits::default(), None).unwrap();
    let fit =
        refine_pawley_with_runtime(&request, &options, Some(&checkpoint), &mut runtime).unwrap();
    assert_eq!(
        fit.diagnostics.convergence_criterion,
        Some("support_projected_gradient")
    );
    // One negative event invalidates certification of the independent jumps.
    request.pattern.observed_y.as_mut().unwrap()[3] = 1.0;
    checkpoint.input = request.clone();
    checkpoint.chi_square_history = vec![calc(&request).residuals.chi_square];
    let mut runtime = RefinementRuntime::new(RefinementLimits::default(), None).unwrap();
    let fit =
        refine_pawley_with_runtime(&request, &options, Some(&checkpoint), &mut runtime).unwrap();
    assert!(
        !fit.diagnostics
            .convergence_criterion
            .unwrap_or("")
            .starts_with("support_")
    );
}
