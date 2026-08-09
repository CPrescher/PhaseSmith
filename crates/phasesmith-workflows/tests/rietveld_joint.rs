//! Joint multi-histogram Rietveld objective contracts and numerical products.

use phasesmith_core::{ConstantWavelengthInstrument, OwnedCwContributions};
use phasesmith_crystallography::{IntegratedIntensityCorrectionModel, UnitCell};
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPhaseDefinition,
};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_io::space_group_by_number;
use phasesmith_model::{PatternRecord, RecordId};
use phasesmith_workflows::{
    AffineConstraint, BackgroundModel, CancellationToken, Constraint, JointRietveldHistogram,
    JointRietveldLayout, JointRietveldRefinementOptions, LatticeBounds, LatticeParameterization,
    ParameterKey, PolynomialBackground, PreparedJointRietveldObjective, RefinementLimits,
    RietveldCalculationOptions, RietveldInput, RietveldInstrumentParameter,
    RietveldParameterSelection, RietveldPhase, RietveldStructuralSelection, TerminationReason,
    calculate_rietveld_pattern, refine_joint_rietveld,
};

fn phase(
    scale: f64,
    scattering_model: BuiltInScatteringModel,
    hkl: Vec<[i32; 3]>,
) -> RietveldPhase {
    let definition = StructuralPhaseDefinition {
        cell: UnitCell {
            a_angstrom: 4.7,
            b_angstrom: 4.7,
            c_angstrom: 4.7,
            alpha_deg: 90.0,
            beta_deg: 90.0,
            gamma_deg: 90.0,
        },
        space_group: space_group_by_number(221).unwrap().space_group,
        multiplicity: vec![6, 12, 8, 6, 24][..hkl.len()].to_vec(),
        hkl,
        fractional_xyz: vec![[0.0, 0.0, 0.0]],
        occupancy: vec![0.9],
        u_iso_angstrom2: vec![0.015],
        anisotropic_mask: vec![false],
        u_aniso_cif_angstrom2: vec![[0.0; 6]],
        scattering_species: vec!["Si".to_owned()],
        scattering_real_offset: Vec::new(),
        scattering_imag_offset: Vec::new(),
        scale,
        coordinate_tolerance: 1.0e-10,
        scattering_model,
        correction_model: IntegratedIntensityCorrectionModel::Neutral,
    };
    RietveldPhase::new_with_site_ids(
        RecordId::new("alpha").unwrap(),
        "Alpha",
        vec![RecordId::new("Si1").unwrap()],
        definition.clone(),
        OwnedCwContributions::neutral(definition.hkl.len()),
    )
    .unwrap()
}

fn histogram(id: &str, neutron: bool) -> JointRietveldHistogram {
    let (start, step, samples, wavelength, scale): (f64, f64, usize, f64, f64) = if neutron {
        (18.0, 0.11, 620, 1.8, 1.3)
    } else {
        (20.0, 0.1, 601, 1.5406, 0.8)
    };
    let x_deg = (0..samples)
        .map(|index| start + f64::from(u32::try_from(index).unwrap()) * step)
        .collect::<Vec<_>>();
    let mask = neutron.then(|| {
        (0..samples)
            .map(|index| index % 17 != 0)
            .collect::<Vec<_>>()
    });
    let input = RietveldInput::new_with_background(
        PatternRecord::new(
            x_deg.clone(),
            Some(vec![0.0; samples]),
            Some(vec![if neutron { 0.7 } else { 0.5 }; samples]),
            mask,
            Some(vec![0.2; samples]),
        )
        .unwrap(),
        ConstantWavelengthInstrument {
            wavelength_angstrom: wavelength,
            u_deg2: if neutron { 3.0e-4 } else { 2.0e-4 },
            v_deg2: -1.0e-4,
            w_deg2: 1.2e-4,
            x_deg: 1.5e-3,
            y_deg: 3.0e-3,
        },
        None,
        MonochromaticPositionCorrection {
            zero_shift_deg: if neutron { -0.015 } else { 0.01 },
            bragg_brentano_mm: None,
            debye_scherrer_micrometre: None,
        },
        BackgroundModel::Polynomial(
            PolynomialBackground::new("main", vec![if neutron { 1.4 } else { 1.0 }, -0.1]).unwrap(),
        ),
        vec![phase(
            scale,
            if neutron {
                BuiltInScatteringModel::NeutronNuclear
            } else {
                BuiltInScatteringModel::XrayNonResonant
            },
            if neutron {
                vec![[1, 0, 0], [1, 1, 0], [2, 0, 0]]
            } else {
                vec![[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0]]
            },
        )],
    )
    .unwrap();
    let parameterization = LatticeParameterization::new(
        input.phases[0].definition().space_group.clone(),
        input.phases[0].definition().cell,
    )
    .unwrap();
    JointRietveldHistogram {
        histogram_id: RecordId::new(id).unwrap(),
        input,
        selection: RietveldParameterSelection::new(
            RietveldStructuralSelection {
                lattice: true,
                occupancy: true,
                u_iso: true,
                phase_scale: true,
                ..RietveldStructuralSelection::default()
            },
            vec![
                RietveldInstrumentParameter::UDeg2,
                RietveldInstrumentParameter::ZeroShiftDeg,
            ],
            true,
            false,
        )
        .unwrap(),
        lattice_bounds: vec![Some(
            LatticeBounds::around(&parameterization, 0.08, 2.0).unwrap(),
        )],
        calculation: RietveldCalculationOptions::new(
            20.0,
            true,
            ExecutionPolicy::new(Some(1), 1).unwrap(),
        )
        .unwrap(),
    }
}

fn histograms() -> Vec<JointRietveldHistogram> {
    vec![histogram("xray", false), histogram("neutron", true)]
}

fn solver_histograms() -> Vec<JointRietveldHistogram> {
    let mut starting = histograms();
    let mut truth = starting.clone();
    for (index, histogram) in truth.iter_mut().enumerate() {
        let mut definition = histogram.input.phases[0].definition().clone();
        definition.cell.a_angstrom = 4.705;
        definition.cell.b_angstrom = 4.705;
        definition.cell.c_angstrom = 4.705;
        definition.scale = if index == 0 { 1.05 } else { 1.55 };
        histogram.input.phases[0] = RietveldPhase::new_with_site_ids(
            RecordId::new("alpha").unwrap(),
            "Alpha",
            vec![RecordId::new("Si1").unwrap()],
            definition.clone(),
            OwnedCwContributions::neutral(definition.hkl.len()),
        )
        .unwrap();
        let calculation =
            calculate_rietveld_pattern(&histogram.input, &histogram.calculation).unwrap();
        starting[index].input.pattern.observed_y = Some(calculation.y);
        starting[index].selection = RietveldParameterSelection::new(
            RietveldStructuralSelection {
                lattice: true,
                phase_scale: true,
                ..RietveldStructuralSelection::default()
            },
            Vec::new(),
            false,
            false,
        )
        .unwrap();
    }
    starting
}

fn solver_options(max_iterations: usize) -> JointRietveldRefinementOptions {
    JointRietveldRefinementOptions::new(
        RefinementLimits::new(max_iterations, 2_000, None, 100).unwrap(),
        1,
        1.0e-12,
        1.0e-10,
        1.0e-6,
        10.0,
        0.3,
        1.0e-10,
        32,
        0.5,
        12,
    )
    .unwrap()
}

#[test]
fn layout_shares_structure_and_namespaces_every_local_parameter() {
    let histograms = histograms();
    let layout = JointRietveldLayout::new(&histograms).unwrap();
    let labels = layout
        .parameters()
        .specs()
        .iter()
        .map(|spec| spec.key().label())
        .collect::<Vec<_>>();
    assert_eq!(
        labels,
        [
            "instrument[xray/cw].u_deg2",
            "instrument[xray/cw].zero_shift_deg",
            "background[xray/main].coefficient_0",
            "background[xray/main].coefficient_1",
            "lattice[alpha].a_angstrom",
            "site[alpha/Si1].occupancy",
            "site[alpha/Si1].u_iso_angstrom2",
            "phase[xray/alpha].scale",
            "instrument[neutron/cw].u_deg2",
            "instrument[neutron/cw].zero_shift_deg",
            "background[neutron/main].coefficient_0",
            "background[neutron/main].coefficient_1",
            "phase[neutron/alpha].scale",
        ]
    );

    let mut values = layout
        .parameters()
        .specs()
        .iter()
        .map(phasesmith_workflows::ParameterSpec::value)
        .collect::<Vec<_>>();
    values[4] = 4.72;
    values[7] = 0.9;
    values[12] = 1.5;
    let updated = layout.apply_values(&histograms, &values).unwrap();
    assert_eq!(
        updated[0].input.phases[0]
            .definition()
            .cell
            .a_angstrom
            .to_bits(),
        4.72_f64.to_bits()
    );
    assert_eq!(
        updated[1].input.phases[0]
            .definition()
            .cell
            .a_angstrom
            .to_bits(),
        4.72_f64.to_bits()
    );
    assert_eq!(
        updated[0].input.phases[0].definition().scale.to_bits(),
        0.9_f64.to_bits()
    );
    assert_eq!(
        updated[1].input.phases[0].definition().scale.to_bits(),
        1.5_f64.to_bits()
    );
}

#[test]
fn joint_jvp_matches_differences_and_vjp_is_the_combined_adjoint() {
    let histograms = histograms();
    let layout = JointRietveldLayout::new(&histograms).unwrap();
    let objective =
        PreparedJointRietveldObjective::new(histograms.clone(), layout.clone()).unwrap();
    let direction = [
        1.0e-4, -0.02, 0.2, -0.1, 0.04, 0.03, 0.002, 0.2, -1.5e-4, 0.03, -0.3, 0.15, -0.25,
    ];
    let products = objective.jvp(&direction).unwrap();
    let values = layout
        .parameters()
        .specs()
        .iter()
        .map(phasesmith_workflows::ParameterSpec::value)
        .collect::<Vec<_>>();
    let step = 1.0e-5;
    let shifted = |sign: f64| {
        values
            .iter()
            .zip(direction)
            .map(|(value, direction)| value + sign * step * direction)
            .collect::<Vec<_>>()
    };
    let plus = layout.apply_values(&histograms, &shifted(1.0)).unwrap();
    let minus = layout.apply_values(&histograms, &shifted(-1.0)).unwrap();
    for histogram_index in 0..histograms.len() {
        let plus = calculate_rietveld_pattern(
            &plus[histogram_index].input,
            &plus[histogram_index].calculation,
        )
        .unwrap();
        let minus = calculate_rietveld_pattern(
            &minus[histogram_index].input,
            &minus[histogram_index].calculation,
        )
        .unwrap();
        for (sample, ((plus, minus), analytical)) in plus
            .y
            .iter()
            .zip(&minus.y)
            .zip(&products[histogram_index].derivative)
            .enumerate()
        {
            let numerical = (plus - minus) / (2.0 * step);
            assert!(
                (numerical - analytical).abs() <= 1.5e-4 * numerical.abs().max(1.0),
                "histogram {histogram_index}, sample {sample}: numerical={numerical:.12e}, analytical={analytical:.12e}"
            );
        }
    }

    let weights = histograms
        .iter()
        .enumerate()
        .map(|(histogram_index, histogram)| {
            (0..histogram.input.pattern.sample_count())
                .map(|sample| {
                    (f64::from(u32::try_from(sample).unwrap()) * 0.13
                        + f64::from(u32::try_from(histogram_index).unwrap()))
                    .sin()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let reverse = objective.vjp(&weights).unwrap();
    let left = products
        .iter()
        .zip(&weights)
        .flat_map(|(product, weights)| product.derivative.iter().zip(weights))
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let right = direction
        .iter()
        .zip(reverse)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    assert!((left - right).abs() <= 2.0e-10 * left.abs().max(1.0));
}

#[test]
fn joint_gradient_and_normal_operator_sum_all_weighted_histograms() {
    let histograms = histograms();
    let layout = JointRietveldLayout::new(&histograms).unwrap();
    let objective =
        PreparedJointRietveldObjective::new(histograms.clone(), layout.clone()).unwrap();
    let direction = vec![0.1; layout.parameters().specs().len()];
    let products = objective.jvp(&direction).unwrap();
    let weighted = products
        .iter()
        .zip(&histograms)
        .map(|(product, histogram)| {
            product
                .derivative
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    if histogram
                        .input
                        .pattern
                        .mask
                        .as_ref()
                        .is_some_and(|mask| !mask[index])
                    {
                        0.0
                    } else {
                        value / histogram.input.pattern.uncertainty.as_ref().unwrap()[index].powi(2)
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let expected = objective.vjp(&weighted).unwrap();
    let damping = 0.3;
    let actual = objective.normal_product(&direction, damping).unwrap();
    for ((actual, expected), direction) in actual.iter().zip(expected).zip(&direction) {
        assert!((actual - (expected + damping * direction)).abs() <= 1.0e-10);
    }

    let evaluated = objective.gradient().unwrap();
    let values = layout
        .parameters()
        .specs()
        .iter()
        .map(phasesmith_workflows::ParameterSpec::value)
        .collect::<Vec<_>>();
    let step = 1.0e-6;
    let objective_at = |sign: f64| {
        let shifted = values
            .iter()
            .zip(&direction)
            .map(|(value, direction)| value + sign * step * direction)
            .collect::<Vec<_>>();
        layout
            .apply_values(&histograms, &shifted)
            .unwrap()
            .iter()
            .map(|histogram| {
                0.5 * calculate_rietveld_pattern(&histogram.input, &histogram.calculation)
                    .unwrap()
                    .metrics
                    .chi_square
            })
            .sum::<f64>()
    };
    let numerical = (objective_at(1.0) - objective_at(-1.0)) / (2.0 * step);
    let analytical = evaluated
        .gradient
        .iter()
        .zip(&direction)
        .map(|(gradient, direction)| gradient * direction)
        .sum::<f64>();
    assert!((numerical - analytical).abs() <= 2.0e-5 * numerical.abs().max(1.0));
}

#[test]
fn invalid_joint_identity_selection_and_shared_structure_are_rejected() {
    let one = vec![histogram("xray", false)];
    assert!(JointRietveldLayout::new(&one).is_err());

    let mut duplicate = histograms();
    duplicate[1].histogram_id = duplicate[0].histogram_id.clone();
    assert!(JointRietveldLayout::new(&duplicate).is_err());

    let mut selection = histograms();
    selection[1].selection.structural.occupancy = false;
    assert!(JointRietveldLayout::new(&selection).is_err());

    let original = histograms();
    let layout = JointRietveldLayout::new(&original).unwrap();
    let mut stale = original.clone();
    stale[1].selection.background = false;
    assert!(layout.apply_values(&stale, &[0.0; 13]).is_err());

    let mut structure = histograms();
    let mut definition = structure[1].input.phases[0].definition().clone();
    definition.occupancy[0] = 0.8;
    structure[1].input.phases[0] = RietveldPhase::new_with_site_ids(
        RecordId::new("alpha").unwrap(),
        "Alpha",
        vec![RecordId::new("Si1").unwrap()],
        definition.clone(),
        OwnedCwContributions::neutral(definition.hkl.len()),
    )
    .unwrap();
    assert!(JointRietveldLayout::new(&structure).is_err());
}

#[test]
fn joint_solver_recovers_shared_cell_and_local_scales_from_one_summed_fit() {
    let histograms = solver_histograms();
    let initial = PreparedJointRietveldObjective::new(
        histograms.clone(),
        JointRietveldLayout::new(&histograms).unwrap(),
    )
    .unwrap()
    .gradient()
    .unwrap()
    .objective;
    let result = refine_joint_rietveld(&histograms, &[], solver_options(15), None, None).unwrap();
    assert!(!result.history.is_empty());
    assert!(
        result.checkpoint.objective < initial * 1.0e-8,
        "initial={initial:.12e}, final={:.12e}, termination={:?}, history={:?}",
        result.checkpoint.objective,
        result.termination_reason,
        result
            .history
            .iter()
            .map(|row| row.objective)
            .collect::<Vec<_>>()
    );
    assert!(
        (result.histograms[0].input.phases[0]
            .definition()
            .cell
            .a_angstrom
            - 4.705)
            .abs()
            < 1.0e-7
    );
    assert!(
        (result.histograms[1].input.phases[0]
            .definition()
            .cell
            .a_angstrom
            - 4.705)
            .abs()
            < 1.0e-7
    );
    assert!(
        (result.histograms[0].input.phases[0].definition().scale - 1.05).abs() < 2.0e-4,
        "xray scale={}",
        result.histograms[0].input.phases[0].definition().scale
    );
    assert!(
        (result.histograms[1].input.phases[0].definition().scale - 1.55).abs() < 2.0e-4,
        "neutron scale={}",
        result.histograms[1].input.phases[0].definition().scale
    );
    let summed = result
        .calculations
        .iter()
        .map(|calculation| 0.5 * calculation.metrics.chi_square)
        .sum::<f64>();
    assert!((summed - result.checkpoint.objective).abs() <= 1.0e-12 * summed.abs().max(1.0));
    assert_eq!(
        result.metrics.chi_square.to_bits(),
        (2.0 * result.checkpoint.objective).to_bits()
    );
    assert_eq!(
        result.metrics.included_samples,
        histograms
            .iter()
            .map(|histogram| {
                histogram
                    .input
                    .pattern
                    .mask
                    .as_ref()
                    .map_or(histogram.input.pattern.sample_count(), |mask| {
                        mask.iter().filter(|included| **included).count()
                    })
            })
            .sum()
    );
    assert!(
        result
            .history
            .windows(2)
            .all(|rows| rows[1].objective < rows[0].objective)
    );
}

#[test]
fn joint_checkpoint_resumes_and_pre_cancel_is_a_normal_unchanged_result() {
    let histograms = solver_histograms();
    let partial = refine_joint_rietveld(&histograms, &[], solver_options(2), None, None).unwrap();
    partial.checkpoint.validate_for(&histograms, &[]).unwrap();
    let mut corrupt = partial.checkpoint.clone();
    corrupt.objective += 1.0;
    assert!(corrupt.validate_for(&histograms, &[]).is_err());
    let resumed = refine_joint_rietveld(
        &histograms,
        &[],
        solver_options(15),
        Some(&partial.checkpoint),
        None,
    )
    .unwrap();
    assert!(resumed.history.len() >= partial.history.len());
    assert!(resumed.checkpoint.objective <= partial.checkpoint.objective);

    let cancellation = CancellationToken::default();
    cancellation.request("joint test cancellation").unwrap();
    let cancelled = refine_joint_rietveld(
        &histograms,
        &[],
        solver_options(5),
        None,
        Some(cancellation),
    )
    .unwrap();
    assert_eq!(cancelled.termination_reason, TerminationReason::Cancelled);
    assert!(cancelled.history.is_empty());
    assert_eq!(cancelled.histograms, histograms);
}

#[test]
fn joint_constraints_link_histogram_local_parameters_in_the_same_solve() {
    let histograms = solver_histograms();
    let constraint = Constraint::Affine(
        AffineConstraint::new(
            ParameterKey::new("phase", "neutron/alpha", "scale").unwrap(),
            ParameterKey::new("phase", "xray/alpha", "scale").unwrap(),
            1.0,
            0.5,
        )
        .unwrap(),
    );
    let result = refine_joint_rietveld(
        &histograms,
        std::slice::from_ref(&constraint),
        solver_options(15),
        None,
        None,
    )
    .unwrap();
    let xray = result.histograms[0].input.phases[0].definition().scale;
    let neutron = result.histograms[1].input.phases[0].definition().scale;
    assert!((neutron - xray - 0.5).abs() < 1.0e-12);
    assert!(
        result
            .free_keys
            .contains(&ParameterKey::new("phase", "xray/alpha", "scale").unwrap())
    );
    assert!(
        !result
            .free_keys
            .contains(&ParameterKey::new("phase", "neutron/alpha", "scale").unwrap())
    );
    result
        .checkpoint
        .validate_for(&histograms, &[constraint])
        .unwrap();
}
