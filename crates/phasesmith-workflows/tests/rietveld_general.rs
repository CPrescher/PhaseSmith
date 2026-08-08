//! Complete native Rietveld parameter/objective contracts.

use phasesmith_core::{ConstantWavelengthInstrument, OwnedCwContributions};
use phasesmith_crystallography::{IntegratedIntensityCorrectionModel, UnitCell};
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPhaseDefinition,
};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_io::space_group_by_number;
use phasesmith_model::{PatternRecord, RecordId};
use phasesmith_workflows::{
    BackgroundModel, DifferentiableBackground, PolynomialBackground,
    PreparedGeneralRietveldObjective, RietveldCalculationOptions, RietveldInput,
    RietveldInstrumentParameter, RietveldParameterLayout, RietveldParameterSelection,
    RietveldPhase, RietveldStructuralSelection, calculate_rietveld_pattern,
};

fn phase(scale: f64) -> RietveldPhase {
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
        hkl: vec![[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0]],
        multiplicity: vec![6, 12, 8, 6],
        fractional_xyz: vec![[0.0, 0.0, 0.0]],
        occupancy: vec![1.0],
        u_iso_angstrom2: vec![0.01],
        anisotropic_mask: vec![false],
        u_aniso_cif_angstrom2: vec![[0.0; 6]],
        scattering_species: vec!["Si".to_owned()],
        scattering_real_offset: Vec::new(),
        scattering_imag_offset: Vec::new(),
        scale,
        coordinate_tolerance: 1.0e-10,
        scattering_model: BuiltInScatteringModel::XrayNonResonant,
        correction_model: IntegratedIntensityCorrectionModel::Neutral,
    };
    RietveldPhase::new(
        RecordId::new("alpha").unwrap(),
        "Alpha",
        definition.clone(),
        OwnedCwContributions::neutral(definition.hkl.len()),
    )
    .unwrap()
}

fn input() -> RietveldInput {
    let x_deg = (0..2_001)
        .map(|index| 20.0 + f64::from(index) * 0.03)
        .collect::<Vec<_>>();
    RietveldInput::new_with_background(
        PatternRecord::new(
            x_deg.clone(),
            Some(vec![0.0; x_deg.len()]),
            Some(vec![0.5; x_deg.len()]),
            None,
            Some(vec![0.2; x_deg.len()]),
        )
        .unwrap(),
        ConstantWavelengthInstrument {
            wavelength_angstrom: 1.5406,
            u_deg2: 2.0e-4,
            v_deg2: -1.0e-4,
            w_deg2: 1.2e-4,
            x_deg: 1.5e-3,
            y_deg: 3.0e-3,
        },
        None,
        MonochromaticPositionCorrection {
            zero_shift_deg: 0.01,
            bragg_brentano_mm: None,
            debye_scherrer_micrometre: None,
        },
        BackgroundModel::Polynomial(PolynomialBackground::new("main", vec![1.0, -0.1]).unwrap()),
        vec![phase(0.8)],
    )
    .unwrap()
}

fn selection() -> RietveldParameterSelection {
    RietveldParameterSelection::new(
        RietveldStructuralSelection {
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
    .unwrap()
}

fn options() -> RietveldCalculationOptions {
    RietveldCalculationOptions::new(20.0, true, ExecutionPolicy::new(Some(1), 1).unwrap()).unwrap()
}

#[test]
fn layout_has_stable_order_and_installs_every_selected_family() {
    let input = input();
    let layout = RietveldParameterLayout::new(&input, &selection(), &[None]).unwrap();
    let labels = layout
        .parameters()
        .specs()
        .iter()
        .map(|spec| spec.key().label())
        .collect::<Vec<_>>();
    assert_eq!(
        labels,
        [
            "instrument[cw].u_deg2",
            "instrument[cw].zero_shift_deg",
            "background[main].coefficient_0",
            "background[main].coefficient_1",
            "phase[alpha].scale",
        ]
    );
    let mut values = layout
        .parameters()
        .specs()
        .iter()
        .map(phasesmith_workflows::ParameterSpec::value)
        .collect::<Vec<_>>();
    values[0] = 3.0e-4;
    values[1] = -0.02;
    values[2] = 1.5;
    values[3] = 0.25;
    values[4] = 1.2;
    let updated = layout.apply_values(&input, &values).unwrap();
    assert_eq!(updated.instrument.u_deg2.to_bits(), 3.0e-4_f64.to_bits());
    assert_eq!(
        updated.position_correction.zero_shift_deg.to_bits(),
        (-0.02_f64).to_bits()
    );
    assert_eq!(
        updated.phases[0].definition().scale.to_bits(),
        1.2_f64.to_bits()
    );
    assert_eq!(updated.background.unwrap().coefficients(), vec![1.5, 0.25]);
}

#[test]
fn complete_jvp_matches_centered_differences_and_vjp_is_adjoint() {
    let input = input();
    let layout = RietveldParameterLayout::new(&input, &selection(), &[None]).unwrap();
    let objective =
        PreparedGeneralRietveldObjective::new(input.clone(), options(), layout.clone()).unwrap();
    let direction = vec![2.0e-4, -0.03, 0.4, -0.2, 0.3];
    let (_, analytical) = objective.jvp(&direction).unwrap();
    let values = layout
        .parameters()
        .specs()
        .iter()
        .map(phasesmith_workflows::ParameterSpec::value)
        .collect::<Vec<_>>();
    let step = 1.0e-5;
    let plus_values = values
        .iter()
        .zip(&direction)
        .map(|(value, direction)| value + step * direction)
        .collect::<Vec<_>>();
    let minus_values = values
        .iter()
        .zip(&direction)
        .map(|(value, direction)| value - step * direction)
        .collect::<Vec<_>>();
    let plus = calculate_rietveld_pattern(
        &layout.apply_values(&input, &plus_values).unwrap(),
        &options(),
    )
    .unwrap();
    let minus = calculate_rietveld_pattern(
        &layout.apply_values(&input, &minus_values).unwrap(),
        &options(),
    )
    .unwrap();
    for (index, ((plus, minus), analytical)) in
        plus.y.iter().zip(&minus.y).zip(&analytical).enumerate()
    {
        let numerical = (plus - minus) / (2.0 * step);
        assert!(
            (numerical - analytical).abs() <= 2.0e-5 * numerical.abs().max(1.0),
            "sample {index}: numerical={numerical:.12e}, analytical={analytical:.12e}"
        );
    }

    let samples = (0..input.pattern.sample_count())
        .map(|index| (f64::from(u32::try_from(index).unwrap()) * 0.17).sin())
        .collect::<Vec<_>>();
    let reverse = objective.vjp(&samples).unwrap();
    let left = analytical
        .iter()
        .zip(&samples)
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
fn invalid_general_selections_fail_before_preparation() {
    let duplicate = RietveldParameterSelection::new(
        RietveldStructuralSelection::default(),
        vec![
            RietveldInstrumentParameter::UDeg2,
            RietveldInstrumentParameter::UDeg2,
        ],
        false,
        false,
    );
    assert!(duplicate.is_err());
    let geometry = RietveldParameterSelection::new(
        RietveldStructuralSelection::default(),
        vec![RietveldInstrumentParameter::SampleDisplacementMm],
        false,
        false,
    )
    .unwrap();
    assert!(RietveldParameterLayout::new(&input(), &geometry, &[None]).is_err());
    assert!(
        RietveldParameterSelection::new(
            RietveldStructuralSelection::default(),
            Vec::new(),
            false,
            true,
        )
        .is_err()
    );
}
