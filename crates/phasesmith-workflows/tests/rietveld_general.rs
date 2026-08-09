//! Complete native Rietveld parameter/objective contracts.

use phasesmith_core::{ConstantWavelengthInstrument, OwnedCwContributions};
use phasesmith_crystallography::{IntegratedIntensityCorrectionModel, UnitCell};
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPhaseDefinition,
};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_io::space_group_by_number;
use phasesmith_model::{FixedWavelengthSpectrum, PatternRecord, RecordId};
use phasesmith_workflows::{
    BackgroundModel, DifferentiableBackground, LatticeBounds, LatticeParameterization,
    PolynomialBackground, PreparedGeneralRietveldObjective, RietveldCalculationOptions,
    RietveldInput, RietveldInstrumentParameter, RietveldParameterLayout,
    RietveldParameterSelection, RietveldPhase, RietveldSamplePhysicsModel,
    RietveldStructuralSelection, calculate_rietveld_pattern,
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

fn spectrum_input() -> RietveldInput {
    let monochromatic = input();
    RietveldInput::new_fixed_spectrum_with_background(
        monochromatic.pattern,
        monochromatic.instrument,
        FixedWavelengthSpectrum::new(vec![1.5406, 1.54439], vec![1.0, 0.48]).unwrap(),
        monochromatic.axial_geometry,
        monochromatic.position_correction,
        monochromatic.background.unwrap(),
        monochromatic.phases,
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

fn with_sample_physics(mut input: RietveldInput) -> RietveldInput {
    input.phases[0] =
        input.phases[0]
            .clone()
            .with_sample_physics(RietveldSamplePhysicsModel::Composite(vec![
                RietveldSamplePhysicsModel::IsotropicSize {
                    crystallite_size_nm: 70.0,
                    shape_factor: 0.9,
                },
                RietveldSamplePhysicsModel::IsotropicMicrostrain {
                    rms_microstrain: 7.0e-4,
                },
                RietveldSamplePhysicsModel::MarchDollase {
                    ratio: 0.82,
                    preferred_axis_hkl: [1.0, 0.3, -0.2],
                },
            ]));
    input
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
fn fixed_spectrum_matches_weighted_monochromatic_phase_sum() {
    let input = spectrum_input();
    let spectrum = calculate_rietveld_pattern(&input, &options()).unwrap();
    let weights = [1.0 / 1.48, 0.48 / 1.48];
    let wavelengths = [1.5406, 1.54439];
    let mut expected = vec![0.0; input.pattern.sample_count()];
    for (weight, wavelength) in weights.into_iter().zip(wavelengths) {
        let mut instrument = input.instrument;
        instrument.wavelength_angstrom = wavelength;
        let mut definition = input.phases[0].definition().clone();
        definition.scale *= weight;
        definition.correction_model = definition.correction_model.with_wavelength(wavelength);
        let phase = RietveldPhase::new(
            RecordId::new("alpha").unwrap(),
            "Alpha",
            definition,
            input.phases[0].contributions().clone(),
        )
        .unwrap();
        let mono = RietveldInput::new(
            input.pattern.clone(),
            instrument,
            input.axial_geometry,
            input.position_correction,
            vec![phase],
        )
        .unwrap();
        let calculation = calculate_rietveld_pattern(&mono, &options()).unwrap();
        for (target, value) in expected.iter_mut().zip(calculation.profile_y) {
            *target += value;
        }
    }
    for (index, (actual, expected)) in spectrum.profile_y.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2.0e-12 * expected.abs().max(1.0),
            "sample {index}: actual={actual:.12e}, expected={expected:.12e}"
        );
    }
}

#[test]
fn fixed_spectrum_general_products_match_differences_and_are_adjoint() {
    let input = with_sample_physics(spectrum_input());
    let selection = RietveldParameterSelection::new(
        RietveldStructuralSelection {
            phase_scale: true,
            ..RietveldStructuralSelection::default()
        },
        vec![
            RietveldInstrumentParameter::UDeg2,
            RietveldInstrumentParameter::ZeroShiftDeg,
        ],
        true,
        true,
    )
    .unwrap();
    let layout = RietveldParameterLayout::new(&input, &selection, &[None]).unwrap();
    let objective =
        PreparedGeneralRietveldObjective::new(input.clone(), options(), layout.clone()).unwrap();
    let direction = [2.0e-4, -0.03, 0.4, -0.2, 8.0, 2.0e-4, 0.1, 0.3];
    assert_eq!(layout.parameters().specs().len(), direction.len());
    let (_, analytical) = objective.jvp(&direction).unwrap();
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
    let plus = calculate_rietveld_pattern(
        &layout.apply_values(&input, &shifted(1.0)).unwrap(),
        &options(),
    )
    .unwrap();
    let minus = calculate_rietveld_pattern(
        &layout.apply_values(&input, &shifted(-1.0)).unwrap(),
        &options(),
    )
    .unwrap();
    for (index, ((plus, minus), analytical)) in
        plus.y.iter().zip(&minus.y).zip(&analytical).enumerate()
    {
        let numerical = (plus - minus) / (2.0 * step);
        assert!(
            (numerical - analytical).abs() <= 1.0e-4 * numerical.abs().max(1.0),
            "sample {index}: numerical={numerical:.12e}, analytical={analytical:.12e}"
        );
    }
    let weights = (0..input.pattern.sample_count())
        .map(|index| (f64::from(u32::try_from(index).unwrap()) * 0.11).sin())
        .collect::<Vec<_>>();
    let reverse = objective.vjp(&weights).unwrap();
    let left = analytical
        .iter()
        .zip(&weights)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let right = direction
        .iter()
        .zip(reverse)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    assert!((left - right).abs() <= 3.0e-10 * left.abs().max(1.0));
}

#[test]
fn fixed_spectrum_rejects_reference_wavelength_lattice_and_wavelength_refinement() {
    let mut invalid = spectrum_input();
    invalid.instrument.wavelength_angstrom = 1.0;
    assert!(invalid.validate().is_err());

    let wavelength = RietveldParameterSelection::new(
        RietveldStructuralSelection::default(),
        vec![RietveldInstrumentParameter::WavelengthAngstrom],
        false,
        false,
    )
    .unwrap();
    assert!(RietveldParameterLayout::new(&spectrum_input(), &wavelength, &[None]).is_err());
    let lattice = RietveldParameterSelection::new(
        RietveldStructuralSelection {
            lattice: true,
            ..RietveldStructuralSelection::default()
        },
        Vec::new(),
        false,
        false,
    )
    .unwrap();
    assert!(RietveldParameterLayout::new(&spectrum_input(), &lattice, &[None]).is_err());
}

#[test]
fn sample_physics_layout_installation_and_complete_products_are_analytical() {
    let input = with_sample_physics(input());
    let selection = RietveldParameterSelection::new(
        RietveldStructuralSelection {
            phase_scale: true,
            ..RietveldStructuralSelection::default()
        },
        Vec::new(),
        false,
        true,
    )
    .unwrap();
    let layout = RietveldParameterLayout::new(&input, &selection, &[None]).unwrap();
    let labels = layout
        .parameters()
        .specs()
        .iter()
        .map(|spec| spec.key().label())
        .collect::<Vec<_>>();
    assert_eq!(
        labels,
        [
            "sample[alpha].isotropic_size.crystallite_size_nm",
            "sample[alpha].isotropic_microstrain.rms",
            "sample[alpha].march_dollase.ratio",
            "phase[alpha].scale",
        ]
    );
    let values = layout
        .parameters()
        .specs()
        .iter()
        .map(phasesmith_workflows::ParameterSpec::value)
        .collect::<Vec<_>>();
    let direction = [8.0, 2.0e-4, 0.1, 0.2];
    let objective =
        PreparedGeneralRietveldObjective::new(input.clone(), options(), layout.clone()).unwrap();
    let (_, analytical) = objective.jvp(&direction).unwrap();
    let step = 1.0e-5;
    let shifted = |sign: f64| {
        values
            .iter()
            .zip(direction)
            .map(|(value, direction)| value + sign * step * direction)
            .collect::<Vec<_>>()
    };
    let plus = calculate_rietveld_pattern(
        &layout.apply_values(&input, &shifted(1.0)).unwrap(),
        &options(),
    )
    .unwrap();
    let minus = calculate_rietveld_pattern(
        &layout.apply_values(&input, &shifted(-1.0)).unwrap(),
        &options(),
    )
    .unwrap();
    for (index, ((plus, minus), analytical)) in
        plus.y.iter().zip(&minus.y).zip(&analytical).enumerate()
    {
        let numerical = (plus - minus) / (2.0 * step);
        assert!(
            (numerical - analytical).abs() <= 8.0e-5 * numerical.abs().max(1.0),
            "sample {index}: numerical={numerical:.12e}, analytical={analytical:.12e}"
        );
    }
    let installed = layout.apply_values(&input, &shifted(1.0)).unwrap();
    let installed_parameters = installed.phases[0]
        .sample_physics()
        .unwrap()
        .parameters()
        .unwrap();
    assert_eq!(
        installed_parameters[0].value.to_bits(),
        shifted(1.0)[0].to_bits()
    );

    let weights = (0..input.pattern.sample_count())
        .map(|index| (f64::from(u32::try_from(index).unwrap()) * 0.13).cos())
        .collect::<Vec<_>>();
    let reverse = objective.vjp(&weights).unwrap();
    let left = analytical
        .iter()
        .zip(&weights)
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
fn march_cell_chain_is_included_in_general_lattice_products() {
    let mut input = input();
    let mut definition = input.phases[0].definition().clone();
    definition.space_group = space_group_by_number(1).unwrap().space_group;
    definition.cell = UnitCell {
        a_angstrom: 4.1,
        b_angstrom: 4.8,
        c_angstrom: 5.7,
        alpha_deg: 82.0,
        beta_deg: 96.0,
        gamma_deg: 74.0,
    };
    input.phases[0] = RietveldPhase::new(
        RecordId::new("alpha").unwrap(),
        "Alpha",
        definition,
        OwnedCwContributions::neutral(4),
    )
    .unwrap()
    .with_sample_physics(RietveldSamplePhysicsModel::MarchDollase {
        ratio: 0.76,
        preferred_axis_hkl: [1.0, 0.2, -0.3],
    });
    let parameterization = LatticeParameterization::new(
        input.phases[0].definition().space_group.clone(),
        input.phases[0].definition().cell,
    )
    .unwrap();
    let bounds = LatticeBounds::around(&parameterization, 0.05, 3.0).unwrap();
    let selection = RietveldParameterSelection::new(
        RietveldStructuralSelection {
            lattice: true,
            ..RietveldStructuralSelection::default()
        },
        Vec::new(),
        false,
        true,
    )
    .unwrap();
    let layout = RietveldParameterLayout::new(&input, &selection, &[Some(bounds)]).unwrap();
    let values = layout
        .parameters()
        .specs()
        .iter()
        .map(phasesmith_workflows::ParameterSpec::value)
        .collect::<Vec<_>>();
    let direction = [0.08, 0.02, -0.03, 0.04, 0.3, -0.2, 0.25];
    assert_eq!(values.len(), direction.len());
    let objective =
        PreparedGeneralRietveldObjective::new(input.clone(), options(), layout.clone()).unwrap();
    let (_, analytical) = objective.jvp(&direction).unwrap();
    let step = 2.0e-5;
    let shifted = |sign: f64| {
        values
            .iter()
            .zip(direction)
            .map(|(value, direction)| value + sign * step * direction)
            .collect::<Vec<_>>()
    };
    let plus = calculate_rietveld_pattern(
        &layout.apply_values(&input, &shifted(1.0)).unwrap(),
        &options(),
    )
    .unwrap();
    let minus = calculate_rietveld_pattern(
        &layout.apply_values(&input, &shifted(-1.0)).unwrap(),
        &options(),
    )
    .unwrap();
    for (index, ((plus, minus), analytical)) in
        plus.y.iter().zip(&minus.y).zip(&analytical).enumerate()
    {
        let numerical = (plus - minus) / (2.0 * step);
        assert!(
            (numerical - analytical).abs() <= 1.5e-4 * numerical.abs().max(1.0),
            "sample {index}: numerical={numerical:.12e}, analytical={analytical:.12e}"
        );
    }
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
    let sample_physics = RietveldParameterSelection::new(
        RietveldStructuralSelection::default(),
        Vec::new(),
        false,
        true,
    )
    .unwrap();
    assert!(
        RietveldParameterLayout::new(&input(), &sample_physics, &[None])
            .unwrap()
            .parameters()
            .specs()
            .is_empty()
    );
}
