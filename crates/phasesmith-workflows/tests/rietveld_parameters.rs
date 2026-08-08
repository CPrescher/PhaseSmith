//! Native Rietveld parameter identity and derivative-transform contracts.

use phasesmith_core::{ConstantWavelengthInstrument, OwnedCwContributions, SupportPolicy};
use phasesmith_crystallography::{
    IntegratedIntensityCorrectionModel, Rational, SpaceGroup, SymmetryOperation, UnitCell,
};
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, PreparedStructuralPatternInputView,
    PreparedStructuralPhase, StructuralPhaseDefinition,
};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::PatternRecord;
use phasesmith_model::RecordId;
use phasesmith_workflows::{
    LatticeBounds, LatticeParameterization, PreparedRietveldObjective, RietveldCalculationOptions,
    RietveldInput, RietveldObjectiveError, RietveldParameterError, RietveldPhase,
    RietveldStructuralLayout, RietveldStructuralSelection, SiteCoordinateModel,
};

fn inversion_group() -> SpaceGroup {
    SpaceGroup::new(vec![
        SymmetryOperation::identity(),
        SymmetryOperation::new([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], [Rational::zero(); 3])
            .unwrap(),
    ])
    .unwrap()
}

fn definition() -> StructuralPhaseDefinition {
    StructuralPhaseDefinition {
        cell: UnitCell {
            a_angstrom: 4.7,
            b_angstrom: 5.1,
            c_angstrom: 6.2,
            alpha_deg: 82.0,
            beta_deg: 87.0,
            gamma_deg: 74.0,
        },
        space_group: inversion_group(),
        hkl: vec![[1, 0, 1], [2, 1, 1], [1, 2, 3]],
        multiplicity: vec![2, 4, 2],
        fractional_xyz: vec![[0.17, 0.23, 0.31], [0.0, 0.0, 0.0]],
        occupancy: vec![0.82, 0.55],
        u_iso_angstrom2: vec![0.012, 0.018],
        anisotropic_mask: vec![false, false],
        u_aniso_cif_angstrom2: vec![[0.0; 6]; 2],
        scattering_species: vec!["Si".to_owned(), "O".to_owned()],
        scattering_real_offset: Vec::new(),
        scattering_imag_offset: Vec::new(),
        scale: 1.4,
        coordinate_tolerance: 1.0e-10,
        scattering_model: BuiltInScatteringModel::XrayNonResonant,
        correction_model: IntegratedIntensityCorrectionModel::Neutral,
    }
}

fn phase() -> RietveldPhase {
    let definition = definition();
    RietveldPhase::new_with_site_ids(
        RecordId::new("alpha").unwrap(),
        "Alpha",
        vec![
            RecordId::new("Si1").unwrap(),
            RecordId::new("origin-O").unwrap(),
        ],
        definition.clone(),
        OwnedCwContributions::neutral(definition.hkl.len()),
    )
    .unwrap()
}

fn objective() -> PreparedRietveldObjective {
    let phase = phase();
    let layout = complete_layout(&phase);
    objective_for_phase(phase, layout).unwrap()
}

fn objective_for_phase(
    phase: RietveldPhase,
    layout: RietveldStructuralLayout,
) -> Result<PreparedRietveldObjective, RietveldObjectiveError> {
    let x_deg = (0..3_001)
        .map(|index| 10.0 + f64::from(index) * 0.03)
        .collect::<Vec<_>>();
    let mut mask = vec![true; x_deg.len()];
    mask[333] = false;
    let pattern = PatternRecord::new(
        x_deg.clone(),
        Some(vec![0.0; x_deg.len()]),
        Some(x_deg.iter().map(|value| 0.4 + value / 500.0).collect()),
        Some(mask),
        Some(x_deg.iter().map(|value| 0.1 + value / 1_000.0).collect()),
    )
    .unwrap();
    let input = RietveldInput::new(
        pattern,
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
            zero_shift_deg: 0.0,
            bragg_brentano_mm: None,
            debye_scherrer_micrometre: None,
        },
        vec![phase],
    )
    .unwrap();
    PreparedRietveldObjective::new(
        input,
        RietveldCalculationOptions::new(20.0, true, ExecutionPolicy::new(Some(1), 2).unwrap())
            .unwrap(),
        layout,
    )
}

fn complete_layout(phase: &RietveldPhase) -> RietveldStructuralLayout {
    let parameterization = LatticeParameterization::new(
        phase.definition().space_group.clone(),
        phase.definition().cell,
    )
    .unwrap();
    let bounds = LatticeBounds::around(&parameterization, 0.05, 3.0).unwrap();
    RietveldStructuralLayout::new(
        std::slice::from_ref(phase),
        RietveldStructuralSelection {
            lattice: true,
            coordinates: true,
            occupancy: true,
            u_iso: true,
            phase_scale: true,
        },
        &[Some(bounds)],
    )
    .unwrap()
}

#[test]
fn explicit_site_ids_and_symmetry_allowed_coordinate_models_are_stable() {
    let phase = phase();
    assert_eq!(phase.site_ids()[0].as_str(), "Si1");
    assert_eq!(phase.site_ids()[1].as_str(), "origin-O");
    let general = SiteCoordinateModel::new(
        &phase.definition().space_group,
        phase.definition().fractional_xyz[0],
        1.0e-10,
    )
    .unwrap();
    let fixed = SiteCoordinateModel::new(
        &phase.definition().space_group,
        phase.definition().fractional_xyz[1],
        1.0e-10,
    )
    .unwrap();
    assert_eq!(general.parameter_names(), ["x", "y", "z"]);
    assert_eq!(
        general.basis(),
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    );
    assert!(!general.is_special_position());
    assert!(fixed.parameter_names().is_empty());
    assert!(fixed.basis().is_empty());
    assert!(fixed.is_special_position());

    let mirror = SpaceGroup::new(vec![
        SymmetryOperation::identity(),
        SymmetryOperation::new([[1, 0, 0], [0, 1, 0], [0, 0, -1]], [Rational::zero(); 3]).unwrap(),
    ])
    .unwrap();
    let plane = SiteCoordinateModel::new(&mirror, [0.2, 0.3, 0.0], 1.0e-10).unwrap();
    assert_eq!(plane.parameter_names(), ["q0", "q1"]);
    assert_eq!(plane.basis(), [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    let definition = definition();
    assert!(matches!(
        RietveldPhase::new_with_site_ids(
            RecordId::new("bad").unwrap(),
            "Bad",
            vec![RecordId::new("only-one").unwrap()],
            definition.clone(),
            OwnedCwContributions::neutral(definition.hkl.len()),
        ),
        Err(phasesmith_workflows::RietveldError::SiteIdCountMismatch)
    ));
    assert!(matches!(
        RietveldPhase::new_with_site_ids(
            RecordId::new("bad").unwrap(),
            "Bad",
            vec![
                RecordId::new("same").unwrap(),
                RecordId::new("same").unwrap()
            ],
            definition.clone(),
            OwnedCwContributions::neutral(definition.hkl.len()),
        ),
        Err(phasesmith_workflows::RietveldError::DuplicateSiteId)
    ));
}

#[test]
fn complete_layout_has_stable_keys_bounds_and_expected_special_position_reduction() {
    let phase = phase();
    let layout = complete_layout(&phase);
    let labels = layout
        .parameters()
        .specs()
        .iter()
        .map(|spec| spec.key().label())
        .collect::<Vec<_>>();

    assert_eq!(labels.len(), 14);
    assert_eq!(
        &labels[..6],
        [
            "lattice[alpha].a_angstrom",
            "lattice[alpha].b_angstrom",
            "lattice[alpha].c_angstrom",
            "lattice[alpha].alpha_deg",
            "lattice[alpha].beta_deg",
            "lattice[alpha].gamma_deg",
        ]
    );
    assert_eq!(
        &labels[6..9],
        [
            "site[alpha/Si1].x",
            "site[alpha/Si1].y",
            "site[alpha/Si1].z",
        ]
    );
    assert!(!labels.iter().any(|label| label.contains("origin-O].q")));
    assert_eq!(labels.last().unwrap(), "phase[alpha].scale");
    assert!(
        layout
            .parameters()
            .specs()
            .iter()
            .all(phasesmith_workflows::ParameterSpec::refine)
    );
}

#[test]
fn transformed_jvp_and_vjp_are_adjoint_consistent() {
    let phase = phase();
    let layout = complete_layout(&phase);
    let direction = (0..layout.parameters().specs().len())
        .map(|index| 0.003 * (f64::from(u32::try_from(index).unwrap()) + 1.0))
        .collect::<Vec<_>>();
    let tangents = layout.native_tangents(&direction).unwrap();
    let execution = ExecutionPolicy::new(Some(1), 2).unwrap();
    let prepared =
        PreparedStructuralPhase::new(phase.definition().clone(), execution.context().clone())
            .unwrap();
    let x_deg = (0..3_001)
        .map(|index| 10.0 + f64::from(index) * 0.03)
        .collect::<Vec<_>>();
    let input = PreparedStructuralPatternInputView {
        x_deg: &x_deg,
        instrument: ConstantWavelengthInstrument {
            wavelength_angstrom: 1.5406,
            u_deg2: 2.0e-4,
            v_deg2: -1.0e-4,
            w_deg2: 1.2e-4,
            x_deg: 1.5e-3,
            y_deg: 3.0e-3,
        },
        axial_geometry: None,
        position_correction: MonochromaticPositionCorrection {
            zero_shift_deg: 0.0,
            bragg_brentano_mm: None,
            debye_scherrer_micrometre: None,
        },
        contributions: phase.contributions().as_view(),
        support: SupportPolicy::FwhmMultiple(20.0),
    };
    let forward = prepared.jvp(&input, &tangents[0]).unwrap();
    let weights = x_deg
        .iter()
        .map(|value| (0.07 * value).sin())
        .collect::<Vec<_>>();
    let reverse = prepared.vjp(&input, &weights).unwrap();
    let projected = layout
        .project_native_gradients(&[&reverse.gradient])
        .unwrap();
    let left = forward
        .d_y
        .iter()
        .zip(&weights)
        .map(|(value, weight)| value * weight)
        .sum::<f64>();
    let right = direction
        .iter()
        .zip(projected)
        .map(|(value, gradient)| value * gradient)
        .sum::<f64>();
    assert!((left - right).abs() < 5.0e-10);
}

#[test]
fn transform_shapes_and_lattice_requirements_fail_structurally() {
    let phase = phase();
    assert!(matches!(
        SiteCoordinateModel::new(
            &phase.definition().space_group,
            [f64::NAN, 0.0, 0.0],
            1.0e-10,
        ),
        Err(RietveldParameterError::InvalidCoordinateModel)
    ));
    assert!(matches!(
        RietveldStructuralLayout::new(
            std::slice::from_ref(&phase),
            RietveldStructuralSelection {
                lattice: true,
                ..RietveldStructuralSelection::default()
            },
            &[None],
        ),
        Err(RietveldParameterError::MissingLatticeBounds)
    ));
    let layout = complete_layout(&phase);
    assert!(matches!(
        layout.native_tangents(&[]),
        Err(RietveldParameterError::DirectionLengthMismatch)
    ));
    assert!(matches!(
        layout.project_native_gradients(&[]),
        Err(RietveldParameterError::PhaseCountMismatch)
    ));
    assert!(matches!(
        layout.project_native_gradients(&[&[]]),
        Err(RietveldParameterError::NativeGradientLengthMismatch)
    ));
}

#[test]
fn physical_values_rebuild_lattice_sites_and_scale_with_bounds() {
    let phase = phase();
    let layout = complete_layout(&phase);
    let mut values = layout
        .parameters()
        .specs()
        .iter()
        .map(phasesmith_workflows::ParameterSpec::value)
        .collect::<Vec<_>>();
    let index = |label: &str| {
        layout
            .parameters()
            .specs()
            .iter()
            .position(|spec| spec.key().label() == label)
            .unwrap()
    };
    values[index("lattice[alpha].a_angstrom")] += 0.02;
    values[index("site[alpha/Si1].x")] += 0.03;
    values[index("site[alpha/Si1].occupancy")] = 0.7;
    values[index("site[alpha/origin-O].u_iso_angstrom2")] = 0.025;
    values[index("phase[alpha].scale")] = 1.1;
    let updated = layout
        .apply_values(std::slice::from_ref(&phase), &values)
        .unwrap();
    let definition = updated[0].definition();
    assert!((definition.cell.a_angstrom - 4.72).abs() < 1.0e-14);
    assert!((definition.fractional_xyz[0][0] - 0.2).abs() < 1.0e-14);
    assert!(
        definition.fractional_xyz[1]
            .iter()
            .all(|value| value.to_bits() == 0.0_f64.to_bits())
    );
    assert_eq!(definition.occupancy[0].to_bits(), 0.7_f64.to_bits());
    assert_eq!(definition.u_iso_angstrom2[1].to_bits(), 0.025_f64.to_bits());
    assert_eq!(definition.scale.to_bits(), 1.1_f64.to_bits());

    values[index("site[alpha/Si1].occupancy")] = -0.1;
    assert!(matches!(
        layout.apply_values(std::slice::from_ref(&phase), &values),
        Err(RietveldParameterError::Parameter(
            phasesmith_workflows::ParameterError::ValueOutsideBounds { .. }
        ))
    ));
}

#[test]
fn prepared_objective_jvp_vjp_and_normal_product_are_consistent() {
    let objective = objective();
    let count = objective.layout().parameters().specs().len();
    let direction = (0..count)
        .map(|index| 0.002 * (f64::from(u32::try_from(index).unwrap()) + 1.0))
        .collect::<Vec<_>>();
    let (_, derivative) = objective.jvp(&direction).unwrap();
    let weights = derivative
        .iter()
        .enumerate()
        .map(|(index, value)| (0.013 * f64::from(u32::try_from(index).unwrap())).cos() * value)
        .collect::<Vec<_>>();
    let reverse = objective.vjp(&weights).unwrap();
    let left = derivative
        .iter()
        .zip(&weights)
        .map(|(a, b)| a * b)
        .sum::<f64>();
    let right = direction
        .iter()
        .zip(&reverse)
        .map(|(a, b)| a * b)
        .sum::<f64>();
    assert!(
        (left - right).abs() < 5.0e-13 * left.abs().max(right.abs()).max(1.0),
        "left={left:.17e}, right={right:.17e}, difference={:.17e}",
        left - right
    );

    let damping = 0.03;
    let actual = objective.normal_product(&direction, damping).unwrap();
    let mut columns = Vec::with_capacity(count);
    for parameter in 0..count {
        let mut unit = vec![0.0; count];
        unit[parameter] = 1.0;
        columns.push(objective.jvp(&unit).unwrap().1);
    }
    let sigma = (0..derivative.len())
        .map(|index| 0.4 + (10.0 + f64::from(u32::try_from(index).unwrap()) * 0.03) / 500.0)
        .collect::<Vec<_>>();
    for parameter in 0..count {
        let expected = columns[parameter]
            .iter()
            .zip(&derivative)
            .enumerate()
            .filter(|(index, _)| *index != 333)
            .map(|(index, (column, value))| column * value / (sigma[index] * sigma[index]))
            .sum::<f64>()
            + damping * direction[parameter];
        assert!(
            (actual[parameter] - expected).abs()
                < 5.0e-12 * actual[parameter].abs().max(expected.abs()).max(1.0)
        );
    }
}

#[test]
fn prepared_objective_gradient_and_invalid_damping_are_explicit() {
    let objective = objective();
    let (calculated, gradient) = objective.gradient().unwrap();
    assert_eq!(calculated.len(), 3_001);
    assert_eq!(
        gradient.len(),
        objective.layout().parameters().specs().len()
    );
    assert!(gradient.iter().all(|value| value.is_finite()));
    assert!(matches!(
        objective.normal_product(&vec![0.0; gradient.len()], f64::NAN),
        Err(RietveldObjectiveError::InvalidDamping)
    ));

    let source = phase();
    let layout = complete_layout(&source);
    let definition = source.definition().clone();
    let other = RietveldPhase::new_with_site_ids(
        RecordId::new("beta").unwrap(),
        "Beta",
        source.site_ids().to_vec(),
        definition.clone(),
        OwnedCwContributions::neutral(definition.hkl.len()),
    )
    .unwrap();
    assert!(matches!(
        objective_for_phase(other, layout),
        Err(RietveldObjectiveError::Parameter(
            RietveldParameterError::PhaseIdentityMismatch
        ))
    ));
}
