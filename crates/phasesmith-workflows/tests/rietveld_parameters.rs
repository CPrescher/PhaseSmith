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
use phasesmith_model::RecordId;
use phasesmith_workflows::{
    LatticeBounds, LatticeParameterization, RietveldParameterError, RietveldPhase,
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
