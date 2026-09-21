//! Public-facade integration coverage for the owned native calculation boundary.

use phasesmith::core::{ConstantWavelengthInstrument, OwnedCwContributions, SupportPolicy};
use phasesmith::crystallography::{
    IntegratedIntensityCorrectionModel, Rational, SpaceGroup, SymmetryOperation, UnitCell,
};
use phasesmith::engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, PreparedStructuralModel,
    PreparedStructuralMultiphase, PreparedStructuralPhase, PreparedStructuralSpectrum,
    StructuralCalculationRequest, StructuralModelInput, StructuralPhaseDefinition,
};
use phasesmith::execution::ExecutionPolicy;

fn phase_definition(scale: f64) -> StructuralPhaseDefinition {
    StructuralPhaseDefinition {
        cell: UnitCell {
            a_angstrom: 4.7,
            b_angstrom: 5.1,
            c_angstrom: 6.2,
            alpha_deg: 82.0,
            beta_deg: 87.0,
            gamma_deg: 74.0,
        },
        space_group: SpaceGroup::new(vec![
            SymmetryOperation::identity(),
            SymmetryOperation::new([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], [Rational::zero(); 3])
                .expect("inversion"),
        ])
        .expect("P-1"),
        hkl: vec![[1, 0, 1], [2, 1, 1], [1, 2, 3]],
        multiplicity: vec![2, 4, 2],
        fractional_xyz: vec![[0.17, 0.23, 0.31], [0.37, 0.11, 0.19]],
        occupancy: vec![0.82, 0.55],
        u_iso_angstrom2: vec![0.012, 0.018],
        anisotropic_mask: vec![false, false],
        u_aniso_cif_angstrom2: vec![[0.0; 6]; 2],
        scattering_species: vec!["Si".to_owned(), "O".to_owned()],
        scattering_real_offset: Vec::new(),
        scattering_imag_offset: Vec::new(),
        scale,
        coordinate_tolerance: 1.0e-10,
        scattering_model: BuiltInScatteringModel::XrayNonResonant,
        correction_model: IntegratedIntensityCorrectionModel::Neutral,
    }
}

fn prepared_multiphase(
    execution: ExecutionPolicy,
) -> Result<PreparedStructuralMultiphase, Box<dyn std::error::Error>> {
    let spectrum = PreparedStructuralSpectrum::new(
        &phase_definition(1.4),
        vec![1.5406, 1.54439],
        &[1.0, 0.5],
        execution.clone(),
    )?;
    let monochromatic =
        PreparedStructuralPhase::new(phase_definition(0.35), execution.context().clone())?;
    Ok(PreparedStructuralMultiphase::new(
        vec![
            PreparedStructuralModel::fixed_spectrum(spectrum),
            PreparedStructuralModel::monochromatic(monochromatic),
        ],
        execution,
    )?)
}

fn calculation_request() -> StructuralCalculationRequest {
    let reflection_count = 3;
    StructuralCalculationRequest {
        x_deg: (0..9_001)
            .map(|index| 10.0 + f64::from(index) * 0.01)
            .collect(),
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
            zero_shift_deg: 0.02,
            bragg_brentano_mm: None,
            debye_scherrer_micrometre: None,
        },
        phase_inputs: vec![
            StructuralModelInput {
                contributions: vec![
                    OwnedCwContributions::neutral(reflection_count),
                    OwnedCwContributions::neutral(reflection_count),
                ],
            },
            StructuralModelInput {
                contributions: vec![OwnedCwContributions::neutral(reflection_count)],
            },
        ],
        support: SupportPolicy::FwhmMultiple(20.0),
        profile_accuracy: phasesmith_core::ProfileAccuracy::default(),
        calculate_axial_derivatives: true,
    }
}

#[test]
fn rust_only_owned_request_calculates_display_ready_multiphase_spectrum()
-> Result<(), Box<dyn std::error::Error>> {
    let serial = prepared_multiphase(ExecutionPolicy::new(Some(1), 2)?)?
        .calculate_request(calculation_request())?;
    let parallel = prepared_multiphase(ExecutionPolicy::new(Some(2), 2)?)?
        .calculate_request(calculation_request())?;

    assert_eq!(parallel, serial);
    assert_eq!(parallel.x_deg.len(), 9_001);
    assert_eq!(parallel.profile_y.len(), parallel.x_deg.len());
    assert_eq!(parallel.phases.len(), 2);
    assert_eq!(parallel.phases[0].structure_factors.intensity.len(), 6);
    assert_eq!(parallel.phases[1].structure_factors.intensity.len(), 3);
    assert!(parallel.profile_y.iter().all(|value| value.is_finite()));
    assert!(parallel.profile_y.iter().any(|value| *value > 0.0));
    for sample in 0..parallel.profile_y.len() {
        let ordered_sum =
            parallel.phases[0].accumulation.y[sample] + parallel.phases[1].accumulation.y[sample];
        assert_eq!(parallel.profile_y[sample].to_bits(), ordered_sum.to_bits());
    }
    Ok(())
}
