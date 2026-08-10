//! Owned native Rietveld calculation-boundary contracts.

use std::process::Command;

use phasesmith_core::{
    ConstantWavelengthInstrument, FcjGeometry, OwnedCwContributionArrays, OwnedCwContributions,
};
use phasesmith_crystallography::{
    IntegratedIntensityCorrectionModel, Rational, SpaceGroup, SymmetryOperation, UnitCell,
};
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPhaseDefinition,
};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::{PatternRecord, RecordId};
use phasesmith_workflows::{
    BackgroundModel, PolynomialBackground, RietveldCalculationOptions, RietveldError,
    RietveldInput, RietveldPhase, RietveldSamplePhysicsModel, calculate_rietveld_pattern,
    estimate_initial_phase_scales,
};

fn instrument() -> ConstantWavelengthInstrument {
    ConstantWavelengthInstrument {
        wavelength_angstrom: 1.5406,
        u_deg2: 2.0e-4,
        v_deg2: -1.0e-4,
        w_deg2: 1.2e-4,
        x_deg: 1.5e-3,
        y_deg: 3.0e-3,
    }
}

#[test]
fn analytical_background_is_added_once_and_exposed_separately() {
    let x_deg = (0..101)
        .map(|index| 20.0 + f64::from(index) * 0.2)
        .collect::<Vec<_>>();
    let fixed = vec![0.25; x_deg.len()];
    let pattern = PatternRecord::new(
        x_deg.clone(),
        Some(vec![0.0; x_deg.len()]),
        None,
        None,
        Some(fixed.clone()),
    )
    .unwrap();
    let model = PolynomialBackground::new("main", vec![1.5, -0.2]).unwrap();
    let request = RietveldInput::new_with_background(
        pattern,
        instrument(),
        None,
        MonochromaticPositionCorrection {
            zero_shift_deg: 0.0,
            bragg_brentano_mm: None,
            debye_scherrer_micrometre: None,
        },
        BackgroundModel::Polynomial(model),
        vec![phase("alpha", 1.0)],
    )
    .unwrap();
    let calculated = calculate_rietveld_pattern(&request, &options(1)).unwrap();
    for (index, background) in calculated.background_y.iter().enumerate() {
        let normalized =
            2.0 * (x_deg[index] - x_deg[0]) / (x_deg[x_deg.len() - 1] - x_deg[0]) - 1.0;
        let expected = fixed[index] + 1.5 - 0.2 * normalized;
        assert!((background - expected).abs() < 1.0e-14);
        assert!((calculated.y[index] - calculated.profile_y[index] - background).abs() < 1.0e-14);
    }
}

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
        .expect("P -1"),
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

fn phase(id: &str, scale: f64) -> RietveldPhase {
    let definition = phase_definition(scale);
    let contributions = OwnedCwContributions::neutral(definition.hkl.len());
    RietveldPhase::new(
        RecordId::new(id).expect("ID"),
        format!("Phase {id}"),
        definition,
        contributions,
    )
    .expect("phase")
}

fn phase_with_a(id: &str, scale: f64, a_angstrom: f64) -> RietveldPhase {
    let mut definition = phase_definition(scale);
    definition.cell.a_angstrom = a_angstrom;
    let contributions = OwnedCwContributions::neutral(definition.hkl.len());
    RietveldPhase::new(
        RecordId::new(id).expect("ID"),
        format!("Phase {id}"),
        definition,
        contributions,
    )
    .expect("phase")
}

fn options(threads: usize) -> RietveldCalculationOptions {
    RietveldCalculationOptions::new(
        20.0,
        true,
        ExecutionPolicy::new(Some(threads), 2).expect("execution"),
    )
    .expect("options")
}

fn input(pattern: PatternRecord, phases: Vec<RietveldPhase>) -> RietveldInput {
    RietveldInput::new(
        pattern,
        instrument(),
        None,
        MonochromaticPositionCorrection {
            zero_shift_deg: 0.02,
            bragg_brentano_mm: None,
            debye_scherrer_micrometre: None,
        },
        phases,
    )
    .expect("input")
}

#[test]
fn initial_phase_scale_estimation_respects_mask_uncertainty_and_background() {
    let x_deg = (0..4_001)
        .map(|index| 10.0 + f64::from(index) * 0.03)
        .collect::<Vec<_>>();
    let background = x_deg
        .iter()
        .map(|x| 3.0 + 0.01 * (x - 10.0))
        .collect::<Vec<_>>();
    let blank = PatternRecord::new(
        x_deg.clone(),
        Some(vec![0.0; x_deg.len()]),
        None,
        None,
        Some(background.clone()),
    )
    .expect("truth pattern");
    let truth = calculate_rietveld_pattern(&input(blank, vec![phase("alpha", 2.5)]), &options(1))
        .expect("truth calculation");
    let uncertainty = (0..x_deg.len())
        .map(|index| {
            0.8 + 0.1 * f64::from(u32::try_from(index % 7).expect("remainder fits in u32"))
        })
        .collect::<Vec<_>>();
    let mask = (0..x_deg.len())
        .map(|index| (250..3_800).contains(&index) && !(1_700..1_900).contains(&index))
        .collect::<Vec<_>>();
    let observed = PatternRecord::new(
        x_deg,
        Some(truth.y),
        Some(uncertainty),
        Some(mask),
        Some(background),
    )
    .expect("observed pattern");
    let starting = input(observed, vec![phase("alpha", 0.15)]);

    let estimated = estimate_initial_phase_scales(&starting, &options(1))
        .expect("single phase scale is identifiable");

    assert!((estimated.scales[0] - 2.5).abs() < 1.0e-10);
    assert_eq!(estimated.included_points, 3_350);
    assert_eq!(estimated.active_phases, 1);
    assert!(estimated.weighted_residual_sum_squares < 1.0e-18);
    assert!((estimated.input.phases[0].definition().scale - 2.5).abs() < 1.0e-10);
}

#[test]
fn initial_phase_scale_estimation_recovers_multiple_non_negative_scales() {
    let x_deg = (0..4_001)
        .map(|index| 10.0 + f64::from(index) * 0.03)
        .collect::<Vec<_>>();
    let blank = PatternRecord::new(
        x_deg.clone(),
        Some(vec![0.0; x_deg.len()]),
        None,
        None,
        Some(vec![0.0; x_deg.len()]),
    )
    .expect("truth pattern");
    let truth = calculate_rietveld_pattern(
        &input(
            blank,
            vec![
                phase_with_a("alpha", 1.7, 4.7),
                phase_with_a("beta", 0.45, 5.35),
            ],
        ),
        &options(1),
    )
    .expect("truth calculation");
    let observed = PatternRecord::new(x_deg, Some(truth.y), None, None, Some(vec![0.0; 4_001]))
        .expect("observed pattern");
    let starting = input(
        observed,
        vec![
            phase_with_a("alpha", 0.01, 4.7),
            phase_with_a("beta", 4.0, 5.35),
        ],
    );

    let estimated = estimate_initial_phase_scales(&starting, &options(1))
        .expect("two phase scales are identifiable");

    assert!((estimated.scales[0] - 1.7).abs() < 1.0e-10);
    assert!((estimated.scales[1] - 0.45).abs() < 1.0e-10);
    assert_eq!(estimated.active_phases, 2);
    assert!(estimated.iterations >= 2);
}

#[test]
fn attached_native_sample_model_drives_calculation_and_derivative_rows() {
    let x_deg = (0..4_001)
        .map(|index| 10.0 + f64::from(index) * 0.03)
        .collect::<Vec<_>>();
    let pattern = PatternRecord::new(
        x_deg.clone(),
        Some(vec![0.0; x_deg.len()]),
        None,
        None,
        Some(vec![0.0; x_deg.len()]),
    )
    .unwrap();
    let neutral = calculate_rietveld_pattern(
        &input(pattern.clone(), vec![phase("alpha", 1.0)]),
        &options(1),
    )
    .unwrap();
    let physical_phase =
        phase("alpha", 1.0).with_sample_physics(RietveldSamplePhysicsModel::Composite(vec![
            RietveldSamplePhysicsModel::IsotropicSize {
                crystallite_size_nm: 55.0,
                shape_factor: 0.9,
            },
            RietveldSamplePhysicsModel::IsotropicMicrostrain {
                rms_microstrain: 8.0e-4,
            },
        ]));
    let physical =
        calculate_rietveld_pattern(&input(pattern, vec![physical_phase]), &options(1)).unwrap();
    assert_ne!(physical.profile_y, neutral.profile_y);
    let global = physical.phases[0]
        .result
        .accumulation
        .derivatives
        .global
        .as_ref()
        .unwrap();
    assert_eq!(global.parameter_count, 9);
    assert!(
        global.values[7 * global.sample_count..]
            .iter()
            .any(|value| *value != 0.0)
    );
}

#[test]
fn removing_sample_physics_preserves_phase_topology_and_fixed_contributions() {
    let original = phase("alpha", 1.0);
    let attached =
        original
            .clone()
            .with_sample_physics(RietveldSamplePhysicsModel::IsotropicMicrostrain {
                rms_microstrain: 5.0e-4,
            });

    let removed = attached.without_sample_physics();

    assert!(removed.sample_physics().is_none());
    assert_eq!(removed.definition(), original.definition());
    assert_eq!(removed.reflection_ids(), original.reflection_ids());
    assert_eq!(removed.contributions(), original.contributions());
}

#[test]
fn multiphase_pattern_is_display_ready_and_worker_deterministic() {
    let x_deg = (0..9_001)
        .map(|index| 10.0 + f64::from(index) * 0.01)
        .collect::<Vec<_>>();
    let background = x_deg
        .iter()
        .map(|value| 0.2 + 1.0e-3 * value)
        .collect::<Vec<_>>();
    let uncertainty = vec![0.5; x_deg.len()];
    let phases = vec![phase("alpha", 1.4), phase("beta", 0.35)];
    let seed_pattern = PatternRecord::new(
        x_deg.clone(),
        Some(vec![0.0; x_deg.len()]),
        Some(uncertainty.clone()),
        None,
        Some(background.clone()),
    )
    .expect("seed pattern");
    let seed = calculate_rietveld_pattern(&input(seed_pattern, phases.clone()), &options(1))
        .expect("seed calculation");
    let observed_pattern = PatternRecord::new(
        x_deg,
        Some(seed.y.clone()),
        Some(uncertainty),
        None,
        Some(background),
    )
    .expect("observed pattern");
    let request = input(observed_pattern, phases);
    let serial = calculate_rietveld_pattern(&request, &options(1)).expect("serial");
    let parallel = calculate_rietveld_pattern(&request, &options(2)).expect("parallel");

    assert_eq!(parallel, serial);
    assert_eq!(serial.phases.len(), 2);
    assert_eq!(serial.phases[0].phase_id.as_str(), "alpha");
    assert_eq!(serial.phases[1].phase_id.as_str(), "beta");
    assert_eq!(serial.profile_y.len(), serial.y.len());
    for sample in 0..serial.y.len() {
        assert_eq!(
            serial.y[sample].to_bits(),
            (serial.profile_y[sample] + serial.background_y[sample]).to_bits()
        );
        assert_eq!(
            serial.profile_y[sample].to_bits(),
            (serial.phases[0].result.accumulation.y[sample]
                + serial.phases[1].result.accumulation.y[sample])
                .to_bits()
        );
    }
    assert!(serial.metrics.chi_square < 1.0e-20);
    assert!(serial.metrics.rwp < 1.0e-14);
}

#[test]
fn invalid_owned_requests_fail_before_calculation() {
    let definition = phase_definition(1.0);
    assert!(matches!(
        RietveldPhase::new(
            RecordId::new("bad").unwrap(),
            "",
            definition.clone(),
            OwnedCwContributions::neutral(definition.hkl.len()),
        ),
        Err(RietveldError::InvalidPhaseName)
    ));
    assert!(matches!(
        RietveldPhase::new(
            RecordId::new("bad").unwrap(),
            "Bad",
            definition,
            OwnedCwContributions::neutral(1),
        ),
        Err(RietveldError::ContributionCountMismatch)
    ));
    let x = vec![20.0, 21.0];
    let no_observations = PatternRecord::new(x.clone(), None, None, None, None).unwrap();
    assert!(matches!(
        input_result(no_observations, vec![phase("alpha", 1.0)]),
        Err(RietveldError::MissingObservations)
    ));
    let observed = PatternRecord::new(x, Some(vec![0.0; 2]), None, None, None).unwrap();
    assert!(matches!(
        input_result(observed.clone(), Vec::new()),
        Err(RietveldError::EmptyPhases)
    ));
    assert!(matches!(
        input_result(observed, vec![phase("alpha", 1.0), phase("alpha", 2.0)],),
        Err(RietveldError::DuplicatePhaseId)
    ));
    assert!(matches!(
        RietveldCalculationOptions::new(0.0, true, ExecutionPolicy::new(Some(1), 2).unwrap()),
        Err(RietveldError::InvalidOptions)
    ));
}

#[test]
fn calculation_revalidates_public_adapter_state() {
    let pattern =
        PatternRecord::new(vec![20.0, 21.0], Some(vec![0.0; 2]), None, None, None).unwrap();
    let mut request = input(pattern.clone(), vec![phase("alpha", 1.0)]);
    request.pattern.x_deg[1] = 19.0;
    assert!(matches!(
        calculate_rietveld_pattern(&request, &options(1)),
        Err(RietveldError::Pattern(_))
    ));

    let mut request = input(pattern.clone(), vec![phase("alpha", 1.0)]);
    request.instrument.wavelength_angstrom = f64::NAN;
    assert!(matches!(
        calculate_rietveld_pattern(&request, &options(1)),
        Err(RietveldError::InvalidInstrument)
    ));

    let mut request = input(pattern.clone(), vec![phase("alpha", 1.0)]);
    request.axial_geometry = Some(FcjGeometry {
        sample_over_radius: -0.01,
        detector_over_radius: 0.01,
    });
    assert!(matches!(
        calculate_rietveld_pattern(&request, &options(1)),
        Err(RietveldError::InvalidAxialGeometry)
    ));

    let mut request = input(pattern.clone(), vec![phase("alpha", 1.0)]);
    request.position_correction.bragg_brentano_mm = Some((0.1, 0.0));
    assert!(matches!(
        calculate_rietveld_pattern(&request, &options(1)),
        Err(RietveldError::InvalidPositionCorrection)
    ));

    let mut request = input(pattern, vec![phase("alpha", 1.0)]);
    request.phases.clear();
    assert!(matches!(
        calculate_rietveld_pattern(&request, &options(1)),
        Err(RietveldError::EmptyPhases)
    ));

    let pattern =
        PatternRecord::new(vec![20.0, 21.0], Some(vec![0.0; 2]), None, None, None).unwrap();
    let request = input(pattern, vec![phase("alpha", 1.0)]);
    let mut invalid_options = options(1);
    invalid_options.support_fwhm = f64::INFINITY;
    assert!(matches!(
        calculate_rietveld_pattern(&request, &invalid_options),
        Err(RietveldError::InvalidOptions)
    ));
}

#[test]
fn residual_contract_honors_mask_and_uncertainty() {
    let x_deg = (0..4_001)
        .map(|index| 10.0 + f64::from(index) * 0.02)
        .collect::<Vec<_>>();
    let seed_pattern = PatternRecord::new(
        x_deg.clone(),
        Some(vec![0.0; x_deg.len()]),
        Some(vec![0.5; x_deg.len()]),
        None,
        None,
    )
    .unwrap();
    let seed =
        calculate_rietveld_pattern(&input(seed_pattern, vec![phase("alpha", 1.0)]), &options(1))
            .unwrap();
    let mut observed = seed.y.clone();
    observed[100] += 5.0;
    observed[200] += 2.0;
    let mut mask = vec![true; x_deg.len()];
    mask[100] = false;
    let pattern = PatternRecord::new(
        x_deg,
        Some(observed),
        Some(vec![0.5; seed.y.len()]),
        Some(mask),
        None,
    )
    .unwrap();
    let calculated =
        calculate_rietveld_pattern(&input(pattern, vec![phase("alpha", 1.0)]), &options(1))
            .unwrap();

    assert!(!calculated.metrics.included[100]);
    assert_eq!(
        calculated.metrics.residual[100].to_bits(),
        (-5.0_f64).to_bits()
    );
    assert_eq!(
        calculated.metrics.residual[200].to_bits(),
        (-2.0_f64).to_bits()
    );
    assert_eq!(
        calculated.metrics.weighted_residual[200].to_bits(),
        (-4.0_f64).to_bits()
    );
    assert_eq!(calculated.metrics.chi_square.to_bits(), 16.0_f64.to_bits());
    assert_eq!(
        calculated.metrics.reduced_chi_square.to_bits(),
        (16.0_f64 / 4_000.0).to_bits()
    );
}

#[test]
fn structural_intermediates_position_correction_and_contributions_are_exposed() {
    let x_deg = (0..4_001)
        .map(|index| 10.0 + f64::from(index) * 0.02)
        .collect::<Vec<_>>();
    let pattern = PatternRecord::new(
        x_deg.clone(),
        Some(vec![0.0; x_deg.len()]),
        None,
        None,
        None,
    )
    .unwrap();
    let definition = phase_definition(1.0);
    let neutral_phase = RietveldPhase::new(
        RecordId::new("neutral").unwrap(),
        "Neutral",
        definition.clone(),
        OwnedCwContributions::neutral(definition.hkl.len()),
    )
    .unwrap();
    let weighted_phase = RietveldPhase::new(
        RecordId::new("weighted").unwrap(),
        "Weighted",
        definition.clone(),
        OwnedCwContributions::new(
            definition.hkl.len(),
            0,
            OwnedCwContributionArrays {
                gaussian_variance_deg2: vec![0.0; definition.hkl.len()],
                lorentzian_fwhm_deg: vec![0.0; definition.hkl.len()],
                intensity_multiplier: vec![2.0, 1.0, 1.0],
                d_gaussian_variance_d_position: vec![0.0; definition.hkl.len()],
                d_lorentzian_fwhm_d_position: vec![0.0; definition.hkl.len()],
                d_intensity_multiplier_d_position: vec![0.0; definition.hkl.len()],
                ..OwnedCwContributionArrays::default()
            },
        )
        .unwrap(),
    )
    .unwrap();
    let mut neutral_input = input(pattern.clone(), vec![neutral_phase]);
    neutral_input.position_correction.zero_shift_deg = 0.0;
    let neutral = calculate_rietveld_pattern(&neutral_input, &options(1)).unwrap();
    let shifted = calculate_rietveld_pattern(
        &input(pattern.clone(), vec![phase("shifted", 1.0)]),
        &options(1),
    )
    .unwrap();
    let weighted =
        calculate_rietveld_pattern(&input(pattern, vec![weighted_phase]), &options(1)).unwrap();

    let neutral_result = &neutral.phases[0].result;
    let shifted_result = &shifted.phases[0].result;
    assert_eq!(
        neutral_result.d_spacing_angstrom.len(),
        definition.hkl.len()
    );
    assert_eq!(neutral_result.two_theta_deg.len(), definition.hkl.len());
    assert_eq!(
        neutral_result.structure_factors.intensity.len(),
        definition.hkl.len()
    );
    for (unshifted, shifted) in neutral_result
        .two_theta_deg
        .iter()
        .zip(&shifted_result.two_theta_deg)
    {
        assert!((shifted - unshifted - 0.02).abs() < 1.0e-12);
    }
    assert_ne!(weighted.profile_y, neutral.profile_y);
    assert!(weighted.profile_y.iter().sum::<f64>() > neutral.profile_y.iter().sum::<f64>());
}

#[test]
#[ignore = "requires an installed NumPy Python interpreter"]
fn calculation_matches_python_structural_workflow_when_configured() {
    let Ok(python) = std::env::var("PHASESMITH_NUMPY_PYTHON") else {
        return;
    };
    let x_deg = (0..3_001)
        .map(|index| 10.0 + f64::from(index) * 0.03)
        .collect::<Vec<_>>();
    let background = x_deg
        .iter()
        .map(|value| 0.1 + value / 900.0)
        .collect::<Vec<_>>();
    let pattern = PatternRecord::new(
        x_deg.clone(),
        Some(vec![0.0; x_deg.len()]),
        None,
        None,
        Some(background),
    )
    .unwrap();
    let native =
        calculate_rietveld_pattern(&input(pattern, vec![phase("alpha", 1.4)]), &options(1))
            .unwrap();

    let script = r#"
from dataclasses import replace
import numpy as np
import phasesmith

group = phasesmith.SpaceGroup([
    phasesmith.SymmetryOperation.identity(),
    phasesmith.SymmetryOperation([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], (0, 0, 0)),
])
structure = phasesmith.CrystalStructure(
    "structure", "P -1", phasesmith.UnitCell(4.7, 5.1, 6.2, 82.0, 87.0, 74.0), group,
    (
        phasesmith.AtomSite("si", "Si1", "Si", "Si", (0.17, 0.23, 0.31), 0.82, 0.012),
        phasesmith.AtomSite("o", "O1", "O", "O", (0.37, 0.11, 0.19), 0.55, 0.018),
    ),
)
reflections = phasesmith.StructuralReflectionBatch(
    ("1,0,1", "2,1,1", "1,2,3"),
    [[1, 0, 1], [2, 1, 1], [1, 2, 3]],
    [2, 4, 2],
)
phase = phasesmith.RietveldPhase(
    "alpha", "Phase alpha", structure, reflections, phasesmith.XrayNonResonant(),
    phasesmith.NeutralIntegratedIntensityCorrection(), 1.4,
)
instrument = phasesmith.ConstantWavelengthInstrument(1.5406, 2e-4, -1e-4, 1.2e-4, 1.5e-3, 3e-3)
experiment = replace(phasesmith.ConstantWavelengthExperiment.x_ray(instrument), zero_shift_deg=0.02)
x = 10.0 + np.arange(3001, dtype=np.float64) * 0.03
pattern = phasesmith.PowderPattern(x, observed_y=np.zeros_like(x), background=0.1 + x / 900.0)
result = phasesmith.calculate_structural_pattern(pattern, experiment, phase)
for values in (
    result.profile_y,
    result.y,
    result.reflections.d_spacing_angstrom,
    result.reflections.two_theta_deg,
    result.reflections.integrated_intensity,
):
    print(" ".join(format(float(value), ".17g") for value in values))
"#;
    let output = Command::new(python)
        .args(["-c", script])
        .env_remove("PYTHONPATH")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Python structural oracle failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines().map(parse_f64_line);
    assert_close_slice(&native.profile_y, &lines.next().unwrap(), 5.0e-10);
    assert_close_slice(&native.y, &lines.next().unwrap(), 5.0e-10);
    assert_close_slice(
        &native.phases[0].result.d_spacing_angstrom,
        &lines.next().unwrap(),
        2.0e-14,
    );
    assert_close_slice(
        &native.phases[0].result.two_theta_deg,
        &lines.next().unwrap(),
        2.0e-13,
    );
    assert_close_slice(
        &native.phases[0].result.structure_factors.intensity,
        &lines.next().unwrap(),
        5.0e-10,
    );
    assert!(lines.next().is_none());
}

fn parse_f64_line(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .map(|value| value.parse::<f64>().unwrap())
        .collect()
}

fn assert_close_slice(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "sample {index}: native={actual:.17e}, Python={expected:.17e}, tolerance={tolerance:.3e}"
        );
    }
}

fn input_result(
    pattern: PatternRecord,
    phases: Vec<RietveldPhase>,
) -> Result<RietveldInput, phasesmith_workflows::RietveldError> {
    RietveldInput::new(
        pattern,
        instrument(),
        None,
        MonochromaticPositionCorrection {
            zero_shift_deg: 0.0,
            bragg_brentano_mm: None,
            debye_scherrer_micrometre: None,
        },
        phases,
    )
}
