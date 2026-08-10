//! Staged effective-profile estimation from a predominantly single-phase pattern.

use phasesmith_core::ConstantWavelengthInstrument;
use phasesmith_crystallography::UnitCell;
use phasesmith_execution::ExecutionPolicy;
use phasesmith_io::space_group_by_number;
use phasesmith_model::PatternRecord;
use phasesmith_workflows::{
    LatticeBounds, LatticeParameterization, LatticeReflectionDomain, LeBailOptions, LeBailPhase,
    ProfileEstimationInput, ProfileEstimationMode, ProfileEstimationOptions,
    ProfileEstimationStageKind, calculate_lebail_pattern, estimate_effective_profile,
    starting_profile_from_fwhm,
};

fn execution() -> ExecutionPolicy {
    ExecutionPolicy::new(Some(1), 2).unwrap()
}

fn truth_instrument() -> ConstantWavelengthInstrument {
    ConstantWavelengthInstrument {
        wavelength_angstrom: 0.4133,
        u_deg2: 0.0,
        v_deg2: 0.0,
        w_deg2: 2.5e-4,
        x_deg: 0.0,
        y_deg: 0.0,
    }
}

fn options(mode: ProfileEstimationMode) -> ProfileEstimationOptions {
    let lebail = LeBailOptions::new(
        40,
        2,
        1.0e-7,
        1.0e-10,
        1.0,
        1.0e-15,
        1.0e-12,
        false,
        1.0 - 1.0e-10,
        false,
        20.0,
        execution(),
    )
    .unwrap()
    .with_profile_controls(1.0e-10, 1.0, 10)
    .unwrap();
    ProfileEstimationOptions::new(mode, false, 0.002, 1.0e-5, 0.98, lebail).unwrap()
}

#[allow(clippy::cast_precision_loss)]
fn linspace(start: f64, endpoint: f64, count: usize) -> Vec<f64> {
    let spacing = (endpoint - start) / (count - 1) as f64;
    (0..count)
        .map(|index| start + index as f64 * spacing)
        .collect()
}

fn phase(instrument: ConstantWavelengthInstrument) -> LeBailPhase {
    let positions = [8.0, 12.0, 17.0, 23.0, 30.0, 38.0];
    LeBailPhase::new(
        "dominant",
        "Dominant phase",
        (0..positions.len())
            .map(|index| format!("r{index}"))
            .collect(),
        (0..positions.len())
            .map(|index| [i32::try_from(index + 1).unwrap(), 1, 0])
            .collect(),
        positions
            .iter()
            .map(|position| {
                instrument.wavelength_angstrom / (2.0 * (0.5_f64 * position).to_radians().sin())
            })
            .collect(),
        positions.to_vec(),
        vec![12.0, 7.0, 5.0, 9.0, 4.0, 6.0],
        1.0,
        Vec::new(),
    )
    .unwrap()
}

fn observed(mask_contaminant: bool) -> (PatternRecord, LeBailPhase) {
    let instrument = truth_instrument();
    let phase = phase(instrument);
    let x = linspace(5.0, 42.0, 7_401);
    let blank = PatternRecord::new(x.clone(), None, None, None, Some(vec![0.1; x.len()])).unwrap();
    let mut y = calculate_lebail_pattern(
        &blank,
        instrument,
        std::slice::from_ref(&phase),
        20.0,
        &execution(),
    )
    .unwrap()
    .y;
    let mask = mask_contaminant.then(|| {
        x.iter()
            .map(|angle| (*angle - 34.0).abs() >= 0.08)
            .collect::<Vec<_>>()
    });
    if mask_contaminant {
        for (angle, value) in x.iter().zip(&mut y) {
            if (*angle - 34.0).abs() < 0.05 {
                *value += 20.0 * (1.0 - (*angle - 34.0).abs() / 0.05);
            }
        }
    }
    (
        PatternRecord::new(x, Some(y), None, mask, Some(blank.background_y)).unwrap(),
        phase,
    )
}

#[test]
fn w_only_recovers_width_and_preserves_poni_wavelength_bitwise() {
    let (pattern, phase) = observed(false);
    let wavelength = truth_instrument().wavelength_angstrom;
    let starting = starting_profile_from_fwhm(wavelength, 0.06).unwrap();
    let input = ProfileEstimationInput::new(pattern, starting, phase).unwrap();
    let result =
        estimate_effective_profile(&input, &options(ProfileEstimationMode::WOnly)).unwrap();

    assert_eq!(
        result.instrument.wavelength_angstrom.to_bits(),
        wavelength.to_bits()
    );
    assert_eq!(result.active_parameters, ["w_deg2"]);
    assert_eq!(result.stages.len(), 1);
    assert_eq!(result.stages[0].kind, ProfileEstimationStageKind::W);
    assert!(result.stages[0].accepted);
    assert!((result.instrument.w_deg2 - truth_instrument().w_deg2).abs() < 3.0e-8);
    assert!(result.lebail.metrics.rwp < 3.0e-5);
    assert!(
        result
            .warnings
            .iter()
            .any(|warning| warning.contains("sample broadening"))
    );
}

#[test]
fn automatic_mode_keeps_the_simpler_model_when_extra_terms_are_unsupported() {
    let (pattern, phase) = observed(false);
    let starting =
        starting_profile_from_fwhm(truth_instrument().wavelength_angstrom, 0.06).unwrap();
    let input = ProfileEstimationInput::new(pattern, starting, phase).unwrap();
    let result =
        estimate_effective_profile(&input, &options(ProfileEstimationMode::Automatic)).unwrap();

    assert_eq!(result.active_parameters, ["w_deg2"]);
    assert_eq!(result.stages.len(), 2);
    assert_eq!(result.stages[1].kind, ProfileEstimationStageKind::Uvw);
    assert!(!result.stages[1].accepted);
    assert!(result.stages[1].decision.starts_with("rejected:"));
}

#[test]
fn pattern_mask_excludes_an_unmodelled_contaminant_peak() {
    let (pattern, phase) = observed(true);
    let starting =
        starting_profile_from_fwhm(truth_instrument().wavelength_angstrom, 0.06).unwrap();
    let input = ProfileEstimationInput::new(pattern, starting, phase).unwrap();
    let result =
        estimate_effective_profile(&input, &options(ProfileEstimationMode::WOnly)).unwrap();

    assert!((result.instrument.w_deg2 - truth_instrument().w_deg2).abs() < 3.0e-8);
    assert!(result.lebail.metrics.rwp < 3.0e-5);
}

#[test]
fn invalid_starting_fwhm_and_fixed_phase_alignment_are_rejected() {
    assert!(starting_profile_from_fwhm(0.4133, 0.0).is_err());
    let (pattern, phase) = observed(false);
    let starting =
        starting_profile_from_fwhm(truth_instrument().wavelength_angstrom, 0.06).unwrap();
    let input = ProfileEstimationInput::new(pattern, starting, phase).unwrap();
    let mut selected = options(ProfileEstimationMode::WOnly);
    selected.align_lattice = true;
    assert!(estimate_effective_profile(&input, &selected).is_err());
}

#[test]
fn optional_nuisance_lattice_alignment_precedes_width_estimation() {
    let starting_cell = UnitCell {
        a_angstrom: 3.995,
        b_angstrom: 3.995,
        c_angstrom: 6.0,
        alpha_deg: 90.0,
        beta_deg: 90.0,
        gamma_deg: 90.0,
    };
    let true_cell = UnitCell {
        a_angstrom: 4.0,
        b_angstrom: 4.0,
        ..starting_cell
    };
    let group = space_group_by_number(123).unwrap().space_group;
    let parameterization = LatticeParameterization::new(group, starting_cell).unwrap();
    let bounds = LatticeBounds::around(&parameterization, 0.04, 5.0).unwrap();
    let domain = LatticeReflectionDomain::new(
        parameterization,
        bounds,
        truth_instrument().wavelength_angstrom,
        [5.0, 42.0],
        1.0,
        true,
        50_000_000,
        1.001,
    )
    .unwrap();
    let starting =
        LeBailPhase::from_lattice_domain("dominant", "Dominant phase", starting_cell, 1.0, domain)
            .unwrap();
    let intensities = starting
        .hkl()
        .iter()
        .map(|hkl| 2.0 + f64::from(hkl[0].unsigned_abs() % 7))
        .collect::<Vec<_>>();
    let starting = starting.with_integrated_intensities(&intensities).unwrap();
    let truth = starting.regenerate_lattice_at_cell(true_cell).unwrap();
    let x = linspace(5.0, 42.0, 7_401);
    let blank = PatternRecord::new(x.clone(), None, None, None, None).unwrap();
    let observed = calculate_lebail_pattern(
        &blank,
        truth_instrument(),
        std::slice::from_ref(&truth),
        20.0,
        &execution(),
    )
    .unwrap();
    let pattern = PatternRecord::new(x, Some(observed.y), None, None, None).unwrap();
    let input = ProfileEstimationInput::new(pattern, truth_instrument(), starting).unwrap();
    let mut selected = options(ProfileEstimationMode::WOnly);
    selected.align_lattice = true;
    selected.lebail = selected
        .lebail
        .with_profile_controls(1.0e-10, 0.05, 10)
        .unwrap();
    let result = estimate_effective_profile(&input, &selected).unwrap();

    assert_eq!(
        result.stages[0].kind,
        ProfileEstimationStageKind::LatticeAlignment
    );
    assert!(result.stages[0].accepted);
    let refined = result.lebail.phases[0].cell().unwrap();
    assert!((refined.a_angstrom - true_cell.a_angstrom).abs() < 2.0e-6);
    assert_eq!(
        result.instrument.wavelength_angstrom.to_bits(),
        truth_instrument().wavelength_angstrom.to_bits()
    );
}
