//! Guarded native structural reflection-topology contracts.

use std::collections::BTreeMap;

use phasesmith_core::{
    ConstantWavelengthInstrument, OwnedCwContributionArrays, OwnedCwContributions,
};
use phasesmith_crystallography::{IntegratedIntensityCorrectionModel, UnitCell};
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPhaseDefinition,
};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_io::space_group_by_number;
use phasesmith_model::{PatternRecord, RecordId};
use phasesmith_workflows::{
    LatticeBounds, LatticeParameterization, LatticeReflectionDomain, ParameterSet,
    RefinementLimits, RietveldCalculationOptions, RietveldCheckpoint, RietveldInput, RietveldPhase,
    RietveldRefinementOptions, RietveldSamplePhysicsModel, RietveldStructuralLayout,
    RietveldStructuralSelection, TerminationReason, calculate_rietveld_pattern, refine_rietveld,
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

fn reference_cell() -> UnitCell {
    UnitCell {
        a_angstrom: 4.0,
        b_angstrom: 4.0,
        c_angstrom: 6.0,
        alpha_deg: 90.0,
        beta_deg: 90.0,
        gamma_deg: 90.0,
    }
}

fn domain(guard_scale: f64) -> (LatticeReflectionDomain, LatticeBounds) {
    let group = space_group_by_number(123).unwrap().space_group;
    let parameterization = LatticeParameterization::new(group, reference_cell()).unwrap();
    let bounds = LatticeBounds::around(&parameterization, 0.2, 5.0).unwrap();
    (
        LatticeReflectionDomain::new(
            parameterization,
            bounds.clone(),
            instrument().wavelength_angstrom,
            [20.0, 90.0],
            1.0,
            true,
            50_000_000,
            guard_scale,
        )
        .unwrap(),
        bounds,
    )
}

fn phase(cell: UnitCell, reflection_domain: LatticeReflectionDomain) -> RietveldPhase {
    let definition = StructuralPhaseDefinition {
        cell,
        space_group: reflection_domain.parameterization().space_group().clone(),
        hkl: Vec::new(),
        multiplicity: Vec::new(),
        fractional_xyz: vec![[0.0, 0.0, 0.0]],
        occupancy: vec![1.0],
        u_iso_angstrom2: vec![0.01],
        anisotropic_mask: vec![false],
        u_aniso_cif_angstrom2: vec![[0.0; 6]],
        scattering_species: vec!["Si".to_owned()],
        scattering_real_offset: Vec::new(),
        scattering_imag_offset: Vec::new(),
        scale: 1.0,
        coordinate_tolerance: 1.0e-10,
        scattering_model: BuiltInScatteringModel::XrayNonResonant,
        correction_model: IntegratedIntensityCorrectionModel::Neutral,
    };
    RietveldPhase::from_lattice_domain(
        RecordId::new("alpha").unwrap(),
        "Alpha",
        vec![RecordId::new("Si1").unwrap()],
        definition,
        reflection_domain,
    )
    .unwrap()
}

fn changed_phase(starting: &RietveldPhase, bounds: &LatticeBounds) -> RietveldPhase {
    let domain = starting.reflection_domain().unwrap();
    bounds
        .corner_values()
        .into_iter()
        .map(|values| domain.parameterization().to_cell(&values).unwrap())
        .map(|cell| starting.regenerate_lattice_at_cell(cell).unwrap().0)
        .find(|candidate| candidate.reflection_ids() != starting.reflection_ids())
        .expect("wide anisotropic bounds must exercise a guarded topology change")
}

#[test]
fn structural_value_installation_regenerates_dynamic_topology() {
    let (reflection_domain, bounds) = domain(1.001);
    let starting = phase(reference_cell(), reflection_domain);
    let changed = changed_phase(&starting, &bounds);
    let layout = RietveldStructuralLayout::new(
        std::slice::from_ref(&starting),
        RietveldStructuralSelection {
            lattice: true,
            ..RietveldStructuralSelection::default()
        },
        &[Some(bounds)],
    )
    .unwrap();
    let target_values = starting
        .reflection_domain()
        .unwrap()
        .parameterization()
        .values_from_cell(changed.definition().cell)
        .unwrap();
    let installed = layout
        .apply_values(std::slice::from_ref(&starting), &target_values)
        .unwrap();
    assert_eq!(installed, vec![changed]);
    assert_ne!(installed[0].reflection_ids(), starting.reflection_ids());
}

#[test]
fn regeneration_transfers_all_sample_physics_arrays_by_stable_id() {
    let (reflection_domain, bounds) = domain(1.001);
    let initial = phase(reference_cell(), reflection_domain);
    let count = initial.reflection_ids().len();
    let indexed = |index: usize| f64::from(u32::try_from(index).unwrap());
    let contributions = OwnedCwContributions::new(
        count,
        1,
        OwnedCwContributionArrays {
            gaussian_variance_deg2: (0..count).map(|index| 0.001 * indexed(index)).collect(),
            lorentzian_fwhm_deg: (0..count).map(|index| 0.002 * indexed(index)).collect(),
            intensity_multiplier: (0..count)
                .map(|index| 1.0 + 0.01 * indexed(index))
                .collect(),
            d_gaussian_variance_d_position: vec![0.1; count],
            d_lorentzian_fwhm_d_position: vec![0.2; count],
            d_intensity_multiplier_d_position: vec![0.3; count],
            d_gaussian_variance_d_parameters: vec![0.4; count],
            d_lorentzian_fwhm_d_parameters: vec![0.5; count],
            d_intensity_multiplier_d_parameters: vec![0.6; count],
        },
    )
    .unwrap();
    let initial = initial.with_contributions(contributions).unwrap();
    let moved = changed_phase(&initial, &bounds);
    let old = initial
        .reflection_ids()
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let arrays = moved.contributions().arrays();
    let mut saw_added = false;
    for (new_index, id) in moved.reflection_ids().iter().enumerate() {
        if let Some(old_index) = old.get(id.as_str()) {
            assert_eq!(
                arrays.intensity_multiplier[new_index].to_bits(),
                (1.0 + 0.01 * indexed(*old_index)).to_bits()
            );
            assert_eq!(
                arrays.d_intensity_multiplier_d_parameters[new_index].to_bits(),
                0.6_f64.to_bits()
            );
        } else {
            saw_added = true;
            assert_eq!(
                arrays.gaussian_variance_deg2[new_index].to_bits(),
                0.0_f64.to_bits()
            );
            assert_eq!(
                arrays.lorentzian_fwhm_deg[new_index].to_bits(),
                0.0_f64.to_bits()
            );
            assert_eq!(
                arrays.intensity_multiplier[new_index].to_bits(),
                1.0_f64.to_bits()
            );
            assert_eq!(
                arrays.d_intensity_multiplier_d_parameters[new_index].to_bits(),
                0.0_f64.to_bits()
            );
        }
    }
    assert!(saw_added || moved.reflection_ids().len() < initial.reflection_ids().len());
}

#[test]
fn regeneration_re_evaluates_attached_models_for_the_new_reflection_batch() {
    let (reflection_domain, bounds) = domain(1.001);
    let initial = phase(reference_cell(), reflection_domain).with_sample_physics(
        RietveldSamplePhysicsModel::IsotropicSize {
            crystallite_size_nm: 65.0,
            shape_factor: 0.9,
        },
    );
    let moved = changed_phase(&initial, &bounds);
    let reflection_count = moved.reflection_ids().len();
    let x_deg = (0..1_401)
        .map(|index| 15.0 + f64::from(index) * 0.06)
        .collect::<Vec<_>>();
    let input = RietveldInput::new(
        PatternRecord::new(
            x_deg.clone(),
            Some(vec![0.0; x_deg.len()]),
            None,
            None,
            Some(vec![0.0; x_deg.len()]),
        )
        .unwrap(),
        instrument(),
        None,
        MonochromaticPositionCorrection {
            zero_shift_deg: 0.0,
            bragg_brentano_mm: None,
            debye_scherrer_micrometre: None,
        },
        vec![moved],
    )
    .unwrap();
    let calculation = calculate_rietveld_pattern(
        &input,
        &RietveldCalculationOptions::new(20.0, false, ExecutionPolicy::new(Some(1), 1).unwrap())
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        calculation.phases[0].result.two_theta_deg.len(),
        reflection_count
    );
    assert_eq!(
        calculation.phases[0]
            .result
            .accumulation
            .derivatives
            .global
            .as_ref()
            .unwrap()
            .parameter_count,
        8
    );
}

#[test]
fn checkpoint_accepts_changed_topology_only_for_identical_domain_contract() {
    let (reflection_domain, bounds) = domain(1.001);
    let starting = phase(reference_cell(), reflection_domain);
    let changed = changed_phase(&starting, &bounds);
    let x_deg = (0..1_001)
        .map(|index| 20.0 + f64::from(index) * 0.07)
        .collect::<Vec<_>>();
    let pattern = PatternRecord::new(
        x_deg.clone(),
        Some(vec![0.0; x_deg.len()]),
        None,
        None,
        Some(vec![0.0; x_deg.len()]),
    )
    .unwrap();
    let input = RietveldInput::new(
        pattern,
        instrument(),
        None,
        MonochromaticPositionCorrection {
            zero_shift_deg: 0.0,
            bragg_brentano_mm: None,
            debye_scherrer_micrometre: None,
        },
        vec![starting],
    )
    .unwrap();
    let calculation =
        RietveldCalculationOptions::new(20.0, false, ExecutionPolicy::new(Some(1), 1).unwrap())
            .unwrap();
    let options = RietveldRefinementOptions::new(
        calculation,
        RefinementLimits::new(1, 20, None, 2).unwrap(),
        1,
        1.0e-12,
        1.0e-10,
        1.0e-6,
        10.0,
        0.3,
        1.0e-10,
        2,
        1.0,
        0,
    )
    .unwrap();
    let checkpoint = RietveldCheckpoint {
        completed_iterations: 0,
        phases: vec![changed.clone()],
        parameters: ParameterSet::new(Vec::new()).unwrap(),
        objective: 0.0,
        damping: options.initial_damping,
        history: Vec::new(),
    };
    let result = refine_rietveld(
        &input,
        RietveldStructuralSelection::default(),
        &[None],
        &options,
        Some(&checkpoint),
        None,
    )
    .unwrap();
    assert_eq!(result.termination_reason, TerminationReason::Converged);
    assert_eq!(result.phases, vec![changed]);

    let (incompatible_domain, _) = domain(1.01);
    let incompatible = phase(result.phases[0].definition().cell, incompatible_domain);
    let incompatible_checkpoint = RietveldCheckpoint {
        phases: vec![incompatible],
        ..checkpoint
    };
    let error = refine_rietveld(
        &input,
        RietveldStructuralSelection::default(),
        &[None],
        &options,
        Some(&incompatible_checkpoint),
        None,
    )
    .unwrap_err();
    assert!(error.to_string().contains("checkpoint"));
}
