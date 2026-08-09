//! Project-level ownership and cross-record validation for native Rietveld state.

use std::collections::BTreeMap;

use phasesmith_core::{ConstantWavelengthInstrument, OwnedCwContributions};
use phasesmith_crystallography::{IntegratedIntensityCorrectionModel, UnitCell};
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPhaseDefinition,
};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_io::space_group_by_number;
use phasesmith_model::{
    ExperimentRecord, FixedWavelengthSpectrum, HistogramRecord, PatternRecord, ProjectRecord,
    ProviderRequirement, RadiationDefinition, RadiationProbe, RecordId, StructuralPhaseRecord,
};
use phasesmith_workflows::{
    RefinementLimits, RietveldAnalysis, RietveldCalculationOptions, RietveldCovarianceOptions,
    RietveldInput, RietveldParameterSelection, RietveldPhase, RietveldProjectError,
    RietveldProjectState, RietveldRefinementOptions, refine_general_rietveld,
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

fn phase() -> RietveldPhase {
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
        hkl: vec![[1, 0, 0]],
        multiplicity: vec![6],
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
    RietveldPhase::new_with_site_ids(
        RecordId::new("alpha").unwrap(),
        "Alpha",
        vec![RecordId::new("Si1").unwrap()],
        definition,
        OwnedCwContributions::neutral(1),
    )
    .unwrap()
}

fn options() -> RietveldRefinementOptions {
    RietveldRefinementOptions::new(
        RietveldCalculationOptions::new(20.0, true, ExecutionPolicy::new(Some(1), 1).unwrap())
            .unwrap(),
        RefinementLimits::new(4, 100, None, 10).unwrap(),
        1,
        1.0e-8,
        1.0e-7,
        1.0e-6,
        10.0,
        0.3,
        1.0e-6,
        20,
        0.25,
        4,
    )
    .unwrap()
}

fn state() -> RietveldProjectState {
    let phase = phase();
    let x_deg = (0..101)
        .map(|index| 20.0 + f64::from(index) * 0.1)
        .collect::<Vec<_>>();
    let pattern = PatternRecord::new(
        x_deg.clone(),
        Some(vec![0.0; x_deg.len()]),
        None,
        None,
        None,
    )
    .unwrap();
    let position = MonochromaticPositionCorrection {
        zero_shift_deg: 0.0,
        bragg_brentano_mm: None,
        debye_scherrer_micrometre: None,
    };
    let input = RietveldInput::new(
        pattern.clone(),
        instrument(),
        None,
        position,
        vec![phase.clone()],
    )
    .unwrap();
    let selection = RietveldParameterSelection::default();
    let native_options = options();
    let solved = refine_general_rietveld(
        &input,
        &selection,
        &[None],
        &[],
        &native_options,
        RietveldCovarianceOptions::default(),
        None,
        None,
    )
    .unwrap();
    RietveldProjectState {
        project: ProjectRecord {
            project_id: RecordId::new("project").unwrap(),
            revision: 3,
            name: "Project".to_owned(),
            histograms: vec![HistogramRecord {
                histogram_id: RecordId::new("histogram").unwrap(),
                name: "Histogram".to_owned(),
                pattern,
                experiment: ExperimentRecord::new(
                    instrument(),
                    RadiationDefinition::Monochromatic {
                        probe: RadiationProbe::Xray,
                        wavelength_angstrom: instrument().wavelength_angstrom,
                    },
                    None,
                    position,
                )
                .unwrap(),
                phase_ids: vec![phase.phase_id().clone()],
            }],
            phases: vec![StructuralPhaseRecord {
                phase_id: phase.phase_id().clone(),
                name: phase.name().to_owned(),
                definition: phase.definition().clone(),
                required_providers: Vec::new(),
            }],
            metadata: BTreeMap::new(),
        },
        analyses: vec![RietveldAnalysis {
            histogram_id: RecordId::new("histogram").unwrap(),
            input,
            selection,
            lattice_bounds: vec![None],
            constraints: Vec::new(),
            options: native_options,
            covariance: RietveldCovarianceOptions::default(),
            checkpoint: Some(solved.checkpoint),
        }],
    }
}

#[test]
fn complete_project_analysis_and_checkpoint_validate_together() {
    state().validate().unwrap();
}

#[test]
fn cross_record_mismatches_are_rejected_before_adapter_use() {
    let mut value = state();
    value.analyses[0].options.limits = RefinementLimits::new(1, 100, None, 10).unwrap();
    value.analyses[0]
        .checkpoint
        .as_mut()
        .unwrap()
        .completed_iterations = 2;
    assert!(matches!(
        value.validate(),
        Err(RietveldProjectError::CheckpointExceedsIterationLimit)
    ));

    let mut value = state();
    value.analyses.push(value.analyses[0].clone());
    assert!(matches!(
        value.validate(),
        Err(RietveldProjectError::DuplicateAnalysis { .. })
    ));

    let mut value = state();
    value.analyses[0].histogram_id = RecordId::new("missing").unwrap();
    assert!(matches!(
        value.validate(),
        Err(RietveldProjectError::UnknownHistogram { .. })
    ));

    let mut value = state();
    value.project.histograms[0].pattern.background_y[0] = 1.0;
    assert!(matches!(
        value.validate(),
        Err(RietveldProjectError::HistogramStateMismatch { .. })
    ));

    let mut value = state();
    value.project.histograms[0].experiment.radiation = RadiationDefinition::FixedSpectrum {
        probe: RadiationProbe::Xray,
        spectrum: FixedWavelengthSpectrum::new(vec![1.5406, 1.5444], vec![1.0, 0.5]).unwrap(),
    };
    assert!(matches!(
        value.validate(),
        Err(RietveldProjectError::UnsupportedRadiation { .. })
    ));

    let mut value = state();
    value.project.phases[0].required_providers =
        vec![ProviderRequirement::new("custom.scattering", "1").unwrap()];
    assert!(matches!(
        value.validate(),
        Err(RietveldProjectError::ExternalProviderRequired { .. })
    ));
}
