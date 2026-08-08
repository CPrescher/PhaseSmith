//! Integration coverage for application-neutral project records.

use std::collections::BTreeMap;

use phasesmith_engine::crystallography::{
    IntegratedIntensityCorrectionModel, SpaceGroup, SymmetryOperation, UnitCell,
};
use phasesmith_engine::profile::ConstantWavelengthInstrument;
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPhaseDefinition,
};
use phasesmith_model::{
    CapabilityReason, DomainError, ExperimentRecord, FixedWavelengthSpectrum, HistogramRecord,
    HostCapabilities, PatternRecord, ProjectRecord, ProviderRequirement, RadiationDefinition,
    RadiationProbe, RecordId, StructuralPhaseRecord,
};

fn id(value: &str) -> RecordId {
    RecordId::new(value).expect("stable ID")
}

fn phase(phase_id: &str, requirements: Vec<ProviderRequirement>) -> StructuralPhaseRecord {
    StructuralPhaseRecord {
        phase_id: id(phase_id),
        name: phase_id.to_owned(),
        definition: StructuralPhaseDefinition {
            cell: UnitCell {
                a_angstrom: 5.0,
                b_angstrom: 5.0,
                c_angstrom: 5.0,
                alpha_deg: 90.0,
                beta_deg: 90.0,
                gamma_deg: 90.0,
            },
            space_group: SpaceGroup::new(vec![SymmetryOperation::identity()]).expect("P1"),
            hkl: vec![[1, 0, 0]],
            multiplicity: vec![2],
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
        },
        required_providers: requirements,
    }
}

fn experiment() -> ExperimentRecord {
    let wavelength = 1.5406;
    ExperimentRecord::new(
        ConstantWavelengthInstrument {
            wavelength_angstrom: wavelength,
            u_deg2: 0.0,
            v_deg2: 0.0,
            w_deg2: 0.01,
            x_deg: 0.0,
            y_deg: 0.0,
        },
        RadiationDefinition::FixedSpectrum {
            probe: RadiationProbe::Xray,
            spectrum: FixedWavelengthSpectrum::new(vec![wavelength, 1.54439], vec![1.0, 0.5])
                .expect("spectrum"),
        },
        None,
        MonochromaticPositionCorrection {
            zero_shift_deg: 0.0,
            bragg_brentano_mm: None,
            debye_scherrer_micrometre: None,
        },
    )
    .expect("experiment")
}

fn histogram(histogram_id: &str, phase_ids: &[&str]) -> HistogramRecord {
    HistogramRecord {
        histogram_id: id(histogram_id),
        name: histogram_id.to_owned(),
        pattern: PatternRecord::new(
            vec![10.0, 10.1, 10.2],
            Some(vec![2.0, 3.0, 4.0]),
            Some(vec![1.0; 3]),
            None,
            None,
        )
        .expect("pattern"),
        experiment: experiment(),
        phase_ids: phase_ids.iter().map(|value| id(value)).collect(),
    }
}

#[test]
fn project_is_multi_histogram_aware_and_validates_references() {
    let project = ProjectRecord {
        project_id: id("project-1"),
        revision: 7,
        name: "Two datasets".to_owned(),
        histograms: vec![
            histogram("bank-1", &["alpha"]),
            histogram("bank-2", &["beta"]),
        ],
        phases: vec![phase("alpha", Vec::new()), phase("beta", Vec::new())],
        metadata: BTreeMap::from([("sample".to_owned(), "reference".to_owned())]),
    };
    project.validate().expect("valid project");

    let mut invalid = project.clone();
    invalid.histograms[0].phase_ids.push(id("missing"));
    assert!(matches!(
        invalid.validate(),
        Err(DomainError::UnknownPhaseReference { .. })
    ));
}

#[test]
fn host_capability_diagnostics_are_explicit_and_stably_ordered() {
    let python_only = ProviderRequirement::new("example.texture", "2").expect("requirement");
    let native = ProviderRequirement::new("native.size", "1").expect("requirement");
    let project = ProjectRecord {
        project_id: id("project-1"),
        revision: 0,
        name: "Capabilities".to_owned(),
        histograms: vec![histogram("pattern", &["alpha"])],
        phases: vec![phase("alpha", vec![python_only.clone(), native.clone()])],
        metadata: BTreeMap::new(),
    };
    project.validate().expect("valid project");

    let diagnostics = project.capability_diagnostics(&HostCapabilities::new([native]));
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].phase_id, id("alpha"));
    assert_eq!(diagnostics[0].requirement, python_only);
    assert_eq!(diagnostics[0].reason, CapabilityReason::ProviderUnavailable);
}

#[test]
fn empty_project_is_valid_but_pattern_and_identity_errors_are_structured() {
    let empty = ProjectRecord {
        project_id: id("new-project"),
        revision: 0,
        name: "New project".to_owned(),
        histograms: Vec::new(),
        phases: Vec::new(),
        metadata: BTreeMap::new(),
    };
    empty.validate().expect("empty project");
    assert!(matches!(
        RecordId::new("invalid ID"),
        Err(DomainError::InvalidId { .. })
    ));
    assert!(matches!(
        PatternRecord::new(vec![1.0, 1.0], None, None, None, None),
        Err(DomainError::UnorderedGrid)
    ));
}

#[test]
fn project_validation_rechecks_directly_constructed_nested_records() {
    let mut project = ProjectRecord {
        project_id: id("project-1"),
        revision: 0,
        name: "Recursive validation".to_owned(),
        histograms: vec![histogram("pattern", &["alpha"])],
        phases: vec![phase("alpha", Vec::new())],
        metadata: BTreeMap::new(),
    };
    project.histograms[0].pattern.background_y.pop();
    assert!(matches!(
        project.validate(),
        Err(DomainError::ArrayLengthMismatch {
            name: "background_y"
        })
    ));

    project.histograms[0] = histogram("pattern", &["alpha"]);
    project.phases[0].definition.cell.a_angstrom = -1.0;
    assert!(matches!(
        project.validate(),
        Err(DomainError::StructuralPhase(
            phasesmith_engine::StructuralPatternError::InvalidCell(_)
        ))
    ));

    project.phases[0] = phase("alpha", Vec::new());
    project.phases[0].definition.scattering_species[0] = "not-an-element".to_owned();
    assert!(matches!(
        project.validate(),
        Err(DomainError::StructuralPhase(_))
    ));
}
