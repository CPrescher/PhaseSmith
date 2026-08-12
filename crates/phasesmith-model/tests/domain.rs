//! Integration coverage for application-neutral project records.

use std::collections::BTreeMap;

use phasesmith_engine::crystallography::{
    IntegratedIntensityCorrectionModel, SpaceGroup, SymmetryOperation, UnitCell,
};
use phasesmith_engine::profile::{ConstantWavelengthInstrument, TofInstrument};
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPhaseDefinition,
};
use phasesmith_model::{
    CapabilityReason, DomainError, ExperimentRecord, FixedWavelengthSpectrum, HistogramRecord,
    HostCapabilities, PatternRecord, ProjectRecord, ProviderRequirement, RadiationDefinition,
    RadiationProbe, RecordId, StructuralPhaseRecord, TofExperimentRecord, TofHistogramRecord,
    TofPatternRecord,
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

fn tof_instrument() -> TofInstrument {
    TofInstrument {
        zero_us: -0.7,
        difc_us_per_angstrom: 5_000.0,
        difa_us_per_angstrom2: -1.5,
        difb_us_angstrom: 0.8,
        alpha_coefficient: 0.18,
        beta0_per_us: 0.04,
        beta1_angstrom4_per_us: 0.000_5,
        betaq_angstrom2_per_us: 0.001,
        sigma0_us2: 1.0,
        sigma1_us2_per_angstrom2: 12.0,
        sigma2_us2_per_angstrom4: 0.05,
        sigmaq_us2_per_angstrom: 0.2,
        x_us_per_angstrom: 0.3,
        y_us_per_angstrom2: 0.05,
        z_us: 0.4,
    }
}

fn tof_histogram(histogram_id: &str, phase_ids: &[&str]) -> TofHistogramRecord {
    TofHistogramRecord {
        histogram_id: id(histogram_id),
        name: histogram_id.to_owned(),
        pattern: TofPatternRecord::new(
            vec![3_000.0, 3_001.0],
            Some(vec![10.0, 11.0]),
            None,
            None,
            None,
        )
        .unwrap(),
        experiment: TofExperimentRecord::new(tof_instrument()).unwrap(),
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
        tof_histograms: Vec::new(),
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
        tof_histograms: Vec::new(),
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
        tof_histograms: Vec::new(),
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
fn tof_pattern_is_typed_in_microseconds_and_validates_aligned_arrays() {
    let pattern = TofPatternRecord::new(
        vec![1_000.0, 1_001.5],
        Some(vec![10.0, 12.0]),
        Some(vec![2.0, 3.0]),
        None,
        None,
    )
    .unwrap();
    assert_eq!(pattern.tof_us, [1_000.0, 1_001.5]);
    assert_eq!(pattern.sample_count(), 2);
    assert!(matches!(
        TofPatternRecord::new(vec![1.0, 1.0], None, None, None, None),
        Err(DomainError::UnorderedGrid)
    ));
    assert!(matches!(
        TofPatternRecord::new(vec![1.0], None, Some(vec![0.0]), None, None),
        Err(DomainError::NonPositiveArray {
            name: "uncertainty"
        })
    ));
}

#[test]
fn mixed_project_keeps_tof_coordinates_and_histogram_ids_explicit() {
    let project = ProjectRecord {
        project_id: id("mixed"),
        revision: 0,
        name: "Mixed coordinates".to_owned(),
        histograms: vec![histogram("cw-bank", &["alpha"])],
        tof_histograms: vec![tof_histogram("tof-bank", &["alpha"])],
        phases: vec![phase("alpha", Vec::new())],
        metadata: BTreeMap::new(),
    };
    project.validate().unwrap();

    let mut duplicate = project.clone();
    duplicate.tof_histograms[0].histogram_id = id("cw-bank");
    assert!(matches!(
        duplicate.validate(),
        Err(DomainError::DuplicateHistogramId { .. })
    ));
}

#[test]
fn project_validation_rechecks_directly_constructed_nested_records() {
    let mut project = ProjectRecord {
        project_id: id("project-1"),
        revision: 0,
        name: "Recursive validation".to_owned(),
        histograms: vec![histogram("pattern", &["alpha"])],
        tof_histograms: Vec::new(),
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
