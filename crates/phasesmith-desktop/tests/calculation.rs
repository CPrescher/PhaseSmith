//! Standalone native desktop calculation and binary-retention tests.

use std::collections::BTreeMap;

use phasesmith_core::ConstantWavelengthInstrument;
use phasesmith_crystallography::{IntegratedIntensityCorrectionModel, UnitCell};
use phasesmith_desktop::{
    CalculationManager, CalculationOptionsInput, DesktopErrorCode, DesktopProjectStore,
    SeriesDtype, SeriesOwner,
};
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPhaseDefinition,
};
use phasesmith_io::space_group_by_number;
use phasesmith_model::{
    ExperimentRecord, HistogramRecord, PatternRecord, ProjectRecord, RadiationDefinition,
    RadiationProbe, RecordId, StructuralPhaseRecord,
};
use phasesmith_workflows::RietveldProjectState;

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

fn project_state() -> RietveldProjectState {
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
    let x_deg = (0..101)
        .map(|index| 20.0 + f64::from(index) * 0.1)
        .collect::<Vec<_>>();
    let pattern = PatternRecord::new(
        x_deg.clone(),
        Some(vec![10.0; x_deg.len()]),
        Some(vec![2.0; x_deg.len()]),
        None,
        Some(vec![1.0; x_deg.len()]),
    )
    .unwrap();
    let position = MonochromaticPositionCorrection {
        zero_shift_deg: 0.0,
        bragg_brentano_mm: None,
        debye_scherrer_micrometre: None,
    };
    RietveldProjectState {
        project: ProjectRecord {
            project_id: RecordId::new("project").unwrap(),
            revision: 0,
            name: "Calculation".to_owned(),
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
                phase_ids: vec![RecordId::new("alpha").unwrap()],
            }],
            phases: vec![StructuralPhaseRecord {
                phase_id: RecordId::new("alpha").unwrap(),
                name: "Alpha".to_owned(),
                definition,
                required_providers: Vec::new(),
            }],
            metadata: BTreeMap::new(),
        },
        analyses: Vec::new(),
    }
}

fn runnable_store() -> DesktopProjectStore {
    let store = DesktopProjectStore::new();
    store.create_project("project", "Empty").unwrap();
    store.replace_project(0, project_state()).unwrap();
    store
}

#[test]
fn calculates_without_mutating_project_and_retains_complete_binary_series() {
    let store = runnable_store();
    let calculations = CalculationManager::new(store.clone());

    let response = calculations
        .calculate_histogram(1, "histogram", CalculationOptionsInput::default())
        .unwrap();

    assert_eq!(response.project_revision, 1);
    assert_eq!(response.sample_count, 101);
    assert_eq!(response.phase_count, 1);
    assert!(response.rp.is_some());
    assert_eq!(store.snapshot().unwrap().revision(), 1);
    let descriptors = calculations
        .calculation_series(response.calculation_id)
        .unwrap();
    assert!(
        descriptors
            .iter()
            .any(|item| item.series_id == "calculated_y")
    );
    assert!(
        descriptors
            .iter()
            .any(|item| item.series_id == "phase/alpha/profile_y")
    );
    assert!(
        descriptors.iter().all(|item| {
            item.dtype == SeriesDtype::Float64Le || item.dtype == SeriesDtype::Uint8
        })
    );
    assert!(descriptors.iter().all(|item| matches!(
        item.owner,
        SeriesOwner::Calculation {
            calculation_id,
            project_revision: 1,
            ref histogram_id,
        } if calculation_id == response.calculation_id && histogram_id == "histogram"
    )));
    let payload = calculations
        .calculation_series_payload(response.calculation_id, "x_deg")
        .unwrap();
    assert_eq!(payload.bytes()[..8], 20.0_f64.to_le_bytes());
}

#[test]
fn retained_results_survive_later_edits_until_explicit_disposal() {
    let store = runnable_store();
    let calculations = CalculationManager::new(store.clone());
    let response = calculations
        .calculate_histogram(1, "histogram", CalculationOptionsInput::default())
        .unwrap();
    let mut next = store.snapshot().unwrap().state().clone();
    next.project.name = "Edited later".to_owned();
    store.replace_project(1, next).unwrap();

    assert!(
        calculations
            .calculation_series(response.calculation_id)
            .is_ok()
    );
    calculations
        .discard_calculation(response.calculation_id)
        .unwrap();
    assert_eq!(
        calculations
            .calculation_series(response.calculation_id)
            .unwrap_err()
            .code,
        DesktopErrorCode::UnknownCalculation
    );
}

#[test]
fn stale_empty_and_invalid_calculations_are_structured() {
    let store = runnable_store();
    let calculations = CalculationManager::new(store.clone());
    let stale = calculations
        .calculate_histogram(0, "histogram", CalculationOptionsInput::default())
        .unwrap_err();
    assert_eq!(stale.code, DesktopErrorCode::RevisionConflict);
    let invalid = calculations
        .calculate_histogram(
            1,
            "histogram",
            CalculationOptionsInput {
                minimum_parallel_phases: 0,
                ..CalculationOptionsInput::default()
            },
        )
        .unwrap_err();
    assert_eq!(invalid.code, DesktopErrorCode::Calculation);

    let mut empty = store.snapshot().unwrap().state().clone();
    empty.project.histograms[0].phase_ids.clear();
    empty.project.phases.clear();
    store.replace_project(1, empty).unwrap();
    let empty = calculations
        .calculate_histogram(2, "histogram", CalculationOptionsInput::default())
        .unwrap_err();
    assert_eq!(empty.code, DesktopErrorCode::Calculation);
}

#[test]
fn calculation_options_have_a_stable_json_contract() {
    let value = serde_json::to_value(CalculationOptionsInput::default()).unwrap();
    assert_eq!(value["support_fwhm"], 20.0);
    assert_eq!(value["threads"], 2);
    let restored: CalculationOptionsInput = serde_json::from_value(value).unwrap();
    assert_eq!(restored, CalculationOptionsInput::default());
}
